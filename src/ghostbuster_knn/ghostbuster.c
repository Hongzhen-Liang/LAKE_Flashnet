#ifdef __KERNEL__
#include <linux/module.h>
#include <linux/random.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <asm/uaccess.h>
#include <linux/delay.h>
#include <linux/ktime.h>
#include "cuda.h"
#include "lake_shm.h"
#define PRINT(...) pr_warn(__VA_ARGS__)
#else
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <errno.h>
static inline uint64_t get_tsns() {
    struct timeval current_time;
    gettimeofday(&current_time, 0);
    return current_time.tv_sec*1000000000 + current_time.tv_usec*1000;
}
void get_random_bytes(char* x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = rand();
}

#define usleep_range(X,Y) sleep(X/1000000)
#define ktime_get_ns() get_tsns()
#define u64 uint64_t
#define vmalloc(X) malloc(X)
#define vfree(X) free((void *)X)
#define kava_alloc(...) malloc(__VA_ARGS__)
#define kava_free(...) free(__VA_ARGS__)
#define PRINT(...) printf(__VA_ARGS__)
#include <cuda.h>
#endif

static char *cubin_path = "/home/hfingler/hf-HACK/src/ghostbuster_knn/knncuda.cubin";
#ifdef __KERNEL__
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to .cubin");
#endif

#define BLOCK_DIM 16
#define WARMS 2
#define RUNS 5

// XXX Need to handle FLOATs eventually
typedef int FLOAT;
typedef u64 DOUBLE;

struct cuda_ctx
{
    CUdevice dev;
    CUcontext ctx;
    CUstream stream;
    CUmodule mod;
    CUfunction compute_dist;
    CUfunction modified_insertion_sort;
    CUfunction compute_sqrt;
};

typedef struct
{
    u64 x;
    u64 y;
    u64 z;
} dim3;

// CUDA vars
struct cuda_ctx ctx;

// KNN vars
static int ref_nb = 16384;
static int query_nb = 4096;
static int k = 16;

int init_cuda(void)
{
    int ret = 0;

    ret = cuInit(0);
    if (ret) {
        PRINT("Err cuInit %d\n", ret);
        goto out;
    }

    ret = cuDeviceGet(&ctx.dev, 0);
    if (ret) {
        PRINT("Err cuDeviceGet %d\n", ret);
        goto out;
    }

    ret = cuCtxCreate(&ctx.ctx, 0, ctx.dev);
    if (ret) {
        PRINT("Err cuCtxCreate %d\n", ret);
        goto out;
    }

    ret = cuModuleLoad(&ctx.mod, cubin_path);
    if (ret) {
        PRINT("Err cuModuleLoad %d\n", ret);
        goto out;
    }

    ret = cuModuleGetFunction(&ctx.compute_dist, ctx.mod,
                               "_Z17compute_distancesPfiiS_iiiS_");
    if (ret) {
        PRINT("Err cuModuleGetFunction %d\n", ret);
        goto out;
    }

    ret = cuModuleGetFunction(&ctx.modified_insertion_sort, ctx.mod,
                               "_Z23modified_insertion_sortPfiPiiiii");
    if (ret) {
        PRINT("Err cuModuleGetFunction 2 %d\n", ret);
        goto out;
    }

    ret = cuModuleGetFunction(&ctx.compute_sqrt, ctx.mod,
                               "_Z12compute_sqrtPfiii");
    if (ret) {
        PRINT("Err cuModuleGetFunction 3 %d\n", ret);
        goto out;
    }

out : return ret;
}
// ==================== End CUDA ====================

// ==================== Start KNN ====================
void initialize_data(FLOAT *ref,
                     int ref_nb,
                     FLOAT *query,
                     int query_nb,
                     int dim)
{
    int i;
    int rand;

    // XXX Resolve floats
    // Generate random reference points
    for (i = 0; i < ref_nb * dim; ++i)
    {
        get_random_bytes(&rand, sizeof(rand));
        ref[i] = 10 * (FLOAT)(rand); // / (DOUBLE) RAND_MAX );
    }

    // XXX Resolve floats
    // Generate random query points
    for (i = 0; i < query_nb * dim; ++i)
    {
        get_random_bytes(&rand, sizeof(rand));
        query[i] = 10 * (FLOAT)(rand); // / (DOUBLE) RAND_MAX );
    }
}

static u64 ctime, ttime;


int knn_cuda( const FLOAT *ref,
              int          ref_nb,
              const FLOAT *query,
              int          query_nb,
              int          dim,
              int          k,
              FLOAT        *knn_dist,
              int          *knn_index )
{
  int ret = 0;

  // Launch params
  dim3 block0, block1, block2;
  dim3 grid0, grid1, grid2;

  // Vars for computation
  CUdeviceptr ref_dev, query_dev, dist_dev, index_dev;
  size_t ref_pitch_in_bytes;
  size_t query_pitch_in_bytes;
  size_t dist_pitch_in_bytes;
  size_t index_pitch_in_bytes;
  // Pitch values
  size_t ref_pitch, query_pitch;
  size_t dist_pitch, index_pitch;
  // Params for pitch (4, 8, or 16)
  size_t element_size_bytes = 16;

  // Allocate global memory
  ret |= cuMemAllocPitch( &ref_dev, &ref_pitch_in_bytes,
                          ref_nb * sizeof( FLOAT ), dim, element_size_bytes );
  ret |= cuMemAllocPitch( &query_dev, &query_pitch_in_bytes,
                          query_nb * sizeof( FLOAT ), dim, element_size_bytes );
  ret |= cuMemAllocPitch( &dist_dev, &dist_pitch_in_bytes,
                          query_nb * sizeof( FLOAT ),
                          ref_nb, element_size_bytes );
  ret |= cuMemAllocPitch( &index_dev, &index_pitch_in_bytes,
                          query_nb * sizeof( int ), k, element_size_bytes );
  if (ret) {
    PRINT( "Memory allocation error\n" );
    goto out;
  }

  // Deduce pitch values
  ref_pitch = ref_pitch_in_bytes / sizeof( FLOAT );
  query_pitch = query_pitch_in_bytes / sizeof( FLOAT );
  dist_pitch = dist_pitch_in_bytes / sizeof( FLOAT );
  index_pitch = index_pitch_in_bytes / sizeof( int );

  // Check pitch values
  if (query_pitch != dist_pitch || query_pitch != index_pitch ) {
    PRINT("Invalid pitch value\n" );
    goto out;
  }

  // Copy reference and query data from the host to the device
  ret |= cuMemcpyHtoDAsync(ref_dev, ref, ref_pitch_in_bytes, 0 );
  ret |= cuMemcpyHtoDAsync(query_dev, query,
                            query_pitch_in_bytes, 0 );

  if (ret) {
    PRINT( "Unable to copy data from host to device\n" );
    goto out;
  }

  // Compute the squared Euclidean distances
  block0 = (dim3) { BLOCK_DIM, BLOCK_DIM, 1 };
  grid0 = (dim3) { query_nb / BLOCK_DIM, ref_nb / BLOCK_DIM, 1 };
  if (query_nb % BLOCK_DIM != 0) {
    grid0.x += 1;
  }
  if (ref_nb % BLOCK_DIM != 0) {
    grid0.y += 1;
  }

  void *args0[] = { &ref_dev, &ref_nb, &ref_pitch,
                   &query_dev, &query_nb, &query_pitch,
                   &dim, &dist_dev };
  cuLaunchKernel( ctx.compute_dist, grid0.x, grid0.y,
                  grid0.z, block0.x, block0.y,
                  block0.z, 0, 0,
                  args0, NULL);
  if ((ret = cuCtxSynchronize())) {
    PRINT( "Unable to execute compute_dist kernel\n" );
    goto out;
  }

  // Sort the distances with their respective indexes
  block1 = (dim3) { 256, 1, 1 };
  grid1 = (dim3) { query_nb / 256, 1, 1 };
  if ( query_nb % 256 != 0 ) 
    grid1.x += 1;
  
  void *args1[] = { &dist_dev, &dist_pitch, &index_dev,
                   &index_pitch, &query_nb, &ref_nb,
                   &k };
  cuLaunchKernel( ctx.modified_insertion_sort, grid1.x, grid1.y,
                  grid1.z, block1.x, block1.y,
                  block1.z, 0, 0,
                  args1, NULL);

  // Compute the square root of the k smallest distances
  block2 = (dim3) { 16, 16, 1 };
  grid2 = (dim3) { query_nb / 16, k / 16, 1 };
  if ( query_nb % 16 != 0 ) {
    grid2.x += 1;
  }
  if ( k % 16 != 0 ) {
    grid2.y += 1;
  }
  void *args2[] = { &dist_dev, &query_nb, &query_pitch, &k };
  cuLaunchKernel( ctx.compute_sqrt, grid2.x, grid2.y,
                  grid2.z, block2.x, block2.y,
                  block2.z, 0, 0,
                  args2, NULL);
  if ( (ret = cuCtxSynchronize())) {
    PRINT( "Unable to execute modified_insertion_sort kernel\n" );
    goto out;
  }

  // Copy k smallest distances / indexes from the device to the host
  ret |= cuMemcpyDtoHAsync( knn_dist, dist_dev,
                            dist_pitch_in_bytes, 0 );
  ret |= cuMemcpyDtoHAsync( knn_index, index_dev,
                            index_pitch_in_bytes, 0 );
  if (ret) {
    PRINT( "Unable to copy data from device to host\n" );
    goto out;
  }

out:
  cuMemFree( ref_dev );
  cuMemFree( query_dev );
  cuMemFree( dist_dev );
  cuMemFree( index_dev ); 

  return ret;
}


// XXX Should time at some point
int test(const FLOAT *ref,
         int ref_nb,
         const FLOAT *query,
         int query_nb,
         int dim,
         int k,
         FLOAT *gt_knn_dist,
         int *gt_knn_index,
         int nb_iterations)
{
    int ret = 0;
    int i;
    int *test_knn_index;
    FLOAT *test_knn_dist;
    //int nb_correct_precisions;
    //int nb_correct_indexes;

    u64 ctimes;
    u64 ttimes;

    // XXX Deal with floats
    // Parameters
    //const FLOAT precision = 0.001f;    // distance error max
    //const FLOAT min_accuracy = 0.999f; // percentage of correct values required
    //FLOAT precision_accuracy;
    //FLOAT index_accuracy;

    // Allocate memory for computed k-NN neighbors
    test_knn_dist = (FLOAT *)kava_alloc(query_nb * k * sizeof(FLOAT));
    test_knn_index = (int *)kava_alloc(query_nb * k * sizeof(int));

    // Allocation check
    if (!test_knn_dist || !test_knn_index)
    {
        PRINT("Error allocating CPU memory for KNN results\n");
        ret = -ENOMEM;
        goto out;
    }
    
    // warm
    PRINT("Computing knn %d times\n", nb_iterations);
    for (i = 0; i < WARMS; ++i)
    {
        ret = knn_cuda(ref, ref_nb, query, query_nb, dim,
                       k, test_knn_dist, test_knn_index);
        PRINT("Computation done on round %d\n", i);
        if (ret != 0)
        {
            PRINT("Computation failed on round %d\n", i);
            goto out;
        }
        else
        {
            PRINT("Computation done on round %d\n", i);
        }
    }

    usleep_range(20, 200);

    ctimes = 0;
    ttimes = 0;
    // Compute k-NN several times
    // PRINT( "Computing knn %d times\n", nb_iterations );
    for (i = 0; i < nb_iterations; ++i)
    {
        if ((ret = knn_cuda(ref, ref_nb, query, query_nb, dim,
                            k, test_knn_dist, test_knn_index)))
        {
            PRINT("Computation failed on round %d\n", i);
            goto out;
        }

        ctimes += ctime;
        ttimes += ttime;
        usleep_range(20, 200);
    }
    PRINT("gpu_%d, %lld, %lld\n", dim, ctimes / (nb_iterations * 1000), ttimes / (nb_iterations * 1000));

out:
    kava_free(test_knn_dist);
    kava_free(test_knn_index);

    return ret;
}

// Allocate input points and output k-NN distances / indexes
int run_knn(void)
{
    int ret = 0;
    int *knn_index;
    FLOAT *ref;
    FLOAT *query;
    FLOAT *knn_dist;
    int knn_index_sz = query_nb * k * sizeof(int);
    int ref_sz;
    int query_sz;
    int knn_dist_sz = query_nb * k * sizeof(FLOAT);
    int i, dim;
    int dims[] = {8, 16, 32, 64, 128};
    int ndims = 5;

    knn_index = (int *)kava_alloc(knn_index_sz);
    knn_dist = kava_alloc(knn_dist_sz);
    for (i = 0; i < ndims; i++)
    {
        dim = dims[i];

        ref_sz = ref_nb * dim * sizeof(FLOAT);
        query_sz = query_nb * dim * sizeof(FLOAT);

        PRINT("Allocate KNN CPU resources\n");
        ref = (FLOAT *)kava_alloc(ref_sz);
        query = (FLOAT *)kava_alloc(query_sz);

        // Allocation checks
        if (!ref || !query || !knn_dist || !knn_index)
        {
            PRINT("Error allocating KNN CPU resources\n");
            ret = -ENOMEM;
            goto out;
        }
        PRINT("Successfully allocated KNN CPU resources\n");

        // Initialize reference and query points with random values
        initialize_data(ref, ref_nb, query, query_nb, dim);
        PRINT("Test KNN execution\n");
        if ((ret = test(ref, ref_nb, query, query_nb, dim,
                        k, knn_dist, knn_index, RUNS)))
        {
            PRINT("KNN execution test failed\n");
            // XXX Should probably use a more idiomatically correct error code
            ret = -ENOENT;
            goto out;
        }
        PRINT("KNN execution test succeeded\n");

        // XXX probably not worth computing ground truth in the kernel
    out:
        kava_free(ref);
        kava_free(query);
    }

    kava_free(knn_dist);
    kava_free(knn_index);

    return 0;
}
// ==================== End KNN ====================

#ifdef __KERNEL__
static int __init ghost_buster_init(void)
{
    int ret = 0;
    if ((ret = init_cuda())) {
        return ret;
    }
    if ((ret = run_knn())) {
        return ret;
    }
    return ret;
}

static void __exit ghost_buster_fini(void)
{
}

module_init(ghost_buster_init);
module_exit(ghost_buster_fini);

MODULE_AUTHOR("Ariel Szekely");
MODULE_DESCRIPTION("A module to detect Spectre attacks"
                   "(aka a Ghost Buster... get it?)");
MODULE_LICENSE("GPL");
MODULE_VERSION(
    __stringify(1) "." __stringify(0) "." __stringify(0) "."
                                                         "0");

#else

int main() {
    int ret = 0;
    if ((ret = init_cuda())) {
        return ret;
    }
    if ((ret = run_knn())) {
        return ret;
    }
    return ret;
}

#endif