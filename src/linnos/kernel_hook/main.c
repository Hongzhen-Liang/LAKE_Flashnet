/*
 * Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
 * Copyright (C) 2022-2024 Henrique Fingler
 * Copyright (C) 2022-2024 Isha Tarte
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <linux/sched/signal.h>
#include <linux/slab.h>
#include <linux/time.h>
#include <linux/ktime.h>
#include <linux/delay.h>
#include <linux/blkdev.h>
#include <linux/string.h>
#include <linux/completion.h>
#include <linux/vmalloc.h>
#include "predictors.h"
#include "lake_shm.h"
#include "queue_depth.h"
#include "helpers.h"

#define SET_SYSCTL_DEBUG 0

extern unsigned long sysctl_lake_enable_linnos;
extern unsigned long sysctl_lake_linnos_debug;

static char *predictor_str = "fake";
module_param(predictor_str, charp, 0444);
MODULE_PARM_DESC(predictor_str, "What predictor to use: fake, cpu, gpu, batchtest, queudepth");

static char *cubin_path = "linnos.cubin";
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to linnos.cubin in case you're using gpu predictor");

int model_size = 0;
module_param(model_size, int, 0444);
MODULE_PARM_DESC(model_size, "what model to use, 0 default, 1 +1, 2 +2");

//adding a model to a device requires:
// 1. include the header with the weights
// 2. put device name in devices
// 3. set the pointers into a new array in weights (dont mess with the ending 0)

#include "sde.h"

/*
  Uncomment only one set of three of these, then go to the
  long *weights[][8] array below and make sure it matches:
  If using default NN, uncomment the lines below //NN (the ones with 4 zeros
  at the end).
  For NN+1, uncomment below //NN+1 (with 2 zeros)
  For NN+2, uncomment below //NN+2 (with no zeros)
*/

#include "weights_header/mix/w_Trace_nvme0n1_1.h"      // don't change here
#include "weights_header/mix/w_Trace_nvme0n1_128.h"    // change here

long *weights[][2][8] = {
	//NN
	{	
		{weight_0_T_nvme0n1_1, weight_1_T_nvme0n1_1, bias_0_nvme0n1_1, bias_1_nvme0n1_1 ,0,0,0,0},           // don't change here
		{weight_0_T_nvme0n1_128, weight_1_T_nvme0n1_128, bias_0_nvme0n1_128, bias_1_nvme0n1_128 ,0,0,0,0},   // change here
	},


	// {weight_0_T_sdb, weight_1_T_sdb, bias_0_sdb, bias_1_sdb ,0,0,0,0},
	// {weight_0_T_sda2, weight_1_T_sda2, bias_0_sda2, bias_1_sda2 ,0,0,0,0},
	// {weight_0_T_nvme2n1, weight_1_T_nvme2n1, bias_0_nvme2n1, bias_1_nvme2n1 ,0,0,0,0},

	// NN+1
	//{weight_0_T_nvme0n1, weight_2_T_nvme0n1, bias_0_nvme0n1, bias_2_nvme0n1, weight_1_T_nvme0n1, bias_1_nvme0n1 ,0,0},
	//{weight_0_T_nvme1n1, weight_2_T_nvme1n1, bias_0_nvme1n1, bias_2_nvme1n1, weight_1_T_nvme1n1, bias_1_nvme1n1 ,0,0},
	//{weight_0_T_nvme2n1, weight_2_T_nvme2n1, bias_0_nvme2n1, bias_2_nvme2n1, weight_1_T_nvme2n1, bias_1_nvme2n1 ,0,0},

	//NN+2
	//{weight_0_T_nvme0n1, weight_3_T_nvme0n1, bias_0_nvme0n1, bias_3_nvme0n1, weight_1_T_nvme0n1, bias_1_nvme0n1 ,weight_2_T_nvme0n1, bias_2_nvme0n1},
	//{weight_0_T_nvme1n1, weight_3_T_nvme1n1, bias_0_nvme1n1, bias_3_nvme1n1, weight_1_T_nvme1n1, bias_1_nvme1n1 ,weight_2_T_nvme1n1, bias_2_nvme1n1},
	//{weight_0_T_nvme2n1, weight_3_T_nvme2n1, bias_0_nvme2n1, bias_3_nvme2n1, weight_1_T_nvme2n1, bias_1_nvme2n1 ,weight_2_T_nvme2n1, bias_2_nvme2n1},

	// for testing..
	//{weight_0_T_sde, weight_1_T_sde, bias_0_sde, bias_1_sde,0,0,0,0},
	//{weight_0_T_sde, weight_1_T_sde, bias_0_sde, bias_1_sde,0,0,0,0}
};

static const char *devices[] = {
	// "/dev/sda",
	"/dev/nvme0n1",   // change here
	// "/dev/sda2",
	// "/dev/nvme2n1",
    //"/dev/vdb",
	//"/dev/vdc",
	0
};

//the predictor function to use
bool (*fptr)(char*,int,long**);

bool is_qdepth = false;
bool is_batch_test = false;
bool is_gpu_inf = false;

/*
 *  Helpers for Batch test
 */
static void batch_test_attach(void) {
	pr_warn("<LAKE trace> attch batch test. \n");
	int i;
	fptr = batch_test;
	window_size_hist = vmalloc(512);
	for (i=0;i<512;i++) window_size_hist[i] = 0;
}
static void batch_test_detach(void) {
	int i;
	for (i=0;i<512;i++)
		if (window_size_hist[i] != 0)
			pr_warn("%d:\t%u\n", i, window_size_hist[i]);
	vfree(window_size_hist);
}

/*
 *  Helpers for queue depth stats
 */
static int qdepth_attach(void) {
	pr_warn("<LAKE trace> attach qdepth. \n");
	int err;
	err = qd_init(); //this sets ptr
	if (err != 0) return err;
	usleep_range(5,10); //lets chill, why not
	sysctl_lake_linnos_debug = 3; //this enables storing batches
	return 0;
}
static void qdepth_detach(void) {
	qd_writeout();
}

/*
 *  Actual hook code
 */
static int parse_arg(void) {
	if (!strcmp("fake", predictor_str)) {
		fptr = fake_prediction_model;
	} else if (!strcmp("cpu", predictor_str)) {
		if (model_size == 0) {
			fptr = cpu_prediction_model;
			no_reject = false;
		}
		else if (model_size == 1) {
			fptr = cpu_prediction_model_plus_1;
			no_reject = true;
		}
		else {
			fptr = cpu_prediction_model_plus_2;
			no_reject = true;
		}
		pr_warn("Inserting CPU prediction with %d extra layers\n", model_size);
	}else if (!strcmp("gpu", predictor_str)) {
		is_gpu_inf = true;
	} else if (!strcmp("batchtest", predictor_str)) {
		pr_warn("Inserting batch test prediction\n");
		is_batch_test = true;
	} else if (!strcmp("queue_depth", predictor_str)) {
		pr_warn("Inserting queue_depth\n");
		//set fake so we go through everything
		is_qdepth = true;
		fptr = fake_prediction_model;
	} else {	
		pr_warn("Invalid predictor argument\n");
		return -2;
	}
	return 0;
}


/*
 *  Helpers for GPU inference
 */
static int gpu_attach(void) {
	int i, ndev=0;
	const char *devs;
	
	fptr = gpu_batch_entry;
	for(devs = devices[0], i=0 ; devs != 0 ; devs = devices[++i]) 
		ndev++;
	pr_warn("<LAKE trace> prepare to attach GPU. \n");
	// pr_warn("<LAKE trace> initing for %d devices\n", ndev);
	multi_initialize_gpu(cubin_path, 256, ndev);
	window_size_hist = vmalloc(256);
	for (i=0;i<256;i++) 
		window_size_hist[i] = 0;
	if(model_size==0) {
		cpu_gpu_threshold = 1;
		max_batch_size = 128;            // change here
	 	// window_size_ns = 5*_us;   
		window_size_ns = 1000*_us;
		no_reject = false;
	} else if (model_size == 1) {
		window_size_ns = 40*_us;
	 	cpu_gpu_threshold = 4;
		max_batch_size = 8;
		no_reject = true;
	} else if (model_size == 2) {
	 	cpu_gpu_threshold = 4;
	 	window_size_ns = 40*_us;
	 	max_batch_size = 6;
		no_reject = true;
	}
	predictors_mgpu_init();
	return 0;
}

static void gpu_detach(void) {
	const char *devs;
	int i;
	pr_warn("Prepare to clean weights of GPU. \n");
	for(devs = devices[0], i=0 ; devs != 0 ; devs = devices[++i]) {
		multi_gpu_cuda_cleanup_dev(&gpu_weights[i][0], i, 1);   // clean weights of low-gran
		multi_gpu_cuda_cleanup_dev(&gpu_weights[i][1], i, 0);   // clean weights of high-gran
	}
	pr_warn("GPU weights-clean succeed! \n");
	
	for (i=0;i<128;i++)
		if (window_size_hist[i] != 0)
			pr_warn("%d:\t%u\n", i, window_size_hist[i]);

	pr_warn("Total trace num: %u\n", n_traces);
	pr_warn("GPU was used %u times\n", n_used_gpu);
	pr_warn("Batch skipped %u times\n", n_skipped);
	// for (i=0;i<NUMBER_DEVICES;i++) {
	// 	pr_warn("IOs on device %d: %u\n", i, ios_on_device[i]);
	// }
	cuCtxDestroy(cuctx);
}
static void gpu_copy_weight(int idx) {
	long **wts_low = weights[idx][0];
	long **wts_high = weights[idx][1];
	pr_warn("<LAKE trace> Copying weights of device idx high %d\n", idx);

	// copy weights for high-granularity inference.
	copy_weights(wts_low, &gpu_weights[idx][0], ONE_IO_LEN);
	pr_warn("<LAKE trace> Copying weights of device idx low %d\n", idx);
	// copy weights for granularity = 1
	copy_weights(wts_high, &gpu_weights[idx][1], LEN_INPUT);	
	pr_warn("<LAKE trace> Finish copying weights of device idx low %d\n", idx);

	first_weight_ptr_to_dev[idx] = wts_low[0];
}

static int attach_to_queue(int idx) {
	struct block_device *dev;
	struct request_queue *q;
	long **wts_low = weights[idx][0];

	pr_warn("<LAKE trace> Attach to queue on %s\n", devices[idx]);
	dev = blkdev_get_by_path(devices[idx], FMODE_READ|FMODE_WRITE, THIS_MODULE);
	if(IS_ERR(dev)) {
		pr_warn("Error getting dev by path (%s): %ld\n", devices[idx], PTR_ERR(dev));
		return -2;
	}
	q = bdev_get_queue(dev);

	//more spaggheti, nice
	if (is_gpu_inf) 
		gpu_copy_weight(idx);
	pr_warn("<LAKE trace> Finish GPU copy weight. \n");


	// for Low granularity
	q->weight_0_T = wts_low[0];
	q->weight_1_T = wts_low[1];
	q->bias_0 = wts_low[2];
	q->bias_1 = wts_low[3];
	pr_warn("<LAKE trace> Finish attach 1st, 2nd layer. \n");

	q->weight_2_T= wts_low[4];
	q->bias_2 = wts_low[5];
	q->weight_3_T = wts_low[6];
	q->bias_3 = wts_low[7];
	pr_warn("<LAKE trace> Finish attach 3rd, 4th layer. \n");

	q->predictor = fptr;
	q->ml_enabled = true;
	sysctl_lake_enable_linnos = true;
	pr_warn("<LAKE trace> Attached!\n");
	return 0;
}

static int gpu_detach_queue(int idx) {
	struct block_device *dev;
	struct request_queue *q;

	pr_warn("Dettaching queue on %s\n", devices[idx]);
	dev = blkdev_get_by_path(devices[idx], FMODE_READ|FMODE_WRITE, THIS_MODULE);
	if(IS_ERR(dev)) {
		pr_warn("Error getting dev by path (%s): %ld\n", devices[idx], PTR_ERR(dev));
		return -1;
	}
	q = bdev_get_queue(dev);

	q->ml_enabled = false;
	sysctl_lake_enable_linnos = false;
	usleep_range(100,200);
	q->predictor = 0;
	q->weight_0_T = 0;
	q->weight_1_T = 0;
	q->bias_0 = 0;
	q->bias_1 = 0;

	q->weight_2_T = 0;
	q->bias_2 = 0;
	q->weight_3_T = 0;
	q->bias_3 = 0;

	pr_warn("Dettached!\n");
	return 0;
}

/**
 * Program main
 */
static int __init hook_init(void)
{
	const char *devs;
	int i, err;

	pr_warn("<LAKE Trace> linnos_gpu hook init. \n");
	sysctl_lake_linnos_debug = SET_SYSCTL_DEBUG;
	err = parse_arg();
	if(err < 0) return -2;

	//special handling
	if(is_batch_test) batch_test_attach();
	if(is_qdepth) 
		if(qdepth_attach() != 0)
			return -2;
	if(is_gpu_inf) gpu_attach();

	for(devs = devices[0], i=0 ; devs != 0 ; devs = devices[++i]) {
		err = attach_to_queue(i);
		if (err) return err;
	}

	return 0;
}

static void __exit hook_fini(void)
{
	const char *devs;
	int i, err;

	sysctl_lake_linnos_debug = 0;
	for(devs = devices[0], i=0 ; devs != 0 ; devs = devices[++i]){
		pr_warn("Prepare to call gpu_detach_queue \n");
		err = gpu_detach_queue(i);
		if (err) {
			return;
		} else {
			pr_warn("gpu detach queue without error. \n");
		}
	}

	if(is_qdepth) {
		pr_warn("Enter is_qdepth\n");
		qdepth_detach();
		pr_warn("qdepth_detach OK\n");
	}
	if(is_batch_test) {
		pr_warn("Enter is_batch_test\n");
		batch_test_detach();
		pr_warn("is_batch_test OK\n");
	}
	if(is_gpu_inf) {
		pr_warn("Enter gpu_detach()\n");
		gpu_detach();
		pr_warn("gpu_detach OK!\n");
	}
}

module_init(hook_init);
module_exit(hook_fini);

MODULE_AUTHOR("Henrique Fingler");
MODULE_DESCRIPTION("Kernel predictor hooks for LAKE-linnos");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
