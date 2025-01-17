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
#include "hello.h"
#include <linux/delay.h>

//#include <linux/bpf.h>
//#include <linux/btf_ids.h>

static int devID = 0;
module_param(devID, int, 0444);
MODULE_PARM_DESC(devID, "GPU device ID in use, default 0");

static char *cubin_path = "hello.cubin";
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to firewall.cubin, default ./firewall.cubin");

static int run_hello(void)
{
    int i, j;
    CUdevice dev;
	int count;
    CUcontext ctx;	
    CUmodule mod;
    CUfunction hello_kernel;
	int* val;	
	CUdeviceptr d_p1;
	u64 t_start, t_stop, t_stop2;

	PRINT(V_INFO, "Running hello world\n");

	// Get the GPU ready
    cuInit(0);
	check_error(cuDeviceGet(&dev, devID), "cuDeviceGet", __LINE__);
	check_error(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate", __LINE__);
    check_error(cuModuleLoad(&mod, cubin_path), "cuModuleLoad", __LINE__);
    check_error(cuModuleGetFunction(&hello_kernel, mod, "_Z12hello_kernelPii"),
            "cuModuleGetFunction", __LINE__);

	val = kava_alloc(10*sizeof(int));
	for (i = 0; i < 10; i++)
		val[i] = 0;

	t_start = ktime_get_ns();
	check_error(cuMemAlloc((CUdeviceptr*) &d_p1, 128), "cuMemAlloc d_p1", __LINE__);
	check_error(cuMemcpyHtoD(d_p1, val, 10), "cuMemcpyHtoD", __LINE__);
	count = 10;
	void *args[] = {
		&d_p1, &count
	};

	for (j=0 ; j < 16 ; j++) {
		check_error(cuLaunchKernel(hello_kernel, 
					1, 1, 1,
					10, 1, 1, 
					0, NULL, args, NULL),
				"cuLaunchKernel", __LINE__);
	}
	t_stop = ktime_get_ns();
	cuCtxSynchronize();
	t_stop2 = ktime_get_ns();

	PRINT(V_INFO, "Times (us): %llu, %llu\n", (t_stop - t_start)/1000, (t_stop2 - t_start)/1000);
	check_error(cuMemcpyDtoH(val, d_p1, 10*sizeof(int)), "cuMemcpyDtoH", __LINE__);
	PRINT(V_INFO, "Printing resulting array: \n");
	for (i = 0; i < 10; i++)
		PRINT(V_INFO, " %d", val[i]);

	cuCtxSynchronize();
	cuMemFree(d_p1);
	kava_free(val);

 	// check_error(cuCtxDestroy(ctx), "cuCtxDestroy", __LINE__);
	return 0;
}


/**
 * Program main
 */
static int __init hello_init(void)
{
	return run_hello();
}

static void __exit hello_fini(void)
{
}

module_init(hello_init);
module_exit(hello_fini);

MODULE_AUTHOR("Henrique Fingler");
MODULE_DESCRIPTION("Example kernel module of using CUDA in lake");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
