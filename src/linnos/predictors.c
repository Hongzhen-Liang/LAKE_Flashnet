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

#include <linux/delay.h>
#include <linux/ktime.h>
#include <linux/vmalloc.h>
#include <asm/fpu/api.h>
#include <linux/completion.h>
#include "predictors.h"
#include "variables.h"
#include "helpers.h"
#include "cuda.h"
#include "lake_shm.h"

int PREDICT_GPU_SYNC = 0;

//debug
volatile bool no_reject;

//batch variables
s64 window_size_ns;
u32 max_batch_size; //this cannot be more than 256 (allocated in main.c)
u32 cpu_gpu_threshold; //less than this we use cpu

//use normal (0), +1 or +2
extern u8 model_size;
u32 n_skipped = 0;
//batch test variables
u32* window_size_hist; //allocated in main.c of kernel_hook, 128 elements
u32 n_used_gpu = 0;
u32 n_traces = 0;
u32 ios_on_device[NUMBER_DEVICES];

u16 current_batch[NUMBER_DEVICES];
spinlock_t batch_entry[NUMBER_DEVICES];
//GPU inference variables
struct GPU_weights gpu_weights[NUMBER_DEVICES][2]; //per-ssd weights, we are not going to have more than NUMBER_DEVICES ssds..

spinlock_t per_batch_lock[NUMBER_DEVICES][MAX_DEV_BATCHES];
struct completion batch_completed[NUMBER_DEVICES][MAX_DEV_BATCHES];
struct completion finalize_batch[NUMBER_DEVICES][MAX_DEV_BATCHES];
u16 n_exited[NUMBER_DEVICES][MAX_DEV_BATCHES];
u16 waiting[NUMBER_DEVICES][MAX_DEV_BATCHES];
u64 window_start_ns[NUMBER_DEVICES][MAX_DEV_BATCHES];
bool use_cpu_instead[NUMBER_DEVICES][MAX_DEV_BATCHES][128];    // hard code here: modify later.
//0=idle, 1=id0 waiting, 2=running

bool batch_closed[NUMBER_DEVICES][MAX_DEV_BATCHES];
s64 first_arrival[NUMBER_DEVICES][MAX_DEV_BATCHES];

#define ia_avg_sz 4
#define ia_avg_shift 2
u32 ia_avgs[NUMBER_DEVICES][ia_avg_sz];
u32 ia_cur[NUMBER_DEVICES];
s64 last_arrival[NUMBER_DEVICES];

u32 cpu_times[] = {7, 101, 196};

struct IO_req_num{
	int vec_nums;
	int full_group_num;
	int lonely_IO_num;
};

void predictors_mgpu_init(void) {
	int i, j;
	for (i=0 ; i < NUMBER_DEVICES ; i++) {
		current_batch[i] = 0;
		ios_on_device[i] = 0;
		last_arrival[i] = 0;
		ia_cur[i] = 0;
		spin_lock_init(&batch_entry[i]);
		for (j=0 ; j < ia_avg_sz ; j++)
			ia_avgs[i][j] = 800*_us; //start large
		for (j=0 ; j < MAX_DEV_BATCHES ; j++) {
			n_exited[i][j] = 0;
			window_start_ns[i][j] = 0;
			waiting[i][j] = 0;
			init_completion(&batch_completed[i][j]);
			init_completion(&finalize_batch[i][j]);
			spin_lock_init(&per_batch_lock[i][j]);
			batch_closed[i][j] = false;
		}
	}
}

int gpu_get_prediction(int dev, int batch, int id) {
	return multi_gpu_outputs[dev][batch][id*64] >=
			(multi_gpu_outputs[dev][batch][id*64+32]) ?  false: true;
}

//hack: weights are actually device pointers here
void multi_gpu_predict_batch(char *__feat_vec, int n_vecs, long **weights_high, int dev, int batch) {
	//do inference
	void *args[] = {
		&weights_high[0], &weights_high[2], &multi_d_input_vec_i[dev][batch], &multi_d_mid_res_i[dev][batch]
	};
	void *args1[] = {
		&weights_high[1], &weights_high[3], &multi_d_mid_res_i[dev][batch], &multi_d_final_res_i[dev][batch]
	};


    check_error(cuLaunchKernel(batch_linnos_mid_layer_kernel,         // => `prediction_mid_layer_batch()` in kernel.cu
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                cu_streams[dev][batch], 
				args, NULL),
			"cuLaunchKernel", __LINE__);

    check_error(cuLaunchKernel(batch_linnos_final_layer_kernel,       // => `prediction_final_layer_batch()` in kernel.cu
				n_vecs, 1, 1,          //blocks
				64, 1, 1,   //threads per block
				0,   //shared mem
                cu_streams[dev][batch], 
				args1, NULL),
			"cuLaunchKernel", __LINE__);
}

void multi_gpu_predict_batch_plus_1(char *__feat_vec, int n_vecs, long **weights_high, int dev, int batch) {
	//do inference
	void *args[] = {
		&weights_high[0], &weights_high[2], &multi_d_input_vec_i[dev][batch], &multi_d_mid_res_i[dev][batch]
	};
	void *args1[] = {
		&weights_high[1], &weights_high[3], &multi_d_mid_res_1_i[dev][batch], &multi_d_final_res_i[dev][batch]
	};
	void *args2[] = {
		&weights_high[4], &weights_high[5], &multi_d_mid_res_i[dev][batch], &multi_d_mid_res_1_i[dev][batch]
	};

    check_error(cuLaunchKernel(batch_linnos_mid_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                cu_streams[dev][batch], 
				args, NULL),
			"cuLaunchKernel", __LINE__);

	check_error(cuLaunchKernel(batch_linnos_mid_layer_1_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                cu_streams[dev][batch], args2, NULL),
			"cuLaunchKernel", __LINE__);

    check_error(cuLaunchKernel(batch_linnos_final_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				64, 1, 1,   //threads per block
				0,   //shared mem
                cu_streams[dev][batch], 
				args1, NULL),
			"cuLaunchKernel", __LINE__);
}

void multi_gpu_predict_batch_plus_2(char *__feat_vec, int n_vecs, long **weights_high, int dev, int batch) {
	//do inference
	void *args[] = {
		&weights_high[0], &weights_high[2], &multi_d_input_vec_i[dev][batch], &multi_d_mid_res_i[dev][batch]
	};
	void *args1[] = {
		&weights_high[1], &weights_high[3], &multi_d_mid_res_2_i[dev][batch], &multi_d_final_res_i[dev][batch]
	};

	void *args2[] = {
		&weights_high[4], &weights_high[5], &multi_d_mid_res_i[dev][batch], &multi_d_mid_res_1_i[dev][batch]
	};

	void *args3[] = {
		&weights_high[6], &weights_high[7], &multi_d_mid_res_1_i[dev][batch], &multi_d_mid_res_2_i[dev][batch]
	};

    check_error(cuLaunchKernel(batch_linnos_mid_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                cu_streams[dev][batch], 
				args, NULL),
			"cuLaunchKernel", __LINE__);

	check_error(cuLaunchKernel(batch_linnos_mid_layer_1_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                cu_streams[dev][batch], args2, NULL),
			"cuLaunchKernel", __LINE__);

	check_error(cuLaunchKernel(batch_linnos_mid_layer_1_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                cu_streams[dev][batch], args3, NULL),
			"cuLaunchKernel", __LINE__);

    check_error(cuLaunchKernel(batch_linnos_final_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				64, 1, 1,   //threads per block
				0,   //shared mem
                cu_streams[dev][batch], 
				args1, NULL),
			"cuLaunchKernel", __LINE__);
}


int re_arrange_features(int n_vecs, int dev_index, int batch_index) {
	/*
		re-arrange the vectors according to the requirement of high-granularity inference.
		@n_vecs:
			number of vectors for this time of inference.
		Each vector is in format of:
			[hist_size_1, hist_size_2, hist_size_3, hist_size_4, IO_size_i, hist_latency_1, hist_latency_2, hist_latency_3, hist_latency_4]
	*/
	int group_id = 0;
	// 1. for the vectors that can form full groups
	for (int i = 0; i < (n_vecs / GRANULARITY) * GRANULARITY; i ++){
		if ((i + 1) % GRANULARITY == 0) {     // we use the history size and latency of the last IO request in a same granularity group.
			// 1. append the history IO size.
			for (int j = 0; j < HIST_SIZE * 3; j++) {    
				multi_inputs_to_gpu[dev_index][batch_index][group_id*LEN_INPUT+j] = multi_inputs_to_gpu_gran_1[dev_index][batch_index][i*ONE_IO_LEN+j];
			}
			// 2. append the current IO size.
			for (int j = 0; j < 3; j++){
				multi_inputs_to_gpu[dev_index][batch_index][group_id*LEN_INPUT+HIST_SIZE*3+(GRANULARITY-1)*3+j] = multi_inputs_to_gpu_gran_1[dev_index][batch_index][i*ONE_IO_LEN+HIST_SIZE*3+j];
			}
			// 3. append the history latency.
			for (int j = 0; j < HIST_SIZE * 4; j++){    
				multi_inputs_to_gpu[dev_index][batch_index][group_id*LEN_INPUT+HIST_SIZE*3+GRANULARITY*3+j] = multi_inputs_to_gpu_gran_1[dev_index][batch_index][i*ONE_IO_LEN+HIST_SIZE*3+3+j];
				// if (group_id*LEN_INPUT+HIST_SIZE*3+GRANULARITY*3+j == 39) {
				// 	pr_warn("multi_inputs_to_gpu[dev_index][batch_index][39] = %li, multi_inputs_to_gpu_gran_1[dev_index][batch_index][30] = %li \n", multi_inputs_to_gpu[dev_index][batch_index][39], multi_inputs_to_gpu_gran_1[dev_index][batch_index][30]);
				// }
			}
			group_id += 1;
		}
		else{
			// append the current IO size.
			for (int j = 0; j < 3; j++){
				multi_inputs_to_gpu[dev_index][batch_index][group_id*LEN_INPUT+HIST_SIZE*3+i*3+j] = multi_inputs_to_gpu_gran_1[dev_index][batch_index][i*ONE_IO_LEN+HIST_SIZE*3+j];
			}
		}
	}
	// pr_warn("<LAKE trace> multi_inputs_to_gpu[dev_index][batch_index][39] = %li. \n", multi_inputs_to_gpu[dev_index][batch_index][39]);
	return group_id;	
}


void do_gpu_inference(int vec_nums, long **weights_high, int dev, int batch_id) {
	/*
		@n_vecs: number of input vectors.
		@weights: weights and bias of model.
		@dev: device index
		@batch_id: id index.
	*/
	multi_copy_inputs_to_gpu(vec_nums, dev, batch_id);
	multi_gpu_predict_batch(0, vec_nums, weights_high, dev, batch_id);
	multi_copy_results_from_gpu(vec_nums, dev, batch_id);
}

void do_gpu_inference_plus_one(int vec_nums, long **weights_high, int dev, int batch_id) {
	multi_copy_inputs_to_gpu(vec_nums, dev, batch_id);
	multi_gpu_predict_batch_plus_1(0, vec_nums, weights_high, dev, batch_id);
	multi_copy_results_from_gpu(vec_nums, dev, batch_id);
}

void do_gpu_inference_plus_two(int vec_nums, long **weights_high, int dev, int batch_id) {
	multi_copy_inputs_to_gpu(vec_nums, dev, batch_id);
	multi_gpu_predict_batch_plus_2(0, vec_nums, weights_high, dev, batch_id);
	multi_copy_results_from_gpu(vec_nums, dev, batch_id);
}

//this is what an IO calls when it calls predict()
bool gpu_batch_entry(char *feat_vec, int n_vecs, long **weights) {
	/*
		@feat_vec: input features
		@n_vecs: number of vectors.
		@weights: weights of model with inference-granularity = 1.
	*/
	// pr_warn("<LAKE trace> [gpu_batch_entry()] Prepare to do gpu inference. \n");
	u16 my_id;
	u16 my_batch;
	bool my_prediction, use_cpu, skip;
	s64 my_arrival;
	u32 i, this_dev=99;
	unsigned long irqflags, err;
	s64 dif;
	bool is_last = false;
	bool inf_fast = false;
	s64 ia_avg = 0;

	int retry_num = 0;

	for(i = 0; i < NUMBER_DEVICES ; i++) {
		if(first_weight_ptr_to_dev[i] == weights[0]) {
			this_dev = i;
			break;
		}
	}
	if (unlikely(this_dev == 99)) {
		pr_warn("COULD NOT FIND DEV\n");
		return false;
	}
	n_traces += 1;

enter_again:
	// lock here.
	spin_lock_irqsave(&batch_entry[this_dev], irqflags);
	my_batch = current_batch[this_dev];
	my_id = waiting[this_dev][my_batch];
	//check this batch out
	
	//should we NOT get in this batch bc its running?
	if (batch_closed[this_dev][my_batch] == true) {
		//lets loop and try another
		current_batch[this_dev] = (current_batch[this_dev]+1) % MAX_DEV_BATCHES;
		//pr_warn("batch is closed, increasing by one to %d\n", current_batch[this_dev]);
		spin_unlock_irqrestore(&batch_entry[this_dev], irqflags);
		udelay(2); //we can afford 2 for a reschedule

		// tmp here to avoid the deadloop.
		retry_num += 1;
		if (retry_num < 20) {
			goto enter_again;
		} else {
			pr_warn("<LAKE trace> loop more than 20 times, give up looping. \n");
			retry_num = 0;
		}
	}

	// change here to make every IO request arrive at the same time.
	// my_arrival = ktime_get_ns();
	my_arrival = 0;

	dif = my_arrival - last_arrival[this_dev];
	ia_cur[this_dev] = (ia_cur[this_dev]+1)% ia_avg_sz ;
	ia_avgs[this_dev][ia_cur[this_dev]] = dif;
	last_arrival[this_dev] = my_arrival;

	ia_avg = 0;
	for (i = 0 ; i < ia_avg_sz ; i++)
	 	ia_avg += ia_avgs[this_dev][i];
	ia_avg = ia_avg >> ia_avg_shift;
	ia_avg = ia_avg / 1000;
	//pr_warn("avg now: %lld us\n", ia_avg/1000);

	i = model_size == 0 ? 1 : model_size;
	//pr_warn("avg %lld.  %lld  <  %lld  %d skip\n", ia_avg, cpu_times[model_size], ia_avg*i, skip);
	skip = cpu_times[model_size] < ia_avg * i;
	skip |= (window_size_ns <= WINDOW_THRESHOLD);

	// Important: Temporary comments this to see the result of GPU inference.
	// if(skip) {
	// 	spin_unlock_irqrestore(&batch_entry[this_dev], irqflags);
	// 	n_skipped++;
	// 	my_prediction = cpu_prediction_model(feat_vec, n_vecs, weights);
	// 	return no_reject ? false : my_prediction;
	// }

	//we can. would we close this batch?
	dif = my_arrival - first_arrival[this_dev][my_batch];
	is_last = dif >= window_size_ns;
	is_last = is_last && my_id; //cant be first
	if (is_last || my_id >= max_batch_size) {
		// pr_warn("<LAKE trace> Be the last one, my_id is %d, max_batch_size = %d. my_arrival = %d, first_arrival[this_dev][my_batch] = %d\n", my_id, max_batch_size, my_arrival, first_arrival[this_dev][my_batch]);
		//pr_warn("i am last of batch %d  time dif? %d  [%lld]!\n", my_batch, is_last, dif);
		//if so, increase current batch
		current_batch[this_dev] = (current_batch[this_dev]+1) % MAX_DEV_BATCHES;
		//we are last, mark batch as full
		is_last = true;
		batch_closed[this_dev][my_batch] = true;
	}
	//we can but not we are not last
	else {
		is_last = false;
	}
	//add one to batch size
	waiting[this_dev][my_batch] += 1;
	if (my_id == 0) {
		//pr_warn("id 0 reiniting batch %d\n", my_batch);
		reinit_completion(&finalize_batch[this_dev][my_batch]);
		reinit_completion(&batch_completed[this_dev][my_batch]);
		use_cpu = true;
		n_exited[this_dev][my_batch] = 0;
		first_arrival[this_dev][my_batch] = ktime_get_ns();
	}
	//Unlock here let others execute
	spin_unlock_irqrestore(&batch_entry[this_dev], irqflags);

	//copy inputs to intermediary buffer, but we need to convert into longs for gpu
	for (i = 0 ; i < ONE_IO_LEN ; i++)
		multi_inputs_to_gpu_gran_1[this_dev][my_batch][my_id*ONE_IO_LEN+i] = (long) feat_vec[i];

	// 1. for the last request:
	// last close everything: the last request will do all the inference work of this batch.
	if (is_last) {
last_req_close:
		//record in histogram
		window_size_hist[waiting[this_dev][my_batch]] += 1;
		//pr_warn(">> closing batch %d size %d\n", my_batch, waiting[this_dev][my_batch]);
		//lonely request :(
		if(waiting[this_dev][my_batch] <= 1) {
			use_cpu = true;
			goto reset_this_batch;
		}
		//not big enough for gpu
		else if(waiting[this_dev][my_batch] < cpu_gpu_threshold) {
			for (int id = 0; id < waiting[this_dev][my_batch]; id++){
				use_cpu_instead[this_dev][my_batch][id] = true;
			}
			use_cpu = true;
		}
		//use the gpu
		else {
			// Multiple IO request to be handled together.
			// if (waiting[this_dev][my_batch] > GRANULARITY) {
			// 	pr_warn("<LAKE trace> Use GPU to calcualate %d vectors!\n", waiting[this_dev][my_batch]);
			// }
			// pr_warn("<LAKE trace> Use GPU to calcualate\n");
			n_used_gpu++;
			pr_warn("[n_traces = %d, n_used_gpu = %d], num_vecs = %d \n", n_traces, n_used_gpu, waiting[this_dev][my_batch]);
			//my_prediction = false; //XXX
			// re-arrange the input features to be align with high-granularity infenrece.
			int full_group_num = re_arrange_features(waiting[this_dev][my_batch], this_dev, my_batch);
			for (int id = 0; id < waiting[this_dev][my_batch]; id++){
				if (id < full_group_num * GRANULARITY) {
					use_cpu_instead[this_dev][my_batch][id] = false;
				} else {
					use_cpu_instead[this_dev][my_batch][id] = true;
				}
			}
			if (my_id < full_group_num * GRANULARITY) {
				use_cpu= false;
			} else {
				use_cpu = true;
			}
			// pr_warn("<LAKE trace> %d vectors are re-arranged, with full-group = %d. \n", waiting[this_dev][my_batch], full_group_num);
			// pr_warn("<LAKE trace> The remaining %d vectors, which cannot form a full-gran group, will be computed by CPU. \n", waiting[this_dev][my_batch] - full_group_num * GRANULARITY);

			// waiting[this_dev][my_batch] means number of vectors to be predicted in a once.
			if (model_size == 0) {
				do_gpu_inference(full_group_num, gpu_weights[this_dev][1].weights, this_dev, my_batch); 
			} 
			else if (model_size == 1) {
				do_gpu_inference_plus_one(full_group_num, gpu_weights[this_dev][1].weights, this_dev, my_batch); 
			}
			else {
				do_gpu_inference_plus_two(full_group_num, gpu_weights[this_dev][1].weights, this_dev, my_batch); 
			}

			// judge whether the result is calculated using GPU.
			if (!use_cpu) {
				my_prediction = gpu_get_prediction(this_dev, my_batch, my_id / GRANULARITY);
				// pr_warn("<LAKE_trace> (id = %d) get the prediction result by gpu.\n", my_id);
			} 
		}

		//let everyone go now
		n_exited[this_dev][my_batch] += 1;
		//pr_warn(" last %d: waking up all\n", my_batch);
		complete_all(&batch_completed[this_dev][my_batch]);

		//wait for everyone to quit
		//pr_warn(" last %d: waiting for everyone to quit\n", my_batch);
		//wait_for_completion(&finalize_batch[this_dev][my_batch]);
		err = wait_for_completion_timeout(&batch_completed[this_dev][my_batch], usecs_to_jiffies((window_size_ns*10)/1000));
		if (err == 0) {
			//pr_warn("!!!!!!!!!!!!!!!!!!!!!!!!!!! LAST WAITED FOR TOO LONG\n");
		}
		
		//pr_warn(" last %d: done \n", my_batch);
reset_this_batch:
		if (use_cpu){
			my_prediction = cpu_prediction_model(feat_vec, n_vecs, weights);
			// if (waiting[this_dev][my_batch] >= cpu_gpu_threshold) {
				// pr_warn("<LAKE_trace> (id = %d) get the prediction result by CPU...\n", my_id);
			// }
		}
		// if (waiting[this_dev][my_batch] >= cpu_gpu_threshold) {
		// 	pr_warn("<LAKE_trace> (id = %d) reset the batch and complete All.\n", my_id);
		// }
		//reset
		waiting[this_dev][my_batch] = 0;
		batch_closed[this_dev][my_batch] = false;
		
		return no_reject ? false : my_prediction;
	}

	// for the request which is not the last one. Maybe this batch will never have a last, so we have to handle it. first may becomes last
	if (my_id == 0) {
		err = wait_for_completion_timeout(&batch_completed[this_dev][my_batch], usecs_to_jiffies((window_size_ns*100)/1000));
		//if this was a timeout, do what the last would to
		if(err == 0) {
			//pr_warn(" id0: timed out\n");
			//race condition: there is a chance id 0 woke up just after someone got in
			//and who got in could be last or not
			// if last, we have to wait
			// if not, we are the last
			spin_lock_irqsave(&per_batch_lock[this_dev][my_batch], irqflags);
			//someone closed this batch, so there is a last already
			if (batch_closed[this_dev][my_batch] == true) {
				//fall through
				//pr_warn(" !!!!!!!!: falling through, id0 timedout but there is last\n");
				spin_unlock_irqrestore(&per_batch_lock[this_dev][my_batch], irqflags);
			}
			//it's either only us or there are more, but they are just waiting
			else { 
				//pr_warn("!!!!!!!!!!!!!!!! id0 : becoming last \n");
				batch_closed[this_dev][my_batch] = true;
				spin_unlock_irqrestore(&per_batch_lock[this_dev][my_batch], irqflags);
				goto last_req_close;
			} 
		}
	}

	//wait until the last wake us up
	//wait_for_completion(&batch_completed[this_dev][my_batch]);
	err = wait_for_completion_timeout(&batch_completed[this_dev][my_batch], usecs_to_jiffies((window_size_ns*5)/1000));
	if (err == 0) {
		//fall through
		//pr_warn("!!!!!!!!!!!!!!!!!!!!!!!! THIS SHOULDNT HAVE HAPPENED  !! %d id %d\n", my_batch, my_id);
	}

	use_cpu = use_cpu_instead[this_dev][my_batch][my_id];
	if (!use_cpu) {
		my_prediction = gpu_get_prediction(this_dev, my_batch, my_id / GRANULARITY);
	}

	//spin_lock_irqsave(&per_batch_lock[this_dev][my_batch], irqflags);
	n_exited[this_dev][my_batch] += 1;
	//pr_warn("%d/%d/%d:  %d/%d left\n", this_dev, my_batch, my_id, n_exited[this_dev][my_batch], this_batch_size[this_dev][my_batch]);
	//we are the last one to exit, inform last
	if (n_exited[this_dev][my_batch] == waiting[this_dev][my_batch]) {
		complete(&finalize_batch[this_dev][my_batch]);
	}
	//spin_unlock_irqrestore(&per_batch_lock[this_dev][my_batch], irqflags);

	if (use_cpu){
		my_prediction = cpu_prediction_model(feat_vec, n_vecs, weights);
	}
			
	return no_reject ? false : my_prediction;
}


//hack: weights are actually device pointers here
void gpu_predict_batch(char *__feat_vec, int n_vecs, long **weights) {
	//do inference
	void *args[] = {
		&weights[0], &weights[2], &d_input_vec_i, &d_mid_res_i
	};
	void *args1[] = {
		&weights[1], &weights[3], &d_mid_res_i, &d_final_res_i
	};

    check_error(cuLaunchKernel(batch_linnos_mid_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);

    check_error(cuLaunchKernel(batch_linnos_final_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				64, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args1, NULL),
			"cuLaunchKernel", __LINE__);
	if(PREDICT_GPU_SYNC == 1) {
		check_error(cuCtxSynchronize(), "cudaDeviceSynchronize", __LINE__);
	}
}

void gpu_predict_batch_plus_1(char *__feat_vec, int n_vecs, long **weights) {
	//do inference
	void *args[] = {
		&weights[0], &weights[2], &d_input_vec_i, &d_mid_res_i
	};
	void *args1[] = {
		&weights[1], &weights[3], &d_mid_res_1_i, &d_final_res_i
	};

	void *args2[] = {
		&weights[4], &weights[5], &d_mid_res_i, &d_mid_res_1_i
	};
    check_error(cuLaunchKernel(batch_linnos_mid_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);

	check_error(cuLaunchKernel(batch_linnos_mid_layer_1_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args2, NULL),
			"cuLaunchKernel", __LINE__);

    check_error(cuLaunchKernel(batch_linnos_final_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				64, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args1, NULL),
			"cuLaunchKernel", __LINE__);
	if(PREDICT_GPU_SYNC == 1) {
		check_error(cuCtxSynchronize(), "cudaDeviceSynchronize", __LINE__);
	}
}

void gpu_predict_batch_plus_2(char *__feat_vec, int n_vecs, long **weights) {
	//do inference
	void *args[] = {
		&weights[0], &weights[2], &d_input_vec_i, &d_mid_res_i
	};
	void *args1[] = {
		&weights[1], &weights[3], &d_mid_res_2_i, &d_final_res_i
	};

	void *args2[] = {
		&weights[4], &weights[5], &d_mid_res_i, &d_mid_res_1_i
	};

	void *args3[] = {
		&weights[6], &weights[7], &d_mid_res_1_i, &d_mid_res_2_i
	};

    check_error(cuLaunchKernel(batch_linnos_mid_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);

	check_error(cuLaunchKernel(batch_linnos_mid_layer_1_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args2, NULL),
			"cuLaunchKernel", __LINE__);

	check_error(cuLaunchKernel(batch_linnos_mid_layer_2_kernel, 
				n_vecs, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args3, NULL),
			"cuLaunchKernel", __LINE__);

    check_error(cuLaunchKernel(batch_linnos_final_layer_kernel, 
				n_vecs, 1, 1,          //blocks
				64, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args1, NULL),
			"cuLaunchKernel", __LINE__);
	if(PREDICT_GPU_SYNC == 1) {
		check_error(cuCtxSynchronize(), "cudaDeviceSynchronize", __LINE__);
	}
}

bool fake_prediction_model(char *feat_vec, int n_vecs, long **weights) {
	//pr_warn("FAKE\n");
	return false;
}
//dont remove dead code
#pragma GCC push_options
#pragma GCC optimize (DEADFLAG)
bool cpu_prediction_model(char *feat_vec, int n_vecs, long **weights) {
	long input_vec_i[LEN_INPUT], mid_res_i[LEN_LAYER_0], final_res_i[LEN_LAYER_1];
	long *weight_0_T_ent, * bias_0_ent, *weight_1_T_ent, * bias_1_ent; 
	int i, j, k, offset;
	bool end;

	for (i=0 ; i<LEN_INPUT; i++) {
		input_vec_i[i] = (long)(feat_vec[i]);
	}

	weight_0_T_ent = weights[0];
	weight_1_T_ent = weights[1];
	bias_0_ent = weights[2];
	bias_1_ent = weights[3];

	for (j = 0, offset=0; j < LEN_LAYER_0; j++, offset+=LEN_INPUT) {
		mid_res_i[j] = 0;
		//loop unroll

		mid_res_i[j] += (input_vec_i[0] == 0 || weight_0_T_ent[offset+0] == 0)? 0 : input_vec_i[0] * weight_0_T_ent[offset+0];
		mid_res_i[j] += (input_vec_i[1] == 0 || weight_0_T_ent[offset+1] == 0)? 0 : input_vec_i[1] * weight_0_T_ent[offset+1];
		mid_res_i[j] += (input_vec_i[2] == 0 || weight_0_T_ent[offset+2] == 0)? 0 : input_vec_i[2] * weight_0_T_ent[offset+2];
		mid_res_i[j] += (input_vec_i[3] == 0 || weight_0_T_ent[offset+3] == 0)? 0 : input_vec_i[3] * weight_0_T_ent[offset+3];
		mid_res_i[j] += (input_vec_i[4] == 0 || weight_0_T_ent[offset+4] == 0)? 0 : input_vec_i[4] * weight_0_T_ent[offset+4];
		mid_res_i[j] += (input_vec_i[5] == 0 || weight_0_T_ent[offset+5] == 0)? 0 : input_vec_i[5] * weight_0_T_ent[offset+5];
		mid_res_i[j] += (input_vec_i[6] == 0 || weight_0_T_ent[offset+6] == 0)? 0 : input_vec_i[6] * weight_0_T_ent[offset+6];
		mid_res_i[j] += (input_vec_i[7] == 0 || weight_0_T_ent[offset+7] == 0)? 0 : input_vec_i[7] * weight_0_T_ent[offset+7];
		mid_res_i[j] += (input_vec_i[8] == 0 || weight_0_T_ent[offset+8] == 0)? 0 : input_vec_i[8] * weight_0_T_ent[offset+8];
		mid_res_i[j] += (input_vec_i[9] == 0 || weight_0_T_ent[offset+9] == 0)? 0 : input_vec_i[9] * weight_0_T_ent[offset+9];
		mid_res_i[j] += (input_vec_i[10] == 0 || weight_0_T_ent[offset+10] == 0)? 0 : input_vec_i[10] * weight_0_T_ent[offset+10];
		mid_res_i[j] += (input_vec_i[11] == 0 || weight_0_T_ent[offset+11] == 0)? 0 : input_vec_i[11] * weight_0_T_ent[offset+11];
		mid_res_i[j] += (input_vec_i[12] == 0 || weight_0_T_ent[offset+12] == 0)? 0 : input_vec_i[12] * weight_0_T_ent[offset+12];
		mid_res_i[j] += (input_vec_i[13] == 0 || weight_0_T_ent[offset+13] == 0)? 0 : input_vec_i[13] * weight_0_T_ent[offset+13];
		mid_res_i[j] += (input_vec_i[14] == 0 || weight_0_T_ent[offset+14] == 0)? 0 : input_vec_i[14] * weight_0_T_ent[offset+14];
		mid_res_i[j] += (input_vec_i[15] == 0 || weight_0_T_ent[offset+15] == 0)? 0 : input_vec_i[15] * weight_0_T_ent[offset+15];
		mid_res_i[j] += (input_vec_i[16] == 0 || weight_0_T_ent[offset+16] == 0)? 0 : input_vec_i[16] * weight_0_T_ent[offset+16];
		mid_res_i[j] += (input_vec_i[17] == 0 || weight_0_T_ent[offset+17] == 0)? 0 : input_vec_i[17] * weight_0_T_ent[offset+17];
		mid_res_i[j] += (input_vec_i[18] == 0 || weight_0_T_ent[offset+18] == 0)? 0 : input_vec_i[18] * weight_0_T_ent[offset+18];
		mid_res_i[j] += (input_vec_i[19] == 0 || weight_0_T_ent[offset+19] == 0)? 0 : input_vec_i[19] * weight_0_T_ent[offset+19];
		mid_res_i[j] += (input_vec_i[20] == 0 || weight_0_T_ent[offset+20] == 0)? 0 : input_vec_i[20] * weight_0_T_ent[offset+20];
		mid_res_i[j] += (input_vec_i[21] == 0 || weight_0_T_ent[offset+21] == 0)? 0 : input_vec_i[21] * weight_0_T_ent[offset+21];
		mid_res_i[j] += (input_vec_i[22] == 0 || weight_0_T_ent[offset+22] == 0)? 0 : input_vec_i[22] * weight_0_T_ent[offset+22];
		mid_res_i[j] += (input_vec_i[23] == 0 || weight_0_T_ent[offset+23] == 0)? 0 : input_vec_i[23] * weight_0_T_ent[offset+23];
		mid_res_i[j] += (input_vec_i[24] == 0 || weight_0_T_ent[offset+24] == 0)? 0 : input_vec_i[24] * weight_0_T_ent[offset+24];
		mid_res_i[j] += (input_vec_i[25] == 0 || weight_0_T_ent[offset+25] == 0)? 0 : input_vec_i[25] * weight_0_T_ent[offset+25];
		mid_res_i[j] += (input_vec_i[26] == 0 || weight_0_T_ent[offset+26] == 0)? 0 : input_vec_i[26] * weight_0_T_ent[offset+26];
		mid_res_i[j] += (input_vec_i[27] == 0 || weight_0_T_ent[offset+27] == 0)? 0 : input_vec_i[27] * weight_0_T_ent[offset+27];
		mid_res_i[j] += (input_vec_i[28] == 0 || weight_0_T_ent[offset+28] == 0)? 0 : input_vec_i[28] * weight_0_T_ent[offset+28];
		mid_res_i[j] += (input_vec_i[29] == 0 || weight_0_T_ent[offset+29] == 0)? 0 : input_vec_i[29] * weight_0_T_ent[offset+29];
		mid_res_i[j] += (input_vec_i[30] == 0 || weight_0_T_ent[offset+30] == 0)? 0 : input_vec_i[30] * weight_0_T_ent[offset+30];

		// apply bias
		mid_res_i[j] += bias_0_ent[j];
		// relu
		if (mid_res_i[j] < 0) {
			mid_res_i[j] = 0;
		}
	}
	
	final_res_i[0] = 0;
	for(k=0; k<LEN_LAYER_0; k += 8) {
		final_res_i[0] += (mid_res_i[k] == 0 || weight_1_T_ent[k] == 0)? 0 : mid_res_i[k] * weight_1_T_ent[k];
		final_res_i[0] += (mid_res_i[k+1] == 0 || weight_1_T_ent[k+1] == 0)? 0 : mid_res_i[k+1] * weight_1_T_ent[k+1];
		final_res_i[0] += (mid_res_i[k+2] == 0 || weight_1_T_ent[k+2] == 0)? 0 : mid_res_i[k+2] * weight_1_T_ent[k+2];
		final_res_i[0] += (mid_res_i[k+3] == 0 || weight_1_T_ent[k+3] == 0)? 0 : mid_res_i[k+3] * weight_1_T_ent[k+3];
		final_res_i[0] += (mid_res_i[k+4] == 0 || weight_1_T_ent[k+4] == 0)? 0 : mid_res_i[k+4] * weight_1_T_ent[k+4];
		final_res_i[0] += (mid_res_i[k+5] == 0 || weight_1_T_ent[k+5] == 0)? 0 : mid_res_i[k+5] * weight_1_T_ent[k+5];
		final_res_i[0] += (mid_res_i[k+6] == 0 || weight_1_T_ent[k+6] == 0)? 0 : mid_res_i[k+6] * weight_1_T_ent[k+6];
		final_res_i[0] += (mid_res_i[k+7] == 0 || weight_1_T_ent[k+7] == 0)? 0 : mid_res_i[k+7] * weight_1_T_ent[k+7];
	}
	// apply bias
	final_res_i[0] += bias_1_ent[0];

	final_res_i[1] = 0;
	for(k=0; k<LEN_LAYER_0; k += 8) {
		final_res_i[1] += (mid_res_i[k] == 0 || weight_1_T_ent[k+256] == 0)? 0 : mid_res_i[k] * weight_1_T_ent[k+256];
		final_res_i[1] += (mid_res_i[k+1] == 0 || weight_1_T_ent[k+257] == 0)? 0 : mid_res_i[k+1] * weight_1_T_ent[k+257];
		final_res_i[1] += (mid_res_i[k+2] == 0 || weight_1_T_ent[k+258] == 0)? 0 : mid_res_i[k+2] * weight_1_T_ent[k+258];
		final_res_i[1] += (mid_res_i[k+3] == 0 || weight_1_T_ent[k+259] == 0)? 0 : mid_res_i[k+3] * weight_1_T_ent[k+259];
		final_res_i[1] += (mid_res_i[k+4] == 0 || weight_1_T_ent[k+260] == 0)? 0 : mid_res_i[k+4] * weight_1_T_ent[k+260];
		final_res_i[1] += (mid_res_i[k+5] == 0 || weight_1_T_ent[k+261] == 0)? 0 : mid_res_i[k+5] * weight_1_T_ent[k+261];
		final_res_i[1] += (mid_res_i[k+6] == 0 || weight_1_T_ent[k+262] == 0)? 0 : mid_res_i[k+6] * weight_1_T_ent[k+262];
		final_res_i[1] += (mid_res_i[k+7] == 0 || weight_1_T_ent[k+263] == 0)? 0 : mid_res_i[k+7] * weight_1_T_ent[k+263];
	}
	// apply bias
	final_res_i[1] += bias_1_ent[1];
    end = (final_res_i[0]>=final_res_i[1])? false: true;
	return no_reject ? false : end; 
}

bool cpu_prediction_model_plus_1(char *feat_vec, int n_vecs, long **weights) {
	long input_vec_i[LEN_INPUT], mid_res_i[LEN_LAYER_0], mid_res_m_1[LEN_LAYER_M_1], final_res_i[LEN_LAYER_1];
	long *weight_0_T_ent, * bias_0_ent, *weight_1_T_ent, * bias_1_ent, *weight_M_1, *bias_M_1; 
	int i, j, k, offset;
	bool end;
	for (i=0 ; i<LEN_INPUT; i++) {
		input_vec_i[i] = (long)(feat_vec[i]);
	}

	weight_0_T_ent = weights[0];
	weight_1_T_ent = weights[1];
	bias_0_ent = weights[2];
	bias_1_ent = weights[3];

	weight_M_1 = weights[4];
	bias_M_1 = weights[5];

	for (j = 0, offset=0; j < LEN_LAYER_0; j++, offset+=LEN_INPUT) {
		mid_res_i[j] = 0;
		//loop unroll

		mid_res_i[j] += (input_vec_i[0] == 0 || weight_0_T_ent[offset+0] == 0)? 0 : input_vec_i[0] * weight_0_T_ent[offset+0];
		mid_res_i[j] += (input_vec_i[1] == 0 || weight_0_T_ent[offset+1] == 0)? 0 : input_vec_i[1] * weight_0_T_ent[offset+1];
		mid_res_i[j] += (input_vec_i[2] == 0 || weight_0_T_ent[offset+2] == 0)? 0 : input_vec_i[2] * weight_0_T_ent[offset+2];
		mid_res_i[j] += (input_vec_i[3] == 0 || weight_0_T_ent[offset+3] == 0)? 0 : input_vec_i[3] * weight_0_T_ent[offset+3];
		mid_res_i[j] += (input_vec_i[4] == 0 || weight_0_T_ent[offset+4] == 0)? 0 : input_vec_i[4] * weight_0_T_ent[offset+4];
		mid_res_i[j] += (input_vec_i[5] == 0 || weight_0_T_ent[offset+5] == 0)? 0 : input_vec_i[5] * weight_0_T_ent[offset+5];
		mid_res_i[j] += (input_vec_i[6] == 0 || weight_0_T_ent[offset+6] == 0)? 0 : input_vec_i[6] * weight_0_T_ent[offset+6];
		mid_res_i[j] += (input_vec_i[7] == 0 || weight_0_T_ent[offset+7] == 0)? 0 : input_vec_i[7] * weight_0_T_ent[offset+7];
		mid_res_i[j] += (input_vec_i[8] == 0 || weight_0_T_ent[offset+8] == 0)? 0 : input_vec_i[8] * weight_0_T_ent[offset+8];
		mid_res_i[j] += (input_vec_i[9] == 0 || weight_0_T_ent[offset+9] == 0)? 0 : input_vec_i[9] * weight_0_T_ent[offset+9];
		mid_res_i[j] += (input_vec_i[10] == 0 || weight_0_T_ent[offset+10] == 0)? 0 : input_vec_i[10] * weight_0_T_ent[offset+10];
		mid_res_i[j] += (input_vec_i[11] == 0 || weight_0_T_ent[offset+11] == 0)? 0 : input_vec_i[11] * weight_0_T_ent[offset+11];
		mid_res_i[j] += (input_vec_i[12] == 0 || weight_0_T_ent[offset+12] == 0)? 0 : input_vec_i[12] * weight_0_T_ent[offset+12];
		mid_res_i[j] += (input_vec_i[13] == 0 || weight_0_T_ent[offset+13] == 0)? 0 : input_vec_i[13] * weight_0_T_ent[offset+13];
		mid_res_i[j] += (input_vec_i[14] == 0 || weight_0_T_ent[offset+14] == 0)? 0 : input_vec_i[14] * weight_0_T_ent[offset+14];
		mid_res_i[j] += (input_vec_i[15] == 0 || weight_0_T_ent[offset+15] == 0)? 0 : input_vec_i[15] * weight_0_T_ent[offset+15];
		mid_res_i[j] += (input_vec_i[16] == 0 || weight_0_T_ent[offset+16] == 0)? 0 : input_vec_i[16] * weight_0_T_ent[offset+16];
		mid_res_i[j] += (input_vec_i[17] == 0 || weight_0_T_ent[offset+17] == 0)? 0 : input_vec_i[17] * weight_0_T_ent[offset+17];
		mid_res_i[j] += (input_vec_i[18] == 0 || weight_0_T_ent[offset+18] == 0)? 0 : input_vec_i[18] * weight_0_T_ent[offset+18];
		mid_res_i[j] += (input_vec_i[19] == 0 || weight_0_T_ent[offset+19] == 0)? 0 : input_vec_i[19] * weight_0_T_ent[offset+19];
		mid_res_i[j] += (input_vec_i[20] == 0 || weight_0_T_ent[offset+20] == 0)? 0 : input_vec_i[20] * weight_0_T_ent[offset+20];
		mid_res_i[j] += (input_vec_i[21] == 0 || weight_0_T_ent[offset+21] == 0)? 0 : input_vec_i[21] * weight_0_T_ent[offset+21];
		mid_res_i[j] += (input_vec_i[22] == 0 || weight_0_T_ent[offset+22] == 0)? 0 : input_vec_i[22] * weight_0_T_ent[offset+22];
		mid_res_i[j] += (input_vec_i[23] == 0 || weight_0_T_ent[offset+23] == 0)? 0 : input_vec_i[23] * weight_0_T_ent[offset+23];
		mid_res_i[j] += (input_vec_i[24] == 0 || weight_0_T_ent[offset+24] == 0)? 0 : input_vec_i[24] * weight_0_T_ent[offset+24];
		mid_res_i[j] += (input_vec_i[25] == 0 || weight_0_T_ent[offset+25] == 0)? 0 : input_vec_i[25] * weight_0_T_ent[offset+25];
		mid_res_i[j] += (input_vec_i[26] == 0 || weight_0_T_ent[offset+26] == 0)? 0 : input_vec_i[26] * weight_0_T_ent[offset+26];
		mid_res_i[j] += (input_vec_i[27] == 0 || weight_0_T_ent[offset+27] == 0)? 0 : input_vec_i[27] * weight_0_T_ent[offset+27];
		mid_res_i[j] += (input_vec_i[28] == 0 || weight_0_T_ent[offset+28] == 0)? 0 : input_vec_i[28] * weight_0_T_ent[offset+28];
		mid_res_i[j] += (input_vec_i[29] == 0 || weight_0_T_ent[offset+29] == 0)? 0 : input_vec_i[29] * weight_0_T_ent[offset+29];
		mid_res_i[j] += (input_vec_i[30] == 0 || weight_0_T_ent[offset+30] == 0)? 0 : input_vec_i[30] * weight_0_T_ent[offset+30];

		// apply bias
		mid_res_i[j] += bias_0_ent[j];
		// relu
		if (mid_res_i[j] < 0) {
			mid_res_i[j] = 0;
		}
	}

	for (j = 0; j < LEN_LAYER_M_1; j++) {
		mid_res_m_1[j] = 0;
		for(int off = 0; off < LEN_LAYER_0; off++) {
			mid_res_m_1[j] += mid_res_i[off]*weight_M_1[j * LEN_LAYER_M_1 + off];
		}

		// apply bias
		mid_res_m_1[j] += bias_M_1[j];
		// relu
		if (mid_res_m_1[j] < 0) {
			mid_res_m_1[j] = 0;
		}
	 }
	
	final_res_i[0] = 0;
	for(k=0; k<LEN_LAYER_0; k += 8) {
		final_res_i[0] += (mid_res_m_1[k] == 0 || weight_1_T_ent[k] == 0)? 0 : mid_res_m_1[k] * weight_1_T_ent[k];
		final_res_i[0] += (mid_res_m_1[k+1] == 0 || weight_1_T_ent[k+1] == 0)? 0 : mid_res_m_1[k+1] * weight_1_T_ent[k+1];
		final_res_i[0] += (mid_res_m_1[k+2] == 0 || weight_1_T_ent[k+2] == 0)? 0 : mid_res_m_1[k+2] * weight_1_T_ent[k+2];
		final_res_i[0] += (mid_res_m_1[k+3] == 0 || weight_1_T_ent[k+3] == 0)? 0 : mid_res_m_1[k+3] * weight_1_T_ent[k+3];
		final_res_i[0] += (mid_res_m_1[k+4] == 0 || weight_1_T_ent[k+4] == 0)? 0 : mid_res_m_1[k+4] * weight_1_T_ent[k+4];
		final_res_i[0] += (mid_res_m_1[k+5] == 0 || weight_1_T_ent[k+5] == 0)? 0 : mid_res_m_1[k+5] * weight_1_T_ent[k+5];
		final_res_i[0] += (mid_res_m_1[k+6] == 0 || weight_1_T_ent[k+6] == 0)? 0 : mid_res_m_1[k+6] * weight_1_T_ent[k+6];
		final_res_i[0] += (mid_res_m_1[k+7] == 0 || weight_1_T_ent[k+7] == 0)? 0 : mid_res_m_1[k+7] * weight_1_T_ent[k+7];
	}
	// apply bias
	final_res_i[0] += bias_1_ent[0];

	final_res_i[1] = 0;
	for(k=0; k<LEN_LAYER_0; k += 8) {
		final_res_i[1] += (mid_res_m_1[k] == 0 || weight_1_T_ent[k+256] == 0)? 0 : mid_res_m_1[k] * weight_1_T_ent[k+256];
		final_res_i[1] += (mid_res_m_1[k+1] == 0 || weight_1_T_ent[k+257] == 0)? 0 : mid_res_m_1[k+1] * weight_1_T_ent[k+257];
		final_res_i[1] += (mid_res_m_1[k+2] == 0 || weight_1_T_ent[k+258] == 0)? 0 : mid_res_m_1[k+2] * weight_1_T_ent[k+258];
		final_res_i[1] += (mid_res_m_1[k+3] == 0 || weight_1_T_ent[k+259] == 0)? 0 : mid_res_m_1[k+3] * weight_1_T_ent[k+259];
		final_res_i[1] += (mid_res_m_1[k+4] == 0 || weight_1_T_ent[k+260] == 0)? 0 : mid_res_m_1[k+4] * weight_1_T_ent[k+260];
		final_res_i[1] += (mid_res_m_1[k+5] == 0 || weight_1_T_ent[k+261] == 0)? 0 : mid_res_m_1[k+5] * weight_1_T_ent[k+261];
		final_res_i[1] += (mid_res_m_1[k+6] == 0 || weight_1_T_ent[k+262] == 0)? 0 : mid_res_m_1[k+6] * weight_1_T_ent[k+262];
		final_res_i[1] += (mid_res_m_1[k+7] == 0 || weight_1_T_ent[k+263] == 0)? 0 : mid_res_m_1[k+7] * weight_1_T_ent[k+263];
	}
	// apply bias
	final_res_i[1] += bias_1_ent[1];

    //return (final_res_i[0]>=final_res_i[1])? false: true;

	end = (final_res_i[0]>=final_res_i[1])? false: true;
	return no_reject ? false : end; 
}

bool cpu_prediction_model_plus_2(char *feat_vec, int n_vecs, long **weights) {
	long input_vec_i[LEN_INPUT], mid_res_i[LEN_LAYER_0], mid_res_m_1[LEN_LAYER_M_1], mid_res_m_2[LEN_LAYER_M_2], final_res_i[LEN_LAYER_1];
	long *weight_0_T_ent, * bias_0_ent, *weight_1_T_ent, * bias_1_ent, *weight_M_1, *bias_M_1, *weight_M_2, *bias_M_2; 
	int i, j, k, offset;
	bool end;
	for (i=0 ; i<LEN_INPUT; i++) {
		input_vec_i[i] = (long)(feat_vec[i]);
	}

	weight_0_T_ent = weights[0];
	weight_1_T_ent = weights[1];
	bias_0_ent = weights[2];
	bias_1_ent = weights[3];

	weight_M_1 = weights[4];
	bias_M_1 = weights[5];

	weight_M_2 = weights[6];
	bias_M_2 = weights[7];

	for (j = 0, offset=0; j < LEN_LAYER_0; j++, offset+=LEN_INPUT) {
		mid_res_i[j] = 0;
		//loop unroll

		mid_res_i[j] += (input_vec_i[0] == 0 || weight_0_T_ent[offset+0] == 0)? 0 : input_vec_i[0] * weight_0_T_ent[offset+0];
		mid_res_i[j] += (input_vec_i[1] == 0 || weight_0_T_ent[offset+1] == 0)? 0 : input_vec_i[1] * weight_0_T_ent[offset+1];
		mid_res_i[j] += (input_vec_i[2] == 0 || weight_0_T_ent[offset+2] == 0)? 0 : input_vec_i[2] * weight_0_T_ent[offset+2];
		mid_res_i[j] += (input_vec_i[3] == 0 || weight_0_T_ent[offset+3] == 0)? 0 : input_vec_i[3] * weight_0_T_ent[offset+3];
		mid_res_i[j] += (input_vec_i[4] == 0 || weight_0_T_ent[offset+4] == 0)? 0 : input_vec_i[4] * weight_0_T_ent[offset+4];
		mid_res_i[j] += (input_vec_i[5] == 0 || weight_0_T_ent[offset+5] == 0)? 0 : input_vec_i[5] * weight_0_T_ent[offset+5];
		mid_res_i[j] += (input_vec_i[6] == 0 || weight_0_T_ent[offset+6] == 0)? 0 : input_vec_i[6] * weight_0_T_ent[offset+6];
		mid_res_i[j] += (input_vec_i[7] == 0 || weight_0_T_ent[offset+7] == 0)? 0 : input_vec_i[7] * weight_0_T_ent[offset+7];
		mid_res_i[j] += (input_vec_i[8] == 0 || weight_0_T_ent[offset+8] == 0)? 0 : input_vec_i[8] * weight_0_T_ent[offset+8];
		mid_res_i[j] += (input_vec_i[9] == 0 || weight_0_T_ent[offset+9] == 0)? 0 : input_vec_i[9] * weight_0_T_ent[offset+9];
		mid_res_i[j] += (input_vec_i[10] == 0 || weight_0_T_ent[offset+10] == 0)? 0 : input_vec_i[10] * weight_0_T_ent[offset+10];
		mid_res_i[j] += (input_vec_i[11] == 0 || weight_0_T_ent[offset+11] == 0)? 0 : input_vec_i[11] * weight_0_T_ent[offset+11];
		mid_res_i[j] += (input_vec_i[12] == 0 || weight_0_T_ent[offset+12] == 0)? 0 : input_vec_i[12] * weight_0_T_ent[offset+12];
		mid_res_i[j] += (input_vec_i[13] == 0 || weight_0_T_ent[offset+13] == 0)? 0 : input_vec_i[13] * weight_0_T_ent[offset+13];
		mid_res_i[j] += (input_vec_i[14] == 0 || weight_0_T_ent[offset+14] == 0)? 0 : input_vec_i[14] * weight_0_T_ent[offset+14];
		mid_res_i[j] += (input_vec_i[15] == 0 || weight_0_T_ent[offset+15] == 0)? 0 : input_vec_i[15] * weight_0_T_ent[offset+15];
		mid_res_i[j] += (input_vec_i[16] == 0 || weight_0_T_ent[offset+16] == 0)? 0 : input_vec_i[16] * weight_0_T_ent[offset+16];
		mid_res_i[j] += (input_vec_i[17] == 0 || weight_0_T_ent[offset+17] == 0)? 0 : input_vec_i[17] * weight_0_T_ent[offset+17];
		mid_res_i[j] += (input_vec_i[18] == 0 || weight_0_T_ent[offset+18] == 0)? 0 : input_vec_i[18] * weight_0_T_ent[offset+18];
		mid_res_i[j] += (input_vec_i[19] == 0 || weight_0_T_ent[offset+19] == 0)? 0 : input_vec_i[19] * weight_0_T_ent[offset+19];
		mid_res_i[j] += (input_vec_i[20] == 0 || weight_0_T_ent[offset+20] == 0)? 0 : input_vec_i[20] * weight_0_T_ent[offset+20];
		mid_res_i[j] += (input_vec_i[21] == 0 || weight_0_T_ent[offset+21] == 0)? 0 : input_vec_i[21] * weight_0_T_ent[offset+21];
		mid_res_i[j] += (input_vec_i[22] == 0 || weight_0_T_ent[offset+22] == 0)? 0 : input_vec_i[22] * weight_0_T_ent[offset+22];
		mid_res_i[j] += (input_vec_i[23] == 0 || weight_0_T_ent[offset+23] == 0)? 0 : input_vec_i[23] * weight_0_T_ent[offset+23];
		mid_res_i[j] += (input_vec_i[24] == 0 || weight_0_T_ent[offset+24] == 0)? 0 : input_vec_i[24] * weight_0_T_ent[offset+24];
		mid_res_i[j] += (input_vec_i[25] == 0 || weight_0_T_ent[offset+25] == 0)? 0 : input_vec_i[25] * weight_0_T_ent[offset+25];
		mid_res_i[j] += (input_vec_i[26] == 0 || weight_0_T_ent[offset+26] == 0)? 0 : input_vec_i[26] * weight_0_T_ent[offset+26];
		mid_res_i[j] += (input_vec_i[27] == 0 || weight_0_T_ent[offset+27] == 0)? 0 : input_vec_i[27] * weight_0_T_ent[offset+27];
		mid_res_i[j] += (input_vec_i[28] == 0 || weight_0_T_ent[offset+28] == 0)? 0 : input_vec_i[28] * weight_0_T_ent[offset+28];
		mid_res_i[j] += (input_vec_i[29] == 0 || weight_0_T_ent[offset+29] == 0)? 0 : input_vec_i[29] * weight_0_T_ent[offset+29];
		mid_res_i[j] += (input_vec_i[30] == 0 || weight_0_T_ent[offset+30] == 0)? 0 : input_vec_i[30] * weight_0_T_ent[offset+30];

		// apply bias
		mid_res_i[j] += bias_0_ent[j];
		// relu
		if (mid_res_i[j] < 0) {
			mid_res_i[j] = 0;
		}
	}

	for (j = 0; j < LEN_LAYER_M_1; j++) {
		mid_res_m_1[j] = 0;
		for(int off = 0; off < LEN_LAYER_0; off++) {
			mid_res_m_1[j] += mid_res_i[off]*weight_M_1[j * LEN_LAYER_M_1 + off];
		}

		// apply bias
		mid_res_m_1[j] += bias_M_1[j];
		// relu
		if (mid_res_m_1[j] < 0) {
			mid_res_m_1[j] = 0;
		}
	 }

	 for (j = 0; j < LEN_LAYER_M_2; j++) {
		mid_res_m_2[j] = 0;
		for(int off = 0; off < LEN_LAYER_M_1; off++) {
			mid_res_m_2[j] += mid_res_m_1[off]*weight_M_2[j * LEN_LAYER_M_2 + off];
		}

		// apply bias
		mid_res_m_2[j] += bias_M_2[j];
		// relu
		if (mid_res_m_2[j] < 0) {
			mid_res_m_2[j] = 0;
		}
	 }
	
	final_res_i[0] = 0;
	for(k=0; k<LEN_LAYER_0; k += 8) {
		final_res_i[0] += (mid_res_m_2[k] == 0 || weight_1_T_ent[k] == 0)? 0 : mid_res_m_2[k] * weight_1_T_ent[k];
		final_res_i[0] += (mid_res_m_2[k+1] == 0 || weight_1_T_ent[k+1] == 0)? 0 : mid_res_m_2[k+1] * weight_1_T_ent[k+1];
		final_res_i[0] += (mid_res_m_2[k+2] == 0 || weight_1_T_ent[k+2] == 0)? 0 : mid_res_m_2[k+2] * weight_1_T_ent[k+2];
		final_res_i[0] += (mid_res_m_2[k+3] == 0 || weight_1_T_ent[k+3] == 0)? 0 : mid_res_m_2[k+3] * weight_1_T_ent[k+3];
		final_res_i[0] += (mid_res_m_2[k+4] == 0 || weight_1_T_ent[k+4] == 0)? 0 : mid_res_m_2[k+4] * weight_1_T_ent[k+4];
		final_res_i[0] += (mid_res_m_2[k+5] == 0 || weight_1_T_ent[k+5] == 0)? 0 : mid_res_m_2[k+5] * weight_1_T_ent[k+5];
		final_res_i[0] += (mid_res_m_2[k+6] == 0 || weight_1_T_ent[k+6] == 0)? 0 : mid_res_m_2[k+6] * weight_1_T_ent[k+6];
		final_res_i[0] += (mid_res_m_2[k+7] == 0 || weight_1_T_ent[k+7] == 0)? 0 : mid_res_m_2[k+7] * weight_1_T_ent[k+7];
	}
	// apply bias
	final_res_i[0] += bias_1_ent[0];

	final_res_i[1] = 0;
	for(k=0; k<LEN_LAYER_0; k += 8) {
		final_res_i[1] += (mid_res_m_2[k] == 0 || weight_1_T_ent[k+256] == 0)? 0 : mid_res_m_2[k] * weight_1_T_ent[k+256];
		final_res_i[1] += (mid_res_m_2[k+1] == 0 || weight_1_T_ent[k+257] == 0)? 0 : mid_res_m_2[k+1] * weight_1_T_ent[k+257];
		final_res_i[1] += (mid_res_m_2[k+2] == 0 || weight_1_T_ent[k+258] == 0)? 0 : mid_res_m_2[k+2] * weight_1_T_ent[k+258];
		final_res_i[1] += (mid_res_m_2[k+3] == 0 || weight_1_T_ent[k+259] == 0)? 0 : mid_res_m_2[k+3] * weight_1_T_ent[k+259];
		final_res_i[1] += (mid_res_m_2[k+4] == 0 || weight_1_T_ent[k+260] == 0)? 0 : mid_res_m_2[k+4] * weight_1_T_ent[k+260];
		final_res_i[1] += (mid_res_m_2[k+5] == 0 || weight_1_T_ent[k+261] == 0)? 0 : mid_res_m_2[k+5] * weight_1_T_ent[k+261];
		final_res_i[1] += (mid_res_m_2[k+6] == 0 || weight_1_T_ent[k+262] == 0)? 0 : mid_res_m_2[k+6] * weight_1_T_ent[k+262];
		final_res_i[1] += (mid_res_m_2[k+7] == 0 || weight_1_T_ent[k+263] == 0)? 0 : mid_res_m_2[k+7] * weight_1_T_ent[k+263];
	}
	// apply bias
	final_res_i[1] += bias_1_ent[1];

    //return (final_res_i[0]>=final_res_i[1])? false: true;
	end = (final_res_i[0]>=final_res_i[1])? false: true;
	return no_reject ? false : end; 
}
#pragma GCC pop_options
bool batch_test(char *feat_vec, int n_vecs, long **weights) {
	return false;
}