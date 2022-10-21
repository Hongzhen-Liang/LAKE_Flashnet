#ifndef __LINNOS_PREDICTORS_H
#define __LINNOS_PREDICTORS_H

#define FEAT_31
#define LEN_INPUT 31
#define LEN_LAYER_0 256
#define LEN_LAYER_0_HALF 128
#define LEN_LAYER_1 2

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/completion.h>
#else
#include <stdbool.h>
#endif

#include "variables.h"

#ifdef __KERNEL__
//these externs are for batching
extern bool* gpu_results;
extern u32* window_size_hist;
extern u32 n_used_gpu;
bool batch_test(char *feat_vec, int n_vecs, long **weights);

extern struct GPU_weights gpu_weights[3];
#endif

bool fake_prediction_model(char *feat_vec, int n_vecs, long **weights);
bool gpu_batch_entry(char *feat_vec, int n_vecs, long **weights);
bool cpu_prediction_model(char *feat_vec, int n_vecs, long **weights);
void gpu_predict_batch(char *__feat_vec, int n_vecs, long **weights);

void predictors_mgpu_init(void);

extern int PREDICT_GPU_SYNC;

#endif