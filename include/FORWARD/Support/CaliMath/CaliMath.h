//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <execinfo.h>
#include <stdlib.h>
#include <math.h>

#define MULTI_THREAD_KL_CALC
#ifdef MULTI_THREAD_KL_CALC
#include <pthread.h>
#include <time.h>
#endif

namespace CaliMath {

#ifdef MULTI_THREAD_KL_CALC
struct mul_thread_inputs{
  long long* hist;
  float* kl;
  long long count;
  long long i;
  long long N;
  long long BINS;
};
#endif

static inline void print_trace()
{
  void *array[10];
  size_t i;

  size_t size = backtrace (array, 10);
  char **strings = backtrace_symbols (array, size);

  printf ("Obtained %lu stack frames.\n", size);

  for (i = 0; i < size; i++)
    printf ("%s\n", strings[i]);

  free (strings);
}

static inline void hang(long long ret)
{
  exit(ret);
}

#define ASSERT(_cond)                                   \
  do {                                                  \
    if (!(_cond)) {                                     \
      printf("ASSERT %s: %d: %s: %s\n",                 \
             __FILE__, __LINE__, __func__, #_cond);     \
      print_trace();                                    \
      fflush(stdout);                                   \
      hang(-1);                                         \
    }                                                   \
  } while(0)

inline float the_max(float *data, long long count) {
  ASSERT(count > 0);
  ASSERT(data != NULL);
  float a_max = fabs(data[0]);
  for (long long i = 1; i < count; i++) {
    a_max = (a_max < fabs(data[i])) ? fabs(data[i]) : a_max;
  }
  return a_max;
}

inline long long the_min_index(float *data, long long count) {
  ASSERT(data != NULL);
  float a_max = data[0];
  long long min_index = 0;
  for (long long i = 1; i < count; i++) {
    if (a_max > data[i]) {
      a_max = data[i];
      min_index = i;
    }
  }
  return min_index;
}

float real_kl_diversity(float *data, long long count);

#ifdef MULTI_THREAD_KL_CALC
// void* kl_calc_thread(void* args_input);
float real_multi_thread_kl_diversity(float *data, long long count, const long long num_bins);
float real_multi_thread_kl_diversity_hist(int *data, float &width, const long long N);
#endif

float kl_diversity(float *data, long long count, long long num_bins);
float kl_diversity_hist(int *data, float width, long long num_bins);

}
