/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

__global__
void min_max_reduce(const float* const d_logLuminance,
                    float* const d_min_logLum,
                    float* const d_max_logLum,
                    int numPixels) {
  int pos = blockDim.x * blockIdx.x + threadIdx.x;
  extern __shared__ float sdata[];  // blockDim size * 2
  float* sdataMin = sdata;  // blockDim size
  float* sdataMax = sdata + blockDim.x;  // blockDim size
  const int tid = threadIdx.x;
  float pixVal;
  if (pos >= numPixels) {
    pos = numPixels - 1;
  }
  pixVal = d_logLuminance[pos];
  sdataMin[tid] = pixVal;
  sdataMax[tid] = pixVal;
  __syncthreads();

  for (unsigned int i = blockDim.x / 2; i > 0; i = i >> 1) {
    if (tid < i) {
      sdataMin[tid] = fmin(sdataMin[tid], sdataMin[tid + i]);
      sdataMax[tid] = fmax(sdataMax[tid], sdataMax[tid + i]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_min_logLum[blockIdx.x] = sdataMin[0];
    d_max_logLum[blockIdx.x] = sdataMax[0];
  }
}

__global__
void min_max_reduce_2nd(const float* const d_in,
                        float* const d_out, int isMin) {
  const int pos = blockDim.x * blockIdx.x + threadIdx.x;
  extern __shared__ float sdata[];  // blockDim size
  const int tid = threadIdx.x;
  sdata[tid] = d_in[pos];
  __syncthreads();

  for (unsigned int i = blockDim.x / 2; i > 0; i = i >> 1) {
    if (tid < i) {
      if (isMin == 1) {
        sdata[tid] = fmin(sdata[tid], sdata[tid + i]);
      } else {
        sdata[tid] = fmax(sdata[tid], sdata[tid + i]);
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    *d_out = sdata[0];
  }
}

const int MAX_THREADS_PER_BLOCK = 1024;

void find_min_max(const float* const d_logLuminance,
                  float &min_logLum, float &max_logLum,
                  const size_t numPixels) {
  if (MAX_THREADS_PER_BLOCK * MAX_THREADS_PER_BLOCK < numPixels) {
    std::cerr << "Too large picture: " << numPixels << std::endl;
    exit(1);
  }
  // 1st step reduce inside block
  int numThreads = MAX_THREADS_PER_BLOCK;
  int numBlocks = (numPixels + numThreads) / numThreads;
  size_t shmSize = numThreads * sizeof(float) * 2;  // sum of 2 arrays
  // Array to store intermediate result
  float* d_min_logLum_mid;  
  float* d_max_logLum_mid;
  checkCudaErrors(cudaMalloc(&d_min_logLum_mid, sizeof(float) * numBlocks));
  checkCudaErrors(cudaMalloc(&d_max_logLum_mid, sizeof(float) * numBlocks));
  checkCudaErrors(cudaMemset(d_min_logLum_mid, 0.f, sizeof(float) * numBlocks));
  checkCudaErrors(cudaMemset(d_max_logLum_mid, 0.f, sizeof(float) * numBlocks));

  min_max_reduce<<<numBlocks, numThreads, shmSize>>>
        (d_logLuminance, d_min_logLum_mid, d_max_logLum_mid, numPixels);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // 2nd step, reduce all the blocks into 1
  numThreads = numBlocks;
  numBlocks = 1;
  shmSize = numThreads * sizeof(float);
  // To store final result
  float* d_min_logLum;
  float* d_max_logLum;
  checkCudaErrors(cudaMalloc(&d_min_logLum, sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_max_logLum, sizeof(float)));
  checkCudaErrors(cudaMemset(d_min_logLum, 0.f, sizeof(float)));
  checkCudaErrors(cudaMemset(d_max_logLum, 0.f, sizeof(float)));

  min_max_reduce_2nd<<<numBlocks, numThreads, shmSize>>>
        (d_min_logLum_mid, d_min_logLum, 1);
  min_max_reduce_2nd<<<numBlocks, numThreads, shmSize>>>
        (d_max_logLum_mid, d_max_logLum, 0);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(&min_logLum, d_min_logLum,
                             sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&max_logLum, d_max_logLum,
                             sizeof(float), cudaMemcpyDeviceToHost));

  // Cleanup
  checkCudaErrors(cudaFree(d_min_logLum_mid));
  checkCudaErrors(cudaFree(d_max_logLum_mid));
  checkCudaErrors(cudaFree(d_min_logLum));
  checkCudaErrors(cudaFree(d_max_logLum));
}

__global__
void histo_kernel_naive(const float* const d_lum,
                        unsigned int* d_histo,
                        const float lumMin,
                        const float lumRange,
                        const int numPixels,
                        const size_t numBins) {
  const int pos = blockDim.x * blockIdx.x + threadIdx.x;
  if (pos >= numPixels) return;

  unsigned int bin = (d_lum[pos] - lumMin) / lumRange * numBins;
  if (bin >= numBins) bin = numBins - 1;
  atomicAdd(&d_histo[bin], 1);
}

__global__
void histo_kernel_shared(const float* const d_lum,
                         unsigned int* d_histo,
                         const float lumMin,
                         const float lumRange,
                         const int numPixels,
                         const size_t numBins) {
  extern __shared__ int s_histo[];
  const int pos = blockDim.x * blockIdx.x + threadIdx.x;
  const int tid = threadIdx.x;

  s_histo[tid] = 0;
  __syncthreads();

  if (pos < numPixels) {
    unsigned int bin = (d_lum[pos] - lumMin) / lumRange * numBins;
    if (bin >= numBins) bin = numBins - 1;
    atomicAdd(&s_histo[bin], 1);
  }
  __syncthreads();

  atomicAdd(&d_histo[tid], s_histo[tid]);
}

void calc_histo(const float* const d_logLuminance,
                unsigned int** d_histo,
                const float min_logLum,
                const float range,
                const size_t numPixels,
                const size_t numBins) {
  checkCudaErrors(cudaMalloc(d_histo, sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemset(*d_histo, 0, sizeof(unsigned int) * numBins));

  const int numThreads = numBins;
  const int numBlocks = (numPixels + numThreads - 1) / numThreads;
  // histo_kernel_naive<<<numBlocks, numThreads>>>
  //       (d_logLuminance, *d_histo, min_logLum, range, numPixels, numBins);

  const size_t shmSize = sizeof(unsigned int) * numBins;
  histo_kernel_shared<<<numBlocks, numThreads, shmSize>>>
        (d_logLuminance, *d_histo, min_logLum, range, numPixels, numBins);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  // int histo[numBins];
  // checkCudaErrors(cudaMemcpy(histo, *d_histo, sizeof(int) * numBins, cudaMemcpyDeviceToHost));

  // std::cout << "hist: " << std::endl;
  // for (int i = 0; i < (int)numBins; ++i) 
  //   std::cout << histo[i] << ", " ;
  // std::cout << std::endl;
}

__global__
void prefix_sum(const unsigned int* const d_in,
                unsigned int* const d_sum,
                const size_t num) {
  // Exclusive prefix sum by Hillis and Steele algorithm
  // GPU Gems chapter 39
  extern __shared__ unsigned int s_buf[];  // double buffer
  unsigned int* bufs[2] = {&s_buf[0], &s_buf[num]};
  int toggle = 1;
  unsigned int* inbuf = bufs[0];
  unsigned int* outbuf = bufs[1];
  const int tid = threadIdx.x;

  inbuf[tid] = tid > 0 ? d_in[tid - 1] : 0;
  __syncthreads();

  for (int offset = 1; offset < num; offset *= 2) {
    toggle = 1 - toggle;
    inbuf = bufs[toggle];
    outbuf = bufs[1 - toggle];

    if (tid >= offset) {
      outbuf[tid] = inbuf[tid] + inbuf[tid - offset];
    } else {
      outbuf[tid] = inbuf[tid];
    }
    __syncthreads();
  }

  d_sum[tid] = outbuf[tid];
}

void calc_cdf(const unsigned int* const d_histo,
              unsigned int* const d_cdf,
              const size_t numBins) {
  if (numBins > MAX_THREADS_PER_BLOCK) {
    std::cerr << "Too large bins: " << numBins << std::endl;
    exit(1);
  }
  size_t shmSize = sizeof(unsigned int) * numBins * 2;  // Double Buffer
  prefix_sum<<<1, numBins, shmSize>>>(d_histo, d_cdf, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  // 1) find the minimum and maximum value in the input logLuminance channel
  //    store in min_logLum and max_logLum
  find_min_max(d_logLuminance, min_logLum, max_logLum, numRows * numCols);

  // 2) subtract them to find the range
  float range_logLum = max_logLum - min_logLum;
  std::cout << "range: " << range_logLum << std::endl;

  // 3) generate a histogram of all the values in the logLuminance channel using
  //    the formula: bin = (lum[i] - lumMin) / lumRange * numBins
  unsigned int* d_histo;
  calc_histo(d_logLuminance, &d_histo, min_logLum, range_logLum,
             numRows * numCols, numBins);

  // 4) Perform an exclusive scan (prefix sum) on the histogram to get
  //    the cumulative distribution of luminance values (this should go in the
  //    incoming d_cdf pointer which already has been allocated for you)
  calc_cdf(d_histo, d_cdf, numBins);

  // 5) cleanup
  checkCudaErrors(cudaFree(d_histo));
}
