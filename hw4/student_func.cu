//Udacity HW 4
//Radix Sorting

#include <stdio.h>
#include "utils.h"
// #include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__
void predicate_kernel(const uint32_t* const d_in,
                      uint8_t* d_predicate,
                      int bit,
                      const size_t num) {
    const int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos >= num) return;

    // True if bit of element is 0
    uint32_t match = d_in[pos] & (1 << bit);
    d_predicate[pos] = match ? 0 : 1;
}

__global__
void count_kernel(const uint8_t* d_predicate,
                  uint32_t* d_count,
                  const size_t num) {    // num needs to be power of 2 
    const int pos = blockDim.x * blockIdx.x + threadIdx.x;

    // Sum (reduce) within each block
    extern __shared__ uint32_t sdata[];  // blockDim size
    const int tid = threadIdx.x;
    sdata[tid] = (pos < num) ? d_predicate[pos] : 0;
    __syncthreads();

    for (uint32_t offset = blockDim.x / 2; offset > 0; offset = offset >> 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_count[blockIdx.x] = sdata[0];
    }
    //    atomicAdd(&d_count[blockIdx.x], d_predicate[pos]);
}

__global__
void prefix_sum_kernel(const unsigned int* const d_in,
                       unsigned int* const d_sum,
                       const size_t num) {
  // Exclusive prefix sum by Hillis and Steele algorithm
  // GPU Gems chapter 39
  extern __shared__ unsigned int s_buf[];  // double buffer
  unsigned int* bufs[2] = {&s_buf[0], &s_buf[num]};
  int toggle = 1;
  unsigned int* inbuf = bufs[0];
  unsigned int* outbuf = bufs[0];
  const int tid = threadIdx.x;

  inbuf[tid] = (tid > 0) ? d_in[tid - 1] : 0;
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

__global__
void prefix_sum_block_kernel(const uint8_t* const d_in,
                             unsigned int* const d_sum,
                             const size_t elemNum) {
    // calc prefix sum within a blcok
    // Exclusive prefix sum by Hillis and Steele algorithm
    // GPU Gems chapter 39
    const int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos >= elemNum) return;

    extern __shared__ unsigned int s_buf[];  // double buffer
    const size_t num = blockIdx.x == gridDim.x ? elemNum - blockIdx.x * blockDim.x : blockDim.x;
    unsigned int* bufs[2] = {&s_buf[0], &s_buf[num]};
    int toggle = 1;
    unsigned int* inbuf = bufs[0];
    unsigned int* outbuf = bufs[0];
    const int tid = threadIdx.x;
    
    inbuf[tid] = tid > 0 ? d_in[pos - 1] : 0;
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
    
    d_sum[pos] = outbuf[tid];
}

__global__
void calc_dest_kernel(const uint8_t* d_predicate,
                      const uint32_t* const d_prefixSum,
                      const uint32_t* const d_count,
                      const uint32_t* const d_cumCount,
                      uint32_t* const d_dest,
                      const size_t numElems) {
    const int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos >= numElems) return;

    // num of digit == 0 in [0.. pos]
    const int num0 = d_cumCount[blockIdx.x] + d_prefixSum[pos];
    if (d_predicate[pos]) {  
        d_dest[pos] = num0; 
    } else {
        // num of digit == 0 in [0.. numElems]
        const int lastBlock = gridDim.x - 1;
        const uint32_t num0Total = d_cumCount[lastBlock] + d_count[lastBlock];
        d_dest[pos] = num0Total + (pos - num0);
    }
}

__global__
void move_kernel(const uint32_t* const d_inputVals, const uint32_t* const d_inputPos,
                 uint32_t* const d_outputVals, uint32_t* const d_outputPos,
                 const uint32_t* const d_dest,
                 const size_t numElems) {
    const int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos >= numElems) return;

    const uint32_t dest = d_dest[pos];
    d_outputVals[dest] = d_inputVals[pos];
    d_outputPos[dest] = d_inputPos[pos];
}

//#define DEBUGP(_X_, _Y_) debugp(#_X_, _X_, _Y_);
#define DEBUGP(_X_, _Y_)

template<typename T>
__global__
void print_kernel(const T* data) {
    const int tid = threadIdx.x;
    printf("%d, ", data[tid]);
}

template<typename T>
void debugp(const char* name, T* data, size_t num) {
    printf("%s : \n", name);
    print_kernel<<<1, num>>>(data);
    cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
    printf("\n\n");
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    const int numBits = sizeof(uint32_t) * 8;
    const int numThreads = 1024;
    //  const int numThreads = 4;
    const int numBlocks = (numElems + numThreads - 1) / numThreads;
    printf("numBlocks: %d, numThreads: %d\n", numBlocks, numThreads);

    uint32_t* d_count;     // [numBlocks] counter of digit 0 for each block
    uint32_t* d_cumCount;  // [numBlocks] counter of digit 0 cumulated from block 0
    uint8_t* d_predicate;  // [numElems] True if digit is 0. Used for prefix sum within a block
    uint32_t* d_prefixSum; // [numElems] exclusive prefix sum of predicate
    uint32_t* d_dest;      // [numElems] Destination position for the element
    checkCudaErrors(cudaMalloc(&d_count, sizeof(uint32_t) * numBlocks));
    checkCudaErrors(cudaMalloc(&d_cumCount, sizeof(uint32_t) * numBlocks));
    checkCudaErrors(cudaMalloc(&d_predicate, sizeof(uint8_t) * numElems));
    checkCudaErrors(cudaMalloc(&d_prefixSum, sizeof(uint32_t) * numElems));
    checkCudaErrors(cudaMalloc(&d_dest, sizeof(uint32_t) * numElems));

    // Ping-pong buffer
    uint32_t* bufVal[2] = {d_inputVals, d_outputVals};
    uint32_t* bufPos[2] = {d_inputPos, d_outputPos};
    int toggle = 1;
    uint32_t* d_inVal;
    uint32_t* d_inPos;
    uint32_t* d_outVal;
    uint32_t* d_outPos;
    // Radix Sort
    for (uint32_t bit = 0; bit < numBits; ++bit) {
        // Swap in and out
        toggle = 1 - toggle;
        d_inVal = bufVal[toggle];
        d_inPos = bufPos[toggle];
        d_outVal = bufVal[1 - toggle];
        d_outPos = bufPos[1 - toggle];
        DEBUGP(d_inVal, numElems);

        // Generate output of predicate. Set True(1) in case digit is 0
        checkCudaErrors(cudaMemset(d_predicate, 0, sizeof(uint8_t) * numElems));
        predicate_kernel<<<numBlocks, numThreads>>>(d_inVal, d_predicate, bit, numElems);
        cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
        DEBUGP(d_predicate, numElems);

        // Count 0 for each block
        checkCudaErrors(cudaMemset(d_count, 0, sizeof(uint32_t) * numBlocks));
        size_t shmSize = sizeof(uint32_t) * numThreads;
        count_kernel<<<numBlocks, numThreads, shmSize>>>(d_predicate, d_count, numElems);
        cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
        DEBUGP(d_count, numBlocks);

        // Calc cumulative count of counter over all blocks
        checkCudaErrors(cudaMemset(d_cumCount, 0, sizeof(uint32_t) * numBlocks));
        shmSize = sizeof(uint32_t) * numBlocks * 2;  // Double Buffer
        prefix_sum_kernel<<<1, numBlocks, shmSize>>>(d_count, d_cumCount, numBlocks);
        cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
        DEBUGP(d_cumCount, numBlocks);

        // Calc prefix_sum within a block
        // which corresponds to the starting offeset for 0 within a block
        checkCudaErrors(cudaMemset(d_prefixSum, 0, sizeof(uint32_t) * numElems));
        shmSize = sizeof(uint32_t) * numThreads * 2;  // Double Buffer
        prefix_sum_block_kernel<<<numBlocks, numThreads, shmSize>>>(d_predicate, d_prefixSum, numElems);
        cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
        DEBUGP(d_prefixSum, numElems);

        // Calc destination of each element over all blocks
        checkCudaErrors(cudaMemset(d_dest, 0, sizeof(uint32_t) * numElems));
        calc_dest_kernel<<<numBlocks, numThreads>>>(d_predicate, d_prefixSum, d_count, d_cumCount,
                                                    d_dest, numElems);
        cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
        DEBUGP(d_dest, numElems);

        // Move the elements based on the destination
        move_kernel<<<numBlocks, numThreads>>>(d_inVal, d_inPos,
                                               d_outVal, d_outPos,
                                               d_dest, numElems);
        cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
        DEBUGP(d_outVal, numElems);
    }
    checkCudaErrors(cudaMemcpy(d_outputVals, d_outVal,
                               sizeof(uint32_t) * numElems, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_outputPos, d_outPos,
                               sizeof(uint32_t) * numElems, cudaMemcpyDeviceToDevice));
    DEBUGP(d_outputVals, numElems);

    checkCudaErrors(cudaFree(d_count));
    checkCudaErrors(cudaFree(d_cumCount));
    checkCudaErrors(cudaFree(d_predicate));
    checkCudaErrors(cudaFree(d_prefixSum));
    checkCudaErrors(cudaFree(d_dest));
}
