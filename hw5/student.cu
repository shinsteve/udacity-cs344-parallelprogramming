/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include <stdio.h>

__global__
void histo_kernel_32bin(const unsigned int *const d_vals,
                        unsigned int* const d_histo,
                        const unsigned int factor,
                        const unsigned int numElemPerThread,
                        const unsigned int numElems) {
    extern __shared__ uint32_t s_histo[];
    const int numBins = 32;
    uint8_t l_histo[numBins];
    const int tid = threadIdx.x;

    if (tid < numBins) s_histo[tid] = 0;
    memset(l_histo, 0, sizeof(uint8_t) * numBins);
    __syncthreads();

    int pos = blockDim.x * numElemPerThread * blockIdx.x + threadIdx.x;
    for (int i = 0; i < numElemPerThread; ++i) {
        if (pos >= numElems) break;
        int bin = d_vals[pos] >> factor;  // Optimization based on assumption
        l_histo[bin]++;
        pos += blockDim.x;
    }
    // Local to Shared
    for (int i = 0; i < numBins; ++i) {
        atomicAdd(&s_histo[i], l_histo[i]);
    }
    __syncthreads();

    // Shared to Global
    if (tid < numBins) {
        atomicAdd(&d_histo[tid], s_histo[tid]);
    }
}

void makeHistoSmallBin(const unsigned int* const d_vals,
                       unsigned int* const d_histo,
                       const size_t numBins,
                       const int factor,
                       const size_t numElems) {
    checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));
    const int numThreads = 1024;
    const int numElemPerThread = 255;  // To make size of 1 bin to be 1 byte
    const int numElemPerBlock = numElemPerThread * numThreads;
    const int numBlocks = (numElems + numElemPerBlock - 1) / numElemPerBlock;
    printf("makeHistoSmallBin(): numBlocks: %d, numThreads: %d\n", numBlocks, numThreads);
    size_t shmSize = sizeof(uint32_t) * numThreads;
    histo_kernel_32bin<<<numBlocks, numThreads, shmSize>>>
        (d_vals, d_histo, factor, numElemPerThread, numElems);
    cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
}

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
                       const unsigned int start,
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

  d_sum[tid] = start + outbuf[tid];
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
    const size_t num = (blockIdx.x == gridDim.x - 1) ? \
          elemNum - blockIdx.x * blockDim.x : blockDim.x;
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
void move_kernel(const uint32_t* const d_inputVals,
                 uint32_t* const d_outputVals,
                 const uint32_t* const d_dest,
                 const size_t numElems) {
    const int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos >= numElems) return;

    const uint32_t dest = d_dest[pos];
    d_outputVals[dest] = d_inputVals[pos];
}

__global__
void shift_kernel(const uint32_t* const d_in,
                  uint32_t* const d_out,
                  const int factor, const size_t numElems) {
    const int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos >= numElems) return;
    uint32_t val = d_in[pos];
    d_out[pos] = val & ((1 << factor) - 1);
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

void coarseSortAndShift(const unsigned int *const d_inputVals,
                        unsigned int* const d_outputVals,
                        const int factor,
                        const unsigned int numElems) {
    // Sort by key : val / 32   <=>   val >> factor
    // Shift val   : val % 32   <=>   val & ((1 << factor) - 1)

    const int numThreads = 1024;
    const int numBlocks = (numElems + numThreads - 1) / numThreads;
    printf("coarseSortAndShift(): numBlocks: %d, numThreads: %d\n", numBlocks, numThreads);

    uint32_t* d_in;        // [numElems] internal buffer for input
    uint32_t* d_out;       // [numElems] internal buffer for output
    uint32_t* d_count;     // [numBlocks] counter of digit 0 for each block
    uint32_t* d_cumCount;  // [numBlocks] counter of digit 0 cumulated from block 0
    uint8_t* d_predicate;  // [numElems] True if digit is 0. Used for prefix sum within a block
    uint32_t* d_prefixSum; // [numElems] exclusive prefix sum of predicate
    uint32_t* d_dest;      // [numElems] Destination position for the element
    checkCudaErrors(cudaMalloc(&d_in,        sizeof(uint32_t) * numElems));
    checkCudaErrors(cudaMalloc(&d_out,       sizeof(uint32_t) * numElems));
    checkCudaErrors(cudaMalloc(&d_count,     sizeof(uint32_t) * numBlocks));
    checkCudaErrors(cudaMalloc(&d_cumCount,  sizeof(uint32_t) * numBlocks));
    checkCudaErrors(cudaMalloc(&d_predicate, sizeof(uint8_t) * numElems));
    checkCudaErrors(cudaMalloc(&d_prefixSum, sizeof(uint32_t) * numElems));
    checkCudaErrors(cudaMalloc(&d_dest,      sizeof(uint32_t) * numElems));

    checkCudaErrors(cudaMemcpy(d_in, d_inputVals,
                               sizeof(uint32_t) * numElems, cudaMemcpyDeviceToDevice));
    // Ping-pong buffer
    uint32_t* bufVal[2] = {d_in, d_out};
    int toggle = 1;
    uint32_t* d_inVal;
    uint32_t* d_outVal;
    // Radix Sort of coarse version
    // NOTE: compare bit >= factor
    for (uint32_t bit = factor; bit < 10; ++bit) {
        // Swap in and out
        toggle = 1 - toggle;
        d_inVal = bufVal[toggle];
        d_outVal = bufVal[1 - toggle];
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
        shmSize = sizeof(uint32_t) * numThreads * 2;  // Double Buffer
        const int numLoop = (numBlocks + numThreads - 1) / numThreads;
        uint32_t startVal = 0;
        for (int i = 0; i < numLoop; ++i) {
            if (i > 0) {
                // Read the last val of cumCount for the next startVal.
                checkCudaErrors(cudaMemcpy(&startVal, d_cumCount + i * numThreads - 1,
                                           sizeof(uint32_t), cudaMemcpyDeviceToHost));
            }
            printf("%d\n", startVal);
            prefix_sum_kernel<<<1, numThreads, shmSize>>>(d_count + i * numThreads,
                                                          d_cumCount + i * numThreads,
                                                          startVal, numThreads);
            cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
        }
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
        move_kernel<<<numBlocks, numThreads>>>(d_inVal, d_outVal, d_dest, numElems);
        cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
        DEBUGP(d_outVal, numElems);
    }

    shift_kernel<<<numBlocks, numThreads>>>(d_outVal, d_outputVals, factor, numElems);
    cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
    DEBUGP(d_outputVals, numElems);

    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaFree(d_count));
    checkCudaErrors(cudaFree(d_cumCount));
    checkCudaErrors(cudaFree(d_predicate));
    checkCudaErrors(cudaFree(d_prefixSum));
    checkCudaErrors(cudaFree(d_dest));
}

void makeHisto2Pass(const unsigned int *const d_vals,
                    unsigned int* const d_histo,
                    const unsigned int numBins,
                    const unsigned int numElems) {
    // Calc histo of large bin by using coarse small bin as first pass
    //
    // NOTE: This implementation is highly optimized by assuming:
    //       numBins == 1024 == 2^10
    //       numBinDivision == 32
    //       numSubBin == 32 == 2^5
    const int numRangeBit = 10;
    const int numDivShiftBit = 5;  // Don't change as the rest of funcs are optimized
    const int numBinDivision = 1 << numDivShiftBit;
    if (numBins != 1024) {
        std::cerr << "Error: numBins is assumed to be 1024, but got: "
                  << numBins << std::endl;
        exit(1);
    }
    if (numBins % numBinDivision != 0) {
        std::cerr << "Error: numBins is not multiple of numBinDivision"
                  << numBins << " % " << numBinDivision << std::endl;
        exit(1);
    }
    const int numSubBin = numBins / numBinDivision;

    // make coarse histogram
    uint32_t h_coarseHisto[numBinDivision];
    uint32_t* d_coarseHisto;     // [numBinDivision]
    checkCudaErrors(cudaMalloc(&d_coarseHisto, sizeof(uint32_t) * numBinDivision));
    makeHistoSmallBin(d_vals, d_coarseHisto, numBinDivision, numRangeBit - numDivShiftBit, numElems);
    checkCudaErrors(cudaMemcpy(h_coarseHisto, d_coarseHisto,
                               sizeof(uint32_t) * numBinDivision, cudaMemcpyDeviceToHost));

    // sort by key of coarse bin location
    // and shit based on bin location
    // After this, element is in range [0..31]
    uint32_t* d_sorted;     // [numElems]
    checkCudaErrors(cudaMalloc(&d_sorted, sizeof(uint32_t) * numElems));
    coarseSortAndShift(d_vals, d_sorted, numDivShiftBit, numElems);

    size_t offset = 0;
    for (int i = 0; i < numBinDivision; ++i) {
        makeHistoSmallBin(d_sorted + offset,
                          d_histo + numSubBin * i,
                          numSubBin,
                          numDivShiftBit,
                          h_coarseHisto[i]);
        offset += h_coarseHisto[i];
    }

    checkCudaErrors(cudaFree(d_coarseHisto));
    checkCudaErrors(cudaFree(d_sorted));
}

__global__
void histo_kernel_shared(const unsigned int *const d_vals,
                        unsigned int* const d_histo,
                        const unsigned int numBins,
                        const unsigned int numElems) {
    extern __shared__ unsigned int s_histo[];
    const int pos = blockDim.x * blockIdx.x + threadIdx.x;
    const int tid = threadIdx.x;

    if (tid < numBins) s_histo[tid] = 0;
    __syncthreads();

    if (pos < numElems) {
        atomicAdd(&s_histo[d_vals[pos]], 1);
    }
    __syncthreads();

    if (tid < numBins) atomicAdd(&d_histo[tid], s_histo[tid]);
}

void makeHistoShm(const unsigned int *const d_vals,
                  unsigned int* const d_histo,
                  const unsigned int numBins,
                  const unsigned int numElems) {
    const int numThreads = numBins;
    const int numBlocks = (numElems + numThreads - 1) / numThreads;
    const size_t shmSize = sizeof(unsigned int) * numBins;
    histo_kernel_shared<<<numBlocks, numThreads, shmSize>>>
        (d_vals, d_histo, numBins, numElems);
    cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
}

__global__
void histo_naive_kernel(const unsigned int *const d_vals,
                        unsigned int* const d_histo,
                        const unsigned int numBins,
                        const unsigned int numElems) {
    const int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos >= numElems) return;

    atomicAdd(&d_histo[d_vals[pos]], 1);
}

void makeHistoNaive(const unsigned int *const d_vals,
                   unsigned int* const d_histo,
                   const unsigned int numBins,
                   const unsigned int numElems) {
    const int numThreads = 1024;
    const int numBlocks = (numElems + numThreads - 1) / numThreads;
    histo_naive_kernel<<<numBlocks, numThreads>>>(d_vals, d_histo, numBins, numElems);
    cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
    // d_histo is initialized by caller side
    // checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));

    makeHistoNaive(d_vals, d_histo, numBins, numElems);
    // On the recent GPU, naive implementation that uses global atomic was the fastest...
    // makeHistoShm(d_vals, d_histo, numBins, numElems);
    // makeHisto2Pass(d_vals, d_histo, numBins, numElems);
}
