//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>
typedef unsigned char uchar;

enum {
    R = 0,
    G,
    B,
    numChannels,
};

const int NUM_ITERATION = 800;

__global__
void preproc_src_kernel(const uchar4* const d_srcRGB,  // IN
                        uchar* const d_srcMask,  // OUT
                        uchar* const d_srcR, uchar* const d_srcG, uchar* const d_srcB,  // OUT
                        const size_t numPixels) {
    const int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos >= numPixels) return;

    uchar4 src = d_srcRGB[pos];
    d_srcR[pos] = src.x;
    d_srcG[pos] = src.y;
    d_srcB[pos] = src.z;
    const int sum = src.x + src.y + src.z;
    d_srcMask[pos] = (sum < 255 * 3) ? 1 : 0;
}

void preprocSrc(const uchar4* const d_srcRGB,
                uchar* const d_srcMask,
                uchar* const d_src[],
                const size_t numPixels) {
    const int numThreads = 1024;
    const int numBlocks = (numPixels + numThreads - 1) / numThreads;
    preproc_src_kernel<<<numBlocks, numThreads>>>
        (d_srcRGB, d_srcMask, d_src[R], d_src[G], d_src[B], numPixels);
    cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
}

__global__
void make_mask_kernel(const uchar* const d_srcMask,  // IN
                      uchar* const d_borderMask,  // OUT
                      uchar* const d_interiorMask,  // OUT
                      const size_t numCols,
                      const size_t numRows) {
    const int2 pos2d = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
                                 blockDim.y * blockIdx.y + threadIdx.y);
    const int pos = numCols * pos2d.y + pos2d.x;
    if (pos2d.x >= numCols || pos2d.y >= numRows) return;
    // d_border/interiorMask should be initialized outside of this kernel
    if (d_srcMask[pos] == 0) return;

    // Casing the image edge
    const int left   = (pos2d.x == 0) ?           pos : pos - 1;
    const int right  = (pos2d.x == numCols - 1) ? pos : pos + 1;
    const int top    = (pos2d.y == 0) ?           pos : pos - numCols;
    const int bottom = (pos2d.y == numRows - 1) ? pos : pos + numCols;
    if (d_srcMask[left] && d_srcMask[right] && d_srcMask[top] && d_srcMask[bottom]) {
        d_interiorMask[pos] = 1;
    } else {
        d_borderMask[pos] = 1;
    }
}

void makeMask(const uchar* const d_srcMask,
              uchar* const d_borderMask,
              uchar* const d_interiorMask,
              const size_t numCols,
              const size_t numRows) {
    // Memset 0 because the kernel updates only masked region
    const size_t numPixels = numCols * numRows;
    checkCudaErrors(cudaMemset(d_borderMask, 0, sizeof(uchar) * numPixels));
    checkCudaErrors(cudaMemset(d_interiorMask, 0, sizeof(uchar) * numPixels));
    const dim3 numThreads(32, 32);
    const dim3 numBlocks((numCols + numThreads.x - 1) / numThreads.x,
                         (numRows + numThreads.y - 1) / numThreads.y);
    make_mask_kernel<<<numBlocks, numThreads>>>
        (d_srcMask, d_borderMask, d_interiorMask, numCols, numRows);
    cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
}

__global__
void preproc_dst_kernel(const uchar4* const d_dstRGB,  // IN
                        uchar* const d_dstR, uchar* const d_dstG, uchar* const d_dstB,  // OUT
                        const uchar* const d_borderMask,  // IN
                        const size_t numPixels) {
    const int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos >= numPixels) return;

    // d_dstR/G/B should be initialized outside of this kernel
    if (d_borderMask[pos] == 0) return;
    uchar4 dst = d_dstRGB[pos];
    d_dstR[pos] = dst.x;
    d_dstG[pos] = dst.y;
    d_dstB[pos] = dst.z;
}

void preprocDst(const uchar4* const d_dstRGB,  // IN
                uchar* const d_dst[],
                const uchar* const d_borderMask,  // IN
                const size_t numPixels) {
    // Memset 0 because the kernel updates only masked region
    for (int ch = 0; ch < numChannels; ++ch) {
        checkCudaErrors(cudaMemset(d_dst[ch], 0, sizeof(uchar) * numPixels));
    }

    const int numThreads = 1024;
    const int numBlocks = (numPixels + numThreads - 1) / numThreads;
    preproc_dst_kernel<<<numBlocks, numThreads>>>
        (d_dstRGB, d_dst[R], d_dst[G], d_dst[B], d_borderMask, numPixels);
    cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
}

__global__
void compute_termg_kernel(const uchar* const d_src,  // IN
                          float* const d_termG,  // OUT
                          const uchar* const d_interiorMask,  // IN
                          const size_t numCols,
                          const size_t numRows) {
    const int2 pos2d = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
                                 blockDim.y * blockIdx.y + threadIdx.y);
    const int pos = numCols * pos2d.y + pos2d.x;
    if (pos2d.x >= numCols || pos2d.y >= numRows) return;
    // d_termG should be initialized outside of this kernel
    if (d_interiorMask[pos] == 0) return;
    // Casing the image edge
    const int left   = (pos2d.x == 0) ?           pos : pos - 1;
    const int right  = (pos2d.x == numCols - 1) ? pos : pos + 1;
    const int top    = (pos2d.y == 0) ?           pos : pos - numCols;
    const int bottom = (pos2d.y == numRows - 1) ? pos : pos + numCols;

    float sum = 4.f * d_src[pos];
    sum -= (float)d_src[left] + (float)d_src[right] + (float)d_src[top] + (float)d_src[bottom];
    d_termG[pos] = sum;
}

void computeTermG(const uchar* const d_src,
                  float* const d_termG,
                  const uchar* const d_interiorMask,
                  const size_t numCols,
                  const size_t numRows) {
    // Memset 0 because the kernel updates only masked region
    checkCudaErrors(cudaMemset(d_termG, 0, sizeof(float) * numCols * numRows));

    const dim3 numThreads(32, 32);
    const dim3 numBlocks((numCols + numThreads.x - 1) / numThreads.x,
                         (numRows + numThreads.y - 1) / numThreads.y);
    compute_termg_kernel<<<numBlocks, numThreads>>>
        (d_src, d_termG, d_interiorMask, numCols, numRows);
    cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
}

__global__
void init_blended_kernel(const uchar* const d_src,
                         float* const d_blended1,
                         const size_t numPixels) {
    const int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos >= numPixels) return;
    d_blended1[pos] = (float)d_src[pos];
}

void initBlended(const uchar* const d_src,
                 float* const d_blended1,
                 const size_t numPixels) {
    const int numThreads = 1024;
    const int numBlocks = (numPixels + numThreads - 1) / numThreads;
    init_blended_kernel<<<numBlocks, numThreads>>>
        (d_src, d_blended1, numPixels);
    cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
}

__global__
void compute_blended_kernel(const uchar* const d_dst,  // IN
                            const float* const d_blended1,  // IN
                            float* const d_blended2,  // OUT
                            const float* const d_termG,  // IN
                            const uchar* const d_interiorMask,  // IN
                            const size_t numCols,
                            const size_t numRows) {
    const int2 pos2d = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
                               blockDim.y * blockIdx.y + threadIdx.y);
    const int pos = numCols * pos2d.y + pos2d.x;
    if (pos2d.x >= numCols || pos2d.y >= numRows) return;
    if (d_interiorMask[pos] == 0) return;

    // Casing the image edge
    const int left   = (pos2d.x == 0) ?           pos2d.x : pos - 1;
    const int right  = (pos2d.x == numCols - 1) ? pos2d.x : pos + 1;
    const int top    = (pos2d.y == 0) ?           pos2d.y : pos - numCols;
    const int bottom = (pos2d.y == numRows - 1) ? pos2d.y : pos + numCols;

    float sum = 0.f;
    sum += d_interiorMask[left]   ? d_blended1[left]   : d_dst[left];
    sum += d_interiorMask[right]  ? d_blended1[right]  : d_dst[right];
    sum += d_interiorMask[top]    ? d_blended1[top]    : d_dst[top];
    sum += d_interiorMask[bottom] ? d_blended1[bottom] : d_dst[bottom];

    float next = (sum + d_termG[pos]) / 4.f;
    d_blended2[pos] = min(max(next, 0.f), 255.f);  // clamp to 8bit
}

void computeBlended(const uchar* const d_dst,
                    const float* const d_blended1,
                    float* const d_blended2,
                    const float* const d_termG,
                    const uchar* const d_interiorMask,
                    const size_t numCols,
                    const size_t numRows) {
    const dim3 numThreads(32, 8);
    const dim3 numBlocks((numCols + numThreads.x - 1) / numThreads.x,
                         (numRows + numThreads.y - 1) / numThreads.y);
    compute_blended_kernel<<<numBlocks, numThreads>>>
        (d_dst, d_blended1, d_blended2, d_termG, d_interiorMask, numCols, numRows);
    cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
}

__global__
void compose_blended_kernel(const float* const d_blendedR,  // IN
                            const float* const d_blendedG,  // IN
                            const float* const d_blendedB,  // IN
                            const uchar4* const d_dstRGB,  // IN
                            uchar4* const d_blendedRGB,  // OUT
                            const uchar* const d_interiorMask,  // IN
                            const size_t numPixels) {
    const int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos >= numPixels) return;

    if (d_interiorMask[pos]) {
        d_blendedRGB[pos] = make_uchar4((uchar)d_blendedR[pos],
                                        (uchar)d_blendedG[pos],
                                        (uchar)d_blendedB[pos],
                                        0);
    } else {
        d_blendedRGB[pos] = d_dstRGB[pos];
    }
}

void composeBlended(const float* const d_blended[],
                    const uchar4* const d_dstRGB,
                    uchar4* const d_blendedRGB,
                    const uchar* const d_interiorMask,
                    const size_t numPixels) {
    const int numThreads = 1024;
    const int numBlocks = (numPixels + numThreads - 1) / numThreads;
    compose_blended_kernel<<<numBlocks, numThreads>>>
        (d_blended[R], d_blended[G], d_blended[B], d_dstRGB, d_blendedRGB, d_interiorMask, numPixels);
    cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
}


void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  /* To Recap here are the steps you need to implement

     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */
    uchar4* d_srcRGB;
    uchar4* d_dstRGB;
    uchar4* d_blendedRGB;
    uchar* d_srcMask;
    uchar* d_borderMask;
    uchar* d_interiorMask;

    // R,G,B separated
    uchar* d_src[numChannels];
    uchar* d_dst[numChannels];
    float* d_termG[numChannels];
    float* d_blended1[numChannels];
    float* d_blended2[numChannels];

    // Allocate device memory
    const size_t numPixels = numRowsSource * numColsSource;
    checkCudaErrors(cudaMalloc(&d_srcRGB,      sizeof(uchar4) * numPixels));
    checkCudaErrors(cudaMalloc(&d_dstRGB,      sizeof(uchar4) * numPixels));
    checkCudaErrors(cudaMalloc(&d_blendedRGB,  sizeof(uchar4) * numPixels));
    checkCudaErrors(cudaMalloc(&d_srcMask,      sizeof(uchar) * numPixels));
    checkCudaErrors(cudaMalloc(&d_borderMask,   sizeof(uchar) * numPixels));
    checkCudaErrors(cudaMalloc(&d_interiorMask, sizeof(uchar) * numPixels));
    for (int ch = 0; ch < numChannels; ++ch) {
        checkCudaErrors(cudaMalloc(&d_src[ch],      sizeof(uchar) * numPixels));
        checkCudaErrors(cudaMalloc(&d_dst[ch],      sizeof(uchar) * numPixels));
        checkCudaErrors(cudaMalloc(&d_termG[ch],    sizeof(float) * numPixels));
        checkCudaErrors(cudaMalloc(&d_blended1[ch], sizeof(float) * numPixels));
        checkCudaErrors(cudaMalloc(&d_blended2[ch], sizeof(float) * numPixels));
    }

    // Copy Host to Device
    checkCudaErrors(cudaMemcpy(d_srcRGB, h_sourceImg,
                               sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_dstRGB, h_destImg,
                               sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

    // Preprocessing
    preprocSrc(d_srcRGB, d_srcMask, d_src, numPixels);
    makeMask(d_srcMask, d_borderMask, d_interiorMask, numColsSource, numRowsSource);
    preprocDst(d_dstRGB, d_dst, d_borderMask, numPixels);

    for (int ch = 0; ch < numChannels; ++ch) {
        // Compute iteration independent Term G
        computeTermG(d_src[ch], d_termG[ch], d_interiorMask, numColsSource, numRowsSource);
        // As initialization, copy src to d_blended1
        initBlended(d_src[ch], d_blended1[ch], numPixels);

        for (int i = 0; i < NUM_ITERATION; ++i) {
            computeBlended(d_dst[ch], d_blended1[ch], d_blended2[ch], d_termG[ch],
                           d_interiorMask, numColsSource, numRowsSource);
            std::swap(d_blended1[ch], d_blended2[ch]);
        }
    }

    // Compose blended image
    composeBlended(d_blended1, d_dstRGB, d_blendedRGB, d_interiorMask, numPixels);

    // Copy Device to Host
    checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedRGB,
                               sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));

    // Free device memory
    checkCudaErrors(cudaFree(d_srcRGB));
    checkCudaErrors(cudaFree(d_dstRGB));
    checkCudaErrors(cudaFree(d_blendedRGB));
    checkCudaErrors(cudaFree(d_srcMask));
    checkCudaErrors(cudaFree(d_borderMask));
    checkCudaErrors(cudaFree(d_interiorMask));
    for (int ch = 0; ch < numChannels; ++ch) {
        checkCudaErrors(cudaFree(d_src[ch]));
        checkCudaErrors(cudaFree(d_dst[ch]));
        checkCudaErrors(cudaFree(d_termG[ch]));
        checkCudaErrors(cudaFree(d_blended1[ch]));
        checkCudaErrors(cudaFree(d_blended2[ch]));
    }
}
