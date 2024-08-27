#include "kernels.h"
#include <cmath>

__global__ void depth_conversion_kernel_impl(const float* PointDepth, const float* DistanceFromCenter, float* PlaneDepth, float f, int height, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < height && j < width) {
        PlaneDepth[i * width + j] = PointDepth[i * width + j] / sqrtf(1.0f + powf(DistanceFromCenter[i * width + j] / f, 2));
    }
}

__global__ void create_point_cloud_kernel_impl(const float* pixel_depth, float* x, float* y, float* z, float FX_DEPTH, float FY_DEPTH, float CX_DEPTH, float CY_DEPTH, int height, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < height && j < width) {
        float depth = pixel_depth[i * width + j] / 1000.0f;

        if (depth > 2.0f) {
            return;
        }

        x[i * width + j] = (j - CX_DEPTH) * depth / FX_DEPTH;
        y[i * width + j] = (i - CY_DEPTH) * depth / FY_DEPTH;
        z[i * width + j] = depth;
    }
}

extern "C" void depth_conversion_kernel_launcher(const float* PointDepth, const float* DistanceFromCenter, float* PlaneDepth, float f, int height, int width) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((height + threadsPerBlock.x - 1) / threadsPerBlock.x, (width + threadsPerBlock.y - 1) / threadsPerBlock.y);
    depth_conversion_kernel_impl<<<blocksPerGrid, threadsPerBlock>>>(PointDepth, DistanceFromCenter, PlaneDepth, f, height, width);
}

extern "C" void create_point_cloud_kernel_launcher(const float* pixel_depth, float* x, float* y, float* z, float FX_DEPTH, float FY_DEPTH, float CX_DEPTH, float CY_DEPTH, int height, int width) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((height + threadsPerBlock.x - 1) / threadsPerBlock.x, (width + threadsPerBlock.y - 1) / threadsPerBlock.y);
    create_point_cloud_kernel_impl<<<blocksPerGrid, threadsPerBlock>>>(pixel_depth, x, y, z, FX_DEPTH, FY_DEPTH, CX_DEPTH, CY_DEPTH, height, width);
}