#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void depth_conversion_kernel_launcher(const float* PointDepth, const float* DistanceFromCenter, float* PlaneDepth, float f, int height, int width);
void create_point_cloud_kernel_launcher(const float* pixel_depth, float* x, float* y, float* z, float FX_DEPTH, float FY_DEPTH, float CX_DEPTH, float CY_DEPTH, int height, int width);

#ifdef __cplusplus
}
#endif

#endif // KERNELS_H
