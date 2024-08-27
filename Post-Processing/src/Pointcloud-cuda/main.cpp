#include <opencv2/opencv.hpp>
#include <open3d/Open3D.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "kernels.h"

void depth_conversion(const cv::Mat& PointDepth, const cv::Mat& DistanceFromCenter, float f, cv::Mat& PlaneDepth, float* PointDepth_device, float* DistanceFromCenter_device, float* PlaneDepth_device) {
    int height = PointDepth.rows;
    int width = PointDepth.cols;

    cudaMemcpy(PointDepth_device, PointDepth.ptr<float>(), height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(DistanceFromCenter_device, DistanceFromCenter.ptr<float>(), height * width * sizeof(float), cudaMemcpyHostToDevice);

    depth_conversion_kernel_launcher(PointDepth_device, DistanceFromCenter_device, PlaneDepth_device, f, height, width);

    cudaMemcpy(PlaneDepth.ptr<float>(), PlaneDepth_device, height * width * sizeof(float), cudaMemcpyDeviceToHost);
}

void create_point_cloud_from_depth(const cv::Mat& depth_image, const std::string& rgb_path, float FX_DEPTH, float FY_DEPTH, float CX_DEPTH, float CY_DEPTH, std::vector<Eigen::Vector3d>& points, std::vector<Eigen::Vector3d>& colors, float* PointDepth_device, float* DistanceFromCenter_device, float* PlaneDepth_device, float* pixel_depth_device, float* x_device, float* y_device, float* z_device) {
    int height = depth_image.rows;
    int width = depth_image.cols;

    cv::Mat pixel_depth_cm(height, width, CV_32F);

    if (depth_image.channels() == 3) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                pixel_depth_cm.at<float>(i, j) = depth_image.at<cv::Vec3b>(i, j)[0] +
                                                 depth_image.at<cv::Vec3b>(i, j)[1] * 256.0f +
                                                 depth_image.at<cv::Vec3b>(i, j)[2] * 256.0f * 256.0f;
            }
        }
    } else if (depth_image.channels() == 1) {
        depth_image.convertTo(pixel_depth_cm, CV_32F);
    } else {
        throw std::runtime_error("Unsupported number of channels in the depth image.");
    }

    pixel_depth_cm *= 10.0f;

    float i_c = static_cast<float>(height) / 2 - 1;
    float j_c = static_cast<float>(width) / 2 - 1;
    cv::Mat DistanceFromCenter(height, width, CV_32F);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            DistanceFromCenter.at<float>(i, j) = sqrtf(powf(i - i_c, 2) + powf(j - j_c, 2));
        }
    }

    cv::Mat PlaneDepth(height, width, CV_32F);
    depth_conversion(pixel_depth_cm, DistanceFromCenter, FX_DEPTH, PlaneDepth, PointDepth_device, DistanceFromCenter_device, PlaneDepth_device);

    cudaMemcpy(pixel_depth_device, PlaneDepth.ptr<float>(), height * width * sizeof(float), cudaMemcpyHostToDevice);

    create_point_cloud_kernel_launcher(pixel_depth_device, x_device, y_device, z_device, FX_DEPTH, FY_DEPTH, CX_DEPTH, CY_DEPTH, height, width);

    std::vector<float> x_host(height * width), y_host(height * width), z_host(height * width);
    cudaMemcpy(x_host.data(), x_device, height * width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_host.data(), y_device, height * width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(z_host.data(), z_device, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    cv::Mat rgb_image = cv::imread(rgb_path);
    if (rgb_image.empty()) {
        throw std::runtime_error("RGB image not found at path: " + rgb_path);
    }
    cv::cvtColor(rgb_image, rgb_image, cv::COLOR_BGR2RGB);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float z_value = z_host[i * width + j];
            if (z_value > 0) {
                Eigen::Vector3d point(x_host[i * width + j], y_host[i * width + j], z_host[i * width + j]);
                points.push_back(point);

                int u = std::min(std::max(static_cast<int>(FX_DEPTH * point[0] / z_value + CX_DEPTH), 0), rgb_image.cols - 1);
                int v = std::min(std::max(static_cast<int>(FY_DEPTH * point[1] / z_value + CY_DEPTH), 0), rgb_image.rows - 1);

                cv::Vec3b color = rgb_image.at<cv::Vec3b>(v, u);
                colors.push_back(Eigen::Vector3d(color[2] / 255.0, color[1] / 255.0, color[0] / 255.0));
            }
        }
    }
}

int main() {
    float FX_DEPTH = 320.0f;
    float FY_DEPTH = 320.0f;
    float CX_DEPTH = 320.0f;
    float CY_DEPTH = 240.0f;

    cv::Mat depth_image = cv::imread("/home/eflinspy/Downloads/MegaScansQuarry/DepthImages/scene_00_0001.png", cv::IMREAD_UNCHANGED);
    if (depth_image.empty()) {
        throw std::runtime_error("Depth image not found.");
    }

    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> colors;

    int height = depth_image.rows;
    int width = depth_image.cols;

    float *PointDepth_device, *DistanceFromCenter_device, *PlaneDepth_device;
    cudaMalloc(&PointDepth_device, height * width * sizeof(float));
    cudaMalloc(&DistanceFromCenter_device, height * width * sizeof(float));
    cudaMalloc(&PlaneDepth_device, height * width * sizeof(float));

    float *pixel_depth_device, *x_device, *y_device, *z_device;
    cudaMalloc(&pixel_depth_device, height * width * sizeof(float));
    cudaMalloc(&x_device, height * width * sizeof(float));
    cudaMalloc(&y_device, height * width * sizeof(float));
    cudaMalloc(&z_device, height * width * sizeof(float));

    create_point_cloud_from_depth(depth_image, "/home/eflinspy/Downloads/MegaScansQuarry/RGBImages/scene_00_0001.png", FX_DEPTH, FY_DEPTH, CX_DEPTH, CY_DEPTH, points, colors, PointDepth_device, DistanceFromCenter_device, PlaneDepth_device, pixel_depth_device, x_device, y_device, z_device);

    cudaFree(PointDepth_device);
    cudaFree(DistanceFromCenter_device);
    cudaFree(PlaneDepth_device);
    cudaFree(pixel_depth_device);
    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(z_device);

    auto point_cloud = std::make_shared<open3d::geometry::PointCloud>();
    point_cloud->points_ = points;
    point_cloud->colors_ = colors;

    open3d::visualization::DrawGeometries({point_cloud}, "Point Cloud");

    return 0;
}
