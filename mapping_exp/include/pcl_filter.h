// 输入参数：cloud - 输入的点云数据
        //           cloud_filtered - 输出的滤波后的点云数据
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
// StatisticalOutlierRemoval
#include <pcl/filters/statistical_outlier_removal.h>

int StatisticalFilter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_filtered) {
    // 创建一个统计滤波器对象
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    // 设置滤波器的输入点云
    sor.setInputCloud(cloud);
    // 设置滤波器的参数
    // sor.setMeanK(50); // 设置在进行统计分析时考虑的临近点的数量
    sor.setMeanK(300); // 设置在进行统计分析时考虑的临近点的数量
    // sor.setStddevMulThresh(1.0); // 设置判断是否为离群点的阈值
    sor.setStddevMulThresh(3.0); // 设置判断是否为离群点的阈值
    // 进行滤波处理，结果保存在cloud_filtered中
    sor.filter(*cloud_filtered);
}

int StatisticalFilter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) {
    // 创建一个统计滤波器对象
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    // 设置滤波器的输入点云
    sor.setInputCloud(cloud);
    // 设置滤波器的参数
    sor.setMeanK(50); // 设置在进行统计分析时考虑的临近点的数量
    sor.setStddevMulThresh(1.0); // 设置判断是否为离群点的阈值
    // 进行滤波处理，结果保存在cloud_filtered中
    sor.filter(*cloud);
}

// 可以多放几种滤波方式对照效果
