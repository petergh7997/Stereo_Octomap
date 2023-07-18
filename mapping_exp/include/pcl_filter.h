// 输入参数：cloud - 输入的点云数据
        //           cloud_filtered - 输出的滤波后的点云数据
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
// StatisticalOutlierRemoval
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>//半径滤波器
#include <pcl/common/time.h>
#include <pcl/filters/median_filter.h> // 中值滤波
#include <pcl/filters/convolution_3d.h>  // 高斯滤波
#include <pcl/search/kdtree.h>
#include <boost/thread/thread.hpp>


int StatisticalFilter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_filtered) {
        pcl::StopWatch time;

    // 创建一个统计滤波器对象
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    // 设置滤波器的输入点云
    sor.setInputCloud(cloud);
    // 设置滤波器的参数
    // sor.setMeanK(50); // 设置在进行统计分析时考虑的临近点的数量
    sor.setMeanK(1000); // 设置在进行统计分析时考虑的临近点的数量
    // sor.setStddevMulThresh(1.0); // 设置判断是否为离群点的阈值
    sor.setStddevMulThresh(5.0); // 设置判断是否为离群点的阈值
    
    // 进行滤波处理，结果保存在cloud_filtered中
    sor.filter(*cloud_filtered);
    std::cout << "统计滤波前有: " << cloud->size() << " 个点 " << std::endl;
    std::cout << "统计滤波后有: " << cloud_filtered->size() << " 个点 " << std::endl;
    std::cout << "运行时间:" << time.getTime() << "毫秒" << std::endl;

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
int VoxelGridFilter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_filtered){
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    // sor.setLeafSize(0.003f, 0.003f, 0.003f); //设置栅格体素的大小
    // sor.setLeafSize(0.01f, 0.01f, 0.01f); //设置栅格体素的大小
    sor.setLeafSize(0.08f, 0.08f, 0.08f); //设置栅格体素的大小
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxel_filtered(new pcl::PointCloud<pcl::PointXYZRGB>); //采样后根
    sor.filter(*cloud_filtered);
}

//半径滤波器
int RadiusFilter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_radius){
    pcl::StopWatch time;
    pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> ror;
    ror.setInputCloud(cloud_in);     // 输入点云
    // ror.setRadiusSearch(0.1);        // 设置半径为0.1m范围内找临近点
    ror.setRadiusSearch(0.1);        // 设置半径为0.1m范围内找临近点
    ror.setMinNeighborsInRadius(15); // 设置查询点的邻域点集数小于10删除
    // ror.setMinNeighborsInRadius(10); // 设置查询点的邻域点集数小于10删除
    ror.filter(*cloud_radius);       // 执行滤波
    //pcl::io::savePCDFileASCII("cloud_radius.pcd", *cloud_radius);
    std::cout << "滤波前有: " << cloud_in->size() << " 个点 " << std::endl;
    std::cout << "滤波后有: " << cloud_radius->size() << " 个点 " << std::endl;
    std::cout << "运行时间:" << time.getTime() << "毫秒" << std::endl;
    // cout << "运行时间:" << time.getTime << "毫秒" << endl;

}

// //中值滤波
// int MedianFilter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_filtered){

//     pcl::MedianFilter <pcl::PointXYZRGB> median;
//     median.setInputCloud(cloud);
//     median.setWindowSize(10);          // 设置过滤器的窗口大小
//     median.setMaxAllowedMovement(0.1f);// 一个点允许沿z轴移动的最大距离 
// 	// pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
//     median.filter(*cloud_filtered);
// }

//高斯滤波
int GaussFilter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_filtered){
    pcl::StopWatch time;
    // -----------------------------基于高斯核函数的卷积滤波实现---------------------------
	// pcl::filters::GaussianKernel<pcl::PointXYZ, pcl::PointXYZ> kernel;
	pcl::filters::GaussianKernel<pcl::PointXYZRGB, pcl::PointXYZRGB> kernel;

	kernel.setSigma(4);//高斯函数的标准方差，决定函数的宽度
	kernel.setThresholdRelativeToSigma(4);//设置相对Sigma参数的距离阈值
	kernel.setThreshold(0.05);//设置距离阈值，若点间距离大于阈值则不予考虑
	// cout << "Kernel made" << endl;
	std::cout << "Kernel made" << std::endl;

	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
	tree->setInputCloud(cloud);
	std::cout << "KdTree made" << std::endl;

    // -------------------------------设置Convolution 相关参数-----------------------------
	pcl::filters::Convolution3D<pcl::PointXYZRGB, pcl::PointXYZRGB, pcl::filters::GaussianKernel<pcl::PointXYZRGB, pcl::PointXYZRGB>> convolution;
	convolution.setKernel(kernel);//设置卷积核
	convolution.setInputCloud(cloud);
	convolution.setNumberOfThreads(8);
	convolution.setSearchMethod(tree);
	convolution.setRadiusSearch(0.01);
	std::cout << "Convolution Start" << std::endl;
	convolution.convolve(*cloud_filtered);
    std::cout << "高斯滤波前有: " << cloud->size() << " 个点 " << std::endl;
    std::cout << "高斯滤波后有: " << cloud_filtered->size() << " 个点 " << std::endl;
    std::cout << "高斯运行时间:" << time.getTime() << "毫秒" << std::endl;
}