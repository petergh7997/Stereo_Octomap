#include <iostream>
#include <fstream>
#include <stdio.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <string>
#include <queue>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h> 
#include <tf/tf.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d/calib3d_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
// for sgm
#include "include/SemiGlobalMatching.h"

using namespace std;
using namespace std::chrono;
using namespace cv;

inline void show_stereo(cv::Mat imgLeft, cv::Mat imgRight, string win){
    if(imgLeft.channels() < 3 || imgRight.channels() < 3){
        cvtColor(imgLeft, imgLeft, CV_GRAY2BGR);
        cvtColor(imgRight, imgRight, CV_GRAY2BGR);
    }
    CV_Assert(imgLeft.type() == imgRight.type());
    cv::Mat mergeImg;
    int merge_rows = imgLeft.rows, merge_cols = imgLeft.cols+10+imgRight.cols;
    mergeImg.create(merge_rows, merge_cols, imgLeft.type());
    imgLeft.copyTo(mergeImg(Rect(0,0,imgLeft.cols,imgLeft.rows)));
    imgRight.copyTo(mergeImg(Rect(imgRight.cols+10, 0,imgRight.cols,imgRight.rows)));
    cv::resize(mergeImg, mergeImg,cv::Size(merge_cols,merge_rows));
    //在合并的图像上绘制水平线
    for(int i = 0; i < mergeImg.rows; i += 30){
        line(mergeImg, Point(0,i), Point(mergeImg.cols, i), Scalar(100, 0, 0), 1, 8);
        }
    cv::imshow(win, mergeImg);
    
}

// bool ComputeDispMat(const cv::Mat &img_left, const cv::Mat &img_right, cv::Mat &res_disp, cv::Mat &disp_Mat){
//     if(img_left.data == nullptr || img_right.data == nullptr){
//         std::cout << "读取影像失败！" << std::endl;
//         return 0;
//     }

//     // 设置 sgbm的参数
//     // Compute the disparity map using SGM algorithm
//     int min_disp = 0; // minimum disparity
//     int num_disp = 64; // max disparity - min disparity
//     int block_size = 5; // window size for computing pixel cost
//     int P1 = 8 * block_size * block_size; // penalty for disparity difference between pixels
//     int P2 = 32 * block_size * block_size; // penalty for large disparity differences
//     int disp12_diff = 1; // maximum allowed difference (in integer pixel units) in the left-right disparity check
//     int pre_filter_cap = 63; // truncation value for the prefiltered image pixels
//     int uniqueness_ratio = 11; // determines how unique the best match is in the match set
//     int speckle_window_size = 0; // the size of the window used to match pixels over time in the speckle filter stage
//     int speckle_range = 0; // the range of difference in values between pixels to consider them a match in the speckle filter stage

//     cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(min_disp, num_disp, block_size, P1, P2, disp12_diff, pre_filter_cap, uniqueness_ratio, speckle_window_size, speckle_range, cv::StereoSGBM::MODE_SGBM);
//     cv::Mat disp_map;
//     sgbm->compute(img_left, img_right, disp_map);

//     // Normalize the disparity map
//     cv::normalize(disp_map, res_disp, 0, 255, cv::NORM_MINMAX, CV_8UC1);

//     // Display the disparity map
//     cv::namedWindow("disparity_map", cv::WINDOW_NORMAL);
//     cv::imshow("disparity_map", res_disp);
//     cv::waitKey(0);

//     return true;
// }

bool ComputeDispMat(const cv::Mat &img_left, const cv::Mat &img_right, cv::Mat &res_disp, cv::Mat &disp_mat){
    if(img_left.data == nullptr || img_right.data == nullptr){
        std::cout << "读取影像失败！" << std::endl;
        return 0;
    }
    // 检查左右图像的行列（图像大小）是否一致
    if(img_left.rows != img_right.rows || img_right.cols != img_left.cols){
        std::cout << "左右影像尺寸不一致！" << std::endl;
        return 0;
    }

    // 定义的视差图大小，宽度由左边图像的列决定，高度由右边图像的行决定
    // 若左右图像的行列一致，则读入, 在 sgm.cpp中定义的 width和height的数据类型为  const sint32
    // static_cast 强制转换
    const sint32 width = static_cast<uint32>(img_left.cols);
    const sint32 height = static_cast<uint32>(img_right.rows);
    // printf("width=%d, height = %d.\n",width,height);

    // 左右影像的灰度数据，用一维数组保存
    // auto bytes_left = new uint8(width * height);
    // auto bytes_right = new uint8(width * height);
    uint8* bytes_left = new uint8[width * height];  
    uint8* bytes_right = new uint8[width * height];
    for(int i = 0; i< height; i++){
        for(int j = 0; j< width; j++){
            // bytes_left[i*width + j] = img_left.at<uint32>(i,j);
            // bytes_right[i*width + j] = img_right.at<uint32>(i,j);
            bytes_left[i * width + j] = img_left.at<uint8>(i,j);
            bytes_right[i * width + j] = img_right.at<uint8>(i,j);
        }
    }

    // SGM 匹配参数设计
    SemiGlobalMatching::SGMOption sgm_option;
    //聚合路径数
    // sgm_option.num_paths = 4;
    sgm_option.num_paths = 8;
    // 候选视差范围
    // sgm_option.min_disparity = 4;
    sgm_option.min_disparity = 0.2;
    sgm_option.max_disparity = 40;
    // Census窗口类型
    sgm_option.census_size = SemiGlobalMatching::Census5x5;
    // sgm_option.census_size = SemiGlobalMatching::Census3x3;

    //一致性检查
    sgm_option.is_check_lr = true;
    sgm_option.lrcheck_thres = 1.0f;
    //唯一性约束
    sgm_option.is_check_unique = true;
    sgm_option.uniqueness_ratio = 0.99;
    // 剔除小连通区
    sgm_option.is_remove_speckles = true;
    sgm_option.min_speckle_aera = 50;
    // 惩罚项
    // sgm_option.p1 = 10;
    sgm_option.p1 = 50;
    // sgm_option.p2 = 150;
    sgm_option.p2_init = 300;
    //视差图填充, 结果不可靠，若工程不建议，若科研可填充
    sgm_option.is_fill_holes = false;

    printf("w = %d, h = %d, d = [%d, %d]\n\n", width, height,sgm_option.min_disparity,sgm_option.max_disparity);

    // 定义sgm匹配视差实例
    SemiGlobalMatching sgm;

    //.............................//
    // 初始化
    printf("SGM initializing....\n");
    auto start = std::chrono::steady_clock::now();
    //初始化主要的设置参数和分配内存什么的
    if(!sgm.Initialize(width, height, sgm_option)){
        std::cout << "SGM初始化失败" << std::endl;
        return 0;
    }      
    auto end = std::chrono::steady_clock::now();
    auto tt = duration_cast<std::chrono::microseconds>(end - start);
    printf("SGM Initializing Done! Timing : %lf s\n\n", tt.count() / 1000.0);

    //..................................//
    // 匹配
    printf("SGM Matching...\n");
    start = std::chrono::steady_clock::now();
    //disparity数组保存子像素的视差结果
    // auto disparity = new float32[uint32(width * height)]();
    float32* disparity = new float32[uint32(width * height)]();
    printf("SGM disparity created...\n");

    // sgm.Match 主要入口，输入左图和右图，输出视差图
    if(!sgm.Match(bytes_left,bytes_right,disparity)){
        std::cout << "SGM匹配失败"<<std::endl;
        return -2;
    }
    end = std::chrono::steady_clock::now();
    tt = duration_cast<std::chrono::microseconds>(end - start);
    printf("SGM Matching Done! Timing : %lf s\n\n", tt.count() / 1000.0);

    //..........................................//
    // 显示视差图
    // 注意，计算点云不能用 disp_mat的数据，是用来显示和保存结果用的---将视差值转换为灰度值。
    // 计算电源要用上面的 disparity数组里的数据，
    disp_mat = cv::Mat(height, width, CV_8UC1);
    res_disp = cv::Mat::zeros(height, width, CV_32FC1);
    float min_disp = width, max_disp = -width;
    for(sint32 i = 0; i < height; i++){
        for(sint32 j = 0; j < width; j++){
            const float32 disp = disparity[i*width + j];
            // if(disp == Invalid_Float){ //这个逻辑错误隐藏的很深！！！
            if(disp != Invalid_Float){
                min_disp = std::min(min_disp, disp);
                max_disp = std::max(max_disp, disp);
            }
        }
    } 

    for(sint32 i = 0; i < height; i++){
        for(sint32 j = 0; j < width; j++){
            const float32 disp = disparity[i*width + j];
            if(disp == Invalid_Float){
                // disp_mat.data[i * width + j] = 0;
                // res_disp.at<float>(i,j) = -1;
                
                // 如果视差无效，请用近邻插值，如果无法插值，则设置为-1或0
                float32 left_disp = 0.0f, right_disp = 0.0f;
                if (j > 0 && disparity[i * width + j - 1] != Invalid_Float) {
                    left_disp = disparity[i * width + j - 1];
                }
                if (j < width - 1 && disparity[i * width + j + 1] != Invalid_Float) {
                    right_disp = disparity[i * width + j + 1];
                }
                if (left_disp != 0.0f && right_disp != 0.0f) {
                    disp_mat.data[i * width + j] = static_cast<uchar>((left_disp + right_disp) / 2.0f);
                    res_disp.at<float>(i,j) = (left_disp + right_disp) / 2.0f;
                } else if (left_disp != 0.0f) {
                    disp_mat.data[i * width + j] = static_cast<uchar>(left_disp);
                    res_disp.at<float>(i,j) = left_disp;
                } else if (right_disp != 0.0f) {
                    disp_mat.data[i * width + j] = static_cast<uchar>(right_disp);
                    res_disp.at<float>(i,j) = right_disp;
                } else {
                    disp_mat.data[i * width + j] = 0;
                    res_disp.at<float>(i,j) = -1; }          
            }
            else{
                // 若视差有效，则在此赋值， 比例系数
                disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
                // cout << "disp_mat" << disp_mat << " "; 
                res_disp.at<float>(i,j) =  disp;
                //cout << "disp" << disp << " "; 
            }
        }
    }
    return 1;
}

int main(int argc, char** argv){
    ros::init(argc, argv,"mapping_node");
    // ros::NodeHandle nh("-");
    ros::NodeHandle nh("~");

    // 从launch文件中读取参数文件的路径
    string pkg_path = argv[1];
    string config_file_name = argv[2];
    string config_file = pkg_path + config_file_name;  // vins eurco param
    cout << "config_file" << config_file << endl;
    ROS_INFO("Config_file: %s", config_file.c_str());

    cv::FileStorage fs_config (config_file, FileStorage::READ);
    string data_path;

    fs_config["pose_graph_path"] >> data_path;
    // cout<< "data_path" << data_path << endl;
    ROS_INFO("Data_path: %s", data_path.c_str());

    ifstream file;
    string pose_graph_file = data_path + "pose_graph.txt";
    string line;

    file.open(pose_graph_file);
    if(!file.is_open())
        ROS_ERROR("Read file failed");
    
    // cv::Mat im_ref, im_cur;
    // cv::Mat im_ref_right, im_cur_right;
    int height = 752;
    int width = 480;

    cv::Mat im_ref = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat im_cur = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat im_ref_right = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat im_cur_right = cv::Mat::zeros(height, width, CV_8UC1);

    // 相机参数 和校正相关的
    cv::Mat K_left, K_right, d_left, d_right, R_clc2, t_clc2, P_left, P_right, R_left, R_right,
            Q, mx_left, my_left, mx_right, my_right;
    cv::Rect validRoiLeft, validRoiRight;
    int cols, rows;
    cv::Size imgSize;
    float focal_length, baseline;
    float focal_length_inv;
    bool rectify;

    focal_length = fs_config["focal_length"];
    baseline = fs_config["baseline"];
    fs_config["image_width"] >> cols;
    fs_config["image_height"] >> rows;
    fs_config["cam0_intrinsics"] >> K_left;   
    fs_config["cam0_distortion_coeffs"] >> d_left;
    fs_config["cam1_intrinsics"] >> K_right;   
    fs_config["cam1_distortion_coeffs"] >> d_right;
    fs_config["R_clc2"] >> R_clc2;   
    fs_config["t_clc2"] >> t_clc2;
    imgSize = cv::Size(cols,rows);
    float cx, cy;
    cx = fs_config["CX"];
    cy = fs_config["CY"];


    //计算校正参数
    cv::stereoRectify(K_left, d_left, K_right, d_right, imgSize, R_clc2, t_clc2,
                    R_left, R_right, P_left, P_right, Q, CV_CALIB_ZERO_DISPARITY,
                    0, imgSize, &validRoiLeft, &validRoiRight);
    cv::initUndistortRectifyMap(K_left, d_left, R_left, P_left, imgSize, CV_32FC1, mx_left, my_left);
    cv::initUndistortRectifyMap(K_right, d_right, R_right, P_right, imgSize, CV_32FC1, mx_right, my_right);

    int cnt = 0;
    // 读入txt 文件 位姿文件 获取每一行 相当于遍历每一帧
    while(getline(file, line)){
        cnt++;
        if(cnt<9){
            continue;
        }
        else{
            cnt = 0;
        }

        //每一行的信息
        if(line.size() == 0){
            break;   //循环体内判断是否空行
        }
        stringstream per_line(line);
        int id;
        double time_stamp;
        double VIO_Tx, VIO_Ty, VIO_Tz;
        double PG_Tx,  PG_Ty, PG_Tz;
        double VIO_Qw, VIO_Qx, VIO_Qy, VIO_Qz;
        double PG_Qw,  PG_Qx, PG_Qy, PG_Qz;
        double loop_info_0, loop_info_1, loop_info_2, loop_info_3;
        double loop_info_4, loop_info_5, loop_info_6, loop_info_7;
        int loop_index;
        int keypoints_num;
        per_line >> id >> time_stamp >>  VIO_Tx >>  VIO_Ty >> VIO_Tz >> PG_Tx >>  PG_Ty >> PG_Tz >>
                                                                            VIO_Qw >> VIO_Qx >> VIO_Qy >> VIO_Qz >> PG_Qw >> PG_Qx >> PG_Qy >> PG_Qz >>
                                                                            loop_index >>
                                                                            loop_info_0 >> loop_info_1 >> loop_info_2 >> loop_info_3 >>
                                                                            loop_info_4  >> loop_info_5 >> loop_info_6 >> loop_info_7 >>
                                                                            keypoints_num;

        string image_path = data_path + to_string(id) + "_image.png";
        string image_path_right = data_path + to_string(id) + "_image_right.png";

        cv::Mat im_cur = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        cv::Mat im_cur_right = cv::imread(image_path_right, cv::IMREAD_GRAYSCALE);

         // 校正左目和右目的图像信息
        cv::Mat imgRect, imgRect_right;
        cv::remap(im_cur, imgRect, mx_left, my_left, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
        cv::remap(im_cur_right, imgRect_right, mx_right, my_right, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

        show_stereo(im_cur, im_cur_right, "Raw");
        show_stereo(imgRect, imgRect_right, "Rectified");

        // 预处理左目图像
 
        // 预处理左目和右目图像，第一步增强光照条件，第二步增强对比度，第三步去除噪声，第四步增强边缘部分的对比度
        cv::Mat imgRect_left_processed, imgRect_right_processed;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(imgRect, imgRect_left_processed);
        clahe->apply(imgRect_right, imgRect_right_processed);
        cv::equalizeHist(imgRect_left_processed, imgRect_left_processed);
        cv::equalizeHist(imgRect_right_processed, imgRect_right_processed);
        cv::GaussianBlur(imgRect_left_processed, imgRect_left_processed, cv::Size(3, 3), 0, 0);
        cv::GaussianBlur(imgRect_right_processed, imgRect_right_processed, cv::Size(3, 3), 0, 0);
        cv::Mat kernel = (cv::Mat_<float>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1);
        cv::filter2D(imgRect_left_processed, imgRect_left_processed, -1, kernel);
        cv::filter2D(imgRect_right_processed, imgRect_right_processed, -1, kernel);
        cv::imshow("imgRect_left_processed",imgRect_left_processed);
        cv::imshow("imgRect_right_processed",imgRect_right_processed);

 
        // 对图像进行一层金字塔优化，加快图像计算的速度，请注意只优化一层
        cv::Mat imgRect_left_processed_pyramid, imgRect_right_processed_pyramid;
        cv::pyrDown(imgRect_left_processed, imgRect_left_processed_pyramid);
        cv::pyrDown(imgRect_right_processed, imgRect_right_processed_pyramid);

        
        // SGM计算视差图
        cv::Mat disp_mat, img_disp, depth_map;
        // ComputeDispMat(imgRect, imgRect_right, disp_mat, img_disp);
        ComputeDispMat(imgRect_left_processed, imgRect_right_processed, disp_mat, img_disp);

        cv::imshow("img_disp",img_disp);
        
        cv::waitKey();
    } 

    return 0;
}