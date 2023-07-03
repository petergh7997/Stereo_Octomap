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
    for(int i = 0; i < mergeImg.rows; i + = 30){
        line(mergeImg, Point(0,i), Point(mergeImg.cols, i), Scalar(100, 0, 0), 1, 8);
        }
    cv::imshow(win, mergeImg);
    
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

        // ROS_INFO("image size: %d",  im_cur.size());
        // ROS_INFO("image type: %s", im_cur.type());

        // ROS_INFO("image_right size: %d",  im_cur_right.size());
        // ROS_INFO("image_right type: %s", im_cur_right.type());

    //     cv::imshow("image cur", im_cur);
    //     if(!im_ref.empty()){
    //         cv::imshow("image ref", im_ref);
    //     }
    //     im_ref = im_cur.clone();

    //    // TODO:  添加右目相机的图片参考帧和当前帧
    //     cv::imshow("im_cur_right", im_cur_right);
    //     if(!im_cur_right.empty()){
    //             cv::imshow("im_ref_right", im_ref_right);                   
    //         }
    //     im_ref_right = im_cur_right.clone();
         // 校正左目和右目的图像信息
        cv::Mat imgRect, imgRect_right;
        cv::remap(im_cur, imgRect, mx_left, my_left, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
        cv::remap(im_cur_right, imgRect_right, mx_right, my_right, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

        show_stereo(im_cur, im_cur_right, "Raw");
        show_stereo(imgRect, imgRect_right, "Rectified");

        cv::waitKey();
    } 

    return 0;
}