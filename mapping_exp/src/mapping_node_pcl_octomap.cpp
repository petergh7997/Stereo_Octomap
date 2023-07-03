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
// #include <eigen3/Dense>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
// for sgm
#include "include/SemiGlobalMatching.h"

#include <QCoreApplication>
#include <QVector>

// 八叉树 库头文件
#include <octomap/OcTree.h>
// pcl滤波器头文件
// #include <pcl/filters/pcl_filter.h>
// #include "pcl_filter.h"
#include "include/pcl_filter.h"

using namespace std;
using namespace std::chrono;
using namespace cv;

// 发布点云，pub_current_cloud和pub_all_cloud变量是对象，可以通过调用其成员函数来发布数据。
ros::Publisher pub_current_cloud, pub_all_cloud;

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
    sgm_option.num_paths = 4;
    // 候选视差范围
    // sgm_option.min_disparity = 4;
    // sgm_option.max_disparity = 32;
    sgm_option.min_disparity = 8;
    sgm_option.max_disparity = 26;
    // Census窗口类型
    sgm_option.census_size = SemiGlobalMatching::Census5x5;
    //一致性检查
    sgm_option.is_check_lr = true;
    sgm_option.lrcheck_thres = 1.0f;
    //唯一性约束
    sgm_option.is_check_unique = true;
    sgm_option.uniqueness_ratio = 0.99;
    // 剔除小连通区
    sgm_option.is_remove_speckles = true;
    sgm_option.min_speckle_aera = 80;
    // 惩罚项
    // sgm_option.p1 = 10;
    sgm_option.p1 = 20;
    // sgm_option.p2 = 150;
    sgm_option.p2_init = 200;
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
            // if(disp == Invalid_Float){
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
                disp_mat.data[i * width + j] = 0;
                res_disp.at<float>(i,j) = -1;
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
// void publishPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pcl_cloud, const ros::Time &time_stamp) {
//   // 创建PointCloud2消息，将PointXYZ类型的点云数据转换为PointCloud2消息
//   sensor_msgs::PointCloud2 msg_cloud;
//   pcl::toROSMsg(*pcl_cloud, msg_cloud);
  
//   // 设置PointCloud2消息的时间戳和参考坐标系信息
//   msg_cloud.header.stamp = time_stamp;
//   msg_cloud.header.frame_id = "world";
//    // 发布PointCloud2消息到指定的topic上
//  pub_current_cloud.publish(msg_cloud);
// }

/*
发布点云消息的函数，接受两个参数：一个是指向pcl::PointCloudpcl::PointXYZRGB类型点云指针的指针，
另一个是std_msgs::Header类型的消息头。
函数中的pub_current_cloud是一个已经创建好的ros::Publisher类型的发布器，用于将消息发布到ROS系统中的某个topic上。

在函数内部，首先调用了pcl::toROSMsg函数将输入的pcl::PointCloudpcl::PointXYZRGB类型点云数据转换为PointCloud2类型的消息dense_point_cloud。然后，将消息头header赋值给PointCloud2消息的header成员。最后，使用已有的ros::Publisher类型的发布器pub_current_cloud，将PointCloud2类型的点云消息dense_point_cloud发布到指定的topic上。

需要注意的是，在调用publish函数之前，确保ros::Publisher类型的发布器已经创建，并且topic已被定义。
*/
// 发布点云的函数定义
void PubCurrentCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std_msgs::Header header){
// 创建PointCloud2消息，将PointXYZ类型的点云数据转换为PointCloud2消息
sensor_msgs::PointCloud2 dense_point_cloud;
pcl::toROSMsg(*cloud, dense_point_cloud);
cout<< "dense_point_cloud is transfered to ROSMsg; \n" << endl;

// 设置PointCloud2消息的时间戳和参考坐标系信息
dense_point_cloud.header = header; // 注意，header是放在后面的
pub_current_cloud.publish(dense_point_cloud);
cout<< "dense_point_cloud is published; \n" << endl;
}

 // 发布所有点云的函数定义
void PubAllCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr all_cloud, std_msgs::Header header){
    // 创建PointCloud2消息，将PointXYZ类型的点云数据转换为PointCloud2消息
    sensor_msgs::PointCloud2 all_dense_point_cloud;
    pcl::toROSMsg(*all_cloud, all_dense_point_cloud);
    cout<< "all_dense_point_cloud is transfered to ROSMsg; \n" << endl;

    // 设置PointCloud2消息的时间戳和参考坐标系信息
    all_dense_point_cloud.header = header; // 注意，header是放在后面的
    // pub_current_cloud.publish(all_dense_point_cloud);
    pub_all_cloud.publish(all_dense_point_cloud);
    cout<< "all_dense_point_cloud is published; \n" << endl;
}

int main(int argc, char** argv){
    ros::init(argc, argv,"mapping_node");
    ros::NodeHandle nh(""); 

// QObject::connect: Cannot queue arguments of type 'QVector<int>'
//(Make sure 'QVector<int>' is registered using qRegisterMetaType().)
    QCoreApplication a(argc, argv);
    qRegisterMetaType<QVector<int>>("QVector<int>");

    // 从launch文件中读取参数文件的路径
    string pkg_path = argv[1];
    string config_file_name = argv[2];
    string config_file = pkg_path + config_file_name;  // vins eurco param
    cout << "config_file" << config_file << endl;
    ROS_INFO("Config_file: %s", config_file.c_str());

    // 设置发布的topic
    pub_current_cloud = nh.advertise<sensor_msgs::PointCloud2>("current_cloud", 1000);
    pub_all_cloud = nh.advertise<sensor_msgs::PointCloud2>("all_cloud", 1000);

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
    // float focal_length, baseline; 
    
    // float focal_length_inv;
    bool rectify;

    float focal_length = fs_config["focal_length"];
    float baseline = fs_config["baseline"];

    float focal_length_inv = 1.0 / focal_length;
    fs_config["image_width"] >> cols;
    fs_config["image_height"] >> rows;
    fs_config["cam0_intrinsics"] >> K_left;   
    fs_config["cam0_distortion_coeffs"] >> d_left;
    fs_config["cam1_intrinsics"] >> K_right;   
    fs_config["cam1_distortion_coeffs"] >> d_right;
    fs_config["R_clc2"] >> R_clc2;   
    fs_config["t_clc2"] >> t_clc2;
    imgSize = cv::Size(cols,rows);

    // 校正之后的内参，来自orb-slam2

    float fx, fy, cx, cy, bf;
    cx = fs_config["Camera.cx"];
    cy = fs_config["Camera.cy"];
    fx = fs_config["Camera.fx"];
    fy = fs_config["Camera.fy"];
    bf = fs_config["Camera.bf"];

    // imu-相机外参
    cv::Mat cvTic0;
    fs_config["T_i_c0"] >> cvTic0;
    cout << "cvTic0 \n" << cvTic0 << endl;
    Eigen::Matrix4d Tic0;
    cv::cv2eigen(cvTic0, Tic0);
    Eigen::Matrix3d R_ic0 = Tic0.block<3,3>(0,0);
    Eigen::Vector3d t_ic0 = Tic0.block<3,1>(0,3);
    Eigen::Quaterniond Q_ic0(R_ic0);
    Q_ic0.normalize();
    cout << "Tic0 \n" << Tic0 << endl;

    //计算校正参数
    cv::stereoRectify(K_left, d_left, K_right, d_right, imgSize, R_clc2, t_clc2,
                    R_left, R_right, P_left, P_right, Q, CV_CALIB_ZERO_DISPARITY,
                    0, imgSize, &validRoiLeft, &validRoiRight);
    
    
    cout << "R_left: \n" << R_left << endl;
    cout << "R_right: \n" << R_right << endl;
    cout << "P_left: \n" << P_left << endl;
    cout << "P_right: \n" << P_right << endl;
                    
    cv::initUndistortRectifyMap(K_left, d_left, R_left, P_left, imgSize, CV_32FC1, mx_left, my_left);
    cv::initUndistortRectifyMap(K_right, d_right, R_right, P_right, imgSize, CV_32FC1, mx_right, my_right);

    // 使用PCL（Point Cloud Library）库创建两个用于存储和处理点云数据的对象：all_cloud 和 cur_cloud_filtered。
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr 是一个指向 pcl::PointCloud<pcl::PointXYZRGB> 类型对象的智能指针。
    // pcl::PointCloud<pcl::PointXYZRGB> 是PCL库中的一个模板类，用于表示点云，其中每个点包含位置（X, Y, Z）和颜色信息（RGB）。
    // new pcl::PointCloud<pcl::PointXYZRGB> 是动态创建一个 pcl::PointCloud<pcl::PointXYZRGB> 类型的对象，并返回指向它的指针。这个指针被用来初始化 all_cloud 和 cur_cloud_filtered。
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr all_cloud (new pcl::PointCloud<pcl::PointXYZRGB> );
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cur_cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB> );


    int cnt = 0;
    // 读入txt 文件 位姿文件 获取每一行 相当于遍历每一帧
    // ros::Rate loop_rate(30);
    ros::Rate loop_rate(100);
       
    while(ros::ok() && getline(file, line)){
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
        // VIO_T 和 VIO_Q的定义
        Eigen::Vector3d VIO_T;
        Eigen::Quaterniond VIO_Q;
        VIO_Q.w() = VIO_Qw;
        VIO_Q.x() = VIO_Qx;
        VIO_Q.y() = VIO_Qy;
        VIO_Q.z() = VIO_Qz;

        VIO_T.x() = VIO_Tx;
        VIO_T.y() = VIO_Ty;
        VIO_T.z() = VIO_Tz;

        // 视差图转换为点云中的 CAM_R和CAM_T
        Eigen::Vector3d CAM_T = VIO_T + t_ic0;
        Eigen::Quaterniond CAM_Q = VIO_Q * Q_ic0;
        Eigen::Matrix3d CAM_R = CAM_Q.toRotationMatrix();

// 定义一个ROS标准消息类型std_msgs::Header的消息变量header，并对其中的frame_id和stamp两个成员变量进行了赋值。

    //frame_id: 用于指定消息所属的参考坐标系，在这里设置成了"world"，表示该消息发生在世界坐标系下。
   // stamp: 用于指定消息的时间戳，这里通过ros::time函数获取输入的time_stamp来设置。

//这个消息变量可以用于ROS系统中的消息传递，比如可以将这个消息发布到某个topic上，也可以作为某个节点接收到消息的一部分。
        std_msgs::Header header;
        header.frame_id = "world";
        // header.stamp = ros::time(time_stamp);
        header.stamp = ros::Time(time_stamp);

        string image_path = data_path + to_string(id) + "_image.png";
        string image_path_right = data_path + to_string(id) + "_image_right.png";

        cv::Mat im_cur = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        cv::Mat im_cur_right = cv::imread(image_path_right, cv::IMREAD_GRAYSCALE);

        // 校正左目和右目的图像的畸变
       // cv::remap函数接受原始图像、校正后的图像、左目和右目的校正映射参数（mx_left、my_left、mx_right、my_right）以及插值方法和边界处理方式作为输入。
      // 它会根据校正映射参数对原始图像进行重映射，从而消除图像中的畸变。校正后的图像imgRect和imgRect_right将用于后续的视差计算和点云生成。 
        cv::Mat imgRect, imgRect_right;
        cv::remap(im_cur, imgRect, mx_left, my_left, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
        cv::remap(im_cur_right, imgRect_right, mx_right, my_right, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

        // 对校正后的左目和右目图像进行预处理，使用滤波器（如高斯滤波器）进行图像去噪，以减少在后续计算中可能引入的错误
        // cv::GaussianBlur(imgRect, imgRect, cv::Size(3,3), 0, 0);
        // cv::GaussianBlur(imgRect_right, imgRect_right, cv::Size(3,3), 0, 0);
        cv::GaussianBlur(imgRect, imgRect, cv::Size(5,5), 0, 0);
        cv::GaussianBlur(imgRect_right, imgRect_right, cv::Size(5,5), 0, 0);          
        // 对校正后的左目和右目图像进行预处理，图像增强：可以应用直方图均衡化等技术，以增强图像的对比度和清晰度，从而有助于更好地提取视差信息。
        cv::equalizeHist(imgRect, imgRect);
        cv::equalizeHist(imgRect_right, imgRect_right);
        // 光照校正：如果左右目图像的光照条件不同，可以进行光照校正，使它们具有相似的亮度和颜色分布。
        // 光照校正
        cv::Mat imgRect_norm, imgRect_right_norm;
        cv::normalize(imgRect, imgRect_norm, 0, 255, cv::NORM_MINMAX);
        cv::normalize(imgRect_right, imgRect_right_norm, 0, 255, cv::NORM_MINMAX);
        imgRect = imgRect_norm;
        imgRect_right = imgRect_right_norm;

        // 构造每一帧点云时，分别将图像进行金字塔向上缩放一层，分别保存结果，以便对比运行速度
        // std::vector<cv::Mat> pyramid_imgRect, pyramid_imgRect_right;
        // for(int i = 0; i < 4; i++){
        //     cv::Mat tmp_imgRect, tmp_imgRect_right;
        //     cv::pyrDown(imgRect, tmp_imgRect, cv::Size(imgRect.cols/2, imgRect.rows/2));
        //     cv::pyrDown(imgRect_right, tmp_imgRect_right, cv::Size(imgRect_right.cols/2, imgRect_right.rows/2));
        //     pyramid_imgRect.push_back(tmp_imgRect);
        //     pyramid_imgRect_right.push_back(tmp_imgRect_right);
        //     // imgRect = tmp_imgRect;
        //     // imgRect_right = tmp_imgRect_right;
        // }
      
        // if (!pyramid_imgRect.empty()) {
        //     imgRect = pyramid_imgRect[1];
        //     imgRect_right = pyramid_imgRect[1];
        // }

 
        // 构造每一帧点云时，将图像进行金字塔向上缩放一层，注意不需要四层，保存结果，以便对比运行速度
        cv::Mat imgRect_pyramid, imgRect_right_pyramid;
        cv::pyrDown(imgRect, imgRect_pyramid, cv::Size(imgRect.cols/2, imgRect.rows/2));
        cv::pyrDown(imgRect_right, imgRect_right_pyramid, cv::Size(imgRect_right.cols/2, imgRect_right.rows/2));
        // cv::pyrDown(imgRect, imgRect_pyramid, cv::Size(imgRect.cols/4, imgRect.rows/4));
        // cv::pyrDown(imgRect_right, imgRect_right_pyramid, cv::Size(imgRect_right.cols/4, imgRect_right.rows/4));
        
        
        show_stereo(im_cur, im_cur_right, "Raw");
        show_stereo(imgRect, imgRect_right, "Rectified");

        // SGM计算视差图
        cv::Mat disp_mat, img_disp, depth_map;
        ComputeDispMat(imgRect, imgRect_right, disp_mat, img_disp);
        cv::imshow("img_disp",img_disp);
        cout<< "img_disp is displayed; \n" << endl;

 
// 检查视差图的无效视差占据图像的比例，如果高于5%，请警告
        // int invalid_disp_count = 0;
        // for(int i = 0; i < height; i++){
        //     for(int j = 0; j < width; j++){
        //         float disp = disp_mat.at<float>(i,j);
        //         if(disp == -1 || disp == 0){
        //             invalid_disp_count++;
        //         }
        //     }
        // }
        // float invalid_disp_ratio = (float)invalid_disp_count / (float)(height * width);
        // if(invalid_disp_ratio > 0.05){
        //     cout << "Warning: Invalid disparity ratio is " << invalid_disp_ratio << endl;
        // }      
        
        // 有了视差信息后，遍历每一张图像的宽和高; 检查每一个视差的有效性，去掉视差无效的点，将其深度赋值为-1;
        // 计算每个像素点的深度，并存入二维深度队列中。

        // Step 1: 计算深度
        const int width = imgRect.cols;
        const int height = imgRect.rows;

        // 初始化深度图
        depth_map = cv::Mat::zeros(height, width, CV_32FC1);

        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                float disp = disp_mat.at<float>(i,j);
                if(disp == -1 || disp == 0){
                    depth_map.at<float>(i,j) = -1;
                    continue;
                }
                float Z = bf/disp;
                depth_map.at<float>(i,j) = Z;
                // cout << disp<<" ";   
            }
        }
        
        // 使用最近邻插值，根据disp == -1或者disp == 0周围的点云深度来估计缺失的点深度的值
        // for(int i = 0; i < height; i++){
        // for(int j = 0; j < width; j++){
        //     float disp = disp_mat.at<float>(i,j);
        //     if(disp == -1 || disp == 0){
        //         // depth_map.at<float>(i,j) = -1;
        //         // 使用最近邻插值，根据disp == -1或者disp == 0周围的点云深度来估计缺失的点深度的值
        //         int window_size = 5; // 设置窗口大小
        //         float sum_depth = 0;
        //         int valid_depth_count = 0;
        //         for(int m = max(0, i - window_size); m <= min(height - 1, i + window_size); m++){
        //             for(int n = max(0, j - window_size); n <= min(width - 1, j + window_size); n++){
        //                 float neighbor_disp = disp_mat.at<float>(m,n);
        //                 if(neighbor_disp > 0){
        //                     float neighbor_depth = bf / neighbor_disp;
        //                     sum_depth += neighbor_depth;
        //                     valid_depth_count++;
        //                 }
        //             }
        //         }
        //         if(valid_depth_count > 0){
        //             depth_map.at<float>(i,j) = sum_depth / valid_depth_count;
        //         } else {
        //             depth_map.at<float>(i,j) = -1;
        //         }
        //         continue;
        //     }

        //     float Z = bf/disp;
        //     depth_map.at<float>(i,j) = Z;

        //     }
        //     // float Z = bf/disp;
        //     // depth_map.at<float>(i,j) = Z;
        //     // cout << disp<<" ";   
        // }
    

        // Step 2: 视差图转换成点云
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cur_cloud(new pcl::PointCloud<pcl::PointXYZRGB> );

        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                float Z = depth_map.at<float>(i,j);
                if(Z == -1){
                    continue;
                }

                float X = (j-cx) * Z * focal_length_inv; // 横着的，图像宽度方向，是x
                float Y = (i-cy) * Z * focal_length_inv; // 竖着的，图像高度方向，是y

                Eigen::Vector3d pt_cam, pt_body, pt_world;
                pt_cam = Eigen::Vector3d(X,Y,Z); // rviz的Y和相机系相反
                pt_body = CAM_R * pt_cam + CAM_T;  // 转换到相机系， 注意这里的坐标变换
                // pt_body = pt_cam; // 如果不转换，点云就飞了

                pcl::PointXYZRGB pt;                
                pt.x = pt_body(0);
                pt.y = pt_body(1);
                pt.z = pt_body(2);
                pt.b = imgRect.at<uchar>(i,j);
                pt.g = imgRect.at<uchar>(i,j);
                pt.r = imgRect.at<uchar>(i,j);
                cur_cloud->points.push_back(pt);
        }
        }
        // 对点云进行去除离群异常点去除

         // 对点云进行降采样
        // Create the filtering object
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud(cur_cloud);
        // sor.setLeafSize(0.01f, 0.01f, 0.01f);
        sor.setLeafSize(0.05f, 0.05f, 0.05f);
        sor.filter(*cur_cloud);

        StatisticalFilter(cur_cloud, cur_cloud_filtered);
        
        // *all_cloud 是对智能指针进行解引用，获取其所指向的对象。
        // 将 all_cloud 所指向的对象和 cur_cloud_filtered 所指向的对象进行相加，然后将结果赋值给 all_cloud 所指向的对象。
        // 这里的 + 操作符是 pcl::PointCloud<pcl::PointXYZRGB> 类型定义的，用于合并两个点云。
        *all_cloud = *all_cloud + *cur_cloud_filtered;
        
      

        // Step 3: 发布，注意写 publisher， 在launch文件里添加 rviz配置
        PubCurrentCloud(cur_cloud, header);
        PubAllCloud(all_cloud,header);
        
        // Step 4: 将每一帧点云转化为octomap地图
        

 
        
           
        // cv::waitKey();
    } 
    // Step 5: 使用pcl库保存拼接后的点云

    return 0;
}