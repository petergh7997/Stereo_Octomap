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
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf/transform_broadcaster.h> 
#include <tf/tf.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d/calib3d_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;
using namespace std::chrono;
using namespace cv;

int main(int argc, char** argv){
    ros::init(argc, argv,"mapping_node");
    // ros::NodeHandle nh("-");
    ros::NodeHandle nh("~");
    
        // 创建 "world" 固定坐标系
    geometry_msgs::TransformStamped init_transformStamped;
    init_transformStamped.header.stamp = ros::Time::now();
    init_transformStamped.header.frame_id = "map"; // 假设原本世界坐标系是map，这里使用之作为父坐标系
    init_transformStamped.child_frame_id = "world";

    init_transformStamped.transform.translation.x = 0.0;
    init_transformStamped.transform.translation.y = 0.0;
    init_transformStamped.transform.translation.z = 0.0;

    init_transformStamped.transform.rotation.x = 0.0;
    init_transformStamped.transform.rotation.y = 0.0;
    init_transformStamped.transform.rotation.z = 0.0;
    init_transformStamped.transform.rotation.w = 1.0;

    // 发送初始的变换关系
    tf2_ros::StaticTransformBroadcaster init_broadcaster;
    init_broadcaster.sendTransform(init_transformStamped);

    std::string parent_frame_id = "world";   // 父坐标系
    std::string left_child_frame_id = "camera_left";    // 左目相机坐标系
    std::string right_child_frame_id = "camera_right";  // 右目相机坐标系

    // 初始化 TransformBroadcaster
    tf2_ros::TransformBroadcaster broadcaster;  

     // 创建位姿信息发布者，发布geometry_msgs/PoseStamped类型的消息
    // ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("pose", 100);
    ros::Publisher pose_pub_left = nh.advertise<geometry_msgs::PoseStamped>("pose_left", 1000);
    ros::Publisher pose_pub_right = nh.advertise<geometry_msgs::PoseStamped>("pose_right", 1000);

    // 创建图像信息发布者，发布sensor_msgs/Image类型的消息
    ros::Publisher img_pub = nh.advertise<sensor_msgs::Image>("image", 1000);
    ros::Publisher img_right_pub = nh.advertise<sensor_msgs::Image>("image_right", 1000);

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

    // 设定循环频率
    ros::Rate loop_rate(30);

    // 读入txt 文件 位姿文件 获取每一行 相当于遍历每一帧
    while(ros::ok() && getline(file, line)){
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

        VIO_Qw = PG_Qw = 1.0;
        VIO_Qx = VIO_Qy = VIO_Qz = PG_Qx = PG_Qy = PG_Qz = 0.0;
        
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
        
        // 发布左目和右目图像信息
        sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", im_cur).toImageMsg();
        img_msg->header.stamp = ros::Time::now();
        img_msg->header.frame_id = "camera_left";
        img_pub.publish(img_msg);

        sensor_msgs::ImagePtr img_right_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", im_cur_right).toImageMsg();
        img_right_msg->header.stamp = ros::Time::now();
        img_right_msg->header.frame_id = "camera_right";
        img_right_pub.publish(img_right_msg);

        // 发布左目的位姿信息
        geometry_msgs::PoseStamped pose_left;
        pose_left.header.stamp = ros::Time::now();
        pose_left.header.frame_id = "world_left";
        pose_left.pose.position.x = VIO_Tx;
        pose_left.pose.position.y = VIO_Ty;
        pose_left.pose.position.z = VIO_Tz;
        pose_left.pose.orientation.w = VIO_Qw;
        pose_left.pose.orientation.x = VIO_Qx;
        pose_left.pose.orientation.y = VIO_Qy;
        pose_left.pose.orientation.z = VIO_Qz;
        pose_pub_left.publish(pose_left);

        // 发布右目的位姿信息
        geometry_msgs::PoseStamped pose_right;
        pose_right.header.stamp = ros::Time::now();
        pose_right.header.frame_id = "world_right";
        pose_right.pose.position.x = PG_Tx;
        pose_right.pose.position.y = PG_Ty;
        pose_right.pose.position.z = PG_Tz;
        pose_right.pose.orientation.w = PG_Qw;
        pose_right.pose.orientation.x = PG_Qx;
        pose_right.pose.orientation.y = PG_Qy;
        pose_right.pose.orientation.z = PG_Qz;
        pose_pub_right.publish(pose_right); 
    
        // 发布左目相机坐标变换消息
        geometry_msgs::TransformStamped left_transform;
        left_transform.header.stamp = ros::Time::now();
        left_transform.header.frame_id = parent_frame_id;
        left_transform.child_frame_id = left_child_frame_id;
        left_transform.transform.translation.x = VIO_Tx;
        left_transform.transform.translation.y = VIO_Ty;
        left_transform.transform.translation.z = VIO_Tz;
        left_transform.transform.rotation.w = VIO_Qw;
        left_transform.transform.rotation.x = VIO_Qx;
        left_transform.transform.rotation.y = VIO_Qy;
        left_transform.transform.rotation.z = VIO_Qz;
        broadcaster.sendTransform(left_transform);

        // 发布右目相机坐标变换消息
        geometry_msgs::TransformStamped right_transform;
        right_transform.header.stamp = ros::Time::now();
        right_transform.header.frame_id = parent_frame_id;
        right_transform.child_frame_id = right_child_frame_id;
        right_transform.transform.translation.x = PG_Tx;
        right_transform.transform.translation.y = PG_Ty;
        right_transform.transform.translation.z = PG_Tz;
        right_transform.transform.rotation.w = PG_Qw;
        right_transform.transform.rotation.x = PG_Qx;
        right_transform.transform.rotation.y = PG_Qy;
        right_transform.transform.rotation.z = PG_Qz;
        broadcaster.sendTransform(right_transform);


        // 读取文件中下一帧的图像数据
        if (!getline(file, line)) {
            break;
        }

        // 等待指定频率的循环
        ros::spinOnce();
        loop_rate.sleep();
    } 

    return 0;
}