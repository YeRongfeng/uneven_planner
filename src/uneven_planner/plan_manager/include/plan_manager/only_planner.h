#pragma once

#include <fstream>
#include <string.h>
#include <random>
#include <time.h>
#include <cmath>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <chrono>

#include <ros/ros.h>
#include <ros/console.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// Service definitions
#include <std_srvs/Trigger.h>

#include "uneven_map/uneven_map.h"
#include "front_end/kino_astar.h"
#include "back_end/alm_traj_opt.h"
#include "mpc_controller/SE2Traj.h"

namespace uneven_planner
{
    // 规划请求结构体（支持批量规划）
    struct PlanningRequest {
        Eigen::Vector3d start_pose;
        Eigen::Vector3d target_pose;
        int env_id;
        int path_id;
        ros::Time timestamp;
    };

    class OnlyPlanner
    {
        private:
            bool in_plan = false;
            std::atomic<bool> batch_planning_enabled{false};  // 批量规划开关
            std::mutex planning_mutex;  // 保护规划状态的互斥锁
            std::queue<PlanningRequest> planning_queue;  // 规划请求队列
            
            double piece_len;
            double mean_vel;
            double init_time_times;
            double yaw_piece_times;
            double init_sig_vel;
            bool enable_optimization;  // 是否启用轨迹优化
            int resample_points;  // 轨迹重采样的固定点数
            double min_planning_distance;  // 最短规划距离
            double max_planning_distance;  // 最远规划距离
            bool use_min_distance_constraint;  // 是否使用最短距离约束
            bool use_max_distance_constraint;  // 是否使用最远距离约束

            // 固定位姿参数
            std::vector<double> start_fixed;  // 起始点固定值 [x, y, yaw]，None用NaN表示
            std::vector<double> end_fixed;    // 目标点固定值 [x, y, yaw]，None用NaN表示
            bool use_start_fixed;  // 是否使用起始点固定
            bool use_end_fixed;    // 是否使用目标点固定

            // 重试机制参数
            int pose_retry_count;      // 当前起终点重试次数
            int max_pose_retries;      // 最大起终点重试次数
            int map_retry_count;       // 当前地图重新生成次数  
            int max_map_retries;       // 最大地图重新生成次数

            Eigen::Vector3d odom_pos;
            string bk_dir;

            UnevenMap::Ptr uneven_map;
            KinoAstar::Ptr kino_astar;
            ALMTrajOpt traj_opt;
            SE2Trajectory opted_traj;

            ros::Publisher traj_pub;
            ros::Publisher success_pub;
            ros::Publisher map_regen_pub;  // 地图重新生成请求发布者
            ros::Subscriber start_sub;
            ros::Subscriber target_sub;
            ros::Publisher                  origin_pub;
            ros::Publisher                  filtered_pub;
            ros::Publisher                  zb_pub;
            ros::Publisher                  so2_test_pub;
            ros::Timer                      vis_timer;
            sensor_msgs::PointCloud2        origin_cloud_msg;
            sensor_msgs::PointCloud2        filtered_cloud_msg;
            visualization_msgs::MarkerArray so2_test_msg;
            visualization_msgs::Marker      zb_msg;    
            bool                            map_ready = false;
            ros::ServiceServer              start_data_srv;   // service to trigger data generation

            // 私有辅助函数
            mpc_controller::SE2Traj createTrajectoryMsg(const SE2Trajectory& traj);
            mpc_controller::SE2Traj createSimpleTrajectoryMsg(const std::vector<Eigen::Vector3d>& path,
                                                             const Eigen::Vector3d& target_state,
                                                             double total_time);

            mpc_controller::SE2Traj createResampledPathMsg(const std::vector<Eigen::Vector3d>& path,
                                                          const Eigen::Vector3d& target_state,
                                                          double total_time);
            mpc_controller::SE2Traj createResampledTrajectoryMsg(const SE2Trajectory& traj);

            // 随机位姿生成相关函数
            void generateRandomPoses(Eigen::Vector3d& start_pose, Eigen::Vector3d& target_pose);
            void planWithRandomPoses();
            void handlePlanningFailure();  // 处理规划失败的函数
            
            // 核心规划逻辑（可被多线程调用）
            bool executePlanning(const Eigen::Vector3d& start_state, const Eigen::Vector3d& end_state,
                               mpc_controller::SE2Traj& traj_msg, int path_id = -1);
            
            // 批量规划处理（非阻塞）
            void processPlanningQueue();

        public:
            void init(ros::NodeHandle& nh, ros::NodeHandle& nh_private);
            void rcvStartPoseCallBack(const geometry_msgs::PoseStampedConstPtr& msg);
            void rcvWpsCallBack(const geometry_msgs::PoseStampedConstPtr& msg);
            void occMapCallback(const nav_msgs::OccupancyGridConstPtr& msg);
            // service callback to trigger startDataGeneration externally
            bool startDataGenerationSrv(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res);

            // 数据生成接口
            void startDataGeneration();
    };
}
