/**
 * @file only_planner_node.cpp
 * @brief 纯路径规划节点主程序
 * @author AI 代码助手 (AI Code Assistant)
 * @date 2025-07-30
 *
 * 该节点负责运行纯路径规划器，不进行运动控制。
 * 接收起始点和目标点位姿，进行路径规划并发布优化后的轨迹。
 */

#include "plan_manager/only_planner.h"
#include <ros/ros.h>
#include <std_srvs/Empty.h>

using namespace uneven_planner;

// 全局规划器指针
OnlyPlanner* g_planner = nullptr;

// 启动数据生成的服务回调
bool startDataGenerationCallback(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res) {
    if (g_planner) {
        g_planner->startDataGeneration();
        return true;
    }
    return false;
}

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "only_planner_node");
    ros::NodeHandle nh;        // 全局命名空间，用于话题重映射
    ros::NodeHandle nh_private("~");  // 私有命名空间，用于参数

    // 创建路径规划器
    OnlyPlanner planner;
    planner.init(nh, nh_private);
    g_planner = &planner;

    // 创建数据生成触发服务
    ros::ServiceServer start_service = nh.advertiseService("start_data_generation", startDataGenerationCallback);

    ROS_INFO("OnlyPlanner node started.");
    ROS_INFO("Call 'rosservice call /only_planner_node/start_data_generation' to start data generation");

    // 运行ROS循环
    ros::spin();

    return 0;
}
