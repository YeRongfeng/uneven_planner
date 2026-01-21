/**
 * @file only_planner.cpp
 * @brief 不平坦地形路径规划管理器类的实现 (Implementation of OnlyPlanner class for uneven terrain path planning)
 * @author AI 代码助手 (AI Code Assistant)
 * @date 2025-07-30
 *
 * 本文件实现了OnlyPlanner类，作为不平坦地形路径规划的主要协调器。
 * 它集成了运动学A*搜索、轨迹优化和ROS通信，用于自主导航，但是不进行运动控制
 *
 * 系统功能包括：
 * - 使用运动学A*算法进行初始路径规划
 * - 针对不平坦表面的平滑运动轨迹优化
 * - 处理里程计和目标航点的ROS消息
 * - 向MPC控制器发布优化后的轨迹
 * 
 * 工作流程：
 * 1. 等待数据集生成节点发布起始点和目标点的位姿
 * 2. 接收到目标点位姿后，进行路径规划
 * 3. 发布优化后的轨迹给到数据集生成节点
 * 4. 等待数据集生成节点发布新的目标点位姿
 * 5. 重复步骤2-4，直到数据集生成节点不再发布目标点位姿
 * 6. 退出
 */

#include "plan_manager/only_planner.h"
#include <std_msgs/Bool.h>
#include <std_srvs/Trigger.h>

namespace uneven_planner
{
    // service callback implementation
    bool OnlyPlanner::startDataGenerationSrv(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res)
    {
        ROS_INFO("Service call /start_data_generation received, starting data generation");
        startDataGeneration();
        res.success = true;
        res.message = "OnlyPlanner: data generation started";
        return true;
    }

    /**
     * @brief 使用ROS参数和组件初始化OnlyPlanner
     * @param nh ROS节点句柄，用于参数获取和通信设置
     *
     * 此方法执行以下初始化步骤：
     * 1. 从ROS参数服务器加载规划参数
     * 2. 创建并初始化核心规划组件（地图、A*、优化器）
     * 3. 设置组件依赖关系和环境引用
     * 4. 建立用于通信的ROS发布者和订阅者
     *
     * @note 在任何规划操作之前必须调用此方法
     */
    void OnlyPlanner::init(ros::NodeHandle& nh, ros::NodeHandle& nh_private)
    {
        ROS_INFO("=== ONLY_PLANNER INIT STARTED ===");

        // 从ROS参数服务器加载规划参数（使用节点私有命名空间，匹配YAML文件结构）
        if (!nh_private.getParam("manager/piece_len", piece_len)) {
            ROS_ERROR("Failed to get manager/piece_len parameter");
            piece_len = 0.3; // 使用YAML文件中的默认值
        }
        if (!nh_private.getParam("manager/mean_vel", mean_vel)) {
            ROS_ERROR("Failed to get manager/mean_vel parameter");
            mean_vel = 0.5; // 使用YAML文件中的默认值
        }
        if (!nh_private.getParam("manager/init_time_times", init_time_times)) {
            ROS_ERROR("Failed to get manager/init_time_times parameter");
            init_time_times = 1.2; // 使用YAML文件中的默认值
        }
        if (!nh_private.getParam("manager/yaw_piece_times", yaw_piece_times)) {
            ROS_ERROR("Failed to get manager/yaw_piece_times parameter");
            yaw_piece_times = 2.0; // 使用YAML文件中的默认值
        }
        if (!nh_private.getParam("manager/init_sig_vel", init_sig_vel)) {
            ROS_ERROR("Failed to get manager/init_sig_vel parameter");
            init_sig_vel = 0.05; // 使用YAML文件中的默认值
        }
        if (!nh_private.getParam("manager/enable_optimization", enable_optimization)) {
            ROS_ERROR("Failed to get manager/enable_optimization parameter");
            enable_optimization = false; // 默认禁用优化（安全起见）
        }
        if (!nh_private.getParam("manager/resample_points", resample_points)) {
            ROS_ERROR("Failed to get manager/resample_points parameter");
            resample_points = 100; // 默认重采样点数
        }
        // 距离约束参数（支持None值）
        use_min_distance_constraint = nh_private.getParam("manager/min_planning_distance", min_planning_distance);
        use_max_distance_constraint = nh_private.getParam("manager/max_planning_distance", max_planning_distance);

        if (!use_min_distance_constraint) {
            min_planning_distance = 0.0;  // 默认值，但不会使用
            ROS_INFO("min_planning_distance not set, no minimum distance constraint");
        }
        if (!use_max_distance_constraint) {
            max_planning_distance = 0.0;  // 默认值，但不会使用
            ROS_INFO("max_planning_distance not set, no maximum distance constraint");
        }

        // 固定位姿参数 - 尝试多种参数路径
        use_start_fixed = nh_private.getParam("manager/start_fixed", start_fixed);
        if (!use_start_fixed) {
            use_start_fixed = nh.getParam("manager/start_fixed", start_fixed);
        }
        if (!use_start_fixed) {
            use_start_fixed = nh.getParam("/manager/start_fixed", start_fixed);
        }

        use_end_fixed = nh_private.getParam("manager/end_fixed", end_fixed);
        if (!use_end_fixed) {
            use_end_fixed = nh.getParam("manager/end_fixed", end_fixed);
        }
        if (!use_end_fixed) {
            use_end_fixed = nh.getParam("/manager/end_fixed", end_fixed);
        }

        ROS_INFO("Parameter loading debug:");
        ROS_INFO("  use_start_fixed: %s", use_start_fixed ? "true" : "false");
        ROS_INFO("  use_end_fixed: %s", use_end_fixed ? "true" : "false");

        // 参数加载调试信息
        if (use_start_fixed) {
            ROS_INFO("Loaded start_fixed: [%.3f, %.3f, %.3f]", start_fixed[0], start_fixed[1], start_fixed[2]);
        }
        if (use_end_fixed) {
            ROS_INFO("Loaded end_fixed: [%.3f, %.3f, %.3f]", end_fixed[0], end_fixed[1], end_fixed[2]);
        }

        if (use_start_fixed && start_fixed.size() != 3) {
            ROS_ERROR("start_fixed must have exactly 3 elements [x, y, yaw]");
            use_start_fixed = false;
        }
        if (use_end_fixed && end_fixed.size() != 3) {
            ROS_ERROR("end_fixed must have exactly 3 elements [x, y, yaw]");
            use_end_fixed = false;
        }
        nh_private.param<string>("manager/bk_dir", bk_dir, "xxx");     // 备份目录路径（默认："xxx"）

        // 初始化重试机制参数
        pose_retry_count = 0;
        max_pose_retries = 5;  // 每个地图最多重试5次起终点

        ROS_INFO("Loaded parameters: piece_len=%.3f, mean_vel=%.3f, init_time_times=%.3f, yaw_piece_times=%.3f, init_sig_vel=%.3f",
                 piece_len, mean_vel, init_time_times, yaw_piece_times, init_sig_vel);
        ROS_INFO("Optimization settings: enable_optimization=%s, resample_points=%d",
                 enable_optimization ? "true" : "false", resample_points);

        // 距离约束信息
        if (use_min_distance_constraint && use_max_distance_constraint) {
            ROS_INFO("Distance constraints: min_distance=%.3f, max_distance=%.3f",
                     min_planning_distance, max_planning_distance);
        } else if (use_min_distance_constraint) {
            ROS_INFO("Distance constraints: min_distance=%.3f, max_distance=None",
                     min_planning_distance);
        } else if (use_max_distance_constraint) {
            ROS_INFO("Distance constraints: min_distance=None, max_distance=%.3f",
                     max_planning_distance);
        } else {
            ROS_INFO("Distance constraints: min_distance=None, max_distance=None");
        }

        // 固定位姿信息
        if (use_start_fixed) {
            ROS_INFO("Start pose constraints: x=%s, y=%s, yaw=%s",
                     (std::isnan(start_fixed[0]) || start_fixed[0] == -999.0) ? "random" : std::to_string(start_fixed[0]).c_str(),
                     (std::isnan(start_fixed[1]) || start_fixed[1] == -999.0) ? "random" : std::to_string(start_fixed[1]).c_str(),
                     (std::isnan(start_fixed[2]) || start_fixed[2] == -999.0) ? "random" : std::to_string(start_fixed[2]).c_str());
        } else {
            ROS_INFO("Start pose constraints: all random");
        }
        if (use_end_fixed) {
            ROS_INFO("End pose constraints: x=%s, y=%s, yaw=%s",
                     (std::isnan(end_fixed[0]) || end_fixed[0] == -999.0) ? "random" : std::to_string(end_fixed[0]).c_str(),
                     (std::isnan(end_fixed[1]) || end_fixed[1] == -999.0) ? "random" : std::to_string(end_fixed[1]).c_str(),
                     (std::isnan(end_fixed[2]) || end_fixed[2] == -999.0) ? "random" : std::to_string(end_fixed[2]).c_str());
        } else {
            ROS_INFO("End pose constraints: all random");
        }

        // 初始化核心规划组件
        uneven_map.reset(new UnevenMap);    // 创建不平坦地形地图处理器
        kino_astar.reset(new KinoAstar);    // 创建运动学A*规划器

        // 调试：检查地图参数是否正确传递
        std::string map_pcd_param, map_file_param;
        if (nh.getParam("uneven_map/map_pcd", map_pcd_param)) {
            ROS_INFO("Map PCD parameter: %s", map_pcd_param.c_str());
        } else {
            ROS_ERROR("Failed to get uneven_map/map_pcd parameter");
        }

        if (nh.getParam("uneven_map/map_file", map_file_param)) {
            ROS_INFO("Map file parameter: %s", map_file_param.c_str());
        } else {
            ROS_ERROR("Failed to get uneven_map/map_file parameter");
        }

        // 调试：检查关键地图参数是否存在
        double xy_res_check, yaw_res_check;
        if (nh.getParam("uneven_map/xy_resolution", xy_res_check)) {
            ROS_INFO("Found xy_resolution parameter: %.6f", xy_res_check);
        } else {
            ROS_ERROR("Failed to find uneven_map/xy_resolution parameter");
        }

        if (nh.getParam("uneven_map/yaw_resolution", yaw_res_check)) {
            ROS_INFO("Found yaw_resolution parameter: %.6f", yaw_res_check);
        } else {
            ROS_ERROR("Failed to find uneven_map/yaw_resolution parameter");
        }

        // 使用ROS参数初始化组件（使用全局命名空间，与成功的plan_manager.cpp一致）
        uneven_map->init(nh);
        kino_astar->init(nh);

        // 设置组件依赖关系
        kino_astar->setEnvironment(uneven_map);    // 为A*规划器提供地图

        // 修复：在初始化轨迹优化器之前验证关键参数是否存在
        // 这是为了防止int_K等参数未正确加载导致的崩溃问题
        // 使用私有命名空间来访问正确的参数路径
        std::vector<std::string> critical_params = {
            "alm_traj_opt/int_K",
            "alm_traj_opt/mem_size",
            "alm_traj_opt/past",
            "alm_traj_opt/max_iter",
            "alm_traj_opt/inner_max_iter"
        };

        bool all_params_loaded = true;
        for (const auto& param : critical_params) {
            // 尝试从私有命名空间加载参数
            if (!nh_private.hasParam(param)) {
                ROS_ERROR("Critical parameter %s not found in private namespace!", param.c_str());
                all_params_loaded = false;
            } else {
                // 验证参数值
                if (param.find("int_K") != std::string::npos) {
                    int int_K_val;
                    nh_private.getParam(param, int_K_val);
                    ROS_INFO("Verified parameter %s = %d", param.c_str(), int_K_val);
                } else if (param.find("mem_size") != std::string::npos) {
                    int mem_size_val;
                    nh_private.getParam(param, mem_size_val);
                    ROS_INFO("Verified parameter %s = %d", param.c_str(), mem_size_val);
                } else if (param.find("past") != std::string::npos) {
                    int past_val;
                    nh_private.getParam(param, past_val);
                    ROS_INFO("Verified parameter %s = %d", param.c_str(), past_val);
                } else if (param.find("max_iter") != std::string::npos) {
                    int max_iter_val;
                    nh_private.getParam(param, max_iter_val);
                    ROS_INFO("Verified parameter %s = %d", param.c_str(), max_iter_val);
                } else if (param.find("inner_max_iter") != std::string::npos) {
                    int inner_max_iter_val;
                    nh_private.getParam(param, inner_max_iter_val);
                    ROS_INFO("Verified parameter %s = %d", param.c_str(), inner_max_iter_val);
                }
            }
        }

        if (!all_params_loaded) {
            ROS_FATAL("Critical parameters missing! Cannot initialize trajectory optimizer safely.");
            ros::shutdown();
            return;
        }

        traj_opt.init(nh_private);                 // 使用私有命名空间初始化轨迹优化器
        traj_opt.setFrontend(kino_astar);          // 将优化器连接到A*前端
        traj_opt.setEnvironment(uneven_map);       // 为优化器提供地图

        // 设置ROS通信
        traj_pub = nh.advertise<mpc_controller::SE2Traj>("optimized_traj", 1);  // 向数据集生成节点发布优化后的轨迹
        success_pub = nh.advertise<std_msgs::Bool>("planning_result", 1);  // 发布规划结果消息
        map_regen_pub = nh.advertise<std_msgs::Bool>("map_regeneration_request", 1);  // 地图重新生成请求发布者
        start_sub = nh.subscribe<geometry_msgs::PoseStamped>("start_pose", 1, &OnlyPlanner::rcvStartPoseCallBack, this);  // 起始点订阅者
        target_sub = nh.subscribe<geometry_msgs::PoseStamped>("target_pose", 1, &OnlyPlanner::rcvWpsCallBack, this);  // 目标订阅者
        // 注册 service 以便从外部触发数据生成（例如 terrain_dataset_generator.py）
        // NOTE: service "start_data_generation" already created in only_planner_node.cpp.
        // To avoid duplicate advertisement in the same process/node, do not register it here.
        // If you want this class to own the service instead, remove the advertise in only_planner_node.cpp.
        // start_data_srv = nh.advertiseService("start_data_generation", &OnlyPlanner::startDataGenerationSrv, this);
        // ROS_INFO("Service advertised: %s/start_data_generation", ros::this_node::getName().c_str());

        ROS_INFO("OnlyPlanner ROS communication setup complete:");
        ROS_INFO("  - Publishing to: %s", traj_pub.getTopic().c_str());
        ROS_INFO("  - Subscribed to start_pose: %s", start_sub.getTopic().c_str());
        ROS_INFO("  - Subscribed to target_pose: %s", target_sub.getTopic().c_str());

        return;
    }

    /**
     * @brief 处理起始点位姿消息的回调函数
     * @param msg 包含起始位置和方向的起始位姿消息
     *
     * 此回调函数接收数据集生成节点发布的起始点位姿，
     * 直接将车辆"传送"到指定位置，更新当前位姿。
     *
     * 转换过程：
     * 1. 直接从消息中提取x, y位置
     * 2. 将四元数方向转换为偏航角
     * 3. 更新车辆当前位姿
     *
     * @note 这实现了车辆的瞬间"传送"功能
     * @note 偏航角以弧度为单位存储在odom_pos(2)中
     */
    void OnlyPlanner::rcvStartPoseCallBack(const geometry_msgs::PoseStampedConstPtr& msg)
    {
        ROS_INFO("=== START POSE CALLBACK TRIGGERED ===");
        ROS_INFO("Received start pose: [%.3f, %.3f, %.3f]",
                 msg->pose.position.x, msg->pose.position.y,
                 atan2(2.0*msg->pose.orientation.z*msg->pose.orientation.w,
                       2.0*pow(msg->pose.orientation.w, 2)-1.0));

        // 从起始位姿消息中提取2D位置
        odom_pos(0) = msg->pose.position.x;
        odom_pos(1) = msg->pose.position.y;

        // 使用与成功的plan_manager.cpp相同的偏航角计算方式
        Eigen::Quaterniond q(msg->pose.orientation.w, \
                             msg->pose.orientation.x, \
                             msg->pose.orientation.y, \
                             msg->pose.orientation.z  );
        Eigen::Matrix3d R(q);
        odom_pos(2) = UnevenMap::calYawFromR(R);

        // ROS_INFO("Vehicle teleported to start pose: [%.3f, %.3f, %.3f]",
        //          odom_pos(0), odom_pos(1), odom_pos(2));
    }

    /**
     * @brief 处理目标航点消息的回调函数
     * @param msg 包含目标位置和方向的目标位姿消息
     *
     * 这是主要的规划回调函数，协调完整的规划流水线：
     * 1. 验证规划前置条件（未在规划中，地图已就绪）
     * 2. 将目标位姿转换为内部表示
     * 3. 执行初始运动学A*搜索
     * 4. 平滑偏航角以避免不连续性
     * 5. 生成用于优化的初始轨迹解
     * 6. 使用SE(2)轨迹优化进行轨迹优化
     * 7. 向MPC控制器发布优化后的轨迹
     *
     * @note 此函数设置in_plan标志以防止并发规划请求
     * @note 如果地图未就绪或正在进行其他规划，则跳过规划
     * @note 使用四元数到偏航角转换公式：atan2(2*qz*qw, 2*qw^2-1)
     */
    void OnlyPlanner::rcvWpsCallBack(const geometry_msgs::PoseStampedConstPtr& msg)
    {
        // ROS_INFO("=== TARGET POSE CALLBACK TRIGGERED ===");
        // ROS_INFO("Received target pose: [%.3f, %.3f, %.3f]",
        //          msg->pose.position.x, msg->pose.position.y,
        //          atan2(2.0*msg->pose.orientation.z*msg->pose.orientation.w,
        //                2.0*pow(msg->pose.orientation.w, 2)-1.0));

        // ROS_INFO("Current vehicle position: [%.3f, %.3f, %.3f]",
        //          odom_pos(0), odom_pos(1), odom_pos(2));

        // ROS_INFO("Checking preconditions: in_plan=%s, map_ready=%s",
        //          in_plan ? "true" : "false",
        //          uneven_map->mapReady() ? "true" : "false");

        // 检查规划前置条件：当前未在规划且地图已就绪
        if (in_plan || !uneven_map->mapReady()) {
            if (in_plan) {
                ROS_WARN("Already in planning, skipping target pose");
            }
            if (!uneven_map->mapReady()) {
                ROS_WARN("Map not ready, skipping target pose");
            }
            return;
        }

        // ROS_INFO("All preconditions met, starting path planning...");

        in_plan = true;  // 设置规划标志以防止并发请求

        // 定义地图边界常量
        const double MAP_BOUNDARY = 19.9999;  // 与地图边界一致

        // 将目标位姿从四元数转换为SE(2)表示 (x, y, yaw)
        Eigen::Vector3d end_state(msg->pose.position.x, \
                                  msg->pose.position.y, \
                                  atan2(2.0*msg->pose.orientation.z*msg->pose.orientation.w, \
                                        2.0*pow(msg->pose.orientation.w, 2)-1.0)             );

        // 在A*规划前检查起始点和目标点是否在地图边界内

        // 检查起始点边界
        if (odom_pos(0) < -MAP_BOUNDARY || odom_pos(0) > MAP_BOUNDARY ||
            odom_pos(1) < -MAP_BOUNDARY || odom_pos(1) > MAP_BOUNDARY) {
            ROS_WARN("Start position out of bounds: [%.3f, %.3f], retrying with new poses",
                     odom_pos(0), odom_pos(1));
            in_plan = false;
            std_msgs::Bool failure_msg;
            failure_msg.data = false;
            success_pub.publish(failure_msg);
            planWithRandomPoses();
            return;
        }

        // 检查目标点边界
        if (end_state(0) < -MAP_BOUNDARY || end_state(0) > MAP_BOUNDARY ||
            end_state(1) < -MAP_BOUNDARY || end_state(1) > MAP_BOUNDARY) {
            ROS_WARN("Target position out of bounds: [%.3f, %.3f], retrying with new poses",
                     end_state(0), end_state(1));
            in_plan = false;
            std_msgs::Bool failure_msg;
            failure_msg.data = false;
            success_pub.publish(failure_msg);
            
            planWithRandomPoses();
            return;
        }

        // 检查起始点和目标点之间的距离是否在允许范围内
        double distance = sqrt(pow(end_state(0) - odom_pos(0), 2) + pow(end_state(1) - odom_pos(1), 2));

        // 检查最短距离约束（如果启用）
        if (use_min_distance_constraint && distance < min_planning_distance) {
            ROS_WARN("Distance %.3f is too short (min: %.3f), retrying with new poses",
                     distance, min_planning_distance);
            in_plan = false;
            std_msgs::Bool failure_msg;
            failure_msg.data = false;
            success_pub.publish(failure_msg);
            planWithRandomPoses();
            return;
        }

        // 检查最远距离约束（如果启用）
        if (use_max_distance_constraint && distance > max_planning_distance) {
            ROS_WARN("Distance %.3f is too far (max: %.3f), retrying with new poses",
                     distance, max_planning_distance);
            in_plan = false;
            std_msgs::Bool failure_msg;
            failure_msg.data = false;
            success_pub.publish(failure_msg);
            planWithRandomPoses();
            return;
        }

        // 调试信息
        ROS_INFO("Planning from [%.3f, %.3f, %.3f] to [%.3f, %.3f, %.3f] (boundary and distance check passed, distance=%.3f)",
                 odom_pos(0), odom_pos(1), odom_pos(2),
                 end_state(0), end_state(1), end_state(2), distance);

        // 添加保护性检查，避免崩溃
        if (!kino_astar) {
            ROS_ERROR("kino_astar is null!");
            std_msgs::Bool failure_msg;
            failure_msg.data = false;
            success_pub.publish(failure_msg);
            in_plan = false;
            return;
        }

        if (!uneven_map) {
            ROS_ERROR("uneven_map is null!");
            std_msgs::Bool failure_msg;
            failure_msg.data = false;
            success_pub.publish(failure_msg);
            in_plan = false;
            return;
        }

        // // 验证输入参数
        // ROS_INFO("Input validation:");
        // ROS_INFO("  Start state: [%.6f, %.6f, %.6f]", odom_pos(0), odom_pos(1), odom_pos(2));
        // ROS_INFO("  End state: [%.6f, %.6f, %.6f]", end_state(0), end_state(1), end_state(2));

        // 检查输入是否包含NaN或无穷大
        if (!std::isfinite(odom_pos(0)) || !std::isfinite(odom_pos(1)) || !std::isfinite(odom_pos(2))) {
            ROS_ERROR("Start state contains invalid values (NaN or Inf)");
            std_msgs::Bool failure_msg;
            failure_msg.data = false;
            success_pub.publish(failure_msg);
            in_plan = false;
            return;
        }

        if (!std::isfinite(end_state(0)) || !std::isfinite(end_state(1)) || !std::isfinite(end_state(2))) {
            ROS_ERROR("End state contains invalid values (NaN or Inf)");
            std_msgs::Bool failure_msg;
            failure_msg.data = false;
            success_pub.publish(failure_msg);
            in_plan = false;
            return;
        }

        // // 测试 kino_astar 对象的基本功能
        // ROS_INFO("Testing kino_astar object...");

        // // 首先检查地图参数是否正确初始化
        // ROS_INFO("Map parameters check:");
        // ROS_INFO("  Map ready: %s", uneven_map->mapReady() ? "true" : "false");

        // 手动计算索引来验证转换逻辑（使用更新的分辨率）
        // double xy_res = 0.05, yaw_res = 0.314;
        // double map_origin_x = -5.0, map_origin_y = -5.0, map_origin_yaw = -M_PI;
        double xy_res = 0.4, yaw_res = 0.314;
        double map_origin_x = -20.0, map_origin_y = -20.0, map_origin_yaw = -M_PI;

        int manual_start_x = floor((odom_pos(0) - map_origin_x) / xy_res);
        int manual_start_y = floor((odom_pos(1) - map_origin_y) / xy_res);
        int manual_start_yaw = floor((odom_pos(2) - map_origin_yaw) / yaw_res);

        int manual_end_x = floor((end_state(0) - map_origin_x) / xy_res);
        int manual_end_y = floor((end_state(1) - map_origin_y) / xy_res);
        int manual_end_yaw = floor((end_state(2) - map_origin_yaw) / yaw_res);

        // ROS_INFO("Manual index calculation:");
        // ROS_INFO("  Start manual: [%d, %d, %d]", manual_start_x, manual_start_y, manual_start_yaw);
        // ROS_INFO("  End manual: [%d, %d, %d]", manual_end_x, manual_end_y, manual_end_yaw);

        // 测试状态到索引的转换
        Eigen::Vector3i start_idx, end_idx;
        try {
            kino_astar->stateToIndex(odom_pos, start_idx);
            kino_astar->stateToIndex(end_state, end_idx);
            // ROS_INFO("State to index conversion completed:");
            // ROS_INFO("  Start index: [%d, %d, %d]", start_idx(0), start_idx(1), start_idx(2));
            // ROS_INFO("  End index: [%d, %d, %d]", end_idx(0), end_idx(1), end_idx(2));

            // 偏航角索引
            start_idx(2) = manual_start_yaw;
            end_idx(2) = manual_end_yaw;
            // ROS_INFO("Corrected indices:");
            // ROS_INFO("  Start corrected: [%d, %d, %d]", start_idx(0), start_idx(1), start_idx(2));
            // ROS_INFO("  End corrected: [%d, %d, %d]", end_idx(0), end_idx(1), end_idx(2));

            // 检查索引是否有效
            if (start_idx(0) == INT_MIN || start_idx(1) == INT_MIN ||
                end_idx(0) == INT_MIN || end_idx(1) == INT_MIN) {
                ROS_ERROR("Invalid index values detected (INT_MIN). Map parameters may not be initialized correctly.");
                std_msgs::Bool failure_msg;
                failure_msg.data = false;
                success_pub.publish(failure_msg);
                in_plan = false;
                return;
            }
        } catch (...) {
            ROS_ERROR("Failed in stateToIndex conversion");
            std_msgs::Bool failure_msg;
            failure_msg.data = false;
            success_pub.publish(failure_msg);
            in_plan = false;
            return;
        }

        ROS_INFO("About to call kino_astar->plan()...");

        // 尝试 A* 路径规划
        std::vector<Eigen::Vector3d> init_path;
        try {
            init_path = kino_astar->plan(odom_pos, end_state);
            // ROS_INFO("A* planning successful, path size: %zu", init_path.size());

            // // 调试：打印A*路径的前几个和后几个点
            // if (!init_path.empty()) {
            //     ROS_INFO("A* path first point: [%.3f, %.3f, %.3f]",
            //              init_path[0].x(), init_path[0].y(), init_path[0].z());
            //     if (init_path.size() > 1) {
            //         ROS_INFO("A* path second point: [%.3f, %.3f, %.3f]",
            //                  init_path[1].x(), init_path[1].y(), init_path[1].z());
            //     }
            //     ROS_INFO("A* path last point: [%.3f, %.3f, %.3f]",
            //              init_path.back().x(), init_path.back().y(), init_path.back().z());
            //     ROS_INFO("Expected start: [%.3f, %.3f, %.3f], Expected end: [%.3f, %.3f, %.3f]",
            //              odom_pos.x(), odom_pos.y(), odom_pos.z(),
            //              end_state.x(), end_state.y(), end_state.z());
            // }
        } catch (const std::exception& e) {
            ROS_WARN("A* planning failed: %s, using fallback straight-line path", e.what());
            // // 生成备用的直线路径
            // int num_points = 10;
            // for (int i = 0; i <= num_points; i++) {
            //     double t = double(i) / double(num_points);
            //     Eigen::Vector3d point;
            //     point(0) = odom_pos(0) + t * (end_state(0) - odom_pos(0));
            //     point(1) = odom_pos(1) + t * (end_state(1) - odom_pos(1));
            //     point(2) = odom_pos(2) + t * (end_state(2) - odom_pos(2));
            //     init_path.push_back(point);
            // }

            handlePlanningFailure();
            return;
        }

        if (init_path.empty())
        {
            ROS_WARN("A* planning failed: no path found, retrying with new poses");
            handlePlanningFailure();
            return;
        }

        // 修复：检查A*生成的轨迹是否有超出地图边界的点
        // 如果有超出边界的点，标记为失败并重新生成
        bool path_out_of_bounds = false;

        // ROS_INFO("Checking A* path boundary compliance...");
        for (size_t i = 0; i < init_path.size(); i++) {
            if (init_path[i](0) < -MAP_BOUNDARY || init_path[i](0) > MAP_BOUNDARY ||
                init_path[i](1) < -MAP_BOUNDARY || init_path[i](1) > MAP_BOUNDARY) {
                ROS_WARN("A* path point %zu out of bounds: [%.3f, %.3f], path rejected",
                         i, init_path[i](0), init_path[i](1));
                path_out_of_bounds = true;
                break;  // 发现一个越界点就足够了
            }
        }

        if (path_out_of_bounds) {
            ROS_WARN("A* generated path contains out-of-bounds points, retrying with new poses");
            in_plan = false;
            // 发布规划失败信号
            std_msgs::Bool failure_msg;
            failure_msg.data = false;
            success_pub.publish(failure_msg);
            // 立即重新生成位姿并重试
            planWithRandomPoses();
            return;
        }

        ROS_INFO("A* path boundary check passed, all %zu points within map bounds", init_path.size());

        // 平滑偏航角以避免路径段之间的大不连续性
        // 这确保偏航角差异保持在[-π/2, π/2]范围内
        double dyaw;
        for (size_t i=0; i<init_path.size()-1; i++)
        {
            dyaw = init_path[i+1].z() - init_path[i].z();

            // 如果偏航角差异太大(>= π/2)则减小
            while (dyaw >= M_PI / 2)
            {
                init_path[i+1].z() -= M_PI * 2;  // 减去2π以保持等效角度
                dyaw = init_path[i+1].z() - init_path[i].z();
            }

            // 如果偏航角差异太小(<= -π/2)则增大
            while (dyaw <= -M_PI / 2)
            {
                init_path[i+1].z() += M_PI * 2;  // 加上2π以保持等效角度
                dyaw = init_path[i+1].z() - init_path[i].z();
            }
        }

        // 将精确的目标点追加到A*路径末尾
        init_path.push_back(end_state);
        // ROS_INFO("Appended exact target point to A* path: [%.3f, %.3f, %.3f], new path size: %zu",
        //          end_state.x(), end_state.y(), end_state.z(), init_path.size());

        // 初始化轨迹边界条件和中间航点
        // 这些矩阵存储位置、速度和加速度约束
        Eigen::Matrix<double, 2, 3> init_xy, end_xy;  // 起始/结束：x,y的[位置, 速度, 加速度]
        Eigen::Vector3d init_yaw, end_yaw;            // 起始/结束：[偏航角, 偏航角速度, 偏航角加速度]
        Eigen::MatrixXd inner_xy;                     // 中间位置航点
        Eigen::VectorXd inner_yaw;                    // 中间偏航角航点
        double total_time;                            // 总轨迹持续时间

        // 设置边界条件：起始点和结束点都使用A*路径的端点（现在包含了精确目标点）
        init_xy << init_path[0].x(), 0.0, 0.0, \
                   init_path[0].y(), 0.0, 0.0;
        end_xy << init_path.back().x(), 0.0, 0.0, \
                  init_path.back().y(), 0.0, 0.0;
        init_yaw << init_path[0].z(), 0.0, 0.0;
        end_yaw << init_path.back().z(), 0.0, 0.0;

        // 基于航向方向设置初始和最终速度
        // 速度大小为init_sig_vel，方向跟随偏航角
        init_xy.col(1) << init_sig_vel * cos(init_yaw(0)), init_sig_vel * sin(init_yaw(0));
        end_xy.col(1) << init_sig_vel * cos(end_yaw(0)), init_sig_vel * sin(end_yaw(0));

        // 将初始路径离散化为中间航点以进行轨迹优化
        // 修复：让位置和偏航角航点数量保持一致，避免轨迹优化中的数据不匹配
        double temp_len_yaw = 0.0;                           // 偏航角离散化的累积长度
        double temp_len_pos = 0.0;                           // 位置离散化的累积长度
        double total_len = 0.0;                              // 总路径长度
        double piece_len_yaw = piece_len / yaw_piece_times;  // 修复：与原版一致
        std::vector<Eigen::Vector2d> inner_xy_node;          // 中间位置航点
        std::vector<double> inner_yaw_node;                  // 中间偏航角航点

        // ROS_INFO("Using consistent waypoint spacing: piece_len=%.3f, piece_len_yaw=%.3f",
        //          piece_len, piece_len_yaw);

        // 遍历初始路径的每个段
        for (int k=0; k<init_path.size()-1; k++)
        {
            // 计算当前路径段的长度
            double temp_seg = (init_path[k+1] - init_path[k]).head(2).norm();
            temp_len_yaw += temp_seg;
            temp_len_pos += temp_seg;
            total_len += temp_seg;

            // 当累积长度超过偏航角片段长度时添加偏航角航点
            while (temp_len_yaw > piece_len_yaw)
            {
                // 在航点位置插值偏航角
                double temp_yaw = init_path[k].z() + (1.0 - (temp_len_yaw-piece_len_yaw) / temp_seg) * (init_path[k+1] - init_path[k]).z();
                inner_yaw_node.push_back(temp_yaw);
                temp_len_yaw -= piece_len_yaw;  // 重置累积长度
            }

            // 当累积长度超过位置片段长度时添加位置航点
            while (temp_len_pos > piece_len)
            {
                // 在航点位置插值位置
                Eigen::Vector3d temp_node = init_path[k] + (1.0 - (temp_len_pos-piece_len) / temp_seg) * (init_path[k+1] - init_path[k]);
                inner_xy_node.push_back(temp_node.head(2));
                // 注意：位置航点的偏航角在上面单独处理
                temp_len_pos -= piece_len;  // 重置累积长度
            }
        }

        // 基于路径长度和期望速度计算总轨迹时间
        total_time = total_len / mean_vel * init_time_times;

        // 检查航点数量，如果过多则直接标记为失败
        const size_t SAFE_WAYPOINT_LIMIT = 160;
        if (inner_xy_node.size() > SAFE_WAYPOINT_LIMIT) {
            ROS_WARN("Too many waypoints generated (xy:%zu), retrying with new poses",
                     inner_xy_node.size());
            handlePlanningFailure();
            return;
        }

        // 调试信息：检查最终的航点数量
        ROS_INFO("Final waypoints: xy_nodes=%zu, yaw_nodes=%zu, total_len=%.3f",
                 inner_xy_node.size(), inner_yaw_node.size(), total_len);

        // 检查航点向量是否为空，如果为空则添加保护措施
        if (inner_xy_node.empty() || inner_yaw_node.empty()) {
            ROS_WARN("No intermediate waypoints generated, adding minimal waypoints for optimization");

            // 添加路径中点作为航点
            if (init_path.size() >= 2) {
                size_t mid_idx = init_path.size() / 2;
                inner_xy_node.push_back(init_path[mid_idx].head(2));
                inner_yaw_node.push_back(init_path[mid_idx](2));
            }
        }

        // 将航点向量转换为Eigen矩阵以进行优化（添加调试信息）
        // ROS_INFO("Setting up matrices: inner_xy_node.size()=%zu, inner_yaw_node.size()=%zu",
        //          inner_xy_node.size(), inner_yaw_node.size());

        if (inner_xy_node.empty() || inner_yaw_node.empty()) {
            ROS_WARN("Empty waypoint vectors detected! xy_size=%zu, yaw_size=%zu, retrying with new poses",
                     inner_xy_node.size(), inner_yaw_node.size());
            handlePlanningFailure();
            return;
        }

        inner_xy.resize(2, inner_xy_node.size());
        inner_yaw.resize(inner_yaw_node.size());

        // ROS_INFO("Matrix resize completed: inner_xy=[2,%zu], inner_yaw=[%zu]",
        //          inner_xy_node.size(), inner_yaw_node.size());

        for (int i=0; i<inner_xy_node.size(); i++)
        {
            inner_xy.col(i) = inner_xy_node[i];
        }
        for (int i=0; i<inner_yaw_node.size(); i++)
        {
            inner_yaw(i) = inner_yaw_node[i];
        }

        // ROS_INFO("Matrix data assignment completed");

        // 根据参数决定是否执行轨迹优化
        if (enable_optimization) {
            // ROS_INFO("Starting trajectory optimization with %zu xy waypoints and %zu yaw waypoints",
            //          inner_xy_node.size(), inner_yaw_node.size());

            try {
                // 使用边界条件和航点执行SE(2)轨迹优化
                int opt_result = traj_opt.optimizeSE2Traj(init_xy, end_xy, inner_xy, \
                                init_yaw, end_yaw, inner_yaw, total_time);
                
                // 检查优化结果
                if (opt_result != 0) {
                    if (opt_result == 1) {
                        ROS_WARN("Trajectory optimization failed due to solver error (NaN/Inf), retrying with new poses");
                    } else if (opt_result == 2) {
                        ROS_WARN("Trajectory optimization failed due to max iterations reached, retrying with new poses");
                    } else {
                        ROS_WARN("Trajectory optimization failed with unknown error code %d, retrying with new poses", opt_result);
                    }
                    handlePlanningFailure();
                    return;
                }
                
                ROS_INFO("Trajectory optimization completed successfully");

                // 检查优化后的轨迹是否有效
                if (traj_opt.getTraj().pos_traj.getPieceNum() == 0 || traj_opt.getTraj().yaw_traj.getPieceNum() == 0) {
                    ROS_WARN("Optimized trajectory is empty, retrying with new poses");
                    handlePlanningFailure();
                    return;
                }

                // 检查优化后的轨迹是否在地图边界内（逐点判断）
                const SE2Trajectory& opt_traj = traj_opt.getTraj();
                bool traj_out_of_bounds = false;
                double total_duration = opt_traj.pos_traj.getTotalDuration();
                int check_points = opt_traj.pos_traj.getPieceNum();
                for (int i = 0; i < check_points; ++i) {
                    // 使用operator[]访问轨迹段，然后获取位置
                    Eigen::Vector2d pos2d = opt_traj.pos_traj[i].getValue(0.0);
                    Eigen::Vector3d pos(pos2d[0], pos2d[1], 0.0);
                    if (pos(0) < -MAP_BOUNDARY || pos(0) > MAP_BOUNDARY ||
                        pos(1) < -MAP_BOUNDARY || pos(1) > MAP_BOUNDARY) {
                        ROS_WARN("Optimized trajectory point %d out of bounds: [%.3f, %.3f], path rejected",
                                 i, pos(0), pos(1));
                        traj_out_of_bounds = true;
                        break; // 发现一个越界点就足够了
                    }
                }
                if (traj_out_of_bounds) {
                    handlePlanningFailure();
                    ROS_WARN("Optimized trajectory contains out-of-bounds points, retrying with new poses");
                    return;
                }

            } catch (const std::exception& e) {
                ROS_WARN("Trajectory optimization failed: %s, retrying with new poses", e.what());
                handlePlanningFailure();
                return;
            }
        } else {
            ROS_WARN("Trajectory optimization disabled, using original trajectory");
        }

        mpc_controller::SE2Traj traj_msg;

        if (enable_optimization) {
            // 获取优化后的轨迹
            SE2Trajectory back_end_traj = traj_opt.getTraj();

            // 可视化轨迹
            traj_opt.visSE2Traj(back_end_traj);
            traj_opt.visSE3Traj(back_end_traj);

            // // 显示轨迹质量指标
            // std::vector<double> max_terrain_value = traj_opt.getMaxVxAxAyCurAttSig(back_end_traj);
            // std::cout << "equal error: "<< back_end_traj.getNonHolError() << std::endl;
            // std::cout << "max vx rate: "<< max_terrain_value[0] << std::endl;
            // std::cout << "max ax rate: "<< max_terrain_value[1] << std::endl;
            // std::cout << "max ay rate: "<< max_terrain_value[2] << std::endl;
            // std::cout << "max cur:     "<< max_terrain_value[3] << std::endl;
            // std::cout << "min cosxi:   "<< -max_terrain_value[4] << std::endl;
            // std::cout << "max sigma:   "<< max_terrain_value[5] << std::endl;

            // 检查轨迹是否存在空间重叠（全局遍历），如果有则舍弃重新规划
            bool has_overlap = false;
            double overlap_threshold = 0.03; // 0.03米作为重叠阈值
            double total_duration = back_end_traj.getTotalDuration();
            double sample_interval = 0.1; // 采样间隔（0.1秒）

            // 收集所有轨迹点
            std::vector<Eigen::Vector2d> trajectory_points;
            for (double t = 0.0; t <= total_duration; t += sample_interval) {
                trajectory_points.push_back(back_end_traj.getPos(t));
            }
            
            // 全局检查是否有重叠（除了相邻点）
            for (size_t i = 0; i < trajectory_points.size() && !has_overlap; i++) {
                for (size_t j = i + 3; j < trajectory_points.size(); j++) { // 跳过相邻点，从i+3开始检查
                    double distance = (trajectory_points[i] - trajectory_points[j]).norm();
                    if (distance < overlap_threshold) {
                        has_overlap = true;
                        double t_i = i * sample_interval;
                        double t_j = j * sample_interval;
                        ROS_WARN("Detected trajectory overlap: t1=%.3f, pos1=[%.3f, %.3f], t2=%.3f, pos2=[%.3f, %.3f], distance=%.3f",
                                 t_i, trajectory_points[i].x(), trajectory_points[i].y(),
                                 t_j, trajectory_points[j].x(), trajectory_points[j].y(), distance);
                        break;
                    }
                }
            }

            if (has_overlap) {
                ROS_WARN("Optimized trajectory contains spatial overlap, discarding and replanning with new poses");
                handlePlanningFailure();
                return;
            }
            
            // ROS_INFO("Trajectory overlap check passed, no spatial overlap detected");

            // 采样优化后的轨迹（固定100个采样点）
            ROS_INFO("Sampling optimized trajectory to 100 points");
            traj_msg = createResampledTrajectoryMsg(back_end_traj);
        } else {
            // 轨迹优化被禁用，直接发送A*路径原始点
            ROS_WARN("Trajectory optimization disabled, sending raw A* path");

            // 直接重采样A*路径到固定数量的点
            traj_msg = createResampledPathMsg(init_path, end_state, total_time);
        }

        // 检查轨迹是否有效（非空）
        if (traj_msg.pos_pts.empty() || traj_msg.angle_pts.empty()) {
            ROS_WARN("Generated trajectory is empty, planning failed");
            handlePlanningFailure();
            return;
        }

        // 检测轨迹是否在地图边界内
        bool traj_out_of_bounds = false;
        for (const auto& pos : traj_msg.pos_pts) {
            if (pos.x < -MAP_BOUNDARY || pos.x > MAP_BOUNDARY ||
                pos.y < -MAP_BOUNDARY || pos.y > MAP_BOUNDARY) {
                ROS_WARN("Trajectory point out of bounds: [%.3f, %.3f], path rejected",
                         pos.x, pos.y);
                traj_out_of_bounds = true;
                break; // 发现一个越界点就足够了
            }
        }
        if (traj_out_of_bounds) {
            ROS_WARN("Generated trajectory contains out-of-bounds points, planning failed");
            handlePlanningFailure();
            return;
        }

        // 发布轨迹消息
        traj_pub.publish(traj_msg);

        // 发布规划成功信号
        std_msgs::Bool success_msg;
        success_msg.data = true;
        success_pub.publish(success_msg);

        // 重置重试计数器（成功规划）
        pose_retry_count = 0;
        map_retry_count = 0;

        // ROS_INFO("Published trajectory with %zu position points and %zu angle points",
        //          traj_msg.pos_pts.size(), traj_msg.angle_pts.size());
        ROS_INFO("Path planning completed successfully");

        in_plan = false;  // 重置规划标志以允许新的规划请求

        return;
    }

    // 随机位姿生成函数
    void OnlyPlanner::generateRandomPoses(Eigen::Vector3d& start_pose, Eigen::Vector3d& target_pose) {
        // 地图边界参数（与launch文件一致）
        const double MAP_X_MIN = -17.6;
        const double MAP_X_MAX = 17.6;
        const double MAP_Y_MIN = -17.6;
        const double MAP_Y_MAX = 17.6;
        const double MIN_DISTANCE = 2.0;

        // 随机数生成器
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<double> x_dist(MAP_X_MIN, MAP_X_MAX);
        static std::uniform_real_distribution<double> y_dist(MAP_Y_MIN, MAP_Y_MAX);
        static std::uniform_real_distribution<double> yaw_dist(-M_PI, M_PI);

        int max_attempts = 100;

        for (int attempt = 0; attempt < max_attempts; ++attempt) {
            // 生成起始位姿（考虑固定值）
            if (use_start_fixed) {
                // 使用-999.0作为"不固定"的标记值，兼容YAML解析
                start_pose.x() = (std::isnan(start_fixed[0]) || start_fixed[0] == -999.0) ? x_dist(gen) : start_fixed[0];
                start_pose.y() = (std::isnan(start_fixed[1]) || start_fixed[1] == -999.0) ? y_dist(gen) : start_fixed[1];
                start_pose.z() = (std::isnan(start_fixed[2]) || start_fixed[2] == -999.0) ? yaw_dist(gen) : start_fixed[2];

                if (start_fixed[0] != -999.0 && !std::isnan(start_fixed[0])) {
                    ROS_INFO("Using fixed start x: %.3f", start_pose.x());
                }
                if (start_fixed[1] != -999.0 && !std::isnan(start_fixed[1])) {
                    ROS_INFO("Using fixed start y: %.3f", start_pose.y());
                }
                if (start_fixed[2] != -999.0 && !std::isnan(start_fixed[2])) {
                    ROS_INFO("Using fixed start yaw: %.3f", start_pose.z());
                }
            } else {
                start_pose.x() = x_dist(gen);
                start_pose.y() = y_dist(gen);
                start_pose.z() = yaw_dist(gen);
            }

            // 生成目标位姿（考虑固定值）
            if (use_end_fixed) {
                target_pose.x() = (std::isnan(end_fixed[0]) || end_fixed[0] == -999.0) ? x_dist(gen) : end_fixed[0];
                target_pose.y() = (std::isnan(end_fixed[1]) || end_fixed[1] == -999.0) ? y_dist(gen) : end_fixed[1];
                target_pose.z() = (std::isnan(end_fixed[2]) || end_fixed[2] == -999.0) ? yaw_dist(gen) : end_fixed[2];

                if (end_fixed[0] != -999.0 && !std::isnan(end_fixed[0])) {
                    ROS_INFO("Using fixed end x: %.3f", target_pose.x());
                }
                if (end_fixed[1] != -999.0 && !std::isnan(end_fixed[1])) {
                    ROS_INFO("Using fixed end y: %.3f", target_pose.y());
                }
                if (end_fixed[2] != -999.0 && !std::isnan(end_fixed[2])) {
                    ROS_INFO("Using fixed end yaw: %.3f", target_pose.z());
                }
            } else {
                target_pose.x() = x_dist(gen);
                target_pose.y() = y_dist(gen);
                target_pose.z() = yaw_dist(gen);
            }

            // 检查距离约束
            double distance = (target_pose.head(2) - start_pose.head(2)).norm();
            bool distance_ok = true;
            if (use_min_distance_constraint && distance < min_planning_distance) distance_ok = false;
            if (use_max_distance_constraint && distance > max_planning_distance) distance_ok = false;
            if (!use_min_distance_constraint && !use_max_distance_constraint && distance < MIN_DISTANCE) distance_ok = false;

            if (!distance_ok) {
                continue; // 尝试下一个随机对
            }

            // 如果地图就绪，则检查occupancy（使用UnevenMap提供的接口）
            bool start_free = true;
            bool target_free = true;
            if (uneven_map && uneven_map->mapReady()) {
                // isInMap + isOccupancyXY: 返回 0 表示 free, 100 表示 occupied, -1 表示不在地图内
                if (!uneven_map->isInMap(start_pose)) {
                    start_free = false;
                } else {
                    int occ = uneven_map->isOccupancyXY(start_pose);
                    if (occ != 0) start_free = false;
                }

                if (!uneven_map->isInMap(target_pose)) {
                    target_free = false;
                } else {
                    int occ = uneven_map->isOccupancyXY(target_pose);
                    if (occ != 0) target_free = false;
                }
            }

            if (start_free && target_free) {
                ROS_INFO("Generated free pose pair (attempt %d): Start=[%.3f, %.3f, %.3f], Target=[%.3f, %.3f, %.3f], Distance=%.3f",
                         attempt + 1, start_pose.x(), start_pose.y(), start_pose.z(),
                         target_pose.x(), target_pose.y(), target_pose.z(), distance);
                return;
            }
            // 否则继续尝试
        }

        ROS_WARN("Failed to generate valid free pose pair after %d attempts, using last generated poses", max_attempts);
    }

    // 自动规划函数：生成随机位姿并进行路径规划
    void OnlyPlanner::planWithRandomPoses() {
        if (in_plan) {
            return;  // 如果正在规划中，跳过
        }

        Eigen::Vector3d start_pose, target_pose;
        generateRandomPoses(start_pose, target_pose);

        // 设置车辆位置为起始位姿
        odom_pos = start_pose;

        // 创建目标位姿消息
        geometry_msgs::PoseStamped target_msg;
        target_msg.header.stamp = ros::Time::now();
        target_msg.header.frame_id = "world";
        target_msg.pose.position.x = target_pose.x();
        target_msg.pose.position.y = target_pose.y();
        target_msg.pose.position.z = 0.0;

        // 将偏航角转换为四元数
        double yaw = target_pose.z();
        target_msg.pose.orientation.w = cos(yaw / 2.0);
        target_msg.pose.orientation.x = 0.0;
        target_msg.pose.orientation.y = 0.0;
        target_msg.pose.orientation.z = sin(yaw / 2.0);

        // 直接调用路径规划回调函数
        rcvWpsCallBack(boost::make_shared<geometry_msgs::PoseStamped>(target_msg));
    }

    // 处理规划失败
    void OnlyPlanner::handlePlanningFailure() {
        // 重置规划标志，允许新的规划请求
        in_plan = false;
        
        // 发布规划失败信号
        std_msgs::Bool failure_msg;
        failure_msg.data = false;
        success_pub.publish(failure_msg);
        
        // 增加起终点重试计数
        pose_retry_count++;
        ROS_WARN("Planning failed, retry count: %d/%d", pose_retry_count, max_pose_retries);

        // // 检查是否需要重新生成地图
        // if (pose_retry_count >= max_pose_retries) {
        //     ROS_WARN("Pose retry count exceeded (%d/%d), attempting map regeneration",
        //              pose_retry_count, max_pose_retries);
            
        //     // 发布地图重新生成请求
        //     std_msgs::Bool regen_msg;
        //     regen_msg.data = true;
        //     map_regen_pub.publish(regen_msg);
            
        //     // 重置位姿重试计数
        //     pose_retry_count = 0;
        //     return;
        // }

        // 继续重试新的起终点
        planWithRandomPoses();
    }

    // 启动数据生成过程
    void OnlyPlanner::startDataGeneration() {
        ROS_INFO("Starting automatic data generation with internal pose generation");
        planWithRandomPoses();
    }

    // 创建标准轨迹消息
    mpc_controller::SE2Traj OnlyPlanner::createTrajectoryMsg(const SE2Trajectory& traj) {
        mpc_controller::SE2Traj traj_msg;
        traj_msg.start_time = ros::Time::now();
        traj_msg.init_v.x = 0.0;
        traj_msg.init_v.y = 0.0;
        traj_msg.init_v.z = 0.0;
        traj_msg.init_a.x = 0.0;
        traj_msg.init_a.y = 0.0;
        traj_msg.init_a.z = 0.0;

        // 添加位置轨迹段
        for (int i = 0; i < traj.pos_traj.getPieceNum(); i++) {
            geometry_msgs::Point pospt;
            Eigen::Vector2d pos = traj.pos_traj[i].getValue(0.0);
            pospt.x = pos[0];
            pospt.y = pos[1];
            traj_msg.pos_pts.push_back(pospt);
            traj_msg.posT_pts.push_back(traj.pos_traj[i].getDuration());
        }
        // 添加最终位置点
        geometry_msgs::Point pospt;
        Eigen::Vector2d pos = traj.pos_traj.getValue(traj.pos_traj.getTotalDuration());
        pospt.x = pos[0];
        pospt.y = pos[1];
        traj_msg.pos_pts.push_back(pospt);

        // 添加偏航角轨迹段
        for(int i = 0; i < traj.yaw_traj.getPieceNum(); i ++){
            geometry_msgs::Point anglept;
            Eigen::VectorXd angle = traj.yaw_traj[i].getValue(0.0);
            anglept.x = angle[0];
            traj_msg.angle_pts.push_back(anglept);
            traj_msg.angleT_pts.push_back(traj.yaw_traj[i].getDuration());
        }
        // 添加最终偏航角点
        geometry_msgs::Point anglept;
        Eigen::VectorXd angle = traj.yaw_traj.getValue(traj.yaw_traj.getTotalDuration());
        anglept.x = angle[0];
        traj_msg.angle_pts.push_back(anglept);

        return traj_msg;
    }

    // 创建基于A*路径的轨迹消息
    mpc_controller::SE2Traj OnlyPlanner::createSimpleTrajectoryMsg(const std::vector<Eigen::Vector3d>& path,
                                                                   const Eigen::Vector3d& target_state,
                                                                   double total_time) {
        mpc_controller::SE2Traj traj_msg;
        traj_msg.start_time = ros::Time::now();
        traj_msg.init_v.x = 0.0;
        traj_msg.init_v.y = 0.0;
        traj_msg.init_v.z = 0.0;
        traj_msg.init_a.x = 0.0;
        traj_msg.init_a.y = 0.0;
        traj_msg.init_a.z = 0.0;

        // 使用A*规划的完整路径点
        if (path.empty()) {
            ROS_WARN("Empty path provided to createSimpleTrajectoryMsg");
            return traj_msg;
        }

        // 添加A*路径的所有点，但确保最后一个点是精确的目标点
        double dt = total_time / (path.size());  // 均匀分布时间

        // 添加A*路径的所有点（包括起点和中间点）
        for (size_t i = 0; i < path.size(); i++) {
            geometry_msgs::Point pos_pt;
            pos_pt.x = path[i](0);
            pos_pt.y = path[i](1);
            pos_pt.z = 0.0;
            traj_msg.pos_pts.push_back(pos_pt);

            geometry_msgs::Point angle_pt;
            angle_pt.x = path[i](2);  // 偏航角
            angle_pt.y = 0.0;
            angle_pt.z = 0.0;
            traj_msg.angle_pts.push_back(angle_pt);

            // 添加时间段（除了最后一个点）
            if (i < path.size() - 1) {
                traj_msg.posT_pts.push_back(dt);
                traj_msg.angleT_pts.push_back(dt);
            }
        }

        // 确保端点精确性：第一个点是精确的起始点，最后一个点是精确的目标点
        if (!traj_msg.pos_pts.empty()) {
            // 确保第一个点是精确的起始点
            traj_msg.pos_pts.front().x = path[0](0);  // 使用A*路径的起点（应该已经是准确的）
            traj_msg.pos_pts.front().y = path[0](1);
            traj_msg.angle_pts.front().x = path[0](2);

            // 确保最后一个点是精确的目标点
            traj_msg.pos_pts.back().x = target_state(0);
            traj_msg.pos_pts.back().y = target_state(1);
            traj_msg.angle_pts.back().x = target_state(2);

            // 添加最后一个时间段
            traj_msg.posT_pts.push_back(dt);
            traj_msg.angleT_pts.push_back(dt);

            // ROS_INFO("Trajectory endpoints: Start=[%.3f, %.3f, %.3f], End=[%.3f, %.3f, %.3f]",
            //          traj_msg.pos_pts.front().x, traj_msg.pos_pts.front().y, traj_msg.angle_pts.front().x,
            //          traj_msg.pos_pts.back().x, traj_msg.pos_pts.back().y, traj_msg.angle_pts.back().x);
        }

        // ROS_INFO("Created simple trajectory with %zu path points (A* path + exact target)",
        //          traj_msg.pos_pts.size());
        return traj_msg;
    }

    // 直接发送A*原始路径点，不进行重采样（用于后续贝塞尔拟合）
    mpc_controller::SE2Traj OnlyPlanner::createResampledPathMsg(const std::vector<Eigen::Vector3d>& path,
                                                               const Eigen::Vector3d& target_state,
                                                               double total_time) {
        mpc_controller::SE2Traj traj_msg;
        traj_msg.start_time = ros::Time::now();
        traj_msg.init_v.x = 0.0;
        traj_msg.init_v.y = 0.0;
        traj_msg.init_v.z = 0.0;
        traj_msg.init_a.x = 0.0;
        traj_msg.init_a.y = 0.0;
        traj_msg.init_a.z = 0.0;

        if (path.empty()) {
            ROS_WARN("Empty path provided to createResampledPathMsg");
            return traj_msg;
        }

        // 确保路径的起始点和结束点是精确的
        std::vector<Eigen::Vector3d> corrected_path = path;
        corrected_path[0] = Eigen::Vector3d(odom_pos.x(), odom_pos.y(), odom_pos.z());  // 精确的起始位姿
        corrected_path.back() = target_state;  // 精确的目标位姿

        ROS_INFO("Sending raw A* path with %zu points to Python for Bezier fitting", corrected_path.size());

        // 直接发送原始路径点，不进行重采样
        double dt = total_time / (corrected_path.size() - 1);

        for (size_t i = 0; i < corrected_path.size(); i++) {
            const Eigen::Vector3d& point = corrected_path[i];

            // 边界检查
            const double RESAMPLED_MAP_BOUNDARY = 17.6;
            if (point(0) < -RESAMPLED_MAP_BOUNDARY || point(0) > RESAMPLED_MAP_BOUNDARY ||
                point(1) < -RESAMPLED_MAP_BOUNDARY || point(1) > RESAMPLED_MAP_BOUNDARY) {
                ROS_WARN("Point %zu out of map bounds: [%.3f, %.3f], path rejected", i,
                         point(0), point(1));
                handlePlanningFailure();
                return traj_msg;
            }

            // 添加位置点
            geometry_msgs::Point pos_pt;
            pos_pt.x = point(0);
            pos_pt.y = point(1);
            pos_pt.z = 0.0;
            traj_msg.pos_pts.push_back(pos_pt);

            // 添加偏航角点
            geometry_msgs::Point angle_pt;
            angle_pt.x = point(2);
            angle_pt.y = 0.0;
            angle_pt.z = 0.0;
            traj_msg.angle_pts.push_back(angle_pt);

            // 添加时间段（最后一个点除外）
            if (i < corrected_path.size() - 1) {
                traj_msg.posT_pts.push_back(dt);
                traj_msg.angleT_pts.push_back(dt);
            }
        }

        ROS_INFO("Sent raw A* path with %zu points (no resampling, will be Bezier fitted in Python)",
                 traj_msg.pos_pts.size());
        return traj_msg;
    }

    // 创建轨迹消息（采样固定数量的轨迹点，用于后续处理）
    mpc_controller::SE2Traj OnlyPlanner::createResampledTrajectoryMsg(const SE2Trajectory& traj) {
        mpc_controller::SE2Traj traj_msg;
        traj_msg.start_time = ros::Time::now();

        double total_duration = traj.pos_traj.getTotalDuration();
        
        // 使用固定的采样点数，避免点数过多
        int num_samples = 100;  // 固定100个采样点
        
        ROS_INFO("Sampling optimized trajectory: duration=%.3f, num_samples=%d (fixed count)",
                 total_duration, num_samples);

        // 均匀采样轨迹点
        for (int i = 0; i < num_samples; i++) {
            double t = (double)i / (double)(num_samples - 1) * total_duration;

            // 获取位置
            Eigen::Vector2d pos = traj.pos_traj.getValue(t);
            geometry_msgs::Point pos_pt;
            pos_pt.x = pos(0);
            pos_pt.y = pos(1);
            pos_pt.z = 0.0;
            traj_msg.pos_pts.push_back(pos_pt);

            // 获取偏航角
            Eigen::VectorXd yaw = traj.yaw_traj.getValue(t);
            geometry_msgs::Point angle_pt;
            angle_pt.x = yaw(0);
            angle_pt.y = 0.0;
            angle_pt.z = 0.0;
            traj_msg.angle_pts.push_back(angle_pt);
        }

        // 设置时间信息（均匀分布）
        double dt = total_duration / (num_samples - 1);
        for (int i = 0; i < num_samples - 1; i++) {
            traj_msg.posT_pts.push_back(dt);
            traj_msg.angleT_pts.push_back(dt);
        }

        // 设置初始速度和加速度
        traj_msg.init_v.x = 0.0;
        traj_msg.init_v.y = 0.0;
        traj_msg.init_v.z = 0.0;

        traj_msg.init_a.x = 0.0;
        traj_msg.init_a.y = 0.0;
        traj_msg.init_a.z = 0.0;

        ROS_INFO("Sent trajectory with %zu points (fixed count sampling)",
                 traj_msg.pos_pts.size());

        return traj_msg;
    }
}