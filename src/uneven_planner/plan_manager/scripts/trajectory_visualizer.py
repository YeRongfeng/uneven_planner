#!/usr/bin/env python3

import rospy
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped
from mpc_controller.msg import SE2Traj
from std_msgs.msg import Bool
import threading
import queue
import time
import os

class TrajectoryVisualizer:
    def __init__(self):
        rospy.init_node('trajectory_visualizer', anonymous=True)
        
        # 可视化参数
        self.enable_visualization = rospy.get_param('~enable_visualization', True)
        self.save_plots = rospy.get_param('~save_plots', True)
        self.plot_dir = rospy.get_param('~plot_dir', '/tmp/trajectory_plots')
        
        if not self.enable_visualization:
            rospy.loginfo("Visualization disabled")
            return
            
        # 数据存储
        self.start_pose = None
        self.target_pose = None
        self.trajectory = None
        self.planning_success = None
        self.current_path_id = 0
        
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_xlim(-5.2, 5.2)  # 与实际地图边界一致
        self.ax.set_ylim(-5.2, 5.2)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Real-time Trajectory Planning Visualization')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # 绘图元素
        self.start_point = None
        self.target_point = None
        self.trajectory_line = None
        self.trajectory_arrows = []
        
        # ROS订阅者
        self.start_sub = rospy.Subscriber('/data_generate_node/start_pose',
                                        PoseStamped, self.start_pose_callback)
        self.target_sub = rospy.Subscriber('/data_generate_node/target_pose',
                                         PoseStamped, self.target_pose_callback)
        self.traj_sub = rospy.Subscriber('/data_generate_node/optimized_traj',
                                       SE2Traj, self.trajectory_callback)
        self.result_sub = rospy.Subscriber('/data_generate_node/planning_result',
                                         Bool, self.result_callback)
        
        rospy.loginfo("Trajectory Visualizer initialized")
        rospy.loginfo("Visualization parameters:")
        rospy.loginfo("  - Enable visualization: %s", self.enable_visualization)
        rospy.loginfo("  - Save plots: %s", self.save_plots)
        rospy.loginfo("  - Plot directory: %s", self.plot_dir)
        
        # 创建保存目录
        if self.save_plots:
            import os
            os.makedirs(self.plot_dir, exist_ok=True)
    

    
    def start_pose_callback(self, msg):
        """接收起始位姿"""
        self.start_pose = msg
        rospy.loginfo("Received start pose: [%.3f, %.3f, %.3f]",
                     msg.pose.position.x, msg.pose.position.y,
                     self.quaternion_to_yaw(msg.pose.orientation))
        # 重置规划状态
        self.planning_success = None
        self.trajectory = None
        # 清除之前的绘图，开始新轨迹
        # self.clear_plot()

    def target_pose_callback(self, msg):
        """接收目标位姿"""
        self.target_pose = msg
        rospy.loginfo("Received target pose: [%.3f, %.3f, %.3f]",
                     msg.pose.position.x, msg.pose.position.y,
                     self.quaternion_to_yaw(msg.pose.orientation))
        # 目标位姿接收后不立即绘图，等待规划结果

    def trajectory_callback(self, msg):
        """接收轨迹数据"""
        self.trajectory = msg
        rospy.loginfo("Received trajectory with %d position points and %d angle points",
                     len(msg.pos_pts), len(msg.angle_pts))
        # 轨迹接收后不立即绘图，等待规划结果确认成功
    
    def result_callback(self, msg):
        """接收规划结果"""
        self.planning_success = msg.data
        if msg.data:
            rospy.loginfo("Planning succeeded for path_%d", self.current_path_id)
            # 只有成功时才更新绘图和保存
            if((self.current_path_id) % 1 == 0):
                self.update_plot()
                if self.save_plots:
                    self.save_current_plot()
                    self.clear_plot()
            self.current_path_id += 1
        else:
            rospy.logwarn("Planning failed for path_%d", self.current_path_id)
            # 失败时不更新绘图，不保存图像
    
    def quaternion_to_yaw(self, q):
        """四元数转偏航角"""
        import math
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def clear_plot(self):
        """清除所有绘图元素，准备绘制新轨迹"""
        if not self.enable_visualization:
            return

        # 清除所有绘图元素
        self.ax.clear()

        # 重新设置图形属性（与实际地图边界一致：-5到5）
        self.ax.set_xlim(-5.2, 5.2)  # 稍微大于地图边界，便于观察
        self.ax.set_ylim(-5.2, 5.2)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Real-time Trajectory Planning Visualization')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        # 重置绘图元素引用
        self.start_point = None
        self.target_point = None
        self.trajectory_line = None
        self.trajectory_arrows = []

        rospy.loginfo("Cleared plot for new trajectory")
    
    def update_plot(self):
        """更新绘图"""
        if not self.enable_visualization:
            return

        # 安全地清除之前的绘图元素
        try:
            if self.start_point:
                self.start_point.remove()
                self.start_point = None
        except (ValueError, AttributeError):
            pass

        try:
            if self.target_point:
                self.target_point.remove()
                self.target_point = None
        except (ValueError, AttributeError):
            pass

        try:
            if self.trajectory_line:
                self.trajectory_line[0].remove()
                self.trajectory_line = None
        except (ValueError, AttributeError, IndexError):
            pass

        # 清除箭头
        for arrow in self.trajectory_arrows:
            try:
                arrow.remove()
            except (ValueError, AttributeError):
                pass
        self.trajectory_arrows.clear()
        
        # 绘制起始点 - 直接使用轨迹的第0个点（位置和角度）
        if self.trajectory and len(self.trajectory.pos_pts) > 0:
            x, y = self.trajectory.pos_pts[0].x, self.trajectory.pos_pts[0].y

            # 修复：直接使用x字段存储的偏航角（与only_planner.cpp一致）
            if len(self.trajectory.angle_pts) > 0:
                angle_pt = self.trajectory.angle_pts[0]
                yaw = angle_pt.x  # 偏航角存储在x字段中
                rospy.loginfo("Start point yaw=%.3f (from angle_pt.x)", yaw)
            else:
                yaw = 0.0
                rospy.logwarn("No angle data available for start point")

            self.start_point = self.ax.scatter(x, y, c='green', s=100, marker='o',
                                             label='Start', zorder=5)
            # 绘制起始方向箭头 - 使用轨迹第0个点的角度
            dx, dy = 0.3 * np.cos(yaw), 0.3 * np.sin(yaw)
            arrow = self.ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1,
                                fc='green', ec='green', alpha=0.7)
            self.trajectory_arrows.append(arrow)

        # 绘制目标点 - 直接使用轨迹的最后一个点（位置和角度）
        if self.trajectory and len(self.trajectory.pos_pts) > 0:
            x, y = self.trajectory.pos_pts[-1].x, self.trajectory.pos_pts[-1].y

            # 修复：直接使用x字段存储的偏航角（与only_planner.cpp一致）
            if len(self.trajectory.angle_pts) > 0:
                angle_pt = self.trajectory.angle_pts[-1]
                yaw = angle_pt.x  # 偏航角存储在x字段中
                rospy.loginfo("End point yaw=%.3f (from angle_pt.x)", yaw)
            else:
                yaw = 0.0
                rospy.logwarn("No angle data available for end point")

            self.target_point = self.ax.scatter(x, y, c='red', s=100, marker='s',
                                              label='Target', zorder=5)
            # 绘制目标方向箭头 - 使用轨迹最后一个点的角度
            dx, dy = 0.3 * np.cos(yaw), 0.3 * np.sin(yaw)
            arrow = self.ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1,
                                fc='red', ec='red', alpha=0.7)
            self.trajectory_arrows.append(arrow)
        
        # 绘制轨迹
        if self.trajectory and len(self.trajectory.pos_pts) > 0:
            # 提取位置点
            x_traj = [pt.x for pt in self.trajectory.pos_pts]
            y_traj = [pt.y for pt in self.trajectory.pos_pts]
            
            # 根据规划结果选择颜色，并添加点数信息
            num_points = len(x_traj)
            if self.planning_success is True:
                color = 'blue'
                alpha = 0.8
                label = f'Trajectory (Success) - Path {self.current_path_id} - {num_points} pts'
            elif self.planning_success is False:
                color = 'orange'
                alpha = 0.6
                label = f'Trajectory (Failed) - Path {self.current_path_id} - {num_points} pts'
            else:
                color = 'gray'
                alpha = 0.5
                label = f'Trajectory (Planning...) - Path {self.current_path_id} - {num_points} pts'
            
            self.trajectory_line = self.ax.plot(x_traj, y_traj, color=color, 
                                              linewidth=2, alpha=alpha, 
                                              label=label, zorder=3)
            
            # 绘制轨迹点
            self.ax.scatter(x_traj, y_traj, c=color, s=20, alpha=alpha, zorder=4)
            
            # 绘制偏航角箭头（每隔几个点绘制一个）
            if len(self.trajectory.angle_pts) > 0:
                step = max(1, len(x_traj) // 8)  # 最多显示8个箭头
                for i in range(0, len(x_traj), step):
                    if i < len(self.trajectory.angle_pts):
                        x, y = x_traj[i], y_traj[i]
                        yaw = self.trajectory.angle_pts[i].x  # 修复：使用正确的角度字段
                        dx, dy = 0.2 * np.cos(yaw), 0.2 * np.sin(yaw)
                        arrow = self.ax.arrow(x, y, dx, dy, head_width=0.05,
                                            head_length=0.05, fc=color, ec=color,
                                            alpha=alpha*0.7, zorder=4)
                        self.trajectory_arrows.append(arrow)
        
        # 更新图例和标题
        self.ax.legend(loc='upper right')
        if self.planning_success is not None:
            status = "SUCCESS" if self.planning_success else "FAILED"
            self.ax.set_title(f'Trajectory Planning - Path {self.current_path_id} - {status}')
        else:
            self.ax.set_title(f'Trajectory Planning - Path {self.current_path_id} - PLANNING...')
        
        # 非交互式模式：只保存图像，不显示窗口
        # 这避免了matplotlib线程问题
    
    def save_current_plot(self):
        """保存当前图像"""
        if not self.save_plots:
            return

        filename = f"{self.plot_dir}/trajectory_path_{self.current_path_id:03d}.png"
        try:
            # 确保图形已经绘制
            self.fig.canvas.draw()
            # 保存图像
            self.fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            rospy.loginfo("Saved plot: %s", filename)
        except Exception as e:
            rospy.logerr("Failed to save plot: %s", str(e))
    
    def run(self):
        """运行可视化"""
        if not self.enable_visualization:
            rospy.spin()
            return

        rospy.loginfo("Running in non-interactive mode (save-only)")

        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down trajectory visualizer")
        finally:
            plt.close('all')

if __name__ == '__main__':
    try:
        visualizer = TrajectoryVisualizer()
        visualizer.run()
    except rospy.ROSInterruptException:
        pass
