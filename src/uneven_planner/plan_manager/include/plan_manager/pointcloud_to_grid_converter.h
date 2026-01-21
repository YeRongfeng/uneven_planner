#ifndef POINTCLOUD_TO_GRID_CONVERTER_H_
#define POINTCLOUD_TO_GRID_CONVERTER_H_

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_srvs/Trigger.h>
#include <vector>

// PCL
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>

// 自定义服务消息
#include <plan_manager/PointCloudToGrid.h>

using namespace std;
using namespace pcl;

typedef pcl::PointXYZI PointType;

class PointCloudToGridConverter
{
private:
    ros::NodeHandle nh_;
    ros::ServiceServer convert_service_;
    
    float grid_coarse_resolution_;  // 粗分辨率（0.4m）用于栅格化
    float grid_fine_resolution_;    // 精细分辨率（0.2m）用于最终输出  
    float voxel_size_;              // 体素降采样大小（0.2m）
    
    PointCloud<PointType>::Ptr processed_pts_;
    PointCloud<Normal>::Ptr cloud_normals_;
    
    // 栅格单元内的点索引
    vector<vector<vector<size_t>>> point_index_within_grid_cell_;

public:
    PointCloudToGridConverter(ros::NodeHandle n);
    ~PointCloudToGridConverter();
    
    // 服务回调函数
    bool convertCallback(plan_manager::PointCloudToGrid::Request &req,
                        plan_manager::PointCloudToGrid::Response &res);
    
    // 点云处理
    void pointProcessing(const sensor_msgs::PointCloud2& pt_msg);
    
    // 栅格生成
    void gridGenerate(float map_min_x, float map_min_y, float map_max_x, float map_max_y,
                     vector<float>& elevation_grid,
                     vector<float>& normal_x_grid,
                     vector<float>& normal_y_grid,
                     vector<float>& normal_z_grid,
                     int& grid_width, int& grid_height);
    
    // 分配点到栅格单元
    void allocatePointsInsideGrid(float min_x, float min_y, float max_x, float max_y,
                                 int width, int height);
    
    // 处理单个栅格单元
    void processGridCell(int grid_x, int grid_y,
                        vector<float>& elevation_grid,
                        vector<float>& normal_x_grid,
                        vector<float>& normal_y_grid,
                        vector<float>& normal_z_grid,
                        int width, int height);
};

#endif // POINTCLOUD_TO_GRID_CONVERTER_H_
