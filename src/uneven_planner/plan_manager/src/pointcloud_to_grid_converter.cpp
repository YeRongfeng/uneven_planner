#include "plan_manager/pointcloud_to_grid_converter.h"
#include <cmath>

PointCloudToGridConverter::PointCloudToGridConverter(ros::NodeHandle n) : nh_(n)
{
    processed_pts_.reset(new PointCloud<PointType>());
    cloud_normals_.reset(new PointCloud<Normal>());
    
    // 设置参数（匹配grid_transformer.cpp）
    grid_coarse_resolution_ = 0.4;   // 粗分辨率用于栅格化
    grid_fine_resolution_ = 0.2;     // 精细分辨率用于最终输出
    voxel_size_ = 0.2;               // 体素降采样大小
    
    // 注册服务
    convert_service_ = nh_.advertiseService("pointcloud_to_grid", 
                                           &PointCloudToGridConverter::convertCallback, 
                                           this);
    
    ROS_INFO("PointCloud to Grid Converter Service started!");
}

PointCloudToGridConverter::~PointCloudToGridConverter()
{
}

bool PointCloudToGridConverter::convertCallback(plan_manager::PointCloudToGrid::Request &req,
                                               plan_manager::PointCloudToGrid::Response &res)
{
    ROS_INFO("Received point cloud conversion request");
    
    try {
        // 1. 处理点云（降采样 + 法向量计算）
        pointProcessing(req.pointcloud);
        
        // 2. 生成栅格地图
        gridGenerate(req.map_min_x, req.map_min_y, req.map_max_x, req.map_max_y,
                    res.elevation_grid, res.normal_x_grid, res.normal_y_grid, res.normal_z_grid,
                    res.grid_width, res.grid_height);
        
        res.resolution = grid_coarse_resolution_;
        res.success = true;
        res.message = "Conversion successful";
        
        ROS_INFO("Point cloud conversion completed successfully. Grid size: %d x %d", 
                 res.grid_width, res.grid_height);
        
    } catch (const std::exception& e) {
        res.success = false;
        res.message = std::string("Conversion failed: ") + e.what();
        ROS_ERROR("%s", res.message.c_str());
    }
    
    return true;
}

void PointCloudToGridConverter::pointProcessing(const sensor_msgs::PointCloud2& pt_msg)
{
    // 转换ROS消息到PCL点云
    PointCloud<PointType>::Ptr raw_pts(new PointCloud<PointType>());
    pcl::fromROSMsg(pt_msg, *raw_pts);
    
    if (raw_pts->empty()) {
        throw std::runtime_error("Received empty point cloud!");
    }
    
    ROS_INFO("Processing point cloud with %zu points", raw_pts->size());
    
    // 1. 体素降采样
    pcl::VoxelGrid<PointType> voxel_grid;
    voxel_grid.setInputCloud(raw_pts);
    voxel_grid.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
    PointCloud<PointType>::Ptr downsampled_cloud(new PointCloud<PointType>());
    voxel_grid.filter(*downsampled_cloud);
    
    ROS_INFO("After downsampling: %zu points", downsampled_cloud->size());
    
    processed_pts_->points = downsampled_cloud->points;
    
    // 2. 计算法向量
    pcl::NormalEstimationOMP<PointType, pcl::Normal> ne;
    ne.setNumberOfThreads(4);
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
    ne.setInputCloud(downsampled_cloud);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(voxel_size_ * 5.0);  // 搜索半径
    ne.setViewPoint(0, 0, 0);
    
    ne.compute(*cloud_normals_);
    
    ROS_INFO("Normal vector calculation completed");
}

void PointCloudToGridConverter::gridGenerate(float map_min_x, float map_min_y, 
                                            float map_max_x, float map_max_y,
                                            vector<float>& elevation_grid,
                                            vector<float>& normal_x_grid,
                                            vector<float>& normal_y_grid,
                                            vector<float>& normal_z_grid,
                                            int& grid_width, int& grid_height)
{
    if (processed_pts_->empty()) {
        throw std::runtime_error("No processed points available!");
    }
    
    // 计算地图尺寸
    float length_x = map_max_x - map_min_x;
    float length_y = map_max_y - map_min_y;
    
    // 首先使用粗分辨率（0.4m）进行栅格化，得到粗栅格
    int coarse_width = static_cast<int>(std::ceil(length_x / grid_coarse_resolution_));
    int coarse_height = static_cast<int>(std::ceil(length_y / grid_coarse_resolution_));
    
    ROS_INFO("Coarse grid dimensions: %.2f x %.2f m", length_x, length_y);
    ROS_INFO("Coarse grid resolution: %.2f m", grid_coarse_resolution_);
    ROS_INFO("Coarse grid cells: %d x %d", coarse_width, coarse_height);
    
    // 分配点到粗栅格单元
    allocatePointsInsideGrid(map_min_x, map_min_y, map_max_x, map_max_y, 
                           coarse_width, coarse_height);
    
    // 创建粗栅格数组
    int coarse_total = coarse_width * coarse_height;
    vector<float> coarse_elevation(coarse_total, std::numeric_limits<float>::quiet_NaN());
    vector<float> coarse_normal_x(coarse_total, std::numeric_limits<float>::quiet_NaN());
    vector<float> coarse_normal_y(coarse_total, std::numeric_limits<float>::quiet_NaN());
    vector<float> coarse_normal_z(coarse_total, std::numeric_limits<float>::quiet_NaN());
    
    // 处理粗栅格单元
    ROS_INFO("Processing %d coarse grid cells...", coarse_total);
    
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
    #endif
    for (int i = 0; i < coarse_height; ++i) {
        for (int j = 0; j < coarse_width; ++j) {
            processGridCell(j, i, coarse_elevation, coarse_normal_x, coarse_normal_y, coarse_normal_z,
                          coarse_width, coarse_height);
        }
    }
    
    ROS_INFO("Coarse grid generation completed!");
    
    // 计算精细栅格尺寸（0.2m分辨率）- 这应该输出205x205
    grid_width = static_cast<int>(std::ceil(length_x / grid_fine_resolution_));
    grid_height = static_cast<int>(std::ceil(length_y / grid_fine_resolution_));
    
    ROS_INFO("Fine grid resolution: %.2f m", grid_fine_resolution_);
    ROS_INFO("Fine grid cells: %d x %d", grid_width, grid_height);
    
    // 使用最近邻插值从粗栅格插值到精细栅格
    int fine_total = grid_width * grid_height;
    elevation_grid.resize(fine_total, std::numeric_limits<float>::quiet_NaN());
    normal_x_grid.resize(fine_total, std::numeric_limits<float>::quiet_NaN());
    normal_y_grid.resize(fine_total, std::numeric_limits<float>::quiet_NaN());
    normal_z_grid.resize(fine_total, std::numeric_limits<float>::quiet_NaN());
    
    // 最近邻插值
    for (int i = 0; i < grid_height; ++i) {
        for (int j = 0; j < grid_width; ++j) {
            // 计算精细格点在地图中的位置
            float fine_x = map_min_x + j * grid_fine_resolution_;
            float fine_y = map_min_y + i * grid_fine_resolution_;
            
            // 找到最近的粗格点
            int coarse_j = static_cast<int>((fine_x - map_min_x) / grid_coarse_resolution_);
            int coarse_i = static_cast<int>((fine_y - map_min_y) / grid_coarse_resolution_);
            
            // 边界检查
            coarse_i = std::max(0, std::min(coarse_i, coarse_height - 1));
            coarse_j = std::max(0, std::min(coarse_j, coarse_width - 1));
            
            int fine_idx = i * grid_width + j;
            int coarse_idx = coarse_i * coarse_width + coarse_j;
            
            elevation_grid[fine_idx] = coarse_elevation[coarse_idx];
            normal_x_grid[fine_idx] = coarse_normal_x[coarse_idx];
            normal_y_grid[fine_idx] = coarse_normal_y[coarse_idx];
            normal_z_grid[fine_idx] = coarse_normal_z[coarse_idx];
        }
    }
    
    ROS_INFO("Fine grid interpolation completed!");
}

void PointCloudToGridConverter::allocatePointsInsideGrid(float min_x, float min_y, 
                                                        float max_x, float max_y,
                                                        int width, int height)
{
    // 重新分配空间
    point_index_within_grid_cell_.clear();
    point_index_within_grid_cell_.resize(width);
    for (int i = 0; i < width; ++i) {
        point_index_within_grid_cell_[i].resize(height);
    }
    
    // 将点分配到对应的栅格单元
    for (size_t i = 0; i < processed_pts_->points.size(); ++i) {
        const PointType& point = processed_pts_->points[i];
        
        // 计算栅格索引
        int grid_x = static_cast<int>((point.x - min_x) / grid_coarse_resolution_);
        int grid_y = static_cast<int>((point.y - min_y) / grid_coarse_resolution_);
        
        // 检查边界
        if (grid_x >= 0 && grid_x < width && grid_y >= 0 && grid_y < height) {
            point_index_within_grid_cell_[grid_x][grid_y].push_back(i);
        }
    }
    
    ROS_INFO("Points allocated to grid cells");
}

void PointCloudToGridConverter::processGridCell(int grid_x, int grid_y,
                                               vector<float>& elevation_grid,
                                               vector<float>& normal_x_grid,
                                               vector<float>& normal_y_grid,
                                               vector<float>& normal_z_grid,
                                               int width, int height)
{
    const auto& point_indices = point_index_within_grid_cell_[grid_x][grid_y];
    int linear_index = grid_y * width + grid_x;
    
    if (point_indices.empty()) {
        // 空单元，保持NaN
        return;
    }
    
    // 找到法向量Z分量最小的点（最倾斜的表面）
    float min_normalized_z = INFINITY;
    size_t min_z_index = 0;
    float highest_elevation = -INFINITY;
    
    for (auto idx : point_indices) {
        // 更新最高点
        if (processed_pts_->points[idx].z > highest_elevation) {
            highest_elevation = processed_pts_->points[idx].z;
        }
        
        // 计算归一化的Z分量
        float nx = cloud_normals_->points[idx].normal_x;
        float ny = cloud_normals_->points[idx].normal_y;
        float nz = cloud_normals_->points[idx].normal_z;
        float norm = std::sqrt(nx*nx + ny*ny + nz*nz);
        
        if (norm > 1e-6) {
            float normalized_z = std::abs(nz / norm);
            if (normalized_z < min_normalized_z) {
                min_normalized_z = normalized_z;
                min_z_index = idx;
            }
        }
    }
    
    // 获取选定点的法向量
    float nx = cloud_normals_->points[min_z_index].normal_x;
    float ny = cloud_normals_->points[min_z_index].normal_y;
    float nz = cloud_normals_->points[min_z_index].normal_z;
    float norm = std::sqrt(nx*nx + ny*ny + nz*nz);
    
    // 存储数据
    elevation_grid[linear_index] = highest_elevation;
    if (norm > 1e-6) {
        normal_x_grid[linear_index] = nx / norm;
        normal_y_grid[linear_index] = ny / norm;
        normal_z_grid[linear_index] = nz / norm;
    }
}
