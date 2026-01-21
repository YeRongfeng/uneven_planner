#include "elevation_generator/grid_transformer.h"
#include <visualization_msgs/Marker.h>
// #include <cmathasdasd.h>
#include <cmath>
using namespace visualization_msgs;

int i=0;
GridTransformer::GridTransformer(ros::NodeHandle n):
system_initialized_(false), nh_(n), 
window_length_(13.0), octree_resolution_(0.4), global_frame_("map"),  //8.0
grid_desired_resolution_(0.2),grid_coarse_resolution_(0.4),
pcd_sub_(n.subscribe("pcd_cloud", 1, &GridTransformer::pointHandler, this)),
pose_sub_(n.subscribe("base_link_pose", 1, &GridTransformer::poseHandler, this)),
octomap_pub_(n.advertise<octomap_msgs::Octomap>("full_octomap",1)),
full_grid_pub_(n.advertise<grid_map_msgs::GridMap>("full_grid_map",1)),
subgrid_pub_(n.advertise<grid_map_msgs::GridMap>("subgrid_map",1)),
marker_pub_(n.advertise<visualization_msgs::MarkerArray>("visualization_marker_zean", 10))   //zean
{
    raw_pts_.reset(new PointCloud<PointType>());
    processed_pts_.reset(new PointCloud<PointType>());
    cloud_normals.reset(new PointCloud<Normal>());
    full_octree_ = new octomap::OcTree(octree_resolution_);
    full_gridmap_ = new grid_map::GridMap();
    line_list= new visualization_msgs::Marker;
    marker_array=new visualization_msgs::MarkerArray;
    interpolated_gridmap_ = new grid_map::GridMap();
    sub_gridmap_ = new grid_map::GridMap();
    publish_rater = n.createTimer(ros::Duration(20.0), &GridTransformer::publishHandler, this); //5  zean 8-14 20:28t
    local_publish_rater = n.createTimer(ros::Duration(0.05), &GridTransformer::publishLocalHandler, this);
    publish_rater.stop();
    local_publish_rater.stop();
    line_list->header.frame_id = "map";
    line_list->ns = "points_and_lines";
    line_list->action = visualization_msgs::Marker::ADD;
    line_list->pose.orientation.w = 1.0;
    line_list->id = 0;
    line_list->header.stamp = ros::Time::now();
    // line_list->type = visualization_msgs::Marker::LINE_LIST;
    line_list->type = visualization_msgs::Marker::ARROW;
    line_list->scale.x = 0.05;
     line_list->scale.y = 0.1;// 箭头头的宽度
    line_list->color.r = 1.0;
    line_list->color.a = 1.0;
}

GridTransformer::~GridTransformer(){}

void GridTransformer::pointHandler(const sensor_msgs::PointCloud2Ptr &pt_map){
    if(!system_initialized_ && raw_pts_->empty()){
        pcl::fromROSMsg<PointType>(*pt_map, *raw_pts_);
        if(raw_pts_->empty()){
            ROS_WARN("Recervied empty points!");
        }
        pointProcessing(raw_pts_);
        publish_rater.start();
        local_publish_rater.start();
    }
}

void GridTransformer::poseHandler(const geometry_msgs::PoseWithCovarianceStampedPtr &pose_conv){
    robot_pose_.header = pose_conv->header;
    robot_pose_.pose = pose_conv->pose.pose;
}

void GridTransformer::pointProcessing(PointCloud<PointType>::Ptr pt_in){
    // Downsampling at first
    pcl::VoxelGrid<PointType> voxelGrid;
    voxelGrid.setInputCloud(pt_in);
    const float voxelSize = 0.2;
    voxelGrid.setLeafSize(voxelSize, voxelSize, voxelSize);
    PointCloud<PointType>::Ptr downsampled_cloud(new PointCloud<PointType>());
    voxelGrid.filter(*downsampled_cloud);
    printf("Downsampling accomplished!\n");
    processed_pts_->points = downsampled_cloud->points;

    // Calculate normal vector
    pcl::NormalEstimationOMP<PointType, pcl::Normal> ne; // 基于omp并行加速，需配置开启OpenMP
    ne.setNumberOfThreads(10);
    // pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
    ne.setInputCloud(downsampled_cloud);

    ne.setSearchMethod(tree);

    // ne.setRadiusSearch(voxelSize*5);    // 上边做了一次体素降采样，所以设置半径时，要考虑到此时的点云空间间距
    ne.setRadiusSearch(voxelSize*2.5);    // SUCCESS 1.5
    // ne.setRadiusSearch(voxelSize*2);    // test
    //ZEAN!

    // Set pts view
    ne.setViewPoint(0,0,0);
    // Calculate normal
    ne.compute(*cloud_normals);
    printf("Normal vector calculation accomplished!\n");
}

void GridTransformer::gridGenerate(){
    full_gridmap_->clearAll();
    interpolated_gridmap_->clearAll();
    if (grid_coarse_resolution_ < 1e-4) {
        throw std::runtime_error("Desired grid map resolution is zero");
    }

    // find point cloud dimensions
    // min and max coordinate in x,y and z direction
    PointType minBound;
    PointType maxBound;
    pcl::getMinMax3D(*processed_pts_, minBound, maxBound);

    // from min and max points we can compute the length
    grid_map::Length length = grid_map::Length(maxBound.x - minBound.x, maxBound.y - minBound.y);

    // we put the center of the grid map to be in the middle of the point cloud
    grid_map::Position position = grid_map::Position((maxBound.x + minBound.x) / 2.0, (maxBound.y + minBound.y) / 2.0);
    // std::cout << "length(0)" << length(0) << std::endl;
    // std::cout << "length(1)" << length(1) << std::endl;
    full_gridmap_->setGeometry(length, grid_coarse_resolution_, position);
    //zean resolution
    full_gridmap_->setFrameId(global_frame_);
    interpolated_gridmap_->setGeometry(length, grid_desired_resolution_, position);
    interpolated_gridmap_->setFrameId(global_frame_);

    ROS_INFO_STREAM("Grid map dimensions: " << full_gridmap_->getLength()(0) << " x " << full_gridmap_->getLength()(1));
    ROS_INFO_STREAM("Grid map resolution: " << full_gridmap_->getResolution());
    ROS_INFO_STREAM("Grid map num cells: " << full_gridmap_->getSize()(0) << " x " << full_gridmap_->getSize()(1));

    allocatePointsInsideGrid();
    ROS_INFO_STREAM("Initialized map point\n");
    // Add elevation layers
    full_gridmap_->add("elevation");
    full_gridmap_->add("normal_x");
    full_gridmap_->add("normal_y");
    full_gridmap_->add("normal_z");
    interpolated_gridmap_->add("elevation");
    interpolated_gridmap_->add("normal_x");
    interpolated_gridmap_->add("normal_y");
    interpolated_gridmap_->add("normal_z");

    grid_map::Matrix& elevationData = full_gridmap_->get("elevation");
    grid_map::Matrix& normalXData = full_gridmap_->get("normal_x");
    grid_map::Matrix& normalYData = full_gridmap_->get("normal_y");
    grid_map::Matrix& normalZData = full_gridmap_->get("normal_z");
    unsigned int linearGridMapSize = full_gridmap_->getSize().prod();
    visualization_msgs::Marker  line_list_;
    //  line_list_.header.frame_id = "/my_frame";
    // line_list_.ns = "points_and_lines";
    // line_list_.action = visualization_msgs::Marker::ADD;
    // line_list_.pose.orientation.w = 1.0;
    // line_list_.id = 1;
    // line_list_.header.stamp = ros::Time::now();
    // line_list_.type = visualization_msgs::Marker::LINE_LIST;
    // line_list_.scale.x = 0.1;
    // line_list_.color.r = 1.0;
    // line_list_.color.a = 1.0;
    // Processing every grid cells
    // Iterate through grid map and calculate the corresponding height based on the point cloud
    #ifndef GRID_MAP_PCL_OPENMP_FOUND
    ROS_WARN_STREAM("OpemMP not found, defaulting to single threaded implementation");
    #else
    omp_set_num_threads(4);
    #pragma omp parallel for schedule(dynamic, 10)
    #endif
    
    //  why not ZEAN !!!!!
    for (unsigned int linearIndex = 0; linearIndex < linearGridMapSize; ++linearIndex) {
        processGridCells(linearIndex, &elevationData, &normalXData, &normalYData, &normalZData, &line_list_);
    }
    //  marker_pub_.publish(*line_list); 
    ROS_INFO_STREAM("Finished adding layer!\n");
}

void GridTransformer::processGridCells(const unsigned int linearGridMapIndex, grid_map::Matrix* gridMapData,
                      grid_map::Matrix* normalXData, grid_map::Matrix* normalYData, grid_map::Matrix* normalZData,visualization_msgs::Marker* line_list_){
    const grid_map::Index index(grid_map::getIndexFromLinearIndex(linearGridMapIndex, full_gridmap_->getSize()));
        boost::mutex::scoped_lock l(pub_marker_mutex_);
    auto point_indices = point_indexWithinGridMapCell_[index.x()][index.y()];
    if(point_indices.size()<1){
        // Empty cell
        // ROS_WARN_STREAM_THROTTLE(10.0, "Less than " << params_.get().gridMap_.minCloudPointsPerCell_ << " points in a cell. Skipping.");
        (*gridMapData)(index(0), index(1)) = std::nan("1");
        (*normalXData)(index(0), index(1)) = std::nan("1");
        (*normalYData)(index(0), index(1)) = std::nan("1");
        (*normalZData)(index(0), index(1)) = std::nan("1");
        return;
    }
    
    geometry_msgs::Point p;
    
    // Fill matrix data
    float min_z = INFINITY;
    float highest_elevation = -1*INFINITY;
    size_t highest_index = 0;
    size_t min_z_index = 0;
    // Find the highest point from ground
    float x,y,z;
    for(auto ind : point_indices){
        if(processed_pts_->points[ind].z > highest_elevation){
            highest_elevation = processed_pts_->points[ind].z;
            highest_index = ind;
        }
        x=cloud_normals->points[ind].normal_x;y=cloud_normals->points[ind].normal_y;z=cloud_normals->points[ind].normal_z;
        if(abs(z/sqrt(x*x+y*y+z*z)) <= min_z){
            min_z = abs(z/sqrt(x*x+y*y+z*z));
            min_z_index = ind;
            p.x=processed_pts_->points[ind].x;
            p.y=processed_pts_->points[ind].y;
            p.z=processed_pts_->points[ind].z;
        }
    }
    //30000 before 1-31   zean    18000 before 2-16
    if(i < 30000) {
        //line_list_->points.push_back(p);
        line_list->id = i;
        line_list->header.stamp = ros::Time::now();
        line_list->points.push_back(p);
        i++;
    } else if(i == 30000) {
        i=0;
        if (marker_array != nullptr) {
            ROS_WARN("!!!!Publishing marker with %lu points", marker_array->markers.size());
            // line_list->header.stamp = ros::Time::now();
            marker_pub_.publish(*marker_array);
            marker_array->markers.clear();
            line_list->id = i;
            line_list->points.push_back(p);
            //    multithreading!!!!
        } else {
            ROS_ERROR("line_list is null");
        }
        i=1;
    }
    x=cloud_normals->points[min_z_index].normal_x;y=cloud_normals->points[min_z_index].normal_y;z=cloud_normals->points[min_z_index].normal_z;
    float norm=sqrt(x*x+y*y+z*z);
    (*gridMapData)(index(0), index(1)) = highest_elevation;
    (*normalXData)(index(0), index(1)) = x/norm;
    (*normalYData)(index(0), index(1)) = y/norm;
    (*normalZData)(index(0), index(1)) = z/norm;


    // (*gridMapData)(index(0), index(1)) = highest_elevation;
    // (*normalXData)(index(0), index(1)) = cloud_normals->points[highest_index].normal_x;
    // (*normalYData)(index(0), index(1)) = cloud_normals->points[highest_index].normal_y;
    // (*normalZData)(index(0), index(1)) = cloud_normals->points[highest_index].normal_z;
    if(z>0)
    {
        p.x+=x/norm;
        p.y+=y/norm;
        p.z+=z/norm;
    }
    else
    {
        p.x-=x;
        p.y-=y;
        p.z-=z;

    }
    if(i<30000)
    {
        // line_list_->points.push_back(p);
        line_list->id = i;
        // line_list->header.stamp = ros::Time::now();
        line_list->points.push_back(p);
         marker_array->markers.push_back(*line_list);
        line_list->points.clear();
        i++;
    }
}

void GridTransformer::allocatePointsInsideGrid(){
    // Allocate space
    const unsigned int dimX = full_gridmap_->getSize().x() + 1;
    const unsigned int dimY = full_gridmap_->getSize().y() + 1;

    // resize vectors
    point_indexWithinGridMapCell_.resize(dimX);

    // allocate pointClouds
    for (unsigned int i = 0; i < dimX; ++i) {
        point_indexWithinGridMapCell_[i].resize(dimY);
    }

    // Allocate 
    for (unsigned int i = 0; i < processed_pts_->points.size(); ++i) {
        const PointType& point = processed_pts_->points[i];
        const double x = point.x;
        const double y = point.y;
        grid_map::Index index;
        full_gridmap_->getIndex(grid_map::Position(x, y), index);
        point_indexWithinGridMapCell_[index.x()][index.y()].push_back(i);
    }
    
}
bool GridTransformer::safeCheck(double x,double y)
{
    if(x>0&&y>0&&x<32.4&&y<18.4)
    return 1;
    return 0;
}

/*new_way


void GridTransformer::interpolateSmoothing(const grid_map::GridMap& coarse_grid, grid_map::GridMap& interpolate_grid){
    float last_elevation, last_x, last_y, last_z;
    grid_map::Size coarse_size = coarse_grid.getSize(); 

    for (grid_map::GridMapIterator iterator(interpolate_grid); !iterator.isPastEnd(); ++iterator) {
        const grid_map::Index index(*iterator);
        grid_map::Position pos;
        interpolate_grid.getPosition(index, pos);
        grid_map::Index coarse_index;
        if(!coarse_grid.getIndex(pos, coarse_index)){
            // Avoid over boundary, use last data
            interpolate_grid.at("elevation", index) = last_elevation;
            interpolate_grid.at("normal_x", index) = last_x;
            interpolate_grid.at("normal_y", index) = last_y;
            interpolate_grid.at("normal_z", index) = last_z;
            continue;
        }
        double xm=coarse_grid.atPosition("normal_x", pos),ym=coarse_grid.atPosition("normal_y", pos),zm=coarse_grid.atPosition("normal_z", pos);
        pos(0)=pos(0)-0.4;
        for(int i=1;i<=3;i++)
        {
            pos(1)-=0.4;
            for(int j=1;j<=3;j++)
            {
                if(safeCheck(pos(0),pos(1)))
                {
                    const double interpolated_normalZ = coarse_grid.atPosition("normal_z", pos);
                    if(interpolated_normalZ<zm)
                    {
                        zm=interpolated_normalZ;
                        xm =coarse_grid.atPosition("normal_x", pos);
                        ym =coarse_grid.atPosition("normal_y", pos);
                    }

                }
                pos(1)+=0.4;
            }
            pos(1)-=0.8;
            pos(0)+=0.4;
        }
        pos(0)-=0.8;

        //INTER_CUBIC 8-11 14:37
        const double interpolated_elevation = coarse_grid.atPosition("elevation", pos, grid_map::InterpolationMethods::INTER_CUBIC);
        interpolate_grid.at("elevation", index) = interpolated_elevation;
        last_elevation = interpolated_elevation;
        // INTER_NEAREST  INTER_LINEAR  INTER_CUBIC_SPLINE  INTER_LANCZOS

        //zean 8-11 14:50
        // const double interpolated_normalX = coarse_grid.atPosition("normal_x", pos, grid_map::InterpolationMethods::INTER_NEAREST);
        // interpolate_grid.at("normal_x", index) = interpolated_normalX;
        // last_x = interpolated_normalX;
        interpolate_grid.at("normal_x", index) = xm;
        last_x = xm;
        // const double interpolated_normalY = coarse_grid.atPosition("normal_y", pos, grid_map::InterpolationMethods::INTER_NEAREST);
        // interpolate_grid.at("normal_y", index) = interpolated_normalY;
        // last_y = interpolated_normalY;
        interpolate_grid.at("normal_y", index) = ym;
        last_y = ym;
        
        // const double interpolated_normalZ = coarse_grid.atPosition("normal_z", pos, grid_map::InterpolationMethods::INTER_NEAREST);
        // interpolate_grid.at("normal_z", index) = interpolated_normalZ;
        // last_z = interpolated_normalZ;
        interpolate_grid.at("normal_z", index) = zm;
        last_z = zm;

        if(coarse_index(0) == coarse_size(0)-1 && coarse_index(1) == coarse_size(1)-1) break;
    }

}
*/

/*  origin */
void GridTransformer::interpolateSmoothing(const grid_map::GridMap& coarse_grid, grid_map::GridMap& interpolate_grid){
    float last_elevation, last_x, last_y, last_z;
    grid_map::Size coarse_size = coarse_grid.getSize(); 

    for (grid_map::GridMapIterator iterator(interpolate_grid); !iterator.isPastEnd(); ++iterator) {
        const grid_map::Index index(*iterator);
        grid_map::Position pos;
        interpolate_grid.getPosition(index, pos);
        grid_map::Index coarse_index;
        if(!coarse_grid.getIndex(pos, coarse_index)){
            // Avoid over boundary, use last data
            interpolate_grid.at("elevation", index) = last_elevation;
            interpolate_grid.at("normal_x", index) = last_x;
            interpolate_grid.at("normal_y", index) = last_y;
            interpolate_grid.at("normal_z", index) = last_z;
            continue;
        }

        //INTER_CUBIC 8-11 14:37
        const double interpolated_elevation = coarse_grid.atPosition("elevation", pos, grid_map::InterpolationMethods::INTER_CUBIC);
        interpolate_grid.at("elevation", index) = interpolated_elevation;
        last_elevation = interpolated_elevation;
        // INTER_NEAREST  INTER_LINEAR  INTER_CUBIC_SPLINE  INTER_LANCZOS

        //zean 8-11 14:50
        const double interpolated_normalX = coarse_grid.atPosition("normal_x", pos, grid_map::InterpolationMethods::INTER_NEAREST);
        interpolate_grid.at("normal_x", index) = interpolated_normalX;
        last_x = interpolated_normalX;
        const double interpolated_normalY = coarse_grid.atPosition("normal_y", pos, grid_map::InterpolationMethods::INTER_NEAREST);
        interpolate_grid.at("normal_y", index) = interpolated_normalY;
        last_y = interpolated_normalY;
        const double interpolated_normalZ = coarse_grid.atPosition("normal_z", pos, grid_map::InterpolationMethods::INTER_NEAREST);
        interpolate_grid.at("normal_z", index) = interpolated_normalZ;
        last_z = interpolated_normalZ;

        if(coarse_index(0) == coarse_size(0)-1 && coarse_index(1) == coarse_size(1)-1) break;
    }

}


void GridTransformer::windowPrune(const grid_map::GridMap& full_grid){
    grid_map::Position robot_position(robot_pose_.pose.position.x, robot_pose_.pose.position.y);
    //
    // std::cout << "Robot Position: " << robot_position.x() << ", " << robot_position.y() << std::endl;
    
    grid_map::Position min_bound, max_bound;
    full_grid.getPosition(grid_map::Index(0, 0), min_bound);
    full_grid.getPosition(grid_map::Index(full_grid.getSize()(0) - 1, full_grid.getSize()(1) - 1), max_bound);
    // std::cout << "Grid Map Bounds: [" << min_bound.x() << ", " << min_bound.y() << "] to [" << max_bound.x() << ", " << max_bound.y() << "]" << std::endl;
    //
    bool isSucess = false;
    *sub_gridmap_ = full_grid.getSubmap(robot_position, grid_map::Length(window_length_, window_length_), isSucess);
    if(!isSucess){
        ROS_WARN("Cannot get submap!\n");
    }
    sub_gridmap_->setPosition(grid_map::Position(robot_pose_.pose.position.x, robot_pose_.pose.position.y));
    sub_gridmap_->setFrameId(global_frame_);
}

void GridTransformer::PCLOctomapConversion(const PointCloud<PointType> &pt_in, octomap::OcTree &octree_out){
    // octomap::OcTree tree( octree_resolution_ );
    if(!pt_in.empty()){
        octree_out.clear();
        for (auto p:pt_in.points)
        {
            // Insert octomap
            octree_out.updateNode( octomap::point3d(p.x, p.y, p.z), true);
        }
        // Update octomap
        octree_out.updateInnerOccupancy();
    }
}

void GridTransformer::publishOctomap(const octomap::OcTree &full_octree){
    octomap_msgs::Octomap octomapMessage;
    octomap_msgs::fullMapToMsg(full_octree, octomapMessage);
    octomapMessage.header.frame_id = global_frame_;
    octomapMessage.header.stamp = ros::Time::now();
    octomap_pub_.publish(octomapMessage);
}

void GridTransformer::OctomapGridMapConversion(const octomap::OcTree &octree_in, grid_map::GridMap &grid_out){

    grid_map::Position3 min_bound;
    grid_map::Position3 max_bound;
    octree_in.getMetricMin(min_bound(0), min_bound(1), min_bound(2));
    octree_in.getMetricMax(max_bound(0), max_bound(1), max_bound(2));

    bool res = grid_map::GridMapOctomapConverter::fromOctomap(octree_in, "elevation", grid_out, &min_bound, &max_bound);
    if (!res) {
        ROS_ERROR("Failed to call convert Octomap.");
        return;
    }
    grid_out.setFrameId(global_frame_);
}

void GridTransformer::publishGridMap(const ros::Publisher &ros_publisher, grid_map::GridMap &full_gridmap){
    // Publish as grid map.
    grid_map_msgs::GridMap full_gridmap_msg_;
    grid_map::GridMapRosConverter::toMessage(full_gridmap, full_gridmap_msg_);
    ros_publisher.publish(full_gridmap_msg_);
}

void GridTransformer::publishHandler(const ros::TimerEvent& e){
    if(!system_initialized_ && !raw_pts_->empty()){
        gridGenerate();
        interpolateSmoothing(*full_gridmap_, *interpolated_gridmap_);
        printf("Grid Generation Finished!\n");
        // //! Octomap method
        // // Convert to octomap
        // PCLOctomapConversion(*raw_pts_, *full_octree_);
        // // Convert to grid map
        // OctomapGridMapConversion(*full_octree_, *full_gridmap_);
        system_initialized_ = true;
    }
    // Publish item
    // publishOctomap(*full_octree_);
    // full_grid_pub_.publish(full_gridmap_msg_);
    publishGridMap(full_grid_pub_, *interpolated_gridmap_);
}

void GridTransformer::publishLocalHandler(const ros::TimerEvent& e){
    if(system_initialized_ && !raw_pts_->empty()){
        windowPrune(*interpolated_gridmap_);
        // Publish item
        publishGridMap(subgrid_pub_, *sub_gridmap_);
        // printf("Published submap!\n");
    }
}