#include "plan_manager/pointcloud_to_grid_converter.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pointcloud_to_grid_converter_node");
    ros::NodeHandle nh;
    
    PointCloudToGridConverter converter(nh);
    
    ROS_INFO("PointCloud to Grid Converter Node started, waiting for service calls...");
    
    ros::spin();
    
    return 0;
}
