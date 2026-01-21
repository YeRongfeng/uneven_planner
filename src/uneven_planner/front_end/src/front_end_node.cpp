#include "front_end/kino_astar.h"
#include <ros/ros.h>

using namespace uneven_planner;

int main( int argc, char * argv[] )
{ 
    ros::init(argc, argv, "front_end_node");
    ros::NodeHandle nh("~");

    // use shared_ptrs instead of copying stack objects (std::mutex in UnevenMap is non-copyable)
    KinoAstar::Ptr kino_astar_ptr = make_shared<KinoAstar>();
    UnevenMap::Ptr uneven_map_ptr = make_shared<UnevenMap>();
    
    uneven_map_ptr->init(nh);
    kino_astar_ptr->init(nh);
    kino_astar_ptr->setEnvironment(uneven_map_ptr);

    ros::spin();

    return 0;
}