#include "uneven_map/uneven_map.h"

namespace uneven_planner
{
    RXS2 UnevenMap::filter(Eigen::Vector3d pos, vector<Eigen::Vector3d> points)
    {
        RXS2 rs2;

        Eigen::Vector3d mean_points = Eigen::Vector3d::Zero();
        for (size_t i=0; i<points.size(); i++)
            mean_points+=points[i];

        mean_points /= (double)points.size();
        rs2.z = mean_points.z();
        if (points.size() < 3 || !mean_points.allFinite())
        {
            rs2.sigma = 1.0;
            rs2.zb.setZero();
            return rs2;
        }

        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        for (size_t i=0; i<points.size(); i++)
        {
            Eigen::Vector3d v = points[i] - mean_points;
            cov += v * v.transpose();
        }
        cov /= (double)points.size();
        Eigen::EigenSolver<Eigen::Matrix3d> es(cov);
        Eigen::Matrix<double, 3, 1> D = es.pseudoEigenvalueMatrix().diagonal();
        Eigen::Matrix3d V = es.pseudoEigenvectors();
        Eigen::MatrixXd::Index evalsMax;
        D.minCoeff(&evalsMax);
        Eigen::Matrix<double, 3, 1> n = V.col(evalsMax);
        n.normalize();
        if (n(2, 0) < 0.0)
            n = -n;
        
        const double eigenvalue_sum = D.sum();
        rs2.sigma = D(evalsMax) / eigenvalue_sum * 3.0;
        if (!std::isfinite(rs2.sigma) || !n.allFinite() ||
            std::fabs(eigenvalue_sum) < 1e-12)
        {
            rs2.sigma = 1.0;
            n = Eigen::Vector3d(0.0, 0.0, 1.0);
        }
        rs2.zb.x() = n(0, 0);
        rs2.zb.y() = n(1, 0);

        return rs2;
    }

    Eigen::Matrix3d UnevenMap::skewSym(Eigen::Vector3d vec)
    {
        Eigen::Matrix3d skem_sym;
        skem_sym << 0.0    , -vec(2), vec(1) , \
                    vec(2) , 0.0    , -vec(0), \
                    -vec(1), vec(0) , 0.0       ;
        return skem_sym;
    }

    // zb, yb = (zb x xyaw).normalized(), xb = yb x zb
    // using Sherman-Morrison formula
    double UnevenMap::calYawFromR(Eigen::Matrix3d R)
    {
        Eigen::Vector2d p(R(0, 2), R(1, 2));
        Eigen::Vector2d b(R(0, 0), R(1, 0));
        Eigen::Vector2d x = (Eigen::Matrix2d::Identity()+p*p.transpose()/(1.0-p.squaredNorm()))*b;
        return atan2(x(1), x(0));
    }

    void UnevenMap::normSO2(double& yaw)
    {
        while (yaw < -M_PI)
            yaw += 2*M_PI;
        while (yaw > M_PI)
            yaw -= 2*M_PI;
        return;
    }

    void UnevenMap::init(ros::NodeHandle& nh)
    {
        nh.getParam("uneven_map/iter_num", iter_num);
        nh.getParam("uneven_map/map_size_x", map_size[0]);
        nh.getParam("uneven_map/map_size_y", map_size[1]);
        nh.getParam("uneven_map/ellipsoid_x", ellipsoid_x);
        nh.getParam("uneven_map/ellipsoid_y", ellipsoid_y);
        nh.getParam("uneven_map/ellipsoid_z", ellipsoid_z);
        nh.getParam("uneven_map/xy_resolution", xy_resolution);
        nh.getParam("uneven_map/yaw_resolution", yaw_resolution);
        nh.getParam("uneven_map/min_cnormal", min_cnormal);
        nh.getParam("uneven_map/max_rho", max_rho);
        nh.getParam("uneven_map/gravity", gravity);
        nh.getParam("uneven_map/mass", mass);
        nh.getParam("uneven_map/map_pcd", pcd_file);
        nh.getParam("uneven_map/map_file", map_file);
        // read option: use external occupancy map via topic
        nh.param<bool>("uneven_map/use_external_occ_subscriber", use_external_occ_subscriber, false);
        nh.param<int>("uneven_map/occ_threshold", occ_threshold, 50);
        origin_pub = nh.advertise<sensor_msgs::PointCloud2>("/origin_map", 1);
        filtered_pub = nh.advertise<sensor_msgs::PointCloud2>("/filtered_map", 1);
        zb_pub = nh.advertise<visualization_msgs::Marker>("/zb_map", 1);
        so2_test_pub = nh.advertise<visualization_msgs::MarkerArray>("/so2_map", 1);
        // NOTE: delay creating the occupancy subscriber until internal buffers are allocated
        // to avoid early callbacks that access uninitialized buffers.
        vis_timer = nh.createTimer(ros::Duration(1.0), &UnevenMap::visCallback, this);
        
        // size
        map_size[2] = 2.0 * M_PI + 5e-2;
        
        // origin and boundary
        min_boundary = -map_size / 2.0;
        max_boundary = map_size / 2.0;
        map_origin = min_boundary;

        // resolution
        xy_resolution_inv = 1.0 / xy_resolution;
        yaw_resolution_inv = 1.0 / yaw_resolution;

        // voxel num
        voxel_num(0) = ceil(map_size(0) / xy_resolution);
        voxel_num(1) = ceil(map_size(1) / xy_resolution);
        voxel_num(2) = ceil(map_size(2) / yaw_resolution);

        // idx
        min_idx = Eigen::Vector3i::Zero();
        max_idx = voxel_num - Eigen::Vector3i::Ones();

        // datas
        int buffer_size  = voxel_num(0) * voxel_num(1) * voxel_num(2);
        map_buffer = vector<RXS2>(buffer_size, RXS2());
        c_buffer   = vector<double>(buffer_size, 1.0);
        occ_buffer = vector<char>(buffer_size, 0);
        occ_r2_buffer = vector<char>(getXYNum(), 0);
        // create external occupancy subscriber only after buffers allocated
        if (use_external_occ_subscriber)
        {
            std::string occ_topic;
            nh.param<std::string>("uneven_map/occ_topic", occ_topic, std::string("/external_occ_grid_hwy"));
            // subscribe to Float32MultiArray 3D HWY message
            occ_sub = nh.subscribe<std_msgs::Float32MultiArray>(occ_topic, 1, &UnevenMap::occMapCallback, this);
            ROS_INFO("UnevenMap: subscribed to external occupancy subscriber on %s", occ_topic.c_str());
        }
        world_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
        world_cloud_plane.reset(new pcl::PointCloud<pcl::PointXY>());
        pcl::PointCloud<pcl::PointXY>::Ptr world_cloud_temp;

        // world cloud process
        pcl::PointCloud<pcl::PointXYZ> cloudMapOrigin;
        pcl::PointCloud<pcl::PointXYZ> cloudMapClipper;

        if (pcd_file.empty())
        {
            ROS_FATAL("UnevenMap: uneven_map/map_pcd is empty; refusing to initialize an empty KD-tree");
            throw std::runtime_error("UnevenMap map_pcd parameter is empty");
        }

        pcl::PCDReader reader;
        const int read_result = reader.read<pcl::PointXYZ>(pcd_file, cloudMapOrigin);
        if (read_result < 0 || cloudMapOrigin.empty())
        {
            ROS_FATAL("UnevenMap: failed to read a non-empty point cloud from '%s'", pcd_file.c_str());
            throw std::runtime_error("UnevenMap failed to load point cloud");
        }

        pcl::CropBox<pcl::PointXYZ> clipper;
        clipper.setMin(Eigen::Vector4f(-10.0, -10.0, -0.01, 1.0));
        clipper.setMax(Eigen::Vector4f(10.0, 10.0, 5.0, 1.0));
        clipper.setInputCloud(cloudMapOrigin.makeShared());
        clipper.filter(cloudMapClipper);
        cloudMapOrigin.clear();

        pcl::VoxelGrid<pcl::PointXYZ> dwzFilter;
        dwzFilter.setLeafSize(0.01, 0.01, 0.01);
        dwzFilter.setInputCloud(cloudMapClipper.makeShared());
        dwzFilter.filter(*world_cloud);
        cloudMapClipper.clear();
        if (world_cloud->empty())
        {
            ROS_FATAL("UnevenMap: point cloud '%s' has no points inside the configured crop box", pcd_file.c_str());
            throw std::runtime_error("UnevenMap point cloud is empty after filtering");
        }

        for (size_t i=0; i<world_cloud->points.size(); i++)
        {
            pcl::PointXY p;
            p.x = world_cloud->points[i].x;
            p.y = world_cloud->points[i].y;
            world_cloud_plane->points.emplace_back(p);
        }
        world_cloud->width = world_cloud->points.size();
        world_cloud->height = 1;
        world_cloud->is_dense = true;
        world_cloud->header.frame_id = "world";
        world_cloud_plane->width = world_cloud_plane->points.size();
        world_cloud_plane->height = 1;
        world_cloud_plane->is_dense = true;
        world_cloud_plane->header.frame_id = "world";
        kd_tree.setInputCloud(world_cloud);
        kd_tree_plane.setInputCloud(world_cloud_plane);
        pcl::toROSMsg(*world_cloud, origin_cloud_msg);

        // construct map: SO(2) --> RXS2
        if (!constructMapInput())
            constructMap();
        
        // occ map: either generate internally or rely on external occupancy subscriber
        if (!use_external_occ_subscriber)
        {
            for (int x=0; x<voxel_num[0]; x++)
                for (int y=0; y<voxel_num[1]; y++)
                    for (int yaw=0; yaw<voxel_num[2]; yaw++)
                    {
                        if (c_buffer[toAddress(x, y, yaw)] < min_cnormal || map_buffer[toAddress(x, y, yaw)].sigma > max_rho)
                        {
                            occ_buffer[toAddress(x, y, yaw)] = 1;
                            occ_r2_buffer[x*voxel_num(1)+y] = 1;
                        }
                    }
        }

        //  to pcl and marker msg
        zb_msg.type = visualization_msgs::Marker::LINE_LIST;
        zb_msg.header.frame_id = "world";
        zb_msg.pose.orientation.w = 1.0;
        zb_msg.scale.x = 0.006;
        zb_msg.color.a = 0.6;
        geometry_msgs::Point p1, p2;
        
        pcl::PointCloud<pcl::PointXYZI> grid_map_filtered;
        pcl::PointXYZI pt_filtered;
        int yaw = floor(M_2_PI*yaw_resolution_inv);
        for (int x=0; x<voxel_num[0]; x++)
            for (int y=0; y<voxel_num[1]; y++)
            {
                if (occ_buffer[toAddress(x, y, yaw)]==1)
                    continue;
                Eigen::Vector3d filtered_p;
                RXS2 rs2 = map_buffer[toAddress(x, y, yaw)];
                double c = c_buffer[toAddress(x, y, yaw)];
                indexToPos(Eigen::Vector3i(x, y, yaw), filtered_p);
                p1.x = pt_filtered.x = filtered_p.x();
                p1.y = pt_filtered.y = filtered_p.y();
                p1.z = pt_filtered.z = rs2.z;
                pt_filtered.intensity = rs2.sigma;
                grid_map_filtered.emplace_back(pt_filtered);

                p2.x = p1.x + 1.5 * xy_resolution * rs2.zb.x();
                p2.y = p1.y + 1.5 * xy_resolution * rs2.zb.y();
                p2.z = p1.z + 1.5 * xy_resolution * c;
                if (x%2==0 && y%2==0)
                {
                    zb_msg.points.emplace_back(p1);
                    zb_msg.points.emplace_back(p2);
                }
            }
        grid_map_filtered.width = grid_map_filtered.points.size();
        grid_map_filtered.height = 1;
        grid_map_filtered.is_dense = true;
        grid_map_filtered.header.frame_id = "world";
        pcl::toROSMsg(grid_map_filtered, filtered_cloud_msg);

        // so2_test_msg
        visualization_msgs::Marker so2_line;
        visualization_msgs::Marker so2_point;
        so2_line.id = 0;
        so2_line.type = visualization_msgs::Marker::LINE_LIST;
        so2_line.header.frame_id = "world";
        so2_line.pose.orientation.w = 1.0;
        so2_line.scale.x = 0.01;
        so2_line.color.a = 0.6;
        so2_point.id = 1;
        so2_point.type = visualization_msgs::Marker::POINTS;
        so2_point.header.frame_id = "world";
        so2_point.pose.orientation.w = 1.0;
        so2_point.scale.x = 0.015;
        so2_point.scale.y = 0.015;
        so2_point.color.a = 1.0;
        so2_point.color.r = 0.8;
        geometry_msgs::Point p0;
        double r_res = 0.8;
        int ri_res = floor(r_res * xy_resolution_inv);
        for (int x=0; x<voxel_num[0]; x+=ri_res)
            for (int y=0; y<voxel_num[1]; y+=ri_res)
                for (int yaw=0; yaw<voxel_num[2]; yaw++)
                {
                    Eigen::Vector3d filtered_p;
                    RXS2 rs2 = map_buffer[toAddress(x, y, yaw)];
                    indexToPos(Eigen::Vector3i(x, y, yaw), filtered_p);
                    p1.x = p0.x = filtered_p.x() + r_res / 2.5 * cos(filtered_p.z());
                    p1.y = p0.y = filtered_p.y() + r_res / 2.5 * sin(filtered_p.z());
                    Eigen::Vector3d zb(rs2.zb(0), rs2.zb(1), c_buffer[toAddress(x, y, yaw)]);
                    Eigen::Vector3d xyaw(cos(filtered_p.z()), sin(filtered_p.z()), 0.0);
                    Eigen::Vector3d yb = zb.cross(xyaw).normalized();
                    Eigen::Vector3d xb = yb.cross(zb);
                    p1.z = p0.z = rs2.z - xb(2) * 0.12;
                    so2_point.points.emplace_back(p0);

                    p2.x = p1.x + 1.5 * xy_resolution * rs2.zb.x();
                    p2.y = p1.y + 1.5 * xy_resolution * rs2.zb.y();
                    p2.z = p1.z + 1.5 * xy_resolution * c_buffer[toAddress(x, y, yaw)];
                    so2_line.points.emplace_back(p1);
                    so2_line.points.emplace_back(p2);
                }
        so2_test_msg.markers.emplace_back(so2_line);
        so2_test_msg.markers.emplace_back(so2_point);

        map_ready = true;
    }

    bool UnevenMap::decodeExternalOccupancy(
        const std_msgs::Float32MultiArray& msg,
        vector<char>& decoded_occ,
        vector<char>& decoded_occ_r2,
        size_t& occupied_voxel_count,
        size_t& occupied_xy_count,
        string& error)
    {
        const int vx = voxel_num(0);
        const int vy = voxel_num(1);
        const int vz = voxel_num(2);
        if (vx <= 0 || vy <= 0 || vz <= 0)
        {
            error = "UnevenMap dimensions are not initialized";
            return false;
        }

        if (msg.layout.dim.size() != 3)
        {
            error = "occupancy_hwy layout must have exactly [height,width,yaw]";
            return false;
        }
        if (msg.layout.data_offset != 0)
        {
            error = "occupancy_hwy data_offset must be zero";
            return false;
        }

        const int msg_h = static_cast<int>(msg.layout.dim[0].size);
        const int msg_w = static_cast<int>(msg.layout.dim[1].size);
        const int msg_yaw_bins =
            static_cast<int>(msg.layout.dim[2].size);
        if (msg.layout.dim[0].label != "height" ||
            msg.layout.dim[1].label != "width" ||
            msg.layout.dim[2].label != "yaw")
        {
            error = "occupancy_hwy labels must be height,width,yaw";
            return false;
        }
        if (msg_h != vy || msg_w != vx || msg_yaw_bins <= 0)
        {
            std::ostringstream oss;
            oss << "occupancy_hwy shape must be H=" << vy
                << ", W=" << vx << ", Y>0; got H=" << msg_h
                << ", W=" << msg_w << ", Y=" << msg_yaw_bins;
            error = oss.str();
            return false;
        }
        const size_t expected =
            static_cast<size_t>(msg_h) * static_cast<size_t>(msg_w) *
            static_cast<size_t>(msg_yaw_bins);
        if (msg.data.size() != expected)
        {
            std::ostringstream oss;
            oss << "occupancy_hwy data length " << msg.data.size()
                << " does not match H*W*Y=" << expected;
            error = oss.str();
            return false;
        }
        if (msg.layout.dim[0].stride !=
                static_cast<uint32_t>(msg_w * msg_yaw_bins) ||
            msg.layout.dim[1].stride !=
                static_cast<uint32_t>(msg_yaw_bins) ||
            msg.layout.dim[2].stride != 1)
        {
            error = "occupancy_hwy strides must be W*Y,Y,1";
            return false;
        }
        for (size_t i = 0; i < msg.data.size(); ++i)
        {
            const float value = msg.data[i];
            if (!std::isfinite(value) || value < 0.0f || value > 1.0f)
            {
                std::ostringstream oss;
                oss << "occupancy_hwy[" << i
                    << "] must be finite and in [0,1]";
                error = oss.str();
                return false;
            }
        }

        decoded_occ.assign(
            static_cast<size_t>(vx) * static_cast<size_t>(vy) *
                static_cast<size_t>(vz),
            0);
        decoded_occ_r2.assign(
            static_cast<size_t>(vx) * static_cast<size_t>(vy), 0);
        occupied_voxel_count = 0;
        occupied_xy_count = 0;
        const double threshold =
            static_cast<double>(occ_threshold) / 100.0;

        // Python flattens C-order (H=row=y, W=column=x, Y=yaw).
        // Fill every internal yaw bin by looking up the source bin containing
        // that bin's physical center. This is periodic and also fills the
        // receiver's 64th bin when the sender has ceil(2*pi/0.1)=63 bins.
        for (int x = 0; x < vx; ++x)
        {
            for (int y = 0; y < vy; ++y)
            {
                bool xy_occupied = false;
                for (int dst_yaw = 0; dst_yaw < vz; ++dst_yaw)
                {
                    double theta =
                        map_origin(2) +
                        (static_cast<double>(dst_yaw) + 0.5) *
                            yaw_resolution;
                    double phase = std::fmod(theta + M_PI, 2.0 * M_PI);
                    if (phase < 0.0)
                        phase += 2.0 * M_PI;
                    int src_yaw = static_cast<int>(std::floor(
                        phase * static_cast<double>(msg_yaw_bins) /
                        (2.0 * M_PI)));
                    src_yaw = std::max(
                        0, std::min(src_yaw, msg_yaw_bins - 1));

                    const size_t src_addr =
                        (static_cast<size_t>(y) *
                             static_cast<size_t>(msg_w) +
                         static_cast<size_t>(x)) *
                            static_cast<size_t>(msg_yaw_bins) +
                        static_cast<size_t>(src_yaw);
                    const bool occupied =
                        static_cast<double>(msg.data[src_addr]) >= threshold;
                    const int dst_addr = toAddress(x, y, dst_yaw);
                    decoded_occ[static_cast<size_t>(dst_addr)] =
                        occupied ? 1 : 0;
                    if (occupied)
                    {
                        ++occupied_voxel_count;
                        xy_occupied = true;
                    }
                }
                const size_t xy_addr =
                    static_cast<size_t>(x) * static_cast<size_t>(vy) +
                    static_cast<size_t>(y);
                decoded_occ_r2[xy_addr] = xy_occupied ? 1 : 0;
                if (xy_occupied)
                    ++occupied_xy_count;
            }
        }
        return true;
    }

    void UnevenMap::occMapCallback(
        const std_msgs::Float32MultiArrayConstPtr& msg)
    {
        vector<char> decoded_occ;
        vector<char> decoded_occ_r2;
        size_t occupied_voxel_count = 0;
        size_t occupied_xy_count = 0;
        string error;
        if (!decodeExternalOccupancy(
                *msg, decoded_occ, decoded_occ_r2,
                occupied_voxel_count, occupied_xy_count, error))
        {
            ROS_ERROR("Rejected external occupancy update: %s", error.c_str());
            return;
        }
        {
            std::lock_guard<std::mutex> lk(occ_mutex);
            occ_buffer.swap(decoded_occ);
            occ_r2_buffer.swap(decoded_occ_r2);
        }
        ROS_INFO(
            "Applied external occupancy: %zu SE(2) voxels, %zu XY cells",
            occupied_voxel_count, occupied_xy_count);
    }

    bool UnevenMap::replaceExternalMap(
        const sensor_msgs::PointCloud2& cloud_msg,
        const std_msgs::Float32MultiArray& occupancy_hwy,
        double min_x,
        double min_y,
        double max_x,
        double max_y,
        double resolution,
        size_t& point_count,
        size_t& occupied_voxel_count,
        size_t& occupied_xy_count,
        string& error)
    {
        const double tolerance = 1e-6;
        if (std::fabs(min_x - min_boundary(0)) > tolerance ||
            std::fabs(min_y - min_boundary(1)) > tolerance ||
            std::fabs(max_x - max_boundary(0)) > tolerance ||
            std::fabs(max_y - max_boundary(1)) > tolerance ||
            std::fabs(resolution - xy_resolution) > tolerance)
        {
            std::ostringstream oss;
            oss << "map contract mismatch: expected bounds ["
                << min_boundary(0) << "," << max_boundary(0) << "]x["
                << min_boundary(1) << "," << max_boundary(1)
                << "] at resolution " << xy_resolution;
            error = oss.str();
            return false;
        }
        if (!cloud_msg.header.frame_id.empty() &&
            cloud_msg.header.frame_id != "map" &&
            cloud_msg.header.frame_id != "world")
        {
            error = "pointcloud frame_id must be map or world";
            return false;
        }

        vector<char> decoded_occ;
        vector<char> decoded_occ_r2;
        if (!decodeExternalOccupancy(
                occupancy_hwy, decoded_occ, decoded_occ_r2,
                occupied_voxel_count, occupied_xy_count, error))
        {
            return false;
        }

        pcl::PointCloud<pcl::PointXYZ> input_cloud;
        try
        {
            pcl::fromROSMsg(cloud_msg, input_cloud);
        }
        catch (const std::exception& exc)
        {
            error = string("failed to decode PointCloud2: ") + exc.what();
            return false;
        }
        vector<int> finite_indices;
        pcl::removeNaNFromPointCloud(
            input_cloud, input_cloud, finite_indices);
        if (input_cloud.empty())
        {
            error = "pointcloud contains no finite XYZ points";
            return false;
        }

        const double box_r =
            std::max(std::max(ellipsoid_x, ellipsoid_y), ellipsoid_z);
        pcl::PointCloud<pcl::PointXYZ> clipped_cloud;
        pcl::CropBox<pcl::PointXYZ> clipper;
        clipper.setMin(Eigen::Vector4f(
            static_cast<float>(min_x - box_r),
            static_cast<float>(min_y - box_r),
            std::numeric_limits<float>::lowest(), 1.0f));
        clipper.setMax(Eigen::Vector4f(
            static_cast<float>(max_x + box_r),
            static_cast<float>(max_y + box_r),
            std::numeric_limits<float>::max(), 1.0f));
        clipper.setInputCloud(input_cloud.makeShared());
        clipper.filter(clipped_cloud);
        if (clipped_cloud.empty())
        {
            error = "pointcloud has no points inside map bounds plus padding";
            return false;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr new_world_cloud(
            new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);
        voxel_filter.setInputCloud(clipped_cloud.makeShared());
        voxel_filter.filter(*new_world_cloud);
        if (new_world_cloud->size() < 3)
        {
            error = "pointcloud has fewer than three filtered points";
            return false;
        }

        pcl::PointCloud<pcl::PointXY>::Ptr new_world_cloud_plane(
            new pcl::PointCloud<pcl::PointXY>());
        new_world_cloud_plane->points.reserve(new_world_cloud->size());
        for (size_t i = 0; i < new_world_cloud->points.size(); ++i)
        {
            pcl::PointXY point;
            point.x = new_world_cloud->points[i].x;
            point.y = new_world_cloud->points[i].y;
            new_world_cloud_plane->points.emplace_back(point);
        }
        new_world_cloud->width = new_world_cloud->points.size();
        new_world_cloud->height = 1;
        new_world_cloud->is_dense = true;
        new_world_cloud->header.frame_id = "world";
        new_world_cloud_plane->width =
            new_world_cloud_plane->points.size();
        new_world_cloud_plane->height = 1;
        new_world_cloud_plane->is_dense = true;
        new_world_cloud_plane->header.frame_id = "world";

        // ros::spin() serializes this service with planning callbacks. Mark
        // the map unavailable before replacing live KD trees and buffers, and
        // expose it only after both terrain and occupancy are complete.
        map_ready = false;
        world_cloud = new_world_cloud;
        world_cloud_plane = new_world_cloud_plane;
        kd_tree.setInputCloud(world_cloud);
        kd_tree_plane.setInputCloud(world_cloud_plane);
        const size_t buffer_size =
            static_cast<size_t>(voxel_num(0)) *
            static_cast<size_t>(voxel_num(1)) *
            static_cast<size_t>(voxel_num(2));
        map_buffer.assign(buffer_size, RXS2());
        c_buffer.assign(buffer_size, 1.0);
        if (!constructMap(false))
        {
            error = "failed to construct terrain map from pointcloud";
            return false;
        }
        {
            std::lock_guard<std::mutex> lk(occ_mutex);
            occ_buffer.swap(decoded_occ);
            occ_r2_buffer.swap(decoded_occ_r2);
        }

        pcl::toROSMsg(*world_cloud, origin_cloud_msg);
        origin_cloud_msg.header.frame_id = "world";
        filtered_cloud_msg = sensor_msgs::PointCloud2();
        filtered_cloud_msg.header.frame_id = "world";
        zb_msg.points.clear();
        so2_test_msg.markers.clear();
        point_count = world_cloud->size();
        map_ready = true;
        return true;
    }

    bool UnevenMap::constructMapInput()
    {
        ifstream pp(map_file);
        if (!pp.good())
        {
            ROS_WARN("map file is empty, begin construct it.");
            return false;
        }
        ifstream fp;
        fp.open(map_file, ios::in);
        string idata, word;
        istringstream sin;
        vector<string> words;
        while (getline(fp, idata))
        {
            sin.clear();
            sin.str(idata);
            words.clear();
            while (getline(sin, word, ','))
            {
                words.emplace_back(word);
            }

            int x = atoi(words[0].c_str());
            int y = atoi(words[1].c_str());
            int yaw = atoi(words[2].c_str());
            double z = stold(words[3]);
            double sigma = stold(words[4]);
            double zba = stold(words[5]);
            double zbb = stold(words[6]);
            if (isInMap(Eigen::Vector3i(x, y, yaw)))
            {
                map_buffer[toAddress(x, y, yaw)] = RXS2(z, sigma, Eigen::Vector2d(zba, zbb));
                if (map_buffer[toAddress(x, y, yaw)].sigma < sigma)
                {
                    map_buffer[toAddress(x, y, yaw)].sigma = sigma;
                }
                c_buffer[toAddress(x, y, yaw)] = sqrt(1.0-zba*zba-zbb*zbb);
            }
        }
        fp.close();

        ROS_INFO("map: SO(2) --> RXS2 done.");

        return true;
    }

    bool UnevenMap::constructMap(bool persist_to_file)
    {
        const double box_r = max(max(ellipsoid_x, ellipsoid_y), ellipsoid_z);
        const Eigen::Vector3d ellipsoid_vecinv(1.0 / ellipsoid_x, 1.0 / ellipsoid_y, 1.0 / ellipsoid_z);
        int cnt=0;

        for (int x=0; x<voxel_num[0]; x++)
            for (int y=0; y<voxel_num[1]; y++)
                for (int yaw=0; yaw<voxel_num[2]; yaw++)
                    for (int iter=0; iter<iter_num; iter++)
                    {
                        Eigen::Vector3d map_pos;
                        RXS2 map_rs2 = map_buffer[toAddress(x, y, yaw)];
                        double map_c = c_buffer[toAddress(x, y, yaw)];
                        indexToPos(Eigen::Vector3i(x, y, yaw), map_pos);
                        
                        Eigen::Vector3d xyaw(cos(map_pos(2)), sin(map_pos(2)), 0.0);
                        Eigen::Vector3d zb(map_rs2.zb(0), map_rs2.zb(1), map_c);
                        Eigen::Vector3d yb = zb.cross(xyaw).normalized();
                        Eigen::Vector3d xb = yb.cross(zb);
                        Eigen::Matrix3d RT;
                        RT.row(0) = xb;
                        RT.row(1) = yb;
                        RT.row(2) = zb;
                        Eigen::Vector3d world_pos(map_pos(0), map_pos(1), map_rs2.z);
                        world_pos.head(2) += xb.head(2) * 0.12;
                        if (!RT.allFinite() || !world_pos.allFinite())
                        {
                            ROS_ERROR(
                                "Non-finite terrain query at grid (%d,%d,%d), "
                                "iteration %d; using conservative flat fallback",
                                x, y, yaw, iter);
                            RXS2 fallback;
                            fallback.z = std::isfinite(map_rs2.z) ? map_rs2.z : 0.0;
                            fallback.sigma = 1.0;
                            map_buffer[toAddress(x, y, yaw)] = fallback;
                            c_buffer[toAddress(x, y, yaw)] = 1.0;
                            continue;
                        }
                        
                        vector<int> Idxs;
                        vector<float> SquaredDists;
                        if (iter == 0)
                        {
                            pcl::PointXY pxy;
                            pxy.x = world_pos(0);
                            pxy.y = world_pos(1);
                            if (kd_tree_plane.nearestKSearch(pxy, 1, Idxs, SquaredDists) > 0)
                            {
                                world_pos(2) = world_cloud->points[Idxs[0]].z;
                            }
                        }

                        // get points and compute, update
                        vector<Eigen::Vector3d> points;
                        pcl::PointXYZ pt;
                        pt.x = world_pos(0);
                        pt.y = world_pos(1);
                        pt.z = world_pos(2);
                        if (kd_tree.radiusSearch(pt, box_r, Idxs, SquaredDists) > 0)
                        {
                            // is in ellipsoid
                            for (size_t i=0; i<Idxs.size(); i++)
                            {
                                Eigen::Vector3d temp_pos(world_cloud->points[Idxs[i]].x, \
                                                         world_cloud->points[Idxs[i]].y, \
                                                         world_cloud->points[Idxs[i]].z );
                                Eigen::Vector3d temp_subtract = temp_pos - world_pos;
                                Eigen::Vector3d temp_inrob = RT*temp_subtract;
                                if (ellipsoid_vecinv.cwiseProduct(temp_inrob).squaredNorm() < 1.0)
                                {
                                    points.emplace_back(temp_pos);
                                }
                            }
                        }
                        if (points.empty())
                        {
                            // std::cout<<"Points empty, but don't worry."<<std::endl;
                            RXS2 rxs2_z;
                            rxs2_z.z = world_pos(2);
                            map_buffer[toAddress(x, y, yaw)] = rxs2_z;
                            c_buffer[toAddress(x, y, yaw)] = map_buffer[toAddress(x, y, yaw)].getC();
                        }
                        else
                        {
                            map_buffer[toAddress(x, y, yaw)] = UnevenMap::filter(map_pos, points);
                            c_buffer[toAddress(x, y, yaw)] = map_buffer[toAddress(x, y, yaw)].getC();
                        }
                        
                        if (iter==0 && cnt++ % 100000 == 0)
                        {
                            cout<<"\033[1;33m map process "<<toAddress(x, y, yaw)*100.0 / (voxel_num[0]*voxel_num[1]*voxel_num[2])<<"%\033[0m"<<endl;
                            cnt=1;
                        }
                    }
        
        if (persist_to_file && !map_file.empty())
        {
            ofstream outf;
            outf.open(map_file, ofstream::out);
            if (!outf.good())
            {
                ROS_ERROR("Failed to open terrain map file '%s' for writing",
                          map_file.c_str());
                return false;
            }
            for (int x=0; x<voxel_num[0]; x++)
                for (int y=0; y<voxel_num[1]; y++)
                    for (int yaw=0; yaw<voxel_num[2]; yaw++)
                    {
                        RXS2 rs2 = map_buffer[toAddress(x, y, yaw)];
                        outf << x << "," << y << "," << yaw << ","
                             << rs2.z << "," << rs2.sigma << ","
                             << rs2.zb.x() << "," << rs2.zb.y() << endl;
                    }
        }

        ROS_INFO("map: SE(2) --> RXS2 done.");

        return true;
    }

    void UnevenMap::visCallback(const ros::TimerEvent& /*event*/)
    {
        if (!map_ready)
            return;
        
        origin_pub.publish(origin_cloud_msg);
        filtered_pub.publish(filtered_cloud_msg);
        zb_pub.publish(zb_msg);
        so2_test_pub.publish(so2_test_msg);
    }
}
