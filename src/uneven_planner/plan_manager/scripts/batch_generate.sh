#!/bin/bash

# 批量数据生成脚本
# 用法: ./batch_generate.sh [map_list]
# 例如: ./batch_generate.sh "desert forest mountain"

# 默认地图列表
DEFAULT_MAPS="desert forest mountain hill volcano"

# 获取地图列表
if [ $# -eq 0 ]; then
    MAPS=$DEFAULT_MAPS
else
    MAPS="$1"
fi

# 默认参数
DEFAULT_PATH_NUM=1000
DEFAULT_START_INDEX=0

echo "==================================="
echo "批量数据生成脚本"
echo "==================================="
echo "地图列表: $MAPS"
echo "每个地图生成路径数: $DEFAULT_PATH_NUM"
echo "起始索引: $DEFAULT_START_INDEX"
echo "==================================="

# 检查ROS环境
if [ -z "$ROS_PACKAGE_PATH" ]; then
    echo "错误: ROS环境未设置，请先运行 source devel/setup.bash"
    exit 1
fi

# 检查包是否存在
if ! rospack find plan_manager > /dev/null 2>&1; then
    echo "错误: plan_manager包未找到，请确保已编译"
    exit 1
fi

# 为每个地图生成数据
for map_name in $MAPS; do
    echo ""
    echo "开始为地图 '$map_name' 生成数据..."
    echo "-----------------------------------"
    
    # 检查地图文件是否存在
    map_file=$(rospack find uneven_map)/maps/${map_name}.map
    pcd_file=$(rospack find uneven_map)/maps/${map_name}.pcd
    
    if [ ! -f "$map_file" ]; then
        echo "警告: 地图文件不存在: $map_file"
        echo "跳过地图: $map_name"
        continue
    fi
    
    if [ ! -f "$pcd_file" ]; then
        echo "警告: PCD文件不存在: $pcd_file"
        echo "跳过地图: $map_name"
        continue
    fi
    
    # 启动数据生成
    echo "启动数据生成进程..."
    roslaunch plan_manager data_generation.launch \
        map_name:=$map_name \
        start_index:=$DEFAULT_START_INDEX \
        path_num:=$DEFAULT_PATH_NUM \
        export_dir:=$(rospack find plan_manager)/data \
        sample_density:=0.1 \
        publish_delay:=2.0 \
        map_x_min:=-8.0 \
        map_x_max:=8.0 \
        map_y_min:=-8.0 \
        map_y_max:=8.0 \
        min_distance:=3.0 &
    
    # 获取进程ID
    LAUNCH_PID=$!
    
    echo "数据生成进程已启动 (PID: $LAUNCH_PID)"
    echo "等待数据生成完成..."
    
    # 等待进程完成
    wait $LAUNCH_PID
    
    # 检查退出状态
    if [ $? -eq 0 ]; then
        echo "✓ 地图 '$map_name' 数据生成完成"
        
        # 检查生成的文件数量
        data_dir=$(rospack find plan_manager)/data/$map_name
        if [ -d "$data_dir" ]; then
            file_count=$(ls -1 "$data_dir"/*.p 2>/dev/null | wc -l)
            echo "  生成了 $file_count 个路径文件"
        fi
    else
        echo "✗ 地图 '$map_name' 数据生成失败"
    fi
    
    echo "-----------------------------------"
    
    # 短暂休息
    sleep 2
done

echo ""
echo "==================================="
echo "批量数据生成完成！"
echo "==================================="

# 显示总结
echo "生成结果总结:"
for map_name in $MAPS; do
    data_dir=$(rospack find plan_manager)/data/$map_name
    if [ -d "$data_dir" ]; then
        file_count=$(ls -1 "$data_dir"/*.p 2>/dev/null | wc -l)
        echo "  $map_name: $file_count 个路径文件"
    else
        echo "  $map_name: 无数据文件"
    fi
done

echo ""
echo "数据位置: $(rospack find plan_manager)/data/"
echo "使用 python3 $(rospack find plan_manager)/scripts/test_data_format.py 测试数据格式"
