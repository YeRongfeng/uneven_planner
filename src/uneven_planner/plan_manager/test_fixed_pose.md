# 固定位姿功能测试指南

## 问题修复

原来的 `.nan` 语法在YAML中解析有问题，现在改用 `-999.0` 作为"不固定"的标记值。

## 测试步骤

### 1. 测试固定起始点朝向

编辑 `params/data_generation.yaml`：

```yaml
manager:
  start_fixed: [-999.0, -999.0, 1.57]  # 固定起始朝向为π/2
  end_fixed: [-999.0, -999.0, 0.0]     # 固定目标朝向为0
```

运行测试：
```bash
roslaunch plan_manager data_generation.launch map_name:=desert path_num:=5
```

### 2. 预期日志输出

启动时应该看到：
```
[INFO] Start pose constraints: x=random, y=random, yaw=1.570000
[INFO] End pose constraints: x=random, y=random, yaw=0.000000
```

生成位姿时应该看到：
```
[INFO] Generated pose pair (attempt 1): Start=[X.XXX, Y.YYY, 1.570], Target=[X.XXX, Y.YYY, 0.000], Distance=Z.ZZZ
```

### 3. 验证要点

- 起始点的yaw应该始终是1.57（约90度）
- 目标点的yaw应该始终是0.0
- x和y坐标应该是随机的
- 距离应该在3.0-5.0米范围内（根据当前配置）

### 4. 测试不同配置

#### 完全固定起始点
```yaml
start_fixed: [0.0, 0.0, 0.0]  # 固定在原点，朝向东
```

#### 只固定目标点位置
```yaml
end_fixed: [5.0, 2.0, -999.0]  # 固定在(5,2)，朝向随机
```

#### 禁用固定位姿
```yaml
# start_fixed: [-999.0, -999.0, 1.57]  # 注释掉
# end_fixed: [-999.0, -999.0, 0.0]     # 注释掉
```

## 故障排除

### 如果固定位姿不生效

1. 检查日志中的"pose constraints"信息
2. 确认参数文件中使用的是 `-999.0` 而不是 `.nan`
3. 确认参数没有被注释掉
4. 重新编译并重启节点

### 如果生成的位姿不符合预期

1. 检查距离约束是否过于严格
2. 检查地图边界是否合理
3. 增加max_attempts来提高成功率

## 调试技巧

1. 设置 `path_num: 1` 进行单次测试
2. 观察日志中的"Generated pose pair"信息
3. 检查固定的分量是否确实保持不变
4. 验证随机分量是否在合理范围内变化
