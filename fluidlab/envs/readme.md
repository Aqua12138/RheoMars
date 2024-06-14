# Target File生成方法
## ShapeMatchingLoss
ShapeMatchingLoss的生成请使用命令
```shell
python fluidlab/run.py --cfg_file configs/exp_gathering.yaml --renderer_type GGUI --record --user_input
```
- 这将会生成一个包含所有帧的x数据以及对应的mat和used
- 这是一个动态的目标

## SDFDensityLoss
SDFDensityLoss的生成请使用命令
```shell
python fluidlab/run.py --cfg_file configs/exp_gathering.yaml --renderer_type GGUI --record_target_grid --user_input
```
- 使用该生成方法会生成出最后一帧的目标的grid mass作为目标grid
- 静态目标

# 训练方法
- 在gathering_env下修改target_file对应的target file名称即可
```shell
python fluidlab/run.py --cfg_file configs/exp_gathering.yaml --renderer_type GGUI --loss_type default --exp_name=gathering_diff_adam
```

# 添加Sensor
- Gridsensor3D：参考Mlagent
## 添加方法
- 在/home/zhx/Project/RheoMars/fluidlab/envs/configs目录下的agent.yaml中添加
```yaml
sensors:
  - type: GridSensor3D
    params:
      cell_arc: 2
      lat_angle_north: 90
      lat_angle_south: 90
      lon_angle: 180
      max_distance: 1
      min_distance: 0
      distance_normalization: 1
```

# 添加环境
- 定义envs环境

# ppo 训练
```shell
python fluidlab/run.py --cfg_file configs/exp_gathering_gridsensor.yaml --renderer_type GGUI --rl ppo --exp_name=test --perc_type sensor
```