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