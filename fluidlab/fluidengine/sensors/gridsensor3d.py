import numpy as np
import taichi as ti
import torch
from fluidlab.configs.macros import *
from pyquaternion import Quaternion
from .gridsensor import GridSensor

from fluidlab.utils.geom import quaternion_to_rotation_matrix

@ti.data_oriented
class GridSensor3D(GridSensor):
    def __init__(self, cell_arc, lat_angle_north, lat_angle_south, lon_angle,
                 max_distance, min_distance, distance_normalization, target_file=None, **kwargs):
        super(GridSensor3D, self).__init__(**kwargs)
        '''
        CellScale: 网格尺寸
        GridSize: 网格检测范围（cellArc latAngleSouth latAngleNorth LonAngle maxDistance minDistance DistanceNormalization）
        RotateWithAgent: 是否随Agent旋转
        agent: Agent
        AgentID: effetor ID
        DetectableTags: 检测物体body tuple
        MaxColliderBufferSize: 最大检测物数量
        DebugColors: 颜色显示，用于debug
        GizmoZOffset: 沿着Z偏移的尺寸
        DataType: 数据类型 目前支持one-hot

        '''
        # Geometry
        self.m_CellArc = cell_arc
        self.m_LatAngleNorth = lat_angle_north
        self.m_LatAngleSouth = lat_angle_south
        self.m_LonAngle = lon_angle
        self.m_MaxDistance = max_distance
        self.m_MinDistance = min_distance
        self.m_DistanceNormalization = distance_normalization
        self.M = (self.m_LonAngle // self.m_CellArc) * 2 # gridsensor m
        self.N = (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc # gridsensor n
        self.target_file = target_file

    def build(self, sim):
        particle_state = ti.types.struct(
            relative_x=ti.types.vector(sim.dim, DTYPE_TI),
            rotated_x=ti.types.vector(sim.dim, DTYPE_TI),
            latitudes=DTYPE_TI,
            longitudes=DTYPE_TI,
            distance=DTYPE_TI
        )

        grid_state = ti.types.struct(
            distance=DTYPE_TI,
            one_hot=DTYPE_TI
        )

        super().build(sim, particle_state, grid_state)

    @ti.kernel
    def transform_point_particle(self, s: ti.i32, f:ti.i32):
        # 计算point相对agent位置
        for p in range(self.n_particles):
            self.particles[s, p].relative_x[0] = self.particle_x[f, p][0] - self.agent.effectors[0].pos[f][0]
            self.particles[s, p].relative_x[1] = self.particle_x[f, p][1] - self.agent.effectors[0].pos[f][1]
            self.particles[s, p].relative_x[2] = self.particle_x[f, p][2] - self.agent.effectors[0].pos[f][2]

            # 获取四元数数据
            a = self.agent.effectors[0].quat[f][0]
            b = -self.agent.effectors[0].quat[f][1]
            c = -self.agent.effectors[0].quat[f][2]
            d = -self.agent.effectors[0].quat[f][3]
            rotation_matrix = quaternion_to_rotation_matrix(a, b, c, d)
            self.particles[s, p].rotated_x = rotation_matrix @ self.particles[s, p].relative_x

    @ti.kernel
    def transform_target_particle(self, s: ti.i32, f: ti.i32):
        for i in range(self.n_particles):
            self.targets[s, i].relative_x[0] = self.tgt_particles_x[i][0] - \
                                                    self.agent.effectors[0].pos[f][0]
            self.targets[s, i].relative_x[1] = self.tgt_particles_x[i][1] - \
                                                    self.agent.effectors[0].pos[f][1]
            self.targets[s, i].relative_x[2] = self.tgt_particles_x[i][2] - \
                                                    self.agent.effectors[0].pos[f][2]

            # 获取四元数数据
            a = self.agent.effectors[0].quat[f][0]
            b = -self.agent.effectors[0].quat[f][1]
            c = -self.agent.effectors[0].quat[f][2]
            d = -self.agent.effectors[0].quat[f][3]
            rotation_matrix = quaternion_to_rotation_matrix(a, b, c, d)
            self.targets[s, i].rotated_x = rotation_matrix @ self.targets[s, i].relative_x

    @ti.kernel
    def compute_lat_lon_particle(self, s: ti.i32):
        for i in range(self.n_particles):
            # 提取局部坐标系中的坐标
            x = self.particles[s, i].rotated_x[0]
            y = self.particles[s, i].rotated_x[1]
            z = self.particles[s, i].rotated_x[2]

            # 计算纬度和经度
            # 计算纬度
            self.particles[s, i].distance = ti.sqrt(x * x + y * y + z * z)
            cos_lat_rad = ti.max(ti.min(y / self.particles[s, i].distance, 1.0), -1.0)
            lat_rad = ti.acos(cos_lat_rad)
            lon_rad = ti.atan2(x, -z)

            self.particles[s, i].latitudes = lat_rad * (
                    180.0 / ti.acos(-1.0))  # acos(-1) is a way to get π in Taichi
            self.particles[s, i].longitudes = lon_rad * (180.0 / ti.acos(-1.0))

    @ti.kernel
    def compute_lat_lon_target(self, s: ti.i32):
        for i in range(self.n_particles):
            # 提取局部坐标系中的坐标
            x = self.targets[s, i].rotated_x[0]
            y = self.targets[s, i].rotated_x[1]
            z = self.targets[s, i].rotated_x[2]

            # 计算纬度和经度
            # 计算纬度
            self.targets[s, i].distance = ti.sqrt(x * x + y * y + z * z)
            cos_lat_rad = ti.max(ti.min(y / self.targets[s, i].distance, 1.0), -1.0)
            lat_rad = ti.acos(cos_lat_rad)
            lon_rad = ti.atan2(x, -z)

            self.targets[s, i].latitudes = lat_rad * (
                    180.0 / ti.acos(-1.0))  # acos(-1) is a way to get π in Taichi
            self.targets[s, i].longitudes = lon_rad * (180.0 / ti.acos(-1.0))

    @ti.kernel
    def normal_distance_particle(self, s: ti.i32):
        # 清空
        # 1. 判断距离是否在球体内
        for p in range(self.n_particles):
            if self.particles[s, p].distance < self.m_MaxDistance and self.particles[s, p].distance > self.m_MinDistance:
                # 2. 判断经度范围和纬度范围
                if (90 - self.particles[s, p].latitudes < self.m_LatAngleNorth and 90 - self.particles[s, p].latitudes >= 0) or \
                        (ti.abs(self.particles[s, p].latitudes - 90) < self.m_LatAngleSouth and ti.abs(self.particles[s, p].latitudes - 90) >= 0):
                    if ti.abs(self.particles[s, p].longitudes) < self.m_LonAngle:
                        # 计算加权距离
                        d = (self.particles[s, p].distance - self.m_MinDistance) / (self.m_MaxDistance - self.m_MinDistance)
                        normal_d = 0.0
                        if self.m_DistanceNormalization == 1:
                            normal_d = 1 - d
                        else:
                            normal_d = 1 - d / (self.m_DistanceNormalization + ti.abs(d)) * (
                                        self.m_DistanceNormalization + 1)
                        # 计算经纬度索引
                        longitude_index = ti.cast(
                            ti.floor((self.particles[s, p].longitudes + self.m_LonAngle) / self.m_CellArc), ti.i32)
                        latitude_index = ti.cast(
                            ti.floor(
                                 (self.particles[s, p].latitudes - (90 - self.m_LatAngleNorth)) / self.m_CellArc),
                            ti.i32)

                        # 使用 atomic_max 更新 normal_distance 的值
                        ti.atomic_max(self.grid_sensor[s, longitude_index, latitude_index, self.sim.particles_i[p].body_id].distance, normal_d)

    @ti.kernel
    def normal_distance_target(self, s: ti.i32):
        # 1. 判断距离是否在球体内
        for p in range(self.n_particles):
            if self.targets[s, p].distance < self.m_MaxDistance and self.targets[
                s, p].distance > self.m_MinDistance:
                # 2. 判断经度范围和纬度范围
                if (90 - self.targets[s, p].latitudes < self.m_LatAngleNorth and 90 - self.targets[
                    s, p].latitudes >= 0) or \
                        (ti.abs(self.targets[s, p].latitudes - 90) < self.m_LatAngleSouth and ti.abs(
                            self.targets[s, p].latitudes - 90) >= 0):
                    if ti.abs(self.targets[s, p].longitudes) < self.m_LonAngle:
                        # 计算加权距离
                        d = (self.targets[s, p].distance - self.m_MinDistance) / (
                                self.m_MaxDistance - self.m_MinDistance)
                        normal_d = 0.0
                        if self.m_DistanceNormalization == 1:
                            normal_d = 1 - d
                        else:
                            normal_d = 1 - d / (self.m_DistanceNormalization + ti.abs(d)) * (
                                    self.m_DistanceNormalization + 1)
                        # 计算经纬度索引
                        longitude_index = ti.cast(
                            ti.floor((self.targets[s, p].longitudes + self.m_LonAngle) / self.m_CellArc), ti.i32)
                        latitude_index = ti.cast(
                            ti.floor(
                                (self.targets[s, p].latitudes - (90 - self.m_LatAngleNorth)) / self.m_CellArc),
                            ti.i32)

                        # 使用 atomic_max 更新 normal_distance 的值
                        ti.atomic_max(self.grid_sensor[s, longitude_index, latitude_index, -1].distance, normal_d)
    @ti.kernel
    def get_sensor_data_kernel(self, s: ti.i32, grid_sensor: ti.types.ndarray()):
        # 这里假设 output 已经是一个正确维度和类型的 Taichi field
        for i, j, k in ti.ndrange((self.m_LonAngle // self.m_CellArc) * 2,
                                  (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc,
                                  self.n_bodies):
            grid_sensor[i, j, k] = self.grid_sensor[s, i, j, k].distance

    def step(self):
        self.transform_point_particle(self.sim.cur_step_global-1, self.sim.cur_substep_local)
        self.compute_lat_lon_particle(self.sim.cur_step_global-1)
        self.normal_distance_particle(self.sim.cur_step_global-1)

        if self.target_file is not None:
            self.transform_target_particle(self.sim.cur_step_global-1, self.sim.cur_substep_local)
            self.compute_lat_lon_target(self.sim.cur_step_global-1)
            self.normal_distance_target(self.sim.cur_step_global-1)

    def step_grad(self):
        if self.target_file is not None:
            self.normal_distance_target.grad(self.sim.cur_step_global-1)
            self.compute_lat_lon_target.grad(self.sim.cur_step_global-1)
            self.transform_target_particle.grad(self.sim.cur_step_global-1, self.sim.cur_substep_local)

        self.normal_distance_particle.grad(self.sim.cur_step_global-1)
        self.compute_lat_lon_particle.grad(self.sim.cur_step_global-1)
        self.transform_point_particle.grad(self.sim.cur_step_global-1, self.sim.cur_substep_local)

    def get_obs(self):
        grid_sensor = torch.zeros((self.M, self.N, self.n_bodies), dtype=torch.float32, device=self.device)
        self.get_sensor_data_kernel(self.sim.cur_step_global-1, grid_sensor)
        return grid_sensor

    def clear_grid_sensor(self):
        self.particles.fill(0)
        self.particles.grad.fill(0)
        if self.target_file is not None:
            self.targets.fill(0)
            self.targets.grad.fill(0)
        self.grid_sensor.fill(0)
        self.grid_sensor.grad.fill(0)

    def reset_grad(self):
        super().reset_grad()
        self.particles.grad.fill(0)
        self.targets.grad.fill(0)
        self.grid_sensor.grad.fill(0)