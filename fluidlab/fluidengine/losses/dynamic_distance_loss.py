import os
import torch
import numpy as np
import taichi as ti
import pickle as pkl
from sklearn.neighbors import KDTree
from fluidlab.fluidengine.simulators import MPMSimulator
from fluidlab.configs.macros import *
from fluidlab.utils.misc import *
import matplotlib.pyplot as plt
from .shapematching_loss import ShapeMatchingLoss

@ti.data_oriented
class DynamicDistanceLoss(ShapeMatchingLoss):
    def __init__(self, type, **kwargs):
        if type == 'diff':
            self.plateau_count_limit     = 5
            self.temporal_expand_speed   = 50
            self.temporal_init_range_end = 50
            self.temporal_range_type     = 'expand'
            self.plateau_thresh          = [0.01, 0.5]
            super(DynamicDistanceLoss, self).__init__(
                matching_mat=MILK_VIS, # 无所谓
                plateau_count_limit=self.plateau_count_limit,
                temporal_expand_speed=self.temporal_expand_speed,
                temporal_init_range_end=self.temporal_init_range_end,
                temporal_range_type='expand',
                plateau_thresh=self.plateau_thresh,
                **kwargs
            )

        elif type == 'default':
            self.temporal_range_type     = 'all'
            super(DynamicDistanceLoss, self).__init__(
                matching_mat=MILK_VIS,
                temporal_range_type='all',
                **kwargs
            )

    @ti.kernel
    def compute_chamfer_loss_kernel(self, s: ti.i32, f: ti.i32):
        for p in range(self.n_particles):
            self.chamfer_loss[s] += ti.pow(self.particle_x[f, p] - self.tgt_particles_x[p], 2).sum()


