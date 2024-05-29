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
from .loss import Loss

@ti.data_oriented
class GatheringLoss(Loss):
    def __init__(
            self,
            matching_mat,
            temporal_range_type     = 'expand',
            temporal_init_range_end = 50,
            plateau_count_limit     = 5,
            temporal_expand_speed   = 50,
            plateau_thresh          = [0.01, 0.5],
            **kwargs,
        ):
        super(GatheringLoss, self).__init__(**kwargs)

        self.matching_mat            = matching_mat
        self.temporal_range_type     = temporal_range_type
        self.temporal_init_range_end = temporal_init_range_end
        self.plateau_count_limit     = plateau_count_limit
        self.temporal_expand_speed   = temporal_expand_speed
        self.plateau_thresh          = plateau_thresh

    def build(self, sim):
        self.density_weight = self.weights['density']
        self.density_loss = ti.field(dtype=DTYPE_TI, shape=(self.max_loss_steps,), needs_grad=True)
        if self.temporal_range_type == 'last':
            self.temporal_range = [self.max_loss_steps-1, self.max_loss_steps]
        elif self.temporal_range_type == 'all':
            self.temporal_range = [0, self.max_loss_steps]
        elif self.temporal_range_type == 'expand':
            self.temporal_range = [0, self.temporal_init_range_end]
            self.best_loss = self.inf
            self.plateau_count = 0

        self.target_density = ti.field(dtype=DTYPE_TI_64, shape=sim.res)
        super(GatheringLoss, self).build(sim)
        self.particle_mass = self.sim.particles_i.mass
        self.grid_mass = ti.field(dtype=DTYPE_TI_64, shape=(*self.res,), needs_grad=True)
        self.grid_mass.from_numpy(self.grids)
        self.iou()
        self._target_iou = self._iou
    @ti.func
    def isnan(self, x):
        return not (x <= 0 or x >= 0)
    def iou(self):
        self._iou = self.iou_kernel()

    @ti.kernel
    def iou_kernel(self) -> ti.float64:
        ma = ti.cast(0., DTYPE_TI_64)
        mb = ti.cast(0., DTYPE_TI_64)
        I = ti.cast(0., DTYPE_TI_64)
        Ua = ti.cast(0., DTYPE_TI_64)
        Ub = ti.cast(0., DTYPE_TI_64)
        for i in ti.grouped(ti.ndrange(*self.res)):
            ti.atomic_max(ma, self.grid_mass[i])
            ti.atomic_max(mb, self.target_density[i])
            I += self.grid_mass[i] * self.target_density[i]
            Ua += self.grid_mass[i]
            Ub += self.target_density[i]
        I = I / ma / mb
        U = Ua / ma + Ub / mb

        return I / (U - I)
    def reset_grad(self):
        super(GatheringLoss, self).reset_grad()
        self.density_loss.grad.fill(0)
        
    def load_target(self, path):
        self.target = pkl.load(open(path, 'rb'))
        self.tgt_particles_x = ti.Vector.field(self.dim, dtype=DTYPE_TI, shape=self.n_particles)
        self.tgt_particles_x.from_numpy(self.target['last_pos'])
        self.grids = self.target['last_grid']
        self.target_density.from_numpy(self.grids)
        print(f'===>  Target loaded from {path}.')

    @ti.kernel
    def clear_losses(self):
        self.density_loss.fill(0)
        self.density_loss.grad.fill(0)

    def compute_step_loss(self, s, f):
        self.compute_grid_mass(f)
        self.iou()
        self.compute_density_loss_kernel(s)

        self.sum_up_loss_kernel(s)

    def compute_step_loss_grad(self, s, f):
        self.sum_up_loss_kernel.grad(s)

        self.compute_density_loss_kernel.grad(s)
        self.compute_grid_mass_grad(f)

    def compute_grid_mass(self, f):
        self.grid_mass.fill(0)
        self.grid_mass.grad.fill(0)
        self.compute_grid_mass_kernel(f)

    def compute_grid_mass_grad(self, f):
        self.compute_grid_mass_kernel.grad(f)

    @ti.kernel
    def compute_grid_mass_kernel(self, f: ti.i32):
        for p in range(self.n_particles):
            if not (self.isnan(self.particle_x[f, p][0]) and self.isnan(self.particle_x[f, p][1]) and self.isnan(
                    self.particle_x[f, p][2])):
                base = (self.particle_x[f, p] * self.sim.inv_dx - 0.5).cast(int)
                fx = self.particle_x[f, p] * self.sim.inv_dx - base.cast(DTYPE_TI)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                for offset in ti.static(ti.grouped(self.sim.stencil_range())):
                    weight = ti.cast(1.0, DTYPE_TI)
                    for d in ti.static(range(self.dim)):
                        weight *= w[offset[d]][d]
                    self.grid_mass[base + offset] += weight * self.particle_mass[p]

    @ti.kernel
    def compute_density_loss_kernel(self, s: ti.i32):
        for I in ti.grouped(ti.ndrange(*self.res)):
            self.density_loss[s] += ti.abs(self.grid_mass[I] - self.target_density[I])
    @ti.kernel
    def sum_up_loss_kernel(self, s: ti.i32):
        self.step_loss[s] += self.density_loss[s] * self.density_weight

    @ti.kernel
    def compute_total_loss_kernel(self, s_start: ti.i32, s_end: ti.i32):
        for s in range(s_start, s_end):
            self.total_loss[None] += self.step_loss[s]

    def get_final_loss(self):
        self.compute_total_loss_kernel(self.temporal_range[0], self.temporal_range[1])
        self.expand_temporal_range()
        
        loss_info = {
            'loss': self.total_loss[None],
            'last_step_loss': self.step_loss[self.max_loss_steps-1],
            'iou': self._iou,
            'target_iou': self._target_iou,
            'temporal_range': self.temporal_range[1],
        }

        return loss_info

    def get_final_loss_grad(self):
        self.compute_total_loss_kernel.grad(self.temporal_range[0], self.temporal_range[1])

    def expand_temporal_range(self):
        if self.temporal_range_type == 'expand':
            loss_improved = (self.best_loss - self.total_loss[None])
            loss_improved_rate = loss_improved / self.best_loss
            if loss_improved_rate < self.plateau_thresh[0] or loss_improved < self.plateau_thresh[1]:
                self.plateau_count += 1
                print('Plateaued!!!', self.plateau_count)
            else:
                self.plateau_count = 0

            if self.best_loss > self.total_loss[None]:
                self.best_loss = self.total_loss[None]

            if self.plateau_count >= self.plateau_count_limit:
                self.plateau_count = 0
                self.best_loss = self.inf

                self.temporal_range[1] = min(self.max_loss_steps, self.temporal_range[1] + self.temporal_expand_speed)
                print(f'temporal range expanded to {self.temporal_range}')
            