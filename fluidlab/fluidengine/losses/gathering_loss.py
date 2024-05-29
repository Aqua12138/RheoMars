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

        self.sdf_loss = ti.field(dtype=DTYPE_TI_64, shape=(self.max_loss_steps,), needs_grad=True)
        self.sdf_weight = self.weights['sdf']

        self.contact_loss = ti.field(dtype=DTYPE_TI_64, shape=(self.max_loss_steps,), needs_grad=True)
        self.min_dist = ti.field(dtype=DTYPE_TI_64, shape=(self.max_loss_steps,), needs_grad=True)
        self.dist_norm = ti.field(dtype=DTYPE_TI_64, shape=(self.max_loss_steps,), needs_grad=True)
        self.contact_weight = self.weights['contact']
        self.soft_contact_loss = self.weights['is_soft_contact']

        if self.temporal_range_type == 'last':
            self.temporal_range = [self.max_loss_steps-1, self.max_loss_steps]
        elif self.temporal_range_type == 'all':
            self.temporal_range = [0, self.max_loss_steps]
        elif self.temporal_range_type == 'expand':
            self.temporal_range = [0, self.temporal_init_range_end]
            self.best_loss = self.inf
            self.plateau_count = 0

        self.target_density = ti.field(dtype=DTYPE_TI_64, shape=sim.res)
        self.target_sdf = ti.field(dtype=DTYPE_TI_64, shape=sim.res)
        self.nearest_point = ti.Vector.field(sim.dim, dtype=DTYPE_TI_64, shape=sim.res)
        self.target_sdf_copy = ti.field(dtype=DTYPE_TI_64, shape=sim.res)
        self.nearest_point_copy = ti.Vector.field(sim.dim, dtype=DTYPE_TI_64, shape=sim.res)

        self.inf = 1000
        super(GatheringLoss, self).build(sim)

        self.primitive = self.agent.effectors[0].mesh
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
        self.sdf_loss.grad.fill(0)
        self.contact_loss.grad.fill(0)
        self.min_dist.grad.fill(0)
        self.dist_norm.grad.fill(0)

    def load_target(self, path):
        self.target = pkl.load(open(path, 'rb'))
        self.tgt_particles_x = ti.Vector.field(self.dim, dtype=DTYPE_TI, shape=self.n_particles)
        self.tgt_particles_x.from_numpy(self.target['last_pos'])
        self.grids = self.target['last_grid']
        self.target_density.from_numpy(self.grids)
        self.update_target()
        print(f'===>  Target loaded from {path}.')

    # -----------------------------------------------------------
    # preprocess target to calculate sdf
    # -----------------------------------------------------------

    def update_target(self):
        self.target_sdf_copy.fill(self.inf)
        for i in range(self.n_grid * 2):
            self.update_target_sdf()
    @ti.func
    def norm(self, x, eps=1e-8):
        return ti.sqrt(x.dot(x) + eps)

    @ti.kernel
    def update_target_sdf(self):
        for I in ti.grouped(self.target_sdf):
            self.target_sdf[I] = self.inf
            grid_pos = ti.cast(I * self.dx, DTYPE_TI_64)
            if self.target_density[I] > 1e-4:  # TODO: make it configurable
                self.target_sdf[I] = 0.
                self.nearest_point[I] = grid_pos
            else:
                for offset in ti.grouped(ti.ndrange(*(((-3, 3),) * self.dim))):
                    v = I + offset
                    if v.min() >= 0 and v.max() < self.n_grid and ti.abs(offset).sum() != 0:
                        if self.target_sdf_copy[v] < self.inf:
                            nearest_point = self.nearest_point_copy[v]
                            dist = self.norm(grid_pos - nearest_point)
                            if dist < self.target_sdf[I]:
                                self.nearest_point[I] = nearest_point
                                self.target_sdf[I] = dist
        for I in ti.grouped(self.target_sdf):
            self.target_sdf_copy[I] = self.target_sdf[I]
            self.nearest_point_copy[I] = self.nearest_point[I]

    # other
    @ti.kernel
    def clear_losses(self):
        self.density_loss.fill(0)
        self.density_loss.grad.fill(0)

        self.sdf_loss.fill(0)
        self.sdf_loss.grad.fill(0)

        self.contact_loss.fill(0)
        self.contact_loss.grad.fill(0)

        if not self.soft_contact_loss:
            self.min_dist.fill(self.inf)
        else:
            self.min_dist.fill(0)
        self.min_dist.grad.fill(0)
        self.dist_norm.fill(0)
        self.dist_norm.grad.fill(0)


    def compute_step_loss(self, s, f):
        self.compute_grid_mass(f)
        self.iou()
        self.compute_density_loss_kernel(s)
        self.compute_sdf_loss_kernel(s)
        self.compute_contact_loss(s, f)

        self.sum_up_loss_kernel(s)

    def compute_step_loss_grad(self, s, f):
        self.sum_up_loss_kernel.grad(s)

        self.compute_contact_loss_grad(s, f)
        self.compute_sdf_loss_kernel.grad(s)
        self.compute_density_loss_kernel.grad(s)
        self.compute_grid_mass_grad(f)

    def compute_grid_mass(self, f):
        self.grid_mass.fill(0)
        self.grid_mass.grad.fill(0)
        self.compute_grid_mass_kernel(f)

    def compute_grid_mass_grad(self, f):
        self.compute_grid_mass_kernel.grad(f)

    def compute_contact_loss(self, s, f):
        if self.soft_contact_loss:
            self.compute_contact_distance_normalize_kernel(s, f)
            self.compute_soft_contact_distance_kernel(s, f)
        else:
            self.compute_contact_distance_kernel(s, f)
        self.compute_contact_loss_kernel(s)

    def compute_contact_loss_grad(self, s, f):
        self.compute_contact_loss_kernel.grad(s)
        if self.soft_contact_loss:
            self.compute_soft_contact_distance_kernel.grad(s, f)
            self.compute_contact_distance_normalize_kernel.grad(s, f)
        else:
            self.compute_contact_distance_kernel.grad(s, f)

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
    def compute_sdf_loss_kernel(self, s: ti.i32):
        for I in ti.grouped(self.grid_mass):
            self.sdf_loss[s] += self.target_sdf[I] * self.grid_mass[I]

    @ti.kernel
    def compute_contact_distance_normalize_kernel(self, s: ti.i32, f: ti.i32):
        for p in range(self.n_particles):
            if not (self.isnan(self.particle_x[f, p][0]) and self.isnan(self.particle_x[f, p][0]) \
                    and self.isnan(self.particle_x[f, p][0])):
                d_ij = max(self.primitive.sdf(f, self.particle_x[f, p]), 0.)
                ti.atomic_add(self.dist_norm[s], self.soft_weight(d_ij))

    @ti.kernel
    def compute_soft_contact_distance_kernel(self, s: ti.i32, f: ti.i32):
        for p in range(self.n_particles):
            if not (self.isnan(self.particle_x[f, p][0]) and self.isnan(self.particle_x[f, p][0]) \
                    and self.isnan(self.particle_x[f, p][0])):
                d_ij = max(self.primitive.sdf(f, self.particle_x[f, p]), 0.)
                ti.atomic_add(self.min_dist[s], d_ij * self.soft_weight(d_ij) / self.dist_norm[s])

    @ti.kernel
    def compute_contact_distance_kernel(self, s: ti.i32, f: ti.i32):
        for p in range(self.n_particles):
            if not (self.isnan(self.particle_x[f, p][0]) and self.isnan(self.particle_x[f, p][0]) \
                    and self.isnan(self.particle_x[f, p][0])):
                d_ij = max(self.primitive.sdf(f, self.particle_x[f, p]), 0.)
                ti.atomic_min(self.min_dist[s], max(d_ij, 0.))

    @ti.kernel
    def compute_contact_loss_kernel(self, s: ti.i32):
        self.contact_loss[s] += self.min_dist[s] ** 2

    @ti.kernel
    def sum_up_loss_kernel(self, s: ti.i32):
        self.step_loss[s] += self.density_loss[s] * self.density_weight
        self.step_loss[s] += self.sdf_loss[s] * self.sdf_weight
        self.step_loss[s] += self.contact_loss[s] * self.contact_weight

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
            