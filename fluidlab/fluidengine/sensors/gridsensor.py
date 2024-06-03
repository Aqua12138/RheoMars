import numpy as np
import taichi as ti
import torch
from fluidlab.configs.macros import *
from pyquaternion import Quaternion
import pickle as pkl

@ti.data_oriented
class GridSensor:
    def __init__(self, horizon):
        self.horizon = horizon+1

    def reset(self):
        self.clear_grid_sensor()

    def reset_grad(self):
        pass

    def build(self, sim, particle_state, grid_state):
        self.sim = sim
        self.device = 'cpu'
        self.dim = sim.dim

        if self.sim.agent is not None:
            self.agent = sim.agent

        if self.sim.particles is not None:
            self.particle_x = sim.particles.x
            self.particle_mat = sim.particles_i.mat
            self.particle_used = sim.particles_ng.used
            self.n_particles = sim.n_particles
            self.n_bodies = sim.n_bodies

        if self.target_file is not None:
            self.load_target(self.target_file)
            self.n_bodies += 1
            self.targets = particle_state.field(shape=(self.horizon, self.n_particles,), needs_grad=True,
                                              layout=ti.Layout.SOA)

        self.particles = particle_state.field(shape=(self.horizon, self.n_particles,), needs_grad=True,
                                              layout=ti.Layout.SOA)
        self.grid_sensor = grid_state.field(shape=(self.horizon, self.M, self.N, self.n_bodies), needs_grad=True,
                                            layout=ti.Layout.SOA)


    def load_target(self, path):
        targets = pkl.load(open(path, 'rb'))
        self.tgt_particles_x = ti.Vector.field(self.dim, dtype=DTYPE_TI, shape=self.n_particles)
        self.tgt_particles_x.from_numpy(targets['last_pos'])
        print(f'===>  Target loaded from {path}.')

    def clear_grid_sensor(self):
        pass