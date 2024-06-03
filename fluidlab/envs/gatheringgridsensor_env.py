import os
import gym
import numpy as np
from .fluid_env import FluidEnv
from yacs.config import CfgNode
from fluidlab.utils.misc import *
from fluidlab.configs.macros import *
from fluidlab.optimizer.policies import *
from fluidlab.fluidengine.taichi_env import TaichiEnv
from fluidlab.fluidengine.losses import *


class GatheringGridSensorEnv(FluidEnv):
    def __init__(self, version, loss=True, loss_type='diff', seed=None, renderer_type='GGUI', perc_type="physics"):

        # Gathering_pos-v0.pkl ShapeMatchingLoss(DynamicDistanceLoss)
        # Gathering_grid-v0.pkl SDFDensityLoss
        target_file = 'Gathering_grid-v0.pkl' # Gathering_pos-v0.pkl

        self.target_file = get_tgt_path(target_file)
        if target_file == "Gathering_pos-v0.pkl":
            self.Loss = DynamicDistanceLoss
            self.weight = {'chamfer': 1}
        elif target_file == "Gathering_grid-v0.pkl":
            self.Loss = SDFDensityLoss
            self.weight = {'density': 1, 'sdf': 100, 'contact': 1, 'is_soft_contact': False}

        if seed is not None:
            self.seed(seed)

        self.horizon = 1000
        self.horizon_action = 1000

        self._n_obs_ptcls_per_body = 1000
        self.loss = loss
        self.loss_type = loss_type
        self.action_range = np.array([-0.0003, 0.0003])
        self.renderer_type = renderer_type
        self.perc_type=perc_type

        # create a taichi env
        self.taichi_env = TaichiEnv(
            dim=3,
            particle_density=1e6,
            max_substeps_local=50,
            gravity=(0.0, -20.0, 0.0),
            horizon=self.horizon,
        )
        self.build_env()
        self.gym_misc()

    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(get_cfg_path('agent_gathering_gridsensor.yaml'))
        if self.target_file is not None:
            for sensor in agent_cfg.sensors:
                sensor["params"]["target_file"] = self.target_file
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    def setup_statics(self):
        self.taichi_env.add_static(
            file='table.obj',
            pos=(0.5, 0.4, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(1, 1, 1),
            material=CONE,
            has_dynamics=True,
        )

    def setup_bodies(self):
        self.taichi_env.add_body(
            type='cube',
            lower=(0.45, 0.50, 0.25),
            upper=(0.55, 0.70, 0.35),
            material=MILK_VIS,
        )

    def setup_boundary(self):
        self.taichi_env.setup_boundary(
            type='cube',
            lower=(0.05, 0.3, 0.05),
            upper=(0.95, 0.95, 0.95),
        )

    def setup_renderer(self):
        if self.renderer_type == 'GGUI':
            self.taichi_env.setup_renderer(
                camera_pos=(-0.15, 2.82, 2.5),
                camera_lookat=(0.5, 0.5, 0.5),
                fov=30,
                lights=[{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                        {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}],
            )
        elif self.renderer_type == 'GL':
            self.taichi_env.setup_renderer(
                type='GL',
                render_particle=True,
                camera_pos=(-0.15, 2.82, 2.5),
                camera_lookat=(0.5, 0.5, 0.5),
                fov=30,
                light_pos=(3.5, 15.0, 0.55),
                light_lookat=(0.5, 0.5, 0.49),
                light_fov=20,
            )
        else:
            raise NotImplementedError

    def setup_loss(self):
        self.taichi_env.setup_loss(
            loss_cls=self.Loss,
            type=self.loss_type,
            target_file=self.target_file,
            weights=self.weight
        )

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array']

        if self.loss is not None:
            tgt_particles = self.loss.tgt_particles_x_f32
        else:
            tgt_particles = None

        return self.taichi_env.render(mode)

    def demo_policy(self, user_input=False):
        if not user_input:
            raise NotImplementedError

        init_p = np.array([0.5, 0.55, 0.2])
        return MousePolicy_vxz(init_p)

    def trainable_policy(self, optim_cfg, init_range):
        return GatheringPolicy(optim_cfg, init_range, self.agent.action_dim, self.horizon_action, self.action_range)

    def get_obs(self):
        obs = []
        for sensor in self.agent.sensors:
            obs.append(sensor.get_obs())
        return obs