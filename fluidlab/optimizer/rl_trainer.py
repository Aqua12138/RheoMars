from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3 import PPO
import os
import gym
import torch
import random
import argparse
import numpy as np

import fluidlab.envs
from fluidlab.utils.logger import Logger
from fluidlab.optimizer.solver import solve_policy
from fluidlab.optimizer.recorder import record_target, replay_policy, replay_target, record_target_grid, debug
from fluidlab.utils.config import load_config

class PPO_trainer:
    def __init__(self, cfg, args):
        env_kwargs = {
            # "seed": cfg["params"]["general"]["seed"],
            "loss": True,
            "loss_type": 'default',
            "renderer_type": "GGUI",
            "seed": cfg.EXP.seed,
            "perc_type" : args.perc_type,
        }
        self.envs = make_vec_env(env_id=cfg.EXP.env_name, n_envs=1, env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv)
        self.model = PPO("MultiInputPolicy", self.envs, verbose=1, tensorboard_log="./ppo_gather_tensorboard/")
    def solver(self):
        self.model.learn(total_timesteps=int(2e6))
        self.model.save("ppo_test2")
