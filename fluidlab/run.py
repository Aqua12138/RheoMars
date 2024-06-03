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
from fluidlab.optimizer.rl_trainer import PPO_trainer
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='test')
    parser.add_argument("--env_name", type=str, default='')
    parser.add_argument("--seed", type=int, default=334)
    parser.add_argument("--cfg_file", type=str, default=None)
    parser.add_argument("--record", action='store_true')
    parser.add_argument("--user_input", action='store_true')
    parser.add_argument("--replay_policy", action='store_true')
    parser.add_argument("--replay_target", action='store_true')
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--renderer_type", type=str, default='GGUI')
    parser.add_argument("--loss_type", type=str, default='default')
    parser.add_argument("--record_target_grid", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--rl", type=str, default='default')
    parser.add_argument("--perc_type", type=str, default='sensor')



    args = parser.parse_args()

    return args

def main():
    args = get_args()
    if args.cfg_file is not None:
        cfg = load_config(args.cfg_file)
    else:
        cfg = None

    if args.record:
        if cfg is not None:
            env = gym.make(cfg.EXP.env_name, seed=cfg.EXP.seed, loss=False, loss_type='diff', renderer_type=args.renderer_type)
        else:
            env = gym.make(args.env_name, seed=args.seed, loss=False, loss_type='diff', renderer_type=args.renderer_type)
        record_target(env, path=args.path, user_input=args.user_input)
    elif args.replay_target:
        if cfg is not None:
            env = gym.make(cfg.EXP.env_name, seed=cfg.EXP.seed, loss=False, loss_type='diff', renderer_type=args.renderer_type)
        else:
            env = gym.make(args.env_name, seed=args.seed, loss=False, loss_type='diff', renderer_type=args.renderer_type)
        replay_target(env)
    elif args.replay_policy:
        if cfg is not None:
            env = gym.make(cfg.EXP.env_name, seed=cfg.EXP.seed, loss=True, loss_type='diff', renderer_type=args.renderer_type)
        else:
            env = gym.make(args.env_name, seed=args.seed, loss=True, loss_type='diff', renderer_type=args.renderer_type)
        replay_policy(env, path=args.path)
    elif args.record_target_grid:
        if cfg is not None:
            env = gym.make(cfg.EXP.env_name, seed=cfg.EXP.seed, loss=False, loss_type='diff', renderer_type=args.renderer_type)
        else:
            env = gym.make(args.env_name, seed=args.seed, loss=False, loss_type='diff', renderer_type=args.renderer_type)
        record_target_grid(env, user_input=args.user_input)
    elif args.debug:
        if cfg is not None:
            env = gym.make(cfg.EXP.env_name, seed=cfg.EXP.seed, loss=True, loss_type='diff', renderer_type=args.renderer_type)
        else:
            env = gym.make(args.env_name, seed=args.seed, loss=True, loss_type='diff', renderer_type=args.renderer_type)
        debug(env)
    elif args.rl == "ppo":
        PPO = PPO_trainer(cfg, args)
        PPO.solver()
    else:
        logger = Logger(args.exp_name)
        env = gym.make(cfg.EXP.env_name, seed=cfg.EXP.seed, loss=True, loss_type=args.loss_type, renderer_type=args.renderer_type)
        solve_policy(env, logger, cfg.SOLVER)

if __name__ == '__main__':
    main()
