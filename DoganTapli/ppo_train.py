from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import numpy as np
import os

#from tensordict.nn import TensorDictModule
#from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
'''from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, set_exploration_mode
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE'''
from tqdm import tqdm

from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from core.agent import Agent
from core.common import estimate_advantages
from core.ppo import ppo_step

from gym.wrappers import AtariPreprocessing
from utils import ZFilter
import argparse
import gym
import time
import math
import pickle

from utils import *


def get_parameters():
    parser = argparse.ArgumentParser(description='PyTorch PPO example')
    parser.add_argument('--env-name', default="Pong", metavar='ENV', 
                        choices=('Pong', 'McPacman'),
                        help='name of the environment to run')
    parser.add_argument('--model-path', metavar='/PATH/TO/MODEL',
                        help='path of pre-trained model')
    parser.add_argument('--render', action='store_true', default=False,
                        help='render the environment')
    parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                        help='log std for the policy (default: -0.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                        help='gae (default: 0.95)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                        help='clipping epsilon for PPO')
    parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                        help='number of threads for agent (default: 4)')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                        help='minimal batch size per PPO update (default: 2048)')
    parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                        help='minimal batch size for evaluation (default: 2048)')
    parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                        help='maximal number of main iterations (default: 500)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                        help="interval between saving model (default: 0, means don't save)")
    parser.add_argument('--gpu-index', type=int, default=None, metavar='N')
    args = parser.parse_args()
    args.env_name = f'{args.env_name}-v0'

    return args


def update_params(batch, i_iter):
    dtype=torch.float32
    states = torch.from_numpy(np.vstack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.vstack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.vstack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.vstack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(states)
        fixed_log_probs = policy_net.get_log_prob(states, actions)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = torch.LongTensor(perm).to(device)

        states, actions, returns, advantages, fixed_log_probs = \
            states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg)


if __name__ == '__main__':
    args = get_parameters()
    device = torch.device('cuda', index=args.gpu_index) \
                        if torch.cuda.is_available() \
                        and args.gpu_index is not None \
                        else torch.device('cpu')

    print(args.env_name)
    env = gym.make(args.env_name)
    env = AtariPreprocessing(env, frame_skip=1, grayscale_obs=False, grayscale_newaxis=True)
    env.unwrapped.np_random = np.random
    state_dim = env.observation_space.shape[0]

    running_state = ZFilter(env.observation_space.shape, clip=5)
    # running_reward = ZFilter((1,), demean=False, clip=10)
    
    """seeding"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    
    """define actor and critic"""
    if args.model_path is None:
        policy_net = DiscretePolicy(env.action_space.n, 512)
        value_net = Value(512)
    else:
        policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
    policy_net.to(device)
    value_net.to(device)
    
    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
    
    # optimization epoch number and batch size for PPO
    optim_epochs = 10
    optim_batch_size = 64
    
    """create agent"""
    agent = Agent(env, policy_net, device, running_state=running_state, num_threads=args.num_threads)
    
    
    
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size, render=args.render)
        t0 = time.time()
        update_params(batch, i_iter)
        t1 = time.time()
        """evaluate with determinstic action (remove noise for exploration)"""
        _, log_eval = agent.collect_samples(args.eval_batch_size, mean_action=True)
        t2 = time.time()
    
        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tT_eval {:.4f}\ttrain_R_min {:.2f}\ttrain_R_max {:.2f}\ttrain_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, t2-t1, log['min_reward'], log['max_reward'], log['avg_reward'], log_eval['avg_reward']))
    
        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net)
            pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(args.env_name)), 'wb'))
            to_device(device, policy_net, value_net)
            break
    
        """clean up gpu memory"""
        torch.cuda.empty_cache()

