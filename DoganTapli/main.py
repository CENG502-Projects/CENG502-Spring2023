from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import numpy as np
import os

from torch import nn
from tqdm import tqdm

from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.attack import AttackNetwork
from core.agent import Agent
from core.common import estimate_advantages_DAP


from gym.wrappers import AtariPreprocessing
from utils import ZFilter
import argparse
import gym
import time
import pickle
import json

from uap import query_uap, uap



from utils import *


def get_parameters():
    parser = argparse.ArgumentParser(description='PyTorch PPO example')
    parser.add_argument('--env-name', default="Pong", metavar='ENV', 
                        choices=('Pong', 'McPacman'),
                        help='name of the environment to run')
    parser.add_argument('--model-path', metavar='/PATH/TO/MODEL',
                        help='path of pre-trained model')
    parser.add_argument('--victim-path', metavar='/PATH/TO/VICTIM',
                        help='path of pre-trained victim', required=True)
    parser.add_argument('--uap-path', metavar='/PATH/TO/UAP',
                        help='path of pre-generated UAPs')
    parser.add_argument('--render', action='store_true', default=False,
                        help='render the environment')
    parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                        help='log std for the policy (default: -0.0)')
    parser.add_argument('--gamma', type=float, default= 0.997, metavar='G',
                        help='discount factor (default: 0.997)')
    parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                        help='gae (default: 0.95)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--learning-rate', type=float, default=0.0007, metavar='G',
                        help='learning rate (default: 0.0007)')
    parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                        help='clipping epsilon for PPO')
    parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                        help='number of threads for agent (default: 1)')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                        help='minimal batch size per PPO update (default: 2048)')
    parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                        help='minimal batch size for evaluation (default: 2048)')
    parser.add_argument('--max-inject-num', type=int, default=21, metavar='N',
                        help='maximal number of injections (default: 21)')
    parser.add_argument('--max-traj-len', type=int, default=1500, metavar='N',
                        help='maximal trajectory length (default: 1500)')  # ?? for Pong (from Fig 4)
    parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                        help='maximal number of main iterations (default: 500)') # 5000 episodes
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                        help="interval between saving model (default: 0, means don't save)")
    parser.add_argument('--gpu-index', type=int, default=None, metavar='N')
    args = parser.parse_args()
    args.env_name = f'ALE/{args.env_name}-v5'

    # labda_r = 0.95 

    return args

def dppo_step(attack_net, hidden, value_net, optimizer_attack, optimizer_value, optim_value_iternum, states, victim_actions,
             returns, advantages, old_switch_log_probs, old_lure_log_probs, clip_epsilon = 0.2, l2_reg = 1e-3):

    
    """update critic"""
    for _ in range(optim_value_iternum):
        values_pred = value_net(states)
        value_loss = (values_pred - returns).pow(2).mean()
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

    
    action_prob, switch_prob, hidden= attack_net(states, victim_actions, hidden)

    """Switch head loss"""
    ratio_sw = torch.exp(torch.log(switch_prob) - old_switch_log_probs)
    surr1_sw = ratio_sw * advantages
    surr2_sw = torch.clamp(ratio_sw, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    loss_sw = -torch.min(surr1_sw, surr2_sw).mean()

    """Lure head loss"""
    lured = action_prob[switch_prob.squeeze(1) > 0.5] 
    masked_advantages = advantages[switch_prob.squeeze(1) > 0.5] # mask advantages ?
    ratio_lr = torch.exp(torch.log(lured.gather(1, victim_actions.long())) - old_lure_log_probs)  # what if game ends before all injections?
    surr1_lr = ratio_lr * masked_advantages 
    surr2_lr = torch.clamp(ratio_lr, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * masked_advantages 
    loss_lr = -torch.min(surr1_lr, surr2_lr).mean()

    # which are added into the overall loss for DPPO.
    # Via minimizing the overall loss, these two sub-policies can be updated in a synchronous fashion.

    # check the signs 
    # our goal is to minimize the reward !! - solved in the estimate_advantages_DAP?
    total_loss = loss_sw + loss_lr

    optimizer_attack.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(attack_net.parameters(), 40) # ??
    optimizer_attack.step()


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
    action_space = env.action_space.n

    running_state = ZFilter(env.observation_space.shape, clip=5)
    running_reward = ZFilter((1,), demean=False, clip=10)
    
    """seeding"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    
    """load victim net"""
    victim_net, _, running_state = pickle.load(open(args.victim_path, "rb"))
    victim_net.to(device)

    if args.uap_path is None:
        # Creation
        list_of_keys = ['"{}_{}"'.format(i,j) for i in range(action_space) for j in range(action_space) if i != j]
        dict_str = '{' + ':[], '.join(list_of_keys) + ':[]}'

        uap_dict = json.loads(dict_str)

        # with the trained victim model collect states with the same action output
        # collect samples from the victim net
        # cluster them with the actions taken by the victim net
        agent_victim = Agent(env, victim_net, device, running_state=running_state, num_threads=args.num_threads)
        for i in range(100): # hyperparam ??
            batch, _ = agent_victim.collect_samples(args.min_batch_size, render=args.render)
            dtype=torch.float32
            states = torch.from_numpy(np.vstack(batch.state)).to(dtype).to(device)
            actions = torch.from_numpy(np.vstack(batch.action).flatten()).to(dtype).to(device)

            state_collections =  [ [] for _ in range(action_space)]
            for j in range(action_space):
                state_collections[j].append(states[actions == j])

        # for each other action call uap function to generate perturbations
        # store them in a dict
        for i in range(action_space):
            for j in range(i, action_space):
                if (i!=j):
                    key = '{}_{}'.format(i, j)
                    uap_dict[key].append( uap(victim_net, torch.stack([item for batch in state_collections[i] for item in batch]), j, action_space) )

                    key = '{}_{}'.format(j, i)
                    uap_dict[key].append( uap(victim_net, torch.stack([item for batch in state_collections[j] for item in batch]), i, action_space) )

        # dump the dictionary once generated the perturbations 
        with open("uap.json", "w") as uap_file:
            json.dump(uap_dict, uap_file)
    else:
        # read from file
        with open(args.uap_path, 'r') as uap_file:
            uap_dict = json.load(uap_file)

    
    attack_net = AttackNetwork(3, action_space)
    value_net = Value(512)
    attack_net.to(device)
    value_net.to(device)

    optimizer_attack = torch.optim.Adam(attack_net.parameters(), lr=args.learning_rate)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
    
    # optimization epoch number and batch size for PPO
    optim_epochs = 10
    optim_batch_size = 64
    
    """create agent"""
    dap_agent = Agent(env, attack_net, device, running_state=running_state, num_threads=args.num_threads)
    
    # random hidden and cell states ??
    hidden = attack_net.init_state()
    
    
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = dap_agent.collect_DAP_samples(hidden, args.min_batch_size, victim_net, args.max_inject_num, args.max_traj_len, uap_dict, render=args.render)
        t0 = time.time()
        dtype=torch.float32
        states = torch.from_numpy(np.vstack(batch.state)).to(dtype).to(device)
        actions = torch.from_numpy(np.vstack(batch.action)).to(dtype).to(device)
        switch = torch.from_numpy(np.vstack(batch.switch)).to(dtype).to(device)
        rewards = torch.from_numpy(np.vstack(batch.reward)).to(dtype).to(device)
        masks = np.vstack(batch.mask).flatten()
        inj_run_out = torch.from_numpy(np.vstack(batch.inj_run_out)).to(dtype).to(device)

        # Trajectory padding
        # Repeat the shorter trajectories (l_max/l_n)-1 times
        # trajectory padding here
        # use masks to find l_max and l_n
        indices = np.where(masks == 0)[0]
        if len(indices):
            if indices[-1] < masks.shape[0]:
                indices = np.append(indices, np.shape(masks)[0])
            prev = indices[0]
            l_max = prev
            for i in indices[1:]:
                if (i - prev -1) > l_max:
                    l_max = i - prev -1
                prev = i
            l_max = args.max_traj_len if l_max > args.max_traj_len else l_max
            
            masks = torch.from_numpy(masks).unsqueeze(1).to(dtype).to(device)

            prev = -1
            states_padding = []
            actions_padding = []
            switch_padding = []
            reward_padding = []
            masks_padding = []
            inj_run_out_padding = []
            for i in indices:
                l_n = i - prev -1
                if l_n < l_max and l_n and ((l_max//l_n)-1):
                    temp = states[prev+1:i]
                    states_padding.append(temp.repeat((l_max//l_n)-1,1,1,1))
                    temp = actions[prev+1:i]
                    actions_padding.append(temp.repeat((l_max//l_n)-1,1))
                    temp = switch[prev+1:i]
                    switch_padding.append(temp.repeat((l_max//l_n)-1,1))
                    temp = rewards[prev+1:i]
                    reward_padding.append(temp.repeat((l_max//l_n)-1,1))
                    temp = masks[prev+1:i]
                    masks_padding.append(temp.repeat((l_max//l_n)-1,1))
                    temp = inj_run_out[prev+1:i]
                    inj_run_out_padding.append(temp.repeat((l_max//l_n)-1,1))
                prev = i
                
            if len(states_padding):
                
                states_padding = torch.cat(states_padding, dim=0)
                states = torch.cat([states,states_padding], dim=0)
                
                actions_padding = torch.cat(actions_padding, dim=0)
                actions = torch.cat([actions,actions_padding], dim=0)
                
                switch_padding = torch.cat(switch_padding, dim=0)
                switch = torch.cat([switch,switch_padding], dim=0)
                
                reward_padding = torch.cat(reward_padding, dim=0)
                rewards = torch.cat([rewards,reward_padding], dim=0)
                
                masks_padding = torch.cat(masks_padding, dim=0)
                masks = torch.cat([masks,masks_padding], dim=0)
                
                inj_run_out_padding = torch.cat(inj_run_out_padding, dim=0)
                inj_run_out = torch.cat([inj_run_out,inj_run_out_padding], dim=0)

        else:
            masks = torch.from_numpy(masks).unsqueeze(1).to(dtype).to(device)

        with torch.no_grad():
            values = value_net(states)

            victim_actions = victim_net(states)
            action_prob, switch_prob, _ = attack_net(states, victim_actions, hidden)

            old_switch_log_probs = torch.log(switch_prob)

            action_prob = action_prob.gather(1, actions.long())
            lured = action_prob[switch.squeeze(1) > 0.5]

            old_lure_log_probs = torch.log(lured)

        # Estimates advantages with trajectory clipping 
        advantages, returns = estimate_advantages_DAP(rewards, masks, values, inj_run_out, args.gamma, args.tau, device)

        # Update model weights 
        # batch by batch !!
        dppo_step(attack_net, hidden, value_net, optimizer_attack, optimizer_value, 1, states, victim_actions, returns, advantages, old_switch_log_probs, old_lure_log_probs)

        t1 = time.time()
    
        if i_iter % args.log_interval == 0:
            print('{}\tT_update {:.4f}\ttrain_R_min {:.2f}\ttrain_R_max {:.2f}\ttrain_R_avg {:.2f}'.format(
                i_iter, t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))
    
        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), attack_net, value_net)
            pickle.dump((attack_net, value_net, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_dap.p'.format(args.env_name)), 'wb'))
            to_device(device, attack_net, value_net)
    
        """clean up gpu memory"""
        torch.cuda.empty_cache()

