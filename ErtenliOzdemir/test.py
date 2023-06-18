import os
import random
import multiprocessing as mp
import torch
import matplotlib.pyplot as plt
from r2d2 import R2D2, State
from environment import create_env
import yaml

device = torch.device("cpu")
torch.set_num_threads(4)


def test(config):
    game_name = config["game_name"]
    save_interval = config["save_interval"]
    save_plot = config["save_plot"]

    env = create_env(game_name, noop_start=True)
    test_round = 5
    pool = mp.Pool(test_round)
    x1, x2, y = [], [], []

    network = R2D2(env.action_space.n)
    network.to(device)
    network.share_memory()
    checkpoint = 1

    while os.path.exists(f"./models_{config['exploration_type']}/{game_name}{checkpoint*save_interval}.pth"):
        state_dict, training_steps, env_steps, time = torch.load(
            f"./models_{config['exploration_type']}/{game_name}{checkpoint*save_interval}.pth"
        )
        x1.append(env_steps * 4)
        x2.append(time / 60)
        network.load_state_dict(state_dict)

        args = [
            (
                network,
                env,
            )
            for _ in range(test_round)
        ]
        rewards = pool.map(test_one_case, args)

        print("env_frames: {}".format(env_steps * 4))
        print("wall-clock time: {:.2}h".format(time / 60))
        print("average reward: {}\n".format(sum(rewards) / test_round))
        y.append(sum(rewards) / test_round)
        checkpoint += 1

    plt.figure()
    plt.title(game_name)
    plt.xlabel("environment frames")
    plt.ylabel("average reward")
    plt.plot(x1, y)

    plt.show()

    if save_plot:
        plt.savefig("./{}.jpg".format(game_name))


def test_one_case(args):
    network, env = args
    obs = env.reset()
    done = False
    agent_state = State(torch.from_numpy(obs).unsqueeze(0), env.action_space.n)
    sum_reward = 0

    while not done:
        q_val, hidden = network(agent_state)

        if random.random() < config.test_epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(q_val, 1).item()

        obs, reward, done, _ = env.step(action)
        
        agent_state.update(obs, action, reward, hidden)
        sum_reward += reward

    return sum_reward


if __name__ == "__main__":

    with open('config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        
    test(config)
