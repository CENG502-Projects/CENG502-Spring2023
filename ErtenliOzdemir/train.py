import random
import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

from actor import Actor
from learner import Learner
from replay_buffer import ReplayBuffer
from environment import create_env
from r2d2 import R2D2

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.set_num_threads(1)


def get_epsilon(
    actor_id: int,
    num_actors: int,
    config
):
    base_eps = config["base_eps"]
    alpha = config["alpha"]
    exponent = 1 + actor_id / (num_actors - 1) * alpha
    return base_eps**exponent


def train(config):
    num_actors = config["num_actors"]
    model = R2D2(create_env(config["game_name"]).action_space.n)
    model.share_memory()
    sample_queue_list = [mp.Queue() for _ in range(num_actors)]
    batch_queue = mp.Queue(8)
    priority_queue = mp.Queue(8)

    buffer = ReplayBuffer(sample_queue_list, batch_queue, priority_queue, config)
    learner = Learner(batch_queue, priority_queue, model, config)
    actors = [
        Actor(get_epsilon(i, num_actors, config), model, sample_queue_list[i], config) for i in range(num_actors)
    ]

    actor_procs = [mp.Process(target=actor.run) for actor in actors]
    for proc in actor_procs:
        proc.start()

    buffer_proc = mp.Process(target=buffer.run)
    buffer_proc.start()

    learner.run()

    buffer_proc.join()

    for proc in actor_procs:
        proc.terminate()


if __name__ == "__main__":

    with open('config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        
    train(config)
