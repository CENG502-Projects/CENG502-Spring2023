import time
import threading
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass

from priority_tree import PriorityTree

@dataclass
class Block:
    obs: np.array
    last_action: np.array
    last_reward: np.array
    action: np.array
    n_step_reward: np.array
    gamma: np.array
    hidden: np.array
    num_sequences: int
    burn_in_steps: np.array
    learning_steps: np.array
    forward_steps: np.array


class ReplayBuffer:
    def __init__(
        self,
        sample_queue_list,
        batch_queue,
        priority_queue,
        config
    ):
        
        buffer_capacity = config["buffer_capacity"]
        alpha = config["prio_exponent"]
        beta = config["importance_sampling_exponent"]
        batch_size = config["batch_size"]

        self.config = config

        self.buffer_capacity = buffer_capacity
        self.sequence_len = config["learning_steps"]
        self.num_sequences = buffer_capacity // self.sequence_len
        self.block_len = config["block_length"]
        self.num_blocks = self.buffer_capacity // self.block_len
        self.seq_pre_block = self.block_len // self.sequence_len

        self.block_ptr = 0

        self.priority_tree = PriorityTree(self.num_sequences, alpha, beta)

        self.batch_size = batch_size

        self.env_steps = 0

        self.num_episodes = 0
        self.episode_reward = 0

        self.training_steps = 0
        self.last_training_steps = 0
        self.sum_loss = 0

        self.lock = threading.Lock()

        self.size = 0
        self.last_size = 0

        self.buffer = [None] * self.num_blocks

        self.sample_queue_list, self.batch_queue, self.priority_queue = (
            sample_queue_list,
            batch_queue,
            priority_queue,
        )

    def __len__(self):
        return self.size

    def run(self):
        background_thread = threading.Thread(target=self.add_data, daemon=True)
        background_thread.start()

        background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        background_thread.start()

        background_thread = threading.Thread(target=self.update_data, daemon=True)
        background_thread.start()

        log_interval = self.config["log_interval"]

        while True:
            print(f"buffer size: {self.size}")
            print(f"buffer update speed: {(self.size-self.last_size)/log_interval}/s")
            self.last_size = self.size
            print(f"number of environment steps: {self.env_steps}")
            if self.num_episodes != 0:
                print(
                    f"average episode return: {self.episode_reward/self.num_episodes:.4f}"
                )
                # print(f'average episode return: {self.episode_reward/self.num_episodes:.4f}')
                self.episode_reward = 0
                self.num_episodes = 0
            print(f"number of training steps: {self.training_steps}")
            print(
                f"training speed: {(self.training_steps-self.last_training_steps)/log_interval}/s"
            )
            if self.training_steps != self.last_training_steps:
                print(
                    f"loss: {self.sum_loss/(self.training_steps-self.last_training_steps):.4f}"
                )
                self.last_training_steps = self.training_steps
                self.sum_loss = 0
            self.last_env_steps = self.env_steps
            print()

            if self.training_steps == self.config["training_steps"]:
                break
            else:
                time.sleep(log_interval)

    def prepare_data(self):
        while self.size < self.config["learning_starts"]:
            time.sleep(1)

        while True:
            if not self.batch_queue.full():
                data = self.sample_batch()
                self.batch_queue.put(data)
            else:
                time.sleep(0.1)

    def add_data(self):
        while True:
            for sample_queue in self.sample_queue_list:
                if not sample_queue.empty():
                    data = sample_queue.get_nowait()
                    self.add(*data)

    def update_data(self):
        while True:
            if not self.priority_queue.empty():
                data = self.priority_queue.get_nowait()
                self.update_priorities(*data)
            else:
                time.sleep(0.1)

    def add(self, block: Block, priority: np.array, episode_reward: float):
        with self.lock:
            idxes = np.arange(
                self.block_ptr * self.seq_pre_block,
                (self.block_ptr + 1) * self.seq_pre_block,
                dtype=np.int64,
            )

            self.priority_tree.update(idxes, priority)

            if self.buffer[self.block_ptr] is not None:
                self.size -= np.sum(self.buffer[self.block_ptr].learning_steps).item()

            self.size += np.sum(block.learning_steps).item()

            self.buffer[self.block_ptr] = block

            self.env_steps += np.sum(block.learning_steps, dtype=np.int32)

            self.block_ptr = (self.block_ptr + 1) % self.num_blocks
            if episode_reward:
                self.episode_reward += episode_reward
                self.num_episodes += 1

    def sample_batch(self):
        """sample one batch of training data"""
        (
            batch_obs,
            batch_last_action,
            batch_last_reward,
            batch_hidden,
            batch_action,
            batch_reward,
            batch_gamma,
        ) = ([], [], [], [], [], [], [])
        burn_in_steps, learning_steps, forward_steps = [], [], []

        with self.lock:

            idxes, is_weights = self.priority_tree.sample(self.batch_size)

            block_idxes = idxes // self.seq_pre_block
            sequence_idxes = idxes % self.seq_pre_block

            for block_idx, sequence_idx in zip(block_idxes, sequence_idxes):

                block = self.buffer[block_idx]

                assert (
                    sequence_idx < block.num_sequences
                ), "index is {} but size is {}".format(
                    sequence_idx, self.seq_pre_block_buf[block_idx]
                )

                burn_in_step = block.burn_in_steps[sequence_idx]
                learning_step = block.learning_steps[sequence_idx]
                forward_step = block.forward_steps[sequence_idx]

                start_idx = block.burn_in_steps[0] + np.sum(
                    block.learning_steps[:sequence_idx]
                )

                obs = block.obs[
                    start_idx - burn_in_step : start_idx + learning_step + forward_step
                ]
                last_action = block.last_action[
                    start_idx - burn_in_step : start_idx + learning_step + forward_step
                ]
                last_reward = block.last_reward[
                    start_idx - burn_in_step : start_idx + learning_step + forward_step
                ]
                obs, last_action, last_reward = (
                    torch.from_numpy(obs),
                    torch.from_numpy(last_action),
                    torch.from_numpy(last_reward),
                )

                start_idx = np.sum(block.learning_steps[:sequence_idx])
                end_idx = start_idx + block.learning_steps[sequence_idx]
                action = block.action[start_idx:end_idx]
                reward = block.n_step_reward[start_idx:end_idx]
                gamma = block.gamma[start_idx:end_idx]
                hidden = block.hidden[sequence_idx]

                batch_obs.append(obs)
                batch_last_action.append(last_action)
                batch_last_reward.append(last_reward)
                batch_action.append(action)
                batch_reward.append(reward)
                batch_gamma.append(gamma)
                batch_hidden.append(hidden)

                burn_in_steps.append(burn_in_step)
                learning_steps.append(learning_step)
                forward_steps.append(forward_step)

            batch_obs = pad_sequence(batch_obs, batch_first=True)
            batch_last_action = pad_sequence(batch_last_action, batch_first=True)
            batch_last_reward = pad_sequence(batch_last_reward, batch_first=True)

            is_weights = np.repeat(is_weights, learning_steps)

            data = (
                batch_obs,
                batch_last_action,
                batch_last_reward,
                torch.from_numpy(np.stack(batch_hidden)).transpose(0, 1),
                torch.from_numpy(np.concatenate(batch_action)).unsqueeze(1),
                torch.from_numpy(np.concatenate(batch_reward)),
                torch.from_numpy(np.concatenate(batch_gamma)),
                torch.ByteTensor(burn_in_steps),
                torch.ByteTensor(learning_steps),
                torch.ByteTensor(forward_steps),
                idxes,
                torch.from_numpy(is_weights.astype(np.float32)),
                self.block_ptr,
                self.env_steps,
            )

        return data

    def update_priorities(
        self, idxes: np.ndarray, td_errors: np.ndarray, old_ptr: int, loss: float
    ):
        """Update priorities of sampled transitions"""
        with self.lock:

            # discard the idxes that already been replaced by new data in replay buffer during training
            if self.block_ptr > old_ptr:
                # range from [old_ptr, self.seq_ptr)
                mask = (idxes < old_ptr * self.seq_pre_block) | (
                    idxes >= self.block_ptr * self.seq_pre_block
                )
                idxes = idxes[mask]
                td_errors = td_errors[mask]
            elif self.block_ptr < old_ptr:
                # range from [0, self.seq_ptr) & [old_ptr, self,capacity)
                mask = (idxes < old_ptr * self.seq_pre_block) & (
                    idxes >= self.block_ptr * self.seq_pre_block
                )
                idxes = idxes[mask]
                td_errors = td_errors[mask]

            self.priority_tree.update(idxes, td_errors)

        self.training_steps += 1
        self.sum_loss += loss