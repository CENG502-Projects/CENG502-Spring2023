import math
import numpy as np
from typing import Tuple

from replay_buffer import Block
from learner import calculate_mixed_td_errors


class LocalBuffer:
    """Store transitions of one episode"""

    def __init__(
        self,
        action_dim: int,
        config
    ):
        
        forward_steps = config["forward_steps"]
        burn_in_steps = config["burn_in_steps"]
        learning_steps = config["learning_steps"]
        gamma = config["gamma"]
        hidden_dim = config["hidden_dim"]
        block_length = config["block_length"]

        self.action_dim = action_dim
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.forward_steps = forward_steps
        self.learning_steps = learning_steps
        self.burn_in_steps = burn_in_steps
        self.block_length = block_length
        self.curr_burn_in_steps = 0
        self.full_reward_buffer = []
        self.full_qval_buffer = []

    def __len__(self):
        return self.size
    
    def __getitem__(self, key):
        idx, buffer_type = key
        if buffer_type == "reward":
            return self.full_reward_buffer[idx]
        elif buffer_type == "qval":
            return self.full_qval_buffer[idx]

    def reset(self, init_obs: np.ndarray):
        self.obs_buffer = [init_obs]
        self.last_action_buffer = [
            np.array([1 if i == 0 else 0 for i in range(self.action_dim)], dtype=bool)
        ]
        self.last_reward_buffer = [0]
        self.hidden_buffer = [np.zeros((2, self.hidden_dim), dtype=np.float32)]
        self.action_buffer = []
        self.reward_buffer = []
        self.qval_buffer = []
        self.curr_burn_in_steps = 0
        self.size = 0
        self.sum_reward = 0
        self.done = False
        
    def reset_full_buffers(self):
        self.full_reward_buffer = []
        self.full_qval_buffer = []

    def add(
        self,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        q_value: np.ndarray,
        hidden_state: np.ndarray,
    ):
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.full_reward_buffer.append(reward)
        self.hidden_buffer.append(hidden_state)
        self.obs_buffer.append(next_obs)
        self.last_action_buffer.append(
            np.array(
                [1 if i == action else 0 for i in range(self.action_dim)], dtype=bool
            )
        )
        self.last_reward_buffer.append(reward)
        self.qval_buffer.append(q_value)
        self.full_qval_buffer.append(q_value)
        self.sum_reward += reward
        self.size += 1

    def finish(self, last_qval: np.ndarray = None) -> Tuple:
        assert self.size <= self.block_length
        # assert len(self.last_action_buffer) == self.curr_burn_in_steps + self.size + 1

        num_sequences = math.ceil(self.size / self.learning_steps)

        max_forward_steps = min(self.size, self.forward_steps)
        n_step_gamma = [self.gamma**self.forward_steps] * (
            self.size - max_forward_steps
        )

        # last_qval is none means episode done
        if last_qval is not None:
            self.qval_buffer.append(last_qval)
            n_step_gamma.extend(
                [self.gamma**i for i in reversed(range(1, max_forward_steps + 1))]
            )
        else:
            self.done = True
            self.qval_buffer.append(np.zeros_like(self.qval_buffer[0]))
            n_step_gamma.extend(
                [0 for _ in range(max_forward_steps)]
            )  # set gamma to 0 so don't need 'done'

        n_step_gamma = np.array(n_step_gamma, dtype=np.float32)

        obs = np.stack(self.obs_buffer)
        last_action = np.stack(self.last_action_buffer)
        last_reward = np.array(self.last_reward_buffer, dtype=np.float32)

        hiddens = np.stack(self.hidden_buffer[slice(0, self.size, self.learning_steps)])

        actions = np.array(self.action_buffer, dtype=np.uint8)

        qval_buffer = np.concatenate(self.qval_buffer)
        reward_buffer = self.reward_buffer + [0 for _ in range(self.forward_steps - 1)]
        n_step_reward = np.convolve(
            reward_buffer,
            [
                self.gamma ** (self.forward_steps - 1 - i)
                for i in range(self.forward_steps)
            ],
            "valid",
        ).astype(np.float32)

        burn_in_steps = np.array(
            [
                min(
                    i * self.learning_steps + self.curr_burn_in_steps,
                    self.burn_in_steps,
                )
                for i in range(num_sequences)
            ],
            dtype=np.uint8,
        )
        learning_steps = np.array(
            [
                min(self.learning_steps, self.size - i * self.learning_steps)
                for i in range(num_sequences)
            ],
            dtype=np.uint8,
        )
        forward_steps = np.array(
            [
                min(self.forward_steps, self.size + 1 - np.sum(learning_steps[: i + 1]))
                for i in range(num_sequences)
            ],
            dtype=np.uint8,
        )
        assert forward_steps[-1] == 1 and burn_in_steps[0] == self.curr_burn_in_steps

        max_qval = np.max(qval_buffer[max_forward_steps : self.size + 1], axis=1)
        max_qval = np.pad(max_qval, (0, max_forward_steps - 1), "edge")
        target_qval = qval_buffer[np.arange(self.size), actions]

        td_errors = np.abs(
            n_step_reward + n_step_gamma * max_qval - target_qval, dtype=np.float32
        )
        priorities = np.zeros(
            self.block_length // self.learning_steps, dtype=np.float32
        )
        priorities[:num_sequences] = calculate_mixed_td_errors(
            td_errors, learning_steps
        )

        # save burn in information for next block
        self.obs_buffer = self.obs_buffer[-self.burn_in_steps - 1 :]
        self.last_action_buffer = self.last_action_buffer[-self.burn_in_steps - 1 :]
        self.last_reward_buffer = self.last_reward_buffer[-self.burn_in_steps - 1 :]
        self.hidden_buffer = self.hidden_buffer[-self.burn_in_steps - 1 :]
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.qval_buffer.clear()
        self.curr_burn_in_steps = len(self.obs_buffer) - 1
        self.size = 0

        block = Block(
            obs,
            last_action,
            last_reward,
            actions,
            n_step_reward,
            n_step_gamma,
            hiddens,
            num_sequences,
            burn_in_steps,
            learning_steps,
            forward_steps,
        )
        return [block, priorities, self.sum_reward if self.done else None]
