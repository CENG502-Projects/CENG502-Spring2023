import os
import time
import threading
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn


def calculate_mixed_td_errors(td_error, learning_steps):
    start_idx = 0
    mixed_td_errors = np.empty(learning_steps.shape, dtype=td_error.dtype)
    for i, steps in enumerate(learning_steps):
        mixed_td_errors[i] = (
            0.9 * td_error[start_idx : start_idx + steps].max()
            + 0.1 * td_error[start_idx : start_idx + steps].mean()
        )
        start_idx += steps

    return mixed_td_errors


class Learner:
    def __init__(
        self,
        batch_queue,
        priority_queue,
        model,
        config
    ):
        
        grad_norm = config["grad_norm"]
        lr = config["lr"]
        eps = config["eps"]
        game_name = config["game_name"]
        target_net_update_interval = config["target_net_update_interval"]
        save_interval = config["save_interval"]

        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online_net = deepcopy(model)
        self.online_net.to(self.device)
        self.online_net.train()
        self.target_net = deepcopy(self.online_net)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr, eps=eps)
        self.loss_fn = nn.MSELoss(reduction="none")
        self.grad_norm = grad_norm
        self.batch_queue = batch_queue
        self.priority_queue = priority_queue
        self.num_updates = 0
        self.done = False

        self.target_net_update_interval = target_net_update_interval
        self.save_interval = save_interval

        self.batched_data = []

        self.shared_model = model

        self.game_name = game_name

    def store_weights(self):
        self.shared_model.load_state_dict(self.online_net.state_dict())

    def prepare_data(self):

        while True:
            if not self.batch_queue.empty() and len(self.batched_data) < 4:
                data = self.batch_queue.get_nowait()
                self.batched_data.append(data)
            else:
                time.sleep(0.1)

    def run(self):
        background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        background_thread.start()
        time.sleep(2)

        start_time = time.time()
        while self.num_updates < self.config["training_steps"]:

            while not self.batched_data:
                time.sleep(1)
            data = self.batched_data.pop(0)

            (
                batch_obs,
                batch_last_action,
                batch_last_reward,
                batch_hidden,
                batch_action,
                batch_n_step_reward,
                batch_n_step_gamma,
                burn_in_steps,
                learning_steps,
                forward_steps,
                idxes,
                is_weights,
                old_ptr,
                env_steps,
            ) = data
            batch_obs, batch_last_action, batch_last_reward = (
                batch_obs.to(self.device),
                batch_last_action.to(self.device),
                batch_last_reward.to(self.device),
            )
            batch_hidden, batch_action = batch_hidden.to(self.device), batch_action.to(
                self.device
            )
            batch_n_step_reward, batch_n_step_gamma = batch_n_step_reward.to(
                self.device
            ), batch_n_step_gamma.to(self.device)
            is_weights = is_weights.to(self.device)

            batch_obs, batch_last_action = batch_obs.float(), batch_last_action.float()
            batch_action = batch_action.long()

            batch_hidden = (batch_hidden[:1], batch_hidden[1:])

            batch_obs = batch_obs / 255

            # double q learning
            with torch.no_grad():
                batch_action_ = (
                    self.online_net.calculate_q_(
                        batch_obs,
                        batch_last_action,
                        batch_last_reward,
                        batch_hidden,
                        burn_in_steps,
                        learning_steps,
                        forward_steps,
                    )
                    .argmax(1)
                    .unsqueeze(1)
                )
                batch_q_ = (
                    self.target_net.calculate_q_(
                        batch_obs,
                        batch_last_action,
                        batch_last_reward,
                        batch_hidden,
                        burn_in_steps,
                        learning_steps,
                        forward_steps,
                    )
                    .gather(1, batch_action_)
                    .squeeze(1)
                )

            target_q = self.value_rescale(
                batch_n_step_reward
                + batch_n_step_gamma * self.inverse_value_rescale(batch_q_)
            )

            batch_q = (
                self.online_net.calculate_q(
                    batch_obs,
                    batch_last_action,
                    batch_last_reward,
                    batch_hidden,
                    burn_in_steps,
                    learning_steps,
                )
                .gather(1, batch_action)
                .squeeze(1)
            )

            loss = (is_weights * self.loss_fn(batch_q, target_q)).mean()

            td_errors = (
                (target_q - batch_q)
                .detach()
                .clone()
                .squeeze()
                .abs()
                .cpu()
                .float()
                .numpy()
            )

            priorities = calculate_mixed_td_errors(td_errors, learning_steps.numpy())

            # automatic mixed precision training
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_norm)
            self.optimizer.step()

            self.num_updates += 1

            self.priority_queue.put((idxes, priorities, old_ptr, loss.item()))

            # store new weights in shared memory
            if self.num_updates % 4 == 0:
                self.store_weights()

            # update target net
            if self.num_updates % self.target_net_update_interval == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            # save model
            if self.num_updates % self.save_interval == 0:
                if not os.path.exists(f"models_{self.config['exploration_type']}"):
                    os.mkdir(f"models_{self.config['exploration_type']}")
                torch.save(
                    (
                        self.online_net.state_dict(),
                        self.num_updates,
                        env_steps,
                        (time.time() - start_time) / 60,
                    ),
                    os.path.join(
                        f"models_{self.config['exploration_type']}", "{}{}.pth".format(self.game_name, self.num_updates)
                    ),
                )

    @staticmethod
    def value_rescale(value, eps=1e-3):
        return value.sign() * ((value.abs() + 1).sqrt() - 1) + eps * value

    @staticmethod
    def inverse_value_rescale(value, eps=1e-3):
        temp = ((1 + 4 * eps * (value.abs() + 1 + eps)).sqrt() - 1) / (2 * eps)
        return value.sign() * (temp.square() - 1)