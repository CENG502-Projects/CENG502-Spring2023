import random
import math
import torch
import torch.nn.functional as F
from scipy.stats import bernoulli

from r2d2 import R2D2, State
from environment import create_env
from local_buffer import LocalBuffer


class Actor:
    def __init__(
        self,
        epsilon: float,
        model,
        sample_queue,
        config
    ):
        
        max_episode_steps = config["max_episode_steps"]
        block_length = config["block_length"]
        exploration_type = config["exploration_type"]
        informed_k = config["informed_k"]
        discount = config["gamma"]
        target_rate = config["target_rate"]

        self.config = config

        self.env = create_env(config["game_name"], noop_start=True)
        self.action_dim = self.env.action_space.n
        self.model = R2D2(self.env.action_space.n)
        self.model.eval()
        self.local_buffer = LocalBuffer(self.action_dim, config)

        self.epsilon = epsilon
        self.shared_model = model
        self.sample_queue = sample_queue
        self.max_episode_steps = max_episode_steps
        self.block_length = block_length
        
        self.exploration_type = exploration_type
        self.informed_k = informed_k
        self.discount = discount
        self.target_rate = target_rate

    def run(self):

        actor_steps = 0

        while True:
            done = False
            agent_state = self.reset()
            episode_steps = 0
            curr_step = 0
            rand = False
            self.local_buffer.reset_full_buffers()

            while not done and episode_steps < self.max_episode_steps:
                with torch.no_grad():
                    q_value, hidden = self.model(agent_state)
                
                # Full exploration/exploitation
                if "experiment-level" in self.exploration_type:  
                    if self.config["start_mode"] == "G":
                        action = torch.argmax(q_value, 1).item()
                    elif self.config["start_mode"] == "X":
                        action = self.env.action_space.sample()
                        
                # Alternate between explore & exploit selected once per episode
                elif "episode-level" in self.exploration_type:
                    # Select the episode mode
                    if episode_steps == 0:
                        episode_mode = "exploit" if random.random() < 0.5 else "explore"
                        
                    if episode_mode == "exploit":
                        action = torch.argmax(q_value, 1).item()
                    else:
                        action = self.env.action_space.sample()
                        
                # Regular epsilon greedy training with irregular exploration steps
                elif "step-level" in self.exploration_type:  
                    if random.random() < self.config["prob_switch_exploit"]:
                        action = self.env.action_space.sample()
                    else:
                        action = torch.argmax(q_value, 1).item()
                        
                # blindly switch to explore mode (either randomly or with a fixed number of steps)
                elif "blind" in self.exploration_type:
                    
                    # Depending on the start mode, initialize the episode
                    if self.config["start_mode"] == "G" and episode_steps <= self.config["exploit_duration"]:
                        action = torch.argmax(q_value, 1).item()
                    elif self.config["start_mode"] == "X" and episode_steps <= self.config["explore_duration"]:
                        action = self.env.action_space.sample()
                        
                    else:
                        # Fixed number of exploit steps & fixed number of explore steps
                        if "fixed" in self.exploration_type:
                            total_length = self.config["explore_duration"] + self.config["exploit_duration"]
                            if self.config["exploit_duration"] < (episode_steps % total_length) < total_length:
                                action = self.env.action_space.sample()
                            else:
                                action = torch.argmax(q_value, 1).item()
                                
                        # Probabilistic number of exploit steps & fixed number of explore steps
                        elif "prob" in self.exploration_type:
                            if curr_step <= episode_steps < curr_step + self.config["explore_duration"]:
                                action = self.env.action_space.sample()
                                rand = True
                            else:
                                action = torch.argmax(q_value, 1).item()
                                rand = False

                            if random.random() < self.config["prob_switch_exploit"] and not rand:
                                curr_step = episode_steps
                
                # Make an informed decision about when to switch to explore mode
                elif "informed" in self.exploration_type:
                    if episode_steps <= self.config["exploit_duration"]:  
                        # Start from exploitation at the beginning of the episode
                        action = torch.argmax(q_value, 1).item()
                    elif curr_step <= episode_steps < curr_step + self.config["explore_duration"]:
                        # If in explore mode, run for a fixed amount of steps
                        action = self.env.action_space.sample()
                    elif (episode_steps % 10) != 0:
                        # Cut down computation of homeostasis by a factor of 10
                        action = torch.argmax(q_value, 1).item()
                    else:  
                        # If in exploit mode, perform homeostasis
                        # Initialize homeostasis variables
                        x_mean = 0
                        x_var = 1
                        x_pos_mean = 1
                        y_t_list = []

                        # Enter homeostasis loop
                        for t in range(episode_steps // 2, episode_steps + 1):  # Cut down computation of homeostasis by a factor of 2
                            # If t is lower than or equal to k, continue with the next value of t
                            if t <= self.informed_k:
                                continue

                            # Calculate V(s_{t-k})
                            try:
                                qvals_prev = torch.from_numpy(self.local_buffer[t - self.informed_k, "qval"][0])
                                p_prev = F.softmax(qvals_prev, dim=-1)
                                v_prev = torch.dot(p_prev, qvals_prev).item()
                                v_prev *= math.pow(self.discount, self.informed_k)
                            except IndexError:
                                print(len(self.local_buffer.full_qval_buffer), t - self.informed_k, flush=True)

                            # Calculate V(s_t)
                            qvals_current = q_value[0]
                            p_current = F.softmax(qvals_current, dim=-1)
                            v_current = torch.dot(p_current, qvals_current)

                            # Calculate the total reward of the actions taken before
                            reward_sum = 0
                            for i in range(1, self.informed_k):
                                try:
                                    reward_sum += \
                                        math.pow(self.discount, i) * self.local_buffer[t - i, "reward"]
                                except IndexError:
                                    print("-", len(self.local_buffer.full_reward_buffer), t - i, flush=True)

                            # Calculate D_promise(t-k, t)
                            d_promise = abs(v_prev - reward_sum - v_current)

                            # Set signal return
                            x_t = d_promise

                            # Calculate the time scale
                            time_scale = min(t, 100 / self.target_rate)

                            # Update moving average
                            x_mean = (1 - (1 / time_scale)) * x_mean + (1 / time_scale) * x_t

                            # Update moving variance
                            x_var = \
                                (1 - (1 / time_scale)) * x_var + (1 / time_scale) * ((x_t - x_mean) ** 2)

                            # Standardise and exponentiate x_pos
                            x_pos = math.exp((x_t - x_mean) / math.sqrt(x_var))

                            # Update transformed moving average
                            x_pos_mean = (1 - (1 / time_scale)) * x_pos_mean + (1 / time_scale) * x_pos

                            # Sample y_t
                            prob = min(1, self.target_rate * (x_pos / x_pos_mean))
                            y_t = bernoulli.rvs(prob, size=1)[0]

                            # Append y_t to y_t_list
                            y_t_list.append(y_t)

                        if y_t_list:
                            # Take the average of switches
                            average_switch = sum(y_t_list) / len(y_t_list)

                            # If average_switch is lower than the target_rate, switch to exploration mode
                            # else, use exploitation
                            if average_switch < self.target_rate:
                                curr_step = episode_steps
                                action = self.env.action_space.sample()
                            else:
                                action = torch.argmax(q_value, 1).item()
    
                # apply action in env
                next_obs, reward, done, _ = self.env.step(action)

                agent_state.update(next_obs, action, reward, hidden)

                episode_steps += 1
                actor_steps += 1

                self.local_buffer.add(
                    action, reward, next_obs, q_value.numpy(), torch.cat(hidden).numpy()
                )

                if done:
                    block = self.local_buffer.finish()
                    self.sample_queue.put(block)

                elif (
                    len(self.local_buffer) == self.block_length
                    or episode_steps == self.max_episode_steps
                ):
                    with torch.no_grad():
                        q_value, hidden = self.model(agent_state)

                    block = self.local_buffer.finish(q_value.numpy())

                    if self.epsilon > 0.01:
                        block[2] = None
                    self.sample_queue.put(block)

                if actor_steps % 400 == 0:
                    self.update_weights()

    def update_weights(self):
        """load the latest weights from shared model"""
        self.model.load_state_dict(self.shared_model.state_dict())

    def reset(self):
        obs = self.env.reset()
        self.local_buffer.reset(obs)

        state = State(torch.from_numpy(obs).unsqueeze(0), self.action_dim)

        return state
