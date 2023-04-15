import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.runners.base_runner import BaseRunner


class TempBuffer:
    def __init__(self, size, n_obs, device):
        self.size = size
        self.rewards = torch.zeros(size, 1, device=device)
        self.dones = torch.zeros(size, 1, device=device).byte()
        self.obs = torch.zeros(size, n_obs, device=device)
        # TODO: info is already empty from env
        self.infos = {}
        self.i = 0

    def pop(self):
        return self.rewards, self.dones, self.infos, self.obs

    def add(self, obs, rewards, dones, infos):
        n = rewards.shape[0]
        e = min(self.i + n, self.size)
        self.rewards[self.i:e].copy_(rewards.unsqueeze(1))
        self.dones[self.i:e].copy_(dones.unsqueeze(1))
        self.obs[self.i:e].copy_(obs.unsqueeze(1))
        # self.infos.append(infos)
        self.i = e

    def is_full(self):
        return self.i == self.size

    def clear(self):
        self.i = 0


class HierarchicalRunner(BaseRunner):
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        self.num_actions = self.env.num_actions if 'num_actions' not in train_cfg else train_cfg['num_actions']

        super().__init__(env, train_cfg, log_dir, device)

    def init_networks(self):
        self.high_obs_idx = self.policy_cfg["high"]["obs_idx"]
        self.mid_obs_idx = self.policy_cfg["mid"]["obs_idx"]
        self.low_obs_idx = self.policy_cfg["low"]["obs_idx"]

        self.low_num_steps = self.policy_cfg["low"]["num_steps"]
        self.mid_num_steps = self.policy_cfg["mid"]["num_steps"]

        high_num_actions = self.policy_cfg["high"]["num_actions"]
        del self.policy_cfg["high"]["num_actions"]
        mid_num_actions = self.policy_cfg["mid"]["num_actions"]
        del self.policy_cfg["mid"]["num_actions"]
        low_num_actions = self.policy_cfg["low"]["num_actions"]
        del self.policy_cfg["low"]["num_actions"]

        high_num_obs = self.policy_cfg["high"]["num_obs"]
        high_num_critic_obs = high_num_obs
        mid_num_obs = self.policy_cfg["mid"]["num_obs"]
        self.mid_num_obs = mid_num_obs
        mid_num_critic_obs = mid_num_obs
        low_num_obs = self.policy_cfg["low"]["num_obs"]
        low_num_critic_obs = low_num_obs

        high_num_steps_per_env = self.policy_cfg["high"]["num_steps_per_env"]
        mid_num_steps_per_env = self.policy_cfg["mid"]["num_steps_per_env"]
        low_num_steps_per_env = self.policy_cfg["low"]["num_steps_per_env"]

        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO

        high_policy_cfg = self.policy_cfg["high"]
        mid_policy_cfg = self.policy_cfg["mid"]
        low_policy_cfg = self.policy_cfg["low"]
        del self.policy_cfg["high"]
        del self.policy_cfg["mid"]
        del self.policy_cfg["low"]

        high_actor_critic: ActorCritic = actor_critic_class(high_num_obs,
                                                            high_num_critic_obs,
                                                            high_num_actions,
                                                            action_activation='sigmoid',
                                                            **high_policy_cfg).to(self.device)
        self.high_alg: PPO = alg_class(high_actor_critic, device=self.device, **self.alg_cfg)

        mid_actor_critic: ActorCritic = actor_critic_class(mid_num_obs,
                                                           mid_num_critic_obs,
                                                           mid_num_actions,
                                                           **mid_policy_cfg).to(self.device)
        self.mid_alg: PPO = alg_class(mid_actor_critic, device=self.device, **self.alg_cfg)

        low_actor_critic: ActorCritic = actor_critic_class(low_num_obs,
                                                           low_num_critic_obs,
                                                           low_num_actions,
                                                           **low_policy_cfg).to(self.device)
        self.low_alg: PPO = alg_class(low_actor_critic, device=self.device, **self.alg_cfg)

        self.high_alg.init_storage(self.env.num_envs, high_num_steps_per_env, [high_num_obs], [high_num_critic_obs],
                                   [high_num_actions])
        self.mid_alg.init_storage(self.env.num_envs, mid_num_steps_per_env, [mid_num_critic_obs],
                                  [mid_num_critic_obs],
                                  [mid_num_actions])
        self.low_alg.init_storage(self.env.num_envs, low_num_steps_per_env, [low_num_obs],
                                  [low_num_obs],
                                  [low_num_actions])

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.high_alg.actor_critic.train()  # switch to train mode (for dropout for example)
        self.mid_alg.actor_critic.train()
        self.low_alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        mid_lenbuffer = deque(maxlen=100)
        mid_rewbuffer = deque(maxlen=100)
        low_rewbuffer = deque(maxlen=100)
        low_lenbuffer = deque(maxlen=100)

        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_mid_rew_sum = torch.zeros_like(cur_reward_sum)
        cur_mid_episode_length = torch.zeros_like(cur_episode_length)
        cur_low_rew_sum = torch.zeros_like(cur_reward_sum)
        cur_low_episode_length = torch.zeros_like(cur_episode_length)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        it_low = torch.tensor(0, dtype=torch.int, device=self.device).unsqueeze(0).repeat(
            self.env.num_envs, 1)

        it_mid = torch.tensor(0, dtype=torch.int, device=self.device).unsqueeze(0).repeat(
            self.env.num_envs, 1)

        i_low = 0
        i_mid = 0

        high_actions = torch.zeros(self.env.num_envs, 4, dtype=torch.float32, device=self.device)
        high_actions[:, 0] = 1  # x
        high_actions[:, 1] = 0.4  # y
        mid_actions = torch.rand(self.env.num_envs, 6, dtype=torch.float32, device=self.device) - 0.5
        low_dones = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        mid_dones = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)

        mid_temp_buffer = TempBuffer(self.env.num_envs, self.mid_num_obs, self.device)

        low_it = torch.zeros_like(low_dones, dtype=torch.int)
        mid_it = torch.zeros_like(low_dones, dtype=torch.int)
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    high_obs = self.get_high_obs(obs, mid_dones)
                    high_critic_obs = high_obs
                    new_high_actions = self.high_alg.act(high_obs, high_critic_obs)
                    high_actions[mid_dones] = new_high_actions[mid_dones]

                    mid_obs = self.get_mid_obs(obs, high_actions, low_dones)
                    # mid_obs = mid_obs.view(self.env.num_envs, -1)[low_dones]
                    mid_critic_obs = mid_obs
                    new_mid_actions = self.mid_alg.act(mid_obs, mid_critic_obs)
                    mid_actions[low_dones] = new_mid_actions[low_dones]

                    low_obs = self.get_low_obs(obs, mid_actions)
                    low_critic_obs = low_obs
                    low_actions = self.low_alg.act(low_obs, low_critic_obs)
                    obs, privileged_obs, high_rewards, dones, infos = self.env.step(low_actions, high_actions,
                                                                                    mid_actions)
                    mid_rewards, low_rewards = self.env.get_reward_mid_low()
                    mid_dones, low_dones = self.env.get_done_mid_low()

                    # if torch.sum(low_dones) > 0:
                    #     mid_temp_buffer.add(mid_obs, mid_rewards[low_dones], mid_dones[low_dones], infos)
                    #     if mid_temp_buffer.is_full():
                    #         self.mid_alg.process_env_step(*mid_temp_buffer.pop())
                    #         mid_temp_buffer.clear()

                    low_dones |= low_it > self.low_num_steps
                    low_dones |= dones
                    low_it[low_dones] = 0
                    low_it = low_it + 1

                    mid_dones |= mid_it > self.mid_num_steps
                    mid_dones |= dones
                    mid_it[mid_dones] = 0
                    mid_it = mid_it + 1

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, high_rewards, dones = obs.to(self.device), critic_obs.to(
                        self.device), high_rewards.to(
                        self.device), dones.to(self.device)
                    low_rewards, mid_rewards, low_dones, mid_dones = low_rewards.to(self.device), mid_rewards.to(
                        self.device), low_dones.to(self.device), mid_dones.to(self.device)

                    self.low_alg.process_env_step(low_rewards, low_dones, infos)

                    self.mid_alg.process_env_step(mid_rewards, mid_dones, infos)

                    self.high_alg.process_env_step(high_rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += high_rewards
                        cur_mid_rew_sum += mid_rewards
                        cur_low_rew_sum += low_rewards
                        cur_episode_length += 1
                        cur_mid_episode_length += 1
                        cur_low_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        low_new_ids = (low_dones > 0).nonzero(as_tuple=False)
                        mid_new_ids = (mid_dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[low_new_ids][:, 0].cpu().numpy().tolist())
                        mid_rewbuffer.extend(cur_mid_rew_sum[mid_new_ids][:, 0].cpu().numpy().tolist())
                        low_rewbuffer.extend(cur_low_rew_sum[low_new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        mid_lenbuffer.extend(cur_mid_episode_length[mid_new_ids][:, 0].cpu().numpy().tolist())
                        low_lenbuffer.extend(cur_low_episode_length[low_new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_low_rew_sum[low_new_ids] = 0
                        cur_mid_rew_sum[mid_new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        cur_mid_episode_length[mid_new_ids] = 0
                        cur_low_episode_length[low_new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.low_alg.compute_returns(low_critic_obs)
                self.mid_alg.compute_returns(mid_critic_obs)
                self.high_alg.compute_returns(high_critic_obs)

                # stop = time.time()
                # collection_time = stop - start
                #
                # start = stop

                # self.high_alg.compute_returns(high_obs)

            # Update the networks
            low_value_loss, low_surrogate_loss = self.low_alg.update()
            mid_value_loss, mid_surrogate_loss = self.mid_alg.update()
            high_value_loss, high_surrogate_loss = self.high_alg.update()

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        high_mean_std = self.high_alg.actor_critic.std.mean()
        mid_mean_std = self.mid_alg.actor_critic.std.mean()
        low_mean_std = self.low_alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/high/value_function', locs['high_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/mid/value_function', locs['mid_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/low/value_function', locs['low_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/high/surrogate', locs['high_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/mid/surrogate', locs['mid_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/low/surrogate', locs['low_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/high/learning_rate', self.high_alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/mid/learning_rate', self.mid_alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/low/learning_rate', self.low_alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/high/mean_noise_std', high_mean_std.item(), locs['it'])
        self.writer.add_scalar('Policy/mid/mean_noise_std', mid_mean_std.item(), locs['it'])
        self.writer.add_scalar('Policy/low/mean_noise_std', low_mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mn_rew_mid', statistics.mean(locs['mid_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mn_rew_low', statistics.mean(locs['low_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mn_epi_len_mid', statistics.mean(locs['mid_lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mn_epi_len_low', statistics.mean(locs['low_lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss High:':>{pad}} {locs['high_value_loss']:.4f}\n"""
                          f"""{'Value function loss Mid:':>{pad}} {locs['mid_value_loss']:.4f}\n"""
                          f"""{'Value function loss Low:':>{pad}} {locs['low_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss High:':>{pad}} {locs['high_surrogate_loss']:.4f}\n"""
                          f"""{'Surrogate loss Mid:':>{pad}} {locs['mid_surrogate_loss']:.4f}\n"""
                          f"""{'Surrogate loss Low:':>{pad}} {locs['low_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std High:':>{pad}} {high_mean_std.item():.2f}\n"""
                          f"""{'Mean action noise std Mid:':>{pad}} {mid_mean_std.item():.2f}\n"""
                          f"""{'Mean action noise std Low:':>{pad}} {low_mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss High:':>{pad}} {locs['high_value_loss']:.4f}\n"""
                          f"""{'Value function loss Mid:':>{pad}} {locs['mid_value_loss']:.4f}\n"""
                          f"""{'Value function loss Low:':>{pad}} {locs['low_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss High:':>{pad}} {locs['high_surrogate_loss']:.4f}\n"""
                          f"""{'Surrogate loss Mid:':>{pad}} {locs['mid_surrogate_loss']:.4f}\n"""
                          f"""{'Surrogate loss Low:':>{pad}} {locs['low_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std High:':>{pad}} {high_mean_std.item():.2f}\n"""
                          f"""{'Mean action noise std Mid:':>{pad}} {mid_mean_std.item():.2f}\n"""
                          f"""{'Mean action noise std Low:':>{pad}} {low_mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'high_model_state_dict': self.high_alg.actor_critic.state_dict(),
            'mid_model_state_dict': self.mid_alg.actor_critic.state_dict(),
            'low_model_state_dict': self.low_alg.actor_critic.state_dict(),
            'high_optimizer_state_dict': self.high_alg.optimizer.state_dict(),
            'mid_optimizer_state_dict': self.mid_alg.optimizer.state_dict(),
            'low_optimizer_state_dict': self.low_alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.high_alg.actor_critic.load_state_dict(loaded_dict['high_model_state_dict'])
        self.mid_alg.actor_critic.load_state_dict(loaded_dict['mid_model_state_dict'])
        self.low_alg.actor_critic.load_state_dict(loaded_dict['low_model_state_dict'])
        if load_optimizer:
            self.high_alg.optimizer.load_state_dict(loaded_dict['high_optimizer_state_dict'])
            self.mid_alg.optimizer.load_state_dict(loaded_dict['mid_optimizer_state_dict'])
            self.low_alg.optimizer.load_state_dict(loaded_dict['low_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_high_obs(self, obs, mid_dones):
        return torch.cat([obs[..., self.high_obs_idx], mid_dones.unsqueeze(1)], dim=1)

    def get_mid_obs(self, obs, high_actions, low_dones, t=None):
        return torch.cat([obs[..., self.mid_obs_idx], high_actions, low_dones.unsqueeze(1)], dim=1)

    def get_low_obs(self, obs, mid_actions, t=None):
        return torch.cat([obs[..., self.low_obs_idx], mid_actions], dim=1)

    def get_inference_policy(self, device=None):
        raise NotImplementedError
        self.high_alg.actor_critic.eval()  # switch to evaluation
        self.mid_alg.actor_critic.eval()  # switch to evaluation
        self.low_alg.actor_critic.eval()  # switch to evaluation

        if device is not None:
            self.low_alg.actor_critic.to(device)
            self.mid_alg.actor_critic.to(device)
            self.high_alg.actor_critic.to(device)

        def policy_fn(obs):
            # Get the current high-level observation from the environment
            high_obs = self.get_high_obs(obs).to(self.device)

            # Get the current mid-level observation from the environment
            mid_obs = self.get_mid_obs(obs).to(self.device)

            # Get the current low-level observation from the environment
            low_obs = self.env.get_observations().to(self.device)

            # Get the current high-level action using the high-level policy function
            high_act = self.high_alg.actor_critic.act_inference(high_obs.to(self.device))

            # Get the current mid-level action using the mid-level policy function
            mid_act = self.mid_alg.actor_critic.act_inference(torch.cat([mid_obs, high_act], dim=1))

            # Generate the low-level action using the low-level policy function
            low_act = self.low_alg.actor_critic.act_inference(torch.cat([low_obs, mid_act], dim=1))

            return low_act

        return policy_fn
