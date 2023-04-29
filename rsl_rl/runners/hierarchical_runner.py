import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.runners.base_runner import BaseRunner
from rsl_rl.modules.meta_learner import MetaLearner


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

        self.high_batch_n = 1 * 15
        self.mid_batch_n = 15
        self.low_batch_n = 1

    def init_networks(self):
        self.high_obs_idx = self.policy_cfg["high"]["obs_idx"]
        self.mid_obs_idx = self.policy_cfg["mid"]["obs_idx"]
        self.low_obs_idx = self.policy_cfg["low"]["obs_idx"]

        self.low_num_steps = self.policy_cfg["low"]["num_steps"]
        self.mid_num_steps = self.policy_cfg["mid"]["num_steps"]
        self.high_num_steps = self.policy_cfg["high"]["num_steps"]
        # self.high_num_steps *= self.env.max_episode_length_s

        high_num_actions = self.policy_cfg["high"]["num_actions"]
        self.high_num_actions = high_num_actions
        del self.policy_cfg["high"]["num_actions"]
        mid_num_actions = self.policy_cfg["mid"]["num_actions"]
        self.mid_num_actions = mid_num_actions
        del self.policy_cfg["mid"]["num_actions"]
        low_num_actions = self.policy_cfg["low"]["num_actions"]
        self.low_num_actions = low_num_actions
        del self.policy_cfg["low"]["num_actions"]

        high_num_obs = self.policy_cfg["high"]["num_obs"]
        self.high_num_obs = high_num_obs
        high_num_critic_obs = high_num_obs
        mid_num_obs = self.policy_cfg["mid"]["num_obs"]
        self.mid_num_obs = mid_num_obs
        mid_num_critic_obs = mid_num_obs
        low_num_obs = self.policy_cfg["low"]["num_obs"]
        self.low_num_obs = low_num_obs
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

        self.meta_learner = None
        if self.policy_cfg["use_meta"]:
            self.meta_learner = MetaLearner(device=self.device, **self.policy_cfg["meta"])

        high_actor_critic: ActorCritic = actor_critic_class(high_num_obs,
                                                            high_num_critic_obs,
                                                            high_num_actions,
                                                            action_activation='tanh',
                                                            min_std=1,
                                                            down_std_action_dim=self.mid_num_actions,
                                                            dropout_prob=0.1,
                                                            **high_policy_cfg).to(self.device)
        high_actor_critic.upp_std_coeff = high_actor_critic.upp_std_coeff.to(self.device)
        high_actor_critic.std_coeff = high_actor_critic.std_coeff.to(self.device)
        self.high_alg: PPO = alg_class(high_actor_critic, device=self.device, **self.alg_cfg['high'])

        mid_actor_critic: ActorCritic = actor_critic_class(mid_num_obs,
                                                           mid_num_critic_obs,
                                                           mid_num_actions,
                                                           action_activation='tanh',
                                                           min_std=1,
                                                           down_std_action_dim=self.low_num_actions,
                                                           dropout_prob=0.5,
                                                           **mid_policy_cfg).to(self.device)
        self.mid_alg: PPO = alg_class(mid_actor_critic, device=self.device, **self.alg_cfg['mid'])
        mid_actor_critic.upp_std_coeff = mid_actor_critic.upp_std_coeff.to(self.device)
        mid_actor_critic.std_coeff = mid_actor_critic.std_coeff.to(self.device)

        low_actor_critic: ActorCritic = actor_critic_class(low_num_obs,
                                                           low_num_critic_obs,
                                                           low_num_actions,
                                                           action_activation='tanh',
                                                           min_std=1,
                                                           dropout_prob=0.5,
                                                           # std_mode='adaptive',
                                                           **low_policy_cfg).to(self.device)
        self.low_alg: PPO = alg_class(low_actor_critic, device=self.device, **self.alg_cfg['low'])
        low_actor_critic.upp_std_coeff = low_actor_critic.upp_std_coeff.to(self.device)
        low_actor_critic.std_coeff = low_actor_critic.std_coeff.to(self.device)

        self.high_alg.init_storage(self.env.num_envs, high_num_steps_per_env, [high_num_obs], [high_num_critic_obs],
                                   [high_num_actions])
        self.mid_alg.init_storage(self.env.num_envs, mid_num_steps_per_env, [mid_num_critic_obs],
                                  [mid_num_critic_obs],
                                  [mid_num_actions])
        self.low_alg.init_storage(self.env.num_envs, low_num_steps_per_env, [low_num_obs],
                                  [low_num_obs],
                                  [low_num_actions])

    def get_step(self, obs):
        return obs[0, -1].item()

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
        rewbuffer = deque(maxlen=int(self.env.max_episode_length))
        lenbuffer = deque(maxlen=int(self.env.max_episode_length))
        mid_lenbuffer = deque(maxlen=int(self.env.max_episode_length))
        mid_rewbuffer = deque(maxlen=int(self.env.max_episode_length))
        low_rewbuffer = deque(maxlen=int(self.env.max_episode_length))
        low_lenbuffer = deque(maxlen=int(self.env.max_episode_length))

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

        high_actions = torch.zeros(self.env.num_envs, self.high_num_actions, dtype=torch.float32, device=self.device)
        high_actions_ori = torch.zeros(self.env.num_envs, self.high_num_actions, dtype=torch.float32,
                                       device=self.device)
        mid_actions = torch.zeros(self.env.num_envs, self.mid_num_actions, dtype=torch.float32, device=self.device)
        mid_actions_ori = torch.zeros(self.env.num_envs, self.mid_num_actions, dtype=torch.float32, device=self.device)
        low_actions = torch.zeros(self.env.num_envs, self.low_num_actions, dtype=torch.float32, device=self.device)
        low_actions_ori = torch.zeros(self.env.num_envs, self.low_num_actions, dtype=torch.float32, device=self.device)
        low_dones = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        low_timeout = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        mid_dones = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        mid_timeout = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)

        mid_rew_buf = torch.zeros(self.env.num_envs, 1, dtype=torch.float32, device=self.device)
        high_rew_buf = torch.zeros(self.env.num_envs, 1, dtype=torch.float32, device=self.device)

        mid_temp_buffer = TempBuffer(self.env.num_envs, self.mid_num_obs, self.device)

        high_it = torch.zeros_like(low_dones, dtype=torch.int)
        low_it = torch.zeros_like(low_dones, dtype=torch.int)
        midn = 0
        lown = 0
        highn = 0
        high_push = 0
        high_add = 0

        mid_surrogate_loss = 0
        mid_value_loss = 0

        high_surrogate_loss = 0
        high_value_loss = 0

        dones = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device).unsqueeze(1)

        low_return = 0
        mid_return = 0
        high_return = 0

        min_a = torch.tensor([-0.1, -0.1], device=self.device)
        max_a = torch.tensor([0.1, 0.1], device=self.device)

        step = 0

        meta_loss = 0

        rew_r = torch.ones(4, dtype=torch.float32, device=self.device)
        rew_r[1] = 0
        rew_r[2] = 0
        rew_r[3] = 0

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            for train_step in range(self.num_steps_per_env):
                with torch.inference_mode():
                    # step = self.get_step(obs)
                    obs[:, -1] /= self.env.max_episode_length
                    hi = step % self.high_num_steps
                    mid_update = (hi == self.high_num_steps - 1)  # or dones[0]
                    high_update = mid_update
                    if hi == 0:
                        high_it[:] = hi
                        high_obs = self.get_high_obs(obs, high_actions, mid_dones)
                        high_critic_obs = high_obs
                        self.high_alg.actor_critic.update_std_max(1 - self.mid_alg.actor_critic.pstd.mean())
                        high_actions_ori = self.high_alg.act(high_obs, high_critic_obs)
                        # high_actions[:] *= self.high_actions_scale
                        high_actions, h_cover, h_clip = self.clip_action(high_actions_ori * 1.1, (min_a, max_a))
                        self.env.update_high_actions(high_actions)
                        h_comb_std_coeff = self._compute_combined_std_coeff(h_cover, h_clip, target_coverage=0.8)
                        self.high_alg.actor_critic.update_std_coeff(h_comb_std_coeff)

                    mi = step % self.mid_num_steps
                    cmi = (step % self.high_num_steps) // self.mid_num_steps
                    if mi == 0:
                        # mid_obs = mid_obs.view(self.env.num_envs, -1)[low_dones]
                        mid_obs = self.get_mid_obs(obs, mid_actions_ori, high_actions, low_dones)
                        mid_critic_obs = mid_obs
                        self.mid_alg.actor_critic.update_std_max(1 - self.low_alg.actor_critic.pstd.mean())
                        mid_actions_ori = self.mid_alg.act(mid_obs, mid_critic_obs)
                        # mid_actions[:] *= self.mid_actions_scale
                        mid_actions, m_cover, m_clip = self.clip_action(mid_actions_ori * 1.1, (-3.14 / 2, 3.14 / 2))
                        self.env.update_mid_actions(mid_actions)
                        m_comb_std_coeff = self._compute_combined_std_coeff(m_cover, m_clip, target_coverage=0.6)
                        self.mid_alg.actor_critic.update_std_coeff(m_comb_std_coeff)

                    cli = (step % self.high_num_steps) % self.mid_num_steps
                    low_update = (cli == self.mid_num_steps - 1)  # or dones[0]
                    mid_update &= low_update
                    mid_timeout[:] = mid_update
                    # for li in range(self.low_num_steps):
                    low_it[:] = cli
                    low_timeout[:] = low_update
                    # mid_low_timeout = mid_timeout & low_timeout
                    low_obs = self.get_low_obs(obs, low_actions_ori, mid_actions_ori, cli)
                    low_critic_obs = low_obs
                    low_actions_ori = self.low_alg.act(low_obs, low_critic_obs)
                    # low_actions_scale = 1.2 
                    low_actions, l_cover, l_clip = self.clip_action(low_actions_ori, (-400, 400))
                    l_comb_std_coeff = self._compute_combined_std_coeff(l_cover, l_clip)
                    self.low_alg.actor_critic.update_std_coeff(l_comb_std_coeff)
                    obs, privileged_obs, high_rewards, dones, infos = self.env.step(low_actions, low_timeout,
                                                                                    mid_timeout,
                                                                                    allow_reset=high_update)

                    new_mid_rewards, low_rewards = self.env.get_reward_mid_low()

                    high_penalty = torch.clip(high_rewards, max=0)
                    high_rewards = torch.clip(high_rewards, min=0)
                    new_high_rewards = high_rewards + high_penalty * rew_r[0]
                    new_mid_rewards += high_penalty * rew_r[1]
                    low_rewards += high_penalty * rew_r[2]

                    high_dones, mid_dones, low_dones = self.env.get_done_levels()

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, new_high_rewards, dones = obs.to(self.device), critic_obs.to(
                        self.device), new_high_rewards.unsqueeze(1).to(
                        self.device), dones.to(self.device)
                    low_rewards, new_mid_rewards, low_dones, mid_dones = low_rewards.to(
                        self.device), new_mid_rewards.unsqueeze(1).to(
                        self.device), low_dones.to(self.device), mid_dones.to(self.device)

                    mid_rew_buf += new_mid_rewards
                    high_rew_buf += new_high_rewards

                    self.low_alg.process_env_step(low_rewards, low_dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += new_high_rewards[:, 0]
                        cur_mid_rew_sum += new_mid_rewards[:, 0]
                        cur_low_rew_sum += low_rewards
                        cur_episode_length += 1
                        cur_mid_episode_length += 1
                        cur_low_episode_length += 1
                        # TODO: this is not accurate
                        new_ids = ((high_update | dones) > 0).nonzero(as_tuple=False)
                        low_new_ids = ((low_update | dones) > 0).nonzero(as_tuple=False)
                        mid_new_ids = ((mid_update | dones) > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
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

                    if low_update:
                        low_return += 1
                        # if self.mid_alg.transition.critic_observations is None:
                        # print(step.detach().cpu().numpy())
                        self.mid_alg.process_env_step(mid_rew_buf, mid_dones.unsqueeze(1), infos)
                        mid_rew_buf[:] = 0

                    if low_return == self.low_batch_n:
                        self.low_alg.compute_returns(low_critic_obs)

                    if mid_update:
                        mid_return += 1
                        high_add += 1

                    if mid_return == self.mid_batch_n:
                        self.mid_alg.compute_returns(mid_critic_obs)

                    if high_update:
                        high_return += 1
                        high_push += 1
                        self.high_alg.process_env_step(high_rew_buf, high_dones.unsqueeze(1), infos)

                        high_rew_buf[:] = 0

                    if high_return == self.high_batch_n:
                        self.high_alg.compute_returns(high_critic_obs)

                if low_return == self.low_batch_n:
                    lown += 1
                    low_value_loss, low_surrogate_loss = self.low_alg.update()
                    low_return = 0

                if mid_return == self.mid_batch_n:
                    midn += 1
                    mid_value_loss, mid_surrogate_loss = self.mid_alg.update()
                    self.low_alg.actor_critic.update_upp_std_coeff(self.mid_alg.actor_critic.down_std_coeff)
                    mid_return = 0

                if high_return == self.high_batch_n:
                    highn += 1
                    high_value_loss, high_surrogate_loss = self.high_alg.update()
                    self.mid_alg.actor_critic.update_upp_std_coeff(self.high_alg.actor_critic.down_std_coeff)

                    if self.meta_learner is not None:
                        all_actions = torch.tensor([statistics.mean(rewbuffer), statistics.mean(mid_rewbuffer),
                                                    statistics.mean(low_rewbuffer), statistics.mean(lenbuffer),
                                                    statistics.mean(mid_lenbuffer), statistics.mean(low_lenbuffer),
                                                    statistics.stdev(rewbuffer), statistics.stdev(mid_rewbuffer),
                                                    statistics.stdev(low_rewbuffer), statistics.stdev(lenbuffer),
                                                    statistics.stdev(mid_lenbuffer), statistics.stdev(low_lenbuffer),
                                                    self.low_alg.actor_critic.std.mean().item(),
                                                    self.mid_alg.actor_critic.std.mean().item(),
                                                    self.high_alg.actor_critic.std.mean().item()],
                                                   dtype=torch.float32, device=self.device)

                        rew_r, meta_loss = self.meta_learner.update(all_actions,
                                                                    (high_rewards + high_penalty).unsqueeze(1))
                    high_return = 0

                step += 1
                if step % self.env.max_episode_length == 0:
                    step = 0

            stop = time.time()
            collection_time = stop - start

            # Learning step
            start = stop

            print('highn: ', highn)
            print('midn: ', midn)
            print('lown: ', lown)
            print('high_push: ', high_push)
            print('high_add: ', high_add)

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

        mid_upp_std_coeff = self.mid_alg.actor_critic.upp_std_coeff.mean()
        low_upp_std_coeff = self.low_alg.actor_critic.upp_std_coeff.mean()
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
        self.writer.add_scalar('Loss/meta/loss', locs['meta_loss'], locs['it'])
        self.writer.add_scalar('Policy/high/mean_noise_std', high_mean_std.item(), locs['it'])
        self.writer.add_scalar('Policy/mid/mean_noise_std', mid_mean_std.item(), locs['it'])
        self.writer.add_scalar('Policy/low/mean_noise_std', low_mean_std.item(), locs['it'])

        self.writer.add_scalar('Policy/low/coverage', locs['l_cover'], locs['it'])
        self.writer.add_scalar('Policy/high/coverage', locs['h_cover'], locs['it'])
        self.writer.add_scalar('Policy/mid/coverage', locs['m_cover'], locs['it'])
        self.writer.add_scalar('Policy/low/clipped_ratio', locs['l_clip'], locs['it'])
        self.writer.add_scalar('Policy/mid/clipped_ratio', locs['m_clip'], locs['it'])
        self.writer.add_scalar('Policy/high/clipped_ratio', locs['h_clip'], locs['it'])
        self.writer.add_scalar('Policy/high/std_coeff', locs['h_comb_std_coeff'], locs['it'])
        self.writer.add_scalar('Policy/mid/std_coeff', locs['m_comb_std_coeff'], locs['it'])
        self.writer.add_scalar('Policy/low/std_coeff', locs['l_comb_std_coeff'], locs['it'])
        self.writer.add_scalar('Policy/high/wstd', self.high_alg.actor_critic.wstd.mean().item(), locs['it'])
        self.writer.add_scalar('Policy/mid/wstd', self.mid_alg.actor_critic.wstd.mean().item(), locs['it'])
        self.writer.add_scalar('Policy/low/wstd', self.low_alg.actor_critic.wstd.mean().item(), locs['it'])
        self.writer.add_scalar('Policy/std_decay', self.low_alg.actor_critic.std_decay.mean().item(), locs['it'])

        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])

        self.writer.add_histogram('Actions/low_act_0_dist', locs['low_actions'][..., 0], locs['it'])
        self.writer.add_histogram('Actions/low_act_1_dist', locs['low_actions'][..., 1], locs['it'])
        self.writer.add_histogram('Actions/low_act_2_dist', locs['low_actions'][..., 2], locs['it'])

        self.writer.add_histogram('Actions/mid_act_0_dist', locs['mid_actions'][..., 0], locs['it'])
        self.writer.add_histogram('Actions/mid_act_1_dist', locs['mid_actions'][..., 1], locs['it'])
        self.writer.add_histogram('Actions/mid_act_2_dist', locs['mid_actions'][..., 2], locs['it'])

        self.writer.add_histogram('Actions/high_act_pos_0_dist', locs['high_actions'][..., 0], locs['it'])
        self.writer.add_histogram('Actions/high_act_pos_1_dist', locs['high_actions'][..., 1], locs['it'])
        # self.writer.add_histogram('Actions/high_act_vel_0_dist', locs['high_actions'][..., 2], locs['it'])
        # self.writer.add_histogram('Actions/high_act_vel_1_dist', locs['high_actions'][..., 3], locs['it'])

        self.writer.add_scalar('Meta/rew_r_high', locs['rew_r'][..., 0], locs['it'])
        self.writer.add_scalar('Meta/rew_r_mid', locs['rew_r'][..., 1], locs['it'])
        self.writer.add_scalar('Meta/rew_r_low', locs['rew_r'][..., 2], locs['it'])
        # self.writer.add_scalar('Meta/stds_r_high', locs['stds_r'][..., 0], locs['it'])
        # self.writer.add_scalar('Meta/stds_r_mid', locs['stds_r'][..., 1], locs['it'])
        # self.writer.add_scalar('Meta/stds_r_low', locs['stds_r'][..., 2], locs['it'])

        if len(locs['rewbuffer']) > 0 and len(locs['lenbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mn_rew_mid', statistics.mean(locs['mid_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mn_rew_low', statistics.mean(locs['low_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mn_epi_len_mid', statistics.mean(locs['mid_lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mn_epi_len_low', statistics.mean(locs['low_lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
            self.writer.add_scalar('Train/curriculum/levels', self.env.curri_level_buf.mean().item(), locs['it'])
            self.writer.add_scalar('Train/success_rate', self.env.success_buf.sum().item(), locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0 and len(locs['lenbuffer']) > 0:
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
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean curriculum level:':>{pad}} {self.env.curri_level_buf.mean().item():.4f}\n""")
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

    def get_high_obs(self, obs, high_actions, mid_dones):
        # t to tensor, with the same shape as mid_dones
        return torch.cat([high_actions, mid_dones.unsqueeze(1), obs[..., self.high_obs_idx]], dim=1)

    def get_mid_obs(self, obs, mid_actions, high_actions, low_dones):
        return torch.cat([mid_actions, high_actions, low_dones.unsqueeze(1), obs[..., self.mid_obs_idx]], dim=1)

    def get_low_obs(self, obs, low_actions, mid_actions, cli):
        t = torch.tensor([cli / 20], dtype=torch.float32, device=obs.device).repeat(obs.shape[0], 1)
        # return torch.cat([t,low_actions, mid_actions, obs[..., self.low_obs_idx]], dim=1)
        return torch.cat([low_actions, mid_actions, obs[..., self.low_obs_idx], t], dim=1)

    def get_inference_policy(self, device=None):
        self.high_alg.actor_critic.eval()  # switch to evaluation
        self.mid_alg.actor_critic.eval()  # switch to evaluation
        self.low_alg.actor_critic.eval()  # switch to evaluation

        if device is not None:
            self.low_alg.actor_critic.to(device)
            self.mid_alg.actor_critic.to(device)
            self.high_alg.actor_critic.to(device)

        def policy_fn(obs, high_actions, mid_dones, mid_actions, low_dones, low_actions, low_timeout, mid_timeout):
            step = self.get_step(obs)
            obs[:, -1] /= self.env.max_episode_length
            hi = step % self.high_num_steps
            mid_update = (hi == self.high_num_steps - 1)  # or dones[0]
            if hi == 0:
                high_obs = self.get_high_obs(obs, high_actions, mid_dones)
                high_critic_obs = high_obs
                high_actions = self.high_alg.act(high_obs, high_critic_obs)
                # high_actions[:] *= self.high_actions_scale
                high_actions = self.env.map_high_actions(high_actions)

            mi = step % self.mid_num_steps
            if mi == 0:
                # mid_obs = mid_obs.view(self.env.num_envs, -1)[low_dones]
                mid_obs = self.get_mid_obs(obs, mid_actions, high_actions, low_dones)
                mid_critic_obs = mid_obs
                mid_actions = self.mid_alg.act(mid_obs, mid_critic_obs)
                # mid_actions[:] *= self.mid_actions_scale
                mid_actions = self.env.map_mid_actions(mid_actions)

            cli = (step % self.high_num_steps) % self.mid_num_steps
            low_update = (cli == self.mid_num_steps - 1)  # or dones[0]
            mid_update &= low_update
            mid_timeout[:] = mid_update
            # for li in range(self.low_num_steps):
            low_timeout[:] = low_update

            low_obs = self.get_low_obs(obs, low_actions, mid_actions)
            low_critic_obs = low_obs
            low_actions = self.low_alg.act(low_obs, low_critic_obs)
            # low_actions[:] *= self.low_actions_scale
            low_actions = self.env.map_low_actions(low_actions)

            return high_actions, mid_actions, low_actions, mid_timeout, low_timeout

        return policy_fn

    def clip_action(self, actions, action_range, scale=True, resample=True):
        coverage = None
        clipped_ratio = None
        if scale:
            coverage = self._calculate_coverage(actions, [-1, 1])
            actions = torch.clamp(actions, -1, 1)
            clipped_index = (actions == -1) | (actions == 1)
            interval = (action_range[1] - action_range[0]) / 2
            mid = (action_range[0] + action_range[1]) / 2
            clipped_actions = actions * interval + mid
        else:
            coverage = self._calculate_coverage(actions, action_range)
            clipped_actions = torch.clamp(actions, action_range[0], action_range[1])
            clipped_index = (clipped_actions == action_range[0]) | (clipped_actions == action_range[1])
            interval = (action_range[1] - action_range[0]) / 2
            mid = (action_range[1] + action_range[0]) / 2

        if resample:
            random_actions = torch.rand_like(actions) * interval * 2 + action_range[0]
            clipped_actions[clipped_index] = random_actions[clipped_index]

        clipped_ratio = torch.sum(clipped_index).float() / actions.numel()

        return clipped_actions, coverage, clipped_ratio

    def _calculate_coverage(self, actions, action_range, threshold=0.01):
        min_val, max_val = action_range
        bins = torch.linspace(min_val, max_val, 100)  # You can adjust the number of bins based on your preference
        hist = torch.histc(actions, bins=bins.shape[0], min=min_val, max=max_val)
        coverage = torch.sum(hist > threshold).float() / hist.shape[0]
        return coverage

    def _compute_combined_std_coeff(self, coverage, clipped_ratio, target_coverage=0.5):
        return (1 - (-target_coverage + coverage)) * (1 - clipped_ratio)
