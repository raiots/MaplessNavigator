from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import os
import sys
sys.path.append('../../')
from scripts.train_ddpg.ddpg_networks import ActorNet, CriticNet

class Agent:
    def __init__(self, s_num, a_num, r_s_num, a_net_dim=(256, 256, 256), c_net_dim=(512, 512, 512),
                 mem_size=1000, b_size=128, t_tau=0.01, t_update_steps=5, r_gamma=0.99,
                 a_lr=0.0001, c_lr=0.0001, e_start=0.9, e_end=0.01, e_decay=0.999,
                 e_rand_decay_start=60000, e_rand_decay_step=1, p_window=50, use_p=False, use_cuda=True):
        self.s_num, self.a_num, self.r_s_num = s_num, a_num, r_s_num
        self.mem_size, self.b_size = mem_size, b_size
        self.t_tau, self.t_update_steps = t_tau, t_update_steps
        self.r_gamma = r_gamma
        self.a_lr, self.c_lr = a_lr, c_lr
        self.e_start, self.e_end, self.e_decay = e_start, e_end, e_decay
        self.e_rand_decay_start = e_rand_decay_start
        self.e_rand_decay_step = e_rand_decay_step
        self.p_window, self.use_p = p_window, use_p
        self.use_cuda = use_cuda
        self.e = e_start
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.use_cuda else "cpu")
        self.mem = deque(maxlen=self.mem_size)
        self.a_net = ActorNet(self.r_s_num, self.a_num, *a_net_dim)
        self.c_net = CriticNet(self.s_num, self.a_num, *c_net_dim)
        self.t_a_net = ActorNet(self.r_s_num, self.a_num, *a_net_dim)
        self.t_c_net = CriticNet(self.s_num, self.a_num, *c_net_dim)
        self._hard_update(self.t_a_net, self.a_net)
        self._hard_update(self.t_c_net, self.c_net)
        self.a_net.to(self.device)
        self.c_net.to(self.device)
        self.t_a_net.to(self.device)
        self.t_c_net.to(self.device)
        self.criterion = nn.MSELoss()
        self.a_optimizer = torch.optim.Adam(self.a_net.parameters(), lr=self.a_lr)
        self.c_optimizer = torch.optim.Adam(self.c_net.parameters(), lr=self.c_lr)
        self.step_ita = 0

    def remember(self, s, rs, a, r, ns, rns, d):
        self.mem.append((s, rs, a, r, ns, rns, d))

    def act(self, state, explore=True, train=True):
        with torch.no_grad():
            state = np.array(state)
            if self.use_p:
                state = self._s2ps(state, 1)
            state = torch.Tensor(state.reshape((1, -1))).to(self.device)
            action = self.a_net(state).to('cpu')
            action = action.numpy().squeeze()
        if train:
            if self.step_ita > self.e_rand_decay_start and self.e > self.e_end:
                if self.step_ita % self.e_rand_decay_step == 0:
                    self.e = self.e * self.e_decay
            noise = np.random.randn(self.a_num) * self.e
            action = noise + (1 - self.e) * action
            action = np.clip(action, [0., 0.], [1., 1.])
        elif explore:
            noise = np.random.randn(self.a_num) * self.e_end
            action = noise + (1 - self.e_end) * action
            action = np.clip(action, [0., 0.], [1., 1.])
        return action.tolist()

    def replay(self):
        s_batch, rs_batch, a_batch, r_batch, ns_batch, rns_batch, d_batch = self._random_minibatch()
        with torch.no_grad():
            na_batch = self.t_a_net(rns_batch)
            next_q = self.t_c_net([ns_batch, na_batch])
            target_q = r_batch + self.r_gamma * next_q * (1. - d_batch)
        self.c_optimizer.zero_grad()
        current_q = self.c_net([s_batch, a_batch])
        c_loss = self.criterion(current_q, target_q)
        c_loss_item = c_loss.item()
        c_loss.backward()
        self.c_optimizer.step()
        self.a_optimizer.zero_grad()
        current_action = self.a_net(rs_batch)
        a_loss = -self.c_net([s_batch, current_action])
        a_loss = a_loss.mean()
        a_loss_item = a_loss.item()
        a_loss.backward()
        self.a_optimizer.step()
        self.step_ita += 1
        if self.step_ita % self.t_update_steps == 0:
            self._soft_update(self.t_a_net, self.a_net)
            self._soft_update(self.t_c_net, self.c_net)
        return a_loss_item, c_loss_item

    def reset_epsilon(self, new_e, new_decay):
        self.e = new_e
        self.e_decay = new_decay

    def save(self, save_dir, episode, run_name):
        try:
            os.mkdir(save_dir)
            print("Dir ", save_dir, " Created")
        except FileExistsError:
            print("Dir", save_dir, " exists")
        torch.save(self.a_net.state_dict(),
                   save_dir + '/' + run_name + '_actor_network_s' + str(episode) + '.pt')
        print("Episode " + str(episode) + " weights saved ...")

    def load(self, load_file_name):
        self.a_net.to('cpu')
        self.a_net.load_state_dict(torch.load(load_file_name))
        self.a_net.to(self.device)

    def _s2ps(self, state_value, batch_size):
        spike_state_value = state_value.reshape((batch_size, self.r_s_num, 1))
        state_spikes = np.random.rand(batch_size, self.r_s_num, self.p_window) < spike_state_value
        poisson_state = np.sum(state_spikes, axis=2).reshape((batch_size, -1))
        poisson_state = poisson_state / self.p_window
        poisson_state = poisson_state.astype(float)
        return poisson_state

    def _random_minibatch(self):
        minibatch = random.sample(self.mem, self.b_size)
        s_batch = np.zeros((self.b_size, self.s_num))
        rs_batch = np.zeros((self.b_size, self.r_s_num))
        a_batch = np.zeros((self.b_size, self.a_num))
        r_batch = np.zeros((self.b_size, 1))
        ns_batch = np.zeros((self.b_size, self.s_num))
        rns_batch = np.zeros((self.b_size, self.r_s_num))
        d_batch = np.zeros((self.b_size, 1))
        for num in range(self.b_size):
            s_batch[num, :] = np.array(minibatch[num][0])
            rs_batch[num, :] = np.array(minibatch[num][1])
            a_batch[num, :] = np.array(minibatch[num][2])
            r_batch[num, 0] = minibatch[num][3]
            ns_batch[num, :] = np.array(minibatch[num][4])
            rns_batch[num, :] = np.array(minibatch[num][5])
            d_batch[num, 0] = minibatch[num][6]
        if self.use_p:
            rs_batch = self._s2ps(rs_batch, self.b_size)
            rns_batch = self._s2ps(rns_batch, self.b_size)
        s_batch = torch.Tensor(s_batch).to(self.device)
        rs_batch = torch.Tensor(rs_batch).to(self.device)
        a_batch = torch.Tensor(a_batch).to(self.device)
        r_batch = torch.Tensor(r_batch).to(self.device)
        ns_batch = torch.Tensor(ns_batch).to(self.device)
        rns_batch = torch.Tensor(rns_batch).to(self.device)
        d_batch = torch.Tensor(d_batch).to(self.device)
        return s_batch, rs_batch, a_batch, r_batch, ns_batch, rns_batch, d_batch

    def _hard_update(self, target, source):
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)

    def _soft_update(self, target, source):
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.t_tau) + param.data * self.t_tau
                )
