import rospy
import time
import os
from torch.utils.tensorboard import SummaryWriter
import sys
import pickle

sys.path.append('../../')
from scripts.train_ddpg.ddpg_agent import Agent
from scripts.environment import GazeboEnvironment
from scripts.utility import *

def do_training(run_name="DDPG_R1", exp_name="Rand_R1", episode_num=(100, 200, 300, 400),
               iteration_num_start=(200, 300, 400, 500), iteration_num_step=(1, 2, 3, 4),
               iteration_num_max=(1000, 1000, 1000, 1000),
               max_speed=0.5, min_speed=0.05, save_steps=10000,
               env_epsilon=(0.9, 0.6, 0.6, 0.6), env_epsilon_decay=(0.999, 0.9999, 0.9999, 0.9999),
               laser_half_num=9, laser_min_dis=0.35, scan_overall_num=36, goal_dis_min_dis=0.3,
               obs_reward=-20, goal_reward=30, goal_dis_amp=15, goal_th=0.5, obs_th=0.35,
               state_num=22, action_num=2, is_pos_neg=False, is_poisson=False, poisson_win=50,
               memory_size=100000, batch_size=256, epsilon_end=0.1, rand_start=10000, rand_decay=0.999,
               rand_step=2, target_tau=0.01, target_step=1, use_cuda=True):

    try:
        os.mkdir('../save_ddpg_weights')
        print("Folder created")
    except:
        print("Folder exists")

    env1_stuff = gen_rand_list_env1(episode_num[0])
    env2_stuff = gen_rand_list_env2(episode_num[1])
    env3_stuff = gen_rand_list_env3(episode_num[2])
    env4_stuff = gen_rand_list_env4(episode_num[3])
    all_poly_list = [env1_stuff[0], env2_stuff[0], env3_stuff[0], env4_stuff[0]]

    all_stuff = pickle.load(open("../random_positions/" + exp_name + ".p", "rb"))
    all_init_list = all_stuff[0]
    all_goal_list = all_stuff[1]
    print("Using random stuff:", exp_name)

    rospy.init_node("train_ddpg")
    env = GazeboEnvironment(laser_scan_half_num=laser_half_num, laser_scan_min_dis=laser_min_dis,
                            scan_dir_num=scan_overall_num, goal_dis_min_dis=goal_dis_min_dis,
                            obs_reward=obs_reward, goal_reward=goal_reward, goal_dis_amp=goal_dis_amp,
                            goal_near_th=goal_th, obs_near_th=obs_th)
    if is_pos_neg:
        rescale_state_num = state_num + 2
    else:
        rescale_state_num = state_num
    agent = Agent(state_num, action_num, rescale_state_num, p_window=poisson_win, use_p=is_poisson,
                  mem_size=memory_size, b_size=batch_size, e_end=epsilon_end,
                  e_rand_decay_start=rand_start, e_decay=rand_decay, e_rand_decay_step=rand_step,
                  t_tau=target_tau, t_update_steps=target_step, use_cuda=use_cuda)

    tb_writer = SummaryWriter()

    total_steps = 0
    total_episode = 0
    env_episode = 0
    env_ita = 0
    ita_per_episode = iteration_num_start[env_ita]
    env.set_new_environment(all_init_list[env_ita],
                            all_goal_list[env_ita],
                            all_poly_list[env_ita])
    agent.reset_epsilon(env_epsilon[env_ita],
                        env_epsilon_decay[env_ita])

    start_time = time.time()
    while True:
        state = env.reset(env_episode)
        if is_pos_neg:
            rescale_state = ddpg_state_2_spike_value_state(state, rescale_state_num)
        else:
            rescale_state = ddpg_state_rescale(state, rescale_state_num)
        episode_reward = 0
        for ita in range(ita_per_episode):
            ita_time_start = time.time()
            total_steps += 1
            raw_action = agent.act(rescale_state)
            decode_action = wheeled_network_2_robot_action_decoder(
                raw_action, max_speed, min_speed
            )
            next_state, reward, done = env.step(decode_action)
            if is_pos_neg:
                rescale_next_state = ddpg_state_2_spike_value_state(next_state, rescale_state_num)
            else:
                rescale_next_state = ddpg_state_rescale(state, rescale_state_num)

            episode_reward += reward
            agent.remember(state, rescale_state, raw_action, reward, next_state, rescale_next_state, done)
            state = next_state
            rescale_state = rescale_next_state

            if len(agent.memory) > batch_size:
                actor_loss_value, critic_loss_value = agent.replay()
                tb_writer.add_scalar('DDPG/actor_loss', actor_loss_value, total_steps)
                tb_writer.add_scalar('DDPG/critic_loss', critic_loss_value, total_steps)
            ita_time_end = time.time()
            tb_writer.add_scalar('DDPG/ita_time', ita_time_end - ita_time_start, total_steps)
            tb_writer.add_scalar('DDPG/action_epsilon', agent.epsilon, total_steps)

            if total_steps % save_steps == 0:
                agent.save("../save_ddpg_weights", total_steps // save_steps, run_name)

            if done or ita == ita_per_episode - 1:
                print("Episode: {}/{}, Avg Reward: {}, Steps: {}"
                      .format(total_episode, episode_num, episode_reward / (ita + 1), ita + 1))
                tb_writer.add_scalar('DDPG/avg_reward', episode_reward / (ita + 1), total_steps)
                break
        if ita_per_episode < iteration_num_max[env_ita]:
            ita_per_episode += iteration_num_step[env_ita]
        if total_episode == 999:
            agent.save("../save_ddpg_weights", 0, run_name)
        total_episode += 1
        env_episode += 1
        if env_episode == episode_num[env_ita]:
            print("Environment ", env_ita, " Training Finished ...")
            if env_ita == 3:
                break
            env_ita += 1
            env.set_new_environment(all_init_list[env_ita],
                                    all_goal_list[env_ita],
                                    all_poly_list[env_ita])
            agent.reset_epsilon(env_epsilon[env_ita],
                                env_epsilon_decay[env_ita])
            ita_per_episode = iteration_num_start[env_ita]
            env_episode = 0
    end_time = time.time()
    print("Finish Training with time: ", (end_time - start_time) / 60, " Min")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--poisson', type=int, default=0)
    args = parser.parse_args()

    USE_CUDA = True
    if args.cuda == 0:
        USE_CUDA = False
    IS_POS_NEG, IS_POISSON = False, False
    if args.poisson == 1:
        IS_POS_NEG, IS_POISSON = True, True
    do_training(use_cuda=USE_CUDA, is_pos_neg=IS_POS_NEG, is_poisson=IS_POISSON)
