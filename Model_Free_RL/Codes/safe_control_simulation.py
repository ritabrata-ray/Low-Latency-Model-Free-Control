import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

from obstacle_2D_sys import *
from policy_gradient import *

from controllers import *

import os

base_path="/Users/ritabrataray/Desktop/Safe control/Model_Free_RL"

GAMMA = 0.7

### Model Parameters

r=0.11  #Obstacle radius
h=0.07  #Obstacle center x-coordinate (abscissa)
k=-0.01 #Obstacle center y-coordinate (ordinate)
d_x=0.5 #Destination abscissa
d_y=-0.2 #Destination ordinate
us_x=0.47 #unsafe start point abscissa
us_y=-0.3 #unsafe start point ordinate
lamda_1=2 #Actual parameters
lamda_2=-1 #Actual parameters
hat_lamda_1=0.2 #Oracle estimate of parameters
hat_lamda_2=-0.1 #Oracle estimate of parameters

qs=100 #state space is discretized 100x100
qa=10  #action space is discretized 10x10

#We repeat for 15 trials

s_num_trials=15

#simulation parameters
s_max_episode_num=5000

s_max_steps = 100
s_numsteps = np.zeros(s_max_episode_num, dtype='int')
s_avg_numsteps = np.zeros(s_max_episode_num)

s_traj_data_x=np.ones((s_max_episode_num,s_max_steps))
s_traj_data_y=np.ones((s_max_episode_num,s_max_steps))
s_traj_data_derivative_x=np.ones((s_max_episode_num,s_max_steps))
s_traj_data_derivative_y=np.ones((s_max_episode_num,s_max_steps))
s_traj_data_phi=np.zeros((s_max_episode_num,s_max_steps))
s_traj_data_dot_phi=np.ones((s_max_episode_num,s_max_steps))

s_unsafe_flag_data=np.zeros(s_max_episode_num)
#s_overall_unsafe=np.zeros(s_num_trials)

learning_rate=3e-4

system = obstacle_avoidance(h, k, r, qs, qa, d_x, d_y)

s0=np.array([-1.0,+1.0])

S, A = system.state_action_space_size()
print("Initial Position:", system.vector_state())
policy_net = PolicyNetwork(system.vector_state().shape[0], A, 128, learning_rate)

for episode in range(s_max_episode_num):
    # if(trial%2==1):
    # if(episode==100):
    # break
    system.force_initialize_state(s0)
    #system.reset()
    state = system.vector_state()
    log_probs = []
    rewards = []
    s_unsafe_flag = 0
    action = np.random.choice(A)
    u_last = system.action_vector(action)

    for steps in range(s_max_steps):
        # env.render()
        state = system.vector_state()
        s_traj_data_x[episode][steps] = state[0]
        s_traj_data_y[episode][steps] = state[1]
        s_traj_data_phi[episode][steps] = system.phi_val()
        s_unsafe_flag = s_unsafe_flag or not (bool(system.is_safe()))
        action, log_prob = policy_net.get_action(system.vector_state())
        # new_state, reward, done, _ = env.step(action)
        # the action chosen is the action number in action-space. Next line does this.
        a = system.action_vector(action)
        # print("predicted action=",a)
        # print("Current phi=",system.phi_val())
        # xd=system.state_derivative()
        a = correction_controller(system, u_last, a, 0.18)
        r, t = system.step(a)
        # a_n = system.get_action_num(a)
        u_last = system.quantize_action(a)
        # log_prob=policy_net.get_log_prob(a_n)  #this line needs a theoretical explanation.
        log_probs.append(log_prob)
        rewards.append(r)
        if (t):
            update_policy(policy_net, rewards, log_probs)
            s_numsteps[episode] = int(steps + 1)
            for j in range(47):
                s_avg_numsteps[episode] += s_numsteps[episode - j]
            s_avg_numsteps[episode] = s_avg_numsteps[episode] / 47
            if (episode < 47):
                s_avg_numsteps[episode] = s_numsteps[episode]
            if episode % 1 == 0:
                sys.stdout.write(
                    "episode: {}, total reward: {}, length: {}\n".format(episode,
                                                                    np.round(np.sum(rewards),decimals=3),
                                                                     steps + 1))
            print("Final Position:", system.vector_state())
            break
        # state = new_state
    if (t == 0):
        update_policy(policy_net, rewards, log_probs)
        s_numsteps[episode] = int(steps + 1)
        for j in range(47):
            s_avg_numsteps[episode] += s_numsteps[episode - j]
        s_avg_numsteps[episode] = s_avg_numsteps[episode] / 47
        if (episode < 47):
            s_avg_numsteps[episode] = s_numsteps[episode]
        if episode % 1 == 0:
            sys.stdout.write(
                "episode: {}, total reward: {}, length: {}\n".format(episode,
                                                                    np.round(np.sum(rewards),decimals=3),
                                                                                steps + 1))
            print("Final Position:", system.vector_state())
    if (s_unsafe_flag):
        print("System was unsafe in this episode!")
        # unsafe_episode=episode
    #s_overall_unsafe[trial] = s_overall_unsafe[trial] or s_unsafe_flag
    s_unsafe_flag_data[episode] = s_unsafe_flag

plt.plot(s_numsteps, label='#steps till destination/horizon')
plt.plot(s_avg_numsteps, label='average #steps for last 10 episodes')
plt.xlabel('Episode')
plt.ylabel('Steps till Destination/Horizon')
plt.title('Safe learning')
plt.legend(loc='upper right', borderpad=0.4, labelspacing=0.7)
# my_path = os.path.abspath("/Users/ritabrataray/Desktop/Safe control/Model_Free_RL")
# plt.savefig(os.path.join(my_path, "Keynote_safe_learning.png"), format="png", bbox_inches="tight")
plt.show()

data_path = os.path.abspath(base_path,"Data/Safe")

np.save(os.path.join(data_path,'safe_simulation_data_15'),s_traj_data_phi)
np.save(os.path.join(data_path,'safe_numsteps_15'),s_numsteps)
np.save(os.path.join(data_path,'safe_average_numsteps_15'),s_avg_numsteps)
np.save(os.path.join(data_path,'safe_simulation_traj_x_15'),s_traj_data_x)
np.save(os.path.join(data_path,'safe_simulation_traj_y_15'),s_traj_data_y)
np.save(os.path.join(data_path,'safe_num_trials'),s_num_trials)
np.save(os.path.join(data_path,'safe_max_episode_num'),s_max_episode_num)
np.save(os.path.join(data_path,'safe_max_steps'),s_max_steps)
