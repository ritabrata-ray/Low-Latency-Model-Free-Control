import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

from benchmark_1_class import *
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

#simulation parameters


max_episode_num=5000
num_trials=10
max_steps = 100
numsteps = np.zeros(max_episode_num, dtype='int')
avg_numsteps = np.zeros(max_episode_num)

traj_data_x=np.ones((max_episode_num,max_steps))
traj_data_y=np.ones((max_episode_num,max_steps))
traj_data_derivative_x=np.ones((max_episode_num,max_steps))
traj_data_derivative_y=np.ones((max_episode_num,max_steps))
traj_data_phi=np.zeros((max_episode_num,max_steps))
traj_data_dot_phi=np.ones((max_episode_num,max_steps))

unsafe_flag_data=np.zeros(max_episode_num)
#overall_unsafe=np.zeros(num_trials)

#learning_rate=np.zeros(num_trials)

learning_rate=3e-4
#training loop


system = obstacle_avoidance(h, k, r, qs, qa, d_x, d_y)
#learning_rate[trial] = (3 + trial * 0) * 0.0001
#s0 = system.vector_state()
s0=np.array([-1.0,+1.0])
S, A = system.state_action_space_size()
print("Initial Position:", system.vector_state())
policy_net = PolicyNetwork(system.vector_state().shape[0], A, 128, learning_rate)
for episode in range(max_episode_num):
        # if(trial%2==1):
        # if(episode==100):
        # break
    system.force_initialize_state(s0)
    state = system.vector_state()
    log_probs = []
    rewards = []
    unsafe_flag = 0
    action = np.random.choice(A)
    u_last = system.action_vector(action)

    for steps in range(max_steps):
        # env.render()
        state = system.vector_state()
        traj_data_x[episode][steps] = state[0]
        traj_data_y[episode][steps] = state[1]
        traj_data_phi[episode][steps] = system.phi_val()
        unsafe_flag = unsafe_flag or not (bool(system.is_safe()))
        action, log_prob = policy_net.get_action(system.vector_state())
            # new_state, reward, done, _ = env.step(action)
            # the action chosen is the action number in action-space. Next line does this.
        a = system.action_vector(action)
            # print("predicted action=",a)
            # print("Current phi=",system.phi_val())
            # xd=system.state_derivative()
            #a = correction_controller(system, u_last, a, 0.18)
        r, t = system.step(a)
            #a_n = system.get_action_num(a)
        u_last = system.quantize_action(a)
            # log_prob=policy_net.get_log_prob(a_n)  #this line needs a theoretical explanation.
        log_probs.append(log_prob)
        rewards.append(r)
        if (t):
            update_policy(policy_net, rewards, log_probs)
            numsteps[episode] = int(steps + 1)
            for j in range(47):
                avg_numsteps[episode] += numsteps[episode - j]
            avg_numsteps[episode] = avg_numsteps[episode] / 47
            if (episode<47):
                avg_numsteps[episode]=numsteps[episode]
            if episode % 1 == 0:
                sys.stdout.write(
                        "episode: {}, total reward: {}, length: {}\n".format(
                                                                            episode,
                                                                            np.round(np.sum(rewards),decimals=3),
                                                                            steps + 1
                                                                            )
                )
            print("Final Position:", system.vector_state())
            break
            # state = new_state
    if (t == 0):
        update_policy(policy_net, rewards, log_probs)
        numsteps[episode] = int(steps + 1)
        for j in range(47):
            avg_numsteps[episode] += numsteps[episode - j]
        avg_numsteps[episode] = avg_numsteps[episode] / 47
        if (episode < 47):
            avg_numsteps[episode] = numsteps[episode]
        if episode % 1 == 0:
            sys.stdout.write(
                    "episode: {}, total reward: {}, length: {}\n".format(
                                                                        episode,
                                                                        np.round(np.sum(rewards),decimals=3),
                                                                        steps + 1
                                                                        )
                            )
            print("Final Position:", system.vector_state())
    if (unsafe_flag):
        print("System was unsafe in this episode!")
            # unsafe_episode=episode
    #overall_unsafe[trial] = overall_unsafe[trial] or unsafe_flag
    unsafe_flag_data[episode] = unsafe_flag

#plt.plot(numsteps, label='#steps till destination/horizon')
plt.plot(avg_numsteps, label='average #steps for last 47 episodes')
plt.xlabel('Episode')
plt.ylabel('Steps till Destination/Horizon')
plt.title('Policy Gradient')
plt.legend(loc='upper right', borderpad=0.4, labelspacing=0.7)
my_path = os.path.join(base_path,"Data/Benchmarks/Benchmark_1_Safety_based_objective")
#plt.savefig(os.path.join(my_path, "Keynote_safe_learning.png"), format="png", bbox_inches="tight")
plt.show()


np.save(os.path.join(my_path,'unsafe_simulation_data_1'),traj_data_phi)
np.save(os.path.join(my_path,'unsafe_numsteps_1'),numsteps)
np.save(os.path.join(my_path,'unsafe_average_numsteps_1'),avg_numsteps)
np.save(os.path.join(my_path,'unsafe_simulation_traj_x_1'),traj_data_x)
np.save(os.path.join(my_path,'unsafe_simulation_traj_y_1'),traj_data_y)
np.save(os.path.join(my_path,'num_trials'),num_trials)
np.save(os.path.join(my_path,'max_episode_num'),max_episode_num)
np.save(os.path.join(my_path,'max_steps'),max_steps)
