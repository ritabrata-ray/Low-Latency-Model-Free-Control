import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.distributions import Categorical
from collections import namedtuple

from benchmark_2_class import *

import os
import sys


# Algorithm hyperparameters
GAMMA = 0.7

# delta, maximum KL divergence
max_d_kl = 0.01


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


#Simulation Parameters
max_episode_num=500
max_steps = 100
num_rollouts=10

numsteps = np.zeros((max_episode_num,num_rollouts), dtype='int')
a_numsteps=np.zeros((max_episode_num)) #to store average num_steps in each epoch averaged over the rollouts.
avg_numsteps = np.zeros(max_episode_num) #for the running average (low-pass filtering)



traj_data_x=np.ones((max_episode_num,num_rollouts,max_steps))
traj_data_y=np.ones((max_episode_num,num_rollouts,max_steps))
traj_data_derivative_x=np.ones((max_episode_num,num_rollouts,max_steps))
traj_data_derivative_y=np.ones((max_episode_num,num_rollouts,max_steps))
traj_data_phi=np.zeros((max_episode_num,num_rollouts,max_steps))
traj_data_dot_phi=np.ones((max_episode_num,num_rollouts,max_steps))

unsafe_flag_data=np.zeros((max_episode_num,num_rollouts,max_steps))
overall_unsafe=0

#simulation environment initilization

system = obstacle_avoidance(h, k, r, qs, qa, d_x, d_y)
s0=np.array([-1.0,+1.0])
S, A = system.state_action_space_size()

state_size=s0.shape[0]
num_actions=A

Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'next_states', ])

def train(epochs=100, num_rollouts=10, max_steps=100):
    mean_total_rewards = []
    global_rollout = 0

    for epoch in range(epochs):
        rollouts = []
        rollout_total_rewards = []

        for t in range(num_rollouts):
            system.force_initialize_state(s0)
            done = False

            samples = []
            steps=0 #loop counter init
            while ((steps<max_steps) and (not done)):

                state = system.vector_state()
                traj_data_x[epoch][t][steps] = state[0]
                traj_data_y[epoch][t][steps] = state[1]
                traj_data_phi[epoch][t][steps] = system.phi_val()
                unsafe_flag_data[epoch][t][steps] = not (bool(system.is_safe()))



                with torch.no_grad():
                    action = get_action(state)

                a=system.action_vector(action)
                next_state, reward, _, done = system.step(a)  #_ ignores the cost

                # Collect samples
                samples.append((state, action, reward, next_state))

                #state = next_state, already done in the beginning of loop
                steps=steps+1 #loop increment

            # Transpose our samples
            states, actions, rewards, next_states = zip(*samples) #each individual rollout data
            numsteps[epoch][t]=int(steps)

            states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
            next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
            actions = torch.as_tensor(actions).unsqueeze(1)
            rewards = torch.as_tensor(rewards).unsqueeze(1)

            rollouts.append(Rollout(states, actions, rewards, next_states)) #this collects the data over all rollouts

            rollout_total_rewards.append(rewards.sum().item()) #collects total rewards for each rollout, not the discounted sum
            global_rollout += 1
            a_numsteps[epoch]=a_numsteps[epoch]+numsteps[epoch][t]

        a_numsteps[epoch]=(a_numsteps[epoch]/num_rollouts) #gets the average number of steps in that epoch
        #Now we do moving average filtering over the last few episodes for noise reduction in the convergence plot.
        for j in range(47):
            avg_numsteps[epoch] += a_numsteps[epoch - j]
        avg_numsteps[epoch] = avg_numsteps[epoch] / 47
        if (epoch < 47):
            avg_numsteps[epoch] = a_numsteps[epoch]
        mtr = np.mean(rollout_total_rewards)
        if epoch % 1 == 0:
            sys.stdout.write(
                "epoch: {}, total reward averaged over {} rollouts: {}, length: {}\n".format(
                                                                                              epoch,
                                                                                              num_rollouts,
                                                                                              np.round(mtr,decimals=3),
                                                                                              a_numsteps[epoch]
                                                                                            )
            )
            print("Final Position:", system.vector_state())

        update_agent(rollouts)
        mean_total_rewards.append(mtr)

    training_epochs=np.arange(epochs)
    plt.plot(training_epochs,a_numsteps,label='average over rollouts')
    #plt.plot(training_epochs, avg_numsteps, label='moving average')
    plt.xlabel('Training Epoch')
    plt.ylabel('Number of steps till destination/Horizon')
    plt.title('TRPO Convergence')
    plt.grid()
    plt.show()

    trajectory_sequence=np.arange(epochs*num_rollouts)
    traj_lengths=np.zeros(epochs*num_rollouts)
    smoothened_traj_lengths = np.zeros(epochs * num_rollouts)
    for e in range(epochs):
        for r in range(num_rollouts):
            traj_lengths[e*num_rollouts+r]=numsteps[e][r]

    for i in range(epochs*num_rollouts):
        if (i < 47):
            smoothened_traj_lengths[i] = traj_lengths[i]
            continue
        for j in range(47):
            smoothened_traj_lengths[i] += traj_lengths[i - j]
        smoothened_traj_lengths[i] = smoothened_traj_lengths[i] / 47

    plt.plot(trajectory_sequence,smoothened_traj_lengths)
    plt.xlabel('Training epochs')
    plt.ylabel('Number of steps till destination/Horizon')
    plt.title('TRPO Convergence')
    plt.show()

    plt.plot(mean_total_rewards)
    plt.show()
#Now we define the actor and critic neural networks
actor_hidden = 32
actor = nn.Sequential(nn.Linear(state_size, actor_hidden),
                      nn.ReLU(),
                      nn.Linear(actor_hidden, num_actions),
                      nn.Softmax(dim=1))

#the forward pass for the actor NN, this is used in the train function
def get_action(state):
    state = torch.tensor(state).float().unsqueeze(0)  # Turn state into a batch with a single element
    dist = Categorical(actor(state))  # Create a distribution from probabilities for actions
    return dist.sample().item()


# Critic takes a state and returns the value function estimate for that state.
critic_hidden = 32
critic = nn.Sequential(nn.Linear(state_size, critic_hidden),
                       nn.ReLU(),
                       nn.Linear(critic_hidden, 1))
critic_optimizer = Adam(critic.parameters(), lr=0.005)


def update_critic(advantages):
    loss = .5 * (advantages ** 2).mean()  # MSE
    critic_optimizer.zero_grad()
    loss.backward()
    critic_optimizer.step()

def apply_update(grad_flattened):   #we don't do adam for the actor updates, rather it uses the natural gradient falttened data
    n = 0
    for p in actor.parameters():
        numel = p.numel()
        g = grad_flattened[n:n + numel].view(p.shape)
        p.data += g
        n += numel

def conjugate_gradient(A, b, delta=0., max_iterations=10):  #Estimates A^{-1}b; see Wikipedia for the algo and proof
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()

    i = 0
    while i < max_iterations:
        AVP = A(p)

        dot_old = r @ r
        alpha = dot_old / (p @ AVP)

        x_new = x + alpha * p

        if (x - x_new).norm() <= delta:
            return x_new

        i += 1
        r = r - alpha * AVP

        beta = (r @ r) / dot_old
        p = r + beta * p

        x = x_new
    return x
#some more auxiliary functions for the final update agent TRPO function (NPG based update)
def estimate_advantages(states, last_state, rewards): #takes a rollout trajectory and separately the last state
    values = critic(states)
    last_value = critic(last_state.unsqueeze(0))
    next_values = torch.zeros_like(rewards)
    for i in reversed(range(rewards.shape[0])): #computing all the Q values in the rollout
        last_value = next_values[i] = rewards[i] + GAMMA * last_value
    advantages = next_values - values  #there is no advantage over the separate last_state
    return advantages


def surrogate_loss(new_probabilities, old_probabilities, advantages):
    return (new_probabilities / old_probabilities * advantages).mean()


def kl_div(p, q):
    p = p.detach()
    return (p * (p.log() - q.log())).sum(-1).mean()  #The mean because it is E[D_KL(pi_theta[.|s]||pi [.|s] |s)] where the
                                                    # expectation is over s: s is coming from the state vsitation measure of
                                                    # the old policy from which the trajectory data is sampled anyway. See TRPO theory.


def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g

#We are now ready for the final actor update function. We compute the natural policy gradient using all the above functions.
#takes rollouts data for that epoch and does the advantage estimation over all the rollouts data in that epoch.
#Updates critic with all this data for estimating the value function corresponding to the old distribution policy of this epoch.
def update_agent(rollouts):
    states = torch.cat([r.states for r in rollouts], dim=0)
    actions = torch.cat([r.actions for r in rollouts], dim=0).flatten()

    advantages = [estimate_advantages(states, next_states[-1], rewards) for states, _, rewards, next_states in rollouts]
    #Above stores the advantages sequence for each rollout trajectory separately; next_states[-1] is the separately given  last state
    advantages = torch.cat(advantages, dim=0).flatten()
    #this last line prepares the advantages as the loss function data using all the rollouts data available from all past episodes

    # Normalize advantages to reduce skewness and improve convergence
    advantages = (advantages - advantages.mean()) / advantages.std()

    update_critic(advantages)

    distribution = actor(states) #probability distribution functions for each state over all the steps/ states visited
                                                                                # over all the rollouts for this episode.
    distribution = torch.distributions.utils.clamp_probs(distribution) #becomes a torch variable
    probabilities = distribution[range(distribution.shape[0]), actions]
    #Above gets a flattenedlist probabilities of all the trajectory steps of all the rollouts in this epoch from the current epoch's distribution.
    # Now we have all the data we need for the algorithm

    # We will calculate the gradient wrt to the new probabilities (surrogate function),
    # so second probabilities should be treated as a constant
    L = surrogate_loss(probabilities, probabilities.detach(), advantages) #this gets the expectation over all the rollouts data of this epoch.
    #when the optimizer works probabilities change but the .detach() keeps the old probabilities intact.
    KL = kl_div(distribution, distribution) #distribution is arrayed over each state in the rollouts data. mean gets the KL div expectation according to the state visitation measure for this epoch.
    #the p.detach() inside the kl_div ensures that the old distribution remains intact while the optimizer works.

    parameters = list(actor.parameters()) #flattened list of all actor NN parameters

    g = flat_grad(L, parameters, retain_graph=True) #gets the policy gradient (flattened list) of the surrogate loss function.
    d_kl = flat_grad(KL, parameters, create_graph=True)  # Create graph, because we will call backward() on it (for HVP)

    def HVP(v): #autograd works only for scalar so use v vector variable multiplication to get the Hessian and v product: HVP.
        return flat_grad(d_kl @ v, parameters, retain_graph=True)

    #So, here we got H, as the Hessian from the HVP function and g the policy gradient of the surrogate loss function as g.
    #so, we are ready to do the following optimization:
        """ 
            max. 1/(1-gamma)*E_{d_pi_old} pi_theta(a_t|s_t)/pi_old(a_t,s_t) A^{old}(s_t,a_t)
            subject to E_{d_pi_old} KL(pi_theta||pi_old) <= delta
        """
        """
            The above constrained optimization problem is linearized and quadtratized as follows:
                  max. g^T x
                  subject to 1/2 x^T H x -delta <= 0,
                  where g and H are defined as above. This QP has closed form solution:
                  x=sqrt(2*delta/(g^T H^{-1} g))*H^{-1} g
                  so update is theta=theta_old+x*(optimal step size)
        """

    search_dir = conjugate_gradient(HVP, g)
    max_length = torch.sqrt(2 * max_d_kl / (search_dir @ HVP(search_dir)))
    max_step = max_length * search_dir   #remember max steps was max horizon length for each rollout, but this is max_step.
    # The above NPG solutions are only used for making an educated guess of the optimal step size as shown below.

    def criterion(step): # A criterion to check if practically using a step size is giving improvements
        apply_update(step)

        with torch.no_grad():
            distribution_new = actor(states)
            distribution_new = torch.distributions.utils.clamp_probs(distribution_new)
            probabilities_new = distribution_new[range(distribution_new.shape[0]), actions]

            L_new = surrogate_loss(probabilities_new, probabilities, advantages)
            KL_new = kl_div(distribution, distribution_new)

        L_improvement = L_new - L

        if L_improvement > 0 and KL_new <= max_d_kl:
            return True

        apply_update(-step)
        return False
    # A smart linear search over i on the 0.9^i space to find the optimal step size practically using the above criterion function.
    i = 0
    while not criterion((0.9 ** i) * max_step) and i < 10:
        i += 1


# The CPO will need a separate optimizer using the cost constraint as well as separate criterion function which will also check if the
# constraint is met.

train(max_episode_num, num_rollouts, max_steps)





