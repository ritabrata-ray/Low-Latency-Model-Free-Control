import math
import numpy as np


class obstacle_avoidance:
    def __init__(self, h, k, r, qs, qa, d_x, d_y):

        # Discrete time step for running dynamics
        self.tau = 0.02

        # Some unknown dynamics parameters hinted by oracle
        self.lambda_1 = 2
        self.lambda_2 = -1

        # Obstacle is circular
        # Obstacle parameters are the obstacle centre coordinates and its radius
        self.obstacle_center_x_coordinate = h
        self.obstacle_center_y_coordinate = k
        self.obstacle_radius = r

        # The agent needs to avoid the obstacle by a distance margin 7% of its radius, maybe we can fix it to a constant
        self.radius_threshold_safety_gap = 0.07 * self.obstacle_radius

        # State Space is a discretized box in 2-D
        self.state_space_discretization = qs
        self.state_space_size = self.state_space_discretization ** 2
        self.state_space = np.zeros((self.state_space_size, 2))
        self.x_lim_left = -1.0
        self.x_lim_right = +1.0
        self.y_lim_left = -1.0
        self.y_lim_right = +1.0
        self.discrete_length_x = (self.x_lim_right - self.x_lim_left) / (self.state_space_discretization)
        self.discrete_length_y = (self.y_lim_right - self.y_lim_left) / (self.state_space_discretization)
        for i in range(self.state_space_discretization):
            for j in range(self.state_space_discretization):
                self.state_space[
                    self.state_space_discretization * i + j, 0] = self.x_lim_left + i * self.discrete_length_x
                self.state_space[
                    self.state_space_discretization * i + j, 1] = self.y_lim_left + j * self.discrete_length_y

        # Action-Space is also a discretized 2-D box
        self.action_space_discretization = qa
        self.action_space_size = self.action_space_discretization ** 2
        self.action_space = np.zeros((self.action_space_size, 2))
        self.u1_lim_left = -7.0
        self.u1_lim_right = +7.0
        self.u2_lim_left = -7.0
        self.u2_lim_right = +7.0
        self.discrete_length_u1 = (self.u1_lim_right - self.u1_lim_left) / (self.action_space_discretization)
        self.discrete_length_u2 = (self.u2_lim_right - self.u2_lim_left) / (self.action_space_discretization)
        for i in range(self.action_space_discretization):
            for j in range(self.action_space_discretization):
                self.action_space[
                    self.action_space_discretization * i + j, 0] = self.u1_lim_left + i * self.discrete_length_u1
                self.action_space[
                    self.action_space_discretization * i + j, 1] = self.u2_lim_left + j * self.discrete_length_u2

        # Random initialization of the state variable
        self.state = np.random.randint(self.state_space_size)
        self.state_prev= self.state
        self.resetting_state = self.state  # Remember this when required to reset

        # The goal of the controller is to reach the destination point
        self.destination_x = d_x
        self.destination_y = d_y

    # Function to check if the system state is in the safe region safe
    def is_safe(self):
        x = self.state_space[self.state][0]
        y = self.state_space[self.state][1]
        val = (x - self.obstacle_center_x_coordinate) ** 2 + (y - self.obstacle_center_y_coordinate) ** 2 - (
                    self.obstacle_radius + self.radius_threshold_safety_gap) ** 2
        if (val > 0):
            return 1
        else:
            return 0

    def phi_val(self):
        x = self.state_space[self.state][0]
        y = self.state_space[self.state][1]
        val = (x - self.obstacle_center_x_coordinate) ** 2 + (y - self.obstacle_center_y_coordinate) ** 2 - (
                self.obstacle_radius + self.radius_threshold_safety_gap) ** 2
        return val

    def grad_phi(self):
        x = self.state_space[self.state][0]
        y = self.state_space[self.state][1]
        h=self.obstacle_center_x_coordinate
        k=self.obstacle_center_y_coordinate
        g=np.zeros(2)
        g[0]=2*(x-h)
        g[1]=2*(y-k)
        return g

    # Function to reset the state variable to the old starting point
    def reset(self):
        self.state = self.resetting_state

    # First we need to quantize the action command to the action space as it may be coming from the correction controller
    def quantize_action(self, action):
        u1 = action[0]
        u2 = action[1]
        u1_normalized = ((u1 - self.u1_lim_left) / (self.discrete_length_u1))
        u1_d = np.floor(u1_normalized + 0.5)  # Nearest integer function
        u2_normalized = ((u2 - self.u2_lim_left) / (self.discrete_length_u2))
        u2_d = np.floor(u2_normalized + 0.5)  # Nearest integer function
        # Correcting border cases as upper and lower limits
        if (u1_d < 0):
            u1_d = 0
        if (u1_d >= self.action_space_discretization):
            u1_d = self.action_space_discretization - 1
        if (u2_d < 0):
            u2_d = 0
        if (u2_d >= self.action_space_discretization):
            u2_d = self.action_space_discretization - 1

        u1_d = int(u1_d)
        u2_d = int(u2_d)

        discretized_action = np.zeros(2)
        discretized_action[0] = self.action_space[self.action_space_discretization * u1_d + u2_d][0]
        discretized_action[1] = self.action_space[self.action_space_discretization * u1_d + u2_d][1]
        return discretized_action

    def get_action_num(self, action):
        u1 = action[0]
        u2 = action[1]
        u1_normalized = ((u1 - self.u1_lim_left) / (self.discrete_length_u1))
        u1_d = np.floor(u1_normalized + 0.5)  # Nearest integer function
        u2_normalized = ((u2 - self.u2_lim_left) / (self.discrete_length_u2))
        u2_d = np.floor(u2_normalized + 0.5)  # Nearest integer function
        # Correcting border cases as upper and lower limits
        if (u1_d < 0):
            u1_d = 0
        if (u1_d >= self.action_space_discretization):
            u1_d = self.action_space_discretization - 1
        if (u2_d < 0):
            u2_d = 0
        if (u2_d >= self.action_space_discretization):
            u2_d = self.action_space_discretization - 1

        u1_d = int(u1_d)
        u2_d = int(u2_d)

        action_num=self.action_space_discretization * u1_d + u2_d
        return action_num

    # Similarly the state varaible may go unquantized after the step, so we quantize it too!
    def quantize_state(self, state):
        x = state[0]
        y = state[1]
        x_normalized = ((x - self.x_lim_left) / (self.discrete_length_x))
        x_d = np.floor(x_normalized + 0.5)  # Nearest integer function
        y_normalized = ((y - self.y_lim_left) / (self.discrete_length_y))
        y_d = np.floor(y_normalized + 0.5)  # Nearest integer function
        # Correcting border cases as upper and lower limits
        if (x_d < 0):
            x_d = 0
            # print("Undershoot")
        if (x_d >= self.state_space_discretization):
            x_d = self.state_space_discretization - 1
            # print("Overshoot")
        if (y_d < 0):
            y_d = 0
            # print("Undershoot")
        if (y_d >= self.state_space_discretization):
            y_d = self.state_space_discretization - 1
            # print("Overshoot")

        x_d = int(x_d)
        y_d = int(y_d)

        # change the state variable right here, the remaining part of the function is useless
        self.state = x_d * self.state_space_discretization + y_d
        # required only if we want to see the state variable separately
        discretized_state = np.zeros(2)
        discretized_state[0] = self.state_space[self.state_space_discretization * x_d + y_d][0]
        discretized_state[1] = self.state_space[self.state_space_discretization * x_d + y_d][1]
        return x_d, y_d, discretized_state

    # Function to run the dynamics
    def step(self, action):
        # We have the following dynamics equation:
        # \dot x1=f1(x1,x2)+lambda_1*u1
        # \dot x2=f2(x1,x2)+lambda_2*u2
        # Here we take f1(x1,x2)=0.1*(x1-x2)^2, f2(x1,x2)=0.007*|x2|, lambda_1=2, lambda_2=-1
        self.state_prev=self.state
        a = self.quantize_action(action)
        x1 = self.state_space[self.state][0]
        x2 = self.state_space[self.state][1]

        dist_to_destination_square = (x1 - self.destination_x) ** 2 + (x2 - self.destination_y) ** 2

        f1 = 0.1 * (x1 - x2) ** 2
        f2 = 0.007 * np.abs(x2)
        dot_x1 = f1 + self.lambda_1 * a[0]
        dot_x2 = f2 + self.lambda_2 * a[1]
        new_state = np.zeros(2)
        x1_n = x1 + dot_x1 * self.tau
        x2_n = x2 + dot_x2 * self.tau
        new_state[0] = x1_n
        new_state[1] = x2_n
        i, j, ns = self.quantize_state(new_state)
        self.state = self.state_space_discretization * i + j

        terminate = 0

        new_dist_to_destination_square = (ns[0] - self.destination_x) ** 2 + (ns[1] - self.destination_y) ** 2

        if (new_dist_to_destination_square < 2.56 * (self.discrete_length_x ** 2 + self.discrete_length_y ** 2)):
            terminate = 1

        reward = 0

        if (dist_to_destination_square > 1.1 * new_dist_to_destination_square):
            reward = 1

        if (dist_to_destination_square < 0.9 * new_dist_to_destination_square):
            reward = -1

        reward=1/(new_dist_to_destination_square+0.01) #trying out new reward function

        if (terminate==1):     #Overwrite to strongly reward termination
            reward=21000

        if (not self.is_safe()):
            #cost = (1000 * self.phi_val()) ** 2  gets uniform safety rate of 0.25, no safer learning; CPO cost
            cost=1
            reward=reward-cost

        return reward, terminate

    def action_space_tour(self, trial_duration=47):
        self.reset()
        total_reward = 0
        trajectory_length = 0
        for i in range(trial_duration):
            a = self.simple_gradient_based_policy()
            r, t = self.step(a)
            # r,t=self.step([sys.action_space[a,0],sys.action_space[a,1]])
            if (self.is_safe()):
                print("Safe at:")
                print(self.state_space[self.state])
            else:
                print("Unsafe at:")
                print(self.state_space[self.state])
            trajectory_length += 1
            total_reward += r
            if (t):
                print("Destination Reached at:")
                print(self.state_space[self.state])
                return total_reward, trajectory_length
        return total_reward, trajectory_length

    def simple_gradient_based_policy(self):
        eta = 7
        action = np.zeros(2)
        action[0] = -0.7 * eta * (self.state_space[self.state][0] - self.destination_x)
        action[1] = 1.4 * eta * (self.state_space[self.state][1] - self.destination_y)
        # the signs are different because it knows the signs of the lambdas
        # to work well needs the ratio of magnitudes as well
        return action

    def vector_state(self):
        return self.state_space[self.state]

    def state_action_space_size(self):
        return self.state_space_size, self.action_space_size

    def action_vector(self,action):
        return self.action_space[action]

    def state_derivative(self):
        x=self.state_space[self.state]
        x0=self.state_space[self.state_prev]
        xc=x-x0
        xd=(1/self.tau)*xc
        return xd

    def force_initialize_state(self,state):
        i, j, s = self.quantize_state(state)
        self.state = self.state_space_discretization * i + j
        #print("Forced Initial Position:", self.state_space[self.state])


#Below lines are just to check if everything is alright
#sys=obstacle_avoidance(0.4,-0.3,0.08,100,10,0.5,-0.2)
#R,L=sys.action_space_tour(77)
#sys.vector_state()
#S,A=sys.state_action_space_size()
#print(R,L)
#print("Performance=",R/L)