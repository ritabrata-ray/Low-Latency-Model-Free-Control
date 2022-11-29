import numpy as np
import os
from obstacle_2D_sys import *
from controllers import *
import matplotlib.pyplot as plt

base_path="/Users/ritabrataray/Desktop/Safe control/Model_Free_RL"

"""
   We have the kinematic 2-D non-linear dynamical system:
     \dot x_1 = f_1(x_1,x_2) + g_11(x_1, x_2)*u_1 g_12(x_1,x_2)*u_2
     \dot x_2 = f_2(x_1,x_2) + g_21(x_1, x_2)*u_1 g_22(x_1,x_2)*u_2
     
     where  for all x_1, x_2:
           g_11 = lambda_1 = 2       g_12 = 0
           g_21 = 0                  g_22 = lambda_2 = -1
           and 
           f_1(x_1,x_2) = 0.1 * (x_1 - x_2) ** 2
           f_2(x_1,x_2) = 0.007 * |x_2|
           
           There is a circular obstacle centered at h = 0.07, k = -0.01, with a radius of r = 0.11.
           Within a close vicinity (7% of the obstacle radius length exterior to the obstacle boundary) of the obstacle
           is considered unsafe and the barrier function \phi gets a non-positive value in the unsafe region. Highly unsafe 
           locations (deeper into the obstacle region) is indicated by large non-negative values of \phi.
           
           In this simulation, we show that for different unsafe initializations of the agent, it is able to quickly recover into the safety
           region, where the quickness of recovery depends on the hyperparameter \eta and the initial point where it started (more recovery
           time when the start location is well inside the obstacle region).
           
           We choose the following unsafe initial locations:
           
           1. x_1, y_1 = -0.04, -0.01
           2. x_2, y_2 =  0.07, +0.10
           3. x_3, y_3 =  0.15, -0.01
           4. x_4, y_4 =  0.07, -0.04
           5. x_5, y_5 = 0.07,  +0.00
           
           We choose a particularly bad nonminal controller, which points it towards the center of the obstacle.
"""


initial_points=np.zeros((5,2))

initial_points[0] = [-0.04, -0.01]
initial_points[1] = [+0.07, +0.10]
initial_points[2] = [+0.15, -0.01]
initial_points[3] = [+0.07, -0.04]
initial_points[4] = [+0.07, +0.00]

T=40
tau=0.002

def nominal_controller(sys):
    u = np.zeros(2)
    state = sys.vector_state()
    point_center = np.zeros(2)
    point_center[0] = h-state[0]
    point_center[1] = k-state[1]
    p_norm = np.sqrt(point_center[0]**2 + point_center[1]**2)
    point_center = point_center/p_norm
    u = point_center*0.05
    return u

time_axis=np.arange(T)*0.02  #0.02 is the time discretization for the agent dynamics

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
qa=100  #action space is discretized 10x10


system = obstacle_avoidance(h, k, r, qs, qa, d_x, d_y)
system.set_time_discretization(tau)

traj_data_phi=np.zeros((5,T))

for i in range(5):
    system.force_initialize_state(initial_points[i])
    u_last=nominal_controller(system)
    for t in range(np.shape(time_axis)[0]):
        u=nominal_controller(system)
        u=recovery_controller( system, u_last, u, theta =0.0, eta=0.01 )
        _,_=system.step(u)
        u_last=u
        traj_data_phi[i][t]=system.phi_val()

plt.plot(time_axis,traj_data_phi[0],label='initial location=(-0.04, -0.01)',color='blue')
plt.plot(time_axis,traj_data_phi[1],label='initial location=(+0.07, +0.10)',color='green')
plt.plot(time_axis,traj_data_phi[4],label='initial location=(+0.07, +0.00)',color='black')
plt.plot(time_axis,traj_data_phi[3],label='initial location=(+0.07, -0.04)',color='yellow')
plt.plot(time_axis,traj_data_phi[2],label='initial location=(+0.15, -0.01)',color='orange')




plt.xlabel("Time")
plt.ylabel("Barrier Function Phi")
plt.title('Recovery of the system from an initial unsafe state')
plt.axhline(y=0,c='r',linestyle='--', label='safe-level')
plt.legend(loc='lower right', borderpad=0.4, labelspacing=0.7)


plt.savefig(os.path.join(base_path,"Plots","Recovery_Plots.png"),format="png", bbox_inches="tight")

plt.show()


#Will get the recovery times for the bar plots.
recovery_time=np.zeros(5)

for i in range(5):
    rt=0
    for t in range(np.shape(time_axis)[0]):
        if (traj_data_phi[i][t]<0):
            rt=time_axis[t]
    recovery_time[i]=rt

initila_unsafe_start_points=['S1', 'S2', 'S3', 'S4', 'S5']
recovery_time_data={'S1': recovery_time[0], 'S2': recovery_time[1], 'S3': recovery_time[4], 'S4': recovery_time[3], 'S5': recovery_time[2]}
points=list(recovery_time_data.keys())
recovery_times=list(recovery_time_data.values())

plt.bar(points, recovery_times, color='maroon',
        width=0.4)

plt.xlabel("Initial Unsafe points")
plt.ylabel("Time needed to reach the safe region")
plt.title("Recovery Times for different unsafe start points")
plt.savefig(os.path.join(base_path,"Plots","Bar_Recovery_Time_Plot.png"),format="png", bbox_inches="tight")
plt.show()





