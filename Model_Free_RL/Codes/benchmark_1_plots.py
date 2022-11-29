import numpy as np
import matplotlib.pyplot as plt
import os

base_path="/Users/ritabrataray/Desktop/Safe control/Model_Free_RL"
data_path=os.path.join(base_path,"Data/Benchmarks/Benchmark_1_Safety_based_objective")
def error_band_data(X,Y):
    if(X.shape[0]!=Y.shape[1]):
        print("Incorrect dimensions of data!")
        return
    N=X.shape[0]
    U=np.zeros(N)
    L=np.zeros(N)
    M=np.zeros(N)
    num_trials=Y.shape[0]
    for i in range(N):
        mean=0
        for j in range(num_trials):
            mean+=Y[j][i]
        mean=(mean/num_trials)
        variance=0
        for j in range(num_trials):
            variance+=(Y[j][i]-mean)**2
        variance=(variance/num_trials)
        std_dev=np.sqrt(variance)
        M[i]=mean
        U[i]=mean+std_dev
        L[i]=mean-std_dev
    return M,U,L

num_trials=np.load(os.path.join(data_path,'num_trials.npy'))
num_trials=int(num_trials)
max_episode_num=np.load(os.path.join(data_path,'max_episode_num.npy'))
max_episode_num=int(max_episode_num)
max_steps=np.load(os.path.join(data_path,'max_steps.npy'))
max_steps=int(max_steps)

#for storage
numsteps = np.zeros((num_trials,max_episode_num), dtype='int')
avg_numsteps = np.zeros((num_trials,max_episode_num))

traj_data_x=np.ones((num_trials,max_episode_num,max_steps))
traj_data_y=np.ones((num_trials,max_episode_num,max_steps))
traj_data_derivative_x=np.ones((num_trials,max_episode_num,max_steps))
traj_data_derivative_y=np.ones((num_trials,max_episode_num,max_steps))
traj_data_phi=np.zeros((num_trials,max_episode_num,max_steps))
traj_data_dot_phi=np.ones((num_trials,max_episode_num,max_steps))








#Loading Unsafe Policy Gradient Simulation Data
traj_data_phi[0]=np.load(os.path.join(data_path,'unsafe_simulation_data_1.npy'))
numsteps[0]=np.load(os.path.join(data_path,'unsafe_numsteps_1.npy'))
avg_numsteps[0]=np.load(os.path.join(data_path,'unsafe_average_numsteps_1.npy'))
traj_data_x[0]=np.load(os.path.join(data_path,'unsafe_simulation_traj_x_1.npy'))
traj_data_y[0]=np.load(os.path.join(data_path,'unsafe_simulation_traj_y_1.npy'))


traj_data_phi[1]=np.load(os.path.join(data_path,'unsafe_simulation_data_2.npy'))
numsteps[1]=np.load(os.path.join(data_path,'unsafe_numsteps_2.npy'))
avg_numsteps[1]=np.load(os.path.join(data_path,'unsafe_average_numsteps_2.npy'))
traj_data_x[1]=np.load(os.path.join(data_path,'unsafe_simulation_traj_x_2.npy'))
traj_data_y[1]=np.load(os.path.join(data_path,'unsafe_simulation_traj_y_2.npy'))

traj_data_phi[2]=np.load(os.path.join(data_path,'unsafe_simulation_data_3.npy'))
numsteps[2]=np.load(os.path.join(data_path,'unsafe_numsteps_3.npy'))
avg_numsteps[2]=np.load(os.path.join(data_path,'unsafe_average_numsteps_3.npy'))
traj_data_x[2]=np.load(os.path.join(data_path,'unsafe_simulation_traj_x_3.npy'))
traj_data_y[2]=np.load(os.path.join(data_path,'unsafe_simulation_traj_y_3.npy'))

traj_data_phi[3]=np.load(os.path.join(data_path,'unsafe_simulation_data_4.npy'))
numsteps[3]=np.load(os.path.join(data_path,'unsafe_numsteps_4.npy'))
avg_numsteps[3]=np.load(os.path.join(data_path,'unsafe_average_numsteps_4.npy'))
traj_data_x[3]=np.load(os.path.join(data_path,'unsafe_simulation_traj_x_4.npy'))
traj_data_y[3]=np.load(os.path.join(data_path,'unsafe_simulation_traj_y_4.npy'))

traj_data_phi[4]=np.load(os.path.join(data_path,'unsafe_simulation_data_5.npy'))
numsteps[4]=np.load(os.path.join(data_path,'unsafe_numsteps_5.npy'))
avg_numsteps[4]=np.load(os.path.join(data_path,'unsafe_average_numsteps_5.npy'))
traj_data_x[4]=np.load(os.path.join(data_path,'unsafe_simulation_traj_x_5.npy'))
traj_data_y[4]=np.load(os.path.join(data_path,'unsafe_simulation_traj_y_5.npy'))

traj_data_phi[5]=np.load(os.path.join(data_path,'unsafe_simulation_data_6.npy'))
numsteps[5]=np.load(os.path.join(data_path,'unsafe_numsteps_6.npy'))
avg_numsteps[5]=np.load(os.path.join(data_path,'unsafe_average_numsteps_6.npy'))
traj_data_x[5]=np.load(os.path.join(data_path,'unsafe_simulation_traj_x_6.npy'))
traj_data_y[5]=np.load(os.path.join(data_path,'unsafe_simulation_traj_y_6.npy'))

traj_data_phi[6]=np.load(os.path.join(data_path,'unsafe_simulation_data_7.npy'))
numsteps[6]=np.load(os.path.join(data_path,'unsafe_numsteps_7.npy'))
avg_numsteps[6]=np.load(os.path.join(data_path,'unsafe_average_numsteps_7.npy'))
traj_data_x[6]=np.load(os.path.join(data_path,'unsafe_simulation_traj_x_7.npy'))
traj_data_y[6]=np.load(os.path.join(data_path,'unsafe_simulation_traj_y_7.npy'))

traj_data_phi[7]=np.load(os.path.join(data_path,'unsafe_simulation_data_8.npy'))
numsteps[7]=np.load(os.path.join(data_path,'unsafe_numsteps_8.npy'))
avg_numsteps[7]=np.load(os.path.join(data_path,'unsafe_average_numsteps_8.npy'))
traj_data_x[7]=np.load(os.path.join(data_path,'unsafe_simulation_traj_x_8.npy'))
traj_data_y[7]=np.load(os.path.join(data_path,'unsafe_simulation_traj_y_8.npy'))

traj_data_phi[8]=np.load(os.path.join(data_path,'unsafe_simulation_data_9.npy'))
numsteps[8]=np.load(os.path.join(data_path,'unsafe_numsteps_9.npy'))
avg_numsteps[8]=np.load(os.path.join(data_path,'unsafe_average_numsteps_9.npy'))
traj_data_x[8]=np.load(os.path.join(data_path,'unsafe_simulation_traj_x_9.npy'))
traj_data_y[8]=np.load(os.path.join(data_path,'unsafe_simulation_traj_y_9.npy'))

traj_data_phi[9]=np.load(os.path.join(data_path,'unsafe_simulation_data_10.npy'))
numsteps[9]=np.load(os.path.join(data_path,'unsafe_numsteps_10.npy'))
avg_numsteps[9]=np.load(os.path.join(data_path,'unsafe_average_numsteps_10.npy'))
traj_data_x[9]=np.load(os.path.join(data_path,'unsafe_simulation_traj_x_10.npy'))
traj_data_y[9]=np.load(os.path.join(data_path,'unsafe_simulation_traj_y_10.npy'))
#num_trials=7
#max_steps = 100
#max_episode_num=5600

my_path=os.path.join(base_path,"Plots")


#Plotting Unsafe Simulation Data

#First plot the fraction of duration in the episode when it was safe

traj_safety_rate = np.zeros((num_trials, max_episode_num))
for trial in range(num_trials):
    for epi in range(max_episode_num):
        usc = 0
        for s in range(numsteps[trial][epi]):
            if (traj_data_phi[trial][epi][s] < 0):
                usc = usc + 1
        traj_safety_rate[trial][epi] = (usc / numsteps[trial][epi])


safety_time=np.zeros((num_trials,max_episode_num))
for epi in range(max_episode_num):
    for trial in range(num_trials):
        n = int(numsteps[trial][epi])
        max = 0
        for s in range(n):
            if(traj_data_phi[trial][epi][s]<0):
                if (s>max):
                    max=s
        safety_time[trial][epi]=max


training_episodes = np.arange(max_episode_num)
M, U, L = error_band_data(training_episodes, traj_safety_rate)
np.save(os.path.join(data_path,'benchmark_1_safety_rate'),M)
np.save(os.path.join(data_path,'benchmark_1_safety_time'),safety_time)
plt.plot(training_episodes, M, label='average over trials')
# plt.plot(training_episodes,U,label='std_dev above average')
# plt.plot(training_episodes,L,label='std_dev below average')
plt.xlabel('Training Episode')
plt.ylabel('Fraction of steps unsafe')
plt.title('Fraction of time duration the system was unsafe')
plt.legend(loc='upper right', borderpad=0.4, labelspacing=0.7)
plt.fill_between(training_episodes, L, U, color='green', alpha=0.1)
#plt.axhline(y=0.0, c='r')
plt.grid()
plt.savefig(os.path.join(my_path,"Fraction_unsafe_Benchmark_1.png"),format="png", bbox_inches="tight")
plt.show()

np.save(os.path.join(my_path,'benchmark_1_safety_rate_M'),M)
np.save(os.path.join(my_path,'benchmark_1_safety_rate_U'),U)
np.save(os.path.join(my_path,'benchmark_1_safety_rate_L'),L)



M,U,L= error_band_data(training_episodes,avg_numsteps)
np.save(os.path.join(data_path,'benchmark_1_convergence_rate'),M)
plt.plot(training_episodes,M,label='average over trials')
#plt.plot(training_episodes,U,label='std_dev above average')
#plt.plot(training_episodes,L,label='std_dev below average')
plt.xlabel('Training Episode')
plt.ylabel('Number of steps till destination/Horizon')
plt.title('Policy Gradient Learning Convergence')
plt.legend(loc='upper right',borderpad=0.4,labelspacing=0.7)
plt.fill_between(training_episodes,L,U,color='green',alpha=0.1)
#plt.axhline(y=0.0,c='r')
plt.grid()
plt.savefig(os.path.join(my_path,"Convergence_Benchmark_1.png"),format="png", bbox_inches="tight")
plt.show()

np.save(os.path.join(my_path,'benchmark_1_convergence_rate_M'),M)
np.save(os.path.join(my_path,'benchmark_1_convergence_rate_U'),U)
np.save(os.path.join(my_path,'benchmark_1_convergence_rate_L'),L)
