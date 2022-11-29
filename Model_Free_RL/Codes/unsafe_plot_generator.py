import numpy as np
import matplotlib.pyplot as plt
import os

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

base_path="/Users/ritabrataray/Desktop/Safe control/Model_Free_RL"
data_path=os.path.join(base_path,"Data/Unsafe")
#Loading Unsafe Policy Gradient Simulation Data
traj_data_phi=np.load(os.path.join(data_path,'unsafe_simulation_data.npy'))
numsteps=np.load(os.path.join(data_path,'unsafe_numsteps.npy'))
avg_numsteps=np.load(os.path.join(data_path,'unsafe_average_numsteps.npy'))
traj_data_x=np.load(os.path.join(data_path,'unsafe_simulation_traj_x.npy'))
traj_data_y=np.load(os.path.join(data_path,'unsafe_simulation_traj_y.npy'))

num_trials=np.load(os.path.join(data_path,'num_trials.npy'))
num_trials=int(num_trials)
max_episode_num=np.load(os.path.join(data_path,'max_episode_num.npy'))
max_episode_num=int(max_episode_num)
max_steps=np.load(os.path.join(data_path,'max_steps.npy'))
max_steps=int(max_steps)

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

training_episodes = np.arange(max_episode_num)
M, U, L = error_band_data(training_episodes, traj_safety_rate)
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
plt.savefig(os.path.join(my_path,"Fraction_unsafe.png"),format="png", bbox_inches="tight")
plt.show()

np.save(os.path.join(my_path,'unsafe_safety_rate_M'),M)
np.save(os.path.join(my_path,'unsafe_safety_rate_U'),U)
np.save(os.path.join(my_path,'unsafe_safety_rate_L'),L)




M,U,L= error_band_data(training_episodes,avg_numsteps)
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
plt.savefig(os.path.join(my_path,"Convergence_unsafe.png"),format="png", bbox_inches="tight")
plt.show()

np.save(os.path.join(my_path,'unsafe_convergence_rate_M'),M)
np.save(os.path.join(my_path,'unsafe_convergence_rate_U'),U)
np.save(os.path.join(my_path,'unsafe_convergence_rate_L'),L)