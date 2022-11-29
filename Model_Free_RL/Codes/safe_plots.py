import numpy as np
import matplotlib.pyplot as plt
import os

base_path="/Users/ritabrataray/Desktop/Safe control/Model_Free_RL"

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



data_path = os.path.join(base_path,"Data/Safe")

num_trials=np.load(os.path.join(data_path,'safe_num_trials.npy'))
num_trials=int(num_trials)
max_episode_num=np.load(os.path.join(data_path,'safe_max_episode_num.npy'))
max_episode_num=int(max_episode_num)
max_steps=np.load(os.path.join(data_path,'safe_max_steps.npy'))
max_steps=int(max_steps)


safe_traj_data_phi=np.zeros((num_trials,max_episode_num,max_steps))

safe_traj_data_phi[0]=np.load(os.path.join(data_path,'safe_simulation_data_1.npy'))
safe_traj_data_phi[1]=np.load(os.path.join(data_path,'safe_simulation_data_2.npy'))
safe_traj_data_phi[2]=np.load(os.path.join(data_path,'safe_simulation_data_3.npy'))
safe_traj_data_phi[3]=np.load(os.path.join(data_path,'safe_simulation_data_4.npy'))
safe_traj_data_phi[4]=np.load(os.path.join(data_path,'safe_simulation_data_5.npy'))
safe_traj_data_phi[5]=np.load(os.path.join(data_path,'safe_simulation_data_6.npy'))
safe_traj_data_phi[6]=np.load(os.path.join(data_path,'safe_simulation_data_7.npy'))
safe_traj_data_phi[7]=np.load(os.path.join(data_path,'safe_simulation_data_8.npy'))
safe_traj_data_phi[8]=np.load(os.path.join(data_path,'safe_simulation_data_9.npy'))
safe_traj_data_phi[9]=np.load(os.path.join(data_path,'safe_simulation_data_10.npy'))
safe_traj_data_phi[10]=np.load(os.path.join(data_path,'safe_simulation_data_11.npy'))
safe_traj_data_phi[11]=np.load(os.path.join(data_path,'safe_simulation_data_12.npy'))
safe_traj_data_phi[12]=np.load(os.path.join(data_path,'safe_simulation_data_13.npy'))
safe_traj_data_phi[13]=np.load(os.path.join(data_path,'safe_simulation_data_14.npy'))
safe_traj_data_phi[14]=np.load(os.path.join(data_path,'safe_simulation_data_15.npy'))




avg_numsteps=np.zeros((num_trials,max_episode_num))
avg_numsteps[0]=np.load(os.path.join(data_path,'safe_average_numsteps_1.npy'))
avg_numsteps[1]=np.load(os.path.join(data_path,'safe_average_numsteps_2.npy'))
avg_numsteps[2]=np.load(os.path.join(data_path,'safe_average_numsteps_3.npy'))
avg_numsteps[3]=np.load(os.path.join(data_path,'safe_average_numsteps_4.npy'))
avg_numsteps[4]=np.load(os.path.join(data_path,'safe_average_numsteps_5.npy'))
avg_numsteps[5]=np.load(os.path.join(data_path,'safe_average_numsteps_6.npy'))
avg_numsteps[6]=np.load(os.path.join(data_path,'safe_average_numsteps_7.npy'))
avg_numsteps[7]=np.load(os.path.join(data_path,'safe_average_numsteps_8.npy'))
avg_numsteps[8]=np.load(os.path.join(data_path,'safe_average_numsteps_9.npy'))
avg_numsteps[9]=np.load(os.path.join(data_path,'safe_average_numsteps_10.npy'))
avg_numsteps[10]=np.load(os.path.join(data_path,'safe_average_numsteps_11.npy'))
avg_numsteps[11]=np.load(os.path.join(data_path,'safe_average_numsteps_12.npy'))
avg_numsteps[12]=np.load(os.path.join(data_path,'safe_average_numsteps_13.npy'))
avg_numsteps[13]=np.load(os.path.join(data_path,'safe_average_numsteps_14.npy'))
avg_numsteps[14]=np.load(os.path.join(data_path,'safe_average_numsteps_15.npy'))

numsteps=np.zeros((num_trials,max_episode_num))
numsteps[0]=np.load(os.path.join(data_path,'safe_numsteps_1.npy'))
numsteps[1]=np.load(os.path.join(data_path,'safe_numsteps_2.npy'))
numsteps[2]=np.load(os.path.join(data_path,'safe_numsteps_3.npy'))
numsteps[3]=np.load(os.path.join(data_path,'safe_numsteps_4.npy'))
numsteps[4]=np.load(os.path.join(data_path,'safe_numsteps_5.npy'))
numsteps[5]=np.load(os.path.join(data_path,'safe_numsteps_6.npy'))
numsteps[6]=np.load(os.path.join(data_path,'safe_numsteps_7.npy'))
numsteps[7]=np.load(os.path.join(data_path,'safe_numsteps_8.npy'))
numsteps[8]=np.load(os.path.join(data_path,'safe_numsteps_9.npy'))
numsteps[9]=np.load(os.path.join(data_path,'safe_numsteps_10.npy'))
numsteps[10]=np.load(os.path.join(data_path,'safe_numsteps_11.npy'))
numsteps[11]=np.load(os.path.join(data_path,'safe_numsteps_12.npy'))
numsteps[12]=np.load(os.path.join(data_path,'safe_numsteps_13.npy'))
numsteps[13]=np.load(os.path.join(data_path,'safe_numsteps_14.npy'))
numsteps[14]=np.load(os.path.join(data_path,'safe_numsteps_15.npy'))

traj_safety_rate = np.zeros((num_trials, max_episode_num))
for trial in range(num_trials):
    for epi in range(max_episode_num):
        sc = 0
        n=int(numsteps[trial][epi])
        for s in range(n):
            if (safe_traj_data_phi[trial][epi][s] < 0):
                    sc = sc + 1
        traj_safety_rate[trial][epi] = (sc / numsteps[trial][epi])
safety_time=np.zeros(max_episode_num)
for epi in range(max_episode_num):
    max=0
    for trial in range(num_trials):
        n = int(numsteps[trial][epi])
        for s in range(n):
            if(safe_traj_data_phi[trial][epi][s]<0):
                if (s>max):
                    max=s
    safety_time[epi]=max






#Plots
my_path=os.path.join(base_path,"Plots")
training_episodes = np.arange(max_episode_num)
M, U, L = error_band_data(training_episodes, traj_safety_rate)
np.save(os.path.join(data_path,'safe_controller_safety_rate'),M)
np.save(os.path.join(data_path,'safe_safety_time'),safety_time)
plt.plot(training_episodes, M, label='average over trials')
# plt.plot(training_episodes,U,label='std_dev above average')
# plt.plot(training_episodes,L,label='std_dev below average')
plt.xlabel('Training Episode')
plt.ylabel('Fraction of episode length')
plt.title('Fractional duration of unsafe system in the presence of correction controller')
plt.legend(loc='upper right', borderpad=0.4, labelspacing=0.7)
plt.fill_between(training_episodes, L, U, color='green', alpha=0.1)
#plt.axhline(y=0.0, c='r')
plt.grid()
plt.savefig(os.path.join(my_path,"Fraction_unsafe.png"),format="png", bbox_inches="tight")
plt.show()

np.save(os.path.join(my_path,'our_algorithm_safety_rate_M'),M)
np.save(os.path.join(my_path,'our_algorithm_safety_rate_U'),U)
np.save(os.path.join(my_path,'our_algorithm_safety_rate_L'),L)

convergence_time=np.arange(max_episode_num)
M,U,L= error_band_data(convergence_time,avg_numsteps)
np.save(os.path.join(data_path,'safe_controller_convergence_rate'),M)
plt.plot(convergence_time,M,label='average over trials')
#plt.plot(training_episodes,U,label='std_dev above average')
#plt.plot(training_episodes,L,label='std_dev below average')
plt.xlabel('Training Episode')
plt.ylabel('Number of steps till destination/Horizon')
plt.title('Convergence of policy gradient with correction controller')
plt.legend(loc='upper right',borderpad=0.4,labelspacing=0.7)
plt.fill_between(convergence_time,L,U,color='green',alpha=0.1)
#plt.axhline(y=0.0,c='r')
plt.grid()
plt.savefig(os.path.join(my_path,"Convergence_safe.png"),format="png", bbox_inches="tight")
plt.show()

np.save(os.path.join(my_path,'our_algorithm_convergence_rate_M'),M)
np.save(os.path.join(my_path,'our_algorithm_convergence_rate_U'),U)
np.save(os.path.join(my_path,'our_algorithm_convergence_rate_L'),L)