import numpy as np
import matplotlib.pyplot as plt
import os

max_episode_num=5000
max_steps=100
num_trials=10

base_path="/Users/ritabrataray/Desktop/Safe control/Model_Free_RL"
data_path=os.path.join(base_path,"Data/Benchmarks/Benchmark_2_CPO")
my_path=os.path.join(base_path,"Plots")

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

convergence_data=np.zeros((num_trials,max_episode_num))
safety_rate_data=np.zeros((num_trials,max_episode_num))
safety_time_data=np.zeros((num_trials,max_episode_num))

#Loading the simulation data
convergence_data[0]=np.load(os.path.join(data_path,'benchmark_2_convergence_rate_1.npy'))
safety_rate_data[0]=np.load(os.path.join(data_path,'benchmark_2_safety_rate_1.npy'))
safety_time_data[0]=np.load(os.path.join(data_path,'benchmark_2_safety_time_1.npy'))

convergence_data[1]=np.load(os.path.join(data_path,'benchmark_2_convergence_rate_2.npy'))
safety_rate_data[1]=np.load(os.path.join(data_path,'benchmark_2_safety_rate_2.npy'))
safety_time_data[1]=np.load(os.path.join(data_path,'benchmark_2_safety_time_2.npy'))

convergence_data[2]=np.load(os.path.join(data_path,'benchmark_2_convergence_rate_3.npy'))
safety_rate_data[2]=np.load(os.path.join(data_path,'benchmark_2_safety_rate_3.npy'))
safety_time_data[2]=np.load(os.path.join(data_path,'benchmark_2_safety_time_3.npy'))

convergence_data[3]=np.load(os.path.join(data_path,'benchmark_2_convergence_rate_4.npy'))
safety_rate_data[3]=np.load(os.path.join(data_path,'benchmark_2_safety_rate_4.npy'))
safety_time_data[3]=np.load(os.path.join(data_path,'benchmark_2_safety_time_4.npy'))

convergence_data[4]=np.load(os.path.join(data_path,'benchmark_2_convergence_rate_5.npy'))
safety_rate_data[4]=np.load(os.path.join(data_path,'benchmark_2_safety_rate_5.npy'))
safety_time_data[4]=np.load(os.path.join(data_path,'benchmark_2_safety_time_5.npy'))

convergence_data[5]=np.load(os.path.join(data_path,'benchmark_2_convergence_rate_6.npy'))
safety_rate_data[5]=np.load(os.path.join(data_path,'benchmark_2_safety_rate_6.npy'))
safety_time_data[5]=np.load(os.path.join(data_path,'benchmark_2_safety_time_6.npy'))

convergence_data[6]=np.load(os.path.join(data_path,'benchmark_2_convergence_rate_7.npy'))
safety_rate_data[6]=np.load(os.path.join(data_path,'benchmark_2_safety_rate_7.npy'))
safety_time_data[6]=np.load(os.path.join(data_path,'benchmark_2_safety_time_7.npy'))

convergence_data[7]=np.load(os.path.join(data_path,'benchmark_2_convergence_rate_8.npy'))
safety_rate_data[7]=np.load(os.path.join(data_path,'benchmark_2_safety_rate_8.npy'))
safety_time_data[7]=np.load(os.path.join(data_path,'benchmark_2_safety_time_8.npy'))

convergence_data[8]=np.load(os.path.join(data_path,'benchmark_2_convergence_rate_9.npy'))
safety_rate_data[8]=np.load(os.path.join(data_path,'benchmark_2_safety_rate_9.npy'))
safety_time_data[8]=np.load(os.path.join(data_path,'benchmark_2_safety_time_9.npy'))

convergence_data[9]=np.load(os.path.join(data_path,'benchmark_2_convergence_rate_10.npy'))
safety_rate_data[9]=np.load(os.path.join(data_path,'benchmark_2_safety_rate_10.npy'))
safety_time_data[9]=np.load(os.path.join(data_path,'benchmark_2_safety_time_10.npy'))

training_episodes = np.arange(max_episode_num)
M, U, L = error_band_data(training_episodes, safety_rate_data)
np.save(os.path.join(data_path,'benchmark_2_safety_rate'),M)
#np.save(os.path.join(data_path,'benchmark_2_safety_time'),safety_time)
plt.plot(training_episodes, M, label='average over trials')
#plt.plot(training_episodes,U,label='std_dev above average')
#plt.plot(training_episodes,L,label='std_dev below average')
plt.xlabel('Training Episode')
plt.ylabel('Fraction of steps unsafe')
plt.title('Fraction of time duration the system was unsafe')
plt.legend(loc='upper right', borderpad=0.4, labelspacing=0.7)
plt.fill_between(training_episodes, L, U, color='green', alpha=0.1)
#plt.axhline(y=0.0, c='r')
#plt.grid()
plt.savefig(os.path.join(my_path,"Fraction_unsafe_Benchmark_2.png"),format="png", bbox_inches="tight")
plt.show()
np.save(os.path.join(my_path,'benchmark_2_safety_rate_M'),M)
np.save(os.path.join(my_path,'benchmark_2_safety_rate_U'),U)
np.save(os.path.join(my_path,'benchmark_2_safety_rate_L'),L)


M,U,L= error_band_data(training_episodes,convergence_data)
np.save(os.path.join(data_path,'benchmark_2_convergence_rate'),M)
plt.plot(training_episodes,M,label='average over trials')
#plt.plot(training_episodes,U,label='std_dev above average')
#plt.plot(training_episodes,L,label='std_dev below average')
plt.xlabel('Training Episode')
plt.ylabel('Number of steps till destination/Horizon')
plt.title('Policy Gradient Learning Convergence')
plt.legend(loc='upper right',borderpad=0.4,labelspacing=0.7)
plt.fill_between(training_episodes,L,U,color='green',alpha=0.1)
#plt.axhline(y=0.0,c='r')
#plt.grid()
plt.savefig(os.path.join(my_path,"Convergence_Benchmark_1.png"),format="png", bbox_inches="tight")
plt.show()
np.save(os.path.join(my_path,'benchmark_2_convergence_rate_M'),M)
np.save(os.path.join(my_path,'benchmark_2_convergence_rate_U'),U)
np.save(os.path.join(my_path,'benchmark_2_convergence_rate_L'),L)