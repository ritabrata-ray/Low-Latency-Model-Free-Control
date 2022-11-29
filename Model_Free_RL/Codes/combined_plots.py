import numpy as np
import matplotlib.pyplot as plt
import os

base_path="/Users/ritabrataray/Desktop/Safe control/Model_Free_RL"
my_path=os.path.join(base_path,"Plots")

#Loading data for plots

unsafe_convergence_rate_M=np.load(os.path.join(my_path,'unsafe_convergence_rate_M.npy'))
unsafe_convergence_rate_U=np.load(os.path.join(my_path,'unsafe_convergence_rate_U.npy'))
unsafe_convergence_rate_L=np.load(os.path.join(my_path,'unsafe_convergence_rate_L.npy'))

our_algorithm_convergence_rate_M=np.load(os.path.join(my_path,'our_algorithm_convergence_rate_M.npy'))
our_algorithm_convergence_rate_U=np.load(os.path.join(my_path,'our_algorithm_convergence_rate_U.npy'))
our_algorithm_convergence_rate_L=np.load(os.path.join(my_path,'our_algorithm_convergence_rate_L.npy'))

benchmark_1_convergence_rate_M=np.load(os.path.join(my_path,'benchmark_1_convergence_rate_M.npy'))
benchmark_1_convergence_rate_U=np.load(os.path.join(my_path,'benchmark_1_convergence_rate_U.npy'))
benchmark_1_convergence_rate_L=np.load(os.path.join(my_path,'benchmark_1_convergence_rate_L.npy'))

benchmark_2_convergence_rate_M=np.load(os.path.join(my_path,'benchmark_2_convergence_rate_M.npy'))
benchmark_2_convergence_rate_U=np.load(os.path.join(my_path,'benchmark_2_convergence_rate_U.npy'))
benchmark_2_convergence_rate_L=np.load(os.path.join(my_path,'benchmark_2_convergence_rate_L.npy'))



unsafe_safety_rate_M=np.load(os.path.join(my_path,'unsafe_safety_rate_M.npy'))
unsafe_safety_rate_U=np.load(os.path.join(my_path,'unsafe_safety_rate_U.npy'))
unsafe_safety_rate_L=np.load(os.path.join(my_path,'unsafe_safety_rate_L.npy'))

our_algorithm_safety_rate_M=np.load(os.path.join(my_path,'our_algorithm_safety_rate_M.npy'))
our_algorithm_safety_rate_U=np.load(os.path.join(my_path,'our_algorithm_safety_rate_U.npy'))
our_algorithm_safety_rate_L=np.load(os.path.join(my_path,'our_algorithm_safety_rate_L.npy'))

benchmark_1_safety_rate_M=np.load(os.path.join(my_path,'benchmark_1_safety_rate_M.npy'))
benchmark_1_safety_rate_U=np.load(os.path.join(my_path,'benchmark_1_safety_rate_U.npy'))
benchmark_1_safety_rate_L=np.load(os.path.join(my_path,'benchmark_1_safety_rate_L.npy'))

benchmark_2_safety_rate_M=np.load(os.path.join(my_path,'benchmark_2_safety_rate_M.npy'))
benchmark_2_safety_rate_U=np.load(os.path.join(my_path,'benchmark_2_safety_rate_U.npy'))
benchmark_2_safety_rate_L=np.load(os.path.join(my_path,'benchmark_2_safety_rate_L.npy'))

max_rollout_num=5000
max_steps=100

rollouts=np.arange(max_rollout_num)


plt.plot(rollouts,unsafe_convergence_rate_M,label='REINFORCE',color='black')
plt.fill_between(rollouts,unsafe_convergence_rate_L,unsafe_convergence_rate_U,color='black',alpha=0.1)
plt.plot(rollouts,benchmark_1_convergence_rate_M, label='REINFORCE with rewards penalized for safety',color='green')
plt.fill_between(rollouts,benchmark_1_convergence_rate_L,benchmark_1_convergence_rate_U,color='green',alpha=0.1)
plt.plot(rollouts,benchmark_2_convergence_rate_M, label='CPO',color='blue')
plt.fill_between(rollouts,benchmark_2_convergence_rate_L,benchmark_2_convergence_rate_U,color='blue',alpha=0.1)
plt.plot(rollouts,our_algorithm_convergence_rate_M,label='Our Algorithm',color='red')
plt.fill_between(rollouts,our_algorithm_convergence_rate_L,our_algorithm_convergence_rate_U,color='red',alpha=0.1)
plt.xlabel("Training Time")
plt.ylabel("Number of steps till destination/Horizon")
plt.title("Convergence of model-free safe RL algorithms")
plt.legend(loc='upper right', borderpad=0.4, labelspacing=0.7)
plt.savefig(os.path.join(my_path,"Combined_Convergence_Plots.png"),format="png", bbox_inches="tight")
plt.show()


plt.plot(rollouts,unsafe_safety_rate_M,label='REINFORCE',color='black')
#plt.fill_between(rollouts,unsafe_safety_rate_L,unsafe_safety_rate_U,color='black',alpha=0.1)
plt.plot(rollouts,benchmark_1_safety_rate_M, label='REINFORCE with rewards penalized for safety',color='green')
#plt.fill_between(rollouts,benchmark_1_safety_rate_L,benchmark_1_safety_rate_U,color='green',alpha=0.1)
plt.plot(rollouts,benchmark_2_safety_rate_M, label='CPO',color='blue')
#plt.fill_between(rollouts,benchmark_2_safety_rate_L,benchmark_2_safety_rate_U,color='blue',alpha=0.1)
plt.plot(rollouts,our_algorithm_safety_rate_M,label='Our Algorithm',color='red')
#plt.fill_between(rollouts,our_algorithm_safety_rate_L,our_algorithm_safety_rate_U,color='red',alpha=0.1)
plt.xlabel("Training Time")
plt.ylabel("Fraction of time unsafe in the training Episode")
plt.title("Safety rates of model-free safe RL algorithms")
plt.legend(loc='center right', borderpad=0.4, labelspacing=0.7)
plt.savefig(os.path.join(my_path,"Combined_Safety_Rate_Plots.png"),format="png", bbox_inches="tight")
plt.show()

#get bar plots of safety rates of all four algorithms

sr_unsafe=0
for i in range(rollouts.shape[0]):
    sr_unsafe+=unsafe_safety_rate_M[i]
sr_unsafe=(sr_unsafe/rollouts.shape[0])

sr_b_1=0
for i in range(rollouts.shape[0]):
    sr_b_1+=benchmark_1_safety_rate_M[i]
sr_b_1=(sr_b_1/rollouts.shape[0])

sr_b_2=0
for i in range(rollouts.shape[0]):
    sr_b_2+=benchmark_2_safety_rate_M[i]
sr_b_2=(sr_b_2/rollouts.shape[0])

sr_our_algorithm=0
for i in range(rollouts.shape[0]):
    sr_our_algorithm+=our_algorithm_safety_rate_M[i]
sr_our_algorithm=(sr_our_algorithm/rollouts.shape[0])

algorithms=['REINFORCE', 'Benchmark 1', 'Benchmark 2', 'Our Algorithm']
mean_safety_rates={'Benchmark 1': sr_b_1, 'Benchmark 2': sr_b_2, 'Our Algorithm': sr_our_algorithm}
algorithms=list(mean_safety_rates.keys())
safety_rates=list(mean_safety_rates.values())

plt.bar(algorithms, safety_rates, color='maroon',
        width=0.4)

plt.xlabel("Model-Free RL Algorithm")
plt.ylabel("Fraction of the duration the system was unsafe")
plt.title("Safety Rates of Different Algorithms")
plt.savefig(os.path.join(my_path,"Bar_Safety_Rate_Plot.png"),format="png", bbox_inches="tight")
plt.show()
