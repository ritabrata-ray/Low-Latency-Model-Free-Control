import numpy as np
import matplotlib.pyplot as plt
import os

base_path="/Users/ritabrataray/Desktop/Safe control/Model_Free_RL"

data_path_safe=os.path.join(base_path, "Data/Safe")
data_path_benchmark_1=os.path.abspath(os.path.join(base_path,"Data/Benchmarks/Benchmark_1_Safety_based_objective"))
data_path_benchmark_2=os.path.abspath(os.path.join(base_path,"Data/Benchmarks/Benchmark_2_CPO"))


safe_safety_rate=np.load(os.path.join(data_path_safe,'safe_controller_safety_rate.npy'))
safe_convergence_rate=np.load(os.path.join(data_path_safe,'safe_controller_convergence_rate.npy'))
safe_safety_time=np.load(os.path.join(data_path_safe,'safe_safety_time.npy'))


b1_safety_rate=np.load(os.path.join(data_path_benchmark_1,'benchmark_1_safety_rate.npy'))
b1_convergence_rate=np.load(os.path.join(data_path_benchmark_1,'benchmark_1_convergence_rate.npy'))
b1_safety_time=np.load(os.path.join(data_path_benchmark_1,'benchmark_1_safety_time.npy'))

b2_safety_rate=np.load(os.path.join(data_path_benchmark_2,'benchmark_2_safety_rate.npy'))
b2_convergence_rate=np.load(os.path.join(data_path_benchmark_2,'benchmark_2_convergence_rate.npy'))
b2_safety_time=np.load(os.path.join(data_path_benchmark_2,'benchmark_2_safety_time.npy'))



total_number_of_rollouts=np.load(os.path.join(data_path_safe,'safe_max_episode_num.npy'))
total_number_of_rollouts=int(total_number_of_rollouts)


training_time=np.arange(total_number_of_rollouts)

my_path=os.path.abspath("/Users/ritabrataray/Desktop/Safe control/Model_Free_RL/Plots")

plt.plot(training_time,safe_convergence_rate,label='our safe controller',color='red')
plt.plot(training_time,b1_convergence_rate,label='benchmark_1: safety-based rewards',color='blue')
plt.plot(training_time,b2_convergence_rate,label='benchmark_2: CPO',color='green')


plt.xlabel("Training Time")
plt.ylabel("Number of steps till destination/Horizon")
plt.title("Convergence of model-free safe RL algorithms")
#plt.axhline(y=0,c='r',linestyle='--', label='safe-level')
plt.legend(loc='upper right', borderpad=0.4, labelspacing=0.7)


plt.savefig(os.path.join(my_path,"Convergence_Plots.png"),format="png", bbox_inches="tight")

plt.show()


plt.plot(training_time,safe_safety_rate,label='our safe controller',color='red')
plt.plot(training_time,b1_safety_rate,label='benchmark_1: safety-based rewards',color='blue')
plt.plot(training_time,b2_safety_rate,label='benchmark_2: CPO',color='green')


plt.xlabel("Training Time")
plt.ylabel("Fraction of time unsafe in the training epoch")
plt.title("Safety rates of model-free safe RL algorithms")
#plt.axhline(y=0,c='r',linestyle='--', label='safe-level')
plt.legend(loc='upper right', borderpad=0.4, labelspacing=0.7)


plt.savefig(os.path.join(my_path,"Safety_Rate_Plots.png"),format="png", bbox_inches="tight")

plt.show()

plt.plot(training_time,safe_safety_time,label='our safe controller',color='red')
plt.plot(training_time,b1_safety_time,label='benchmark_1: safety-based rewards',color='blue')
plt.plot(training_time,b2_safety_time,label='benchmark_2: CPO',color='green')


plt.xlabel("Training Time")
plt.ylabel("Time Steps after which the system is always safe")
plt.title("Safety Times of model-free safe RL algorithms")
#plt.axhline(y=0,c='r',linestyle='--', label='safe-level')
plt.legend(loc='upper right', borderpad=0.4, labelspacing=0.7)


plt.savefig(os.path.join(my_path,"Safety_Time_Plots.png"),format="png", bbox_inches="tight")

plt.show()