import matplotlib.pyplot as plt
from Q_learning.QLearning import call_env as q_env
from Value_Iteration.ValueIteration import value_iteration
import numpy as np

train_reward, test_reward = q_env(iterations=1000,gamma=0.9,render=False)
print(f'Q_learning Reward Average = {np.average(test_reward)}')
value_iter_rewards, value_iter_avg = value_iteration(epsilon=1e-6 ,gamma=0.9,iteration=1000,render=False)
print("Value Iteration Average rewards:", value_iter_avg)


fig, axs = plt.subplots(3, 1, figsize=(10, 7))

# Plot Q-Learning Learning Phase
axs[0].plot(train_reward, label='Q_Learning Learning Phase')
axs[0].set_title('Q-Learning Learning Phase')
axs[0].set_xlabel('Episodes')
axs[0].set_ylabel('Rewards')
axs[0].legend()

# Plot Q-Learning Testing Phase
axs[1].plot(test_reward, label='Q_Learning Testing Phase')
axs[1].set_title('Q-Learning Testing Phase')
axs[1].set_xlabel('Episodes')
axs[1].set_ylabel('Rewards')
axs[1].legend()

# Plot Value Iteration
axs[2].plot(value_iter_rewards, label='Value Iteration')
axs[2].set_title('Value Iteration')
axs[2].set_xlabel('Episodes')
axs[2].set_ylabel('Rewards')
axs[2].legend()

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()