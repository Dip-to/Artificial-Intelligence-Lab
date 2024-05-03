import numpy as np
import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt

class ValueIterationAgent:
    def __init__(self, env, gamma=0.8, epsilon=1e-3, iteration=100, render=False):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.v_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.max_iterations = iteration
        self.render = render

    def value_iteration(self):
        delta_history = []
        num_states = self.env.observation_space.n
        num_actions = self.env.action_space.n
        V = np.zeros(num_states)

        state_convergence = defaultdict(list)  # Stores convergence for each state

        for i in range(self.max_iterations):
            for s in range(num_states):
                delta = self.epsilon
                previous_values = defaultdict(lambda: np.zeros(num_actions), self.v_values)
                v = max(self.v_values[s])

                for a in range(num_actions):
                    future_reward = []
                    current_reward = []

                    for trans_prob, next_state, reward_prob, done in self.env.unwrapped.P[s][a]:
                        future_reward.append(trans_prob * max(previous_values[next_state]))
                        current_reward.append(trans_prob * reward_prob)

                    sum_of_future_rewards = sum(future_reward)
                    sum_of_current_rewards = sum(current_reward)
                    self.v_values[s][a] = sum_of_current_rewards + self.gamma * sum_of_future_rewards

                    delta = max(delta, abs(sum_of_current_rewards - self.v_values[s][a]))

                if delta != 0 and (delta > 0.15 or len(delta_history) < 5):
                    delta_history.append(delta)

                state_convergence[s].append(self.v_values[s].copy())

        iterations = 1000
        all_rewards = []

        for i_episode in range(iterations):
            obs, info = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(obs)
                obs, reward, done, _, info = self.env.step(action)
                total_reward += reward

            all_rewards.append(total_reward)

            if self.render:
                self.env.render()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot state convergence for each state
        for state, convergence in state_convergence.items():
            axes[0].plot(range(len(convergence)), convergence, label=f"State {state}")

        axes[0].set_title('Convergence of State Values')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Value')

        # Plot delta history
        axes[1].plot(range(len(delta_history)), delta_history)
        axes[1].set_title('Convergence of delta')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Delta')

        plt.tight_layout()
        plt.show()


        return all_rewards, np.average(all_rewards)

    def choose_action(self, obs):
        return np.argmax(self.v_values[obs])

def value_iteration(gamma=0.8, epsilon=1e-6, iteration=100, render=False):
    env = gym.make('FrozenLake-v1', is_slippery=True)
    agent = ValueIterationAgent(env, gamma, epsilon, iteration, render)
    return agent.value_iteration()
