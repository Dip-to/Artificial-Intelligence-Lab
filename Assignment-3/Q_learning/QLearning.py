import random
from collections import defaultdict
import numpy as np
import gymnasium as gym

class QLearningAgent():
    def __init__(self, env, gamma=0.99, alpha=0.1, eps=0.9):
        self.gamma = gamma
        self.env = env
        self.q_vals = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))
        self.alpha = alpha
        self.eps = eps

    def choose_action(self, state):
        if random.random() < self.eps:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_vals[state])
        return action

    def learn(self, state, action, reward, next_state):
        max_val = np.max(self.q_vals[next_state])
        curr_new_val = reward + self.gamma * max_val
        old_value = self.q_vals[state][action]
        self.q_vals[state][action] = old_value + self.alpha * (curr_new_val - old_value)


def call_env(iterations, gamma,render=False):
    env = gym.make('FrozenLake-v1', is_slippery=True)
    agent = QLearningAgent(env=env, gamma=gamma)
    
    total_rewards = []
    for i in range(iterations):
        state, info = env.reset()
        total_reward = 0
        done = False
        while not done:
            current_state = state
            action = agent.choose_action(state)
            state, reward, done, _, info = env.step(round(action))
            agent.learn(current_state, action, reward, state)
            total_reward += reward

        total_rewards.append(total_reward)


    all_rewards = []
    agent.eps = 0
    for i in range(iterations):
        state, info = env.reset()
        total_reward = 0
        done = False
        while not done:
            current_state = state
            action = agent.choose_action(state)
            state, reward, done, _, info = env.step(round(action))
            total_reward += reward
        all_rewards.append(total_reward)
        # if render:
        #         env.render()
    env.close()

    return total_rewards, all_rewards
