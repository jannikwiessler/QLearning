import gym
import numpy as np
import os
from q_table import Qtable


env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 2000

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = Qtable(env.observation_space.high, env.observation_space.low, [20, 20], env.action_space.n)

for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        os.system("cls")
        print(f"Current episode: {episode}")
        render = True
    else: 
        render = False
    states = env.reset()
    done = False
    while not done:
        action = q_table.get_action(states)
        new_states, reward, done, _ = env.step(action) # state containing car's position and velocity
        if render:
            env.render()
        if not done:
            max_future_q = q_table.get_max_q_value(new_states)
            current_q = q_table.get_max_q_value(states)
            new_q = (1- LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table.set_q_value(states, action, new_q)
        elif new_states[0] >= env.goal_position:
            print(f"Made it on episode: {episode}")
            q_table.set_q_value(states, action, 0)
        states = new_states
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING_
        epsilon-=epsilon_decay_value

env.close()


