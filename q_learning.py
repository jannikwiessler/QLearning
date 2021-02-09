import numpy as np
import os
import matplotlib.pyplot as plt 

eps = {"init_value": 0.5, "startepisode_of_decaying_eps": 1, "endepisode_of_decaying_eps": 500}

class QLeaning():
    def __init__(self, environment, q_table, learning_rate=0.1, discount=0.95, episodes=2000, eps=eps,gui=False):
        self._interim_show_every = 200
        self._interim_save_every = 200
        self._learning_rate = learning_rate
        self._discount = discount
        self._num_episodes = episodes
        self._epsilon_dict = eps
        self._eps = eps["init_value"]
        self._q_table = q_table
        self._env = environment
        self._episode_reward = [] 
        self._aggr_episode_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
        self._calc_decay_value_of_eps()
        self._os_has_gui = gui

    def _calc_decay_value_of_eps(self):
        start = self._epsilon_dict["startepisode_of_decaying_eps"]
        end = self._epsilon_dict["endepisode_of_decaying_eps"]
        self._epsilon_decay_value = self._eps / (end - start)

    def __update_epsilon_value(self):    
        start = self._epsilon_dict["startepisode_of_decaying_eps"]
        end = self._epsilon_dict["endepisode_of_decaying_eps"]  
        current_episode = self._current_episode 
        if end >= current_episode >= start:
            self._eps-=self._epsilon_decay_value

    def __run_episode(self):
        current_episode = self._current_episode
        if current_episode % self._interim_show_every == 0 and self._os_has_gui:
            print(f"Current episode: {current_episode}")
            render = True
        else: 
            render = False
        states = self._env.reset()
        done = False
        episode_reward = 0
        while not done:
            if np.random.random() > self._eps:
                action = self._q_table.get_action(states)
            else:
                action = np.random.randint(0,self._q_table.get_number_of_actions())
            new_states, reward, done, _ = self._env.step(action) # state containing car's position and velocity
            episode_reward += reward
            if render:
                self._env.render()
            if not done:
                max_future_q = self._q_table.get_max_q_value(new_states)
                current_q = self._q_table.get_max_q_value(states)
                learning_rate = self._learning_rate
                discount = self._discount
                new_q = (1- learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
                self._q_table.set_q_value(states, action, new_q)
            elif new_states[0] >= self._env.goal_position:
                print(f"Made it on episode: {current_episode}")
                self._q_table.set_q_value(states, action, 0)
            states = new_states
        self._episode_reward.append(episode_reward)
        self.__update_epsilon_value()

    def __save_aggr_epsiode_reward(self):
        ep_rewards = self._episode_reward
        average_reward = sum(ep_rewards[-self._interim_save_every:])/len(ep_rewards[-self._interim_save_every:])
        self._aggr_episode_rewards['ep'].append(self._current_episode)
        self._aggr_episode_rewards['avg'].append(average_reward)
        self._aggr_episode_rewards['min'].append(min(ep_rewards[-self._interim_save_every:]))
        self._aggr_episode_rewards['max'].append(max(ep_rewards[-self._interim_save_every:]))
        print(f"Episode: {self._current_episode} avg: {average_reward} min: {min(ep_rewards[-self._interim_save_every:])} max: {max(ep_rewards[-self._interim_save_every:])}")

    def run_training(self):
        for episode in range(self._num_episodes):
            self._current_episode = episode
            self.__run_episode()
            if not episode % self._interim_save_every:
                self.__save_aggr_epsiode_reward()
        self._env.close()

    def show_training_results(self):
        plt.plot(self._aggr_episode_rewards['ep'], self._aggr_episode_rewards['avg'], label="avg")
        plt.plot(self._aggr_episode_rewards['ep'], self._aggr_episode_rewards['min'], label="min")
        plt.plot(self._aggr_episode_rewards['ep'], self._aggr_episode_rewards['max'], label="max")
        plt.legend(loc=2)
        plt.show()
