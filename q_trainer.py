import gymnasium as gym
from state_space_representation import state_space_representation
import pickle
from GridSpace import GridSpace
from GridWorldEnv import GridWorldEnv
from Q import Q
import numpy as np
import win32api
import matplotlib.pyplot as plt

class q_trainer:
    def __init__(self, env_label, state_data_filename, n_states_in_each_dim, weigh_reward_with_time,
                 PARAM_number_of_episodes=0, PARAM_episode_max_length=2000, PARAM_learning_rate=0.999, 
                 PARAM_discount_factor=0.89, PARAM_epsilon=0.2, show_windows=False,
                 ): 
        assert env_label in (0,1)
        
        self.env_label = env_label
        self.show_windows = show_windows

        # initialize environment, state space and action space
        match env_label:
            case 0:
                if (not self.show_windows):
                    self.env = gym.make('CartPole-v1')
                else:
                    self.env = gym.make('CartPole-v1', render_mode='human')
                    # The cartpole state space contains continuous
                    # valued components and thus needs to be binned.
                self.state_space_repr = state_space_representation(
                    env=gym.make('CartPole-v1'),
                    num_states_in_each_dim=n_states_in_each_dim
                )
                if (PARAM_number_of_episodes == 0):
                    self.PARAM_number_of_episodes = 1000 if not self.show_windows else 10
            case 1:
                self.env = GridWorldEnv()
                self.state_space_repr = GridSpace()
                if (PARAM_number_of_episodes == 0):
                    self.PARAM_number_of_episodes = 10

        self.state_data_filename = state_data_filename
        
        self.PARAM_number_of_episodes = PARAM_number_of_episodes
        self.PARAM_episode_max_length = PARAM_episode_max_length
        self.PARAM_learning_rate = PARAM_learning_rate
        self.PARAM_discount_factor = PARAM_discount_factor
        self.PARAM_epsilon = PARAM_epsilon
        

        self.q = Q(
            state_data_filename=self.state_data_filename,
            state_space_repr=self.state_space_repr,
            action_space_repr=self.env.action_space,
            learning_rate=self.PARAM_learning_rate, 
            discount_factor=self.PARAM_discount_factor,
            epsilon=self.PARAM_epsilon
        )


        self.weigh_reward_with_time = weigh_reward_with_time

        self.mean_reward_per_session = []
        return

    def save_q_table(self):
        with open(self.state_data_filename, 'wb') as f:
            pickle.dump(self.q.q_table, f)
        return
    
    def run_session(self):
        reward_history = []
        n_states_start = len(self.q.q_table)
        for episode in range(self.PARAM_number_of_episodes):
            observation, _ = self.env.reset()
            
            episode_delta = 0
            episode_reward = 0
            # misses_at_start = self.q.state_lookup_misses
            for t in range(self.PARAM_episode_max_length):
                if ((self.env_label == 0) and self.show_windows):
                    self.env.render()
                action = self.q.best_action(observation)
                next_observation, reward, terminate_episode_signal, _, step_info = self.env.step(action)
                time_weighted_reward = reward + t
                feedback = time_weighted_reward if self.weigh_reward_with_time else reward
                
                # if (terminate_episode_signal and (t < 200)):
                #     feedback -= (30000 * t)
                episode_delta += abs(self.q.update(observation, action, feedback, next_observation))
                
                # state_watch = (observation, self.env.action_space_labels[action], reward, next_observation, delta)
                # print(state_watch)
                

                episode_reward += reward
                observation = next_observation

                if terminate_episode_signal:
                    break
                
                if not (reward == 1):
                    pass
            
            reward_history.append(episode_reward)

            # misses_during_episode = self.q.state_lookup_misses - misses_at_start
            episodes_left = self.PARAM_number_of_episodes - episode - 1
            discovered_states = len(self.q.q_table) - n_states_start
            # perc_state_space_discovered = len(self.q.q_table) / undiscovered_states
            # print(episode_delta)
            # exp_info = (episodes_left, undiscovered_states, perc_state_space_discovered, episode_delta, episode_reward, t)
            exp_info = (episodes_left, discovered_states, episode_delta, episode_reward, t) 
            print(exp_info)
            if ((self.env_label == 1) and (self.show_windows)):
                self.gridworld_heatmap()
        
        session_reward = np.mean(reward_history)
        # print((session_reward))
        # self.mean_reward_per_session.append(session_reward)
        n_states_end = len(self.q.q_table)
        states_found = n_states_end - n_states_start
        # print(f'States found: {states_found}')
        
        stat = (session_reward, states_found)
        with open('stats.txt', '+a') as f:
            f.write("," + str(stat))
        print(stat)
        self.save_q_table()
        win32api.MessageBox(0, 'Session Done', 'DoneMessage', 0x00001000)
        return

    def gridworld_heatmap(self):
        self.for_heatmap = []
        for i in range(self.state_space_repr.n_rows):
            row = []
            for j in range(self.state_space_repr.n_cols):
                state = self.q.q_table[(i,j)]
                avg_value = np.mean(list(state.values()))
                row.append(avg_value)
            self.for_heatmap.append(row)
        fig, ax = plt.subplots()
        im = ax.imshow(self.for_heatmap)

        for i in range(self.state_space_repr.n_rows):
            for j in range(self.state_space_repr.n_cols):
                text_to_show = "{:0.5f}".format(self.for_heatmap[i][j])
                ax.text(j, i, text_to_show,
                    ha='center', va='center', color='w',
                    
                )

        plt.show()
        return