import random
import pickle
from GridSpace import GridSpace

class Q:
    def __init__(self, state_data_filename, state_space_repr, action_space_repr, learning_rate, 
                 discount_factor, epsilon, 
                 ):
        self.state_space_repr = state_space_repr
        self.action_space_repr = action_space_repr
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_init_value = lambda  : 0 #random.random

        try:
            with open(state_data_filename, 'rb') as f:
                table = pickle.load(f)
                self.q_table = table
        except FileNotFoundError:
            self.q_table = {}
        
            # if env is GridSpace just initialize
            # all states at one time
            if (type(self.state_space_repr) == GridSpace):
                for i in range(self.state_space_repr.n_rows):
                    for j in range(self.state_space_repr.n_cols):
                        state = (i,j)
                        self.q_table[state] = {i:self.q_init_value() for i in range(self.action_space_repr.n)} 

        # self.history_statistics = {
        # }
        self.state_lookup_misses = 0
        # self.delta_history = []
        return

    def best_action(self, observation):
        curr_state = self.state_space_repr.map_observation_to_repr(observation)
        
        if curr_state not in self.q_table:
            self.state_lookup_misses += 1
            #place current state in table
            self.q_table[curr_state] = {i:self.q_init_value() for i in range(self.action_space_repr.n)}
        
        if random.random() < self.epsilon:
            return self.action_space_repr.sample()

        all_action_values_equal = True
        best_action = 0
        best_value = self.q_table[curr_state][best_action]
        
        for action in self.q_table[curr_state]:
            value = self.q_table[curr_state][action] 
            if (value > best_value):
                all_action_values_equal = False
                best_action = action
                best_value = value

        if (all_action_values_equal):
            best_action = self.action_space_repr.sample()

        return best_action
    
    def update(self, observation, action, reward, next_observation):
        curr_state = self.state_space_repr.map_observation_to_repr(observation)
        next_state = self.state_space_repr.map_observation_to_repr(next_observation)

        
        if curr_state not in self.q_table:
            #place current state in table
            self.state_lookup_misses += 1
            self.q_table[curr_state] = {i:self.q_init_value() for i in range(self.action_space_repr.n)}

        if next_state not in self.q_table:
            #place next state in table
            self.state_lookup_misses += 1
            self.q_table[next_state] = {i:self.q_init_value() for i in range(self.action_space_repr.n)}

        
        #update q value
        q_before_update = self.q_table[curr_state][action]
        current_value_information = (1 - self.learning_rate) * self.q_table[curr_state][action]
        temporal_diff_target = self.learning_rate * (reward + self.discount_factor * max(self.q_table[next_state].values()) )
        self.q_table[curr_state][action] = current_value_information + temporal_diff_target  

        delta = self.q_table[curr_state][action] - q_before_update
        # print( (old_value, self.q_table[repr][action], delta, temporal_diff_target), sep="", end=" " )
        # self.delta_history.append(delta)
        return delta