# This class was written for openai gym cartpole environment.
# The 'state space' is a superset of the observation space.
# Observations from the cartpole envirionment consists
# of 4 continuous components.
# States are discretized into intervals.

import pickle

# Samples environment in order to partition
# the observation space into intervals bounded
# by extreme values observed during sampling,
# and then by negative and positive infinity. 
class state_space_representation:
    def __init__(self, env, num_states_in_each_dim=30, 
                 n_samples=1000, episode_max_length=1000
        ):
        self.num_dims = env.observation_space.shape[0]
        self.num_states_in_each_dim = num_states_in_each_dim
        self.complexity = (num_states_in_each_dim+2) ** self.num_dims
        self.n_samples = n_samples
        
        try:
            filename = self.filename_from_params()
            with open(filename, 'rb') as f:
                state = pickle.load(f)
                self.min_max_bounds = state[0]
                self.interval_lengths = state[1]
                self.states_by_dim = state[2]
        except:
            self.min_max_bounds = [[float('inf'), float('-inf')] for i in range(self.num_dims)]        
            # Estimate bounds of each dimension through sampling
            self.learn_space_bounds(env, episode_max_length)
            self.finalize()
            self.save()
        return
    
    def filename_from_params(self):
        filename = "ss" 
        filename += str(self.num_states_in_each_dim)
        filename += "_"
        filename += str(self.n_samples)
        filename += ".pkl"
        return filename

    def save(self):
        filename = self.filename_from_params()
        with open(filename, 'wb') as f:
            state = [
                self.min_max_bounds, 
                self.interval_lengths,
                self.states_by_dim
                ]
            pickle.dump(state, f)
        return

    # update oberseved bounds of each dimensions
    def update_dim_bounds(self, observation):
        for i in range(self.num_dims):
            if (observation[i] < self.min_max_bounds[i][0]):
                self.min_max_bounds[i][0] = observation[i]
            if (observation[i] > self.min_max_bounds[i][1]):
                self.min_max_bounds[i][1] = observation[i]
        return

    def learn_space_bounds(self, env, episode_max_length):
        for episode in range(self.n_samples):
            observation, _ = env.reset()
            self.update_dim_bounds(observation)

            for t in range(episode_max_length):
                random_action = env.action_space.sample()
                observation, reward, terminate_episode_signal, _, step_info = env.step(random_action)
                self.update_dim_bounds(observation)
                
                if terminate_episode_signal:
                    break
            print((self.n_samples - episode,))
        return

    def map_observation_to_repr(self, observation):
        #Given an observation, return the state space representation
        # that it belongs to.
        assert (len(observation) == self.num_dims)
        repr = []

        for i in range(self.num_dims):
            for j in range(len(self.states_by_dim[i])):
                low, high = self.states_by_dim[i][j]
                if ( (observation[i] >= low) and (observation[i] < high) ):
                    repr.append(j)
                    break
        repr = tuple(repr)
        return repr

    def finalize(self):
        self.states_by_dim  = []

        # --UNUSED
        #The larger this number, the more elements will exist
        # within the partition of each state space component.
        # self.state_space_component_complexitity_multiplier = 4
        # self.ssccm = self.state_space_component_complexitity_multiplier
        # --END OF UNUSED

        self.interval_lengths = [self.min_max_bounds[i][1] - self.min_max_bounds[i][0] for i in range(self.num_dims)]

        for i in range(self.num_dims):
            # Given that this code was written for the cartpole env,
            # Every component of the cartpole observation space 
            # are all larger than 1.
            # Thus the number of elements in each partition will always al least
            # at least the ssccm. 
            low, high = self.min_max_bounds[i]

            states = [(-1 * float('inf'), low) ]
            
            interval_step_size = self.interval_lengths[i] / self.num_states_in_each_dim 

            for j in range(self.num_states_in_each_dim):
                interval_low = low + (j * interval_step_size)
                interval_high = low + ((j + 1) * interval_step_size)
                interval = (interval_low, interval_high)
                states.append(interval)

            #There is a difference between these values that
            #shouldn't exist
            states[0] = (states[0][0], states[1][0])

            #Should be true, but a guarantee was not proven  
            assert (len(states) > 0)
            
            # print((states[-1][1], high, states[-1][1] - high))

            states.append((states[-1][1], float('inf')))
            self.states_by_dim.append(states)
        
        return