import gymnasium as gym
from gymnasium.utils.play import play

play(gym.make('CartPole-v1', render_mode='rgb_array'), keys_to_action={
    'a': 0,
    'd': 1
})