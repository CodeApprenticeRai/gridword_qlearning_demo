from q_trainer import q_trainer
import time


if __name__ == "__main__":
    start_time = time.time()
    env_label=0
    n_states_in_each_dim=30
    PARAM_number_of_episodes = 10
    weigh_reward_with_time = True
    state_data_filename=str(env_label) + "q" + str(n_states_in_each_dim) 
    state_data_filename += "_" + str(int(weigh_reward_with_time)) + ".pkl"
    trainer = q_trainer(
        env_label=env_label,
        state_data_filename=state_data_filename,
        n_states_in_each_dim=n_states_in_each_dim,
        weigh_reward_with_time=weigh_reward_with_time,
        PARAM_number_of_episodes=PARAM_number_of_episodes,
        PARAM_learning_rate=0.99,
        PARAM_discount_factor=0.9,
        PARAM_epsilon=0.5,
        show_windows=True
    )
    trainer.run_session()
    run_time = time.time() - start_time
    print(f'runtime: {run_time}, {run_time/PARAM_number_of_episodes}')