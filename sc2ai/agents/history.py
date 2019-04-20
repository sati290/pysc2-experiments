import numpy as np


class History:
    def __init__(self, trajectory_length, batch_size, env_spec):
        self.trajectory_length = trajectory_length
        self.batch_size = batch_size

        shape = (trajectory_length, batch_size)

        self.observations = {name: np.zeros(shape + spec.shape, dtype=np.float32)
                             for name, spec in env_spec.observation_spec.items()}
        self.last_observations = {name: np.zeros((batch_size,) + spec.shape, dtype=np.float32)
                                  for name, spec in env_spec.observation_spec.items()}
        self.actions = {name: np.zeros(shape, dtype=np.int32) for name in env_spec.action_spec}
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.episode_ends = np.zeros(shape, dtype=np.bool)

        self.traj_step = 0
        self.batch_idx = 0

    def reset(self):
        self.traj_step = 0
        self.batch_idx = 0

    def append(self, obs, actions, rewards, next_obs, episode_ends):
        assert self.traj_step < self.trajectory_length
        assert self.batch_idx < self.batch_size

        num_envs = len(rewards)
        assert self.batch_size % num_envs == 0

        batch_start = self.batch_idx
        batch_end = batch_start + num_envs

        for name, o in obs.items():
            self.observations[name][self.traj_step, batch_start:batch_end] = o

        for name, a in actions.items():
            self.actions[name][self.traj_step, batch_start:batch_end] = a

        self.rewards[self.traj_step, batch_start:batch_end] = rewards
        self.episode_ends[self.traj_step, batch_start:batch_end] = episode_ends

        if self.traj_step + 1 == self.trajectory_length:
            for name, o in next_obs.items():
                self.last_observations[name][batch_start:batch_end] = o

            self.traj_step = 0
            self.batch_idx += num_envs
        else:
            self.traj_step += 1

    def batch_ready(self):
        assert self.batch_idx <= self.batch_size
        return self.batch_idx == self.batch_size
