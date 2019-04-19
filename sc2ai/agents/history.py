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

        self.step = 0

    def reset(self):
        self.step = 0

    def append(self, obs, actions, reward, next_obs, episode_end):
        assert self.step < self.trajectory_length * self.batch_size

        traj_step = self.step % self.trajectory_length
        batch_idx = (self.step // self.trajectory_length) % self.batch_size

        for name, o in obs.items():
            self.observations[name][traj_step, batch_idx] = o

        for name, a in actions.items():
            self.actions[name][traj_step, batch_idx] = a

        self.rewards[traj_step, batch_idx] = reward
        self.episode_ends[traj_step, batch_idx] = episode_end

        if traj_step + 1 == self.trajectory_length:
            for name, o in next_obs.items():
                self.last_observations[name][batch_idx] = o

        self.step += 1

    def batch_ready(self):
        assert self.step <= self.trajectory_length * self.batch_size
        return self.step == self.trajectory_length * self.batch_size
