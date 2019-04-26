from multiprocessing import Process, Pipe
import numpy as np


def _worker(env_fn, env_config, pipe):
    env = env_fn(env_config)
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'step':
                result = env.step(data)
                pipe.send(result)
            elif cmd == 'reset':
                result = env.reset()
                pipe.send(result)
            elif cmd == 'get_spec':
                pipe.send(env.spec)
            elif cmd == 'close':
                env.close()
                break
            else:
                raise NotImplementedError
    finally:
        env.close()


class VecEnv:
    def __init__(self, env_fn, env_config, num_envs=4):
        self._pipes, self._worker_pipes = zip(*[Pipe() for _ in range(num_envs)])
        self._procs = [Process(target=_worker, args=(env_fn, env_config, p)) for p in self._worker_pipes]
        for p in self._procs:
            p.daemon = True
            p.start()

        self._pipes[0].send(('get_spec', None))
        self.spec = self._pipes[0].recv()

    def close(self):
        for p in self._pipes:
            p.send(('close', None))
        for proc in self._procs:
            proc.join()

    def reset(self):
        for p in self._pipes:
            p.send(('reset', None))
        results = self._receive()
        return _reshape_obs(results)

    def step(self, actions):
        actions = [[{k: v[i] for k, v in a.items()} for a in actions] for i in range(len(self._pipes))]
        for p, a in zip(self._pipes, actions):
            p.send(('step', a))
        results = self._receive()
        obs, rewards, dones = zip(*results)
        return _reshape_obs(obs), list(zip(*rewards)), dones

    def _receive(self):
        return [p.recv() for p in self._pipes]


def _reshape_obs(obs):
    num_obs = len(obs[0])
    keys = obs[0][0].keys()
    out = []
    for i in range(num_obs):
        out.append({k: np.stack([o[i][k] for o in obs]) for k in keys})

    return out
