
class Runner:
    def __init__(
            self,
            env,
            agent
    ):
        self.env = env
        self.agent = agent

        self.obs = None

    def train(self, run_context, num_steps):
        if not self.obs:
            self.obs = self.env.reset()

        for _ in range(num_steps):
            action = self.agent.get_action(run_context, self.obs[0])

            next_obs, reward, episode_end = self.env.step([action])

            self.agent.on_step(run_context, self.obs[0], action, reward[0], next_obs[0], episode_end)

            self.obs = next_obs
