from collections import namedtuple
from absl import flags
import gin
import numpy as np
from pysc2.lib.features import parse_agent_interface_format, SCREEN_FEATURES, MINIMAP_FEATURES, Features, FeatureType
from pysc2.env.environment import StepType
from pysc2.env.sc2_env import SC2Env, Agent, Bot, Race, Difficulty
from pysc2.lib.actions import FunctionCall

FLAGS = flags.FLAGS

flags.DEFINE_boolean('visualize', False, '')

EnvironmentSpec = namedtuple('EnvironmentSpec', ['action_spec', 'spaces'])
SpaceDesc = namedtuple('SpaceDesc', ['name', 'index', 'shape', 'features'])


@gin.configurable()
class SC2Environment:
    def __init__(self, screen_size=16, minimap_size=16):
        self._env = None
        self._aif = parse_agent_interface_format(feature_screen=screen_size, feature_minimap=minimap_size)

        features = Features(agent_interface_format=self._aif)
        obs_spec = features.observation_spec()

        self.spec = EnvironmentSpec(features.action_spec(), [
            SpaceDesc('screen', 0, obs_spec['feature_screen'], SCREEN_FEATURES),
            SpaceDesc('minimap', 1, obs_spec['feature_minimap'], MINIMAP_FEATURES)
        ])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self._env = SC2Env(map_name='MoveToBeacon', agent_interface_format=self._aif, players=[
            Agent(Race.protoss)
        ], visualize=FLAGS.visualize)

    def stop(self):
        if self._env:
            self._env.close()

    def reset(self):
        return self._env.reset()

    def step(self, actions):
        return self._env.step([self._actions_to_sc2(a) for a in actions])

    def _actions_to_sc2(self, actions):
        function = actions[0].item()
        args = [
            list(np.unravel_index(actions[1][arg.name].item(), arg.sizes))
            for arg in self.spec.action_spec.functions[function].args
        ]

        return FunctionCall(function, args)

    def _palettes(self):
        feat_palettes = [[None] * len(s.features) for s in env_spec.spaces]
        for s in env_spec.spaces:
            for f in s.features:
                palette = f.palette
                if len(palette) < f.scale:
                    palette = np.append(f.palette, [[255, 0, 255] * (f.scale - len(f.palette))], axis=0)
                feat_palettes[s.index][f.index] = tf.constant(palette, dtype=tf.uint8,
                                                              name='{}_{}_palette'.format(s.name, f.name))
