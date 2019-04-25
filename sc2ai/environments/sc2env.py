import sys
from collections import namedtuple, OrderedDict
import atexit
from absl import flags
import gin
import numpy as np
from pysc2.lib.features import parse_agent_interface_format, SCREEN_FEATURES, MINIMAP_FEATURES, Features, FeatureType
from pysc2.env.environment import StepType
from pysc2.lib.actions import FunctionCall, FUNCTIONS

EnvironmentSpec = namedtuple('EnvironmentSpec', ['action_spec', 'observation_spec'])
ObservationSpec = namedtuple('ObservationSpec', ['id', 'shape', 'is_spatial', 'features'])
FeatureSpec = namedtuple('FeatureSpec', ['index', 'scale', 'is_categorical'])
ActionSpec = namedtuple('ActionSpec', ['id', 'sizes', 'obs_space', 'args_mask'])


@gin.configurable
class SC2Environment:
    def __init__(self, screen_size=16, minimap_size=16, function_set='minigames', visualize=False):
        self._env = None
        self._aif = parse_agent_interface_format(feature_screen=screen_size, feature_minimap=minimap_size)
        self._visualize = visualize

        if function_set == 'all':
            self._func_ids = [f.id for f in FUNCTIONS]
        elif function_set == 'minigames':
            self._func_ids = [0, 1, 2, 3, 4, 6, 7, 12, 13, 42, 44, 50, 91, 183, 234, 309, 331, 332, 333, 334, 451, 452, 490]
        else:
            raise ValueError

        sc2_features = Features(agent_interface_format=self._aif)
        sc2_action_spec = sc2_features.action_spec()
        sc2_obs_spec = sc2_features.observation_spec()

        fn_args_mask = np.zeros((len(self._func_ids), len(sc2_action_spec.types) + 1), dtype=np.bool)
        fn_args_mask[:, 0] = 1
        for i, func_id in enumerate(self._func_ids):
            used_args = [a.id + 1 for a in FUNCTIONS[func_id].args]
            fn_args_mask[i, used_args] = 1
        action_spec = [('function_id', ActionSpec(0, (len(self._func_ids),), None, fn_args_mask))]
        for t in sc2_action_spec.types:
            if t.name == 'screen' or t.name == 'screen2':
                space = 'screen'
            elif t.name == 'minimap':
                space = 'minimap'
            else:
                space = None

            action_spec.append((t.name, ActionSpec(len(action_spec), t.sizes, space, None)))
        action_spec = OrderedDict(action_spec)

        def feature_spec(features):
            return [FeatureSpec(f.index, f.scale, f.type == FeatureType.CATEGORICAL) for f in features]

        obs_spec = OrderedDict([
            ('screen', ObservationSpec(0, sc2_obs_spec['feature_screen'], True, feature_spec(SCREEN_FEATURES))),
            ('minimap', ObservationSpec(1, sc2_obs_spec['feature_minimap'], True, feature_spec(MINIMAP_FEATURES))),
            ('available_actions', ObservationSpec(2, (len(self._func_ids),), False, None)),
            ('player', ObservationSpec(3, sc2_obs_spec['player'], False, None))
        ])

        self.spec = EnvironmentSpec(action_spec, obs_spec)

    def start(self):
        from pysc2.env.sc2_env import SC2Env, Agent, Race

        if not flags.FLAGS.is_parsed():
            flags.FLAGS(sys.argv)

        self._env = SC2Env(map_name='MoveToBeacon', agent_interface_format=self._aif, players=[
            Agent(Race.protoss)
        ], visualize=self._visualize)

        atexit.register(self._env.close)

    def stop(self):
        if self._env:
            self._env.close()
            atexit.unregister(self._env.close)

    def reset(self):
        obs, rewards, done = self._wrap_obs(self._env.reset())
        return obs

    def step(self, actions):
        sc2_actions = [self._actions_to_sc2(a) for a in actions]

        obs = self._env.step(sc2_actions)

        obs, rewards, done = self._wrap_obs(obs)
        if done:
            obs = self.reset()

        return obs, rewards, done

    def _wrap_obs(self, obs):
        def wrap(o):
            available_actions = np.zeros(self.spec.observation_spec['available_actions'].shape, dtype=np.int32)
            func_ids = [i for i, func_id in enumerate(self._func_ids) if func_id in o.observation['available_actions']] # TODO: this is too slow when using all function ids
            available_actions[func_ids] = 1

            return {
                'screen': np.asarray(o.observation['feature_screen']),
                'minimap': np.asarray(o.observation['feature_minimap']),
                'available_actions': available_actions,
                'player': np.asarray(o.observation['player'])
            }

        return [wrap(o) for o in obs], [o.reward for o in obs], obs[0].step_type == StepType.LAST

    def _actions_to_sc2(self, actions):
        def convert_arg(value, spec):
            if len(spec.sizes) == 2:
                value = np.unravel_index(value, spec.sizes)
                value = np.flip(value)
                return list(value)
            else:
                return [value]
        function = self._func_ids[actions['function_id']]
        args = [
            convert_arg(actions[arg.name].item(), self.spec.action_spec[arg.name])
            for arg in FUNCTIONS[function].args
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
