from collections import namedtuple, OrderedDict
import atexit
import gin
import numpy as np
from pysc2.lib.features import parse_agent_interface_format, SCREEN_FEATURES, MINIMAP_FEATURES, Features, FeatureType
from pysc2.env.environment import StepType
from pysc2.lib.actions import FunctionCall, FUNCTIONS

EnvironmentSpec = namedtuple('EnvironmentSpec', ['action_spec', 'observation_spec'])
ObservationSpec = namedtuple('ObservationSpec', ['id', 'shape', 'is_spatial', 'features'])
ActionSpec = namedtuple('ActionSpec', ['id', 'sizes', 'obs_space', 'args_mask'])


@gin.configurable()
class SC2Environment:
    def __init__(self, screen_size=16, minimap_size=16, visualize=False):
        self._env = None
        self._aif = parse_agent_interface_format(feature_screen=screen_size, feature_minimap=minimap_size)
        self._visualize = visualize

        sc2_features = Features(agent_interface_format=self._aif)
        sc2_action_spec = sc2_features.action_spec()
        sc2_obs_spec = sc2_features.observation_spec()

        fn_args_mask = np.zeros((len(sc2_action_spec.functions), len(sc2_action_spec.types) + 1), dtype=np.bool)
        fn_args_mask[:, 0] = 1
        for f in sc2_action_spec.functions:
            used_args = [a.id + 1 for a in f.args]
            fn_args_mask[f.id, used_args] = 1
        action_spec = [('function_id', ActionSpec(0, (len(sc2_action_spec.functions),), None, fn_args_mask))]
        for t in sc2_action_spec.types:
            if t.name == 'screen' or t.name == 'screen2':
                space = 'screen'
            elif t.name == 'minimap':
                space = 'minimap'
            else:
                space = None

            action_spec.append((t.name, ActionSpec(len(action_spec), t.sizes, space, None)))
        action_spec = OrderedDict(action_spec)

        obs_spec = OrderedDict([
            ('screen', ObservationSpec(0, sc2_obs_spec['feature_screen'], True, SCREEN_FEATURES)),
            ('minimap', ObservationSpec(1, sc2_obs_spec['feature_minimap'], True, MINIMAP_FEATURES)),
            ('available_actions', ObservationSpec(2, (len(sc2_action_spec.functions),), False, None)),
            ('player', ObservationSpec(3, sc2_obs_spec['player'], False, None))
        ])

        self.spec = EnvironmentSpec(action_spec, obs_spec)

    def start(self):
        from pysc2.env.sc2_env import SC2Env, Agent, Race

        self._env = SC2Env(map_name='MoveToBeacon', agent_interface_format=self._aif, players=[
            Agent(Race.protoss)
        ], visualize=self._visualize)

        atexit.register(self._env.close)

    def stop(self):
        if self._env:
            self._env.close()
            atexit.unregister(self._env.close)

    def reset(self):
        return self._wrap_obs(self._env.reset())

    def step(self, actions):
        sc2_actions = [self._actions_to_sc2(a) for a in actions]
        obs = self._env.step(sc2_actions)
        return self._wrap_obs(obs)

    def _wrap_obs(self, obs):
        def wrap(o):
            available_actions = np.zeros(self.spec.observation_spec['available_actions'].shape, dtype=np.int32)
            available_actions[o.observation['available_actions']] = 1

            return {
                'screen': o.observation['feature_screen'],
                'minimap': o.observation['feature_minimap'],
                'available_actions': available_actions,
                'player': o.observation['player']
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
        function = actions['function_id'].item()
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
