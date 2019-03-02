from absl import app
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Reshape, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from pysc2.lib import actions
from pysc2.lib.features import parse_agent_interface_format, SCREEN_FEATURES, MINIMAP_FEATURES, FeatureType
from pysc2.env.environment import StepType
from pysc2.env.sc2_env import SC2Env, Agent, Bot, Race, Difficulty


def build_model(obs_spec):
    use_conv = False

    screen_shape = obs_spec['feature_screen']
    minimap_shape = obs_spec['feature_minimap']

    screen_input = Input(shape=screen_shape, name='screen_input')
    minimap_input = Input(shape=minimap_shape, name='minimap_input')

    if use_conv:
        screen_conv = Conv2D(8, (1, 1), data_format='channels_first', name='screen_conv')(screen_input)
        minimap_conv = Conv2D(4, (1, 1), data_format='channels_first', name='minimap_conve')(minimap_input)

        concat_inputs = Concatenate()([Flatten()(screen_conv), Flatten()(minimap_conv)])
    else:
        concat_inputs = Concatenate()([Flatten()(screen_input), Flatten()(minimap_input)])

    dense = Dense(512, activation='relu')(concat_inputs)

    if use_conv:
        screen_output = Dense(np.prod(screen_conv.shape[1:4]), activation='linear')(dense)
        screen_output = Reshape(screen_conv.shape[1:4], name='screen_output')(screen_output)
        screen_output = Conv2D(screen_shape[0], (1, 1), data_format='channels_first')(screen_output)

        minimap_output = Dense(np.prod(minimap_conv.shape[1:4]), activation='linear')(dense)
        minimap_output = Reshape(minimap_conv.shape[1:4], name='minimap_output')(minimap_output)
        minimap_output = Conv2D(minimap_shape[0], (1, 1), data_format='channels_first')(minimap_output)
    else:
        screen_output = Dense(np.prod(screen_shape), activation='linear')(dense)
        screen_output = Reshape(screen_shape, name='screen_output')(screen_output)

        minimap_output = Dense(np.prod(minimap_shape), activation='linear')(dense)
        minimap_output = Reshape(minimap_shape, name='minimap_output')(minimap_output)

    return Model(
        inputs=[screen_input, minimap_input],
        outputs=[screen_output, minimap_output]
    )


def normalize_features(input, feature_spec):
    output = np.ndarray(input.shape)
    for f in feature_spec:
        output[f.index] = input[f.index] / f.scale

    return output


def main(args):
    log_dir = 'logs/' + time.strftime('%Y%m%d-%H%M%S', time.localtime())

    sess = K.get_session()

    env = SC2Env(map_name='Simple64', agent_interface_format=[
        parse_agent_interface_format(feature_screen=16, feature_minimap=16)
    ], players=[
        Agent(Race.protoss),
        Bot(Race.protoss, Difficulty.easy)
    ], visualize=True)

    try:
        action_spec = env.action_spec()[0]
        obs_spec = env.observation_spec()[0]

        model = build_model(obs_spec)
        model.summary()

        opt = Adam(lr=0.0001)
        model.compile(optimizer=opt, loss='mean_squared_error')

        summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)

        global_step = 0
        while True:
            obs = env.reset()

            for j in range(1000):
                global_step = global_step + 1

                if obs[0].step_type == StepType.LAST:
                    break

                action = np.random.choice(obs[0].observation.available_actions)
                args = [[np.random.randint(0, size) for size in arg.sizes] for arg in
                        action_spec.functions[action].args]

                prev_obs = obs
                obs = env.step([actions.FunctionCall(action, args)])

                normed_screen = normalize_features(obs[0].observation['feature_screen'], SCREEN_FEATURES)
                normed_minimap = normalize_features(obs[0].observation['feature_minimap'], MINIMAP_FEATURES)

                ret = model.train_on_batch(
                    [[normed_screen], [normed_minimap]],
                    [[normed_screen], [normed_minimap]]
                )

                print(ret)

                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = ret[0]
                summary_value.tag = 'loss'
                summary_writer.add_summary(summary, global_step=global_step)

                #pred = model.predict_on_batch([[obs[0].observation['feature_screen']], [obs[0].observation['feature_minimap']]])

    finally:
        env.close()


if __name__ == '__main__':
    app.run(main)
