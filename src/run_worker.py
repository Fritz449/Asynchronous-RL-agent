import tensorflow as tf
from run_agent import FLAGS
from utils import cluster_spec, dump_object, load_object
from network import Network
from environments import GymEnvironment, AtariGymEnvironment, RamGymEnvironment
import numpy as np
import scipy.signal
import time
from redis import Redis


def apply_adam_updates(variables_server, gradients, learning_rate, epsilon=1e-6):
    update_steps = increment_shared_variable(variables_server, 'update_steps')
    learning_rate = learning_rate * ((1 - FLAGS.beta_2 ** update_steps) ** 0.5) / (1 - FLAGS.beta_1 ** update_steps)
    for i, gradient in enumerate(gradients):
        momentum = load_object(variables_server.get('momentum_{}'.format(i)))
        momentum = FLAGS.beta_2 * momentum + (1 - FLAGS.beta_2) * gradient * gradient
        variables_server.set('momentum_{}'.format(i), dump_object(momentum))
        velocity = load_object(variables_server.get('velocity_{}'.format(i)))
        velocity = FLAGS.beta_1 * velocity + (1 - FLAGS.beta_1) * gradient
        variables_server.set('velocity_{}'.format(i), dump_object(velocity))
        weight = load_object(variables_server.get('weight_{}'.format(i)))
        new_weight = weight - velocity * learning_rate / ((momentum ** 0.5) + epsilon)
        variables_server.set('weight_{}'.format(i), dump_object(new_weight))
    return update_steps


def update_target_weights(variables_server, coef=FLAGS.softness_target_update):
    count_of_weights = load_object(variables_server.get('count_of_weights'))
    for i in range(count_of_weights):
        weight = load_object(variables_server.get('weight_{}'.format(i)))
        target_weight = load_object(variables_server.get('target_weight_{}'.format(i)))
        new_target_weight = (1 - coef) * target_weight + coef * weight
        variables_server.set('target_weight_{}'.format(i), dump_object(new_target_weight))


def add_to_xp(variables_server, state, action, reward, next_state, terminal):
    transition = [state, action, reward, next_state, terminal]
    variables_server.lpush("transitions", dump_object(transition))
    if variables_server.llen("transitions") >= FLAGS.buffer_max_size + 250 and FLAGS.thread == 0:
        variables_server.ltrim("transitions", 100, FLAGS.buffer_max_size + 99)


def increment_shared_variable(variables_server, name, increment=1):
    value = load_object(variables_server.get(name))
    new_value = value + increment
    variables_server.set(name, dump_object(new_value))
    return new_value


def get_xp_batch(variables_server):
    xp_size = min(FLAGS.buffer_max_size, variables_server.llen("transitions"))
    indexes_batch = np.random.randint(xp_size, size=FLAGS.batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = [], [], [], [], []
    for i in range(FLAGS.batch_size):
        transition = load_object(variables_server.lindex("transitions", indexes_batch[i]))
        state_batch.append(transition[0])
        action_batch.append(transition[1])
        reward_batch.append(transition[2])
        next_state_batch.append(transition[3])
        terminal_batch.append(transition[4])
    state_batch = np.array(state_batch)
    action_batch = np.array(action_batch)
    reward_batch = np.array(reward_batch)
    next_state_batch = np.array(next_state_batch)
    terminal_batch = np.array(terminal_batch)
    return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch


def save_shared_weights(variables_server):
    count_of_weights = load_object(variables_server.get('count_of_weights'))
    for i in range(count_of_weights):
        weight = load_object(variables_server.get('weight_{}'.format(i)))
        np.save('weight_{}'.format(i), weight)


def get_shared_weights(variables_server):
    count_of_weights = load_object(variables_server.get('count_of_weights'))
    weights = []
    for i in range(count_of_weights):
        weights.append(load_object(variables_server.get('weight_{}'.format(i))))
    return weights


def get_target_weights(variables_server):
    count_of_weights = load_object(variables_server.get('count_of_weights'))
    weights = []
    for i in range(count_of_weights):
        weights.append(load_object(variables_server.get('target_weight_{}'.format(i))))
    return weights


def save_agent(variables_server):
    if not os.path.exists(FLAGS.save_name):
        os.makedirs(FLAGS.save_name)
    target_weights = get_target_weights(variables_server)
    for i, weight in enumerate(target_weights):
        np.save(FLAGS.save_name + '/weight_{}'.format(i), weight)


import os

if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    spec = tf.train.ClusterSpec(cluster_spec(n_workers=FLAGS.n_threads, port=FLAGS.port))
    server = tf.train.Server(spec, job_name='worker', task_index=FLAGS.thread)
    sess = tf.InteractiveSession(server.target)
    time.sleep(3)

    if FLAGS.env_class == 'gym':
        env_class = GymEnvironment
    if FLAGS.env_class == 'atari-gym':
        env_class = AtariGymEnvironment
    if FLAGS.env_class == 'ram-atari-gym':
        env_class = RamGymEnvironment
    environment = env_class(FLAGS.env_name)
    test_env = env_class(FLAGS.env_name)
    state_dim = environment.get_state_dim()
    action_dim = environment.get_action_dim()

    network_parameters = (state_dim, action_dim, FLAGS.batch_size, FLAGS.critic_loss_coef, FLAGS.entropy_coef)

    with tf.device(tf.train.replica_device_setter(1, worker_device='job:worker/task:{}'.format(FLAGS.thread))):
        with tf.variable_scope("global"):
            network = Network('thread_{}'.format(FLAGS.thread), sess, *network_parameters, initialize=True)
    with tf.device(tf.train.replica_device_setter(1, worker_device='job:ps/task:0')):
        with tf.variable_scope("global"):
            target_network = Network('target_network', sess, *network_parameters, initialize=True)

    print('Networks of thread_{} initialized'.format(FLAGS.thread))
    variables_server = Redis(port=12000)

    weights = network.weights

    network.assign_weights(get_shared_weights(variables_server))
    target_network.assign_weights(get_target_weights(variables_server))
    total_reward = 0
    actions_made = np.zeros(action_dim)
    obs = environment.reset()
    while True:
        time_to_update = False
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []
        for _ in range(FLAGS.n_steps):
            # eps = max(0., min(1, 1 - (update_steps.eval() - FLAGS.epoch_time * 10) / (FLAGS.epoch_time * 50)))
            eps = 0
            action = network.get_action(np.copy(obs), eps)

            actions_made[action] += 1
            next_obs, reward, done = environment.step(action)
            states.append(np.copy(obs))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.copy(next_obs))
            terminals.append(done)
            add_to_xp(variables_server, obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            if done:
                break

        states_batch = np.array(states)
        actions_batch = np.array(actions)

        if done:
            rewards.append(0)
        else:
            rewards.append(network.compute_value(obs.reshape((1,) + obs.shape)))
        values_batch = scipy.signal.lfilter([1], [1, -FLAGS.gamma], rewards[::-1], axis=0)[::-1].astype('float32')[:-1]

        if FLAGS.a3c_learning_rate > 0:
            gradients, loss, td_error = network.compute_a2c_outputs(values_batch, states_batch, actions_batch)
            update_step = apply_adam_updates(variables_server, gradients, FLAGS.a3c_learning_rate)
            if update_step % FLAGS.epoch_time == 0:
                time_to_update = True
            network.assign_weights(get_shared_weights(variables_server))

        if FLAGS.dddqn_learning_rate > 0 and variables_server.llen("transitions") * 2 > FLAGS.buffer_max_size:
            network.assign_weights(get_shared_weights(variables_server))
            target_network.assign_weights(get_target_weights(variables_server))
            # Sample transitions
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = get_xp_batch(variables_server)
            output_target = target_network.compute_q_values(next_state_batch)
            q_argmax_online = np.argmax(network.compute_q_values(next_state_batch), axis=1)
            q_max_batch = output_target[np.arange(FLAGS.batch_size), q_argmax_online]

            target_q_batch = (reward_batch + (1 - terminal_batch) * FLAGS.gamma * q_max_batch)
            gradients, loss, td_error = network.compute_dqn_outputs(target_q_batch, state_batch, action_batch, None)
            update_step = apply_adam_updates(variables_server, gradients, FLAGS.dddqn_learning_rate)
            if update_step % FLAGS.epoch_time == 0:
                time_to_update = True
            network.assign_weights(get_shared_weights(variables_server))

        update_target_weights(variables_server)

        if time_to_update:
            target_network.assign_weights(get_target_weights(variables_server))
            index = 0
            total_test_reward = 0
            obs = test_env.reset()
            print(target_network.compute_q_values(obs.reshape((1,) + obs.shape)))
            for _ in range(1):
                done_test = False
                tot_r = 0
                while not done_test:
                    action = target_network.get_action(np.copy(obs), 0, True)
                    next_obs, reward, done_test = test_env.step(action)
                    obs = next_obs
                total_test_reward += test_env.get_total_reward()
                obs = test_env.reset()
            print("Mean reward of the test episodes:", total_test_reward / 1.,
                  np.sum(np.absolute(load_object(variables_server.get('target_weight_1')))))
            save_agent(variables_server)

        if done:
            total_reward = environment.get_total_reward()
            increment_shared_variable(variables_server, 'episodes')
            # print('Reward of an episode ', load_object(variables_server.get('episodes')), 'is ' + str(total_reward),
            #       actions_made,
            #       load_object(variables_server.get('update_steps')),
            #       )
            tot_r = 0
            actions_made = np.zeros(action_dim)
            obs = environment.reset()
