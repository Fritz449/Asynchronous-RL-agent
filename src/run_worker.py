import tensorflow as tf
from run_agent import FLAGS
from utils import cluster_spec
from network import Network
from environments import GymEnvironment, AtariGymEnvironment
import numpy as np
import scipy.signal
import time


def get_adam_updates(shared_variables, update_steps, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-6):
    learning_rate = learning_rate * ((1 - beta2 ** update_steps) ** 0.5) / (1 - beta1 ** update_steps)
    decrements = []
    shared_weights, shared_momentum, shared_velocity = shared_variables
    for momentum, velocity in zip(shared_momentum, shared_velocity):
        decrements.append(velocity.eval() * learning_rate / ((momentum.eval() ** 0.5) + epsilon))
    return decrements


import os

if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    spec = tf.train.ClusterSpec(cluster_spec(n_workers=FLAGS.n_threads))
    server = tf.train.Server(spec, job_name='worker', task_index=FLAGS.thread)
    sess = tf.InteractiveSession(server.target)
    tf.logging.set_verbosity(tf.logging.ERROR)
    time.sleep(3)

    if FLAGS.env_class == 'gym':
        env_class = GymEnvironment
    if FLAGS.env_class == 'atari-gym':
        env_class = AtariGymEnvironment
    environment = env_class(FLAGS.env_name)
    obs = environment.reset()
    test_env = env_class(FLAGS.env_name)
    state_dim = environment.get_state_dim()
    action_dim = environment.get_action_dim()
    network_parameters = (state_dim, action_dim, FLAGS.batch_size, FLAGS.critic_loss_coef, FLAGS.entropy_coef)
    with tf.device('job:ps/task:0'):
        with tf.variable_scope('global'):
            xp_replay = tf.RandomShuffleQueue(FLAGS.buffer_max_size, 0,
                                              [tf.float32, tf.float32, tf.int32, tf.float32, tf.int32],
                                              shapes=[state_dim, state_dim, (), (), ()],
                                              names=['state', 'next_state', 'action', 'reward', 'terminal'],
                                              shared_name='xp_replay')
            new_state = tf.placeholder(tf.float32, state_dim)
            new_next_state = tf.placeholder(tf.float32, state_dim)
            new_action = tf.placeholder(tf.int32)
            new_reward = tf.placeholder(tf.float32)
            new_terminal = tf.placeholder(tf.int32)
            new_states = tf.placeholder(tf.float32, (None,) + state_dim)
            new_next_states = tf.placeholder(tf.float32, (None,) + state_dim)
            new_actions = tf.placeholder(tf.int32, (None,))
            new_rewards = tf.placeholder(tf.float32, (None,))
            new_terminals = tf.placeholder(tf.int32, (None,))
            add_to_xp_op = xp_replay.enqueue(
                {'state': new_state,
                 'next_state': new_next_state,
                 'action': new_action,
                 'reward': new_reward,
                 'terminal': new_terminal})
            return_to_xp_op = xp_replay.enqueue_many(
                {'state': new_states,
                 'next_state': new_next_states,
                 'action': new_actions,
                 'reward': new_rewards,
                 'terminal': new_terminals})
            get_batch_op = xp_replay.dequeue_many(FLAGS.batch_size)
            delete_tr_op = xp_replay.dequeue()
            queue_indexes = tf.FIFOQueue(FLAGS.buffer_max_size, tf.int32, shapes=(), shared_name='queue_ix')
    worker_device = "/job:worker/task:{}".format(FLAGS.thread)
    with tf.device(tf.train.replica_device_setter(1, worker_device='job:worker/task:{}'.format(FLAGS.thread))):
        with tf.variable_scope("global"):
            network = Network('thread_{}'.format(FLAGS.thread), sess, *network_parameters, initialize=True)
            target_network = Network('thread_{}_target'.format(FLAGS.thread), sess, *network_parameters,
                                     initialize=True)

    print('Networks of thread_{} initialized'.format(FLAGS.thread))
    with tf.device(tf.train.replica_device_setter(1, worker_device='job:worker/task:{}'.format(FLAGS.thread))):
        with tf.variable_scope('global'):
            weights = network.weights
            shared_weights = []
            shared_targets = []
            shared_momentum = []
            shared_velocity = []
            update_momentum_ops = []
            update_velocity_ops = []
            update_weight_ops = []
            gradient_phs = []
            steps_phs = []
            scalar_ph = tf.placeholder(tf.float32, shape=())
            for i, weight in enumerate(weights):
                w_shape = weight.get_shape()
                shared_weights.append(tf.get_variable('weight_{}'.format(i), w_shape, tf.float32))
                shared_targets.append(tf.get_variable('target_{}'.format(i), w_shape, tf.float32))
                # Shared momentum and velocity for Adam optimizer
                shared_momentum.append(tf.get_variable('momentum_{}'.format(i), w_shape, tf.float32))
                shared_velocity.append(tf.get_variable('velocity_{}'.format(i), w_shape, tf.float32))
                gradient_phs.append(tf.placeholder(tf.float32, shape=w_shape))
                update_velocity_ops.append(shared_velocity[i].assign(
                    shared_velocity[i] * FLAGS.beta_1 + (1 - FLAGS.beta_1) * gradient_phs[i]))
                update_momentum_ops.append(shared_momentum[i].assign(
                    shared_momentum[i] * FLAGS.beta_2 + (1 - FLAGS.beta_2) * gradient_phs[i] * gradient_phs[i]))
                steps_phs.append(tf.placeholder(tf.float32, shape=w_shape))
                update_weight_ops.append(shared_weights[i].assign(shared_weights[i] - steps_phs[i]))
            update_target_op = [sh_t.assign(0.001 * sh_w + 0.999 * sh_t) for sh_w, sh_t in
                                zip(shared_weights, shared_targets)]
            deq_cursor = queue_indexes.dequeue()
            scalar_int_ph = tf.placeholder(tf.int32, shape=())
            fill_queue = queue_indexes.enqueue_many(tf.range(0, FLAGS.buffer_max_size))
            update_steps = tf.get_variable('update_steps', (), tf.int32, tf.zeros_initializer())
            episodes = tf.get_variable('episodes', (), tf.int32, tf.zeros_initializer())
            size = tf.get_variable('size_of_xp', (), tf.int32, tf.zeros_initializer())
            cursor = tf.get_variable('cursor_of_xp', (), tf.int32, tf.zeros_initializer())

    buffer_max_size = tf.constant(FLAGS.buffer_max_size)
    increment_size = size.assign(tf.minimum(size + 1, buffer_max_size), use_locking=True)
    increment_cursor = cursor.assign(tf.mod(cursor + 1, buffer_max_size), use_locking=True)
    increment_update_steps = [update_steps.assign(update_steps + 1, use_locking=True)]
    increment_episode_steps = episodes.assign(episodes + 1, use_locking=True)

    shared_variables = (shared_weights, shared_momentum, shared_velocity)
    index_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size,))
    prob_ph = tf.placeholder(tf.float32, (FLAGS.batch_size,))
    network.assign_weights(shared_weights)
    target_network.assign_weights(shared_targets)
    tot_r = 0
    actions_made = np.zeros(action_dim)
    while update_steps.eval() < 1e9:
        time_to_update = False

        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []
        for _ in range(FLAGS.n_steps):
            eps = max(0., min(1, 1 - (update_steps.eval() - FLAGS.epoch_time * 10) / (FLAGS.epoch_time * 50)))
            # eps = 0
            action = network.get_action(np.copy(obs), eps)

            actions_made[action] += 1
            next_obs, reward, done = environment.step(action)
            sess.run([increment_size])
            cursor_value = sess.run(deq_cursor)
            if cursor_value == FLAGS.buffer_max_size - 1:
                sess.run(fill_queue)
            if xp_replay.size().eval() >= FLAGS.buffer_max_size - 500 and FLAGS.dddqn_learning_rate > 0:
                sess.run(delete_tr_op)
            if cursor_value % (FLAGS.n_steps * FLAGS.epoch_time) == 0 and size.eval() * 2 > FLAGS.buffer_max_size:
                time_to_update = True

            states.append(np.copy(obs))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.copy(next_obs))
            terminals.append(done)
            obs = next_obs
            tot_r += reward
            if done:
                break

        states_batch = np.array(states)
        next_states_batch = np.array(next_states)
        actions_batch = np.array(actions)
        rewards_batch = np.array(rewards)
        terminals_batch = np.array(terminals)
        if FLAGS.dddqn_learning_rate > 0:
            sess.run(return_to_xp_op,
                     feed_dict={new_states: states_batch, new_next_states: next_states_batch,
                                new_actions: actions_batch, new_rewards: rewards_batch, new_terminals: terminals_batch})
        if done:
            rewards.append(0)
        else:
            rewards.append(network.compute_value(obs.reshape((1,) + obs.shape)))
        values_batch = scipy.signal.lfilter([1], [1, -FLAGS.gamma], rewards[::-1], axis=0)[::-1].astype('float32')[:-1]

        if FLAGS.a3c_learning_rate > 0:
            gradients, loss, td_error = network.compute_a2c_outputs(values_batch, states_batch, actions_batch)
            sess.run(update_momentum_ops + update_velocity_ops, feed_dict=dict(zip(gradient_phs, gradients)))
            sess.run(increment_update_steps)
            steps = get_adam_updates(shared_variables, update_steps.eval(), FLAGS.a3c_learning_rate)
            sess.run(update_weight_ops, feed_dict=dict(zip(steps_phs, steps)))
            network.assign_weights(shared_weights)

        if FLAGS.dddqn_learning_rate > 0 and size.eval() * 2 > FLAGS.buffer_max_size:
            network.assign_weights(shared_weights)
            target_network.assign_weights(shared_targets)
            # Sample transitions

            batch = sess.run(get_batch_op)
            state_batch, next_state_batch, action_batch, reward_batch, done_batch = batch['state'], batch['next_state'], \
                                                                                    batch['action'], batch['reward'], \
                                                                                    batch['terminal']
            sess.run(return_to_xp_op,
                     feed_dict={new_states: state_batch, new_next_states: next_state_batch,
                                new_actions: action_batch, new_rewards: reward_batch, new_terminals: done_batch})
            output_target = target_network.compute_q_values(next_state_batch)
            q_argmax_online = np.argmax(network.compute_q_values(next_state_batch), axis=1)
            q_max_batch = output_target[np.arange(FLAGS.batch_size), q_argmax_online]
            # q_max_batch = np.max(output_target, axis=1)

            target_q_batch = (reward_batch + (1 - done_batch) * FLAGS.gamma * q_max_batch)
            # print (target_q_batch)
            gradients, loss, td_error = network.compute_dqn_outputs(target_q_batch, state_batch, action_batch, None)
            sess.run(update_momentum_ops + update_velocity_ops, feed_dict=dict(zip(gradient_phs, gradients)))
            sess.run(increment_update_steps)
            steps = get_adam_updates(shared_variables, update_steps.eval(), FLAGS.dddqn_learning_rate)
            sess.run(update_weight_ops, feed_dict=dict(zip(steps_phs, steps)))
            network.assign_weights(shared_weights)
        sess.run(update_target_op)
        target_network.assign_weights(shared_targets)
        if time_to_update:
            index = 0
            for weight in shared_weights:
                np.save('weight_{}'.format(index), weight.eval())
                index += 1
            total_reward = 0
            for _ in range(1):
                obs = test_env.reset()
                done_test = False
                tot_r = 0
                while not done_test:
                    action = target_network.get_action(np.copy(obs), 0, True)
                    next_obs, reward, done_test = test_env.step(action)
                    obs = next_obs

                    tot_r += reward
                total_reward += test_env.get_total_reward()
            print("Mean reward of the test episodes:", total_reward / 1.,
                  np.sum(np.absolute(shared_weights[-2].eval())), np.sum(np.absolute(shared_weights[-1].eval())),
                  max(0.1, min(1, 1 - (update_steps.eval() - FLAGS.epoch_time * 10) / (FLAGS.epoch_time * 50))))

        if done:
            total_reward = environment.get_total_reward()
            sess.run(increment_episode_steps)
            print('Reward of an episode ', episodes.eval(), 'is ' + str(total_reward), xp_replay.size().eval(),
                  np.sum(np.absolute(shared_weights[-2].eval())), np.max(np.absolute(shared_weights[-1].eval())),
                  actions_made,
                  max(0.1, min(1, 1 - (update_steps.eval() - FLAGS.epoch_time * 10) / (FLAGS.epoch_time * 50))),
                  update_steps.eval()
                  )
            tot_r = 0
            actions_made = np.zeros(action_dim)
            obs = environment.reset()
