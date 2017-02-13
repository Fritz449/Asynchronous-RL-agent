import tensorflow as tf
from environments import GymEnvironment, AtariGymEnvironment
from utils import launch_workers, cluster_spec
from network import Network
import numpy as np

flags = tf.flags
flags.DEFINE_integer('thread', 0, 'number of thread')
flags.DEFINE_string('env_class', 'gym', 'environment type')
flags.DEFINE_string('env_name', 'CartPole-v0', 'gym environment name')
flags.DEFINE_integer('n_threads', 6, 'number of workers')
flags.DEFINE_integer('port', 12332, 'starting port')
flags.DEFINE_integer('n_steps', 20, 'agent parameter')
flags.DEFINE_float('gamma', 0.99, 'agent parameter')
flags.DEFINE_float('dddqn_learning_rate', 0.0, 'network parameter')
flags.DEFINE_float('a3c_learning_rate', 0.01, 'network parameter')
flags.DEFINE_integer('batch_size', 32, 'network parameter')
flags.DEFINE_integer('buffer_max_size', 30000, 'agent parameter')
flags.DEFINE_integer('epoch_time', 100, 'agent parameter')
flags.DEFINE_float('critic_loss_coef', 0.5, 'agent parameter')
flags.DEFINE_float('entropy_coef', 0.001, 'agent parameter')
flags.DEFINE_float('beta_1', 0.9, 'optimizer parameter')
flags.DEFINE_float('beta_2', 0.999, 'optimizer parameter')
FLAGS = flags.FLAGS
import os

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if FLAGS.env_class == 'gym':
        env_class = GymEnvironment
    if FLAGS.env_class == 'atari-gym':
        env_class = AtariGymEnvironment
    test_environment = env_class(FLAGS.env_name)
    state_dim = test_environment.get_state_dim()
    action_dim = test_environment.get_action_dim()

    print('Starting parameter server...')
    spec = tf.train.ClusterSpec(cluster_spec(n_workers=FLAGS.n_threads))
    server = tf.train.Server(spec, job_name='ps', task_index=0)
    sess = tf.InteractiveSession(server.target)
    tf.logging.set_verbosity(tf.logging.ERROR)
    network_parameters = (
        state_dim, action_dim, FLAGS.batch_size, FLAGS.critic_loss_coef, FLAGS.entropy_coef)
    worker_device = 'job:ps/task:0'
    with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
        with tf.variable_scope('global'):
            network = Network('network', sess, *network_parameters, initialize=False)
    with tf.device('job:ps/task:0'):
        with tf.variable_scope('global'):
            xp_replay = tf.RandomShuffleQueue(FLAGS.buffer_max_size, 0,
                                              [tf.float32, tf.float32, tf.int32, tf.float32, tf.int32],
                                              shapes=[state_dim, state_dim, (), (), ()],
                                              names=['state', 'next_state', 'action', 'reward', 'terminal'],
                                              shared_name='xp_replay')
            queue_indexes = tf.FIFOQueue(FLAGS.buffer_max_size, tf.int32, shapes=(), shared_name='queue_ix')
    with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
        with tf.variable_scope('global'):
            print('Global variables initializing...')
            weights = network.weights
            shared_weights = []
            shared_target = []
            for i, weight in enumerate(weights):
                w_shape = weight.get_shape()
                # Shared weight
                shared_w = tf.get_variable('weight_{}'.format(i), w_shape, tf.float32, tf.zeros_initializer())
                shared_t = tf.get_variable('target_{}'.format(i), w_shape, tf.float32, tf.zeros_initializer())
                shared_weights.append(shared_w)
                shared_target.append(shared_t)
                # Shared momentum and velocity for Adam optimizer
                shared_m = tf.get_variable('momentum_{}'.format(i), w_shape, tf.float32, tf.zeros_initializer())
                shared_v = tf.get_variable('velocity_{}'.format(i), w_shape, tf.float32, tf.zeros_initializer())
            print('Global experience replay initializing...')
            xp_shape = (FLAGS.buffer_max_size,) + state_dim
            update_steps = tf.get_variable('update_steps', (), tf.int32, tf.zeros_initializer())
            episodes = tf.get_variable('episodes', (), tf.int32, tf.zeros_initializer())
            size = tf.get_variable('size_of_xp', (), tf.int32, tf.zeros_initializer())
            cursor = tf.get_variable('cursor_of_xp', (), tf.int32, tf.zeros_initializer())

    print('Starting asynchronous training...')
    processes = launch_workers(n_workers=FLAGS.n_threads)
    sess.run(tf.global_variables_initializer())
    sess.run(queue_indexes.enqueue_many(tf.range(0, FLAGS.buffer_max_size)))
    print('initialized')
    with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
        with tf.variable_scope('global'):
            print('Global weights initializing...')
            weights = network.weights
            for i, weight in enumerate(weights):
                sess.run(shared_weights[i].assign(weight.eval()))
                sess.run(shared_target[i].assign(weight.eval()))
            print('Global weights initialized.')
    server.join()
    print("the end")
