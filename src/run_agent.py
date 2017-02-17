import tensorflow as tf
from environments import GymEnvironment, AtariGymEnvironment
from utils import launch_workers, cluster_spec, dump_object, load_object
from network import Network
import subprocess
from redis import Redis
import time
import numpy as np

flags = tf.flags
flags.DEFINE_boolean('load_agent', False, 'whether agent will try to load weights')
flags.DEFINE_integer('test_games', 10, 'how many games agent should play for testing')
flags.DEFINE_string('save_name', 'pgq', 'save directory')
flags.DEFINE_integer('thread', 0, 'number of thread')
flags.DEFINE_string('env_class', 'gym', 'environment type')
flags.DEFINE_string('env_name', 'CartPole-v0', 'gym environment name')
flags.DEFINE_integer('n_threads', 6, 'number of workers')
flags.DEFINE_integer('port', 12001, 'starting port')
flags.DEFINE_integer('n_steps', 4, 'agent parameter')
flags.DEFINE_float('gamma', 0.99, 'agent parameter')
flags.DEFINE_float('dddqn_learning_rate', 0.00025, 'network parameter')
flags.DEFINE_float('softness_target_update', .001, 'how fast target network update its weights at every step')
flags.DEFINE_float('a3c_learning_rate', 0.0, 'network parameter')
flags.DEFINE_integer('batch_size', 32, 'network parameter')
flags.DEFINE_integer('buffer_max_size', 3000, 'max size of xp-replay buffer')
flags.DEFINE_integer('epoch_time', 500, 'how often to test target network')
flags.DEFINE_float('critic_loss_coef', 0.5, 'a3c parameter')
flags.DEFINE_float('entropy_coef', 0.001, 'a3c parameter')
flags.DEFINE_float('beta_1', 0.9, 'adam parameter')
flags.DEFINE_float('beta_2', 0.999, 'adam parameter')
FLAGS = flags.FLAGS
import os


def try_to_load_agent(variables_server):
    if not os.path.exists(FLAGS.save_name):
        print('This save doesn\'t exist!')
        return False
    count_of_weights = load_object(variables_server.get('count_of_weights'))
    try:
        for i in range(count_of_weights):
            weight = np.load(FLAGS.save_name + '/weight_{}.npy'.format(i))
            variables_server.set('weight_{}'.format(i), dump_object(weight))
            variables_server.set('target_weight_{}'.format(i), dump_object(weight))
        print('Weights are successfully downloaded from checkpoint')
        return True
    except:
        print('Something went wrong')
        return False


def initialize_weights(variables_server, weights):
    for i, weight in enumerate(weights):
        variables_server.set('weight_{}'.format(i), dump_object(weight.eval()))
        variables_server.set('target_weight_{}'.format(i), dump_object(weight.eval()))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    spec = tf.train.ClusterSpec(cluster_spec(n_workers=FLAGS.n_threads, port=FLAGS.port))
    server = tf.train.Server(spec, job_name='ps', task_index=0)
    sess = tf.InteractiveSession(server.target)

    if FLAGS.env_class == 'gym':
        env_class = GymEnvironment
    if FLAGS.env_class == 'atari-gym':
        env_class = AtariGymEnvironment
    test_environment = env_class(FLAGS.env_name)
    state_dim = test_environment.get_state_dim()
    action_dim = test_environment.get_action_dim()

    print('Starting parameter server...')
    cmd_server = 'redis-server --port 12000'
    p = subprocess.Popen(cmd_server, shell=True, preexec_fn=os.setsid)
    variables_server = Redis(port=12000)

    print('Global variables creating...')
    network_parameters = (sess, state_dim, action_dim, FLAGS.batch_size, FLAGS.critic_loss_coef, FLAGS.entropy_coef)
    network = Network('network', *network_parameters, initialize=False)
    weights = network.weights
    for i, weight in enumerate(weights):
        w_shape = weight.get_shape()
        # Shared weight
        variables_server.set('weight_{}'.format(i), dump_object(np.zeros(w_shape)))
        variables_server.set('target_weight_{}'.format(i), dump_object(np.zeros(w_shape)))
        # Shared momentum and velocity for Adam optimizer
        variables_server.set('momentum_{}'.format(i), dump_object(np.zeros(w_shape)))
        variables_server.set('velocity_{}'.format(i), dump_object(np.zeros(w_shape)))
    variables_server.set('update_steps', dump_object(0))
    variables_server.set('episodes', dump_object(0))
    variables_server.set('count_of_weights', dump_object(len(weights)))
    print('Starting asynchronous training...')
    processes = launch_workers(n_workers=FLAGS.n_threads)
    sess.run(tf.global_variables_initializer())
    print('Global weights initializing...')
    if FLAGS.load_agent:
        print('Trying to load agent...')
        if not try_to_load_agent(variables_server):
            initialize_weights(variables_server, weights)
    else:
        initialize_weights(variables_server, weights)

    print('Global weights initialized.')
    time.sleep(100000000)
    print("the end")
