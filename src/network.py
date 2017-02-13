import numpy as np
import tensorflow as tf
import os


def weight_variable(name, shape):
    if len(shape) == 4:
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True),
                               dtype=tf.float32)
    else:
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                               dtype=tf.float32)


def bias_variable(name, shape):
    return tf.get_variable(name, initializer=tf.constant(0.1, shape=shape), dtype=tf.float32)


def conv_layer(name, input_tensor, kernel_shape, elu=True, max_pooling=None, strides=1):
    kernel = weight_variable(name + "_kernel", kernel_shape)
    biases = bias_variable(name + "_bias", (kernel_shape[3],))
    convolution = tf.nn.conv2d(input_tensor, kernel, strides=[1, strides, strides, 1], padding='SAME')
    result = convolution + biases
    if elu:
        result = tf.nn.elu(result)

    if max_pooling is not None:
        return tf.nn.max_pool(result, ksize=[1, max_pooling, max_pooling, 1],
                              strides=[1, max_pooling, max_pooling, 1],
                              padding='SAME')
    return result


def fc_layer(name, input_tensor, n_out, elu=True):
    weights = weight_variable(name + "_weights", shape=(int(input_tensor.get_shape()[1]), n_out))
    biases = bias_variable(name + "_bias", shape=(n_out,))
    result = tf.matmul(input_tensor, weights) + biases
    if elu:
        result = tf.nn.elu(result)
    return result


class Network:
    def create_conv_model(self):
        # This is the place where neural network model initialized
        self.l1 = conv_layer(self.name + '_conv1', self.state_in, (8, 8, 4, 32), strides=4)
        self.l2 = conv_layer(self.name + '_conv2', self.l1, (4, 4, 32, 64), strides=2)
        self.l3 = conv_layer(self.name + '_conv3', self.l2, (3, 3, 64, 64), strides=1)
        self.h = tf.reshape(self.l3, [tf.shape(self.l3)[0],
                                      int(self.l3.get_shape()[1] * self.l3.get_shape()[2] * self.l3.get_shape()[3])])
        self.hidden = fc_layer(self.name + '_hidden', self.h, 512)
        self.policy = tf.nn.softmax(fc_layer(self.name + '_policy', self.hidden, self.action_dim, elu=False))
        self.value = fc_layer(self.name + '_value', self.hidden, 1, elu=False)
        self.q_values = self.entropy_coef * (tf.log(self.policy + 1e-8) +
                                             (tf.tile(tf.reduce_sum(tf.log(self.policy + 1e-8) * self.policy, axis=[1],
                                                                    keep_dims=True), [1, self.action_dim]))
                                             + self.value)

    def create_fc_model(self):
        # This is the place where neural network model initialized
        # self.hidden = fc_layer(self.name + '_hidden', self.state_in, 128)
        self.hidden = self.state_in
        self.policy = tf.nn.softmax(fc_layer(self.name + '_policy', self.hidden, self.action_dim, elu=False))
        self.value = fc_layer(self.name + '_value', self.hidden, 1, elu=False)
        self.q_values = self.entropy_coef * (tf.log(self.policy + 1e-18) +
                                             (tf.tile(tf.reduce_sum(tf.log(self.policy + 1e-18) * self.policy, axis=[1],
                                                                    keep_dims=True), [1, self.action_dim]))
                                             + self.value)

    def __init__(self, name, sess, state_dim, action_dim, batch_size=32, critic_loss_coef=0.5, entropy_coef=0.001,
                 initialize=True):

        # Assign network features
        self.sess = sess
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef
        # Create input for training
        with tf.variable_scope(self.name):
            self.state_in = tf.placeholder(shape=((None,) + self.state_dim), dtype='float32')
            self.actions = tf.placeholder(shape=(None,), dtype='int32')
            self.value_target = tf.placeholder(shape=(None,), dtype='float32')
            # These weights are for weighted update
            self.weights_loss = tf.placeholder(shape=(None,),
                                               dtype='float32')
            self.q_value_target = tf.placeholder(shape=(None,), dtype='float32')

        # Initialize model
        if len(state_dim) == 3:
            self.create_conv_model()
        else:
            self.create_fc_model()

        self.weights = [t for t in tf.trainable_variables() if t.name.startswith('global/' + self.name)]
        self.weights = sorted(self.weights, key=lambda x: x.name)
        self.update_weights_ops = []
        self.update_placeholders = []
        for index in range(len(self.weights)):
            weight = self.weights[index]
            placeholder = tf.placeholder('float32', shape=weight.get_shape())
            self.update_placeholders.append(placeholder)
            self.update_weights_ops.append(weight.assign(placeholder))
        # Get q-values for corresponding actions
        action_opinion = self.policy
        value_opinion = tf.reshape(self.value, [-1])

        # CRITIC
        self.td_error = self.value_target - value_opinion
        self.critic_loss = 0.5 * tf.reduce_sum(self.td_error ** 2)

        # ACTOR
        # entropy terms
        log_prob_all = tf.log(action_opinion + 1e-8)
        entropy = -1. * tf.reduce_sum(log_prob_all * action_opinion, axis=[1])
        # objective part
        batch_numbering = tf.range(tf.shape(action_opinion)[0])
        indices = tf.stack([batch_numbering, self.actions], axis=1)
        log_prob = tf.gather_nd(log_prob_all, indices)
        advantage = tf.stop_gradient(self.td_error)
        actor_loss = -1. * (log_prob * advantage + entropy_coef * entropy)
        self.actor_loss = tf.reduce_sum(actor_loss)

        self.total_a3c_loss = self.actor_loss + critic_loss_coef * self.critic_loss
        q_value = tf.gather_nd(self.q_values, indices)
        self.td_q_error = self.q_value_target - q_value
        self.dqn_loss = tf.reduce_sum(self.weights_loss * (self.td_q_error ** 2))

        self.optimizer = tf.train.GradientDescentOptimizer(1)
        gradients_a3c = self.optimizer.compute_gradients(self.total_a3c_loss)
        gradients_a3c = [gr for gr in gradients_a3c if gr[0] is not None]
        gradients_a3c = sorted(gradients_a3c, key=lambda x: x[1].name)
        gradients_a3c = [g[0] for g in gradients_a3c]
        for i in range(len(gradients_a3c)):
            gradients_a3c[i] = tf.clip_by_norm(gradients_a3c[i], 10)

        self.gradients_a3c = gradients_a3c
        self.dqn_optimizer = tf.train.GradientDescentOptimizer(1)
        gradients_dqn = self.dqn_optimizer.compute_gradients(self.dqn_loss)
        gradients_dqn = [gr for gr in gradients_dqn if gr[0] is not None]
        gradients_dqn = sorted(gradients_dqn, key=lambda x: x[1].name)
        gradients_dqn = [g[0] for g in gradients_dqn]
        for i in range(len(gradients_dqn)):
            gradients_dqn[i] = tf.clip_by_norm(gradients_dqn[i], 10)
        self.gradients_dqn = gradients_dqn
        if initialize:
            sess.run(tf.variables_initializer(self.weights))

    def compute_a2c_outputs(self, value_target, state_in, actions):
        return self.sess.run([self.gradients_a3c, self.total_a3c_loss, self.td_error], feed_dict={self.actions: actions,
                                                                                                  self.state_in: state_in,
                                                                                                  self.value_target: value_target,
                                                                                                  })

    def compute_dqn_outputs(self, q_value_target, state_in, actions, weights):
        if weights is None:
            weights = np.ones((state_in.shape[0],))
        return self.sess.run([self.gradients_dqn, self.dqn_loss, self.td_q_error], feed_dict={self.actions: actions,
                                                                                              self.state_in: state_in,
                                                                                              self.q_value_target: q_value_target,
                                                                                              self.weights_loss: weights
                                                                                              })

    def compute_q_values(self, states_input):
        return self.q_values.eval(feed_dict={self.state_in: states_input})

    def compute_value(self, obs):
        return self.value.eval(feed_dict={self.state_in: obs})[0]

    def get_action(self, obs, eps=0., greedy=False):
        if np.random.rand() <= eps:
            return np.random.randint(self.action_dim)
        else:
            probs = self.sess.run([self.policy], feed_dict={self.state_in: obs.reshape((1,) + obs.shape)})[0][0]
            if greedy:
                return np.argmax(probs)
            probs = probs - np.finfo(np.float32).epsneg
            histogram = np.random.multinomial(1, probs)
            return histogram.argmax()

    def get_q_action(self, obs, eps, greedy=False):
        if np.random.rand() <= eps:
            return np.random.randint(self.action_dim)
        else:
            q_values = self.compute_q_values(obs.reshape((1,) + obs.shape))[0]
            if greedy:
                return np.argmax(q_values)
            return np.argmax(q_values)

    def assign_weights(self, new_weights, coef=1.):
        for i in range(len(self.weights)):
            weight = self.weights[i]
            new_weight = new_weights[i]
            placeholder = self.update_placeholders[i]
            update_op = self.update_weights_ops[i]
            self.sess.run(update_op, feed_dict={placeholder: (1 - coef) * weight.eval() + coef * new_weight.eval()})
