import numpy as np
from keras import backend as Theano
from keras.layers import Dense, Input, Convolution2D, Flatten, merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adadelta, RMSprop, Adam, SGD
from keras.regularizers import l1, l2
from keras.initializations import normal, glorot_uniform
import os


class Network:
    def create_conv_model(self):
        # This is the place where neural network model initialized
        init = 'glorot_uniform'
        self.state_in = Input(self.state_dim)
        self.l1 = Convolution2D(32, 8, 8, activation='elu', init=init, subsample=(4, 4), border_mode='same')(
            self.state_in)
        self.l2 = Convolution2D(64, 4, 4, activation='elu', init=init, subsample=(2, 2), border_mode='same')(
            self.l1)
        # self.l3 = Convolution2D(64, 3, 3, activation='relu', init=init, subsample=(1, 1), border_mode='same')(
        #     self.l2)
        self.l3 = self.l2
        self.h = Flatten()(self.l3)
        self.hidden = Dense(256, init=init, activation='elu')(self.h)
        self.value = Dense(1, init=init)(self.hidden)
        self.policy = Dense(self.action_dim, init=init, activation='softmax')(self.hidden)
        self.q_values = self.entropy_coef * (Theano.log(self.policy + 1e-18) -
                                             Theano.tile(Theano.sum(Theano.log(self.policy + 1e-18) * self.policy,
                                                                    axis=[1], keepdims=True), (1, self.action_dim)))
        self.q_values = self.q_values + Theano.tile(self.value, (1, self.action_dim))
        self.model = Model(self.state_in, output=[self.policy, self.value])

    def create_fc_model(self):
        # This is the place where neural network model initialized
        init = 'glorot_uniform'
        self.state_in = Input(self.state_dim)
        self.hidden = Dense(256, init=init, activation='elu')(self.state_in)
        self.value = Dense(1)(self.hidden)
        self.policy = Dense(self.action_dim, init=init, activation='softmax')(self.hidden)

        self.q_values = self.entropy_coef * (Theano.log(self.policy + 1e-18) -
                                             Theano.tile(Theano.sum(Theano.log(self.policy + 1e-18) * self.policy,
                                                                    axis=[1], keepdims=True), (1, self.action_dim)))
        # print (type(Theano.sum(Theano.log(self.policy + 1e-18) * self.policy,
        #                                                 axis=[1], keepdims=True)))
        # print(Theano.function([self.state_in], [Theano.sum(Theano.log(self.policy + 1e-18) * self.policy,
        #                                                 axis=[1], keepdims=True)])([np.zeros((32,) + self.state_dim)])[0].shape)
        # 1/0
        self.q_values = self.q_values + Theano.tile(self.value, (1, self.action_dim))
        self.model = Model(self.state_in, output=[self.policy, self.value])

    def __init__(self, name, state_dim, action_dim, batch_size=32, critic_loss_coef=0.5, entropy_coef=0.001,
                 initialize=True):

        # Assign network features
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef
        # Create input for training
        self.actions = Input(shape=(None,), dtype='int32')
        self.value_target = Input(shape=(None,), dtype='float32')
        self.q_values_target = Input(shape=(None,), dtype='float32')

        # Initialize model
        if len(state_dim) == 3:
            self.create_conv_model()
        else:
            self.create_fc_model()

        self.weights = self.model.trainable_weights
        self.reg_loss = 0
        for weight in self.weights:
            self.reg_loss += 0.001 * Theano.sum(weight ** 2)
        self.weights = sorted(self.weights, key=lambda x: x.name)
        # print ([x.name for x in self.weights])
        # Get q-values for corresponding actions
        action_opinion = self.policy
        value_opinion = Theano.reshape(self.value, [-1])

        # CRITIC
        self.td_error = self.value_target - value_opinion
        self.critic_loss = 0.5 * Theano.sum(self.td_error ** 2)

        # ACTOR
        # entropy terms
        log_prob_all = Theano.log(action_opinion + 1e-18)
        entropy = -1. * Theano.sum(log_prob_all * action_opinion, axis=[1])
        # objective part
        action_mask = Theano.T.eq(Theano.T.arange(self.action_dim).reshape((1, -1)),
                                  self.actions.reshape((-1, 1))).astype(Theano.T.config.floatX)
        log_prob = Theano.T.sum(log_prob_all * action_mask, axis=1, keepdims=True)
        advantage = Theano.stop_gradient(self.td_error)
        actor_loss = -1. * (log_prob * advantage + entropy_coef * entropy)
        self.actor_loss = Theano.sum(actor_loss)

        self.total_a3c_loss = self.actor_loss + critic_loss_coef * self.critic_loss + self.reg_loss
        q_value = Theano.T.sum(self.q_values * action_mask, axis=1, keepdims=True)
        self.td_q_error = self.q_values_target - q_value
        self.dqn_loss = Theano.sum((self.td_q_error ** 2)) + self.reg_loss

        self.optimizer = SGD(lr=1)
        gradients_a3c = Theano.gradients(self.total_a3c_loss, self.weights)
        self.gradients_a3c = gradients_a3c

        gradients_dqn = Theano.gradients(self.dqn_loss, self.weights)
        self.gradients_dqn = gradients_dqn

        self.a2c_outputs = Theano.function([self.state_in, self.actions, self.value_target],
                                           self.gradients_a3c)

        self.dqn_outputs = Theano.function([self.state_in, self.actions, self.q_values_target],
                                           self.gradients_dqn)

        self.get_q_values = Theano.function([self.state_in], [self.q_values])
        self.get_value = Theano.function([self.state_in], [self.value])
        self.get_policy = Theano.function([self.state_in], [self.policy])

    def compute_a2c_outputs(self, value_target, state_in, actions):
        actions = actions.reshape(actions.shape + (1,))
        value_target = value_target.reshape((1,) + value_target.shape)
        return self.a2c_outputs([state_in, actions, value_target])

    def compute_dqn_outputs(self, q_value_target, state_in, actions):
        actions = actions.reshape(actions.shape + (1,))
        q_value_target = q_value_target.reshape(q_value_target.shape + (1,))
        return self.dqn_outputs([state_in, actions, q_value_target])

    def compute_q_values(self, state_input):
        return self.get_q_values([state_input])[0]

    def compute_value(self, state_input):
        return self.get_value([state_input])[0]

    def get_action(self, state, eps=0., greedy=False):
        if np.random.rand() <= eps:
            return np.random.randint(self.action_dim)
        else:
            probs = self.get_policy([state.reshape((1,) + state.shape)])[0][0]
            # print (probs, 'policy')
            # print (self.get_value([state.reshape((1,) + state.shape)]))
            # print (self.get_q_values([state.reshape((1,) + state.shape)])[0], 'qs')
            # 1/0
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
            weight.set_value((coef * new_weight + (1 - coef) * weight.get_value()).astype('float32'))
