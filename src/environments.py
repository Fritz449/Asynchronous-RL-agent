import gym

import numpy as np
import cv2


class AtariGymEnvironment:
    def __init__(self, name, width=84, height=84, buffer_size=4, frame_skip=4):
        self.observation = np.zeros((width, height))
        self.name = name
        self.width = width
        self.height = height
        self.buffer_size = buffer_size
        self.env = gym.make(name)
        self.state_dim = (width, height, buffer_size)
        self.buf = np.zeros(self.state_dim)
        self.total_reward = 0
        self.done = True
        self.frame_skip = frame_skip

    def step(self, action):
        next_obs, reward, done, _ = self.env.step(action)
        self.buf = np.roll(self.buf, 1, axis=2)
        self.buf[:, :, 0] = cv2.resize(cv2.cvtColor(next_obs, cv2.COLOR_RGB2GRAY), (self.width, self.height)) / 255.
        self.total_reward += reward
        reward = np.clip(reward, -1, 1)
        self.done = done
        return self.buf, reward, done

    def reset(self, random_starts=10):
        self.env.reset()
        self.total_reward = 0
        self.buf = np.zeros(self.state_dim)
        for i in range(np.random.randint(random_starts)):
            self.step(np.random.randint(0, self.env.action_space.n))
        return self.buf

    def get_state_dim(self):
        return self.state_dim

    def done(self):
        return self.done

    def get_action_dim(self):
        return self.env.action_space.n

    def get_total_reward(self):
        return self.total_reward

    def get_observation(self):
        return self.buf


class GymEnvironment:
    def __init__(self, name):
        self.name = name
        self.env = gym.make(name)
        self.state_dim = self.env.observation_space.shape
        self.obs = np.zeros(self.state_dim)
        self.total_reward = 0
        self.done = True

    def step(self, action):
        next_obs, reward, done, _ = self.env.step(action)
        self.obs = next_obs
        self.total_reward += reward
        self.done = done
        return next_obs, reward, done

    def reset(self, random_starts=0):
        self.obs = self.env.reset()
        self.total_reward = 0
        for i in range(random_starts):
            self.step(np.random.randint(0, self.env.action_space.n))
        return self.obs

    def get_state_dim(self):
        return self.state_dim

    def done(self):
        return self.done

    def get_action_dim(self):
        return self.env.action_space.n

    def get_total_reward(self):
        return self.total_reward

    def get_observation(self):
        return self.obs

class RamGymEnvironment:
    def __init__(self, name):
        self.name = name
        self.env = gym.make(name)
        self.state_dim = self.env.observation_space.shape
        self.obs = np.zeros(self.state_dim)
        self.total_reward = 0
        self.done = True

    def step(self, action):
        next_obs, reward, done, _ = self.env.step(action)
        next_obs= next_obs.astype('float32')/255.
        self.obs = next_obs
        self.total_reward += reward
        self.done = done
        return next_obs, reward, done

    def reset(self, random_starts=0):
        self.obs = self.env.reset().astype('float32')/255.
        self.total_reward = 0
        for i in range(random_starts):
            self.step(np.random.randint(0, self.env.action_space.n))
        return self.obs

    def get_state_dim(self):
        return self.state_dim

    def done(self):
        return self.done

    def get_action_dim(self):
        return self.env.action_space.n

    def get_total_reward(self):
        return self.total_reward

    def get_observation(self):
        return self.obs
