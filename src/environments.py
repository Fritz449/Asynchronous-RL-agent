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


class ALEEnvironment:
    def __init__(self, name, width=84, height=84, buffer_size=4, disp=False):
        rom_file = '/home/fritz/PycharmProjects/RL-agent/roms/' + name + '.bin'
        from ale_python_interface import ALEInterface
        self.name = name
        self.width = width
        self.height = height
        self.buffer_size = buffer_size
        self.state_dim = (width, height, buffer_size)
        self.env = ALEInterface()
        self.env.setInt('random_seed', np.random.randint(10000000))
        self.env.setBool('color_averaging', True)
        self.env.setInt('frame_skip', 4)
        self.env.setFloat('repeat_action_probability', 0)
        self.env.setBool('display_screen', disp)
        self.env.loadROM(rom_file)
        self.actions = self.env.getMinimalActionSet()
        self.buf = np.zeros(self.state_dim)
        self.total_reward = 0

    def step(self, action):
        reward = self.env.act(self.actions[action])
        screen = self.env.getScreenGrayscale()
        self.buf = np.roll(self.buf, 1, axis=2)
        self.buf[:, :, 0] = cv2.resize(screen, (self.width, self.height))
        reward = np.clip(reward, -1, 1)
        self.total_reward += reward
        return self.buf, reward, self.env.game_over()

    def reset(self, random_starts=10):
        self.env.reset_game()
        self.total_reward = 0
        self.buf = np.zeros(self.state_dim)
        for i in range(np.random.randint(random_starts)):
            self.step(np.random.randint(0, self.env.action_space.n))
        return self.buf

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return len(self.actions)

    def done(self):
        return self.env.game_over()

    def get_observation(self):
        return self.buf

    def get_total_reward(self):
        return self.total_reward
