import numpy as np
import gym
from kaggle_environments import make

class ConnectFourGym(gym.Env):
    def __init__(self, agent2=None):
        super(ConnectFourGym, self).__init__()
        
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([None, agent2])
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        self.inarow = ks_env.configuration.inarow

        self.action_space = gym.spaces.Discrete(self.columns)
        self.observation_space = gym.spaces.Box(low=0, high=2, 
                                                shape=(1, self.rows, self.columns), dtype=np.int32)

        self.reward_range = (-10, 1)
        self.spec = None
        self.metadata = None

    def reset(self):
        self.obs = self.env.reset()
        board = np.array(self.obs['board']).reshape(1, self.rows, self.columns)
        return board.astype(np.float32)  

    def step(self, action):
        is_valid = self.obs['board'][action] == 0
        if is_valid:
            self.obs, old_reward, done, _ = self.env.step(action)
            reward = self._custom_reward(old_reward, done)
        else:
            reward, done, _ = -10, True, {}

        board = np.array(self.obs['board']).reshape(1, self.rows, self.columns).astype(np.float32)
        return board, reward, done, _

    def _custom_reward(self, old_reward, done):
        if old_reward == 1:
            return 1.0
        elif done:
            return -1.0
        else:
            return 1.0 / (self.rows * self.columns)

class ConnectFiveGym(gym.Env):
    def __init__(self, agent2=None):
        super(ConnectFiveGym, self).__init__()

        ks_env = make("connectx", debug=True)
        ks_env.configuration.inarow = 5
        ks_env.configuration.rows = 8
        ks_env.configuration.columns = 8

        self.env = ks_env.train([None, agent2])
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        self.inarow = ks_env.configuration.inarow

        self.action_space = gym.spaces.Discrete(self.columns)
        self.observation_space = gym.spaces.Box(low=0, high=2, 
                                                shape=(1, self.rows, self.columns), dtype=np.int32)

        self.reward_range = (-10, 1)
        self.spec = None
        self.metadata = None

    def reset(self):
        self.obs = self.env.reset()
        board = np.array(self.obs['board']).reshape(1, self.rows, self.columns)
        return board.astype(np.float32)  

    def step(self, action):
        is_valid = self.obs['board'][action] == 0
        if is_valid:
            self.obs, old_reward, done, _ = self.env.step(action)
            reward = self._custom_reward(old_reward, done)
        else:
            reward, done, _ = -10, True, {}

        board = np.array(self.obs['board']).reshape(1, self.rows, self.columns).astype(np.float32)
        return board, reward, done, _

    def _custom_reward(self, old_reward, done):
        if old_reward == 1:
            return 1.0
        elif done:
            return -1.0
        else:
            return 1.0 / (self.rows * self.columns)
