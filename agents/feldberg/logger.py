import gymnasium as gym
from gymnasium.utils.play import play
import numpy as np
# Ver https://gymnasium.farama.org/api/utils/ para mas info

class Logger:
    def __init__(self):
        self.actions = []
        self.obs_t = []
        self.obs_tp1 = []
        self.rewards = []
        self.terminated = []
        self.truncated = []
        self.info = []

    def log_callback(self, obs_t, obs_tp1, action, rew, teminated, truncated, info):
        self.actions.append(action)
        if type(obs_t) != tuple:
            self.obs_t.append(obs_t)
        else:
            self.obs_t.append(obs_t[0])
        self.obs_tp1.append(obs_tp1)
        self.rewards.append(rew)
        self.terminated.append(teminated)
        self.truncated.append(truncated)
        self.info.append(info)
        return action

logger = Logger()

play(gym.make('CentipedeNoFrameskip-v4', render_mode='rgb_array'), zoom=3, fps=60, callback=logger.log_callback)
#play(gym.make('ALE/Centipede-v5'...  # El enviroment que usamos nosotros (es mas dificil de jugar a mano)

# Para ver los datos que se guardaron, por ej:
print(logger.actions)

np.save('actions/actions_7.npy', logger.actions)
np.save('states/states_7.npy', logger.obs_t)
