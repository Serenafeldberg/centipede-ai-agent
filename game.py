import gymnasium as gym
from agents.feldberg.agent import FeldbergAgent
from agents.feldberg.state_extractor import StateExtractor

agent = FeldbergAgent()
extractor = StateExtractor()
env = gym.make('ALE/Centipede-v5', render_mode='human')
obs, _ = env.reset()

while True:
    state = extractor.extract(obs)
    action = agent.action(state)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(agent.name())