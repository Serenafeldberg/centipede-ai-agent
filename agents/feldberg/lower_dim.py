import numpy as np
from state_extractor import StateExtractor
from tqdm import tqdm

# states_ = np.load('agents/feldberg/39501_states.npy')
# actions_ = np.load('agents/feldberg/39501_actions.npy')

states_ = np.load('agents/feldberg/states_pre/states_start.npy', allow_pickle=True)
actions_ = np.load('agents/feldberg/actions_pre/actions_start.npy', allow_pickle=True)


if len(states_) != len(actions_):
    if len(states_) > len(actions_):
        states_ = states_[:-1]
    else:
        actions_ = actions_[:-1]

extractor = StateExtractor()

states = []
actions = []

for i in tqdm(range(len(states_)), desc='Extracting states'):
    s = extractor.extract(states_[i])
    states.append(s)

states = np.array(states)
np.save('agents/feldberg/states/start_states_new.npy', states)

for action in tqdm(actions_, desc='Extracting actions'):
    one_hot = np.eye(18)[action]
    actions.append(one_hot)

actions = np.array(actions)
np.save('agents/feldberg/actions/start_actions_new.npy', actions)


