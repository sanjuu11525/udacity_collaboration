from unityagents import UnityEnvironment
import numpy as np
import torch
import os

from agent import MADDPG

class Params:
    """Set up configuration here."""
    def __init__(self):
        self.__dict__.update(**{
            'buffer_size' : int(1e4),  # replay buffer size
            'batch_size'  : 256,       # minibatch size
            'gamma'       : 0.99,      # discount factor
            'tau'         : 1e-3,      # for soft update of target parameters
            'lr'          : 1e-4,      # learning rate 
            'update_every' : 1,        # how often to update the network
})

if __name__ == '__main__':

    # env setup
    env_file_name = os.path.abspath("Tennis_Linux/Tennis.x86_64")
    env = UnityEnvironment(file_name=env_file_name, no_graphics=False)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    device = torch.device("cpu")

    # MADDPG agent
    agent = MADDPG(state_size=state_size, action_size=action_size, params=Params(), device=device)
    # model path
    model_path_0 = os.path.abspath('checkpoint/checkpoint_0.pth')
    model_path_1 = os.path.abspath('checkpoint/checkpoint_1.pth')
    agent.load_agent(model_path_0, model_path_1)

    for i_episode in range(1, 20):
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        state = env_info.vector_observations               # get the current state
        score = np.zeros(num_agents)
        while True:
            action = agent.act(state)
            env_info   = env.step(action)[brain_name]      # send the action to the environment
            next_state = env_info.vector_observations      # get the next state
            reward     = env_info.rewards                  # get the reward
            done       = env_info.local_done               # see if episode has finished
            state = next_state
            score += reward
            if np.any(done):
                break
        
        print(f'episode: {i_episode}, score: {score}')
