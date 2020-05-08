import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque

from network import *


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device, seed=0):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.tensor([e.state for e in experiences if e is not None], dtype=torch.float, device=self.device)
        actions = torch.tensor([e.action for e in experiences if e is not None], dtype=torch.float, device=self.device)
        rewards = torch.tensor([e.reward for e in experiences if e is not None], dtype=torch.float, device=self.device)
        next_states = torch.tensor([e.next_state for e in experiences if e is not None], dtype=torch.float, device=self.device)
        dones = torch.tensor([e.done for e in experiences if e is not None], dtype=torch.float, device=self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        self.state = x + dx
        return self.state * self.scale


class DDPGAgent():
    """Agent class for making actions."""

    def __init__(self, state_size, action_size, params, device, seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.device = device
        
        self.noise = OUNoise(action_size)
        
        self.critic = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params.lr)
        
        self.actor = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params.lr)

    def reset(self):
        """Reset noise"""
        self.noise.reset()

    def load_actor(self, checkpoint):
        """Load actor for inference only.
        
        Params
        ======
            checkpoint (string): model path
        """
        
        model = torch.load(checkpoint)
        self.actor.load_state_dict(model)

    def act(self, state, noise=0.0):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            noise (float): noise for exploration
        """
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)

        self.actor.eval()
        with torch.no_grad():
            action_values = self.actor(state).squeeze(0).cpu().data.numpy()
        self.actor.train()

        # action with noise decay
        action_values = action_values + noise * self.noise.noise()
        
        return np.clip(action_values, -1, 1)
            

class MADDPG:
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, params, device, seed=0):
        
        # Multi-agent setup
        self.maddpg_agent = [DDPGAgent(state_size, action_size, params, device, seed),
                             DDPGAgent(state_size, action_size, params, device, seed)]

        # Replay buffer
        self.memory = ReplayBuffer(action_size=action_size     , buffer_size=params.buffer_size,
                                   batch_size=params.batch_size, device=device)

        self.t_step = 0
        
        self.params = params
    def reset(self):
        for agent in self.maddpg_agent:
            agent.reset()
                             
    def load_agent(self, checkpoint_0, checkpoint_1):
        """Load actors for inference only"""
        self.maddpg_agent[0].load_actor(checkpoint_0)
        self.maddpg_agent[1].load_actor(checkpoint_1)                  
            
    def step(self, state, action, reward, next_state, done):
        """Queue experience in reply memory and make train the model.
        
        Params
        ======
            state (array_like): current state
            action (array_like): action to next state
            reward (array_like): given reward by the action
            next_state (array_like): next state
            done (array_like): if the episodic task done
        """
        
        # Save global experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.params.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.params.batch_size:
                experiences = self.memory.sample()
                for agent_id in range(len(self.maddpg_agent)):
                    self.learn(agent_id, experiences)

    def act(self, states, noise=0.0):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """ 

        actions = [agent.act(state, noise) for agent, state in zip(self.maddpg_agent, states)]
            
        return actions

    def learn(self, agent_id, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            agent_id (int): id of agent
        """
        # dimension of states      (num_batch, num_agent, 24)
        # dimension of actions     (num_batch, num_agent, 2)
        # dimension of rewards     (num_batch, 2)
        # dimension of next_states (num_batch, num_agent, 24)
        # dimension of dones       (num_batch, 2)
        states, actions, rewards, next_states, dones = experiences
        
        # select agent for learning
        agent = self.maddpg_agent[agent_id]
                             
        # centralized information
        # dimension of full states (num_batch, num_agent * 24)
        full_states = states.view(self.params.batch_size, -1)
        full_next_states = next_states.view(self.params.batch_size, -1)
        # dimension of actions (num_batch, num_agent * 2)
        full_actions = actions.view(self.params.batch_size, -1)
    
        # compute and minimize the action value
        full_next_actions = []
        for id_, agent_ in enumerate(self.maddpg_agent):
            full_next_actions.append(agent_.actor_target(next_states[:,id_,:]))

        full_next_actions = torch.cat(full_next_actions, dim=1)
        next_Qs = agent.critic_target(full_next_states, full_next_actions).detach()

        target_Qs = rewards[:, agent_id].view(-1,1) + self.params.gamma * next_Qs * (1 - dones[:, agent_id].view(-1,1))
        expected_Qs = agent.critic(full_states, full_actions)
        
        loss_critic = F.mse_loss(expected_Qs, target_Qs)
        agent.critic_optimizer.zero_grad()
        loss_critic.backward()
        agent.critic_optimizer.step()

        # compute and maximize the policy
        tmp = []
        for id_, agent_ in enumerate(self.maddpg_agent):
            if agent_id == id_:
                # update current agent
                tmp.append(agent_.actor(states[:,id_,:]))
            else:
                # detach other agents
                tmp.append(agent_.actor(states[:,id_,:]).detach())
                
        full_current_actions = torch.cat(tmp, dim=1)

        loss_actor = -agent.critic(full_states, full_current_actions).mean()
        agent.actor_optimizer.zero_grad()    
        loss_actor.backward()
        agent.actor_optimizer.step()
        
        self.soft_update(agent.critic, agent.critic_target, self.params.tau)
        self.soft_update(agent.actor, agent.actor_target, self.params.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            