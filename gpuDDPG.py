from mlagents_envs.environment import UnityEnvironment
import numpy as np
from mlagents_envs.base_env import ActionTuple
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F


# This is a non-blocking call that only loads the environment.
print("connecting to env...")
env = UnityEnvironment(file_name="gpubuild.x86_64", seed=1, side_channels=[], no_graphics=True)
# Start interacting with the environment.
print("connected to env")
env.reset()
knight_one_names = env.behavior_specs.keys()
print(knight_one_names)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Actor(nn.Module):
    def __init__(self, state_size, action_size, action_bound, visual_input_shape):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(visual_input_shape[2], 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv_output_size = self._get_conv_output_size(visual_input_shape)

        self.fc1 = nn.Linear(self.conv_output_size + state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)
        self.action_bound = action_bound
        self.to(device)

    def _get_conv_output_size(self, shape):
        x = torch.rand(1, shape[2], shape[0], shape[1]).to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return int(np.prod(x.size()))

    def forward(self, visual_input, vector_input):
        x1 = torch.relu(self.conv1(visual_input))
        x1 = torch.relu(self.conv2(x1))
        x1 = x1.reshape(x1.size(0), -1)

        x = torch.cat((x1, vector_input), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.action_bound
        return x

class Critic(nn.Module):
    def __init__(self, state_size, action_size, visual_input_shape):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(visual_input_shape[2], 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv_output_size = self._get_conv_output_size(visual_input_shape)

        self.fc1 = nn.Linear(self.conv_output_size + state_size + action_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.to(device)

    def _get_conv_output_size(self, shape):
        x = torch.rand(1, shape[2], shape[0], shape[1]).to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return int(np.prod(x.size()))

    def forward(self, visual_input, vector_input, action):
        x1 = torch.relu(self.conv1(visual_input))
        x1 = torch.relu(self.conv2(x1))
        x1 = x1.view(x1.size(0), -1)

        x = torch.cat((x1, vector_input, action), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DDPG:
    def __init__(self, state_size, visual_input_shape, action_size, action_bound):
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        self.memory = RingBuffer(max_size=20000)
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        
        self.actor = Actor(state_size, action_size, action_bound, visual_input_shape).to(device)
        self.target_actor = Actor(state_size, action_size, action_bound, visual_input_shape).to(device)
        self.critic = Critic(state_size, action_size, visual_input_shape).to(device)
        self.target_critic = Critic(state_size, action_size, visual_input_shape).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        self.update_target_network(self.target_actor, self.actor, 1.0)
        self.update_target_network(self.target_critic, self.critic, 1.0)
    
    def update_target_network(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
    
    def memorize(self, state, action, reward, next_state, done):
        action = np.array(action).flatten()
        self.memory.add((state, action, reward, next_state, done))
    
    def act(self, visual_input, vector_input):
        print(f"vis input shape: {visual_input.shape}")
        visual_input = torch.FloatTensor(visual_input).to(device)
        if visual_input.dim() == 3:
            visual_input = visual_input.unsqueeze(0)  # Add batch dimension if needed
        visual_input = visual_input.permute(0, 3, 1, 2)  # Rearrange dimensions
        
        vector_input = torch.FloatTensor(vector_input).unsqueeze(0).to(device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(visual_input, vector_input).cpu().data.numpy().flatten()
        self.actor.train()
        return action + np.random.normal(0, 0.1, size=self.action_size)
    
    def replay(self):
        print("replaying experience")
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = self.memory.sample(self.batch_size)

        print(f"minibatch shape: ({len(minibatch)}, {len(minibatch[0])})")
        print(f"state shape: ({len(minibatch[0][0][0])}, {len(minibatch[0][0][1])})")
        print(f"action shape: {len(minibatch[0][1])}")
        print(f"reward: {minibatch[0][2]}")
        print(f"next state shape: ({len(minibatch[0][3][0])}, {len(minibatch[0][3][1])})")
        print(f"done: {minibatch[0][4]}")
    
        visual_inputs = torch.FloatTensor(np.array([m[0][0] for m in minibatch])).to(device)
        visual_inputs = visual_inputs.permute(0, 3, 1, 2)  # Rearrange dimensions
        vector_inputs = torch.FloatTensor(np.array([m[0][1] for m in minibatch])).to(device)
        actions = torch.FloatTensor(np.array([m[1] for m in minibatch])).to(device)
        rewards = torch.FloatTensor(np.array([m[2] for m in minibatch])).unsqueeze(1).to(device)
        next_visual_inputs = torch.FloatTensor(np.array([m[3][0] for m in minibatch])).to(device)
        next_visual_inputs = next_visual_inputs.permute(0, 3, 1, 2)
        next_vector_inputs = torch.FloatTensor(np.array([m[3][1] for m in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([m[4] for m in minibatch])).unsqueeze(1).to(device)
        
        # Update Critic
        next_actions = self.target_actor(next_visual_inputs, next_vector_inputs)
        next_q_values = self.target_critic(next_visual_inputs, next_vector_inputs, next_actions)
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        q_expected = self.critic(visual_inputs, vector_inputs, actions)
        critic_loss = nn.MSELoss()(q_expected, q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        actions_pred = self.actor(visual_inputs, vector_inputs)
        actor_loss = -self.critic(visual_inputs, vector_inputs, actions_pred).mean()
        
        print(f"actor loss: {actor_loss}, critic loss: {critic_loss}")
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update of target networks
        self.update_target_network(self.target_actor, self.actor, self.tau)
        self.update_target_network(self.target_critic, self.critic, self.tau)

    def save(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {filename}")

class RingBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.index = 0

    def add(self, item):
        if len(self.buffer) < self.max_size:
            self.buffer.append(item)
        else:
            self.buffer[self.index] = item
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
# Get the state and action sizes
wooden_knight = list(env.behavior_specs.keys())[0]
grass_knight = list(env.behavior_specs.keys())[1]

behavior_spec = env.behavior_specs[wooden_knight]

# Extract observation and action space sizes
actuator_obs_space = behavior_spec.observation_specs[0][0][0]
visual_obs_space = behavior_spec.observation_specs[1].shape
print(f"VISOBS: {visual_obs_space}")
action_space = behavior_spec.action_spec.continuous_size
action_bound = 100000 #normalizes action bounds to [-100000, 100000]

# Create an instance of the custom network
agent = DDPG(actuator_obs_space, tuple(visual_obs_space), action_space, action_bound)

num_episodes = 100
for episode in range(num_episodes):
    print(f"Starting episode:{episode}")
    
    env.reset()
    decision_steps_wooden, terminal_steps_wooden = env.get_steps(wooden_knight)
    decision_steps_grass, terminal_steps_grass = env.get_steps(grass_knight)
    first = True
    
    while len(decision_steps_wooden) > 0 and len(decision_steps_grass) > 0:
         # Generate actions if we have decision steps to go
        for agent_id in decision_steps_wooden.agent_id:
            obs = torch.FloatTensor(decision_steps_wooden.obs[0][agent_id]).to(device)
            visual_obs = torch.FloatTensor(decision_steps_wooden.obs[1][agent_id]).to(device)

            action = agent.act(visual_obs, obs)
            action_tuple = ActionTuple(continuous=np.expand_dims(action, axis=0))
            env.set_action_for_agent(wooden_knight, agent_id, action_tuple)

            if first:
                prev_obs = torch.zeros((len(decision_steps_wooden) * 2, actuator_obs_space)).to(device)
                prev_visual_obs = torch.zeros((len(decision_steps_wooden) * 2, visual_obs_space[0], visual_obs_space[1], visual_obs_space[2])).to(device)
                prev_action = torch.zeros((len(decision_steps_wooden) * 2, action_space)).to(device)
                first = False

            print(f"shapes: o: {prev_obs.shape}, vo: {prev_visual_obs.shape}, a: {prev_action.shape}")
            
            agent.memorize((prev_visual_obs[agent_id].cpu().numpy(), prev_obs[agent_id].cpu().numpy()), 
                           prev_action[agent_id].cpu().numpy(), 0, 
                           (visual_obs.cpu().numpy(), obs.cpu().numpy()), False)
            print("action obs visobs shapes:")
            print(action.shape)
            print(obs.shape)
            print(visual_obs.shape)
            prev_action[agent_id] = torch.FloatTensor(action).to(device)
            prev_obs[agent_id] = obs
            prev_visual_obs[agent_id] = visual_obs
            
        for agent_id in decision_steps_grass.agent_id:
            obs = torch.FloatTensor(decision_steps_grass.obs[1][agent_id - 6]).to(device)
            visual_obs = torch.FloatTensor(decision_steps_grass.obs[0][agent_id - 6]).to(device)

            action = agent.act(visual_obs, obs)
            action_tuple = ActionTuple(continuous=np.expand_dims(action, axis=0))
            env.set_action_for_agent(grass_knight, agent_id, action_tuple)

            if first:
                prev_obs = torch.zeros((len(decision_steps_wooden) * 2, actuator_obs_space)).to(device)
                prev_visual_obs = torch.zeros((len(decision_steps_wooden) * 2, visual_obs_space[0], visual_obs_space[1], visual_obs_space[2])).to(device)
                prev_action = torch.zeros((len(decision_steps_wooden) * 2, action_space)).to(device)
                first = False

            agent.memorize((prev_visual_obs[agent_id - 6].cpu().numpy(), prev_obs[agent_id - 6].cpu().numpy()), 
                           prev_action[agent_id - 6].cpu().numpy(), 0, 
                           (visual_obs.cpu().numpy(), obs.cpu().numpy()), False)
            prev_action[agent_id] = torch.FloatTensor(action).to(device)
            print(f"shapes: o: {prev_obs.shape}, vo: {prev_visual_obs.shape}, a: {prev_action.shape}")
            prev_obs[agent_id] = obs
            prev_visual_obs[agent_id] = visual_obs

        # Step the environment
        for _ in range(50):
            env.step()

        decision_steps_wooden, terminal_steps_wooden = env.get_steps(wooden_knight)
        decision_steps_grass, terminal_steps_grass = env.get_steps(grass_knight)

        # Collect rewards and train the agent
        if len(terminal_steps_wooden) > 0 or len(terminal_steps_grass) > 0:
            for agent_id in terminal_steps_wooden.agent_id:
                reward = terminal_steps_wooden[agent_id].reward
                done = terminal_steps_wooden[agent_id].interrupted
                next_obs = torch.zeros_like(obs)  # Terminal state has no next state
                next_visual_obs = torch.zeros_like(visual_obs)
                agent.memorize((visual_obs.cpu().numpy(), obs.cpu().numpy()), 
                               action, reward, 
                               (next_visual_obs.cpu().numpy(), next_obs.cpu().numpy()), done)

            for agent_id in terminal_steps_grass.agent_id:
                reward = terminal_steps_grass[agent_id].reward
                done = terminal_steps_grass[agent_id].interrupted
                next_obs = torch.zeros_like(obs)  # Terminal state has no next state
                next_visual_obs = torch.zeros_like(visual_obs)
                agent.memorize((visual_obs.cpu().numpy(), obs.cpu().numpy()), 
                               action, reward, 
                               (next_visual_obs.cpu().numpy(), next_obs.cpu().numpy()), done)

            break

        agent.replay()

env.close()

import os
import time
if not os.path.exists('saved_ddpg_models'):
    os.makedirs('saved_ddpg_models')
folder = "saved_ddpg_models/"
model_name = "ddpg_model" + time.strftime("%Y%m%d-%H%M%S") + ".pth"
agent.save(folder + "/" + model_name)