from mlagents_envs.environment import UnityEnvironment
import numpy as np
from mlagents_envs.base_env import ActionTuple
import torch
import torch.nn as nn
import torch.optim as optim
import random

# This is a non-blocking call that only loads the environment.
print("connecting to env...")
env = UnityEnvironment(file_name="build2/Knights of Papers.exe", seed=1, side_channels=[])
# Start interacting with the environment.
env.reset()
knight_one_names = env.behavior_specs.keys()
    
class Actor(nn.Module):
    def __init__(self, state_size, action_size, action_bound, visual_input_shape):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv_output_size = self._get_conv_output_size(visual_input_shape)

        self.fc1 = nn.Linear(self.conv_output_size + state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.action_bound = action_bound

    def _get_conv_output_size(self, shape):
        x = torch.rand(1, *shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.size()))

    def forward(self, visual_input, vector_input):
        x1 = torch.relu(self.conv1(visual_input))
        x1 = torch.relu(self.conv2(x1))
        x1 = torch.relu(self.conv3(x1))
        x1 = x1.view(x1.size(0), -1)

        x = torch.cat((x1, vector_input), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.action_bound
        return x

class Critic(nn.Module):
    def __init__(self, state_size, action_size, visual_input_shape):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv_output_size = self._get_conv_output_size(visual_input_shape)

        self.fc1 = nn.Linear(self.conv_output_size + state_size + action_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def _get_conv_output_size(self, shape):
        x = torch.rand(1, *shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.size()))

    def forward(self, visual_input, vector_input, action):
        x1 = torch.relu(self.conv1(visual_input))
        x1 = torch.relu(self.conv2(x1))
        x1 = torch.relu(self.conv3(x1))
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
        
        self.actor = Actor(state_size, action_size, action_bound, visual_input_shape)
        self.target_actor = Actor(state_size, action_size, action_bound, visual_input_shape)
        self.critic = Critic(state_size, action_size, visual_input_shape)
        self.target_critic = Critic(state_size, action_size, visual_input_shape)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        self.update_target_network(self.target_actor, self.actor, 1.0)
        self.update_target_network(self.target_critic, self.critic, 1.0)
    
    def update_target_network(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
    
    def memorize(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))
    
    def act(self, visual_input, vector_input):
        visual_input = torch.FloatTensor(visual_input).unsqueeze(0)
        vector_input = torch.FloatTensor(vector_input).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(visual_input, vector_input).cpu().data.numpy().flatten()
        self.actor.train()
        return action + np.random.normal(0, 0.1, size=self.action_size)  # Adding noise for exploration
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = self.memory.sample(self.batch_size)
        
        visual_inputs = torch.FloatTensor(np.array([m[0][0] for m in minibatch]))
        vector_inputs = torch.FloatTensor(np.array([m[0][1] for m in minibatch]))
        actions = torch.FloatTensor(np.array([m[1] for m in minibatch]))
        rewards = torch.FloatTensor(np.array([m[2] for m in minibatch])).unsqueeze(1)
        next_visual_inputs = torch.FloatTensor(np.array([m[3][0] for m in minibatch]))
        next_vector_inputs = torch.FloatTensor(np.array([m[3][1] for m in minibatch]))
        dones = torch.FloatTensor(np.array([m[4] for m in minibatch])).unsqueeze(1)
        
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

#we use ringbuffer instead of deque on the recommendation of ghliu ddpg impl
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
visual_obs_space = behavior_spec.observation_specs[1][0]
action_space = behavior_spec.action_spec.continuous_size
action_bound = 1.0 #normalizes action bounds to [-1, 1]

# Create an instance of the custom network
agent = DDPG(actuator_obs_space, visual_obs_space, action_space, action_bound)

num_episodes = 100
for episode in range(num_episodes):
    print(f"Starting episode:{episode}")
    
    env.reset()
    decision_steps_wooden, terminal_steps_wooden = env.get_steps(wooden_knight)
    decision_steps_grass, terminal_steps_grass = env.get_steps(grass_knight)

    while len(decision_steps_wooden) > 0 and len(decision_steps_grass) > 0:
         # Generate actions if we have decision steps to go
        for agent_id in decision_steps_wooden.agent_id:
            obs = decision_steps_wooden.obs[0][agent_id]
            visual_obs = decision_steps_wooden.obs[1][agent_id]

            action = agent.act(visual_obs, obs)
            action_tuple = ActionTuple(continuous=np.expand_dims(action, axis=0))
            env.set_action_for_agent(wooden_knight, agent_id, action_tuple)

        for agent_id in decision_steps_grass.agent_id:
            obs = decision_steps_grass.obs[1][agent_id - 6]
            visual_obs = decision_steps_grass.obs[0][agent_id - 6]

            action = agent.act(visual_obs, obs)
            action_tuple = ActionTuple(continuous=np.expand_dims(action, axis=0))
            env.set_action_for_agent(grass_knight, agent_id, action_tuple)

        # Step the environment
        env.step()

        decision_steps_wooden, terminal_steps_wooden = env.get_steps(wooden_knight)
        decision_steps_grass, terminal_steps_grass = env.get_steps(grass_knight)

        # Collect rewards and train the agent
        if len(terminal_steps_wooden) > 0 or len(terminal_steps_grass) > 0:
            for agent_id in terminal_steps_wooden.agent_id:
                reward = terminal_steps_wooden[agent_id].reward
                done = terminal_steps_wooden[agent_id].interrupted
                next_obs = np.zeros_like(obs)  # Terminal state has no next state
                next_visual_obs = np.zeros_like(visual_obs)
                agent.memorize((visual_obs, obs), action, reward, (next_visual_obs, next_obs), done)

            for agent_id in terminal_steps_grass.agent_id:
                reward = terminal_steps_grass[agent_id].reward
                done = terminal_steps_grass[agent_id].interrupted
                next_obs = np.zeros_like(obs)  # Terminal state has no next state
                next_visual_obs = np.zeros_like(visual_obs)
                agent.memorize((visual_obs, obs), action, reward, (next_visual_obs, next_obs), done)

            break

        agent.replay()

env.close()

torch.save({
    'episodes': num_episodes,
    'model_state_dict': agent.state_dict()
    # any other metrics or variables you need
}, 'saved_models/ddpg.pth')