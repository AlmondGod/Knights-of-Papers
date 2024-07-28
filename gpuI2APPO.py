import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Environment setup (same as before)
env = UnityEnvironment(file_name="gpubuild.x86_64", seed=1, side_channels=[], no_graphics=True)
print("env connected")
env.reset()
wooden_knight = list(env.behavior_specs.keys())[0]
grass_knight = list(env.behavior_specs.keys())[1]
behavior_spec = env.behavior_specs[wooden_knight]

print(f"Vector observation shape: {behavior_spec.observation_specs[0].shape}")
print(f"Visual observation shape: {behavior_spec.observation_specs[1].shape}")
print(f"Action space size: {behavior_spec.action_spec.continuous_size}")

vector_obs_size = 154
visual_obs_shape = (10, 10)
action_size = 25

# I2A Components
class EnvironmentModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(EnvironmentModel, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, state_size)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        next_state = self.fc3(x)
        return next_state

class RolloutEncoder(nn.Module):
    def __init__(self, state_size, hidden_size=256):
        super(RolloutEncoder, self).__init__()
        self.lstm = nn.LSTM(state_size, hidden_size, batch_first=True)
        
    def forward(self, rollouts):
        _, (h_n, _) = self.lstm(rollouts)
        return h_n.squeeze(0)

class I2APolicy(nn.Module):
    def __init__(self, state_size, action_size, rollout_size, hidden_size=256):
        super(I2APolicy, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv_output_size = self._get_conv_output_size(visual_obs_shape)
        
        self.fc1 = nn.Linear(self.conv_output_size + state_size + rollout_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)
        
    def _get_conv_output_size(self, shape):
        x = torch.rand(1, 3, *shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return int(np.prod(x.size()))
    
    def forward(self, vector_input, visual_input, rollout_encoding):
        x1 = F.relu(self.conv1(visual_input))
        x1 = F.relu(self.conv2(x1))
        x1 = x1.view(x1.size(0), -1)
        
        x = torch.cat([x1, vector_input, rollout_encoding], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        action_probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        
        return action_probs, value

class I2A(nn.Module):
    def __init__(self, state_size, action_size, rollout_size=5, hidden_size=256):
        super(I2A, self).__init__()
        self.env_model = EnvironmentModel(state_size, action_size).to(device)
        self.rollout_encoder = RolloutEncoder(state_size).to(device)
        self.policy = I2APolicy(state_size, action_size, hidden_size).to(device)
        self.rollout_size = rollout_size
        
    def imagine(self, state, action):
        rollouts = []
        current_state = state
        
        for _ in range(self.rollout_size):
            next_state = self.env_model(current_state, action)
            rollouts.append(next_state)
            current_state = next_state
        
        return torch.stack(rollouts, dim=1)
    
    def forward(self, vector_input, visual_input):
        batch_size = vector_input.size(0)
        imagined_states = self.imagine(vector_input, torch.zeros(batch_size, action_size).to(device))
        rollout_encoding = self.rollout_encoder(imagined_states)
        return self.policy(vector_input, visual_input, rollout_encoding)

# PPO Components
class PPO:
    def __init__(self, state_size, action_size, rollout_size=5, hidden_size=256, lr=3e-4, gamma=0.99, epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.i2a = I2A(state_size, action_size, rollout_size, hidden_size).to(device)
        self.optimizer = optim.Adam(self.i2a.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
    
    def select_action(self, vector_input, visual_input):
        vector_input = torch.FloatTensor(vector_input).unsqueeze(0).to(device)
        visual_input = torch.FloatTensor(visual_input).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_probs, _ = self.i2a(vector_input, visual_input)
        
        action_dist = Normal(action_probs, torch.ones_like(action_probs) * 0.1)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(dim=-1)
        
        return action.cpu().numpy().flatten(), log_prob.cpu().item()
    
    def update(self, states, visual_states, actions, old_log_probs, rewards, dones):
        states = torch.FloatTensor(states).to(device)
        visual_states = torch.FloatTensor(visual_states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        action_probs, values = self.i2a(states, visual_states)
        action_dist = Normal(action_probs, torch.ones_like(action_probs) * 0.1)
        new_log_probs = action_dist.log_prob(actions).sum(dim=-1)
        
        advantages = rewards - values.detach()
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(values, rewards)
        entropy = action_dist.entropy().mean()
        
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# Training loop
ppo = PPO(vector_obs_size, action_size)
num_episodes = 2500

for episode in range(num_episodes):
    print(f"Starting episode: {episode}")
    env.reset()
    decision_steps_wooden, terminal_steps_wooden = env.get_steps(wooden_knight)
    decision_steps_grass, terminal_steps_grass = env.get_steps(grass_knight)
    
    episode_states, episode_visual_states, episode_actions, episode_log_probs, episode_rewards, episode_dones = [], [], [], [], [], []
    
    while len(decision_steps_wooden) > 0 and len(decision_steps_grass) > 0:
        for agent_id in decision_steps_wooden.agent_id:
            obs = decision_steps_wooden.obs[0][agent_id]
            visual_obs = decision_steps_wooden.obs[1][agent_id].transpose(2, 0, 1)  # Change to (C, H, W)
            
            action, log_prob = ppo.select_action(obs, visual_obs)
            action_tuple = ActionTuple(continuous=np.expand_dims(action, axis=0))
            env.set_action_for_agent(wooden_knight, agent_id, action_tuple)
            
            reward = decision_steps_wooden.reward[agent_id]
            done = len(terminal_steps_wooden) > 0 or len(terminal_steps_grass) > 0
            
            episode_states.append(obs)
            episode_visual_states.append(visual_obs)
            episode_actions.append(action)
            episode_log_probs.append(log_prob)
            episode_rewards.append(reward)
            episode_dones.append(done)
        
        # Similar process for grass_knight agents...
        
        env.step()
        decision_steps_wooden, terminal_steps_wooden = env.get_steps(wooden_knight)
        decision_steps_grass, terminal_steps_grass = env.get_steps(grass_knight)
        
        if len(terminal_steps_wooden) > 0 or len(terminal_steps_grass) > 0:
            break
    
    # Update PPO
    loss = ppo.update(
        np.array(episode_states),
        np.array(episode_visual_states),
        np.array(episode_actions),
        np.array(episode_log_probs),
        np.array(episode_rewards),
        np.array(episode_dones)
    )
    
    print(f"Episode {episode} completed. Loss: {loss}")

# Save the model
torch.save(ppo.i2a.state_dict(), "i2a_ppo_model.pth")
print("Model saved")