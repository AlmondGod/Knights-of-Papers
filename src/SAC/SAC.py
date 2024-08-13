import os
import sys
from io import StringIO
from mlagents_envs.environment import UnityEnvironment

class SilentUnityEnvironment(UnityEnvironment):
    def __init__(self, *args, **kwargs):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        super().__init__(*args, **kwargs)
        sys.stdout = self._stdout
        sys.stderr = self._stderr

    def step(self):
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        result = super().step()
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        return result

# Use this instead of UnityEnvironment
env = SilentUnityEnvironment(file_name="environment-builds/macos/build8.app", seed=1, side_channels=[], no_graphics=True)
from mlagents_envs.environment import UnityEnvironment
import numpy as np
from mlagents_envs.base_env import ActionTuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# This is a non-blocking call that only loads the environment.
# env = UnityEnvironment(file_name="build8.app", seed=1, side_channels=[], no_graphics=True)
print("env connected")
# for behavior_name in env.behavior_specs:
#     spec = env.behavior_specs[behavior_name]
#     print(f"Behavior {behavior_name}:")
#     for i, obs_spec in enumerate(spec.observation_specs):
#         print(f"  Observation {i} shape: {obs_spec.shape}")
# Start interacting with the environment.
env.reset()
knight_one_names = env.behavior_specs.keys()

class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action, visual_input_shape):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv_output_size = self._get_conv_output_size(visual_input_shape)

        self.fc1 = nn.Linear(self.conv_output_size + state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_size)
        self.log_std = nn.Linear(256, action_size)
        self.max_action = max_action

    def _get_conv_output_size(self, shape):
        x = torch.rand(1, 3, *shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return int(np.prod(x.size()))

    def forward(self, vector_input, visual_input):
        x1 = torch.relu(self.conv1(visual_input))
        x1 = torch.relu(self.conv2(x1))
        x1 = x1.view(x1.size(0), -1)

        x = torch.cat((x1, vector_input), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # Clamping for numerical stability
        return mean, log_std #gives us prob dist for stochastic action selection
    
    def sample(self, vector_input, visual_input):
        mean, log_std = self.forward(vector_input, visual_input)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.sample() #sample from the dist
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)  # Enforcing action bounds
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class DualCritic(nn.Module):
    def __init__(self, state_size, action_size, visual_input_shape):
        super(DualCritic, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv_output_size = self._get_conv_output_size(visual_input_shape)

        self.fc1 = nn.Linear(self.conv_output_size + state_size + action_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.conv3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.fc4 = nn.Linear(self.conv_output_size + state_size + action_size, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def _get_conv_output_size(self, shape):
        x = torch.rand(1, 3, *shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return int(np.prod(x.size()))

    def forward(self, vector_input, visual_input, action):
        x1 = torch.relu(self.conv1(visual_input))
        x1 = torch.relu(self.conv2(x1))
        x1 = x1.view(x1.size(0), -1)

        xa = torch.cat((x1, vector_input, action), dim=1)

        q1 = torch.relu(self.fc1(xa))
        q1 = torch.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        x2 = torch.relu(self.conv3(visual_input))
        x2 = torch.relu(self.conv4(x2))
        x2 = x2.view(x2.size(0), -1)
        xb = torch.cat((x2, vector_input, action), dim=1)

        q2 = torch.relu(self.fc4(xb))
        q2 = torch.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2
    
class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
    
    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, vision, actions, next_states, next_vision, rewards, dones = [], [], [], [], [], [], []
        
        for i in ind:
            s, v, a, ns, nv, r, d = self.storage[i]
            states.append(np.array(s))
            vision.append(np.array(v))
            actions.append(np.array(a))
            next_states.append(np.array(ns))
            next_vision.append(np.array(nv))
            rewards.append(np.array(r))
            dones.append(np.array(d))

        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(vision)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(next_vision)),
            torch.FloatTensor(np.array(rewards).reshape(-1, 1)),
            torch.FloatTensor(np.array(dones).reshape(-1, 1))
        )

class SAC:
    def __init__(self, state_size, visual_input_shape, max_action, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.visual_shape = visual_input_shape
        self.tau = 0.005 #soft update coefficient (how much of og network the target takes)
        self.discount = 0.99 #how much the agent values future rewards (smalles prioritizes immediate rewards)
        self.alpha = 0.2 #entropy coefficient (how much the agent values exploration)
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        
        self.actor = Actor(state_size, action_size, max_action, visual_input_shape)
        self.dualcritics = DualCritic(state_size, action_size, visual_input_shape)
        self.actor_loss = 0

        self.target_critics = DualCritic(state_size, action_size, visual_input_shape)
        self.target_critics.load_state_dict(self.dualcritics.state_dict()) #sets target and regular to have same params at the start
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(list(self.dualcritics.parameters()), lr=self.critic_lr)
    
    def select_action(self, visual_input, vector_input):
        visual_input = torch.FloatTensor(visual_input).unsqueeze(0)
        vector_input = torch.FloatTensor(vector_input).unsqueeze(0)
        action, _ = self.actor.sample(vector_input, visual_input)
        return action.detach()

    def train(self, replay_buffer, batch_size = 256):
        state, vision, action, next_state, next_vision, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_state, next_vision)
            q1_next, q2_next = self.target_critics(next_state, next_vision, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            expected_q = reward + (not_done * self.discount * q_next)
        
        current_q1, current_q2 = self.dualcritics(state, vision, action)
        critic_loss = nn.functional.mse_loss(current_q1, expected_q) + nn.functional.mse_loss(current_q2, expected_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi = self.actor.sample(state, vision)
        q1_pi, q2_pi = self.dualcritics(state, vision, np.squeeze(pi, axis=1))
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_pi - min_q_pi).mean()

        self.actor_loss = actor_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.dualcritics.parameters(), self.target_critics.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


wooden_knight = list(env.behavior_specs.keys())[0]
grass_knight = list(env.behavior_specs.keys())[1]
behavior_spec = env.behavior_specs[wooden_knight]

# Print observation shapes
print(f"Vector observation shape: {behavior_spec.observation_specs[0].shape}")
print(f"Visual observation shape: {behavior_spec.observation_specs[1].shape}")
print(f"Action space size: {behavior_spec.action_spec.continuous_size}")

# Extract observation and action space sizes
actuator_obs_space = behavior_spec.observation_specs[0][0][0]
visual_obs_space = behavior_spec.observation_specs[1].shape[1:]
action_space = behavior_spec.action_spec.continuous_size

experience_replay = ReplayBuffer()
max_action = 10000
# Create an instance of the custom network

vector_obs_size = 154
visual_obs_shape = (10, 10)
action_size = 25

model = SAC(vector_obs_size, visual_obs_shape, max_action, action_size)
checkpoint = torch.load('saved_SAC_models/sac_model20240803-164158.pth', map_location=torch.device('cpu'))

if isinstance(checkpoint, SAC):
    # If the saved object is an SAC instance
    model = checkpoint
    print("loaded model")

actor_losses = []

num_episodes = 1000
# Main loop to interact with the environment
#every action decision, sample the prestate (obs + vision), action, reward, poststate (obs + vision), and done from the replay buffer
for episode in range(num_episodes):
    step = 0
    print(f"Starting episode:{episode}")
    env.reset()
    decision_steps_wooden, terminal_steps_wooden = env.get_steps(wooden_knight)
    decision_steps_grass, terminal_steps_grass = env.get_steps(grass_knight)

    while len(decision_steps_wooden) > 0 and len(decision_steps_grass) > 0:
        #generate actions if we have decision steps to go
        for agent_id in decision_steps_wooden.agent_id:
            obs = torch.tensor(decision_steps_wooden.obs[0][agent_id], dtype=torch.float32)
            visual_obs = torch.tensor(decision_steps_wooden.obs[1][agent_id], dtype=torch.float32).permute(2, 0, 1)  # Change to (C, H, W)

            with torch.no_grad():
                action = np.expand_dims(model.select_action(visual_obs, obs).cpu().numpy()[0] * max_action, axis=0)
                
            action_tuple = ActionTuple(continuous=action)
            env.set_action_for_agent(wooden_knight, agent_id, action_tuple)

            reward = decision_steps_wooden.reward[agent_id]
            done = len(terminal_steps_wooden) > 0 or len(terminal_steps_grass) > 0
            next_obs = torch.tensor(decision_steps_wooden.obs[0][agent_id], dtype=torch.float32)
            next_visual_obs = torch.tensor(decision_steps_wooden.obs[1][agent_id], dtype= torch.float32).permute(2, 0, 1)
            experience_replay.add((obs, visual_obs, action[0], next_obs, next_visual_obs, reward, done))

        for agent_id in decision_steps_grass.agent_id:  
            obs = torch.tensor(decision_steps_grass.obs[1][agent_id - 6], dtype=torch.float32)
            visual_obs = torch.tensor(decision_steps_grass.obs[0][agent_id - 6], dtype= torch.float32).permute(2, 0, 1)

            with torch.no_grad():
                action = np.expand_dims(model.select_action(visual_obs, obs).cpu().numpy()[0] * max_action, axis=0)

            action_tuple = ActionTuple(continuous=action)
            env.set_action_for_agent(grass_knight, agent_id, action_tuple)
            
            reward = decision_steps_grass.reward[agent_id - 6]
            if (agent_id == 8):
                print(reward)
            next_obs = torch.tensor(decision_steps_wooden.obs[0][agent_id - 6], dtype=torch.float32)
            next_visual_obs = torch.tensor(decision_steps_wooden.obs[1][agent_id - 6], dtype= torch.float32).permute(2, 0, 1)
            done = len(terminal_steps_wooden) > 0 or len(terminal_steps_grass) > 0
            experience_replay.add((obs, visual_obs, action[0], next_obs, next_visual_obs, reward, done))

        # Step the environment
        for _ in range(40):
            env.step()
        step += 1

        if step % 100 == 0:
            model.train(experience_replay, batch_size=256)

        decision_steps_wooden, terminal_steps_wooden = env.get_steps(wooden_knight)
        decision_steps_grass, terminal_steps_grass = env.get_steps(grass_knight)
        
        #upon episode end, compute loss and backpropagate
        if len(terminal_steps_wooden) > 0 or len(terminal_steps_grass) > 0: 
            break
    
    print(f"current loss: {model.actor_loss.item()}")
    actor_losses.append(model.actor_loss.item())

print("finished")

import time

folder = "saved_SAC_models/"
model_name = "sac_model" + time.strftime("%Y%m%d-%H%M%S") + ".pth"
torch.save(model, folder + model_name)
torch.save({
    'actor_state_dict': model.actor.state_dict(),
    'critic_state_dict': model.dualcritics.state_dict(),
    'target_critic_state_dict': model.target_critics.state_dict(),
    'actor_optimizer_state_dict': model.actor_optimizer.state_dict(),
    'critic_optimizer_state_dict': model.critic_optimizer.state_dict(),
    'episode': episode,  # Assuming you're tracking episode number
    'hyperparameters': {
        'tau': model.tau,
        'discount': model.discount,
        'alpha': model.alpha,
        'actor_lr': model.actor_lr,
        'critic_lr': model.critic_lr
    }
}, folder + "full_state_" + model_name)

with(open(folder + "actor_losses_sac_model" + time.strftime("%Y%m%d-%H%M%S") + ".txt", "w")) as f:
    for loss in actor_losses:
        f.write(str(loss) + "\n")

print("model saved")