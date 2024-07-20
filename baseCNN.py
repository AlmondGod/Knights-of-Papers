from mlagents_envs.environment import UnityEnvironment
import numpy as np
from mlagents_envs.base_env import ActionTuple
import torch
import torch.nn as nn
import torch.optim as optim

# This is a non-blocking call that only loads the environment.
print("connecting to env...")
env = UnityEnvironment(file_name="build4.app", seed=1, side_channels=[])
# Start interacting with the environment.
env.reset()
knight_one_names = env.behavior_specs.keys()
    
class CNN(nn.Module):
    def __init__(self, observation_space, action_space, visual_input_shape):
        super(CNN, self).__init__()
        
        # Define CNN for visual input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)  # Single channel input
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        
        # Flattened size after convolutions
        conv_output_size = self._get_conv_output_size(visual_input_shape)

        self.fc1 = nn.Linear(conv_output_size + observation_space, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space)

    def _get_conv_output_size(self, shape):
        x = torch.rand(1, *shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.size()))

    def forward(self, visual_input, vector_input):
        # Process visual input
        x1 = torch.relu(self.conv1(visual_input))
        x1 = torch.relu(self.conv2(x1))
        x1 = torch.relu(self.conv3(x1))
        x1 = x1.view(x1.size(0), -1).flatten()  # Flatten the conv output
        
        # Concatenate visual and vector inputs
        x = torch.cat((x1, vector_input), dim=0)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

wooden_knight = list(env.behavior_specs.keys())[0]
grass_knight = list(env.behavior_specs.keys())[1]

behavior_spec = env.behavior_specs[wooden_knight]

# Extract observation and action space sizes
actuator_obs_space = behavior_spec.observation_specs[0][0][0]
visual_obs_space = behavior_spec.observation_specs[1][0]
action_space = behavior_spec.action_spec.continuous_size

# Create an instance of the custom network
model = CNN(actuator_obs_space, action_space, visual_obs_space)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_episodes = 100
# Main loop to interact with the environment
for episode in range(num_episodes):
    print(f"Starting episode:{episode}")
    env.reset()
    decision_steps_wooden, terminal_steps_wooden = env.get_steps(wooden_knight)
    decision_steps_grass, terminal_steps_grass = env.get_steps(grass_knight)

    while len(decision_steps_wooden) > 0 and len(decision_steps_grass) > 0:
        #generate actions if we have decision steps to go
        for agent_id in decision_steps_wooden.agent_id:
            obs = torch.tensor(decision_steps_wooden.obs[0][agent_id], dtype=torch.float32)
            visual_obs = torch.tensor(decision_steps_wooden.obs[1][agent_id], dtype= torch.float32)

            with torch.no_grad():
                action = np.expand_dims(model(visual_obs, obs).numpy(), axis=0)
                
            action_tuple = ActionTuple(continuous=action)
            env.set_action_for_agent(wooden_knight, agent_id, action_tuple)

        for agent_id in decision_steps_grass.agent_id:  
            obs = torch.tensor(decision_steps_grass.obs[1][agent_id - 6], dtype=torch.float32)
            visual_obs = torch.tensor(decision_steps_grass.obs[0][agent_id - 6], dtype= torch.float32)

            with torch.no_grad():
                action = np.expand_dims(model(visual_obs, obs).numpy(), axis=0)
            
            action_tuple = ActionTuple(continuous=action)
            env.set_action_for_agent(grass_knight, agent_id, action_tuple)

        # Step the environment
        env.step()

        decision_steps_wooden, terminal_steps_wooden = env.get_steps(wooden_knight)
        decision_steps_grass, terminal_steps_grass = env.get_steps(grass_knight)
        
        #upon episode end, compute loss and backpropagate
        if len(terminal_steps_wooden) > 0 or len(terminal_steps_grass) > 0: 
            for reward in terminal_steps_wooden.reward:
                # Compute the loss and backpropagate
                optimizer.zero_grad()
                loss = torch.tensor(-reward, dtype=torch.float32, requires_grad=True) # Simple negative reward as loss
                print(f"Loss (Wooden Knight): {loss}")  
                loss.backward()
                optimizer.step()

            for reward in terminal_steps_grass.reward:
                optimizer.zero_grad()
                loss = torch.tensor(-reward, dtype=torch.float32, requires_grad=True)  # Simple negative reward as loss
                print(f"Loss (Grass Knight): {loss}")  
                loss.backward()
                optimizer.step()
            
            break

env.close()

torch.save({
    'episodes': num_episodes,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    # any other metrics or variables you need
}, 'saved_models/base_cnn.pth')