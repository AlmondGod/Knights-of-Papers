import torch
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

# Import the DDPG and SAC classes from your existing files
from src.DDPG.DDPG import DDPG 
from src.SAC.SAC import SAC    

# Load the saved models
ddpg_checkpoint = torch.load('saved_models/ddpg.pth', map_location=torch.device('cpu'))
sac_checkpoint = torch.load('saved_SAC_models/full_state_sac_model.pth', map_location=torch.device('cpu'))

# Initialize the agents
# You might need to adjust these initializations based on your actual class structures
ddpg_agent = DDPG(ddpg_checkpoint['model_state_dict'])
sac_agent = SAC(sac_checkpoint['actor_state_dict'], sac_checkpoint['critic_state_dict'])

# Connect to the Unity environment
env = UnityEnvironment(file_name="environment-builds/macos/build8.app", seed=1, side_channels=[], no_graphics=False)
env.reset()

# Get behavior names
behavior_names = list(env.behavior_specs.keys())
wooden_knight = behavior_names[0]
grass_knight = behavior_names[1]

num_episodes = 10
for episode in range(num_episodes):
    print(f"Starting episode: {episode + 1}")
    env.reset()
    decision_steps_wooden, terminal_steps_wooden = env.get_steps(wooden_knight)
    decision_steps_grass, terminal_steps_grass = env.get_steps(grass_knight)

    episode_reward = 0
    step = 0
    while len(decision_steps_wooden) > 0 and len(decision_steps_grass) > 0:
        # DDPG for wooden knight
        for agent_id in decision_steps_wooden.agent_id:
            obs = decision_steps_wooden.obs[0][agent_id]
            visual_obs = decision_steps_wooden.obs[1][agent_id]
            
            action = ddpg_agent.act(visual_obs, obs)
            action_tuple = ActionTuple(continuous=np.expand_dims(action, axis=0))
            env.set_action_for_agent(wooden_knight, agent_id, action_tuple)

        # SAC for grass knight
        for agent_id in decision_steps_grass.agent_id:
            obs = decision_steps_grass.obs[1][agent_id - 6]
            visual_obs = decision_steps_grass.obs[0][agent_id - 6]
            
            action = sac_agent.select_action(visual_obs, obs)
            action_tuple = ActionTuple(continuous=np.expand_dims(action, axis=0))
            env.set_action_for_agent(grass_knight, agent_id, action_tuple)

        # Step the environment
        env.step()
        step += 1

        decision_steps_wooden, terminal_steps_wooden = env.get_steps(wooden_knight)
        decision_steps_grass, terminal_steps_grass = env.get_steps(grass_knight)

        # Collect rewards
        for agent_id in decision_steps_wooden.agent_id:
            episode_reward += decision_steps_wooden[agent_id].reward
        for agent_id in decision_steps_grass.agent_id:
            episode_reward += decision_steps_grass[agent_id].reward

        # Check for episode termination
        if len(terminal_steps_wooden) > 0 or len(terminal_steps_grass) > 0:
            break

    print(f"Episode {episode + 1} finished. Total steps: {step}, Total reward: {episode_reward}")

env.close()
print("Evaluation complete.")