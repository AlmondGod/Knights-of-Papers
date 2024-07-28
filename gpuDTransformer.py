from mlagents_envs.environment import UnityEnvironment
import numpy as np
from mlagents_envs.base_env import ActionTuple
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import time

env = UnityEnvironment(file_name="gpubuild.x86_64", seed=1, side_channels=[], no_graphics=True)
env.reset()
print("yuh")
knight_one_names = env.behavior_specs.keys()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# #pseudocode: 
# #Algorithm 1 Decision Transformer Pseudocode (for continuous actions)
# # R, s, a, t: returns -to -go , states , actions , or timesteps
# # transformer : transformer with causal masking (GPT)
# # embed_s , embed_a , embed_R : linear embedding layers
# # embed_t : learned episode positional embedding
# # pred_a : linear action prediction layer
# # main model
# def DecisionTransformer (R , s , a , t ):
    # # compute embeddings for tokens
    # pos_embedding = embed_t ( t ) # per - timestep ( note : not per - token )
    # s_embedding = embed_s ( s ) + pos_embedding
    # a_embedding = embed_a ( a ) + pos_embedding
    # R_embedding = embed_R ( R ) + pos_embedding
    # # interleave tokens as (R_1 , s_1 , a_1 , ... , R_K , s_K )
    # input_embeds = stack ( R_embedding , s_embedding , a_embedding )
    # # use transformer to get hidden states
    # hidden_states = transformer ( input_embeds = input_embeds )
    # # select hidden states for action prediction tokens
    # a_hidden = unstack ( hidden_states ). actions
    # # predict action
    # return pred_a ( a_hidden )

# # training loop
# for (R , s , a , t ) in dataloader : # dims : ( batch_size , K, dim )
    # a_preds = DecisionTransformer (R , s , a , t )
    # loss = mean (( a_preds - a )**2) # L2 loss for continuous actions
    # optimizer . zero_grad (); loss . backward (); optimizer . step ()

# # evaluation loop
# target_return = 1 # for instance , expert - level return
# R , s , a , t , done = [ target_return ] , [ env . reset ()] , [] , [1] , False
# while not done : # autoregressive generation / sampling
    # # sample next action
    # action = DecisionTransformer (R , s , a , t )[ -1] # for cts actions
    # new_s , r , done , _ = env . step ( action )
    # # append new tokens to sequence
    # R = R + [ R [ -1] - r] # decrement returns -to -go with reward
    # s , a , t = s + [ new_s ] , a + [ action ] , t + [ len ( R )]
    # R , s , a , t = R [ - K :] , ... # only keep context length of K

#---------------------------------MODEL INITIALIZATION-------------------------------------
class CNN(nn.Module):
    def __init__(self, visual_input_shape, embed_dim):
        super(CNN, self).__init__()

        channels, height, width = visual_input_shape[2], visual_input_shape[0], visual_input_shape[1]
        print(channels, height, width)
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        
        conv_output_size = self._get_conv_output_size(visual_input_shape)

        self.fc1 = nn.Linear(conv_output_size, embed_dim)

    def _get_conv_output_size(self, shape):
        x = torch.rand(1, *shape)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        return int(np.prod(x.size()))
        # print(shape)
        # x = torch.rand(1, *shape)
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # return int(np.prod(x.size()))

    def forward(self, visual_input):
        # Process visual input
        batch_size, seq_len, channels, height, width = visual_input.shape
        print(f"visual input shape: {visual_input.shape}")
        
        visual_input = visual_input.view(batch_size * seq_len, channels, height, width)
        
        # Process visual input through convolutional layers
        x = nn.functional.relu(self.conv1(visual_input))
        x = nn.functional.relu(self.conv2(x))
        # x = nn.functional.relu(self.conv3(x))
        
        # Flatten the conv output
        x = x.view(x.size(0), -1)  # Flatten the conv output
        
        # Fully connected layer
        x = nn.functional.relu(self.fc1(x))
        
        # Reshape output to separate batch and sequence dimensions: [B*T, EmbedDim] -> [B, T, EmbedDim]
        x = x.view(batch_size, seq_len, -1)
        return x

#influenced by nikhilbarhate99 min-decision-transformer and decision transformer paper
#write this based on the pseudocode above
class MaskedCausalAttention(nn.Module): #This specifically is straight from min-decision-transformer, thank you to him!
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = nn.functional.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out

class Block(nn.Module):
    def __init__(self, sequence_len, embed_dim, n_heads, dropout):
        super(Block, self).__init__()
        self.sequence_len = sequence_len
        self.attention = MaskedCausalAttention(embed_dim, sequence_len, n_heads, dropout)
        # self.attention = nn.MultiheadAttention(embed_dim, n_heads, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        print(x.shape)
        x = x + self.attention(x)
        # x = self.attention(x, attn_mask = attn_mask, is_causal=True)[0] + x
        x = self.ln1(x)
        x = self.mlp(x) + x
        x = self.ln2(x)
        
        return x

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, visual_dim, action_dim, n_blocks, n_heads, embed_dim, context_len, dropout, max_timesteps):
        super(DecisionTransformer, self).__init__()
        self.state_dim = state_dim
        self.visual_dim = visual_dim
        self.action_dim = action_dim
        self.hidden_dim = embed_dim

        input_len = (4 * context_len) - 1 #each input is k of state, visual, action, rewards to go except kth has no action (we predict it)
        attention_blocks = [Block(input_len, embed_dim, n_heads, dropout) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*attention_blocks)

        #projection heads to convert inputs of varying sizes to embeddings
        self.ln = nn.LayerNorm(embed_dim) #layernorm for the concatenated embeds
        self.embed_states = nn.Linear(state_dim, embed_dim)
        self.embed_vision = CNN(visual_dim, embed_dim) #vision embedding created through CNN
        self.embed_actions = nn.Linear(action_dim, embed_dim)
        self.embed_returns_to_go = nn.Linear(1, embed_dim)
    
        self.embed_timesteps = nn.Embedding(max_timesteps, embed_dim) #we add timestep embedding to each corresponding
        
        #Action prediction head (we could predict states and returns to go but no need)
        self.predict_action = nn.Sequential(
            nn.Linear(embed_dim, action_dim),
            nn.Tanh()
            )
        
        self.transformer = self.transformer.to(device)
        self.ln = self.ln.to(device)
        self.embed_states = self.embed_states.to(device)
        self.embed_vision = self.embed_vision.to(device)
        self.embed_actions = self.embed_actions.to(device)
        self.embed_returns_to_go = self.embed_returns_to_go.to(device)
        self.embed_timesteps = self.embed_timesteps.to(device)
        self.predict_action = self.predict_action.to(device)

    def forward(self, returns_to_go, states, visions, actions, timestep):
        print(timestep.shape)
        print("Max timestep:", t.max().item())
        print("Min timestep:", t.min().item())

        returns_to_go = returns_to_go.to(device)
        states = states.to(device)
        visions = visions.to(device)
        actions = actions.to(device)
        timestep = timestep.to(device)
        
        time_embed = self.embed_timesteps(timestep)
        state_embed = self.embed_states(states) + time_embed
        action_embed = self.embed_actions(actions) + time_embed[:, :actions.shape[1]]
        print(f"action embed: {action_embed.shape}")
        returns_to_go_embed = self.embed_returns_to_go(returns_to_go) + time_embed
        #separate visual data multiply batch and sequence dimension
        vision_embed = self.embed_vision(visions) + time_embed

        # input_embeds = torch.cat((returns_to_go_embed[0], state_embed[0], visual_embed[0], action_embed[0]), dim=0)
        # for i in range(1, len(visual_embed)):
        #     input_embeds = torch.cat((input_embeds, torch.cat((returns_to_go_embed[i], state_embed[i], visual_embed[i], action_embed[i]), dim=0)), dim=0)
        
        # input_embeds = torch.cat((input_embeds, torch.cat((returns_to_go_embed[-1], state_embed[-1], visual_embed[-1]), dim=0)), dim=0)
        interleaved = []
        for i in range(returns_to_go_embed.shape[1]):
            interleaved.append(returns_to_go_embed[:, i, :].unsqueeze(1))
            interleaved.append(state_embed[:, i, :].unsqueeze(1))
            interleaved.append(vision_embed[:, i, :].unsqueeze(1))
            if i < returns_to_go_embed.shape[1]-1:  # Only add actions up to S-1
                interleaved.append(action_embed[:, i, :].unsqueeze(1))

        # Concatenate along the sequence dimension to form the full interleaved sequence
        interleaved_sequence = torch.cat(interleaved, dim=1)

        input_embeds = self.ln(interleaved_sequence) #layernorm

        print(input_embeds.shape)
        print(f"input_embeds: {input_embeds}")

        hidden_states = self.transformer(input_embeds)

        actions_pred = self.predict_action(hidden_states)
        
        return actions_pred


#---------------------------------DATA INITIALIZATION-------------------------------------
# Get the state and action sizes
wooden_knight = list(env.behavior_specs.keys())[0]
grass_knight = list(env.behavior_specs.keys())[1]
behavior_spec = env.behavior_specs[wooden_knight]

# Extract observation and action space sizes
actuator_obs_size = behavior_spec.observation_specs[0][0][0]
visual_obs_size = behavior_spec.observation_specs[1][0]
visual_obs_size = (visual_obs_size[2], visual_obs_size[0], visual_obs_size[1])
action_size = behavior_spec.action_spec.continuous_size

total_timesteps = 500
decision_interval = 5
env_data = []

grass_returns_to_go = []
grass_states = []
grass_visions = []
grass_actions = []
wood_returns_to_go = []
wood_states = []
wood_visions = []
wood_actions = []
timesteps = []
episode_ends = []

for episode in range(1):
    print(f"Starting episode:{episode}")
    env.reset()
    decision_steps_wooden, terminal_steps_wooden = env.get_steps(wooden_knight)
    decision_steps_grass, terminal_steps_grass = env.get_steps(grass_knight)

    wood_rewards = []
    grass_rewards = []
    while len(decision_steps_wooden) > 0 and len(decision_steps_grass) > 0:
        #generate actions if we have decision steps to go
        for agent_id in decision_steps_wooden.agent_id:
            obs = torch.tensor(decision_steps_wooden.obs[0][agent_id], dtype=torch.float32)
            visual_obs = torch.tensor(decision_steps_wooden.obs[1][agent_id], dtype= torch.float32)
            visual_obs = visual_obs.permute(2, 0, 1).unsqueeze(0)
            print(f"visobs shape {visual_obs.shape}")

            with torch.no_grad():
                action = np.random.rand(1, action_size)

            wood_states.append(decision_steps_wooden.obs[0][agent_id])
            wood_visions.append(decision_steps_wooden.obs[1][agent_id])
            wood_actions.append(action[0])
            wood_rewards.append(decision_steps_wooden.reward[agent_id])

            action_tuple = ActionTuple(continuous=action)
            env.set_action_for_agent(wooden_knight, agent_id, action_tuple)
            
        for agent_id in decision_steps_grass.agent_id:  
            obs = torch.tensor(decision_steps_grass.obs[1][agent_id - 6], dtype=torch.float32)
            visual_obs = torch.tensor(decision_steps_grass.obs[0][agent_id - 6], dtype= torch.float32)
            visual_obs = visual_obs.permute(2, 0, 1).unsqueeze(0)
                
            with torch.no_grad():
                action = np.random.rand(1, action_size)

            grass_states.append(decision_steps_grass.obs[1][agent_id - 6])
            grass_visions.append(decision_steps_grass.obs[0][agent_id - 6])
            grass_actions.append(action[0])
            grass_rewards.append(decision_steps_grass.reward[agent_id - 6])
                
            action_tuple = ActionTuple(continuous=action)
            env.set_action_for_agent(grass_knight, agent_id, action_tuple)

        # Step the environment
        for _ in range(decision_interval):
            env.step()

        decision_steps_wooden, terminal_steps_wooden = env.get_steps(wooden_knight)
        decision_steps_grass, terminal_steps_grass = env.get_steps(grass_knight)
        
        #upon episode end, compute loss and backpropagate
        if len(terminal_steps_wooden) > 0 or len(terminal_steps_grass) > 0: 
    
            both_timesteps = [i for i in range(len(wood_rewards))]

            #this should have entries which are the sum of all rewards at this index and after
            view=np.flip(wood_rewards, 0)
            np.cumsum(view, 0, out=view)
            wood_returns_to_go = wood_returns_to_go + wood_rewards
            
            view=np.flip(grass_rewards, 0)
            np.cumsum(view, 0, out=view)
            grass_returns_to_go = grass_returns_to_go + grass_rewards

            #each index this contains is the index at which we start new episode for sequence extraction
            episode_ends.append(len(wood_states)) 
            timesteps = timesteps + both_timesteps

            # wood_states=torch.stack(wood_states)
            # wood_visions=torch.stack(wood_visions)
            # wood_actions=torch.stack(wood_actions)
            # wood_returns_to_go=torch.tensor(wood_returns_to_go)
            # both_timesteps=torch.tensor(both_timesteps)
            # grass_states=torch.stack(grass_states)
            # grass_visions=torch.stack(grass_visions)
            # grass_actions=torch.stack(grass_actions)
            # grass_returns_to_go=torch.tensor(grass_returns_to_go)

            # #interleave tokens as (state, vision, action, reward, timestep)
            # tensorlist = [wood_states, wood_visions, wood_actions, wood_returns_to_go, both_timesteps]
            # length = wood_states.shape[0]
            # wood_data = [val for tup in zip(*tensorlist) for val in tup]
            # print(f"wood data shape: {len(wood_data)}")
            # tensorlist = [grass_states, grass_visions, grass_actions, grass_returns_to_go, both_timesteps]
            # length = grass_states.shape[0]
            # grass_data = [val for tup in zip(*tensorlist) for val in tup]
            
            # #add to env_data
            # env_data.append(wood_data)
            # env_data.append(grass_data)
            
            break

env.close()

print("data collection finished")

batch_size = 4
learning_rate = 1e-4
wt_decay = 1e-4

n_blocks = 3
n_heads = 2
embed_dim = 64
dropout = 0.1
K = 20
context_len = 4 * K 

#-----------------------------SEQUENCE INITIALIZATION--------------------------------
# #initialize dataloader to hold (R, s, v, a, t)  dims : ( batch_size , K, dim )
# #now we break env_data into batches of size batch_size * K
# #env_data has tensors each of which is an episode, each episode contains token tensors of state, vision, action, rewards to go, timestep
# #each data input is a sequence of length context_len of those tokens, and each batch has batch_size of these inputs
# #to maximize the use of our data, we want to make data inputs for all length-K subsequences of the episode
# #but batches should all contain unique subsequences
# #but every sequence should end on an action is the issue!

# subsequences = []

# for episode in range(len(env_data)):
#     episode_len = len(env_data[episode])
#     for i in range(episode_len - context_len):
#         subsequences.append(env_data[episode][i:i+context_len])
        
# random.shuffle(subsequences)#shuffle since we don't want high correlation between sequences in training

# action_shape = (1, action_size)


#we first need to extract all possible sequences from the data
wood_state_sequences = []
wood_vision_sequences = []
wood_action_sequences = []
wood_returns_to_go_sequences = []
timestep_sequences = []
grass_state_sequences = []
grass_vision_sequences = []
grass_action_sequences = []
grass_returns_to_go_sequences = []

for episode_end in episode_ends:
    i = K
    while (i < episode_end):
        wood_state_sequences.append(wood_states[i - K:i])
        wood_vision_sequences.append(wood_visions[i - K:i])
        wood_action_sequences.append(wood_actions[i - K:i]) #action sequences are one less length since we try to predict the last action
        wood_returns_to_go_sequences.append(wood_returns_to_go[i - K:i])
        timestep_sequences.append(timesteps[i - K:i])
        grass_state_sequences.append(grass_states[i - K:i])
        grass_vision_sequences.append(grass_visions[i - K:i])
        grass_action_sequences.append(grass_actions[i - K:i])
        grass_returns_to_go_sequences.append(grass_returns_to_go[i - K:i])
        i += K

sequence_lengths = [
    len(wood_state_sequences), len(wood_vision_sequences),
    len(wood_action_sequences), len(wood_returns_to_go_sequences),
    len(timestep_sequences), len(grass_state_sequences),
    len(grass_vision_sequences), len(grass_action_sequences),
    len(grass_returns_to_go_sequences)
]

if len(set(sequence_lengths)) != 1:
    print("Sequence lengths are not all equal:")
    print(f"wood_state_sequences: {len(wood_state_sequences)}")
    print(f"wood_vision_sequences: {len(wood_vision_sequences)}")
    print(f"wood_action_sequences: {len(wood_action_sequences)}")
    print(f"wood_returns_to_go_sequences: {len(wood_returns_to_go_sequences)}")
    print(f"timestep_sequences: {len(timestep_sequences)}")
    print(f"grass_state_sequences: {len(grass_state_sequences)}")
    print(f"grass_vision_sequences: {len(grass_vision_sequences)}")
    print(f"grass_action_sequences: {len(grass_action_sequences)}")
    print(f"grass_returns_to_go_sequences: {len(grass_returns_to_go_sequences)}")
else:
    print(f"All sequences have length {sequence_lengths[0], sequence_lengths[1], sequence_lengths[2], sequence_lengths[3], sequence_lengths[4], sequence_lengths[5], sequence_lengths[6], sequence_lengths[7], sequence_lengths[8]}")

#now shuffle these sequences
zipped = list(zip(wood_state_sequences, wood_vision_sequences, wood_action_sequences, wood_returns_to_go_sequences, timestep_sequences, grass_state_sequences, grass_vision_sequences, grass_action_sequences, grass_returns_to_go_sequences))
random.shuffle(zipped)
wood_state_sequences, wood_vision_sequences, wood_action_sequences, wood_returns_to_go_sequences, timestep_sequences, grass_state_sequences, grass_vision_sequences, grass_action_sequences, grass_returns_to_go_sequences = zip(*zipped)
wood_state_sequences, wood_vision_sequences, wood_action_sequences, wood_returns_to_go_sequences, timestep_sequences, grass_state_sequences, grass_vision_sequences, grass_action_sequences, grass_returns_to_go_sequences = list(wood_state_sequences), list(wood_vision_sequences), list(wood_action_sequences), list(wood_returns_to_go_sequences), list(timestep_sequences), list(grass_state_sequences), list(grass_vision_sequences), list(grass_action_sequences), list(grass_returns_to_go_sequences)

sequence_lengths = [
    len(wood_state_sequences), len(wood_vision_sequences),
    len(wood_action_sequences), len(wood_returns_to_go_sequences),
    len(timestep_sequences), len(grass_state_sequences),
    len(grass_vision_sequences), len(grass_action_sequences),
    len(grass_returns_to_go_sequences)
]

if len(set(sequence_lengths)) != 1:
    print("Sequence lengths are not all equal:")
    print(f"wood_state_sequences: {len(wood_state_sequences)}")
    print(f"wood_vision_sequences: {len(wood_vision_sequences)}")
    print(f"wood_action_sequences: {len(wood_action_sequences)}")
    print(f"wood_returns_to_go_sequences: {len(wood_returns_to_go_sequences)}")
    print(f"timestep_sequences: {len(timestep_sequences)}")
    print(f"grass_state_sequences: {len(grass_state_sequences)}")
    print(f"grass_vision_sequences: {len(grass_vision_sequences)}")
    print(f"grass_action_sequences: {len(grass_action_sequences)}")
    print(f"grass_returns_to_go_sequences: {len(grass_returns_to_go_sequences)}")
else:
    print(f"All sequences have length {sequence_lengths[0], sequence_lengths[1], sequence_lengths[2], sequence_lengths[3], sequence_lengths[4], sequence_lengths[5], sequence_lengths[6], sequence_lengths[7], sequence_lengths[8]}")
#------------------------------BATCH INITIALIZATION------------------------------------
#each batch is a tuple of (s, v, R, a, t) where each element is a list of tensors of length batch_size, 
# and each of those tensors contains k tokens, and each token is of its given size
#now, to create batches, separate out the states, visions, action, returns to go, and timesteps
#for each sequence, and put these in corresponding rows in the 
#subsequence items are each of length 5K
# batches are of size 5 * batch_size * K * unique input_dims
dataloader = []
num_batches = int(len(wood_state_sequences) / (batch_size * K))
# num_batches = sequence_lengths[0]

print(f"num_batches: {num_batches}")
print(f"batch_size: {batch_size}")
print(f"K: {K}")
for i in range(num_batches):
    batch_wood = ([], [], [], [], [])
    batch_grass = ([], [], [], [], [])
    for j in range(batch_size):
        print("error info")
        print(i * batch_size + j)
        print(len(wood_returns_to_go_sequences))
        batch_wood[0].append(wood_returns_to_go_sequences[i * batch_size + j])
        batch_wood[1].append(wood_state_sequences[i * batch_size + j])
        batch_wood[2].append(wood_vision_sequences[i * batch_size + j])
        batch_wood[3].append(wood_action_sequences[i * batch_size + j])
        batch_wood[4].append(timestep_sequences[i * batch_size + j])
        batch_grass[0].append(grass_returns_to_go_sequences[i * batch_size + j])
        batch_grass[1].append(grass_state_sequences[i * batch_size + j])
        batch_grass[2].append(grass_vision_sequences[i * batch_size + j])
        batch_grass[3].append(grass_action_sequences[i * batch_size + j])
        batch_grass[4].append(timestep_sequences[i * batch_size + j])
    dataloader.append(batch_wood)
    dataloader.append(batch_grass)

# for i in range(int(len(subsequences) / 5 * batch_size)):
#     #each batch is a tuple of (s, v, R, a, t) where each element is a list of tensors of length batch_size, 
#     # and each of those tensors contains k tokens, and each token is of its given size
#     batch = ([], [], [], [], []) 

#     #thus, we now need to iterate through this list of batch_size sequences we have,
#     #and for each sequence we separate out the states, visions, actions, returns to go, and timesteps
#     #and put these in corresponding rows in the batch
#     subsequences = subsequences[i*batch_size:(i+1)*batch_size]
#     for subsequence in subsequences:
#         for token in subsequence:
#             batch_sequence = []
#             if token.shape == actuator_obs_size:
#                 batch_sequence = batch[0]
#             elif token.shape == visual_obs_size:
#                 batch_sequence = batch[1]
#             elif token.shape == action_shape:
#                 batch_sequence = batch[3]
#             elif token.shape == 1: #returns to go shape
#                 batch_sequence = batch[2]
#             else:
#                 batch_sequence = batch[4]
#             batch_sequence.append(token)
#     dataloader.append(batch)

print("dataloader created")

#initialize model and optimizer
# timesteps = int (total_timesteps / decision_interval)
print(f"visual obs size: {visual_obs_size}")
timesteps = 500
model = DecisionTransformer(actuator_obs_size, visual_obs_size, action_size, n_blocks, n_heads, embed_dim, context_len, dropout, timesteps).to(device)
print("model done")
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wt_decay)
print("optimizer done")
mse_loss = nn.MSELoss()

#---------------------------------TRAINING LOOP-------------------------------------
losses = []
#training loop
for (R, s, v, a, t) in dataloader:
    #alter model to take in an entire batch
    print(f"R shape: ({len(R)}, {len(R[0])})")
    print(f"s shape: ({len(s)}, {len(s[0])}, {len(s[0][0])})")
    print(f"v shape: ({len(v)}, {len(v[0])}, {len(v[0][0])})")
    print(f"a shape: ({len(a)}, {len(a[0])}, {len(a[0][0])})")
    print(f"t shape: ({len(t)}, {len(t[0])})")
    print(t[0])
    R = torch.tensor(R, dtype=torch.float32).unsqueeze(-1).to(device)
    t = torch.tensor(t, dtype=torch.long).to(device)
    s = torch.tensor(np.array(s), dtype=torch.float32).to(device)
    v = torch.tensor(np.array(v), dtype=torch.float32).to(device)
    a_true = torch.tensor(np.array(a)[:, 19, :], dtype=torch.float32).to(device)
    a = torch.tensor(np.array(a)[:, :19, :], dtype=torch.float32).to(device)


    action_prediction = model(R, s, v, a, t)
    action_prediction = action_prediction[:, -1, :]#temporary solution
    loss = mse_loss(action_prediction, a_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

import os
if not os.path.exists('saved_transformer_models'):
    os.makedirs('saved_transformer_models')
folder = "saved_transformer_models/"
model_name = "sac_model" + time.strftime("%Y%m%d-%H%M%S") + ".pth"
torch.save(model, folder + model_name)
torch.save({
    'data_size': len(wood_states),
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': losses,
    # any other metrics or variables you need
}, "saved_transformer_models/decision_transformer_" + time.strftime("%Y%m%d-%H%M%S") + ".pth")