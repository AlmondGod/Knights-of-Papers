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
print("connected!")
env.reset()
knight_one_names = env.behavior_specs.keys()

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

#influenced by nikhilbarhate99 min-decision-transformer and decision transformer paper
#write this based on the pseudocode above
class Block(nn.Module):
    def __init__(self, sequence_len, embed_dim, n_heads, dropout):
        super(Block, self).__init__()
        self.sequence_len = sequence_len
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_mask = torch.tril(torch.ones(self.sequence_len, self.sequence_len))
        x = self.attention(x, attn_mask = attn_mask, is_causal=True)[0] + x
        x = self.ln1(x)
        x = self.mlp(x) + x
        x = self.ln2(x)

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, n_blocks, n_heads, embed_dim, context_len, dropout, timesteps):
        super(DecisionTransformer, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = embed_dim

        input_len = 3 * context_len #each input is state, action, rewards to go
        attention_blocks = [Block(input_len, embed_dim, n_heads, dropout) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*attention_blocks)

        #projection heads to convert inputs of varying sizes to embeddings
        self.ln = nn.LayerNorm(embed_dim)
        self.embed_states = nn.Linear(state_dim, embed_dim)
        self.embed_actions = nn.Linear(action_dim, embed_dim)
        self.embed_returns_to_go = nn.Linear(1, embed_dim)
        print(timesteps)
        self.embed_timesteps = nn.Embedding(timesteps, embed_dim)
        
        #Prediction heads (these determine our output)
        self.predict_returns_to_go = nn.Linear(embed_dim, 1)
        self.predict_state = nn.Linear(embed_dim, state_dim)
        self.predict_action = nn.Sequential(
            nn.Linear(embed_dim, action_dim),
            nn.Tanh()
            )

    def forward(self, timesteps, states, actions, returns_to_go):
        time_embed = self.embed_timesteps(timesteps)
        state_embed = self.embed_states(states) + time_embed
        action_embed = self.embed_actions(actions) + time_embed
        returns_to_go_embed = self.embed_returns_to_go(returns_to_go) + time_embed

        input_embeds = torch.cat([state_embed, action_embed, returns_to_go_embed], dim=2)

        input_embeds = self.ln(input_embeds)

        hidden_states = self.transformer(input_embeds)

        actions_pred = self.predict_action(hidden_states[:, :, self.state_dim:self.state_dim+self.action_dim])
        returns_to_go_pred = self.predict_returns_to_go(hidden_states[:, :, -1])
        states_pred = self.predict_state(hidden_states[:, :, :self.state_dim])

        return actions_pred, returns_to_go_pred, states_pred

 
# Get the state and action sizes
wooden_knight = list(env.behavior_specs.keys())[0]
print(f"specs: {list(env.behavior_specs.keys())}")
grass_knight = list(env.behavior_specs.keys())[1]

behavior_spec = env.behavior_specs[wooden_knight]

# Extract observation and action space sizes
actuator_obs_size = behavior_spec.observation_specs[0][0][0]
visual_obs_size = behavior_spec.observation_specs[1][0][0] * behavior_spec.observation_specs[1][0][1] * behavior_spec.observation_specs[1][0][2]
state_size = actuator_obs_size + visual_obs_size
action_size = behavior_spec.action_spec.continuous_size

#initialize dataloader to hold (R, s, a, t) # dims : ( batch_size , K, dim )

batch_size = 32
learning_rate = 1e-4
wt_decay = 1e-4

#initialize model and optimizer
n_blocks = 3
n_heads = 2
embed_dim = 64
dropout = 0.1
context_len = 20 #value of K
total_timesteps = 500
decision_interval = 5
timesteps = int (total_timesteps / decision_interval)
model = DecisionTransformer(state_size, action_size, n_blocks, n_heads, embed_dim, context_len, dropout, timesteps)
print("model done")

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wt_decay)

print("optimizer done")

#TODO: implement dataloader to hold (R, s, a, t) # dims : ( batch_size , K, dim )
dataloader = []

#TODO: implement training loop
