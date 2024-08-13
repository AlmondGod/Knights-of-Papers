# Knights of Papers: 
## Duel of Reinforcement Learning Algorithms

Algorithms: DDPG, Decision Transformer, I2A with PPO, SAC
Using Unity MLAgents simulator

### Installation

Environments exist for macos, windows, and linux. 
You must use Python > 3.10.5, tested on Python 3.10.14

`git clone https://github.com/AlmondGod/Knights-of-Papers knights-of-papers`
`cd knights-of-papers`
`python3 -m venv venv`
`source venv/bin/activate`
`pip install --upgrade pip`
`pip install -r requirements.txt`

And run any desired algorithm in src!

# Training
Each algorithm is trained by self-play in a 8-agent environment over 3000-5000 environment episodes
Used vast.ai-rented NVIDIA RTX 4090

# Visualizations
To see a trained algorithm, use the equivlent run-saved script or one of the duels scripts, and make sure the environment is set to no_graphics=True


