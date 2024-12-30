<img width="1000" alt="Screenshot 2024-12-30 at 4 26 17 PM" src="https://github.com/user-attachments/assets/cc4fcfd6-0701-4178-8665-9b3012aabb92" />

# Knights of Papers: Duel of Reinforcement Learning Algorithms

Reinforcement Learning Self-Play on 24-DOF Unity combat humanoids. See [this video](https://youtu.be/kpDfXqX7h1U) on the creation of this repository's contents!

Algorithms: \
[DDPG](./src/DDPG/): [paper](https://arxiv.org/abs/1509.02971) \
[Decision Transformer](./src/Decision-Transformer/): [paper](https://arxiv.org/abs/2106.01345) \
[I2A with PPO](./src/I2A-PPO/): [paper](https://arxiv.org/abs/1707.06203), [paper](https://arxiv.org/abs/1707.06347) \
[SAC](./src/SAC/): [paper](https://arxiv.org/abs/1801.01290)

## Installation

Environments exist for Macos, Windows, and Linux. 
You must use Python > 3.10.5, tested on Python 3.10.14

```
git clone https://github.com/AlmondGod/Knights-of-Papers knights-of-papers
cd knights-of-papers
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Relevant File Map 
[src](src): training algorithms

saved_x_model: saved models which can be run using the running files in the same folder

[environment-builds](environment-builds): (explained below) available for [linux](environment-builds/linux/), [macos](environment-builds/macos), and [windows](environment-builds/windows/). To use them simply link the filepath of the desired build in the relevant script's environment setup. 

[Assets](Assets): Unity files and C# code used to construct the environment

## Environment
This project uses Unity Machine Learning Simulator, which can be downloaded [here](https://unity.com/download) (not necessary for the function of this repository, only if anyone wants to modify the environment). Each body has 9 joints and movement is restricted according to typical human joint dimensions. Rewards are gained by 
1. Touching one's sword to the opponent (+10 and episode end)
2. Maintaining one's head above their body in the y dimension (+0.1, -0.1 otherwise)
3. Moving closer to the opponent than in the previous timestep (+0.1, -0.1 otherwise)

The robots start two body lengths from each other and must converge to fight. Walls surround the small arena, which, if contacted, result in a -10 reward penalty and the episode ending. The agents are automatically reset after 1000 timesteps.

Environment builds are necessary to allow the Unity ML Python API to interact with the environments using custom algorithms. 

For Macos, use the highest numbered build (currently [build8.app](environment-builds/macos/build8.app)), for Linux use [gpubuildclose.x86_64](environment-builds/linux/gpubuildclose.x86_64), and for windows, either suffices.

## Training
Each algorithm is trained by self-play in a 60-agent environment over 1500-5000 environment episodes, each with 1000 timesteps. Models trained using vast.ai-rented NVIDIA RTX 4090.

## Visualizations
To see a trained algorithm, use the equivalent run-saved script or one of the duels scripts, and make sure the environment is set to no_graphics=True
