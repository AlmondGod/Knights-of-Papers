# Knights of Papers: 
## Duel of Reinforcement Learning Algorithms

See [this video](https://youtu.be/kpDfXqX7h1U) on the creation and training of this repository's contents!

Algorithms: 

[DDPG](./src/DDPG/) 

[Decision Transformer](./src/Decision-Transformer/) 

[I2A with PPO](./src/I2A-PPO/)

[SAC](./src/SAC/)

## Installation

Environments exist for macos, windows, and linux. 
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

saved_algorithm_model: saved models which can be run using the running files in the same folder

[environment-builds](environment-builds): (explained below) available for [linux](environment-builds/linux/), [macos](environment-builds/macos), and [windows](environment-builds/windows/). To use them simply link the filepath of the desired build in the relevant script's environment setup. 

[Assets](Assets): Unity files and C# code used to construct the environment

## Environment
This project uses Unity Machine Learning Simulator, which can be downloaded [here](https://unity.com/download) (not necessary for the function of this repository, only if anyone wants to modify the environment). Each body has 9 joints and movement restricted according to typical human joint dimensions. Rewards are gained by 
1. Touching ones' sword to the opponent (+10 and episode end)
2. Maintaining ones' head above their body in the y dimension (+0.1, -0.1 otherwise)
3. Moving closer to the opponent than in the previous timestep (+0.1, -0.1 otherwise)

The robots start two body-lengths from each other and must converge to fight. Walls surround the small arena, which, if contacted, result in a -10 reward penalty and the episode ending. The agents are automatically reset after 1000 timesteps.

Environment builds are necessary to allow the Unity ML Python API to ineract with the environments using custom algorithms. 

For macos, use the highest numbered build (currently [build8.app](environment-builds/macos/build8.app)), for linux use [gpubuildclose.x86_64](environment-builds/linux/gpubuildclose.x86_64), and for windows, either suffices.

## Training
Each algorithm is trained by self-play in a 60-agent environment over 15000-5000 environment episodes, each with 1000 timesteps. Models trained using vast.ai-rented NVIDIA RTX 4090.

## Visualizations
To see a trained algorithm, use the equivlent run-saved script or one of the duels scripts, and make sure the environment is set to no_graphics=True