# moonlander_rl

A reinforcement learning algorithm which is trained to play Lunar Lander.
Programmed in Python with the PyTorch library used for the neural network, and OpenAI gym used for the environment.
The environment contains a discrete action space, having actions for doing nothing, thursting up, thrusting right, and thrusting left.
The observation space (state) simply constists of the coordinates of the agent, its velocity, heading, angular velocity, and whether or not each of its legs are in contact
with the ground.
