This project has been developed using the concepts of reinforcement learning to predict the rocket's movement. the rocket moves against a few constraints such as Thrust, Rotation and weight of the rocket. The reinforcement learning approach is based on the reward basis. Rewards have been allocated based on each action performed by the rocket

1. Actions
In the rocket simulation, the rocket deals with 6 actions such as,

Move upward
Move downward
Move left
Move right
Rotate clockwise
Rotate counterclockwise
2. Crashed events
Here rocket moves within the range of width=800 and height=600 environment. If the rocket is out of the region then we consider the rocket has crashed.

If the angle of the spinning of the rocket exceeds more than 90 degrees then we consider the rocket has crashed.

3. Landed events
To be landed safely the rocket should landed within the range of the window.

The rocket should maintain an angle not exceeding 45 degrees with the rocket axis

4. Reward Function
In this model has been developed using the DQN reinforcement learning approach and the rewards have been allocated based on each action taken by the rocket. There is a dynamic allocating system to maintain the rocket by avoiding rotating and encouraging the rocket to keep a stable position.

If the rocket is out of the region then provide a -20 reward
If the rocket is spinning then allocate a -10 reward
If the rocket lands clos to the center of the width it gets highest reward like wise based on the position the rocket learns the next action( Bellman equation)
Required libraries

1. Numpy
2. Torch
3. PyGame
4. GYM
