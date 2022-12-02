## Lunar Lander with RL 

My Thesis for my Master's degree in Control and Computing, UoA:
Application of Reinforcement Learning methods in solving a Lunar Lander game


#### The game allows you either to play it yourself or implement various flavours of the DQN algorithm (vanilla DQN, standard DQN or DDQN), in order to solve it.
Moreover, it contains the implementation of a PID controller that tries to solve the Lunar Lander game as well.

**Requirements**:
- Python3
- Modules:
    - pygame
    - gym
    - keras
    

#### Explaining repo

The project is consisted of the following files/folders:

- **thesis.pdf**
- **game** 
    - **ai_mode** : contains the source code for the implementation of the RL agent 
    - **game_mode** : contains the source code for the game as well as the necessary image and sound files
    - **start_human.py** 
    - **start_training_vanillaDQN.py**
    - **start_training_DQN.py**  
    - **start_training_DDQN.py**
    - **start_PID.py**


#### How-to run

For playing the game yourself():  
```
python3 start_human.py 
```

For training the RL agents using vanilla DQN or standard DQN or DDQN algorithm, and then test the trained model:
```
python3 start_training_vanillaDQN.py 
or
python3 start_training_DQN.py 
or
python3 start_training_DDQN.py 
```

For running the PID controller:
```
python3 start_PID.py  
```


