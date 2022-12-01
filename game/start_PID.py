import gym
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
#from ai_mode.env import Env
from ai_mode.simple_env import Env

class Data():
    """tracks elements of the state"""
    def __init__(self):
        self.states = []
    
    def add(self,state):
        self.states.append(state)
        
    def graph1(self):
        states = np.array(self.states).reshape(len(self.states),-1)
        plt.rcParams.update({'font.size': 17}) 
        plt.figure(figsize=(15, 8))
        plt.plot(states[:,0],label='x')
        plt.plot(states[:,1],label='y')
        plt.legend()
        plt.grid()
        plt.title('PID Control')
        plt.ylabel('Value')
        plt.xlabel('Steps')
        plt.show('pid.png')

    def graph2(self):
        states = np.array(self.states).reshape(len(self.states),-1)
        plt.rcParams.update({'font.size': 17}) 
        plt.figure(figsize=(15, 8))
        plt.plot(states[:,2],label='vx')
        plt.plot(states[:,3],label='vy')
        plt.legend()
        plt.grid()
        plt.title('PID Control')
        plt.ylabel('Value')
        plt.xlabel('Steps')
        plt.show('pid.png')
    
    def graph3(self):
        states = np.array(self.states).reshape(len(self.states),-1)
        plt.rcParams.update({'font.size': 17}) 
        plt.figure(figsize=(15, 8))
        plt.plot(states[:,4],label='theta')
        plt.legend()
        plt.grid()
        plt.title('PID Control')
        plt.ylabel('Value')
        plt.xlabel('Steps')
        plt.show('pid.png')

    def graph4(self):
        states = np.array(self.states).reshape(len(self.states),-1)
        plt.rcParams.update({'font.size': 17}) 
        plt.figure(figsize=(15, 8))
        plt.plot(states[:,5],label='w')
        plt.legend()
        plt.grid()
        plt.title('PID Control')
        plt.ylabel('Value')
        plt.xlabel('Steps')
        plt.show('pid.png')


def pid(state, params):
    """ calculates settings based on pid control """
    # PID parameters
    kp_y = params[0]  # proportional altitude
    kd_y = params[1]  # derivative altitude
    kp_angle = params[2]  # proportional angle
    kd_angle = params[3]  # derivative angle

    # Calculate setpoints (target values)
    target_y = np.abs(state[0])
    target_angle = (.25*np.pi)*(state[0]+state[2])

    # Calculate error values
    y_error = (target_y - state[1])
    angle_error = (target_angle - state[4])
    
    # Use PID to get adjustments
    y_adj = kp_y*y_error + kd_y*state[3]
    angle_adj = kp_angle*angle_error + kd_angle*state[5]

    a = np.array([y_adj, angle_adj])
    a = np.clip(a, -1, +1)
    
    # Selecting action from action_space
    # If either of the legs have contact
    if state[6] or state[7]:
        action = 0
    # Given an action a = `np.array([main, lateral])`, the main engine will be turned off completely if `main < 0`
    # and the throttle scales affinely from 50% to 100% for `0 <= main <= 1` (the main engine doesn't work  with less than 50% power).
    #Similarly, if `-0.5 < lateral < 0.5`, the lateral boosters will not fire at all. If `lateral < -0.5`, the left
    #booster will fire, and if `lateral > 0.5`, the right booster will fire. Again, the throttle scales affinely
    #from 50% to 100% between -1 and -0.5 (and 0.5 and 1, respectively).
    action = 0
    if((a[0] >= 0) and (a[0] <=1)): action = 3
    elif(a[1] < -0.5): action = 1
    elif(a[1] > 0.5): action = 2
    
    return action

def run(params, env, verbose=False):
    """ runs an episode given pid parameters """
    data = Data() 
    done = False
    state = env.reset()
    if verbose:
        env.render()
        sleep(.005)
    data.add(state)
    total = 0
    while not done:
        action = pid(state, params)
        state,reward,done,_ = env.step(action)
        total += reward
        if verbose:
            env.render()
            sleep(.005)
        data.add(state)
    return total, data

def optimize(params, current_score, env, step):
    """ runs a step of randomized hill climbing """

    # add gaussian noise (less noise as n_steps increases)
    test_params = params + np.random.normal(0,80.0/step,size=params.shape)
    
    # test params over 5 trial avg
    scores = []
    for trial in range(3):
        score,_ = run(test_params,env)
        scores.append(score)
    avg = np.mean(scores)
    
    # update params if improved
    if avg > current_score:
        return test_params,avg
    else:
        return params,current_score

def main():
    # Setup environment
    env = Env()
    # Seed RNGs
    np.random.seed(0)

    # Random Hill Climb over params
    params = np.array([0,0,0,0])
    score = -2000 # bad starting score
    for steps in range(2000):
        params,score = optimize(params,score,env,steps+1)
        if steps%10 == 0:
            print("Step:",steps,"Score:",score,"Params:",params)

    pre = 0
    # Get data for final run
    scores = []
    for trial in range(100):
        score, data = run(params, env, True)
        if(score > pre):
            dis = data

        scores.append(score)
    
    print("Average Score:",np.mean(scores))
    
    dis.graph1()
    dis.graph2()
    dis.graph3()
    dis.graph4()

    scores = np.array(scores)
    mean = np.empty(100)
    mean.fill(np.mean(scores))

    x = np.arange(0, 100)

    # plotting
    plt.rcParams.update({'font.size': 17}) 
    plt.figure(figsize=(15, 8))
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(x, scores, color ="blue")
    plt.plot(x, mean, color ="orange")
    plt.show()    
    

if __name__ == '__main__':
    main()
