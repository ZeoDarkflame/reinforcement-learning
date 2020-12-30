import numpy as np
import gym
from gym_minigrid.wrappers import *
import matplotlib
import time
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')		#using this as showing figures does not work on agg, need to install tkinter

name = 'MiniGrid-FourRooms-v0'  # name of the setting from MINI-GRIDWORLD
global successes
successes = 0

# 0 is rotate left 90deg
# 1 is rotate right 90deg
# 2 is move forward
# 3-6 do nothing for now
# looking right is direction 0
# looking down is direction 1
# looking left is direction 2
# looking up is direction 3

try:
    sar_vals = np.load('sar_vals.npy')
    print('loading')
except:
    sar_vals = np.zeros((11,11,4,3))
    print('recreating')


def policy(x,y,dir,epsilon):
    randnum = np.random.rand()
    
    if randnum < epsilon:
        return np.random.randint(0,3)
    else:
        return np.argmax(sar_vals[x][y][dir])
    

def gen_episode(env,alpha,gamma,epsilon,render = False):
    global successes
    env.reset()
    done = False
    if(render):
        env.render()
    time.sleep(1)
    steps = 0
    x = env.agent_pos[0]
    y = env.agent_pos[1]
    dir = env.agent_dir

    while(not done):
        action = policy(x,y,dir,epsilon)
        _ , rew, done, _ = env.step(action)
        if rew == 0:
            rew = -1
        steps += 1
        if(render):
            env.render()
        newx = env.agent_pos[0]
        newy = env.agent_pos[1]
        newdir = env.agent_dir
        sar_vals[x,y,dir,action] += alpha*(rew + gamma*(sar_vals[newx,newy,newdir,policy(newx,newx,newdir,epsilon)]) - sar_vals[x,y,dir,action])
        
        x = newx
        y = newy
        dir = newdir
        
    #print(rew,steps)
    if(rew > 0):
        successes += 1
    return rew,steps


if __name__ == "__main__":
    rewards_list = []
    steps_list = []
    env = gym.make(name)
    
    alpha = 0.8
    
    gamma = 0.8
    
    epochs = 500
    
    show_on = [0,250,350,499]
    
    for i in range(epochs):
        print(str(i),end = '\r')
        if(i in show_on):
            reward , steps = gen_episode(env,alpha,gamma,(50/(50+i)),True)
        else:
            reward , steps = gen_episode(env,alpha,gamma,(50/(50+i)),False)
        rewards_list.append(reward)
        steps_list.append(steps)
    np.save('sar_vals.npy',sar_vals)

    # saving results
    plt.clf()
    plt.plot(rewards_list)
    plt.show()
    plt.savefig('sar_rewards.png')
    plt.clf()
    plt.plot(steps_list)
    plt.show()
    plt.savefig('sar_steps.png')
    np.save('sar_rewards',np.array(rewards_list))
    np.save('sar_steps',np.array(steps_list))


# run the policy without exploration
'''
env = gym.make(name)
gen_episode(env,0.8,0.8,0,True)
'''
#np.save('sar_vals.npy',sar_vals)
