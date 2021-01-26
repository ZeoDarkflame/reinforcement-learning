import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as functions
import torch.optim as optim
import random

# the observed state is a vector of shape (4,)
# 0 is move left
# 1 is move right
# reward +1 for everytime the episode is not terminated

memory = []

def policy(ob,net,epsilon):
    randnum = np.random.rand()
    if randnum < epsilon:
        return np.random.randint(0,2)
    else:
        with torch.no_grad():
            return np.argmax(net(torch.from_numpy(ob).float()).numpy())

def gen_episode(env,gamma,epsilon,net,net_cache,render = False, not_training = False):

    done = False
    ob = env.reset()
    global memory
    step_count = 0
    actions = []

    while(not done):
        if(render):
            env.render()
            time.sleep(0.05)
        action = policy(ob,net,epsilon)    # pass network to policy function for forward pass 
        actions.append(action)
        newob, rew, done, _ = env.step(action)
        memory.append((ob,action,rew,newob,done))
        ob = newob
        step_count += 1
        if(not not_training):
            train_network(net,net_cache,optimizer,memory,render)
    if(render):
        print(actions)

    
    return step_count
    
        
        

class NNet(nn.Module): ## making Network
    
    def __init__(self):
        super(NNet,self).__init__()
        self.fc1 = nn.Linear(4,6)
        self.fc2 = nn.Linear(6,2)
        
    def forward(self,x):
        x = self.fc1(x)
        x = functions.relu(x)
        x = self.fc2(x)
        x = functions.relu(x)
        
        return x
    
def train_network(net,net_cache,optimizer,rep_memory,render=False):   # Y = r_t + gamma*max(Q(S_t+1))
    
    memory = []
    try:
        memory = random.sample(rep_memory,64)
    except:
        memory = rep_memory

    X = []
    Y = []
    actions = []
    rewards = []
    terminated = []
    for i in memory:
        X.append(i[0])
        actions.append(i[1])
        rewards.append(i[2])
        Y.append(i[3])
        terminated.append(i[4])
    X = np.array(X)
    Y = np.array(Y)
    
    with torch.no_grad():
        otp = net_cache(torch.from_numpy(X).float()).numpy()
        Y = net_cache(torch.from_numpy(Y).float()).numpy()
    Y = np.max(Y,axis=1,keepdims = True)
    Y = 1+0.9*Y
    
    for i in range(len(terminated)):
        if(terminated[i]):
            otp[i][actions[i]] = 0
        else:
            otp[i][actions[i]] = Y[i]

    #optimizer defined in main
    loss = nn.MSELoss()
    X = torch.from_numpy(X).float()
    otp = torch.from_numpy(otp).float()
    costs = []
    optimizer.zero_grad()
    output = net(X)
    cost = loss(output,otp)
    cost.backward()
    optimizer.step()
    costs.append(cost.item())
    if(render):
        print(costs)    

def is_solved(env,net,net_cache,gamma):
    steps = []
    for i in range(100):
        steps.append(gen_episode(env,gamma,0,net,net_cache,not_training=True))
    cumu_rew = sum(steps)/len(steps)

    print(cumu_rew)


if __name__ == '__main__':

    net = NNet()
    net_cache = NNet()
    name = 'CartPole-v0' # name of environment
    env = gym.make(name)
    print(net(torch.from_numpy(env.reset()).float()))
    steps = []

    optimizer = torch.optim.SGD(net.parameters(),lr = 0.05)
    #gen_episode(env,0.9,0,net,True,True)
    for i in range(800):
        #print(i,end = '\r')
        if(i%25 == 0):
            net_cache.load_state_dict(net.state_dict())
        if(i%100 == 0):
            steps.append(gen_episode(env,0.9,200/(200+i),net,net_cache,True))
        else:
            steps.append(gen_episode(env,0.9,200/(200+i),net,net_cache))
    input('Confirm....')
    gen_episode(env,0.9,0,net,net_cache,True)
    plt.plot(steps)
    i = input('do you want to save model')
    if (int(i) == 0):
        print('saving')
        torch.save(net.state_dict(),'DQN.pth') 
    plt.savefig('DQNSteps.png')     #save the plot

    #these commands evaluate the agent
    #net.load_state_dict(torch.load('DQN.pth'))
    #is_solved(env,net,net_cache,0.9)