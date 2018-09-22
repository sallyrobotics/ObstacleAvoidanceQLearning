
#Importing the libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

#Creating architecture of NN
#Implimenting the init function
class Network(nn.Module):
    #3 actions - right, left, forward
    #input size is 5 - 2 for orientation, one for forward input, left, and right. If want to get a 360 deg view, add another.
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.np_action = nb_action
        self.fc1 = nn.Linear(self, input_size, 30)
        self.fc2 = nn.Linear(self, 30, nb_action)
        #Here, I've taken only 1 hidden layer with 30 neurons, make more fully connected layers if you want to with any number of neurons.
        
    def forward(self, state):
        #x represents hidden neurons
        x = f.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values
    
#Implementing Experience replay
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        #Always contains a fixed number => we use a push function to do that
        #Memory can be 100 or 1000 or 100000 or whatever, we use 100 cuz computationally easier
        
    def push(self, event):
        self.memory.append(event)
        #Event has 4 parts - st, s(t+1), a(t) and R(t)
        if len(self.memory) > self.capacity:
            del self.memory[0]
      #getting random samples from the previous memory will greatly improve the model
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory , batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
    
#Implementing Deep Q learning
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimiser = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0 
        self.last_reward = 0

#Funtion which selects the right action at each time
    def select_action(self, state):
        probs = f.softmax(self.model(Variable(state , volatile = True))*7) #7 is the temperature paramet
        action = probs.multinomial()
        return action.data[0,0]
    

        
    