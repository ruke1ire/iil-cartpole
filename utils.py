from collections import namedtuple, deque
import random
import torch

Transition = namedtuple('Transition',('state', 'action', 'demonstration_flag', 'reward', 'termination_flag', 'next_state'))

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, **kwargs):
        """Save a transition"""
        self.memory.append(Transition(**kwargs))

    def sample(self, batch_size):
        memory_size = len(self.memory)
        if(batch_size > memory_size):
            return random.sample(self.memory, memory_size)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
   
def create_optimizer(name, kwargs, parameters):
    optim_init = getattr(torch.optim, name)
    optimizer = optim_init(parameters, **kwargs)
    return optimizer