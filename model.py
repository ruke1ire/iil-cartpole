import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NNActor(nn.Module):
    def __init__(self, intervention_thresholds = None):
        super().__init__()
        hidden_size = 100
        self.linear = nn.Linear(4,hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size,1)

        if(intervention_thresholds is not None):
            self.threshold = intervention_thresholds
    
    def forward(self, state, noise = 0.0):
        device = state.device
        x = F.silu(self.linear(state))
        x = F.silu(self.hidden(x))
        output = torch.tanh(self.output(x))
        noise_tensor = 2*torch.rand(1).to(device) - 1
        output = noise*noise_tensor + (1-noise)*output
        return output

    def check_bad_state(self, state):
        # np.array([x, x_dot, theta, theta_dot])
        #if(abs(state[0]) >= 2.0) or (abs(state[1]) >= 3) or (abs(state[2]) >= 10*2*math.pi/360) or (abs(state[3]) >= 4):
        if(abs(state[0]) >= self.threshold['x']) or (abs(state[1]) >= self.threshold['x_dot']) or (abs(state[2]) >= self.threshold['theta']) or (abs(state[3]) >= self.threshold['theta_dot']):
            return True
        else:
            return False
    
class NNCritic(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 100
        self.linear = nn.Linear(5,hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size,1)
    
    def forward(self, state, action):
        state_action = torch.concat((state,action), dim = 1)
        x = F.silu(self.linear(state_action))
        x = F.silu(self.hidden(x))
        x = self.output(x)
        return x