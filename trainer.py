import torch
import torch.nn.functional as F
import pickle
from abc import ABC, abstractmethod

from utils import *

class Trainer(ABC):
    @abstractmethod
    def __init__(self):
        '''
        Initializes the components needed for the algorithm. 
        Every component and variable should be set here and should not be changed later on.
        If a certain variable/component is to be changed, the algorithm object should be re-initilized
        '''
        pass

    @abstractmethod
    def train_one_mini_batch(self, models, replay_buffer):
        '''
        Train the algorithm on a single epoch
        
        models -- dict of models Eg: dict(expert = <nn.Module>, agent = <nn.Module>)
        replay_buffer --  ReplayBuffer
        '''
        pass

class SLRL(Trainer):
    def __init__(self, 
                actor_model,
                critic_model,
                actor_optimizer_name,
                actor_optimizer_kwargs,
                critic_optimizer_name,
                critic_optimizer_kwargs,
                replay_buffer,
                device,
                discount = 0.99,
                tau = 0.005,
                noise = 0.2,
                actor_update_period = 2,
                only_rl = False,
                batch_size = 24,
                ):

        self.discount = discount
        self.tau = tau
        self.noise = noise
        self.actor_update_period = actor_update_period
        self.batch_size = batch_size
        self.only_rl = only_rl
        self.device = device

        self.actor_model = actor_model.to(self.device)
        self.critic_1_model = critic_model.to(self.device)
        self.critic_2_model = pickle.loads(pickle.dumps(self.critic_1_model)).to(self.device)
        self.actor_target_model = pickle.loads(pickle.dumps(self.actor_model)).to(self.device)
        self.critic_1_target_model = pickle.loads(pickle.dumps(self.critic_1_model)).to(self.device)
        self.critic_2_target_model = pickle.loads(pickle.dumps(self.critic_1_model)).to(self.device)

        self.actor_optimizer = create_optimizer(actor_optimizer_name, actor_optimizer_kwargs, self.actor_model.parameters())
        self.critic_1_optimizer = create_optimizer(critic_optimizer_name, critic_optimizer_kwargs, self.critic_1_model.parameters())
        self.critic_2_optimizer = create_optimizer(critic_optimizer_name, critic_optimizer_kwargs, self.critic_2_model.parameters())

        self.replay_buffer = replay_buffer

        self.i = 0
    
    def train_one_mini_batch(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        state, action, demonstration_flag, reward, termination_flag, next_state = Transition(*zip(*transitions))

        with torch.no_grad():
            state = torch.stack(state).to(self.device)
            action = torch.stack(action).to(self.device)
            demonstration_flag = torch.stack(demonstration_flag)
            reward = torch.tensor(reward).to(self.device)
            termination_flag = torch.stack(termination_flag).to(self.device)
            next_state = torch.stack(next_state).to(self.device)

            # 1. Compute target actions from target actor P'(s(t+1))
            next_actions = self.actor_target_model(state = next_state, noise = self.noise)

            # 2. Compute Q-value of next state using the  target critics Q'(s(t+1), P'(s(t+1)))
            next_q1 = self.critic_1_target_model(state = next_state, action = next_actions)
            next_q2 = self.critic_2_target_model(state = next_state, action = next_actions)

            # 3. Use smaller Q-value as the Q-value target")
            next_q = torch.min(next_q1, next_q2)

            # 4. Compute current Q-value with the reward")
            target_q = reward + self.discount * next_q * (1-termination_flag)

        # 5.1 Compute Q-value from critics Q(s_t, a_t)")
        q1 = self.critic_1_model(state = state, action = action)

        # 6.1 Compute MSE loss for the critics")
        critic_1_loss = ((q1[demonstration_flag == 0.0] - target_q[demonstration_flag == 0])**2).mean()

        # 7.1 Optimize critic")
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        # 5.2 Compute Q-value from critics Q(s_t, a_t)")
        q2 = self.critic_2_model(state = state, action = action)

        # 6.2 Compute MSE loss for the critics")
        critic_2_loss = ((q2[demonstration_flag == 0.0] - target_q[demonstration_flag == 0.0])**2).mean()

        # 7.2 Optimize critic")
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 8. Compute actor actions")
        actor_action = self.actor_model(state = state)

        # 9. Compute actor loss")
        if(self.only_rl == False):
            actor_se = (actor_action - action)**2
            actor_se[demonstration_flag == 0.0] = actor_se[demonstration_flag == 0.0]*0.0
            actor_loss_ = actor_se.mean()
        else:
            actor_loss_ = None

        if(self.i % self.actor_update_period == (self.actor_update_period - 1)):
            # 10. Compute the negative critic values using the real critic")
            negative_value = -self.critic_1_model(
                                state = state,
                                action = actor_action)[demonstration_flag == 0.0]
            negative_value = negative_value.mean()
        else:
            negative_value = None

        if(actor_loss_ is None):
            actor_loss = negative_value
        else:
            if(negative_value is not None):
                actor_loss = actor_loss_ + negative_value
            else:
                actor_loss = actor_loss_

        # 11. Optimize actor")
        if(actor_loss is not None):
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # 12. Update target networks")
        for param, target_param in zip(self.critic_1_model.parameters(), self.critic_1_target_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2_model.parameters(), self.critic_2_target_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor_model.parameters(), self.actor_target_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)	

        self.i += 1

class SL(Trainer):
    def __init__(self, 
                actor_model,
                actor_optimizer_name,
                actor_optimizer_kwargs,
                replay_buffer,
                device,
                only_supervised = False,
                batch_size = 24
                ):
        '''
        Supervised Learning (SL) algorithm implementation.
        '''

        self.only_supervised = only_supervised
        self.batch_size = batch_size
        self.device = device

        self.actor_model = actor_model.to(self.device)
        self.actor_optimizer = create_optimizer(actor_optimizer_name, actor_optimizer_kwargs, self.actor_model.parameters())
        self.replay_buffer = replay_buffer

        self.i = 0
    
    def train_one_mini_batch(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        state, action, demonstration_flag, reward, termination_flag, next_state = Transition(*zip(*transitions))

        with torch.no_grad():
            state = torch.stack(state).to(self.device)
            action = torch.stack(action).to(self.device)
            demonstration_flag = torch.stack(demonstration_flag)

        # 1. Compute actor actions")
        actor_action = self.actor_model(state = state)

        # 2. Compute actor loss")
        actor_se = (actor_action - action)**2

        if(self.only_supervised == True):
            if(sum(demonstration_flag == 1.0).item() == 0):
                return
            else:
                actor_loss = actor_se[demonstration_flag == 1.0].mean()
        else:
            actor_se[demonstration_flag == 0.0] = actor_se[demonstration_flag == 0.0]*0.1
            actor_loss = actor_se.mean()

        # 3. Optimize actor")
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.i += 1
    