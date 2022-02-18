import torch
import numpy as np

from utils import *
from logger import *

class IIL_algorithm:
    def __init__(self, env, trainer, expert, replay_buffer, noise = 0.0, logger = None):
        self.env = env
        self.trainer = trainer
        self.device = self.trainer.device
        self.agent = self.trainer.actor_model
        self.expert = expert
        self.replay_buffer = replay_buffer
        self.noise = noise
        self.logger = logger
    
    def run(self, num_of_episodes, offset_steps = 0):
        avg_len = 0
        avg_sup = 0
        avg_agn = 0
        for ep_no in range(1, num_of_episodes+1):
            print(f"Episode No.: {ep_no}", end="\t")
            count = 0
            count_supervised = 0
            next_demo = False
            done = False
            next_state = self.env.reset()
            while done == False:
                demo = next_demo
                state = next_state

                state_tensor = torch.tensor(state, dtype = torch.float32).to(self.device)

                if(demo == False):
                    action_tensor = self.agent(state_tensor, noise = self.noise)
                else:
                    action_tensor = self.expert(state_tensor)
                    count_supervised += 1

                next_state, reward, done, info = self.env.step(np.array(action_tensor.detach().cpu(), dtype = np.float32))

                if(self.expert.check_bad_state(next_state) == True):
                    next_demo = True
                else:
                    next_demo = False
                
                if(next_demo == True and demo == False) or done == True:
                    termination_flag = True
                else:
                    termination_flag = False

                self.replay_buffer.push(
                    state = state_tensor,
                    action = action_tensor,
                    demonstration_flag = torch.tensor(demo, dtype = torch.float32),
                    reward = reward,
                    termination_flag = torch.tensor(termination_flag, dtype = torch.float32),
                    next_state = torch.tensor(next_state, dtype = torch.float32),
                )

                if(self.trainer is not None):
                    self.trainer.train_one_mini_batch()
                
                count += 1

            print(f"[Episode Length: {count}]", end = "\t")
            print(f"[Supervised Frames: {count_supervised}]")
            print(f"[Agent Frames: {count-count_supervised}]")
            offset_steps += count_supervised

            log_dict = {"Episode Length": count, "Supervised Frames": count_supervised, "Agent Frames": count-count_supervised, "Steps": offset_steps}
            self.logger.log(DataType.dict, data = log_dict, key = None)
            avg_len += count/num_of_episodes
            avg_sup += count_supervised/num_of_episodes
            avg_agn += (count-count_supervised)/num_of_episodes
        
        return avg_len, avg_sup, avg_agn, offset_steps

class RL_algorithm:
    def __init__(self, env, trainer, replay_buffer, noise = 0.0, logger = None):
        self.env = env
        self.trainer = trainer
        self.device = self.trainer.device
        self.agent = self.trainer.actor_model
        self.replay_buffer = replay_buffer
        self.noise = noise
        self.logger = logger
    
    def run(self, num_of_episodes):
        avg_len = 0
        for ep_no in range(1, num_of_episodes+1):
            print(f"Episode No.: {ep_no}", end="\t")
            count = 0
            done = False
            next_state = self.env.reset()
            while done == False:
                state = next_state

                state_tensor = torch.tensor(state, dtype = torch.float32).to(self.device)

                action_tensor = self.agent(state_tensor, noise = self.noise)

                next_state, reward, done, info = self.env.step(np.array(action_tensor.detach().cpu(), dtype = np.float32))
                
                self.replay_buffer.push(
                    state = state_tensor,
                    action = action_tensor,
                    demonstration_flag = torch.tensor(False, dtype = torch.float32),
                    reward = reward,
                    termination_flag = torch.tensor(done, dtype = torch.float32),
                    next_state = torch.tensor(next_state, dtype = torch.float32),
                )

                if(self.trainer is not None):
                    self.trainer.train_one_mini_batch()
                
                count += 1

            print(f"[Episode Length: {count}]")
            self.logger.log(DataType.num, data = count, key = "Episode Length")
            avg_len += count/num_of_episodes

        return avg_len

class Test_algorithm:
    def __init__(self, env, tester):
        self.env = env
        self.tester = tester
    
    def run(self, num_of_episodes, models):
        agent = models['actor']
        for traj_no in range(1, num_of_episodes+1):
            print(f"Generating trajectory no. : {traj_no}")
            done = False
            next_state = self.env.reset()
            while done == False:
                state = next_state

                state_tensor = torch.tensor(state, dtype = torch.float32)

                action_tensor = agent(state_tensor)

                next_state, reward, done, info = self.env.step(np.array(action_tensor, dtype = np.float32))