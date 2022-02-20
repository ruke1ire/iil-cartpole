import torch
import numpy as np
import math

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
        self.global_step = 0
        self.global_supervisor_step = 0
    
    def run(self, num_of_episodes):
        avg_len = 0
        avg_sup = 0
        avg_agn = 0
        for ep_no in range(1, num_of_episodes+1):
            print(f"Episode No.: {ep_no}", end="\t")
            episode_step = 0
            episode_supervisor_step = 0
            episode_agent_step = 0
            episode_agent_step_list = []
            next_demo = False
            done = False
            next_state = self.env.reset()
            while done == False:
                demo = next_demo
                state = next_state

                state_tensor = torch.tensor(state, dtype = torch.float32).to(self.device)

                if(demo == False):
                    action_tensor = self.agent(state_tensor, noise = self.noise)
                    episode_agent_step += 1
                else:
                    action_tensor = self.expert(state_tensor, noise = self.noise)
                    episode_supervisor_step += 1

                next_state, reward, done, info = self.env.step(np.array(action_tensor.detach().cpu(), dtype = np.float32))

                if(self.expert.check_bad_state(next_state) == True):
                    next_demo = True
                else:
                    next_demo = False
                
                if(next_demo == True and demo == False) or done == True:
                    termination_flag = True
                    episode_agent_step_list.append(episode_agent_step)
                    episode_agent_step = 0
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
                    for i in range(50):
                        self.trainer.train_one_mini_batch()
                
                episode_step += 1

            self.global_step += episode_step
            self.global_supervisor_step += episode_supervisor_step
            episode_agent_step_avg = sum(episode_agent_step_list)/len(episode_agent_step_list)

            avg_len += episode_step/num_of_episodes
            avg_sup += episode_supervisor_step/num_of_episodes
            avg_agn += episode_agent_step_avg/num_of_episodes

            print(f"[Episode Length: {episode_step}]", end = "\t")
            print(f"[Supervised Steps: {episode_supervisor_step}]", end = "\t")
            print(f"[Average Agent Steps: {episode_agent_step_avg}]")

            log_dict = {
                "Episode Length": episode_step, 
                "Supervised Frames": episode_supervisor_step, 
                "Agent Frames": episode_agent_step_avg, 
                "Supervised Steps": self.global_supervisor_step,
                "Steps": self.global_step}

            self.logger.log(DataType.dict, data = log_dict, key = None)
        
        return avg_len, avg_sup, avg_agn, self.global_step, self.global_supervisor_step

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
    def __init__(self, env, agent, logger = None):
        self.env = env
        self.agent = agent
        self.logger = logger
    
    def run(self, num_of_episodes):
        avg_avg_error = 0
        for traj_no in range(1, num_of_episodes+1):
            done = False
            next_state = self.env.reset()
            sum_error = np.zeros(4)
            steps = 0
            while done == False:
                state = next_state

                state_tensor = torch.tensor(state, dtype = torch.float32)

                action_tensor = self.agent(state_tensor)

                next_state, reward, done, info = self.env.step(np.array(action_tensor.detach().cpu(), dtype = np.float32))

                normalized_abs_next_state = abs(next_state/np.array([2.4, 1, 15*math.pi/180, 1]))
                sum_error += normalized_abs_next_state

                steps += 1

            avg_error = sum_error/steps
            avg_avg_error += avg_error/num_of_episodes

            log_dict = {
                "Average Theta Error": avg_error[2], 
                "Average X Error": avg_error[0], 
                }

            self.logger.log(DataType.dict, data = log_dict, key = None)
        
        print("Average Error: ", avg_avg_error)
        return avg_avg_error