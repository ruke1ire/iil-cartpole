from model import *
from algorithms import *
from utils import *
from trainer import *
from env import * 
from logger import *

import torch
import os
import sys
import math
import pickle

run_name = "slrl_training_noisy"
run_id = 7

model_save_path = f'data/model/{run_name}/{run_id}'
logs_save_path = f'data/logs/{run_name}'
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(logs_save_path, exist_ok=True)

env = MaxStepContinuousCartPoleEnv()

actor_model = NNActor()
critic_model = NNCritic()

intervention_threshold = dict(
    x = 2.0,
    x_dot = 3.0,
    theta = 12*2*math.pi/360,
    theta_dot = 4.0
)

expert_model = NNActor(intervention_thresholds=intervention_threshold)
expert_model.load_state_dict(torch.load("data/model/expert_training/5/12700.pth"))
expert_model.eval()

trainer_config = dict(
    actor_optimizer_name = "Adam",
    actor_optimizer_kwargs = dict(lr = 1e-6, weight_decay = 1e-4),
    critic_optimizer_name = "Adam",
    critic_optimizer_kwargs = dict(lr = 1e-4),
    discount = 0.99, 
    tau = 0.005, 
    noise = 0.2, 
    actor_update_period = 2,
    batch_size = 24,
    )

algorithm_config = dict(
    replay_buffer_size = int(1e6),
    noise = 0.2)

replay_buffer = ReplayBuffer(algorithm_config['replay_buffer_size'])

trainer = SLRL(
    actor_model,
    critic_model,
    replay_buffer = replay_buffer,
    only_rl = False,
    device = 'cpu',
    **trainer_config)

logger = WandbLogger(name = run_name, id = run_id, config_dict = dict(trainer = trainer_config, algorithm = algorithm_config, intervention_threshold = intervention_threshold))

algo = IIL_algorithm(env, trainer, expert_model, replay_buffer, noise = algorithm_config['noise'], logger = logger)

logs = []
save_id = 0
N = 1
while True:
    avg_len, avg_sup, avg_agn, steps, supervised_steps = algo.run(N)
    logs.append((avg_len, avg_sup, avg_agn, steps, supervised_steps))
    pickle.dump(logs, open(f'{logs_save_path}/{run_id}.p',"wb"))
    if(avg_agn >= 2900):
        torch.save(trainer.actor_model.state_dict(), f'{model_save_path}/{save_id}.pth')
        sys.exit()
    save_id += N