from model import *
from algorithms import *
from utils import *
from trainer import *
from env import * 
from logger import *

import torch
import pickle
import os
import math
import sys

run_name = "bc_training"
run_id = 101

model_save_path = f'data/model/{run_name}/{run_id}'
logs_save_path = f'data/logs/{run_name}'
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(logs_save_path, exist_ok=True)

env = MaxStepContinuousCartPoleEnv()

actor_model = NNActor()
critic_model = NNCritic()

expert_model = NNActor()
expert_model.load_state_dict(torch.load("data/model/expert_training/5/12700.pth"))
expert_model.eval()

trainer_config = dict(
    actor_optimizer_name = "Adam",
    actor_optimizer_kwargs = dict(lr = 1e-3, weight_decay = 1e-4),
    batch_size = 24,
    )

algorithm_config = dict(
    replay_buffer_size = int(1e6),
    noise = 0.2)

replay_buffer = ReplayBuffer(algorithm_config['replay_buffer_size'])

trainer = SL(
    actor_model,
    replay_buffer = replay_buffer,
    only_supervised = True,
    device = 'cpu',
    **trainer_config)

logger = WandbLogger(name = run_name, id = run_id, config_dict = dict(trainer = trainer_config, algorithm = algorithm_config))

algo = SL_algorithm(env, trainer, expert_model, replay_buffer, noise = algorithm_config['noise'], logger = logger)

logs = []
save_id = 0
N = 1
while True:
    avg_agn, supervised_steps = algo.run(N)
    logs.append((avg_agn, supervised_steps))
    pickle.dump(logs, open(f'{logs_save_path}/{run_id}.p', "wb"))
    torch.save(trainer.actor_model.state_dict(), f'{model_save_path}/{save_id}.pth')
    save_id += N
