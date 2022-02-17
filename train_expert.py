from model import *
from algorithms import *
from utils import *
from trainer import *
from env import * 
from logger import *

import torch
import os

run_name = "expert_training"
run_id = 5

save_path = f'data/model/{run_name}/{run_id}'
os.makedirs(save_path, exist_ok=True)

env = MaxStepContinuousCartPoleEnv()

actor_model = NNActor()
critic_model = NNCritic()

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
    only_rl = True,
    device = 'cpu',
    **trainer_config)

logger = WandbLogger(name = run_name, id = run_id, config_dict = dict(trainer = trainer_config, algorithm = algorithm_config))

algo = RL_algorithm(env, trainer, replay_buffer, noise = algorithm_config['noise'], logger = logger)

save_id = 0
N = 10
while True:
    avg_len = algo.run(N)
    if(avg_len == 3000):
        torch.save(trainer.actor_model.state_dict(), f'{save_path}/{save_id}.pth')
    save_id += N