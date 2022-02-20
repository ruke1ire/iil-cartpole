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
import glob

run_name = "rl_training_noisy"
run_id = 3

model_load_path = f'data/model/{run_name}/{run_id}'
logs_save_path = f'data/logs/{run_name}/test'
os.makedirs(logs_save_path, exist_ok=True)
models = sorted(glob.glob(f'{model_load_path}/*'))
print(models)

env = MaxStepContinuousCartPoleEnv()

actor_model = NNActor()
actor_model.load_state_dict(torch.load(models[-1]))

logger = WandbLogger(name = run_name+"_test", id = run_id, config_dict = dict())

algo = Test_algorithm(env, actor_model, logger = logger)

logs = []
avg_error = algo.run(100)
logs.append((avg_error))
pickle.dump(logs, open(f'{logs_save_path}/{run_id}.p',"wb"))
sys.exit()
