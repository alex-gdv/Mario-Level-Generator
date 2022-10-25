import numpy as np
import os
import pandas as pd
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from mariopuzzle.generator import Generator
from mariopuzzle.mario_level_repairer.repairer import Repairer
from mariopuzzle.mario_level_repairer.CNet import CNet
from mariopuzzle.utils import * 
from mario_play_env import Mario_Play_Env, Skip_Frame
from mario_pcg_env2 import Mario_PCG_Env
from root import rootpath

DIR_PLAY = "play18"
ALGO = PPO
ALGO_NAME = "ppo"
TIMESTEPS = 204800
SKIP = 10

MODEL_PATH = f"{rootpath}/models/"
DATA_PATH = f"{rootpath}/data/pcg_data/ppo/"
PATH_TO_SOLVER1 = MODEL_PATH + "solver/ppo/best_model_1"
PATH_TO_SOLVER2 = MODEL_PATH + "play18/ppo/best_model_204800"
SOLVER1 = PPO.load(PATH_TO_SOLVER1)
SOLVER2 = PPO.load(PATH_TO_SOLVER2)

G = Generator()
R = Repairer()

class Random:
    def __init__(self):
        pass
    
    def load(self):
        return Random

    def predict(self):
        return np.random.uniform(-1, 1, 32), None

algos = {0:(PPO, "ppo"), 1:(Random, "random")}

def create_level_imgs(model, path):
    os.makedirs(path, exist_ok=True)
    for i in range(10):
        obs_zeros = np.zeros((32), dtype=np.float32)
        obs_random = np.random.uniform(-1, 1, 32)
        curr_level_1 = R.repair(G.generate(obs_zeros))
        curr_level_2 = R.repair(G.generate(obs_random))
        for _ in range(7):
            action, _ = model.predict(obs_zeros)
            obs_zeros = action
            curr_level_1 = np.concatenate((curr_level_1, R.repair(G.generate(action))), 1)
            action, _ = model.predict(obs_random)
            obs_random = action
            curr_level_2 = np.concatenate((curr_level_2, R.repair(G.generate(action))), 1)
        saveLevelAsImage(curr_level_1[:, 14:], path + f"level_zeros_{i}")
        saveLevelAsImage(curr_level_2[:, 14:], path + f"level_random_{i}")

def make_env1(seed, solver):
    play_env = Mario_Play_Env(collect_data=False)
    play_env = Skip_Frame(play_env, skip=SKIP)
    play_env = DummyVecEnv([lambda: play_env])
    play_env = VecFrameStack(play_env, n_stack=4, channels_order="first")
    env = Mario_PCG_Env(solver, play_env, 16*2, f"{rootpath}/data/garbage/ppo/",
                                env_num=seed, episode_length=10, diversity_num_segments=3,
                                novelty_num_segments=3, adversarial=False)
    env.seed(seed)
    return env

def make_env(seed, solver):
    def _f():
        play_env = Mario_Play_Env(collect_data=False)
        play_env = Skip_Frame(play_env, skip=SKIP)
        play_env = DummyVecEnv([lambda: play_env])
        play_env = VecFrameStack(play_env, n_stack=4, channels_order="first")
        env = Mario_PCG_Env(solver, play_env, 16*2, f"{rootpath}/data/garbage/ppo/",
                                    env_num=seed, episode_length=10, diversity_num_segments=3,
                                    novelty_num_segments=3, adversarial=False)            
        env.seed(seed)
        return env
    return _f

def mean_std(lst):
    avg = sum(lst) / len(lst)
    std = math.sqrt(sum([(num-avg)**2 for num in lst])/len(lst))
    return avg, std

def eval_model(exp, solver1=True):

    model = exp["algo"][0].load(MODEL_PATH + exp["name"]  + "/" + exp["algo"][1] + "/" + f"best_model_{exp['best_model']}")
    # create_level_imgs(model, f"{DATA_PATH}{exp['data_path']}/levels/")
    avg_p = []
    avg_d = []
    avg_n = []
    avg_num_enemies = []
    avg_num_gaps = []
    s = SOLVER1 if solver1 else SOLVER2
    # env = DummyVecEnv([make_env(seed, s) for seed in range(1)])
    env = make_env1(0, s)
    print("begin eval of ", exp["name"])
    for i in range(50):
        if i > 0:
            # avg_p.append(info[0]["data"][-1][0] / 10)
            # avg_d.append(sum(info[0]["data"][-1][1]) / counter)
            # avg_n.append(sum(info[0]["data"][-1][2]) / counter)
            # avg_num_enemies.append(info[0]["data"][-1][-2] / counter)
            # avg_num_gaps.append(info[0]["data"][-1][-1] / counter)
            avg_p.append(info["data"][-1][0] / 10)
            avg_d.append(sum(info["data"][-1][1]) / counter)
            avg_n.append(sum(info["data"][-1][2]) / counter)
            avg_num_enemies.append(info["data"][-1][-2] / counter)
            avg_num_gaps.append(info["data"][-1][-1] / counter)
        counter = 0
        obs = env.reset()
        while True:
            counter += 1
            action, _ = model.predict(obs)
            obs, _, done, info = env.step(action)
            if done:
                break
    f = open(f"eval_{exp['name']}.txt", "a")
    avg_p, std_p = mean_std(avg_p)
    avg_d, std_d = mean_std(avg_d)
    avg_n, std_n = mean_std(avg_n)
    avg_num_enemies, std_num_enemies = mean_std(avg_num_enemies)
    avg_num_gaps, std_num_gaps = mean_std(avg_num_gaps)
    f.write(f"{exp['name']}, solver1:{solver1}, avg p: {avg_p}, {std_p}, avg d: {avg_d}, {std_d}, avg n: {avg_n}, {std_n}, avg num enemies: {avg_num_enemies}, {std_num_enemies}, avg num gaps: {avg_num_gaps}, {std_num_gaps}\n")
    f.close()

if __name__ == "__main__":
    exps = [{"name":"pcg_stc_one1", "algo":algos[0], "best_model":14080, "hyperparameters":["stc", "pop=1"], "data_path":"stc_one"},
            {"name":"pcg_stc_coop_hybrid_four4", "algo":algos[0], "best_model":10240, "hyperparameters":["hybrid", "pop=4"], "data_path":"hybrid_four"},
            {"name":"pcg_coop_four4", "algo":algos[0], "best_model":10880, "hyperparameters":["coop", "pop=4"], "data_path":"coop_four"},
            {"name":"pcg_stc_four4", "algo":algos[0], "best_model":53120, "hyperparameters":["stc", "pop=4"], "data_path":"static_four"},
            {"name":"pcg_coop_one4", "algo":algos[0], "best_model":2560, "hyperparameters":["coop", "pop=1"], "data_path":"coop_one"},
            ]
    exp = {"name":"random", "algo":(Random,"random"), "best_model":14080, "hyperparameters":["stc", "pop=1"], "data_path":"random"}
    eval_model(exp, solver1=False)
    # for exp in exps:
    #   eval_model(exp, solver1=False)
    # eval_model(exps[2], solver1=False)
    # for exp in exps:
    #     create_level_imgs(exp["algo"][0].load(MODEL_PATH + exp["name"]  + "/" + exp["algo"][1] + "/" + f"best_model_{exp['best_model']}"), f"{DATA_PATH}{exp['data_path']}/levels/")
