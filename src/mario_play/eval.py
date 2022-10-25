import os
import pandas as pd
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import math

from mario_play_env import Mario_Play_Env, Skip_Frame
from root import rootpath

def mean_std(lst):
    avg = sum(lst) / len(lst)
    std = math.sqrt(sum([(num-avg)**2 for num in lst])/len(lst))
    return avg, std

class Random:
    def __init__(self):
        pass
    
    def load(self):
        return Random()

    def predict(self, _):
        return random.randint(0,9), None
    
class Right_Jump_Boost:
    def __init__(self):
        pass

    def load(self):
        return Right_Jump_Boost()

    def predict(self, _):
        return 6, None

def load_level_names():
    files = []
    res = []
    for r, d, f in os.walk(f"{rootpath}/super_mario_python/levels"):
        for file in f:
            files.append(os.path.join(r, file))
    for f in files:
        res.append(os.path.split(f)[1].split(".")[0])
    return res

def eval_agent(mpath, algo, skip, dpath):
    os.makedirs(dpath, exist_ok=True)
    data = []
    model = algo.load(mpath)
    levels = load_level_names()
    overall_avg_pct_comp = []
    overall_win_rate = []
    for level in levels:
        env = Mario_Play_Env()
        env.set_level_name(level)
        # env = Skip_Frame(env, skip=skip)
        # env = DummyVecEnv([lambda: env])
        # env = VecFrameStack(env, n_stack=4, channels_order="first")
        curr_avg_pct_comp = 0
        curr_win_rate = 0
        win = False
        for _ in range(1):
            obs = env.reset()
            while True:
                action, _ = model.predict(obs)
                obs, _, done, info = env.step(action)
                if done:
                    # pct_comp = info[0]["data"][0][1] / info[0]["data"][0][3] * 100
                    pct_comp = info["data"][0][1] / info["data"][0][3] * 100
                    curr_avg_pct_comp += pct_comp
                    overall_avg_pct_comp.append(pct_comp)
                    # win = info[0]["data"][0][4] if not win else win
                    win = info["data"][0][4] if not win else win
                    curr_win_rate += win
                    overall_win_rate.append(win)
                    break
        # print(f"{level} \twin: {win} \tpercentage completed: {pct_comp:.2f}")
        data.append((level, curr_avg_pct_comp / 10, curr_win_rate / 10, win))

    avgp, stdp = mean_std(overall_avg_pct_comp)
    avgr, stdr = mean_std(overall_win_rate)
    print(dpath, avgp, stdp, avgr, stdr)
    # data.append(("Average", overall_avg_pct_comp, overall_win_rate, False))
    df = pd.DataFrame(data)
    df.to_csv(dpath + "eval_results.csv")
    return avgp

if __name__ == "__main__":
    # exps = [("play16", "skip=30"), ("play17", "skip=60"), ("play18", "skip=10")]
    # exps = [("play10", 80000, 4), ("play11", 80000, 8), ("play11", 80000, 8), ("play12", 600000, 16),
    #             ("play13", 25000, 8), ("play15", 100000, 30), ("play16", 10240, 30), ("play17", 5120, 60),
    #             ("play18", 40960, 10)]
    # exps = [("play16", 71680, 30), ("play17", 46080, 60), ("play18", 204800, 10), ("solver", 1, 10)]
    max_pct_comp = -1
    best_exp = ""
    lst = []
    # for exp in exps:
    #     print(f"Experiment: {exp[0]}")
    #     temp = eval_agent(f"{rootpath}/models/{exp[0]}/ppo/best_model_{exp[1]}", PPO, exp[2],
    #                         f"{rootpath}/data/{exp[0]}/ppo/")
    #     lst.append((temp, exp[0]))
    #     if temp > max_pct_comp:
    #         max_pct_comp = temp
    #         best_exp = exp[0]
    
    # temp = eval_agent("", Random, 1, f"{rootpath}/data/Eval_Play/Random/")
    # if temp > max_pct_comp:
    #     max_pct_comp = temp
    #     best_exp = "Random"
    # lst.append((temp, "Random"))
    temp = eval_agent("", Right_Jump_Boost, 1, f"{rootpath}/data/Eval_Play/Right_Jump_Boost/")
    if temp > max_pct_comp:
        max_pct_comp = temp
        best_exp = "Right_Jump_Boost"
    lst.append((temp, "Right_Jump_Boost"))
    print(f"Best exp: {best_exp}, progress: {max_pct_comp}")
    print(lst.sort())