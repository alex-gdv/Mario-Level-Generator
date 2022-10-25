import os
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from mario_play_env import Mario_Play_Env, Skip_Frame
from root import rootpath

# play10: skip=4 reward=1 (wrong n_steps for a2c)
# play11: skip=8 reward=1 (wrong n_steps for a2c)
# play12: skip=16 reward=1
# play13: skip=8 reward=1
# play14: skip=60 reward=1
# play15: skip=30
###################################################
# changed steps to adjust for higher frame skip value
# new total timesteps = 10,000
# new n_steps = 32
# new batch size = 4
# new save_freq = 1,000
###################################################
# play16: skip=30, steps=256, batch size=16
# play17: skip=60, steps=128, batch size=8
# play18: skip=10, steps=512, batch size=32

algos = {0:(PPO, "ppo"), 1:(A2C, "a2c")}
DIR = "play17"
SKIP = 60

def load_level_names():
    files = []
    res = []
    for r, d, f in os.walk(f"{rootpath}/super_mario_python/levels"):
        for file in f:
            files.append(os.path.join(r, file))
    for f in files:
        res.append(os.path.split(f)[1].split(".")[0])
    return res

def visual_test(timesteps, algo, algo_name, path=None):
    if path is not None:
        env = Mario_Play_Env(visuals=True, collect_data=False)
        env.set_level_name(path)
        env = DummyVecEnv([lambda: Skip_Frame(env, skip=SKIP)])
    else:
        env = DummyVecEnv([lambda: Skip_Frame(Mario_Play_Env(visuals=True, collect_data=False), skip=SKIP)])
    env = VecFrameStack(env, n_stack=4, channels_order="first")
    model = algo.load(f"{rootpath}/models/{DIR}/{algo_name}/best_model_{timesteps}", env)
    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
            break

def test(timesteps, algo, algo_name):
    log = open(f"{rootpath}/mario_play_test_logs/{algo_name}_{timesteps}.txt", "w")
    env = Mario_Play_Env()
    model = algo.load(f"{rootpath}/saved_models/{algo_name}/best_model_{timesteps}", env)
    levels = load_level_names()
    for level in levels:
        env.set_level_name(level)
        obs = env.reset()
        while True:
            action, _ = model.predict(obs)
            obs, _, done, info = env.step(action)
            if done:
                if info["level completed"]:
                    line = f"{level}: completed\n"
                elif info["game over"]:
                    line = f"{level}: game over\n"
                else:
                    line = f"{level}: too many actions\n"
                log.write(line)
                break
    log.close()

if __name__ == "__main__":
    # demo 1
    # visual_test(46080, algos[0][0], algos[0][1], "level1-1")
    # demo 2
    # visual_test(46080, algos[0][0], algos[0][1], "level3-1")
    # demo 3
    # visual_test(46080, algos[0][0], algos[0][1], "level8-1")
    # demo 4
    # visual_test(46080, algos[0][0], algos[0][1], "levelLvl-coop4")
    # demo 5
    visual_test(46080, algos[0][0], algos[0][1], "levelLvl-static4")
