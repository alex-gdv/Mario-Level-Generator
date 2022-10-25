from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from mario_play_env import Mario_Play_Env, Skip_Frame
from callbacks import Play_Callback
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


NPROC = 4
ALGOS = {0:(PPO, "ppo"), 1:(A2C, "a2c")}
DIR = "play20"

def make_env(seed, skip):
    def _f():
        env = Mario_Play_Env()
        env = Skip_Frame(env, skip=skip)
        env.seed(seed)
        return env
    return _f

def train(skip, n_steps, batch_size, algo, algo_name):
    callback = Play_Callback(model_save_freq=n_steps*NPROC*10, model_path=f"{rootpath}/models/{DIR}/{algo_name}/",
                            data_save_freq=n_steps*NPROC, data_path=f"{rootpath}/data/{DIR}/{algo_name}/") 
    env = SubprocVecEnv([make_env(seed, skip) for seed in range(NPROC)])
    # env = DummyVecEnv([lambda: Mario_Play_Env()])
    # env = VecFrameStack(env, n_stack=4, channels_order="first")
    model = algo("CnnPolicy", env, learning_rate=1e-6, n_steps=n_steps, batch_size=batch_size, verbose=1, tensorboard_log=f"{rootpath}/logs/{DIR}/")
    model.learn(total_timesteps=n_steps*NPROC*100, callback=callback)

if __name__ == "__main__":
    train(skip=4, n_steps=256, batch_size=32, algo=ALGOS[0][0], algo_name=ALGOS[0][1])
