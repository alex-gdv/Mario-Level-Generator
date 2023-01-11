
import torch

from ..common.wrappers import NoopResetEnv, MaxAndSkipEnv, WarpFrame, ScaledFloatFrame, FrameStack
from .env import Mario_Play_Env

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_env(visuals=False):
    env = Mario_Play_Env(visuals=visuals)
    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    env = FrameStack(env, k=4)
    return env
