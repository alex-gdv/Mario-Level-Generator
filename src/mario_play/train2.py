
import torch

from ..common.wrappers import NoopResetEnv, MaxAndSkipEnv, WarpFrame, ScaledFloatFrame, FrameStack
from .env import Mario_Play_Env

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_env():
    env = Mario_Play_Env()
    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    env = FrameStack(env)
    return env
