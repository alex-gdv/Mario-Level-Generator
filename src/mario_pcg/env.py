import numpy as np
import os
from datetime import datetime
from collections import deque
from gym import Env, spaces

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from mariopuzzle.generator import Generator
from mariopuzzle.mario_level_repairer.repairer import Repairer
from mariopuzzle.mario_level_repairer.CNet import CNet
from mariopuzzle.utils import *
from mariopuzzle.mario_level_repairer.level_process import arr_to_str
from map import map_string_to_json

class Mario_PCG_Env(Env):
    def __init__(self, solver, play_env, solver_timesteps, save_path, env_num=0, episode_length=100,
    diversity_num_segments=5, novelty_num_segments=10, adversarial=False, collect_data=True):
        super(Mario_PCG_Env, self).__init__()
        self.observation_shape = self.action_shape = (32)
        self.observation_space = spaces.Box(
            low=np.negative(np.ones(self.observation_shape, dtype=np.float32)),
            high=np.ones(self.observation_shape, dtype=np.float32),
            dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.negative(np.ones(self.action_shape, dtype=np.float32)),
            high=np.ones(self.action_shape, dtype=np.float32),
            dtype=np.float32)
        self.episode_length = episode_length
        self.level_height = self.window_height = self.window_width = 14
        self.level_width = self.window_width * (episode_length + 1)
        self.diversity_num_segments = diversity_num_segments
        self.novelty_num_segments = novelty_num_segments
        self.novelty_history = deque(maxlen=1000)
        self.generator = Generator()
        self.repairer = Repairer()
        self.solver = solver
        self.play_env = play_env
        self.solver_timesteps = solver_timesteps
        self.adversarial = adversarial
        self.seg_count = 0
        self.save_path = save_path
        self.env_num = env_num
        self.data = None
        if collect_data:
            self.data = []
        self.reset()

    def generate_random_segment(self):
        rand_vec = np.random.uniform(-1, 1, 32)
        segment = self.generator.generate(rand_vec)
        segment = self.repairer.repair(segment)
        return rand_vec, segment

    def save_solver(self):
        self.solver.save(f"{self.save_path}solver{self.env_num}")

    def train_solver(self, path):
        self.play_env.envs[0].env.set_level_name(path)
        self.solver.learn(self.solver_timesteps)

    def play_level(self, path):
        self.play_env.envs[0].env.set_level_name(path)
        obs = self.play_env.reset()
        while True:
            action, _ = self.solver.predict(obs)
            obs, _, done, info = self.play_env.step(action)
            if done and info[0]["win"]:
                self.play_env.close()
                return True
            elif done:
                self.play_env.close()
                return False

    def check_playability(self, segment):
        mini_level = segment
        num = min(3, len(self.recent_segments))
        for i in range(num):
            mini_level = np.concatenate((self.level[:, (self.seg_count-i-1)*self.window_width:(self.seg_count-i)*self.window_width], mini_level), 1)
        slevel = arr_to_str(mini_level)
        now = datetime.now()
        now = now.strftime("%d_%H_%M_%S")
        map_string_to_json(slevel, f"Seg-{now}", generated=True, path=self.save_path+f"segments{self.env_num}/")
        seg_name = f"LevelSeg-{now}"
        seg_path = f"../../../{self.save_path}/segments{self.env_num}/{seg_name}"
        return self.play_level(seg_path), seg_path

    def reset(self):
        if self.seg_count > 1:
            slevel = arr_to_str(self.level[:, 0:self.seg_count*self.window_width])
            now = datetime.now()
            now = now.strftime("%d_%H_%M_%S")
            level_dir = f"levels{self.env_num}/"
            map_string_to_json(slevel, f"Lvl-{now}_{self.seg_count}", generated=True, path=self.save_path+level_dir)
            saveLevelAsImage(self.level[:, self.window_width:self.seg_count*self.window_width],
                                self.save_path + level_dir + f"level_{now}")
        self.done = False
        self.info = {"P":0, "D":[], "N":[], "playable":False, "data":self.data, "adversarial":self.adversarial}
        self.rew_sum = 0
        self.steps = 0
        self.level = np.zeros((self.level_height, self.level_width), dtype=np.uint8)
        self.recent_segments = deque(maxlen=20)
        segment = np.full((14, 14), 2)
        segment[13, :] = segment[13, :] * 0
        self.level[:, 0:self.window_width] = segment
        self.recent_segments.append(lv2Map(segment))
        self.seg_count = 1
        return np.zeros(self.observation_shape, dtype=np.float32)
        # while True:
        #     self.seg_count = 0
        #     self.level = np.zeros((self.level_height, self.level_width), dtype=np.uint8)
        #     self.recent_segments = deque(maxlen=20)
        #     latent_space_vec, segment = self.generate_random_segment()
        #     self.seg_count += 1
        #     self.level[:, 0:self.window_width] = segment
        #     self.recent_segments.append(lv2Map(segment))
        #     latent_space_vec, segment = self.generate_random_segment()
        #     if self.check_playability(segment):
        #         self.seg_count += 1
        #         self.level[:, self.window_width:2*self.window_width] = segment
        #         self.recent_segments.append(lv2Map(segment))
        #         return latent_space_vec
    
    def calculate_norm_novelty(self, value):
        # code from mariopuzzle
        self.novelty_history.append(value)
        max_n = max(self.novelty_history)
        min_n = min(self.novelty_history)
        if max_n == min_n:
            return 0
        return (value-min_n)/(max_n-min_n)

    def calculate_novelty_reward(self, segment):
        if len(self.recent_segments) > 1:
            reward = 0
            segment = lv2Map(segment)
            # code from mariopuzzle
            scores = []
            for s in self.recent_segments:
                scores.append(calKLFromMap(s, segment))
            scores.sort()
            num = min(self.novelty_num_segments, len(scores))
            for i in range(num):
                reward += scores[i]
            reward /= num
            reward = self.calculate_norm_novelty(reward)
            return reward
        return 0

    def calculate_diversity_reward(self, segment):
        reward = 0
        # code from mariopuzzle
        num = min(self.diversity_num_segments, self.seg_count-1)
        fac = 1
        decay = 0.9
        s = 0
        for i in range(1, num+1):
            val = calKLFromMap(lv2Map(segment), self.recent_segments[-i])
            reward += fac * kl_fn(val, i-1)
            s += fac
            fac *= decay
        reward /= s
        return reward

    def num_gaps(self):
        return np.sum(self.level[-1,:self.window_width*self.seg_count] == 2)

    def num_enemies(self):
        return np.sum(self.level == 5)

    def save_data(self):
        self.data.append((self.info["P"], self.info["D"], self.info["N"], self.info["playable"], self.rew_sum, self.num_enemies(), self.num_gaps()))

    def step(self, action):
        segment = self.generator.generate(action)
        segment = self.repairer.repair(segment)
        playable, path = self.check_playability(segment)
        reward = 0
        self.steps += 1
        if not playable:
            if self.adversarial:
                self.train_solver(path)
                playable, _ = self.check_playability(segment)
                self.done = not playable
            else:
                self.done = True
        if playable:
            self.seg_count += 1
            rew_P = 1
            rew_D = self.calculate_novelty_reward(segment)
            rew_N = self.calculate_diversity_reward(segment)
            self.info["P"] = self.steps
            self.info["D"].append(rew_D)
            self.info["N"].append(rew_N)
            reward = rew_P + rew_D + rew_N
            if self.steps == self.episode_length:
                self.done = True
                self.info["playable"] = True
            try:
                self.level[:, (self.seg_count-1)*self.window_width:self.seg_count*self.window_width] = segment
            except ValueError:
                print(ValueError.args[0])
                print(f"seg_count: {self.seg_count}\tsteps: {self.steps}\tlen(recent_segments:{len(self.recent_segments)}")
                self.done = True
            self.recent_segments.append(lv2Map(segment))
        self.rew_sum += reward
        if self.done and self.data is not None:
            self.save_data()
        return action, reward, self.done, self.info
