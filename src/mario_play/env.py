import numpy as np
import pygame
import os
import random
from gym import Env, spaces, Wrapper
from torchvision import transforms

from common.utils import *
from super_mario_python.classes.Dashboard import Dashboard
from super_mario_python.classes.Level import Level
from super_mario_python.classes.Sound import Sound
from super_mario_python.entities.MarioAI import MarioAI

class Mario_Play_Env():
    def __init__(self, visuals=False, collect_data=False):
        super(Mario_Play_Env, self).__init__()
        self.observation_shape = (1, 84, 84)
        self.observation_space = spaces.Box(
                            low=np.zeros(self.observation_shape, dtype=np.uint8),
                            high=np.full(self.observation_shape, 255, dtype=np.uint8),
                            dtype=np.uint8)
        self.action_space = spaces.Discrete(10)
        self._level_name = None
        self._visuals = visuals
        if not self._visuals:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"
        self.data = None
        if collect_data:
            # "rew_sum", "blocks_travelled", "steps", "level_length", "win"
            self.data = []

    def set_level_name(self, name):
        self._level_name = name
    
    def reset(self):
        pygame.init()
        self.done = False
        self.info = {"game_over":False, "max_actions": False, "win":False, "data":self.data}
        self.steps = 0
        self.curr_seg = 1
        self.curr_x = 0
        self.max_x = 0
        self.rew_sum = 0
        self.screen = pygame.display.set_mode(WINDOW_SIZE) if self._visuals else pygame.display.set_mode(WINDOW_SIZE, flags=pygame.HIDDEN)
        self.dashboard = Dashboard(f"{rootpath}/super_mario_python/img/font.png", 8, self.screen)
        self.sound = Sound()
        self.level = Level(self.screen, self.sound, self.dashboard)
        if self._level_name is None:
            level_names = load_level_names()
            random_level = random.randint(0, len(level_names)-1)
            name = level_names[random_level]
        else:
            name = self._level_name
        self.level.loadLevel(name) 
        self.mario = MarioAI(0, 0, self.level, self.screen, self.dashboard, self.sound)
        self.clock = pygame.time.Clock()
        return self.get_observation()

    def get_observation(self):
        frame = pygame.surfarray.array3d(self.screen.copy()).swapaxes(0, 1)
        preprocess = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Grayscale(),
                                    transforms.Resize((84, 84)),
                                    transforms.PILToTensor()])
        obs = preprocess(frame).numpy()
        return obs

    def get_reward(self):
        # reward 1
        reward = self.mario.rect.x - self.curr_x
        # reward 2
        # reward = self.mario.rect.x - self.curr_x if self.mario.rect.x != self.curr_x else -5
        # reward 3
        # reward = 1 if self.mario.rect.x > self.curr_x else -1
        # reward 4
        # reward = 0
        # if self.curr_x // 14 * 32 * self.curr_seg >= 1:
        #     reward = 10
        #     self.curr_seg += 1
        self.curr_x = self.mario.rect.x
        if self.mario.restart:
            self.info["game_over"] = True
            self.done = True
            reward -= 100
        elif self.steps > self.level.levelLength * 32:
            self.info["max_actions"] = True
            self.done = True
            reward -= 100
        elif self.mario.rect.x  + 32 == self.level.levelLength * 32:
            self.info["win"] = True
            self.done = True
            reward += 100
        reward = max(-100, min(100, reward))
        return reward

    def save_data(self):
        # blocks_travelled = math.ceil((self.max_x + 32) / 32)
        self.data.append((self.rew_sum, self.max_x, self.steps, self.level.levelLength*32, self.info["win"]))

    def step(self, action):
        self.mario.input.receiveInput(action)
        pygame.event.pump()
        pygame.display.set_caption("Super Mario running with {:d} FPS".format(int(self.clock.get_fps())))
        self.level.drawLevel(self.mario.camera)
        self.dashboard.update()
        self.mario.update()
        pygame.display.update()
        max_frame_rate = 60
        self.clock.tick(max_frame_rate)
        self.steps += 1
        reward = self.get_reward()
        obs = self.get_observation()
        self.rew_sum += reward
        if self.curr_x > self.max_x:
            self.max_x = self.curr_x
        if self.done and self.data is not None:
            self.save_data()
        return obs, reward, self.done, self.info
    
    def close(self):
        pygame.quit()
