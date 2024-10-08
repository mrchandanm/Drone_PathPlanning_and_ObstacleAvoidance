"""
2D Quadcopter AI by Alexandre Sajus

More information at:
https://github.com/AlexandreSajus/Quadcopter-AI

This is a gym environment based on drone_game (see Human/drone_game.py for details)
It is to be used with a DQN agent
The goal is to reach randomly positoned targets
"""

import os
from math import sin, cos, pi, sqrt
from random import randrange

import numpy as np
import gym
from gym import spaces

import pygame
from pygame.locals import *

class droneEnv(gym.Env):
    def __init__(self, render_every_frame, mouse_target):
        super(droneEnv, self).__init__()

        self.render_every_frame = render_every_frame
        self.mouse_target = mouse_target

        # Initialize Pygame, load sprites
        pygame.init()
        self.screen = pygame.display.set_mode((1300, 500))
        self.FramePerSec = pygame.time.Clock()

        self.player = pygame.image.load(os.path.join("assets/sprites/drone_old.png"))
        self.player.convert()

        self.target = pygame.image.load(os.path.join("assets/sprites/target_old.png"))
        self.target.convert()

        self.bird = pygame.image.load(os.path.join("assets/sprites/bird_png.png"))
        self.bird.convert()

        pygame.font.init()
        self.myfont = pygame.font.SysFont("Comic Sans MS", 20)

        # Physics constants
        self.FPS = 60
        self.gravity = 0.08
        self.thruster_amplitude = 0.04
        self.diff_amplitude = 0.003
        self.thruster_mean = 0.04
        self.mass = 1
        self.arm = 25

        # Initialize variables
        (self.a, self.ad, self.add) = (0, 0, 0)
        (self.x, self.xd, self.xdd) = (50, 0, 0)
        (self.y, self.yd, self.ydd) = (300, 0, 0)
        self.xt = randrange(1100, 1200)
        self.yt = randrange(150, 400)

        self.reset_bird()  # New function to randomize bird's position

        # Initialize game variables
        self.target_counter = 0
        self.reward = 0
        self.time = 0
        self.time_limit = 20
        if self.mouse_target is True:
            self.time_limit = 1000

        # Action space: thrust amplitude and thrust difference in float values between -1 and 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        # Observation space now includes the bird's position (9 variables)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,))

    def reset(self):
        # Reset variables
        (self.a, self.ad, self.add) = (0, 0, 0)
        (self.x, self.xd, self.xdd) = (50, 0, 0)
        (self.y, self.yd, self.ydd) = (300, 0, 0)
        self.xt = randrange(1100, 1250)
        self.yt = randrange(150, 400)

        self.reset_bird()  # Reset bird position when resetting environment

        self.target_counter = 0
        self.reward = 0
        self.time = 0

        return self.get_obs()

    def reset_bird(self):
        """Randomize bird's initial position and reset its movement."""
        self.xb = 1350  # Start from right
        self.yb = randrange(100, 400)  # Random y-position

    def get_obs(self) -> np.ndarray:
        """
        Calculates the observations including bird's position
        """
        angle_to_up = self.a / 180 * pi
        velocity = sqrt(self.xd**2 + self.yd**2)
        angle_velocity = self.ad
        distance_to_target = (
            sqrt((self.xt - self.x) ** 2 + (self.yt - self.y) ** 2) / 500
        )
        angle_to_target = np.arctan2(self.yt - self.y, self.xt - self.x)
        angle_target_and_velocity = np.arctan2(
            self.yt - self.y, self.xt - self.x
        ) - np.arctan2(self.yd, self.xd)
        distance_to_bird = (
            sqrt((self.xb - self.x) ** 2 + (self.yb - self.y) ** 2) / 500
        )  # Distance to bird
        angle_to_bird=np.arctan2(self.yb - self.y, self.xb - self.x)
        angle_bird_and_velocity = np.arctan2(
            self.yb - self.y, self.xb - self.x
        ) - np.arctan2(self.yd, self.xd)


        return np.array(
            [
                angle_to_up,
                velocity,
                angle_velocity,
                distance_to_target,
                angle_to_target,
                angle_target_and_velocity,
                distance_to_target,
                distance_to_bird,  # Bird distance in observations
                angle_to_bird,  # Bird's x position
                angle_bird_and_velocity,  # Bird's y position
            ]
        ).astype(np.float32)

    def step(self, action):
        self.reward = 0.0
        (action0, action1) = (action[0], action[1])

        # Act every 5 frames
        for _ in range(5):
            self.time += 1 / 60

            if self.mouse_target is True:
                self.xt, self.yt = pygame.mouse.get_pos()

            # Initialize accelerations
            self.xdd = 0
            self.ydd = self.gravity
            self.add = 0
            thruster_left = self.thruster_mean
            thruster_right = self.thruster_mean

            thruster_left += action0 * self.thruster_amplitude
            thruster_right += action0 * self.thruster_amplitude
            thruster_left += action1 * self.diff_amplitude
            thruster_right -= action1 * self.diff_amplitude

            # Calculating accelerations
            self.xdd += (
                -(thruster_left + thruster_right) * sin(self.a * pi / 180) / self.mass
            )
            self.ydd += (
                -(thruster_left + thruster_right) * cos(self.a * pi / 180) / self.mass
            )
            self.add += self.arm * (thruster_right - thruster_left) / self.mass

            self.xd += self.xdd
            self.yd += self.ydd
            self.ad += self.add
            self.x += self.xd
            self.y += self.yd
            self.a += self.ad

            # Bird movement from right to left
            self.xb -= 1
            if self.xb < -50:
                self.reset_bird()

            dist_to_target = sqrt((self.x - self.xt) ** 2 + (self.y - self.yt) ** 2)
            dist_to_bird = sqrt((self.x - self.xb) ** 2 + (self.y - self.yb) ** 2)

            # Reward for reaching the target
            if dist_to_target < 50:
                self.reward += 100
                self.xt = randrange(1150, 1250)
                self.yt = randrange(150, 400)
                self.target_counter += 1
                self.reset()

            # Penalty if drone crashes into the bird
            if dist_to_bird < 50:
                self.reward -= 500
                done = True
                break

            # Reward per step survived
            self.reward += 1 / 60
            # Penalty for distance from target
            self.reward -= dist_to_target / (100 * 60)

            if self.time > self.time_limit:
                done = True
                break
            elif dist_to_target > 1500:
                self.reward -= 1000
                done = True
                break
            else:
                done = False

            if self.render_every_frame:
                self.render("yes")

        info = {}
        return self.get_obs(), self.reward, done, info

    def render(self, mode):
        pygame.event.get()
        self.screen.fill(0)
        self.screen.blit(self.target, (self.xt - self.target.get_width() // 2, self.yt - self.target.get_height() // 2))
        player_copy = pygame.transform.rotate(self.player, self.a)
        self.screen.blit(player_copy, (self.x - player_copy.get_width() // 2, self.y - player_copy.get_height() // 2))
        self.screen.blit(self.bird, (self.xb, self.yb))

        textsurface = self.myfont.render("Collected: " + str(self.target_counter), False, (255, 255, 255))
        self.screen.blit(textsurface, (20, 20))
        textsurface3 = self.myfont.render("Time: " + str(int(self.time)), False, (255, 255, 255))
        self.screen.blit(textsurface3, (20, 50))

        pygame.display.update()
        self.FramePerSec.tick(self.FPS)
    def close(self):
        pass

       
