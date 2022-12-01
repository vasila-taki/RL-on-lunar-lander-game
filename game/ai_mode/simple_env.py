import gym
import numpy as np
import math
import pygame
import sys
import operator
from gym import spaces
from game_mode.game import Game
from game_mode.game import Config

def check_press_quit():
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()


class Env(gym.Env):
    
	def __init__(self):
		self.game = Game('ai')
		self.action_space = gym.spaces.Discrete(4)
	
		low = np.array(
			[
				# these are bounds for position
				# realistically the environment should have ended
				# long before we reach more than 50% outside
				-1.5,
				-1.5,
				-3.5,
				-3.5,
				-math.pi / 2,
				-(Config.ROTATE_ANGLE + 1),
				-0.0,
				-0.0
			]).astype(np.float32)
        
		high = np.array(
			[
				# these are bounds for position
				# realistically the environment should have ended
				# long before we reach more than 50% outside
				1.5,
				1.5,
				3.5,
				3.5,
				math.pi / 2,
				Config.ROTATE_ANGLE + 1,	
				1.0,	
				1.0
			]).astype(np.float32)

		# useful range is -1 .. +1, but spikes can be higher
		self.observation_space = spaces.Box(low, high)


	def step(self, action):

		self.game.lander.next_action(action)
		rt, se = self.game.update()

		if(se == 1):
			w = -Config.ROTATE_ANGLE
		elif(se == 2):
			w = Config.ROTATE_ANGLE
		else:
			w = 0
     
		# lander.y = 0 is on the top of the screen 
		state = [
			(self.game.lander.x - Config.SCREEN_WIDTH / 2) / (Config.SCREEN_WIDTH / 2),
			((self.game.environment.landing_pad_y * self.game.environment.terrain_block_size) - (self.game.lander.y + (self.game.lander.lander_height / 2)))/ (self.game.environment.landing_pad_y * self.game.environment.terrain_block_size),
			self.game.lander.vx,
			self.game.lander.vy,
			w,
			(self.game.lander.angle / 90) * (math.pi / 2),
			1.0 if self.game.lander.left_contact_ok else 0.0,
			1.0 if self.game.lander.right_contact_ok else 0.0
		]

		assert len(state) == 8

		# Encourage lander to use as little fuel as possible
		# total fuel = 500
		fuel_conservation = self.game.lander.fuel / 500

		reward = 0.0

        # Encourage agent to approach the surface instead of hanging in the air
		if(state[3] >= 0):
			distance_reward = 1 - abs(state[1])**0.5
			reward = distance_reward * fuel_conservation

		done = False

		if((self.game.lander.exploded) and (not self.game.lander.hit_terrain)):
			done = True
			reward = -100.0
			#print(state)
			print("YOU LOST, out of screen or fuel")

		if (self.game.lander.hit_terrain):
			done = True
			#print(state)
            
			if(self.game.lander.landed):
				# Land successfully; give a big reward
				landing_reward = 100.0
				# Multiply percentage of remaining fuel
				reward = landing_reward * fuel_conservation
				print("YOU WON")
			elif(self.game.lander.velocity_ok and self.game.lander.angle_ok):
				reward = -10.0
				print("YOU LOST, out of landing pad")
			else:
				reward = -100.0
				print("YOU LOST, lander crashed")
        
		info = {}
		return np.array(state, dtype=np.float32), reward, done, info

    
	def render(self):

		self.game.render()
		check_press_quit()


	def reset(self):
        
		self.prev_shaping = None
		self.game.reset()
		self.game.start()
		self.game.run_game()

		state = [
			(self.game.lander.x - Config.SCREEN_WIDTH / 2) / (Config.SCREEN_WIDTH / 2), 
			((self.game.environment.landing_pad_y * self.game.environment.terrain_block_size) - (self.game.lander.y + (self.game.lander.lander_height / 2)))/ (self.game.environment.landing_pad_y * self.game.environment.terrain_block_size), 
			self.game.lander.vx, 
			self.game.lander.vy, 	
			(self.game.lander.angle / 90) * (math.pi / 2),
			0, 
			1.0 if self.game.lander.left_contact_ok else 0.0, 
			1.0 if self.game.lander.right_contact_ok else 0.0]    

		return state




