import sys
from random import choice
import pygame
import os
from pygame.locals import *
import random
import math


class Environment():
    
    def __init__(self):

        self.planet_high_low_image = pygame.image.load('game_mode/planet_high_low.png').convert_alpha()
        self.planet_high_mid_image = pygame.image.load('game_mode/planet_high_mid.png').convert_alpha()
        self.planet_low_high_image = pygame.image.load('game_mode/planet_low_high.png').convert_alpha()
        self.planet_low_mid_image = pygame.image.load('game_mode/planet_low_mid.png').convert_alpha()
        self.planet_mid_image = pygame .image.load('game_mode/planet_mid.png').convert_alpha()
        self.planet_mid_high_image = pygame.image.load('game_mode/planet_mid_high.png').convert_alpha()
        self.planet_mid_low_image = pygame.image.load('game_mode/planet_mid_low.png').convert_alpha()
        self.planet_blank_image = pygame.image.load('game_mode/planet_blank.png').convert_alpha()
        self.planet_landing_block_image = pygame.image.load('game_mode/planet_landing_block.png').convert_alpha()

        # Initialise variables
        self.terrain_block_size = self.planet_blank_image.get_rect().width
        # arithmos apo terrain blocks
        self.terrain_blocks = int(Config.SCREEN_WIDTH / self.terrain_block_size)

        self.planet = []


    def new_level(self, lvl):

        if lvl <= len(Config.levels):
            self.landing_blocks = Config.levels[lvl]['landing_blocks']
        else:      
            self.landing_blocks = Config.levels[len(Config.levels)]['landing_blocks']


        # Random planet, landing pad and start location
        # landing pad start: arithmos pou deixnei sto terrain block pou einai to 1o tou landing pad
        self.landing_pad_start = int((self.terrain_blocks / 2) - (self.landing_blocks / 2))
        self.setup_planet()
        
        # landing begins in the area outside the landing pad's x coordinates
        self.lander_column = self.random_lander_loc()
        lander_x = self.lander_column * self.terrain_block_size

        return lander_x


    # Set up a planet terrain as list
    def setup_planet(self):

        del self.planet[:]
            
        self.terrain_choice = ''
            
        self.planet_images = {'hi_lo': self.planet_high_low_image,
                               'hi_mid': self.planet_high_mid_image,
                               'lo_hi': self.planet_low_high_image,
                               'lo_mid': self.planet_low_mid_image,
                               'mid': self.planet_mid_image,
                               'mid_lo': self.planet_mid_low_image,
                               'mid_hi': self.planet_mid_high_image,
                               'blank': self.planet_blank_image,
                               'landing': self.planet_landing_block_image} 

        self.new_block_last_mid = ['mid', 'higher', 'lower']

        self.new_block_last_higher = ['higher', 'higher', 'higher', 'higher',
                                       'higher', 'higher', 'higher', 'higher',
                                       'higher', 'higher', 'mid', 'lower']
            
        self.new_block_last_lower = ['lower', 'lower', 'lower', 'lower',
                                      'lower', 'lower', 'lower', 'lower',
                                      'lower', 'lower', 'mid', 'higher']
        
        # initialization
        self.landing_pad_y = int(Config.MAX_TERRAIN_HEIGHT - ((Config.MAX_TERRAIN_HEIGHT - Config.MIN_TERRAIN_HEIGHT) / 2))

        for block in range(self.terrain_blocks):
            self.terrain_block = {'terrain_type': self.planet_images['landing'],'terrain_y': self.landing_pad_y}
            self.planet.append(self.terrain_block)

        self.last_block = 'mid'
        self.terrain_y_coord = self.landing_pad_y

        for block in range(int(self.terrain_blocks / 2), -1, -1):
            if ((block <= self.landing_pad_start - 1) or (self.landing_pad_start + self.landing_blocks <= block)):
                if self.last_block == 'mid':
                    self.new_block = random.choice(self.new_block_last_mid)
                elif self.last_block == 'higher':
                    self.new_block = random.choice(self.new_block_last_higher)
                elif self.last_block == 'lower':
                    self.new_block = random.choice(self.new_block_last_lower)

                if self.last_block == 'mid':
                    if self.new_block == 'mid':
                        self.terrain_choice = self.planet_images['mid']
                
                    elif self.new_block == 'higher':
                        self.terrain_choice = self.planet_images['mid_lo']
                    elif self.new_block == 'lower':
                        self.terrain_choice = self.planet_images['mid_hi']
                elif self.last_block == 'higher':
                    if self.new_block == 'mid':
                        self.terrain_choice = self.planet_images['lo_mid']                
                    elif self.new_block == 'higher':
                        self.terrain_choice = self.planet_images['hi_lo']
                        self.terrain_y_coord -= 1
                    elif self.new_block == 'lower':
                        self.terrain_choice = self.planet_images['lo_hi']
                elif self.last_block == 'lower':
                    if self.new_block == 'mid':
                        self.terrain_choice = self.planet_images['hi_mid']
                    elif self.new_block == 'higher':
                        self.terrain_choice = self.planet_images['hi_lo']
                    elif self.new_block == 'lower':
                        self.terrain_choice = self.planet_images['lo_hi']
                        self.terrain_y_coord += 1

                if self.terrain_y_coord < Config.MIN_TERRAIN_HEIGHT:
                    self.terrain_y_coord = Config.MIN_TERRAIN_HEIGHT
                    self.terrain_choice = self.planet_images['hi_mid']
                    self.new_block = random.choice(self.new_block_last_lower)
                
                if self.terrain_y_coord > Config.MAX_TERRAIN_HEIGHT - 1:
                    self.terrain_y_coord = Config.MAX_TERRAIN_HEIGHT - 1
                    self.terrain_choice = self.planet_images['lo_mid']
                    self.new_block = random.choice(self.new_block_last_higher)    
            
                self.terrain_block = {'terrain_type': self.terrain_choice,'terrain_y': self.terrain_y_coord}

                self.planet[block] = self.terrain_block
                
                self.last_block = self.new_block 

        self.last_block = 'mid'
        self.terrain_y_coord = self.landing_pad_y

        for block in range(int(self.terrain_blocks / 2), self.terrain_blocks):
            if ((block <= self.landing_pad_start - 1) or (self.landing_pad_start + self.landing_blocks <= block)):
                if self.last_block == 'mid':
                    self.new_block = random.choice(self.new_block_last_mid)
                elif self.last_block == 'higher':
                    self.new_block = random.choice(self.new_block_last_higher)
                elif self.last_block == 'lower':
                    self.new_block = random.choice(self.new_block_last_lower)

                if self.last_block == 'mid':
                    if self.new_block == 'mid':
                        self.terrain_choice = self.planet_images['mid']
                
                    elif self.new_block == 'higher':
                        self.terrain_choice = self.planet_images['mid_hi']
                    elif self.new_block == 'lower':
                        self.terrain_choice = self.planet_images['mid_lo']
                elif self.last_block == 'higher':
                    if self.new_block == 'mid':
                        self.terrain_choice = self.planet_images['hi_mid']                
                    elif self.new_block == 'higher':
                        self.terrain_choice = self.planet_images['lo_hi']
                        self.terrain_y_coord -= 1
                    elif self.new_block == 'lower':
                        self.terrain_choice = self.planet_images['hi_lo']
                elif self.last_block == 'lower':
                    if self.new_block == 'mid':
                        self.terrain_choice = self.planet_images['lo_mid']
                    elif self.new_block == 'higher':
                        self.terrain_choice = self.planet_images['lo_hi']
                    elif self.new_block == 'lower':
                        self.terrain_choice = self.planet_images['hi_lo']
                        self.terrain_y_coord += 1

                if self.terrain_y_coord < Config.MIN_TERRAIN_HEIGHT:
                    self.terrain_y_coord = Config.MIN_TERRAIN_HEIGHT
                    self.terrain_choice = self.planet_images['hi_mid']
                    self.new_block = random.choice(self.new_block_last_lower)
                
                if self.terrain_y_coord > Config.MAX_TERRAIN_HEIGHT - 1:
                    self.terrain_y_coord = Config.MAX_TERRAIN_HEIGHT - 1
                    self.terrain_choice = self.planet_images['lo_mid']
                    self.new_block = random.choice(self.new_block_last_higher)    
            
                self.terrain_block = {'terrain_type': self.terrain_choice,'terrain_y': self.terrain_y_coord}

                self.planet[block] = self.terrain_block
                
                self.last_block = self.new_block 

    
    # Generate a random starting x column for the lander
    # arithmos pou deixnei se ena block that is not within the landing pad or the buffer zones
    def random_lander_loc(self):
        
        maximum_right_col = self.terrain_blocks - Config.RIGHT_BUFFER_ZONE - self.landing_blocks
        lander_column = random.randint(Config.LEFT_BUFFER_ZONE, maximum_right_col)
        
        while self.landing_pad_start <= lander_column <= self.landing_pad_start + self.landing_blocks:
            lander_column = random.randint(Config.LEFT_BUFFER_ZONE, maximum_right_col)
        
        return lander_column

       
class Lander:

    def __init__(self):
        
        # Load images
        self.lander_image = pygame.image.load('game_mode/lander.png').convert_alpha()
        self.thrust_image = pygame.image.load('game_mode/thrust_lander.png').convert_alpha()
        self.exploded_lander_image = pygame.image.load('game_mode/exploded_lander.png').convert_alpha()
        
        # Load sounds
        self.thrust_sound = pygame.mixer.Sound('game_mode/thrust.ogg')
        self.explosion_sound = pygame.mixer.Sound('game_mode/explosion.ogg')
        self.landed_sound = pygame.mixer.Sound('game_mode/landed.ogg')

        # Initialise variables
        self.display_lander_image = self.lander_image
        self.lander_rect = self.display_lander_image.get_rect()
        self.lander_width = self.lander_image.get_rect().width
        self.lander_height = self.lander_image.get_rect().height

        self.mass = 5
        self.gravity = Config.KGRAVITY
        self.engine_power = Config.ENGINE_POWER


    def next_action(self, action):
        # SPACE key pressed - rocket thrust
        if action == 3:
            self.rocket_thrust = True
        else: 
            self.rocket_thrust = False
                    
        # Left or right key pressed - rotate ship, ROTATE_ANGLE = 4
        if action == 2:
            self.side_engines = 2
                        
        elif action == 1:
            self.side_engines = 1

        else: 
            self.side_engines = 0

        
    def new_level(self, lvl, lander_x):
        
        if lvl < len(Config.levels):
            self.fuel = Config.levels[lvl]['fuel']
            self.landing_velocity = Config.levels[lvl]['landing_velocity']
            self.landing_angle = Config.levels[lvl]['landing_angle']
        else:            
            self.fuel = Config.levels[len(Config.levels)]['fuel']            
            self.landing_velocity = Config.levels[len(Config.levels)]['landing_velocity']
            self.landing_angle = Config.levels[len(Config.levels)]['landing_angle']
                    
        # change level variables
        self.angle = 0
        self.x = lander_x
        self.y = Config.LANDER_START_Y
        self.vx = 0
        self.vy = 0

        self.rocket_thrust = False
        self.side_engines = 0
        
        self.exploded = False
        self.left_contact_ok = False
        self.right_contact_ok = False
        self.hit_terrain = False
        self.landed = False
        self.fuel_left = True
        self.velocity_ok = True
        self.landing_location_ok = True
        self.angle_ok = True
        self.on_screen = True

    
    # Update lander location from previous action
    def update_location(self):
        
        self.x += self.vx * Config.DT
        self.lander_left_x = self.x - self.lander_width / 2
        self.lander_right_x = self.x + self.lander_width / 2
        
        self.y += self.vy * Config.DT
        self.lander_top_y = self.y - self.lander_width / 2
        
        self.lander_rect.centerx = self.x
        self.lander_rect.centery = self.y


    # Check lander has not left screen left, right or top- if yes it explodes
    def is_in_screen(self):
        if self.lander_left_x < 0 or self.lander_right_x > Config.SCREEN_WIDTH:
            self.vx = 0
            self.exploded = True
            self.on_screen = False
            #print("out of screen")

        if self.lander_top_y < 0:
            self.vx = 0
            self.vy = 0
            self.exploded = True
            self.on_screen = False
            

    # Check for terrain collision by looping through all terrain blocks
    def has_collided(self, planet, terrain_block_size, landing_pad_start, landing_blocks):
        for block_number, block in enumerate(planet):
        
            terrain_y = block.get('terrain_y')
            block_x = block_number * terrain_block_size
            block_y = terrain_y * terrain_block_size
            terrain_rect = pygame.Rect(block_x, block_y, terrain_block_size, terrain_block_size)
            
            if self.lander_rect.colliderect(terrain_rect):
                self.hit_terrain = True
                
                # Check if landing velocity acceptable
                if abs(self.vx) > self.landing_velocity or self.vy > self.landing_velocity:
                    self.exploded = True
                    self.velocity_ok = False
                    #print("hit terrain, bad velocity")
                
                # Check if landing angle acceptable
                if abs(self.angle) > self.landing_angle:
                    self.exploded = True
                    self.angle_ok = False
                    #print("hit terrain, bad angle")
                
                # Check if lander is within landing pad location
                self.landing_pad_left = (landing_pad_start - 1) * terrain_block_size #!!
                self.landing_pad_right = (landing_pad_start + landing_blocks + 2) * terrain_block_size #!!
                if self.lander_left_x >= self.landing_pad_left and self.lander_right_x <= self.landing_pad_right:
                    if self.exploded is False:
                        self.landed = True
                        #print("FUCK YEAH")

                        if self.angle < 0:
                            self.left_contact_ok = True
                        elif self.angle > 0: 
                            self.right_contact_ok = True
                        else:
                            self.right_contact_ok = True
                            self.left_contact_ok = True
                else:
                    self.exploded = True
                    self.landing_location_ok = False
                    #print("hit terrain, out of landing pad")
                
                self.vx = 0.0
                self.vy = 0.0


    def update_physics(self, game_mode='human'):
        if self.exploded is False:
            
            # If rocket thrust, calculate new velocity
            if self.rocket_thrust is True:

                fx = 0 + self.mass * self.engine_power * math.sin(math.radians(-self.angle))
                fy = - 2 * self.mass * self.gravity - self.mass * self.engine_power * math.cos(math.radians(self.angle))
            
                ax = fx / self.mass
                ay = fy / self.mass

                self.vx = self.vx + ax * Config.DT
                self.vy = self.vy + ay * Config.DT
                
                
                self.display_lander_image = pygame.transform.rotate(self.thrust_image, self.angle)
                
                # Reduce fuel 
                #if(game_mode == 'human'):
                self.fuel -= 1
                if self.fuel == 0:
                    self.exploded = True
                    self.fuel_left = False
                    #print("run out of fuel")

                
                # Play thrust sound effect
                if pygame.mixer.get_busy() == 0:
                    self.thrust_sound.play()
        
            else:
                if self.side_engines == 1:
                    self.angle -= Config.ROTATE_ANGLE

                    if self.angle < -90:
                        self.angle = -90

                elif self.side_engines == 2:
                    self.angle += Config.ROTATE_ANGLE

                    if self.angle > +90:
                        self.angle = +90

                self.display_lander_image = pygame.transform.rotate(self.lander_image, self.angle)
                self.thrust_sound.stop()


        else:
            self.display_lander_image = pygame.transform.rotate(self.exploded_lander_image, self.angle)

        # Gravity affects the y velocity
        fy = self.gravity * self.mass

        ay = fy / self.mass

        self.vy = self.vy + ay * Config.DT

        rt , se = self.report_state()
        
        # initialize for next event
        self.rocket_thrust = False
        self.side_engines = 0

        return rt, se


    def report_state(self):

        rt = self.rocket_thrust
        se = self.side_engines

        return rt, se


class Game:
    
    def __init__(self, game_mode):
        
        # setup game
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        pygame.init()

        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        pygame.display.set_caption('Lunar Lander')

        pygame.key.set_repeat(10, 20)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Courier', 16)

        self.background_image = pygame.image.load('game_mode/background.png').convert()

        self.game_mode = game_mode

        # setup environment
        self.environment = Environment()
        # setup lander
        self.lander = Lander()

        self.level = 1
        self.change_level()

        self.hi_score = 0

        self.reset()

     
    def change_level(self):
        
        if (self.level == 1):
            self.level_score = 0
            self.total_score = 0
        else:
            self.total_score += self.level_score

        lander_x = self.environment.new_level(self.level)

        self.lander.new_level(self.level, lander_x)

        self.level_end_sound_played = False
                
        
    # Reset game to the initial state and stop it 
    def reset(self):
        self.game_active = False
        
    
    def start(self):
        
        self.game_active = True
        

    def run_game(self):
        
        if(self.game_mode == 'human'):
            # Main game loop
            while True:
                self.check_events()
                rt, se = self.update()
                self.render()
        else:
            self.level == 1
            self.change_level()


    def check_events(self):
        
        for event in pygame.event.get():
            self.key_pressed = pygame.key.get_pressed()
            
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if self.game_active:
                if event.type == pygame.KEYDOWN:
                    
                    # SPACE key pressed - rocket thrust
                    if self.key_pressed[pygame.K_SPACE]:
                        self.lander.next_action(3) 
                    
                    # Left or right key pressed - rotate ship, ROTATE_ANGLE = 4
                    elif self.key_pressed[pygame.K_LEFT]:
                        self.lander.next_action(2) 
                        
                    elif self.key_pressed[pygame.K_RIGHT]:
                        self.lander.next_action(1) 
                        
                    else:
                        self.lander.next_action(0) 


                    # RETURN pressed and end of level/game
                    if self.key_pressed[pygame.K_RETURN] and (self.lander.landed or self.lander.hit_terrain is True):
                                    
                        # Successful landing - level up
                        if self.lander.landed is True:
                            self.level += 1
                        # Crashed - back to level 1
                        elif self.lander.hit_terrain is True:
                            self.level = 1

                        self.change_level()

                        pygame.mixer.stop()



    def update(self):
        # 1) Update lander location from previous action
        self.lander.update_location()
        
        # 2) Check lander has not left screen left, right or top
        self.lander.is_in_screen()

        # 3) Check for terrain collision by looping through all terrain blocks
        self.lander.has_collided(self.environment.planet, self.environment.terrain_block_size, self.environment.landing_pad_start, self.environment.landing_blocks)

        # 4) If after previous move there isn't collision, update physics from current move
        rt, se = self.lander.update_physics(self.game_mode)

        return rt, se


    def render(self):
        if(self.game_mode == "human"):
            # Display background
            self.screen.blit(self.background_image, [0, 0])
            
            # Display planet
            for block_number, block in enumerate(self.environment.planet):
                terrain_type = block.get('terrain_type')
                terrain_y = block.get('terrain_y')
                
                block_x = block_number * self.environment.terrain_block_size
                block_y = terrain_y * self.environment.terrain_block_size
                
                self.screen.blit(terrain_type, [block_x, block_y])
                
                # Display blank blocks below terrain surface
                for blank_block in range(terrain_y + 1, Config.MAX_TERRAIN_HEIGHT):
                    block_x = block_number * self.environment.terrain_block_size
                    block_y = blank_block * self.environment.terrain_block_size
                    self.screen.blit(self.environment.planet_blank_image, [block_x, block_y])

            # Display lander
            self.screen.blit(self.lander.display_lander_image, self.lander.lander_rect)
                
            # Landed messages
            if self.lander.hit_terrain is True:
                if self.game_mode == 'human':
                
                    if self.lander.landed is True:
                        self.display_successful_landing()
                        self.return_message_text = 'Hit return for next level.'
                        self.level_score = self.lander.fuel * self.level + (self.lander.landing_angle - abs(self.lander.angle)) * 10 * self.level
                        
                    else:
                        self.display_failed_landing(self.lander.on_screen, self.lander.fuel_left, self.lander.velocity_ok, self.lander.landing_location_ok, self.lander.angle_ok)
                        self.level_score = 0
                        self.return_message_text = 'Hit return for new game.'
                        
                    if self.total_score + self.level_score > self.hi_score:
                        self.hi_score = self.total_score + self.level_score
                        
                    
                    self.display_game_end(self.level_score, self.total_score + self.level_score, self.hi_score, self.return_message_text)
                
            else:
                self.display_status(self.lander.fuel, self.lander.vx, self.lander.landing_velocity, self.lander.vy, self.lander.angle, self.lander.landing_angle, self.level)
                
            if(self.lander.exploded is True):
                if self.level_end_sound_played is False:
                    self.lander.explosion_sound.play()
                    self.level_end_sound_played = True

            if(self.lander.landed is True):
                if self.level_end_sound_played is False:
                    self.lander.landed_sound.play()
                    self.level_end_sound_played = True

            pygame.display.update()
            if(self.game_mode == 'human'):
                self.clock.tick(60)
        
    
    # Successful landing message
    def display_successful_landing(self):
        self.message_text = 'Congratulations, a successful landing.'
        self.text = self.font.render(self.message_text, True, Config.GREEN)
        
        self.text_rect = self.text.get_rect()
        self.message_x = (Config.SCREEN_WIDTH - self.text_rect.width) / 2
        self.message_y = (Config.SCREEN_HEIGHT - self.text_rect.height) / 2 - Config.TEXT_LINE_SPACE * 6
        self.screen.blit(self.text, [self.message_x, self.message_y])


    # Failed landing message
    def display_failed_landing(self, on_screen, fuel_left, velocity_ok, landing_location_ok, angle_ok):
        
        self.message_text = 'YOU failed to land the spacecraft.'
        self.text = self.font.render(self.message_text, True, Config.GREEN)
        
        self.text_rect = self.text.get_rect()
        self.message_x = (Config.SCREEN_WIDTH - self.text_rect.width) / 2
        self.message_y = (Config.SCREEN_HEIGHT - self.text_rect.height) / 2 - Config.TEXT_LINE_SPACE * 7
        self.screen.blit(self.text, [self.message_x, self.message_y])
        
        self.fault_report_head = 'Fault Report'
        self.text = self.font.render(self.fault_report_head, True, Config.AMBER)
        
        self.text_rect = self.text.get_rect()
        self.message_x = (Config.SCREEN_WIDTH - self.text_rect.width) / 2
        self.message_y = (Config.SCREEN_HEIGHT - self.text_rect.height) / 2 - Config.TEXT_LINE_SPACE * 3
        self.screen.blit(self.text, [self.message_x, self.message_y])
        
        self.fault_text = ''
        if on_screen is False:
            self.fault_text += 'Left planet orbit'
        else:
            if fuel_left is False:
                self.fault_text += 'No fuel.'
            if velocity_ok is False:
                self.fault_text += 'Approach velocity.'
            if landing_location_ok is False:
                self.fault_text += 'Missed landing pad.'
            if angle_ok is False:
                self.fault_text += 'Approach angle.'

        self.text = self.font.render(self.fault_text, True, Config.AMBER)
        self.text_rect = self.text.get_rect()
        self.message_x = (Config.SCREEN_WIDTH - self.text_rect.width) / 2
        self.message_y = (Config.SCREEN_HEIGHT - self.text_rect.height) / 2 - Config.TEXT_LINE_SPACE * 2
        self.screen.blit(self.text, [self.message_x, self.message_y])
        
    # End of game message
    def display_game_end(self, level_score, game_score, hi_score, message):

        self.score_text = 'Scores: Level ' + str(level_score) + ' - Game ' + str(game_score) + ' - Hi ' + str(hi_score)
        self.text = self.font.render(self.score_text, True, Config.GREEN)
        
        self.text_rect = self.text.get_rect()
        self.message_x = (Config.SCREEN_WIDTH - self.text_rect.width) / 2
        self.message_y = (Config.SCREEN_HEIGHT - self.text_rect.height) / 2 - Config.TEXT_LINE_SPACE * 5
        self.screen.blit(self.text, [self.message_x, self.message_y])
        
        self.text = self.font.render(message, True, Config.GREEN)
        self.text_rect = self.text.get_rect()
        self.message_x = (Config.SCREEN_WIDTH - self.text_rect.width) / 2
        self.message_y = (Config.SCREEN_HEIGHT - self.text_rect.height) / 2 + Config.TEXT_LINE_SPACE * 1
        self.screen.blit(self.text, [self.message_x, self.message_y])


    # On screen data display: Fuel, H Velocity, V Velocity, Angle and Level
    def display_status(self, fuel, velocity_x, landing_velocity, velocity_y, angle, landing_angle, level):

        self.fuel_text = 'Fuel: ' + str(fuel)
        if fuel < Config.FUEL_WARNING:
            self.text = self.font.render(self.fuel_text, True, Config.AMBER)
        else:
            self.text = self.font.render(self.fuel_text, True, Config.GREEN)

        self.screen.blit(self.text, [Config.DATA_DISPLAY_LEFT, Config.TEXT_LINE_SPACE * 2])

        self.rounded_vel_x = round(velocity_x, 2)
        self.h_velocity_text = 'H Vel: ' + str(self.rounded_vel_x)
        if abs(velocity_x) > landing_velocity:
            self.text = self.font.render(self.h_velocity_text, True, Config.AMBER)
        else:
            self.text = self.font.render(self.h_velocity_text, True, Config.GREEN)
        
        self.screen.blit(self.text, [Config.DATA_DISPLAY_LEFT, Config.TEXT_LINE_SPACE * 3])


        self.rounded_vel_y = round(velocity_y, 2)
        self.v_velocity_text = 'V Vel: ' + str(self.rounded_vel_y)
        if velocity_y > landing_velocity:
            self.text = self.font.render(self.v_velocity_text, True, Config.AMBER)
        else:
            self.text = self.font.render(self.v_velocity_text, True, Config.GREEN)
        
        self.screen.blit(self.text, [Config.DATA_DISPLAY_LEFT, Config.TEXT_LINE_SPACE * 4])
        
        self.angle_text = 'Angle: ' + str(-angle)
        if abs(self.lander.angle) > self.lander.landing_angle:
            self.text = self.font.render(self.angle_text, True, Config.AMBER)
        else:
            self.text = self.font.render(self.angle_text, True, Config.GREEN)
        self.screen.blit(self.text, [Config.DATA_DISPLAY_LEFT, Config.TEXT_LINE_SPACE * 5])
        
        self.level_text = 'Level: ' + str(self.level)
        self.text = self.font.render(self.level_text, True, Config.GREEN)
        self.screen.blit(self.text, [Config.DATA_DISPLAY_LEFT, Config.TEXT_LINE_SPACE * 6])

    

class Config:
    # Define the colours
    WHITE = (255, 255, 255)
    GREEN = (84, 216, 61)
    AMBER = (255, 153, 0)

    # Define constants
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 480

    # not the same unit with SCREEN_WIDTH 
    DATA_DISPLAY_LEFT = 16
    TEXT_LINE_SPACE = 24

    FUEL_WARNING = 50
    # gravity expressed in (m/s^2 / 10) //Earth's gravity would be 0.981 in this case 
    KGRAVITY = 0.02
    ENGINE_POWER = 0.05

    MAX_TERRAIN_HEIGHT = 120
    MIN_TERRAIN_HEIGHT = 90

    
    LEFT_BUFFER_ZONE = 12
    RIGHT_BUFFER_ZONE = 12

    LANDER_START_Y = 40
    ROTATE_ANGLE = 4

    DT = 1

    levels = { 1 : {'fuel': 500, 'landing_blocks': 20, 'landing_velocity': 1.0, 'landing_angle': 24},
               2 : {'fuel': 400, 'landing_blocks': 16, 'landing_velocity': 1.0, 'landing_angle': 24},
               3 : {'fuel': 350, 'landing_blocks': 16, 'landing_velocity': 0.8, 'landing_angle': 20},
               4 : {'fuel': 300, 'landing_blocks': 16, 'landing_velocity': 0.8, 'landing_angle': 16},
               5 : {'fuel': 250, 'landing_blocks': 12, 'landing_velocity': 0.8, 'landing_angle': 12},
               6 : {'fuel': 200, 'landing_blocks': 12, 'landing_velocity': 0.6, 'landing_angle': 8},
               7 : {'fuel': 200, 'landing_blocks': 8, 'landing_velocity': 0.5, 'landing_angle': 0},
              }


