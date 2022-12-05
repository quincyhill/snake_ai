import pygame
import random
from enum import Enum, auto
from collections import namedtuple
import numpy as np

pygame.init()

# Font just for visual
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = auto()
    LEFT = auto()
    UP = auto()
    DOWN = auto()
    
# Ok this works the way it currently does
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
MAX_FRAME_ITERATION = 100
SPEED = 100

class SnakeGame:
    
    def __init__(self, w=1200, h=720):
        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI')

        self.clock = pygame.time.Clock()
        self.reset()
        
        
        
    def reset(self):
        """Initializes the game state"""
        # init game state
        # Always start the instance heading to the right

        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)

        # Our snake is a list of points
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None

        # Randomly place our food on the screen, could also try multiple sources of food
        self._place_food()

        # Always start on frame 0 for each
        self.frame_iteration = 0

    def _place_food(self):
        """Spawns food on the screen"""
        x = random.randint(0, (self.w-BLOCK_SIZE ) // BLOCK_SIZE ) * BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE ) // BLOCK_SIZE ) * BLOCK_SIZE

        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        """Takes action and updates the game state"""
        self.frame_iteration += 1

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0

        game_over = False

        if self.is_collision() or self.frame_iteration > MAX_FRAME_ITERATION * len(self.snake):
            # End game condition
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            # we have eaten so we need to place a new piece of food, and increase score
            self.score += 1
            reward = 10
            self._place_food()
        else:
            # pop tail otherwise snake grows forever each frame
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        """Collision detection game is over if this is hit"""
        if pt is None:
            pt = self.head
        
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # hits itself
        if self.head in self.snake[1:]:
            # lol can't hit its own head so check if head hits body somewhere
            return True
        
        return False
        
    def _update_ui(self):
        """Updates the UI"""
        self.display.fill(BLACK)
        
        for part in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(part.x, part.y, BLOCK_SIZE, BLOCK_SIZE))

            # Good enough, allows it to scale
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(part.x + BLOCK_SIZE * .125 , part.y + BLOCK_SIZE * .125 , BLOCK_SIZE * .75 , BLOCK_SIZE * .75))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)

        # draw visually
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        """Moves the snake, given an action"""
        # [straight, right, left]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        dir_index = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            # keep current direction, no change
            new_dir = clock_wise[dir_index] 
        elif np.array_equal(action, [0, 1, 0]):
            # turn right
            next_idx = (dir_index + 1) % 4
            
            # right turn r -> d -> l -> u
            new_dir = clock_wise[next_idx]
        else:
            # Assumes [0, 0, 1] maybe I should constrain since this could be [2,0,0]
            # turn left
            next_idx = (dir_index - 1) % 4
            
            # left turn r -> u -> l -> d
            new_dir = clock_wise[next_idx]
        
        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
