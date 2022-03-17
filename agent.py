import torch
import random
import numpy as np
import matplotlib.pyplot as plt

# data structure to store memory
from collections import deque

from snake_game import SnakeGameAI, Direction, Point, BLOCK_SIZE


MAX_MEMORY_SIZE = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self) -> None:
        self.n_games = 0
        
        # control the randomness
        self.epsilon = 0
        
        # discount rate
        self.gamma = 0
        
        # popleft when memory is full
        self.memory = deque(maxlen=MAX_MEMORY_SIZE)
        
        # TODO: model 
        self.model = None

        # TODO: trainer
        self.trainer = None
    
    def get_state(self, game: SnakeGameAI):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            
            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            
            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food left
            game.food.x < game.head.x,
            
            # Food right
            game.food.x > game.head.x,
            
            # Food up
            game.food.y < game.head.y,
            
            # Food down
            game.food.y > game.head.y,
        ]
        
        return np.array(state, dtype=int)

    
    def remember(self, state, action, reward, next_state, game_over):
        # Again if exceeds the max size, pop left aka the oldest one
        self.memory.append((state, action, reward, next_state, game_over))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # return a list of tuples 
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, games_over = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, games_over)
        

    def train_short_memory(self, state, action, reward, next_state, game_over):
        # Train for one game step
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves: tradeoff between exploration and exploitation
        
        # Can change this to whatever
        # The more games we have the smaller the epsilon will get and the less likely the agent will explore
        self.epsilon = 80 - self.n_games
        
    
        
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Convert state to tensor
            state0 = torch.tensor(state, dtype=torch.float32)

            prediction = self.model.predict(state0)
            
            # Convert to only one number
            move = torch.argmax(prediction).item()
            
            final_move[move] = 1

        return final_move

def train():
    # used for plotting
    plot_scores = []
    
    plot_mean_scores = []
    
    total_score = 0
    
    best_score = 0
    
    agent = Agent()
    
    game = SnakeGameAI()

    # train till I quit
    while True:

        # get old state
        state_old = agent.get_state(game)
        
        # get move
        final_move = agent.get_action(state_old)
        
        # perform move and get new state
        reward, game_over, score = game.play_step(final_move)
        
        state_new = agent.get_state(game)
        
        # train short memeory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)
        
        # remember
        agent.remember(state_old, final_move, reward, state_new, game_over)
        
        if game_over:
            # Train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > best_score:
                best_score = score
                # TODO: agent.model.save()
                
            print('Game: ', agent.n_games, 'Score: ', score, 'Best: ', best_score)
            
            # plotting


if __name__ == '__main__':
    train()