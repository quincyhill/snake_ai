import torch
import random
import numpy as np

# data structure to store memory
from collections import deque

from snake_game import SnakeGame, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot


MAX_MEMORY_SIZE = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self) -> None:
        self.n_games = 0
        
        # control the randomness
        self.epsilon = 0
        
        # discount rate, can mess around but keep it < 1
        self.gamma = 0.9
        
        # popleft when memory is full
        self.memory = deque(maxlen=MAX_MEMORY_SIZE)
        
        # TODO: model 
        self.model = Linear_QNet(11, 256, 3)

        # TODO: trainer
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        # Load the model
        self.model.load()

        # lets see
        self.model.eval()
    
    def get_state(self, game: SnakeGame):
        """All the possible states that the agent can be in"""

        head = game.snake[0]

        # points above, lower, left and right to the head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        # Add an addional state thats danger if snake is about to hit itself
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
        """Store the memory"""
        # Again if exceeds the max size, pop left aka the oldest one
        self.memory.append((state, action, reward, next_state, game_over))
    
    def train_long_memory(self):
        """Train the model on the long memory"""
        if len(self.memory) > BATCH_SIZE:
            # return a list of tuples 
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, game_overs)
        

    def train_short_memory(self, state, action, reward, next_state, game_over):
        """Train the model on the short memory"""
        # Train for one game step
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        """Get the action from the model"""
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

            prediction = self.model(state0)
            
            # Convert to only one number
            move = torch.argmax(prediction).item()
            
            final_move[move] = 1

        return final_move

def train():
    """Train the model"""
    # used for plotting
    plot_scores = []
    
    plot_mean_scores = []
    
    total_score = 0
    
    best_score = 0
    
    agent = Agent()
    
    game = SnakeGame()

    # train till I quit
    while True:

        # get old state
        state_old = agent.get_state(game)
        
        # get move
        final_move = agent.get_action(state_old)
        
        # perform move and get new state
        reward, game_over, score = game.play_step(final_move)
        
        # get new state
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
                agent.model.save()
                
            print('Game: ', agent.n_games, 'Score: ', score, 'Best: ', best_score)
            
            plot_scores.append(score)
            total_score += score
            mean_score  = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()