import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor):
        """Forward pass of the neural network"""
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        """Save the model to the file"""
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        print('model_saved')

    # Load the model from the file
    def load(self, file_name='model.pth'):
        """Load the model from the file"""
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))
        self.eval()

class QTrainer:
    def __init__(self, model: Linear_QNet, lr, gamma) -> None:
        self.model = model
        self.lr = lr
        self.gamma = gamma
        
        #pytorch optimzation step
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Loss function, mean squared error loss = (Q_new - Q_old)^2
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, game_over):
        """Train the model"""
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        
        # (n, x)
        
        if len(state.shape) == 1:
            # only have one number, have form (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over,  )
            
        # 1: predicted Q values with the current state
        pred = self.model(state)
        
        target = pred.clone()
        
        for i in range(len(game_over)):
            Q_new = reward[i]
            if not game_over[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action).item()] = Q_new
        
        # 2: Q_new =  r + y * max(next_predicted_Q_value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        
        self.optimizer.step()