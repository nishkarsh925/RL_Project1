import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Create models directory if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

# LSTM Model for Sales Forecasting
class SalesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(SalesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# DQN Model for Price Optimization
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# RL Agent for Dynamic Pricing
class DynamicPricingDQN:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.data['Report Date'] = pd.to_datetime(self.data['Report Date'])
        self.data.sort_values('Report Date', inplace=True)
        
        # Feature Engineering
        self.data['Rolling_Median_Price'] = self.data['Product Price'].rolling(7).median().bfill()
        self.data.fillna(0, inplace=True)
        
        self.min_price = self.data['Product Price'].min()
        self.max_price = self.data['Product Price'].max()
        
        # LSTM for Sales Forecasting
        self.lstm_model = SalesLSTM(input_size=2)  # (Price, Previous Sales)
        self.lstm_optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.001)
        self.lstm_criterion = nn.MSELoss()
        
        # Load LSTM model if exists
        if os.path.exists("models/lstm_model.pth"):
            self.lstm_model.load_state_dict(torch.load("models/lstm_model.pth"))
            print("Loaded saved LSTM model.")
        
        # Q-Learning Parameters
        self.gamma = 0.99
        self.batch_size = 32
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Action Space: [-10%, -5%, 0, +5%, +10%]
        self.actions = [-0.1, -0.05, 0, 0.05, 0.1]
        
        # Q-Network & Optimizer
        self.q_model = QNetwork(input_dim=4, output_dim=len(self.actions))
        self.q_optimizer = optim.Adam(self.q_model.parameters(), lr=0.001)
        self.q_criterion = nn.MSELoss()
        
        # Load Q-Network model if exists
        if os.path.exists("models/q_network.pth"):
            self.q_model.load_state_dict(torch.load("models/q_network.pth"))
            print("Loaded saved Q-Network model.")
        
        # Experience Replay Memory
        self.memory = deque(maxlen=5000)

    def predict_sales(self, price, prev_sales):
        """Predict next-day sales using LSTM"""
        self.lstm_model.eval()
        input_tensor = torch.FloatTensor([[price, prev_sales]]).unsqueeze(0)
        with torch.no_grad():
            return self.lstm_model(input_tensor).item()
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(len(self.actions)))  # Explore
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.q_model(state_tensor)).item()  # Exploit
    
    def train_q_network(self):
        """Train Q-Network with experience replay"""
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in batch:
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)
            action_index = action  
            
            with torch.no_grad():
                target_q = reward + self.gamma * torch.max(self.q_model(next_state_tensor))
            
            predicted_q = self.q_model(state_tensor)[action_index]
            loss = self.q_criterion(predicted_q, target_q)
            
            self.q_optimizer.zero_grad()
            loss.backward()
            self.q_optimizer.step()
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def run_simulation(self, episodes=1000):
        """Train RL model and return the best price prediction for tomorrow"""
        for _ in range(episodes):
            sample = self.data.sample(1).iloc[0]
            current_price = sample['Product Price']
            prev_sales = sample['Total Sales']
            predicted_sales = self.predict_sales(current_price, prev_sales)
            
            state = [current_price, prev_sales, sample['Organic Conversion Percentage'], predicted_sales]
            action_index = self.choose_action(state)
            next_price = np.clip(current_price * (1 + self.actions[action_index]), self.min_price, self.max_price)
            next_sales = self.predict_sales(next_price, prev_sales)
            
            reward = (next_sales - prev_sales) / max(prev_sales, 1)  # Sales growth reward
            if next_price > current_price:
                reward += 0.2  # Encourage price increase
            
            next_state = [next_price, next_sales, sample['Organic Conversion Percentage'], predicted_sales]
            self.memory.append((state, action_index, reward, next_state))
            self.train_q_network()
        
        # Save trained models
        torch.save(self.lstm_model.state_dict(), "models/lstm_model.pth")
        torch.save(self.q_model.state_dict(), "models/q_network.pth")
        print("Models saved successfully!")
        
        # Predict best price for tomorrow
        best_price = max(self.data['Product Price'], key=lambda p: self.q_model(torch.FloatTensor([p, 1, 1, 1])).max().item())
        return best_price

# Run RL Model
if __name__ == "__main__":
    pricing_agent = DynamicPricingDQN('woolballhistory.csv')
    optimal_price = pricing_agent.run_simulation(episodes=5000)
    print(f"Optimal Product Price for Tomorrow: ${optimal_price:.2f}")
