import gymnasium as gym
import random
import torch
import torch.nn.functional as torchFunc
import torch.optim as optim
from MyCNN import MyCNN 
import numpy as np
import matplotlib.pyplot as plt

cnn = MyCNN()


loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr=1e-3)
gamma = 0.9
epsilon = 1.0

def model(observation):
    image_tensor = torch.tensor(observation['image']).float()
    image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0)
    direction_tensor = torch.tensor(observation['direction'])
    direction_tensor = torchFunc.one_hot(direction_tensor, num_classes=4).float()
    direction_tensor = direction_tensor.unsqueeze(0)
    qValue = cnn.forward(image_tensor, direction_tensor)
    
    return qValue



#steps
#0 - turn left
#1 - turn right
#2 - go forward

#lets first start with the regular empty grid and see if it converges - let's plot
#then we can use the empty - random grid. 
#then we can use the obstacle grid
#let's create a custom grid where the green and the player position changes with each epoch.
	

# action = process_action(observation)

epochs = 5000
losses = []
env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")

total_rewards = []
average_losses = []

for i in range(epochs):
    observation, info = env.reset(seed=42)
    state1 = observation
    status = 1
    total_reward = 0
    decay_amount = 0.000001
    while status == 1:
        qval = model(state1)
        
        if random.random() < epsilon:
            action_ = np.random.randint(0, 4)
        else:
            action_ = torch.argmax(qval.detach()).item()
    
        next_observation, reward, terminated, truncated, info = env.step(action_)
        state2 = next_observation
    
        with torch.no_grad():
            newQ = model(state2)
            maxQ = torch.max(newQ).item()

        # Compute the target value Y
        Y = reward if terminated else reward + (gamma * maxQ)
        Y = torch.tensor([Y], dtype=torch.float32)
        
        # Q-value for the current state-action pair
        X = qval.squeeze()[action_].unsqueeze(0)

        # Compute loss and update model
        loss = loss_fn(X, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_reward += reward
        losses.append(loss.item())
        state1 = state2
        
    
        
        # Check for termination
        if terminated:
            status = 0
            
        # Epsilon decay
        if epsilon > 0.1:
            epsilon -= decay_amount
    
    total_rewards.append(total_reward)
    average_loss = sum(losses) / len(losses)
    average_losses.append(average_loss)
    
    if i % 10 == 0:  # Print info every 10 episodes
        print(f"Epoch {i}, Avg Loss: {average_loss}, Total Reward: {total_reward}, Epsilon: {epsilon}")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(total_rewards)
plt.title('Total Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(1, 2, 2)
plt.plot(average_losses)
plt.title('Average Loss per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Loss')

plt.show()
        
            
env.close()

 

