import gymnasium as gym
import minigrid
import time


#0 - turn left
#1 - turn right
#2 - go forward
	
env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")


observation, info = env.reset(seed=42)

observation, reward, terminated, truncated, info = env.step(2)
print(reward)
observation, reward, terminated, truncated, info = env.step(2)
print(reward)
observation, reward, terminated, truncated, info = env.step(1)
print(reward)
observation, reward, terminated, truncated, info = env.step(2)
print(reward)
observation, reward, terminated, truncated, info = env.step(1)
print(reward)
observation, reward, terminated, truncated, info = env.step(2)
print(reward)
observation, reward, terminated, truncated, info = env.step(0)
print(reward)
observation, reward, terminated, truncated, info = env.step(2)
print(reward)
observation, reward, terminated, truncated, info = env.step(0)
print(reward)
observation, reward, terminated, truncated, info = env.step(2)
print(reward)

# observation, reward, terminated, truncated, info = env.step(2)
# print(reward)

     
   
    
       
env.close()

 

