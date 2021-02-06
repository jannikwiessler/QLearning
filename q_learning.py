import gym
from q_table import Qtable


env = gym.make("MountainCar-v0")
env.reset()

print(env.observation_space.high)
print(env.observation_space.low)

print(env.action_space.n)

q_table_obj = Qtable(env.observation_space.high, env.observation_space.low, [20, 20], env.action_space.n)
q_table = q_table_obj.get_q_table()

done = False

action = 2 # Accelerate to the Right
while not done:
    new_state, reward, done, _ = env.step(action) # state containing car's position and velocity
   # print(new_state)
    env.render()

env.close()


