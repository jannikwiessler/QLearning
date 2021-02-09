# check if os has gaphical interface 
# (there might be no gui on linux servers)
try:
    import pyglet.gl
    GUIAVAILIABLE = True
except ImportError:
    GUIAVAILIABLE = False
import gym
from q_learning import QLeaning
from q_table import Qtable

env = gym.make("MountainCar-v0")

q_table = Qtable(env.observation_space.high, env.observation_space.low, [20, 20], env.action_space.n)
learning_engine = QLeaning(environment=env,q_table=q_table,gui=GUIAVAILIABLE)
learning_engine.run_training()
if GUIAVAILIABLE:
    learning_engine.show_training_results()