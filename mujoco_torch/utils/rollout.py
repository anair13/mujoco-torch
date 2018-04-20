import numpy as np
import pickle
from ss.envs.ball_env import BallEnv
from ss.envs.box_env import BoxEnv
import click

class Rollout:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        pass

    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def play(self):
        pass

    # TODO: make saving and loading efficient if it becomes a problem
    # def __getstate__(self):
    #     exclude_vars = set(["env"])
    #     args = {}
    #     for k in self.init_args:
    #         if k not in exclude_vars:
    #             args[k] = self.init_args[k]
    #     return {'tf': self.get_save_tf(), 'init': args}

    # def __setstate__(self, state):
    #     self.__init__(**state['init'])

    #     self.sess = tf.InteractiveSession() # for now just make ourself a session
    #     self.sess.run(tf.global_variables_initializer())
    #     self.restore_tf(state['tf'])
    #     self.actor_optimizer.sync()
    #     self.critic_optimizer.sync()

@click.command()
@click.argument('rollout_pickle_file')
def play(rollout_pickle_file):
    f = open(rollout_pickle_file, "rb")
    rollouts = pickle.load(f)
    env = BoxEnv()
    for r in rollouts:
        for i in range(len(r.states)):
            s = r.states[i]
            env.set_state(*s)
            env.render()

if __name__ == "__main__":
    play()
