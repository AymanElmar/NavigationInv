from environment.navigation.Env import MultiAgentGraphEnv
from environment.navigation.Scenarionavigation import Scenario
from training.wrappers import GraphSubprocVecEnv, GraphDummyVecEnv
from environment.navigation.Env import GraphMPEEnv
import numpy as np


def make_train_env(all_args):
    def get_env_fn(rank: int):
        def init_env():
            env = GraphMPEEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return GraphDummyVecEnv([get_env_fn(0)])
    else:
        return GraphSubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
        )

class Args:
    def __init__(self):
        self.num_agents: int = 3
        self.world_size = 2
        self.num_scripted_agents = 0
        self.num_obstacles: int = 3
        self.collaborative: bool = False
        self.max_speed: Optional[float] = 2
        self.collision_rew: float = 5
        self.goal_rew: float = 5
        self.min_dist_thresh: float = 0.1
        self.use_dones: bool = False
        self.episode_length: int = 250
        self.max_edge_dist: float = 1
        self.graph_feat_type: str = "global"
        self.verbose: bool = True
        self.n_rollout_threads: int = 3
        self.seed=1
if __name__ == '__main__':
    all_args = Args()


    # makeshift argparser



    nthrd = make_train_env(all_args)
    
    print(nthrd.observation_space)
    print(nthrd.action_space)
    print(nthrd.action_space[0])
    # create multiagent environment
    # render call to create viewer window
    class random_policy:
        def __init__(self, env, agent_id):
            self.env = env
            self.agent_id = agent_id
        def action(self):
            return self.env.action_space[self.agent_id].sample()
        def all_actions(self):
            all_actions=[]
            for action_space in self.env.action_space:
                all_actions.append(action_space.sample())
            return all_actions

    obs_stack, agent_id_stack, node_obs_stack, adj_stack = nthrd.reset()
    stp = 0
    while True:
        act_stack = []
        for i in range(all_args.n_rollout_threads):
            act_stack.append(random_policy(nthrd, i).all_actions())
        nthrd.step_async(act_stack)
        try:
            while not nthrd.waiting:
                pass
        except:
            pass
        obs_stack, agent_id_stack, node_obs_stack, adj_stack, reward_stack, done_stack, info_stack = nthrd.step_wait()
        stp += 1
        if stp > 100: 
            print("action shape:", np.array(act_stack).shape)
            print("Obs shape:", obs_stack.shape)
            print("Node Obs shape:", node_obs_stack.shape)
            print("Adjacency shape:", adj_stack.shape)
            print("Reward:", reward_stack)
            print("Done:", done_stack)
            break

