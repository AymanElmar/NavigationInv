from navigation.Env import MultiAgentGraphEnv
from navigation.Scenarionavigation import Scenario

# makeshift argparser
class Args:
    def __init__(self):
        self.num_agents: int = 5
        self.world_size = 2
        self.num_scripted_agents = 0
        self.num_obstacles: int = 0
        self.collaborative: bool = False
        self.max_speed: Optional[float] = 2
        self.collision_rew: float = 5
        self.goal_rew: float = 5
        self.min_dist_thresh: float = 0.1
        self.use_dones: bool = False
        self.episode_length: int = 250
        self.max_edge_dist: float = 1
        self.graph_feat_type: str = "global"
        self.only_nav: bool = True

args = Args()

scenario = Scenario()
# create world
world = scenario.make_world(args)
# create multiagent environment
env = MultiAgentGraphEnv(
    world=world,
    reset_callback=scenario.reset_world,
    reward_callback=scenario.reward,
    observation_callback=scenario.observation,
    graph_observation_callback=scenario.graph_observation,
    info_callback=scenario.info_callback,
    done_callback=scenario.done,
    id_callback=scenario.get_id,
    update_graph=scenario.update_graph,
    shared_viewer=False,)
# render call to create viewer window
env.render()
class random_policy:
    def __init__(self, env, agent_id):
        self.env = env
        self.agent_id = agent_id
    def action(self, obs):
        return self.env.action_space[self.agent_id].sample()
# create interactive policies for each agent
policies = [random_policy(env, i) for i in range(env.n)]
# execution loop
obs_n, agent_id_n, node_obs_n, adj_n = env.reset()
stp = 0
while True:
    # query for action from each agent's policy
    act_n = []
    dist_mag = env.world.cached_dist_mag

    for i, policy in enumerate(policies):
        act_n.append(policy.action(obs_n[i]))
    # step environment
    obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n = env.step(act_n)
    # render all agent views
    env.render()
    stp += 1
    # display rewards
    if all(done_n):
        obs_n, agent_id_n, node_obs_n, adj_n = env.reset()
print(act_n[0])
print("Observation:", obs_n, "Obs shape:",args.num_agents,'*',  obs_n[0].shape)
print("Node Observation:", node_obs_n, "Node Obs shape:",args.num_agents,'*',  node_obs_n[0].shape)
print("Adjacency matrix:", adj_n, "Adjacency shape:",args.num_agents,'*', adj_n[0].shape)
print("Reward:", reward_n)
print("Done:", done_n)
print(env.observation_space)
print(env.share_observation_space)