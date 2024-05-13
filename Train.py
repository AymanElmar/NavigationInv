#!/usr/bin/env python
import argparse
from distutils.util import strtobool
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import os, sys

sys.path.append(os.path.abspath(os.getcwd()))

from config import get_config, graph_config
from environment.navigation.Env import GraphMPEEnv
from training.wrappers import GraphSubprocVecEnv, GraphDummyVecEnv



def print_dash(num_dash: int = 50):
    """
    Print "______________________"
    for num_dash times
    """
    print("_" * num_dash)

def print_box(string, num_dash: int = 50):
    """
    print the given string as:
    _____________________
    string
    _____________________
    """
    print_dash(num_dash)
    print(string)
    print_dash(num_dash)



def print_args(args: argparse.Namespace):
    """
    Print the args in a pretty table
    """
    box_dist = 50
    print_dash(box_dist)
    half_len = int((box_dist - len("Arguments") - 5) / 2)
    print("||" + " " * half_len + "Arguments" + " " * half_len + " ||")
    print_dash(box_dist)
    for k, v in vars(args).items():
        len_line = len(f"{k}: {str(v)}")
        print("|| " + f"{k}: {str(v)}" + " " * (box_dist - len_line - 5) + "||")
    print_dash(box_dist)



def connected_to_internet(url: str = "http://www.google.com/", timeout: int = 5):
    import requests
    """
    Check if system is connected to the internet
    Used when running code on MIT Supercloud
    """
    try:
        _ = requests.get(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        print("No internet connection available.")
    return False






"""Train script for MPEs."""


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

def make_eval_env(all_args: argparse.Namespace):
    def get_env_fn(rank: int):
        def init_env():
            env = GraphMPEEnv(all_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return GraphDummyVecEnv([get_env_fn(0)])
    else:
        return GraphSubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
        )

def parse_args(args, parser):

    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="navigation",
        help="Which scenario to run on",
    )
    parser.add_argument("--num_agents", type=int, default=3, help="number of players")
    parser.add_argument(
        "--num_obstacles", type=int, default=3, help="Number of obstacles"
    )
    parser.add_argument(
        "--collaborative",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="are the agents collaborative or competitive",
    )
    parser.add_argument(
        "--max_speed",
        type=float,
        default=2,
        help="Max speed for agents. NOTE that if this is None, "
        "then max_speed is 2 with discrete action space",
    )
    parser.add_argument(
        "--collision_rew",
        type=float,
        default=5,
        help="The reward to be negated for collisions with other "
        "agents and obstacles",
    )
    parser.add_argument(
        "--goal_rew",
        type=float,
        default=5,
        help="The reward to be added if agent reaches the goal",
    )
    parser.add_argument(
        "--min_dist_thresh",
        type=float,
        default=0.05,
        help="The minimum distance threshold to classify whether "
        "agent has reached the goal or not",
    )
    parser.add_argument(
        "--use_dones",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether we want to use the 'done=True' "
        "when agent has reached the goal or just return False like "
        "the `simple.py` or `simple_spread.py`",
    )

    all_args = parser.parse_known_args(args)[0]

    return all_args, parser


def main(args):
    parser = get_config()
    all_args, parser = parse_args(args, parser)
    all_args, parser = graph_config(args, parser)
    assert (
            all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy
        ), "check recurrent policy!"

    assert (
        all_args.share_policy == True
        and all_args.scenario_name == "simple_speaker_listener"
    ) == False, (
        "The simple_speaker_listener scenario can not use shared policy. "
        "Please check the config.py."
    )

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print_box("Choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print_box("Choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    if all_args.verbose:
        print_args(all_args)




    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        # for supercloud when no internet_connection
        if not connected_to_internet():
            import json

            # save a json file with your wandb api key in your
            # home folder as {'my_wandb_api_key': 'INSERT API HERE'}
            # NOTE this is only for running on systems without internet access
            # have to run `wandb sync wandb/run_name` to sync logs to wandboard
            with open("./keys.json") as json_file:
                key = json.load(json_file)
                my_wandb_api_key = key["my_wandb_api_key"]  # NOTE change here as well
            os.environ["WANDB_API_KEY"] = my_wandb_api_key
            os.environ["WANDB_MODE"] = "dryrun"
            os.environ["WANDB_SAVE_CODE"] = "true"

        print_box("Creating wandboard...")
        run = wandb.init(
            config=all_args,
            project=all_args.project_name,
            # project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name)
            + "_"
            + str(all_args.experiment_name)
            + "_seed"
            + str(all_args.seed),
            # group=all_args.scenario_name,
            dir=str(run_dir),
            # job_type="training",
            reinit=True,
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }


    # run experiments
    from training.Runner import GMPERunner as Runner


    runner = Runner(config)

    if all_args.verbose:
        print_box("Actor Network", 80)
        if type(runner.policy) == list:
            print_box(runner.policy[0].actor, 80)
            print_box("Critic Network", 80)
            print_box(runner.policy[0].critic, 80)
        else:
            print_box(runner.policy.actor, 80)
            print_box("Critic Network", 80)
            print_box(runner.policy.critic, 80)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
