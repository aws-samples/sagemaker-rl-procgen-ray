import numpy as np
import json
import os
import collections
import re
import sys
sys.path.append("neurips2020-procgen-starter-kit")

import gym
import sagemaker
import boto3

from rollout import default_policy_agent_mapping, keep_going, DefaultMapping, RolloutSaver
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.env import MultiAgentEnv
try:
    from ray.rllib.evaluation.episode import _flatten_action
except Exception:
    # For newer ray versions
    from ray.rllib.utils.space_utils import flatten_to_single_ndarray as _flatten_action

from ray.rllib.evaluation.worker_set import WorkerSet

from source.custom.callbacks import CustomCallbacks

def get_latest_sagemaker_training_job(name_contains):
    sagemaker_session = sagemaker.Session()
    sagemaker_client = boto3.client('sagemaker')
    response = sagemaker_client.list_training_jobs(
        NameContains=name_contains,
        StatusEquals='Completed'
    )
    training_jobs = response['TrainingJobSummaries']
    assert len(training_jobs) > 0, "Couldn't find any completed training jobs with '{}' in name.".format(name_contains)
    latest_training_job = training_jobs[0]['TrainingJobName']
    return latest_training_job

def download_ray_checkpoint(checkpoint_dir, s3_bucket, latest_training_job):
    # Get last checkpoint
    checkpoint_data = "{}/{}/output/intermediate/training".format(s3_bucket, latest_training_job)
    checkpoint_bucket_key = "/".join(checkpoint_data.split("/")[1:]) + "/"

    s3 = boto3.client('s3')
    intermediate = s3.list_objects_v2(Bucket=s3_bucket, Prefix=checkpoint_bucket_key, Delimiter='//')

    last_checkpoint_num = 0
    last_checkpoint_key = None

    for content in intermediate['Contents']:
        # Check params.json
        if "params.json" in content["Key"]:
            with open('checkpoint/params.json', 'wb') as data:
                s3.download_fileobj(s3_bucket, content["Key"], data)

        # Find the last checkpoint
        checkpoint = re.search(r"checkpoint-([0-9]+)", content["Key"])
        if checkpoint is not None:
            checkpoint_num = checkpoint.group(1)
            if int(checkpoint_num) > last_checkpoint_num:
                last_checkpoint_num = int(checkpoint_num)
                last_checkpoint_key = content["Key"]

    with open('{}/checkpoint-{}'.format(checkpoint_dir, last_checkpoint_num), 'wb') as data:
        s3.download_fileobj(s3_bucket, last_checkpoint_key, data)
    with open('{}/checkpoint-{}.tune_metadata'.format(checkpoint_dir, last_checkpoint_num), 'wb') as data:
        s3.download_fileobj(s3_bucket, last_checkpoint_key+".tune_metadata", data)
    
    return last_checkpoint_num

def get_model_config():
    with open(os.path.join("checkpoint", "params.json")) as f:
        config = json.load(f)
        
    config["monitor"] = False
    config["num_workers"] = 1
    config["num_gpus"] = 0

    if 'callbacks' in config:
        callback_cls_str = config['callbacks'] # "<class 'custom.callbacks.CustomCallbacks'>",
        callback_cls = callback_cls_str.split("'")[-2].split(".")[-1] # CustomCallbacks
        config['callbacks'] = eval(callback_cls)
            
    return config

def rollout(agent,
            env_name,
            num_steps,
            num_episodes=0,
            saver=None,
            no_render=True,
            video_dir=None):
    # Adapted from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/master/rollout.py#L349
    policy_agent_mapping = default_policy_agent_mapping
    
    if saver is None:
        saver = RolloutSaver()

    if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
        #env = agent.workers.local_worker().env
        env = gym.make(env_name, render_mode="rgb_array")
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
    else:
        env = gym.make(env_name)
        multiagent = False
        try:
            policy_map = {DEFAULT_POLICY_ID: agent.policy}
        except AttributeError:
            raise AttributeError(
                "Agent ({}) does not have a `policy` property! This is needed "
                "for performing (trained) agent rollouts.".format(agent))
        use_lstm = {DEFAULT_POLICY_ID: False}

    action_init = {
        p: _flatten_action(m.action_space.sample())
        for p, m in policy_map.items()
    }

    steps = 0
    episodes = 0
    rgb_array = []
    
    while keep_going(steps, num_steps, episodes, num_episodes):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        saver.begin_rollout()
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        episode_steps = 0
        while not done and keep_going(steps, num_steps, episodes,
                                      num_episodes):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    a_action = _flatten_action(a_action)  # tuple actions
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)
            episode_steps += 1
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            if not no_render:
                rgb_array.append(env.render(mode='rgb_array'))
            saver.append_step(obs, action, next_obs, reward, done, info)
            steps += 1
            obs = next_obs
        saver.end_rollout()
        print("Episode #{}: reward: {} steps: {}".format(episodes, reward_total, episode_steps))
        if done:
            episodes += 1
    return rgb_array