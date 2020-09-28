# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# or in the "license" file accompanying this file. This file is distributed 
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
# express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

import os
import json
import gym
import ray

from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from procgen_ray_launcher import ProcgenSageMakerRayLauncher

from ray_experiment_builder import RayExperimentBuilder

from utils.loader import load_algorithms, load_preprocessors
try:
    from custom.envs.procgen_env_wrapper import ProcgenEnvWrapper
except ModuleNotFoundError:
    from envs.procgen_env_wrapper import ProcgenEnvWrapper

class MyLauncher(ProcgenSageMakerRayLauncher):
    def register_env_creator(self):        
        register_env(
            "stacked_procgen_env",  # This should be different from procgen_env_wrapper
            lambda config: gym.wrappers.FrameStack(ProcgenEnvWrapper(config), 4)
        )

    def _get_ray_config(self):
        return {
            "ray_num_cpus": 8, # adjust based on selected instance type
            "ray_num_gpus": 1,
            "eager": False,
             "v": True, # requried for CW to catch the progress
        }

    def _get_rllib_config(self):
        return {
            "experiment_name": "training",
            "run": "PPO",
            "env": "procgen_env_wrapper",
            "stop": {
                # 'time_total_s': 60,
                'training_iteration': 500,
                },
            "checkpoint_freq": 20,
            "checkpoint_at_end": True,
            "keep_checkpoints_num": 5,
            "queue_trials": False,
            "config": {
                # === Environment Settings ===
                "gamma": 0.999,
                "lambda": 0.95,
                "lr": 5.0e-4,
                "num_sgd_iter": 3,
                "kl_coeff": 0.0,
                "kl_target": 0.01,
                "vf_loss_coeff": 0.5,
                "entropy_coeff": 0.01,
                "clip_param": 0.2,
                "vf_clip_param": 0.2,
                "grad_clip": 0.5,
                "observation_filter": "NoFilter",
                "vf_share_layers": True,
                "soft_horizon": False,
                "no_done_at_end": False,
                "normalize_actions": False,
                "clip_actions": True,
                "ignore_worker_failures": True,
                "use_pytorch": False,
                "sgd_minibatch_size": 2048, # 8 minibatches per epoch
                "train_batch_size": 16384, # 2048 * 8
                # === Settings for Model ===
                "model": {
                    "custom_model": "impala_cnn_tf",
                },
                # === Settings for Rollout Worker processes ===
                "num_workers": 6, # adjust based on total number of CPUs available in the cluster, e.g., p3.2xlarge has 8 CPUs
                "rollout_fragment_length": 140,
                "batch_mode": "truncate_episodes",
                # === Advanced Resource Settings ===
                "num_envs_per_worker": 12,
                "num_cpus_per_worker": 1,
                "num_cpus_for_driver": 1,
                "num_gpus_per_worker": 0.1,
                # === Settings for the Trainer process ===
                "num_gpus": 0.3, # adjust based on number of GPUs available in a single node, e.g., p3.2xlarge has 1 GPU
                # === Exploration Settings ===
                "explore": True,
                "exploration_config": {
                    "type": "StochasticSampling",
                },
                # === Settings for the Procgen Environment ===
                "env_config": {
                    # See https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/master/experiments/procgen-starter-example.yaml#L34 for an explaination.
                    "env_name": "coinrun",
                    "num_levels": 0,
                    "start_level": 0,
                    "paint_vel_info": False,
                    "use_generated_assets": False,
                    "center_agent": True,
                    "use_sequential_levels": False,
                    "distribution_mode": "easy"
                }
            }
        }
    
    def register_algorithms_and_preprocessors(self):
        try:
            from custom.algorithms import CUSTOM_ALGORITHMS
            from custom.preprocessors import CUSTOM_PREPROCESSORS
            from custom.models.impala_cnn_tf import ImpalaCNN
        except ModuleNotFoundError:
            from algorithms import CUSTOM_ALGORITHMS
            from preprocessors import CUSTOM_PREPROCESSORS
            from models.impala_cnn_tf import ImpalaCNN

        load_algorithms(CUSTOM_ALGORITHMS)
        load_preprocessors(CUSTOM_PREPROCESSORS)
        ModelCatalog.register_custom_model("impala_cnn_tf", ImpalaCNN)

    def get_experiment_config(self):
        params = dict(self._get_ray_config())
        params.update(self._get_rllib_config())
        reb = RayExperimentBuilder(**params)
        return reb.get_experiment_definition()


if __name__ == "__main__":
    MyLauncher().train_main()
