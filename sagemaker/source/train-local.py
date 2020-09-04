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

#!/usr/bin/env python

import os
from pathlib import Path
import yaml

import ray
from ray.tune.tune import _make_scheduler, run_experiments
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf, try_import_torch

from utils.loader import load_envs, load_models, load_algorithms, load_preprocessors
from ray_experiment_builder import RayExperimentBuilder

# Try to import both backends for flag checking/warnings.
tf = try_import_tf()
torch, _ = try_import_torch()

from custom.models.my_vision_network import MyVisionNetwork

"""
Note : This script has been adapted from :
    https://github.com/ray-project/ray/blob/master/rllib/train.py and https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/master/train.py
"""

load_envs(os.path.join("custom", "envs"))
load_models((os.path.join("custom", "models"))) # Load models
# Load custom algorithms
from custom.algorithms import CUSTOM_ALGORITHMS
load_algorithms(CUSTOM_ALGORITHMS)
# Load custom preprocessors
from custom.preprocessors import CUSTOM_PREPROCESSORS
load_preprocessors(CUSTOM_PREPROCESSORS)

print(ray.rllib.contrib.registry.CONTRIBUTED_ALGORITHMS)

def run():
    ModelCatalog.register_custom_model("my_vision_network", MyVisionNetwork)
    config={
        "model":{
            "custom_model": "my_vision_network",
            "conv_filters": [[16, [5, 5], 4], [32, [3, 3], 1], [256, [3, 3], 1]],
            "custom_preprocessor": None
        }
    }
        
    reb = RayExperimentBuilder(**config)
    experiments, args, verbose = reb.get_experiment_definition()
    
    ray.init(
        address=args.ray_address,
        object_store_memory=args.ray_object_store_memory,
        memory=args.ray_memory,
        redis_max_memory=args.ray_redis_max_memory,
        num_cpus=args.ray_num_cpus,
        num_gpus=args.ray_num_gpus)

    run_experiments(
        experiments,
        scheduler=_make_scheduler(args),
        queue_trials=args.queue_trials,
        resume=args.resume,
        verbose=verbose,
        concurrent=True)

if __name__ == "__main__":
    run()
