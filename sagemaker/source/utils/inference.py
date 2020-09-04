import numpy as np
import tensorflow as tf

def get_action(net, state, prev_action, prev_reward):
    state = np.expand_dims(state, 0)
    
    if prev_action == None:
        prev_action = -1
    if prev_reward == None:
        prev_reward = -1

    obs = {"obs": tf.convert_to_tensor(state, dtype=tf.float32)}

    predict = net.signatures["serving_default"](
        observations=tf.convert_to_tensor(state, dtype=tf.float32),
        seq_lens=tf.constant([0], dtype=tf.int32), 
        prev_action=tf.constant([prev_action], dtype=tf.int64),
        prev_reward=tf.constant([prev_reward], dtype=tf.float32), 
        is_training=tf.constant(False, dtype=tf.bool))

    action = predict["actions_0"].numpy()[0]
    return action

import sagemaker
import boto3
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

