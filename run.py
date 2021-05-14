import matplotlib.pyplot as plt 
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
from president import PresidentEnv
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

## Adapted from 'The TF-Agents Authors' © 2021
## https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial

## DQN Architecture Method
# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_in', distribution='truncated_normal')
    )

## Data Collection Methods
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


if __name__ == "__main__":
    ## Enable GPU Acceleration
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    ## Define Players' Policy
    policy = 'random'

    ## Hyperparameters
    num_iterations = 20000 # @param {type:"integer"}

    initial_collect_steps = 100  # @param {type:"integer"} 
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 0.025  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}
    
    ## Create Environment
    env = PresidentEnv(policy)
    train_env = tf_py_environment.TFPyEnvironment(PresidentEnv(policy))
    eval_env = tf_py_environment.TFPyEnvironment(PresidentEnv(policy))
    
    ## DQN Architecture
    fc_layer_params = (1024, 512, 256, 128, 64)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=0.5, mode='fan_in', distribution='truncated_normal')
        # kernel_initializer=tf.keras.initializers.RandomUniform(
        #     minval=-0.25, maxval=0.25),
        # bias_initializer=tf.keras.initializers.Constant(-0.25)
    )
    q_net = sequential.Sequential(dense_layers + [q_values_layer])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=0.9, 
                                         beta_2=0.999, 
                                         epsilon=1e-07)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate
    #                                     # momentum=0.9,
    #                                     # nesterov=True
    #                                     )
    train_step_counter = tf.Variable(0)
    
    ## DQN Compile Agent
    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)
    agent.initialize()
    
    ## Policy Setup
    eval_policy = agent.policy
    collect_policy = agent.collect_policy
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())
    ## Replay Buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)
    
    collect_data(train_env, random_policy, replay_buffer, 
                 initial_collect_steps)
    
    ## Create TensorFlow Dataset
    dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size,
    num_steps=2,
    single_deterministic_pass=False).prefetch(3)
    iterator = iter(dataset)
    
    ## Train DQN Agent
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_data(train_env, agent.collect_policy, replay_buffer,
                     collect_steps_per_iteration)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

    ## Save Preparation
    path = os.path.dirname(__file__)
    os.chdir(path)
    if not os.path.exists('figures'):
        os.makedirs('figures')
    now = datetime.now()
    date = now.strftime("(%m.%d-%H.%M.%S)")
    
    ## Data Visualization
    iterations = range(0, num_iterations + 1, eval_interval)
    plt.figure(figsize=(10, 5), dpi=180)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=5)
    plt.title(f'Average Return Against {policy.title()} Policy')
    plt.savefig(f'{path}/figures/{policy}_policy_{date}.png')
