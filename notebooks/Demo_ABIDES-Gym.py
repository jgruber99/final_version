#!/usr/bin/env python
# coding: utf-8

# # General Presentation

# ## Introduction

# This notebook is a demo of the use of ABIDES-Gym for training and testing a RL algorithm.
# 
# Details about ABIDES-Gym can be found on [arXiv](https://arxiv.org/pdf/2110.14771.pdf).
# 
# We assume the setup instructions in [ReadMe](https://github.com/jpmorganchase/abides-jpmc-public#readme) have been followed prior to running this notebook.
# 
# At the moment, ABIDES-Gym offers two trading environments:
# - Daily Investor Environment: This environment presents an example of the classic problem wherean investor tries to make money buying and selling a stock through-out a single day. 
# - Execution Environment: This environment presents an example of the optimal execution of a parent order problem. 
# 
# We will use the [ABIDES-Gym Execution environment](https://github.com/jpmorganchase/abides-jpmc-public/blob/main/abides-gym/abides_gym/envs/markets_execution_environment_v0.py) as an illustration of how to use ABIDES-Gym

# ## Problem setup 
# 
# 
# 
# This environment presents an example of the optimal execution of a parent order problem. The agent has either an initial inventory of the stocks it tries to trade out of or no initial inventory and tries to acquire a target number of shares. The goal is to realize this task while minimizing transaction cost from spreads and market impact. It does so by splitting the parent order into several smaller child orders.
# 
# 
# We can define the problem parameters:
# - $parentOrderSize$: Total size the agent has to execute (either buy or sell).
# - $direction$: direction of the parentOrder (buy or sell).
# - $timeWindow$: Time length the agent is given to proceed with parentOrderSize execution.
# - $entryPrice$ is the midPrice_t at the beginning of the episode
# - $nearTouch_t$ is the highest bidPrice if direction = buy else is the lowest askPrice
# - $penalty$: it is a constant penalty per non-executed share at the end of the timeWindow.
# 
# 
# 
# 
# 
# 

# ## Action Space

# The environment allows for three simple actions: "MARKET ORDER", "DO NOTHING" and "LIMIT ORDER". They are defined as follow:
# - "MARKET ORDER": the agent places a market order of size $childOrderSize$ in the direction $direction$. (Instruction to buy or sell immediately at the current best available price)
# - "LIMIT ORDER": the agent places a limit order of size $childOrderSize$ in the direction $direction$ at the price level $nearTouch_t$. (Buy or sell only at a specified price or better, does not guarantee execution)
# - "DO NOTHING": no action is taken.
# 
# Before sending a "MARKET ORDER" or a "LIMIT ORDER", the agent will cancel any living order still in the Order Book.

# ## Reward
# 
# $$R_{total} = R_{slippage} + R_{Late Penalty} = R_{slippage} + \lambda \times Quantity_{NotExecuted}$$
# 
# with: 
# 
# - $ \lambda = penaltyFactor $

# We define the step reward (slippage reward componant) as:
# 
# $$reward_t= \frac{PNL_t}{parentOrderSize}$$
# 
# with:
# 
# $$PNL_t = \sum_{o \in O_t } numside\cdot(entryPrice - fillPrice_{o})* quantity_{o}))$$
#     
# where $numside=1$ if direction is buy else $numside=0$ and $O_t$ is the set of orders executed between step $t-1$ and $t$
#     
#     
# We also define an episode update reward that is computed at the end of the episode. Denoting $O^{episode}$ the set of all orders executed in the episode, it is defined as: 
# 
# - 0 if $\sum_{o \in O^{episode}} quantity_{o} = parentOrderSize$
#         
# - else $|penalty \times ( parentOrderSize - \sum_{o \in O^{episode}} quantity_{o})|$
#         

# ## State Space

# The experimental agent perceives the market through the state representation:
#         
# $$s(t)=( holdingsPct_t, timePct_t, differencePct_t, imbalance5_t, imbalanceAll_t, priceImpact_t, spread_t, directionFeature_t, R^{k}_t)$$
#        
# where:
# - $holdingsPct_t = \frac{holdings_t}{parentOrderSize}$: the execution advancement
# - $timePct_t=\frac{t - startingTime}{timeWindow}$: the time advancement
# - $differencePct_t = holdingsPct_t - timePct_t$ 
# - $priceImpact_t = midPrice_t - entryPrice$
# - $imbalance_t=\frac{bids \ volume}{ bids \ volume + asks \ volume}$ using the first 3 levels of the order book. Value is respectively set to $0, 1, 0.5$ for no bids, no asks and empty book.
# - $spread_t=bestAsk_t-bestBid_t$
# - $directionFeature_t= midPrice_t-lastTransactionPrice_t$
# - $R^k_t=(r_t,...,r_{t-k+1})$ series of mid price differences, where $r_{t-i}=mid_{t-i}-mid_{t-i-1}$. It is set to 0 when undefined. By default $k=3$ 
#     

# # Imports

# In[3]:


# Only a few import are needed

import gym
import abides_gym

# In[9]:



# In[7]:


import gym

# # Exploration of ABIDES-Gym with Gym runner

# ## Import the specific environment and set the global seed 

# In[ ]:


#import abides gym 
env = gym.make(
        "markets-execution-v0",
        background_config="rmsc04"
    )
#set the general seed
env.seed(0)

# ## Start the simulation
# 
# A simulation is composed of different steps before enabling the Gym agent to act in the environment:
# - start the day (00:00:00): No agent can act on the market 
# - open the market (09:30:00): Background agents can act on the market
# - authorize the Gym agent to act (09:30:00 + N minutes) with N=5 minutes here
# 
# The sequence of actions is called by the classic Gym function:
# ```python
# .reset()
# ```
# 
# It returns the state of the market at time 09:30:00 + N minutes.

# In[ ]:


state = env.reset()

# In[ ]:


#The state features are as follow
#holdings_pct, time_pct, diff_pct, imbalance_all, imbalance_5, price_impact, spread, directiom, returns

print(state)

# ## Agent Action

# In[ ]:


env.env.action_space

# Three actions are available for the agent:
# - 0: MKT order of size order_fixed_size
# - 1: LMT order of size order_fixed_size
# - 2: DO NOTHING

# ## Step in the Simulation

# The step is called by the classic Gym function:
# ```python
# .step(action)
# ```
# 
# It returns the classic:
# - the state of the market after taking the action and waiting until the next time step
# - the reward obtained during the step 
# - whether the environment episode is finished 
# - info dictionnary: in default mode, return interesting metrics to track during the training of the agent. May be used in debug_mode for more information passed through

# In[ ]:


# agent does action 0 
state, reward, done, info = env.step(0)

# In[ ]:


print('State:')
print(state)
print('-------------------------------------')
print('Reward:')
print(reward)
print('-------------------------------------')
print('Done:')
print(done)
print('-------------------------------------')
print('Info:')
print(info)

# # Training an RL algorithm in ABIDES: Example with RLlib

# Remark: The notebook demonstrates the use of [RLlib](https://docs.ray.io/en/latest/rllib-algorithms.html) to train the execution agent, however any open-source library or own code will be compatible with ABIDES-Gym. 

# ## Imports

# In[ ]:


import ray
from ray import tune

# Import to register environments
import abides_gym


from ray.tune.registry import register_env
# import env 

from abides_gym.envs.markets_execution_environment_v0 import (
    SubGymMarketsExecutionEnv_v0,
)

# ## Runner
# 
# RLlib enables easy parallelization with for example here the training of a DQN agent in RMSC04 doing a grid_search on the value of the learning rate. 
# 
# We also run for 3 different global seeds to verify the robustness of our learning.
# 
# 
# **Note**: 
# - Would recommend to copy this cell in a python script and let it run in the back with [screen](https://linuxize.com/post/how-to-use-linux-screen/) or [tmux](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/) for instance. 
# - If you want to execute the cell, turn the execute_cell_flag to True

# In[ ]:


execute_cell_flag = False 

if execute_cell_flag:

    ray.shutdown()
    ray.init()

    """
    DQN's default:
    train_batch_size=32, sample_batch_size=4, timesteps_per_iteration=1000 -> workers collect chunks of 4 ts and add these to the replay buffer (of size buffer_size ts), then at each train call, at least 1000 ts are pulled altogether from the buffer (in batches of 32) to update the network.
    """
    register_env(
        "markets-execution-v0",
        lambda config: SubGymMarketsExecutionEnv_v0(**config),
    )


    name_xp = "dqn_execution_demo_3" #change to your convenience

    tune.run(
        "DQN",
        name=name_xp,
        resume=False,
        stop={"training_iteration": 100},  
        checkpoint_at_end=True,
        checkpoint_freq=5,
        config={
            "env": "markets-execution-v0",
            "env_config": {"background_config":"rmsc04",
                           "timestep_duration":"10S",
                           "execution_window": "04:00:00",
                           "parent_order_size": 20000,
                           "order_fixed_size": 50,
                           "not_enough_reward_update":-100,#penalty
             },
            "seed": tune.grid_search([1, 2, 3]),
            "num_gpus": 0,
            "num_workers": 0,
            "hiddens": [50, 20],
            "gamma": 1,
            "lr": tune.grid_search([0.001,0.0001, 0.01]),
            "framework": "torch",
            "observation_filter": "MeanStdFilter",
        },
    )

# The users can track its experimentation on:
# - ray dashboard: port 8265
# - TensorBoard: port 6006 
# 
# Tensorboard can be launched with the following command in terminal:
# 
# ```bash
# tensorboard --logdir=~/ray_results/{name_xp}
# ```
# 

# ![Screen%20Shot%202021-10-21%20at%205.57.58%20PM.png](attachment:Screen%20Shot%202021-10-21%20at%205.57.58%20PM.png)

# ![Screen%20Shot%202021-10-21%20at%209.12.48%20PM.png](attachment:Screen%20Shot%202021-10-21%20at%209.12.48%20PM.png)

# # Test of the algorithm - Rollouts

# This part illustrates the implementation of 4 hand-crafted policies and 1 Reinforcement Learning agent learned with RRlib.

# ## Define or load a policy

# In[ ]:


import numpy as np 
np.random.seed(0)
import ray.rllib.agents.dqn as dqn
from ray.tune import Analysis


#each policy needs to have:
# - an attribute name 
# - a method get_action


class policyPassive:
    def __init__(self):
        self.name = 'passive'
        
    def get_action(self, state):
        return 1
        
class policyAggressive:
    def __init__(self):
        self.name = 'aggressive'
        
    def get_action(self, state):
        return 0
    
class policyRandom:
    def __init__(self):
        self.name = 'random'
        
    def get_action(self, state):
        return np.random.choice([0,1])
    
class policyRandomWithNoAction:
    def __init__(self):
        self.name = 'random_no_action'
        
    def get_action(self, state):
        return np.random.choice([0,1, 2])
    
class policyRL:
    """
    policy learned during the training
    get the best policy from training {name_xp}
    Use this policy to compute action
    """
    def __init__(self):
        self.name = 'rl'
        
        ##### LOADING POLICY 
        # cell specific to ray to 
        #this part is to get a path
        # https://github.com/ray-project/ray/issues/4569#issuecomment-480453957
        name_xp = "dqn_execution_demo_4" 
        
        data_folder = f"~/ray_results/{name_xp}"
        analysis = Analysis(data_folder)
        trial_dataframes = analysis.trial_dataframes
        trials = list(trial_dataframes.keys())
        best_trial_path = analysis.get_best_logdir(metric='episode_reward_mean', mode='max')
        #can replace by string here - any checkpoint of your choice 
        best_checkpoint = analysis.get_best_checkpoint(trial = best_trial_path, mode='max')
        
        # define the meta parameters of the policy in order to set the trained
        config = dqn.DEFAULT_CONFIG.copy()
        config["framework"]= "torch"
        config["observation_filter"]= "MeanStdFilter"
        config["hiddens"]= [50, 20]
        config["env_config"]= {"background_config":"rmsc04",
                               "timestep_duration":"10S",
                               "execution_window": "04:00:00",
                               "parent_order_size": 20000,
                               "order_fixed_size": 50,
                               "not_enough_reward_update":-100,#penalty
                 }

        self.trainer = dqn.DQNTrainer(config=config, env = "markets-execution-v0")
        #load policy from checkpoint
        self.trainer.restore(best_checkpoint)
        
        
    def get_action(self, state):
        return self.trainer.compute_action(state)    
        

# ## Testing (rollout) infrastructure
# 
# **Note**: This part may be included in ABIDES-GYM source code in the future.

# In[ ]:


def generate_env(seed):
    """
    generates specific environment with the parameters defined and set the seed
    """
    env = gym.make(
            "markets-execution-v0",
            background_config="rmsc04",
        timestep_duration="10S",
                           execution_window= "04:00:00",
                           parent_order_size= 20000,
                           order_fixed_size= 50,
                           not_enough_reward_update=-100,#penalty

        )

    env.seed(seed)
    
    return env

# In[ ]:


from collections.abc import MutableMapping
import pandas as pd

def flatten_dict(d: MutableMapping, sep: str= '.') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict

def run_episode(seed = None, policy=None):
    """
    run fully one episode for a given seed and a given policy
    """
    
    env = generate_env(seed)
    state = env.reset()
    done = False
    episode_reward = 0 
    
    while not done:
        action = policy.get_action(state)
        state, reward, done, info = env.step(action)
        episode_reward += reward
    
    #could add a few more... 
    output = flatten_dict(info) 
    output['episode_reward'] = episode_reward
    output['name'] = policy.name
    return output
        
        
    

# In[ ]:


from p_tqdm import p_map
import pandas as pd 
from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
#from ray.util.multiprocessing import Pool



def run_N_episode(N):
    """
    run in parallel N episode of testing for the different policies defined in policies list
    heads-up: does not work yet for rllib policies - pickle error
    #https://stackoverflow.com/questions/28821910/how-to-get-around-the-pickling-error-of-python-multiprocessing-without-being-in

    need to run the rllib policies with following cell (not in parralel)
    
    """
    #define policies 
    policies = [policyAggressive(), policyRandom(), policyPassive(), policyRandomWithNoAction()]
    seeds = [i for i in range(N)]
    
    tests = [{"policy":policy, 'seed':seed} for policy in policies for seed in seeds]
    
    def wrap_run_episode(param):
        return run_episode(**param)
    
    outputs = p_map(wrap_run_episode, tests)
    #outputs = Pool().map(wrap_run_episode, tests)
    
    return outputs

N = 50
outputs = run_N_episode(N) 



# In[ ]:


#cannot do in parrallel here for now - pickling errors

from tqdm import tqdm
for i in tqdm(range(N)):
    outputs.append(run_episode(seed = i, policy=policyRL()))

# ## Plotting results

# In[ ]:


#convert results in a dataframe with all metrics
df = pd.DataFrame(outputs)
df['ep_len'] = df['action_counter.action_0'] + df['action_counter.action_1'] + df['action_counter.action_2']
df.head()

# In[ ]:


#group by policy name and sort
df_g = df.groupby(by='name').mean()

list_interest =  ['episode_reward', 'time_pct', 
               'holdings_pct', 'late_penalty_reward', 
               'imbalance_5', 'spread', 'action_counter.action_0', 'action_counter.action_1',
              'action_counter.action_2']

df_g.sort_values(by = 'episode_reward', ascending=False)[list_interest]

# Example of rollout comparison (output of the cell):
# ![Screen%20Shot%202021-10-22%20at%2011.52.52%20AM.png](attachment:Screen%20Shot%202021-10-22%20at%2011.52.52%20AM.png)

# In[ ]:


import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

# In[ ]:


#distribution plots
for metric in list_interest:
    print(f'======================={metric}========================')
    sns.displot(data = df, x=metric, hue='name', bins=10)
    plt.show()
    sns.displot(data = df, x=metric, hue='name',  kind="kde")
    plt.show()

# Example of rollout distribution plots (output of the cell):
# ![Screen%20Shot%202021-10-22%20at%2011.53.09%20AM.png](attachment:Screen%20Shot%202021-10-22%20at%2011.53.09%20AM.png)

# ![Screen%20Shot%202021-10-22%20at%2011.53.20%20AM.png](attachment:Screen%20Shot%202021-10-22%20at%2011.53.20%20AM.png)

# # Future ABIDES-Gym-Markets environment

# Plan for ABIDE-Gym environments to offer raw l2 data instead of computed features as state space. 
# These new environement will be called markets-XXX-v1.
