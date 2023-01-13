#!/usr/bin/env python
# coding: utf-8

# This is a tutorial for basic use of abides_markets simulation.
# It uses the simulator directly without the OpenAI Gym interface

# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from abides_core import abides
from abides_core.utils import parse_logs_df, ns_date, str_to_ns, fmt_ts
from abides_markets.configs import rmsc04

# ## Build runnable configuration

# Here we are generating/building a config from the config file rmsc04. 
# The config object is a dictionnary containing key elements like, start time, end time, agents to be used, latency and computation delay models.

# In[3]:


config = rmsc04.build_config()

# In[4]:


config.keys()

# ## Running simulation

# Once the config is ready it can be run using the abides runner function.
# It instanciates a simulation kernel, runs the configuration and returns an end_state that mostly contains pointers to the different agent objects.
# The agents are in their final state, their internal variables can be accessed to extract informations of interests like logs.

# In[4]:


end_state = abides.run( config )

# ## Retrieving results from end_state

# By convention in abides_markets configuration the first agent is the exchange agent. 
# It contains an order_book. By default it logs its history.

# In[5]:


order_book = end_state["agents"][0].order_books["ABM"]

# ### Order book history L1

# L1 data snapshots for every tick can be extracted
# ( best bid and ask price and quantity )

# In[6]:


L1 = order_book.get_L1_snapshots()

# Here we plot the time series of the best bid and best ask price thoughout the simulation

# In[42]:


best_bids = pd.DataFrame(L1["best_bids"],columns=["time","price","qty"])
best_asks = pd.DataFrame(L1["best_asks"],columns=["time","price","qty"])

## All times are in ns from 1970, remove the date component to put them in ns from midnight
best_bids["time"] = best_bids["time"].apply( lambda x: x - ns_date(x) )
best_asks["time"] = best_asks["time"].apply( lambda x: x - ns_date(x) )

plt.plot(best_bids.time,best_bids.price)
plt.plot(best_asks.time,best_asks.price)

band = 100
plt.ylim(100_000-band,100_000+band)

time_mesh = np.arange(
    str_to_ns("09:30:00"),
    str_to_ns("10:10:00"),
    1e9*60*10
)
_=plt.xticks(time_mesh, [ fmt_ts(time).split(" ")[1] for time in time_mesh], rotation=60 )

# ### Order book history L2

# L2 data snapshots for every tick can be extracted
# ( bids and asks price and quantity for every orderbook level. Here max depth logged is a parameter of the simulation and max number of levels we want to retrieve from the orderbook after the simulation is a parameter too)

# In[8]:


L2 = order_book.get_L2_snapshots(nlevels=10)

# As an illustration we plot the time series of the fifth best bid price and fifth best ask price throughout the simulation

# In[43]:


## plotting fifth best bid and fifth best ask
times = [ t - ns_date(t) for t in L2["times"] ]
plt.scatter( times, L2["bids"][:,5,0], s=.5 )
plt.scatter( times, L2["asks"][:,5,0], s=.5 )

band = 100
plt.ylim(100_000-band,100_000+band)

_=plt.xticks(time_mesh, [ fmt_ts(time).split(" ")[1] for time in time_mesh], rotation=60 )

# ### Looking at agents logs

# All agents can be inspected to retrieve desired information. 
# The utility parse_logs_df for instance provides a quick way to retrieve and aggregate the log variables of each agent in a single dataframe

# In[10]:


logs_df = parse_logs_df( end_state )

# #### Histogram of order submission times for noise agents

# As an illustration we retrieve the submission times of all the orders sent by noise agent and display the histogram of all these times

# In[44]:


plt.hist( logs_df[ (logs_df.agent_type == "NoiseAgent") & (logs_df.EventType=="ORDER_SUBMITTED") ].EventTime.apply(lambda x: x - ns_date(x) )  )

_=plt.xticks(time_mesh, [ fmt_ts(time).split(" ")[1] for time in time_mesh], rotation=60 )





# We proceed the same way for value agents as well

# In[45]:


plt.hist( logs_df[ (logs_df.agent_type == "ValueAgent") & (logs_df.EventType=="ORDER_SUBMITTED") ].EventTime.apply(lambda x: x - ns_date(x) )  )

_=plt.xticks(time_mesh, [ fmt_ts(time).split(" ")[1] for time in time_mesh], rotation=60 )
