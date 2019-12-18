NUMBER_OF_PHYSICAL_NODE_TYPES = 3
NUMBER_OF_VIRTUAL_NODE_TYPES = 12

NODES_PER_ISP = 15
ISP_RADIUS = 500 #km
ALPHA = 1
BETA = 1

MIN_CUTOFF = 1
MAX_CUTOFF = 5

NUMBER_OF_ISPS = 5
AVERAGE_PEERING_NODES_PER_ISP = 1.5
MEAN_NUMBER_OF_PEERING_LINKS = 4

VIRTUAL_NODE_TYPES = [0,1,2,3,4,5,6,7,8,9]
PROB_OF_FEASIBILITY_ISP = 0.5
PROB_OF_FEASIBILITY_NODE = 0.5

PROB_NODE_CAPACITY_VERIFICATION = 0.1

INTER_LINKS_COSTS = [11,12,13,14,15]
INTRA_LINKS_COSTS = [6,7,8,9] #[100,200,300,400] #

MAX_COMPUTING_DEMAND_PER_NODE = 10
MAX_BANDIDTH_DEMAND_PER_PAIRS = 10
MAX_COST_PER_VIRTUAL_NODE = 10


CAPACITIES_OF_PHYSICAL_NODES = [30,40,50]#[11] #[50,70,80,90,100,110,120,130,140] #[1,2,3,4,5,6,7,8,9,10] #[20,30,40,50]
CAPACITIES_OF_PHYSICAL_LINKS = [50,100] 
CAPACITIES_OF_PEERING_LINKS = [200,400] 

N_VIRTUAL_NODES = 2
PROB_OF_VIRTUAL_LINK = 0.5

ALPHA_COST = 1
BETA_COST = 1

PATIENCE = 50000#number of consecutive epochs without improvement in the RL approach to early stop the training

#RL parameters 
EPS_EXTERNAL_NODE = 0.99
LEARNING_RATE_EXTERNAL_NODE = 0.125#0.1#0.1
DISCOUNT_FACTOR_EXTERNAL_NODE = 0.125#0.1#0.1#0.5#0.9#0.00001
NUM_OF_POSSIBLE_ACTIONS_EXTERNAL_NODE = 3

EPS_EXTERNAL_PATH = 0.95
LEARNING_RATE_EXTERNAL_PATH = 0.125#0.1#0.1
DISCOUNT_FACTOR_EXTERNAL_PATH = 0.125#0.1#0.1#0.5#0.00001
NUM_OF_POSSIBLE_ACTIONS_EXTERNAL_PATH = 3

EPS_INTERNAL_NODE = 0.8
LEARNING_RATE_INTERNAL_NODE = 0.125#0.1#0.5#0.2#0.1
DISCOUNT_FACTOR_INTERNAL_NODE = 0.125#0.1#0.1#0.5#0.00001
NUM_OF_POSSIBLE_ACTIONS_INTERNAL_NODE = 3

EPS_INTERNAL_PATH = 0.95
LEARNING_RATE_INTERNAL_PATH = 0.125#0.1#0.1
DISCOUNT_FACTOR_INTERNAL_PATH = 0.125#0.1#0.1#0.5#0.00001
NUM_OF_POSSIBLE_ACTIONS_INTERNAL_PATH = 3


# good parameter
"""
INFINITY_VALUE = 10**4 #10**2
ROUND_EXTERNAL_NODE = 1000 #every round_external_node the orchestrator re-assigns vfs to the isps  
ROUND_EXTERNAL_PATH = 500 #every round_external_path the orchestrator re-assigns the virtual paths to the peering nodes  
ROUND_INTERNAL_NODE = 200#1*10**3

INFINITY_VALUE = 10**4 #10**2
ROUND_EXTERNAL_NODE = 200 #every round_external_node the orchestrator re-assigns vfs to the isps  
ROUND_EXTERNAL_PATH = 50 #every round_external_path the orchestrator re-assigns the virtual paths to the peering nodes  
ROUND_INTERNAL_NODE = 10#1*10**3
"""

"""
INFINITY_VALUE = 10**4 #10**2
ROUND_EXTERNAL_NODE = 600 #every round_external_node the orchestrator re-assigns vfs to the isps  
ROUND_EXTERNAL_PATH = 200 #every round_external_path the orchestrator re-assigns the virtual paths to the peering nodes  
ROUND_INTERNAL_NODE = 50 #50#1*10**3
"""

"""
INFINITY_VALUE = 5*10**4 #10**2
ROUND_EXTERNAL_NODE = 10#10#3*60 #every round_external_node the orchestrator re-assigns vfs to the isps  
ROUND_EXTERNAL_PATH = 10#10#3*20 #every round_external_path the orchestrator re-assigns the virtual paths to the peering nodes  
ROUND_INTERNAL_NODE = 10#10#3*5#1*10**3
"""
INFINITY_VALUE = 1000 #1*10**4 #10**2
ROUND_EXTERNAL_NODE_PRIVATE_O1 = 10 #every round_external_node the orchestrator re-assigns vfs to the isps  
ROUND_EXTERNAL_PATH_PRIVATE_O1 = 50 #every round_external_path the orchestrator re-assigns the virtual paths to the peering nodes  
ROUND_INTERNAL_NODE_PRIVATE_O1 = 1#1*10**3
ROUND_INTERNAL_LINK_PRIVATE_O1 = 1000

ROUND_EXTERNAL_NODE_PRIVATE_O2 = 10 #every round_external_node the orchestrator re-assigns vfs to the isps  
ROUND_EXTERNAL_PATH_PRIVATE_O2 = 50 #every round_external_path the orchestrator re-assigns the virtual paths to the peering nodes  
ROUND_INTERNAL_NODE_PRIVATE_O2 = 1#1*10**3
ROUND_INTERNAL_LINK_PRIVATE_O2 = 500

ROUND_EXTERNAL_NODE_PRIVATE_O3 = 10 #every round_external_node the orchestrator re-assigns vfs to the isps  
ROUND_EXTERNAL_PATH_PRIVATE_O3 = 50 #every round_external_path the orchestrator re-assigns the virtual paths to the peering nodes  
ROUND_INTERNAL_NODE_PRIVATE_O3 = 1#1*10**3
ROUND_INTERNAL_LINK_PRIVATE_O3 = 100

ROUND_EXTERNAL_NODE_PRIVATE_O4 = 10 #every round_external_node the orchestrator re-assigns vfs to the isps  
ROUND_EXTERNAL_PATH_PRIVATE_O4 = 50 #every round_external_path the orchestrator re-assigns the virtual paths to the peering nodes  
ROUND_INTERNAL_NODE_PRIVATE_O4 = 1#1*10**3
ROUND_INTERNAL_LINK_PRIVATE_O4 = 1

ROUND_EXTERNAL_NODE = 10 #every round_external_node the orchestrator re-assigns vfs to the isps  
ROUND_EXTERNAL_PATH = 50 #every round_external_path the orchestrator re-assigns the virtual paths to the peering nodes  
ROUND_INTERNAL_NODE = 1#1*10**3
ROUND_INTERNAL_LINK = 1

N_EPOCHS = 5*10**5
N_SIMULATIONS = 50