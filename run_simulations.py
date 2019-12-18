from utils import *
from utils_RL import * 
from utils_BM import * 
from config import *
from matplotlib import pyplot as plt 
from time import time 

#RL parameters 
eps_external_path = EPS_EXTERNAL_PATH#0.9999999
learning_rate_external_path = LEARNING_RATE_EXTERNAL_PATH
discount_factor_external_path = DISCOUNT_FACTOR_EXTERNAL_PATH
num_of_possible_actions_external_path = NUM_OF_POSSIBLE_ACTIONS_EXTERNAL_PATH

eps_external_node = EPS_EXTERNAL_NODE
learning_rate_external_node = LEARNING_RATE_EXTERNAL_NODE 
discount_factor_external_node = DISCOUNT_FACTOR_EXTERNAL_NODE
num_of_possible_actions_external_node = NUM_OF_POSSIBLE_ACTIONS_EXTERNAL_NODE

eps_internal_path = EPS_INTERNAL_PATH 
learning_rate_internal_path = LEARNING_RATE_INTERNAL_PATH 
discount_factor_internal_path = DISCOUNT_FACTOR_INTERNAL_PATH 
num_of_possible_actions_internal_path = NUM_OF_POSSIBLE_ACTIONS_INTERNAL_PATH

eps_internal_node = EPS_INTERNAL_NODE 
learning_rate_internal_node = LEARNING_RATE_INTERNAL_NODE
discount_factor_internal_node = DISCOUNT_FACTOR_INTERNAL_NODE 
num_of_possible_actions_internal_node = NUM_OF_POSSIBLE_ACTIONS_INTERNAL_PATH

patience = PATIENCE
prob_node_capacity_verification = PROB_NODE_CAPACITY_VERIFICATION

nodes = NODES_PER_ISP
types_of_physical_nodes = np.arange(NUMBER_OF_PHYSICAL_NODE_TYPES)
radius = ISP_RADIUS
alpha = ALPHA 
beta = BETA
number_of_ISPs = NUMBER_OF_ISPS 
average_peering_nodes_per_ISP = AVERAGE_PEERING_NODES_PER_ISP
total_number_of_peering_nodes = int(average_peering_nodes_per_ISP*number_of_ISPs)
virtual_node_types = VIRTUAL_NODE_TYPES
n_virtual_nodes = N_VIRTUAL_NODES 
prob_of_virtual_link = PROB_OF_VIRTUAL_LINK 
types_of_virtual_nodes = VIRTUAL_NODE_TYPES 
max_cost_per_virtual_node = MAX_COST_PER_VIRTUAL_NODE
capacities_of_physical_nodes = CAPACITIES_OF_PHYSICAL_NODES 
capacities_of_physical_links = CAPACITIES_OF_PHYSICAL_LINKS
capacities_of_peering_links = CAPACITIES_OF_PEERING_LINKS

prob_of_feasibility_isp = PROB_OF_FEASIBILITY_ISP 
prob_of_feasibility_node = PROB_OF_FEASIBILITY_NODE

inter_links_costs = INTER_LINKS_COSTS 
intra_links_costs = INTRA_LINKS_COSTS 
max_computing_demand_per_node = MAX_COMPUTING_DEMAND_PER_NODE
max_bandwidth_demand_per_pairs = MAX_BANDIDTH_DEMAND_PER_PAIRS

infinity_value = INFINITY_VALUE
n_epochs = N_EPOCHS
round_external_node = ROUND_EXTERNAL_NODE
round_external_path = ROUND_EXTERNAL_PATH
round_internal_node = ROUND_INTERNAL_NODE
round_internal_link = ROUND_INTERNAL_LINK

round_external_node_private_o1 = ROUND_EXTERNAL_NODE_PRIVATE_O1
round_external_path_private_o1 = ROUND_EXTERNAL_PATH_PRIVATE_O1
round_internal_node_private_o1 = ROUND_INTERNAL_NODE_PRIVATE_O1
round_internal_link_private_o1 = ROUND_INTERNAL_LINK_PRIVATE_O1

round_external_node_private_o2 = ROUND_EXTERNAL_NODE_PRIVATE_O2
round_external_path_private_o2 = ROUND_EXTERNAL_PATH_PRIVATE_O2
round_internal_node_private_o2 = ROUND_INTERNAL_NODE_PRIVATE_O2
round_internal_link_private_o2 = ROUND_INTERNAL_LINK_PRIVATE_O2

round_external_node_private_o3 = ROUND_EXTERNAL_NODE_PRIVATE_O3
round_external_path_private_o3 = ROUND_EXTERNAL_PATH_PRIVATE_O3
round_internal_node_private_o3 = ROUND_INTERNAL_NODE_PRIVATE_O3
round_internal_link_private_o3 = ROUND_INTERNAL_LINK_PRIVATE_O3

round_external_node_private_o4 = ROUND_EXTERNAL_NODE_PRIVATE_O4
round_external_path_private_o4 = ROUND_EXTERNAL_PATH_PRIVATE_O4
round_internal_node_private_o4 = ROUND_INTERNAL_NODE_PRIVATE_O4
round_internal_link_private_o4 = ROUND_INTERNAL_LINK_PRIVATE_O4

alpha_cost = ALPHA_COST 
beta_cost = BETA_COST

min_cutoff = MIN_CUTOFF
max_cutoff = MAX_CUTOFF

all_final_costs = []
all_final_costs_ilp1 = []
all_final_costs_ilp2 = []
all_sim_times = []

what_to_run = ['LID','FID','RL','RL_private_o1','RL_private_o2','RL_private_o3','RL_private_o4']
#what_to_run = ['RL','LID']

results = {}
n_simulations = N_SIMULATIONS
for n_sim in range(n_simulations):

	print('sim',n_sim)
	t0 = time()

	results[n_sim] = {}
	results[n_sim]['LID'] = {}
	results[n_sim]['LID']['time'] = None 
	results[n_sim]['LID']['cost'] = None 

	results[n_sim]['FID'] = {}
	results[n_sim]['FID']['time'] = None 
	results[n_sim]['FID']['cost'] = None 

	results[n_sim]['RL'] = {}
	results[n_sim]['RL']['time'] = None 
	results[n_sim]['RL']['cost'] = []

	"""
	results[n_sim]['RL_private_o1'] = {}
	results[n_sim]['RL_private_o1']['time'] = None 
	results[n_sim]['RL_private_o1']['cost'] = []

	results[n_sim]['RL_private_o2'] = {}
	results[n_sim]['RL_private_o2']['time'] = None 
	results[n_sim]['RL_private_o2']['cost'] = []	
	"""

	results[n_sim]['RL_private_o1'] = {}
	results[n_sim]['RL_private_o1']['time'] = None 
	results[n_sim]['RL_private_o1']['cost'] = []
	

	results[n_sim]['RL_private_o2'] = {}
	results[n_sim]['RL_private_o2']['time'] = None 
	results[n_sim]['RL_private_o2']['cost'] = []

	results[n_sim]['RL_private_o3'] = {}
	results[n_sim]['RL_private_o3']['time'] = None 
	results[n_sim]['RL_private_o3']['cost'] = []

	results[n_sim]['RL_private_o4'] = {}
	results[n_sim]['RL_private_o4']['time'] = None 
	results[n_sim]['RL_private_o4']['cost'] = []

	while 1:
		virtual_graph, dict_virtual_node_types, repeat = define_virtual_graph(n_virtual_nodes, prob_of_virtual_link,types_of_virtual_nodes)
		if repeat == 0:
			break 
	
	res = get_adjacencies_matrices(number_of_ISPs,average_peering_nodes_per_ISP,alpha,beta,radius,types_of_physical_nodes, nodes,virtual_graph, capacities_of_physical_nodes,capacities_of_physical_links,capacities_of_peering_links,types_of_virtual_nodes,prob_of_feasibility_isp)
	feasibile_vfs_types_per_isp, all_adjacency_matrices, info_per_physical_node, dict_local_to_global_node, global_position_of_nodes, peering_links, dict_peering_to_local, dict_local_to_peering, peering_nodes, dict_of_capacities_of_physical_nodes, capacity_of_all_links, stop = res 	

	if not stop:
		
		where_vfs_can_be_hosted = get_feasible_association_vfs_isps(dict_virtual_node_types,feasibile_vfs_types_per_isp)
		feasibility_matrix = define_feasibility_matrix(all_adjacency_matrices,info_per_physical_node,prob_of_feasibility_isp,prob_of_feasibility_node,where_vfs_can_be_hosted,infinity_value)
		cost_per_virtual_node, demand_per_virtual_node = define_virtual_node_cost_matrix(all_adjacency_matrices,virtual_node_types,number_of_ISPs,info_per_physical_node,max_computing_demand_per_node,n_virtual_nodes,dict_virtual_node_types,max_cost_per_virtual_node)
		cost_per_link_matrix = define_physical_links_cost_matrix(all_adjacency_matrices,info_per_physical_node,inter_links_costs, intra_links_costs)
		cost_per_virtual_links, bandwidth_requests_between_virtual_nodes, shortest_path_per_each_pair_of_peering = define_bandwidth_cost_for_all_possible_assignments_in_peering_network(cost_per_link_matrix,all_adjacency_matrices,virtual_node_types,max_bandwidth_demand_per_pairs,dict_peering_to_local,dict_local_to_peering,n_virtual_nodes)
		
		print(bandwidth_requests_between_virtual_nodes,'bw')

		go = 1 # go only if every vf can be hosted by at least an ISP, otherwise the solution will always be infeasible
		for vf in where_vfs_can_be_hosted:
			if len(where_vfs_can_be_hosted[vf]) < 1:
				go = 0

		print("qui")

		if go:

			if 'LID' in what_to_run:

				t_start_lid = time()

				x_solution_ilp1, y_solution_ilp1 = solve_ilp1(where_vfs_can_be_hosted,cost_per_virtual_node,cost_per_virtual_links,all_adjacency_matrices,feasibility_matrix,dict_local_to_peering,infinity_value,info_per_physical_node)
				#cost_ilp1 = compute_cost_ilp1(x_solution_ilp1,y_solution_ilp1,cost_per_virtual_node,cost_per_virtual_links,dict_local_to_peering,info_per_physical_node,where_vfs_can_be_hosted,infinity_value)

				""" # i replae this piece of code with another one beacuse the ilp1 is giving inconsistencies problems! so it is not correct to compute the traversed nodes in this way
				traversed_links = []
				for local_p1 in range(y_solution_ilp1.shape[0]):
					for local_p2 in range(y_solution_ilp1.shape[1]):
						if np.sum(y_solution_ilp1[local_p1,local_p2]) > 0:

							gl_p1 = dict_local_to_peering[local_p1]
							gl_p2 = dict_local_to_peering[local_p2]

							traversed_links.append([gl_p1,gl_p2])
				"""

				vfs_assigned_per_isp = {}
				for vf in range(x_solution_ilp1.shape[1]):
					hosting_peering = np.where(x_solution_ilp1[:,vf] > 0)[0][0]
					hosting_peering = dict_local_to_peering[hosting_peering]
					owner_isp = info_per_physical_node[hosting_peering]['ISP']

					if owner_isp not in vfs_assigned_per_isp:
						vfs_assigned_per_isp[owner_isp] = []
					vfs_assigned_per_isp[owner_isp].append(vf)


				cost_ilp1, traversed_links = compute_cost_ilp1_replaced(x_solution_ilp1,cost_per_virtual_node,cost_per_virtual_links,dict_local_to_peering,info_per_physical_node,where_vfs_can_be_hosted,infinity_value,vfs_assigned_per_isp,all_adjacency_matrices,bandwidth_requests_between_virtual_nodes)

				cost_ilp2 = 0
				cost_ilp2_embedding = 0

				phy_node_per_vf_node = {}
				for curr_isp in vfs_assigned_per_isp:
					x_solution_ilp2, f_solution_ilp2, virtual_traffic_matrix_inside_curr_isp, dict_global_virtual_id_to_local_virtual_id = solve_ilp2(all_adjacency_matrices,vfs_assigned_per_isp,curr_isp,x_solution_ilp1,dict_local_to_peering,info_per_physical_node,bandwidth_requests_between_virtual_nodes,dict_local_to_global_node,feasibility_matrix,demand_per_virtual_node,dict_of_capacities_of_physical_nodes,capacity_of_all_links)
					
					#print("sum f",np.sum(f_solution_ilp2))
					cost, cost_embedding, cost_links, curr_traversed_links = compute_cost_ilp2(x_solution_ilp2,f_solution_ilp2,curr_isp,feasibility_matrix,vfs_assigned_per_isp,dict_local_to_global_node,demand_per_virtual_node,cost_per_virtual_node,virtual_traffic_matrix_inside_curr_isp,cost_per_link_matrix)
					cost_ilp2 += cost 
					cost_ilp2_embedding += cost_embedding

					"""
					#print("where are hosted the vfs?")
					for gl_vf in vfs_assigned_per_isp[curr_isp]:
						loc_vf = dict_global_virtual_id_to_local_virtual_id[gl_vf]
						loc_phy = np.where(x_solution_ilp2[:,loc_vf]> 0)[0][0]
						glob_phy = dict_local_to_global_node['ISP_' + str(curr_isp) + '_node_' + str(loc_phy)]
						phy_node_per_vf_node[gl_vf] = glob_phy

						#print(gl_vf,'in ',glob_phy,'in isp',curr_isp)
					"""
					for el in curr_traversed_links:
						if el not in traversed_links:
							traversed_links.append(el)

				#print(phy_node_per_vf_node)
				#draw_graph(all_adjacency_matrices,phy_node_per_vf_node,traversed_links)

				#print(cost_ilp1,cost_ilp2)
				total_cost = cost_ilp1 + cost_ilp2
				#print("total cost",total_cost,'of which ilp1',cost_ilp1,'and ilp2',cost_ilp2)
				#print("cost embedding ilp2 ",cost_ilp2_embedding)
				print("total cost LID is ",total_cost)

				t_finish_lid = time()

				results[n_sim]['LID']['time'] = t_finish_lid - t_start_lid 
				results[n_sim]['LID']['cost'] = total_cost

			if 'FID' in what_to_run:

				t_start_fid = time()

				x_solution, f_solution = solve_ILP_FID(all_adjacency_matrices,bandwidth_requests_between_virtual_nodes,feasibility_matrix,demand_per_virtual_node,dict_of_capacities_of_physical_nodes,capacity_of_all_links)

				total_cost = compute_cost_ILP_FID(x_solution,f_solution,feasibility_matrix,demand_per_virtual_node,cost_per_virtual_node,bandwidth_requests_between_virtual_nodes,cost_per_link_matrix)
				print("total cost FID is ",total_cost)

				traversed_links = []
				for phy_node1 in range(f_solution.shape[0]):
					for phy_node2 in range(f_solution.shape[1]):
						if np.sum(f_solution[phy_node1,phy_node2]) > 0:
							traversed_links.append([phy_node1,phy_node2])

				#print("sum ",np.sum(f_solution))

				phy_node_per_vf_node = {}
				for vf in range(x_solution.shape[1]):
					phy = np.where(x_solution[:,vf]>0)[0][0]
					phy_node_per_vf_node[vf] = phy

				#print(phy_node_per_vf_node)
				#draw_graph(all_adjacency_matrices,phy_node_per_vf_node,traversed_links)

				t_finish_fid = time()

				results[n_sim]['FID']['time'] = t_finish_fid - t_start_fid
				results[n_sim]['FID']['cost'] = total_cost

			if 'RL' in what_to_run:
				#t_start_rl = time()
				out_ = preprocess_data_for_RL(cost_per_virtual_node,info_per_physical_node,number_of_ISPs,all_adjacency_matrices)

				mapping_singledomain_to_multidomain_node, mapping_multidomain_to_singledomain_node, all_isps, N_nodes_dict, cost_per_vnf  = out_
				vnf_graph = all_adjacency_matrices['virtual_graph']
				multidomain_adj = all_adjacency_matrices['multidomain']

				current_states, Q_dict, overall_mapping_dict = get_initial_states_and_Q(all_adjacency_matrices,all_isps,virtual_graph,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix,min_cutoff,max_cutoff, num_of_possible_actions_internal_node, num_of_possible_actions_internal_path, num_of_possible_actions_external_node, num_of_possible_actions_external_path)
				
				initial_states  = copy.copy(current_states) 
				initial_Q = copy.copy(Q_dict)
				initial_dict = copy.copy(overall_mapping_dict)
				
				total_cost, results[n_sim]['RL']['cost'],tot_time = learning_function_(current_states, Q_dict, overall_mapping_dict,results[n_sim]['RL']['cost'], round_external_node, round_external_path, round_internal_node,round_internal_link,all_adjacency_matrices,all_isps,virtual_graph,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix,min_cutoff,max_cutoff, num_of_possible_actions_internal_node, num_of_possible_actions_internal_path, num_of_possible_actions_external_node, num_of_possible_actions_external_path, n_epochs,eps_external_node,eps_external_path,eps_internal_node,eps_internal_path,learning_rate_internal_node, learning_rate_external_node,learning_rate_internal_path,learning_rate_external_path,discount_factor_internal_node, discount_factor_external_node, discount_factor_internal_path, discount_factor_external_path,bandwidth_requests_between_virtual_nodes,cost_per_link_matrix,cost_per_virtual_node,demand_per_virtual_node,dict_of_capacities_of_physical_nodes,infinity_value,capacity_of_all_links,patience,prob_node_capacity_verification)

			if 'RL_private_o1' in what_to_run:
				#t_start_rl = time()
				out_ = preprocess_data_for_RL(cost_per_virtual_node,info_per_physical_node,number_of_ISPs,all_adjacency_matrices)

				mapping_singledomain_to_multidomain_node, mapping_multidomain_to_singledomain_node, all_isps, N_nodes_dict, cost_per_vnf  = out_
				vnf_graph = all_adjacency_matrices['virtual_graph']
				multidomain_adj = all_adjacency_matrices['multidomain']

				current_states, Q_dict, overall_mapping_dict = get_initial_states_and_Q(all_adjacency_matrices,all_isps,virtual_graph,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix,min_cutoff,max_cutoff, num_of_possible_actions_internal_node, num_of_possible_actions_internal_path, num_of_possible_actions_external_node, num_of_possible_actions_external_path)
				
				initial_states  = copy.copy(current_states) 
				initial_Q = copy.copy(Q_dict)
				initial_dict = copy.copy(overall_mapping_dict)
				
				total_cost, results[n_sim]['RL_private_o1']['cost'],tot_time = learning_function_private(current_states, Q_dict, overall_mapping_dict,results[n_sim]['RL_private_o1']['cost'], round_external_node_private_o1, round_external_path_private_o1, round_internal_node_private_o1,round_internal_link_private_o1,all_adjacency_matrices,all_isps,virtual_graph,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix,min_cutoff,max_cutoff, num_of_possible_actions_internal_node, num_of_possible_actions_internal_path, num_of_possible_actions_external_node, num_of_possible_actions_external_path, n_epochs,eps_external_node,eps_external_path,eps_internal_node,eps_internal_path,learning_rate_internal_node, learning_rate_external_node,learning_rate_internal_path,learning_rate_external_path,discount_factor_internal_node, discount_factor_external_node, discount_factor_internal_path, discount_factor_external_path,bandwidth_requests_between_virtual_nodes,cost_per_link_matrix,cost_per_virtual_node,demand_per_virtual_node,dict_of_capacities_of_physical_nodes,infinity_value,capacity_of_all_links,patience,prob_node_capacity_verification)

			if 'RL_private_o2' in what_to_run:
				#t_start_rl = time()
				out_ = preprocess_data_for_RL(cost_per_virtual_node,info_per_physical_node,number_of_ISPs,all_adjacency_matrices)

				mapping_singledomain_to_multidomain_node, mapping_multidomain_to_singledomain_node, all_isps, N_nodes_dict, cost_per_vnf  = out_
				vnf_graph = all_adjacency_matrices['virtual_graph']
				multidomain_adj = all_adjacency_matrices['multidomain']

				current_states, Q_dict, overall_mapping_dict = get_initial_states_and_Q(all_adjacency_matrices,all_isps,virtual_graph,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix,min_cutoff,max_cutoff, num_of_possible_actions_internal_node, num_of_possible_actions_internal_path, num_of_possible_actions_external_node, num_of_possible_actions_external_path)
				
				initial_states  = copy.copy(current_states) 
				initial_Q = copy.copy(Q_dict)
				initial_dict = copy.copy(overall_mapping_dict)
				
				total_cost, results[n_sim]['RL_private_o2']['cost'],tot_time = learning_function_private(current_states, Q_dict, overall_mapping_dict,results[n_sim]['RL_private_o2']['cost'], round_external_node_private_o2, round_external_path_private_o2, round_internal_node_private_o2,round_internal_link_private_o2,all_adjacency_matrices,all_isps,virtual_graph,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix,min_cutoff,max_cutoff, num_of_possible_actions_internal_node, num_of_possible_actions_internal_path, num_of_possible_actions_external_node, num_of_possible_actions_external_path, n_epochs,eps_external_node,eps_external_path,eps_internal_node,eps_internal_path,learning_rate_internal_node, learning_rate_external_node,learning_rate_internal_path,learning_rate_external_path,discount_factor_internal_node, discount_factor_external_node, discount_factor_internal_path, discount_factor_external_path,bandwidth_requests_between_virtual_nodes,cost_per_link_matrix,cost_per_virtual_node,demand_per_virtual_node,dict_of_capacities_of_physical_nodes,infinity_value,capacity_of_all_links,patience,prob_node_capacity_verification)

			if 'RL_private_o3' in what_to_run:
				#t_start_rl = time()
				out_ = preprocess_data_for_RL(cost_per_virtual_node,info_per_physical_node,number_of_ISPs,all_adjacency_matrices)

				mapping_singledomain_to_multidomain_node, mapping_multidomain_to_singledomain_node, all_isps, N_nodes_dict, cost_per_vnf  = out_
				vnf_graph = all_adjacency_matrices['virtual_graph']
				multidomain_adj = all_adjacency_matrices['multidomain']

				current_states, Q_dict, overall_mapping_dict = get_initial_states_and_Q(all_adjacency_matrices,all_isps,virtual_graph,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix,min_cutoff,max_cutoff, num_of_possible_actions_internal_node, num_of_possible_actions_internal_path, num_of_possible_actions_external_node, num_of_possible_actions_external_path)
				
				initial_states  = copy.copy(current_states) 
				initial_Q = copy.copy(Q_dict)
				initial_dict = copy.copy(overall_mapping_dict)
				
				total_cost, results[n_sim]['RL_private_o3']['cost'],tot_time = learning_function_private(current_states, Q_dict, overall_mapping_dict,results[n_sim]['RL_private_o3']['cost'], round_external_node_private_o3, round_external_path_private_o3, round_internal_node_private_o3,round_internal_link_private_o3,all_adjacency_matrices,all_isps,virtual_graph,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix,min_cutoff,max_cutoff, num_of_possible_actions_internal_node, num_of_possible_actions_internal_path, num_of_possible_actions_external_node, num_of_possible_actions_external_path, n_epochs,eps_external_node,eps_external_path,eps_internal_node,eps_internal_path,learning_rate_internal_node, learning_rate_external_node,learning_rate_internal_path,learning_rate_external_path,discount_factor_internal_node, discount_factor_external_node, discount_factor_internal_path, discount_factor_external_path,bandwidth_requests_between_virtual_nodes,cost_per_link_matrix,cost_per_virtual_node,demand_per_virtual_node,dict_of_capacities_of_physical_nodes,infinity_value,capacity_of_all_links,patience,prob_node_capacity_verification)

			if 'RL_private_o4' in what_to_run:
				#t_start_rl = time()
				out_ = preprocess_data_for_RL(cost_per_virtual_node,info_per_physical_node,number_of_ISPs,all_adjacency_matrices)

				mapping_singledomain_to_multidomain_node, mapping_multidomain_to_singledomain_node, all_isps, N_nodes_dict, cost_per_vnf  = out_
				vnf_graph = all_adjacency_matrices['virtual_graph']
				multidomain_adj = all_adjacency_matrices['multidomain']

				current_states, Q_dict, overall_mapping_dict = get_initial_states_and_Q(all_adjacency_matrices,all_isps,virtual_graph,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix,min_cutoff,max_cutoff, num_of_possible_actions_internal_node, num_of_possible_actions_internal_path, num_of_possible_actions_external_node, num_of_possible_actions_external_path)
				
				initial_states  = copy.copy(current_states) 
				initial_Q = copy.copy(Q_dict)
				initial_dict = copy.copy(overall_mapping_dict)
				
				total_cost, results[n_sim]['RL_private_o4']['cost'],tot_time = learning_function_private(current_states, Q_dict, overall_mapping_dict,results[n_sim]['RL_private_o4']['cost'], round_external_node_private_o4, round_external_path_private_o4, round_internal_node_private_o4,round_internal_link_private_o4,all_adjacency_matrices,all_isps,virtual_graph,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix,min_cutoff,max_cutoff, num_of_possible_actions_internal_node, num_of_possible_actions_internal_path, num_of_possible_actions_external_node, num_of_possible_actions_external_path, n_epochs,eps_external_node,eps_external_path,eps_internal_node,eps_internal_path,learning_rate_internal_node, learning_rate_external_node,learning_rate_internal_path,learning_rate_external_path,discount_factor_internal_node, discount_factor_external_node, discount_factor_internal_path, discount_factor_external_path,bandwidth_requests_between_virtual_nodes,cost_per_link_matrix,cost_per_virtual_node,demand_per_virtual_node,dict_of_capacities_of_physical_nodes,infinity_value,capacity_of_all_links,patience,prob_node_capacity_verification)




				"""
				for round_external_node_ in [5]:
					for round_external_path_ in [20]:
						for round_internal_node_ in [1,5]:
							for eps_external_node_ in [0.8,0.9]:
								for eps_external_path_ in [0.99]:
									for eps_internal_node_ in [0.8,0.9]:
										for eps_internal_path_ in [0.99]:
				
											id_ = 'ex_node_' + str(round_external_node_) + '_' + str(eps_external_node_) + '_ex_path_' + str(round_external_path_) + '_' + str(eps_external_path_) + '_int_node' + str(round_internal_node_) + '_' + str(eps_internal_node_) 			

											results[n_sim]['RL'][id_] = {}
											results[n_sim]['RL'][id_]['time'] = None 
											results[n_sim]['RL'][id_]['cost'] = []

											total_cost, results[n_sim]['RL'][id_]['cost'],tot_time = learning_function_(initial_states, initial_Q, initial_dict,results[n_sim]['RL']['cost'], round_external_node_, round_external_path_, round_internal_node_,all_adjacency_matrices,all_isps,virtual_graph,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix,min_cutoff,max_cutoff, num_of_possible_actions_internal_node, num_of_possible_actions_internal_path, num_of_possible_actions_external_node, num_of_possible_actions_external_path, n_epochs,eps_external_node_,eps_external_path_,eps_internal_node_,eps_internal_path_,learning_rate_internal_node, learning_rate_external_node,learning_rate_internal_path,learning_rate_external_path,discount_factor_internal_node, discount_factor_external_node, discount_factor_internal_path, discount_factor_external_path,bandwidth_requests_between_virtual_nodes,cost_per_link_matrix,cost_per_virtual_node,demand_per_virtual_node,dict_of_capacities_of_physical_nodes,infinity_value,capacity_of_all_links)

											np.save('results_simulations',results)
				"""
				#t_finish_rl = time()
				"""
				repeat_rl = 1
				while 1:

					print("repeat?")
					repeat_rl = input()

					if 'y' in repeat_rl:

						print("external node")
						round_external_node = input() 
						round_external_node = int(round_external_node)

						print("internal node")
						round_internal_node = input() 
						round_internal_node = int(round_internal_node)

						print("external path")
						round_external_path = input() 
						round_external_path = int(round_external_path)					

						results[n_sim]['RL']['cost'] = []
						total_cost, results[n_sim]['RL']['cost'],tot_time = learning_function_(results[n_sim]['RL']['cost'], round_external_node, round_external_path, round_internal_node,all_adjacency_matrices,all_isps,virtual_graph,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix,min_cutoff,max_cutoff, num_of_possible_actions_internal_node, num_of_possible_actions_internal_path, num_of_possible_actions_external_node, num_of_possible_actions_external_path, n_epochs,eps_external_node,eps_external_path,eps_internal_node,eps_internal_path,learning_rate_internal_node, learning_rate_external_node,learning_rate_internal_path,learning_rate_external_path,discount_factor_internal_node, discount_factor_external_node, discount_factor_internal_path, discount_factor_external_path,bandwidth_requests_between_virtual_nodes,cost_per_link_matrix,cost_per_virtual_node,demand_per_virtual_node,dict_of_capacities_of_physical_nodes,infinity_value,capacity_of_all_links)
			
					else:
						break
				"""

				#results[n_sim]['RL']['time'] = tot_time
				
				#results[n_sim]['RL']['cost'].append[total_cost]
	np.save('results_simulations',results)