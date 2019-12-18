import numpy as np
import copy 
import networkx as nx 
from matplotlib import pyplot as plt 

def get_initial_states_and_Q(all_adjacency_matrices,all_isps,virtual_graph,num_of_possible_actions,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix):

	states = {}
	Q_dict = {}
	all_mapping_dictionaries = {} 

	N_ISPs = len(list(all_adjacency_matrices.keys())) -3 
	number_of_virtual_nodes = all_adjacency_matrices['virtual_graph'].shape[0]

	states['orchestrator'] = {}
	Q_dict['orchestrator'] = {}

	for vf_id_number in range(number_of_virtual_nodes):
		vf_id = 'vf_' + str(vf_id_number)
		states['orchestrator'][vf_id] = np.zeros(N_ISPs)
		curr_state = np.random.choice(np.arange(len(states['orchestrator'][vf_id]))) #np.argmax(states['orchestrator'][vf_id])
		states['orchestrator'][vf_id][curr_state] = 1
		Q_dict['orchestrator'][vf_id] = np.zeros([N_ISPs,num_of_possible_actions])

	links_for_isp = {}
	adj_global = all_adjacency_matrices['multidomain']
	tot_num_of_phy_nodes = adj_global.shape[0]

	adj_matrix_peering = all_adjacency_matrices['peering_network']
	for isp_id in range(N_ISPs):
		print(isp_id)
		isp = 'ISP_' + str(isp_id)
		
		adj_matrix_curr_isp = all_adjacency_matrices[isp_id]
		G = nx.from_numpy_matrix(adj_matrix_curr_isp)

		num_physical_nodes_for_curr_isp = adj_matrix_curr_isp.shape[0]

		peering_nodes_of_curr_isp = []
		peering_links_of_curr_isp = []

		for peering_p1 in range(adj_matrix_peering.shape[0]):
			for peering_p2 in range(adj_matrix_peering.shape[0]):

				if adj_matrix_peering[peering_p1,peering_p2] > 0:

					global_peering_p1 = dict_local_to_peering[peering_p1]
					global_peering_p2 = dict_local_to_peering[peering_p2]

					owner_isp_p1 = info_per_physical_node[global_peering_p1]['ISP']
					owner_isp_p1 = 'ISP_' + str(owner_isp_p1)

					owner_isp_p2 = info_per_physical_node[global_peering_p2]['ISP']
					owner_isp_p2 = 'ISP_' + str(owner_isp_p2)

					if owner_isp_p1 == isp:
						peering_nodes_of_curr_isp.append(global_peering_p1)
						peering_links_of_curr_isp.append([global_peering_p1,global_peering_p2])

		Q_dict[isp] = {}
		states[isp] = {} 
		all_mapping_dictionaries[isp] = {}

		tot_num_of_phy_nodes = feasibility_matrix.shape[0]
		for vf1 in range(number_of_virtual_nodes):
			
			suitable_phy_nodes_for_vf1 = []	
			dict_mapping_between_row_id_and_global_phy_node = {} #key: vf_id, value: global identifier of the node 

			vf_id1 = 'vf_' + str(vf1)
			id_row = 0
			for phy_node_id in range(tot_num_of_phy_nodes):
				owner_isp = info_per_physical_node[phy_node_id]['ISP']
				if owner_isp == isp_id:
					if feasibility_matrix[phy_node_id,vf1] < 2: #i.e., feasibile association
						suitable_phy_nodes_for_vf1.append(phy_node_id)
						dict_mapping_between_row_id_and_global_phy_node[id_row] = phy_node_id
						id_row += 1 

			suitable_phy_nodes_for_vf1 = list(set(suitable_phy_nodes_for_vf1))
			number_of_suitable_phy_nodes = len(suitable_phy_nodes_for_vf1)

			if number_of_suitable_phy_nodes > 0:				
				state_vector = np.zeros(number_of_suitable_phy_nodes)
				curr_state = np.random.choice(np.arange(number_of_suitable_phy_nodes))
				state_vector[curr_state] = 1 
				Q_matrix = np.zeros([number_of_suitable_phy_nodes,num_of_possible_actions])
			else:
				state_vector = [] 
				Q_matrix = []

			Q_dict[isp][vf_id1] = Q_matrix
			states[isp][vf_id1] = state_vector
			all_mapping_dictionaries[isp][vf_id1] = dict_mapping_between_row_id_and_global_phy_node

		for vf1 in range(number_of_virtual_nodes):
			print(vf1)
			for vf2 in range(number_of_virtual_nodes):
				print(vf2)
				vf_id1 = 'vf_' + str(vf1)
				vf_id2 = 'vf_' + str(vf2)

				if vf1 != vf2:

					curr_virtual_path = vf_id1 + '_' + vf_id2 
					Q_dict[isp][curr_virtual_path] = {}
					states[isp][curr_virtual_path] = {} 
					all_mapping_dictionaries[isp][curr_virtual_path] = {}

					suitable_path_when_vf1_and_vf2_are_in_isp = []
					suitable_path_when_only_vf1_is_in_isp = []
					suitable_path_when_only_vf2_is_in_isp = []
					suitable_path_when_neither_vf1_nor_vf2_are_in_isp = []
					
					dict_mapping_between_row_id_and_path_when_vf1_and_vf2_are_in_isp = {}
					row_id_when_vf1_and_vf2_are_in_isp = 0

					dict_mapping_between_row_id_and_path_when_only_vf1_is_in_isp = {}
					row_id_when_only_vf1_is_in_isp = 0

					dict_mapping_between_row_id_and_path_when_only_vf2_is_in_isp = {}
					row_id_when_only_vf2_is_in_isp = 0

					dict_mapping_between_row_id_and_path_when_neither_vf1_nor_vf2_are_in_isp = {}
					row_id_when_neither_vf1_nor_vf2_are_in_isp = 0

					for node_u in range(num_physical_nodes_for_curr_isp):
						
						global_phy_u = dict_local_to_global_node[isp + '_node_' + str(node_u)]
						f1 = feasibility_matrix[global_phy_u,vf1]
						f1_vf2 = feasibility_matrix[global_phy_u,vf2]
						
						for node_v in range(num_physical_nodes_for_curr_isp):

							global_phy_v = dict_local_to_global_node[isp + '_node_' + str(node_v)]						
							f2 = feasibility_matrix[global_phy_v,vf2]

							#case 1: both vf1 and vf2 are assigned to the current ISP 
							if f1 < 2 and f2 < 2:

								all_paths_between_u_and_v = []
								if node_u != node_v:
									for cutoff in range(2,5):
										curr_path_between_u_and_v = list(nx.all_simple_paths(G, node_u, node_v, cutoff=cutoff))
										for curr_path in curr_path_between_u_and_v:
											all_paths_between_u_and_v.append(curr_path)
										if len(all_paths_between_u_and_v) > 1:
											break 

								if node_u == node_v:
									all_paths_between_u_and_v.append([node_u,node_u])

								for curr_path_ in all_paths_between_u_and_v:

									global_curr_path_ = []
									for node_in_path in curr_path_:
										global_curr_path_.append(dict_local_to_global_node[isp + '_node_' + str(node_in_path)])

									suitable_path_when_vf1_and_vf2_are_in_isp.append(global_curr_path_)

									curr_path = ''
									for el in global_curr_path_:
										curr_path += str(el) + '_'
									curr_path = curr_path[:-1]

									dict_mapping_between_row_id_and_path_when_vf1_and_vf2_are_in_isp[row_id_when_vf1_and_vf2_are_in_isp] = curr_path
									row_id_when_vf1_and_vf2_are_in_isp += 1 
	
						#case 2: only vf1 is assigned to the current ISP 
						if f1 < 2:
							all_paths_between_u_and_peerings = []
							for peering_p in peering_nodes_of_curr_isp:
								local_peering_p = info_per_physical_node[peering_p]['single_domain_node_identifier']

								if node_u != local_peering_p:

									for cutoff in range(2,5):
										curr_path_between_u_local_peering_p = list(nx.all_simple_paths(G, node_u, local_peering_p, cutoff=cutoff))
										for curr_path in curr_path_between_u_local_peering_p:
											all_paths_between_u_and_peerings.append(curr_path)
										if len(all_paths_between_u_and_peerings) > 3:
											break 

								if node_u == local_peering_p:
									all_paths_between_u_and_peerings.append([node_u,node_u])

								for curr_path_ in all_paths_between_u_and_peerings:

									global_curr_path_ = []
									for node_in_path in curr_path_:
										global_curr_path_.append(dict_local_to_global_node[isp + '_node_' + str(node_in_path)])

									considered_peering = global_curr_path_[-1]
									for peering_link in peering_links_of_curr_isp:
										if peering_link[0] == considered_peering:

											global_curr_path_1 = copy.copy(global_curr_path_)
											global_curr_path_1.append(peering_link[1])

											suitable_path_when_only_vf1_is_in_isp.append(global_curr_path_1)

											curr_path = ''
											for el in global_curr_path_1:
												curr_path += str(el) + '_'
											curr_path = curr_path[:-1]

											dict_mapping_between_row_id_and_path_when_only_vf1_is_in_isp[row_id_when_only_vf1_is_in_isp] = curr_path
											row_id_when_only_vf1_is_in_isp += 1 

						#case 3: only vf2 is assigned to the current ISP 
						if f1_vf2 < 2:
							all_paths_between_u_and_peerings = []
							for peering_p in peering_nodes_of_curr_isp:
								local_peering_p = info_per_physical_node[peering_p]['single_domain_node_identifier']

								if node_u != local_peering_p:

									for cutoff in range(2,5):
										curr_path_between_u_local_peering_p = list(nx.all_simple_paths(G, local_peering_p, node_u, cutoff=cutoff))
										for curr_path in curr_path_between_u_local_peering_p:
											all_paths_between_u_and_peerings.append(curr_path)
										if len(all_paths_between_u_and_peerings) > 3:
											break 

								if node_u == local_peering_p:
									all_paths_between_u_and_peerings.append([node_u,node_u])

								for curr_path_ in all_paths_between_u_and_peerings:

									global_curr_path_ = []
									for node_in_path in curr_path_:
										global_curr_path_.append(dict_local_to_global_node[isp + '_node_' + str(node_in_path)])

									suitable_path_when_only_vf2_is_in_isp.append(global_curr_path_)

									curr_path = ''
									for el in global_curr_path_:
										curr_path += str(el) + '_'
									curr_path = curr_path[:-1]

									dict_mapping_between_row_id_and_path_when_only_vf2_is_in_isp[row_id_when_only_vf2_is_in_isp] = curr_path
									row_id_when_only_vf2_is_in_isp += 1 

					
					all_paths_between_peering1_and_peering2 = [] 
					for peering_p1 in peering_nodes_of_curr_isp:
						for peering_p2 in peering_nodes_of_curr_isp:
							local_peering_p1 = info_per_physical_node[peering_p1]['single_domain_node_identifier']
							local_peering_p2 = info_per_physical_node[peering_p2]['single_domain_node_identifier']

							if local_peering_p1 != local_peering_p2:
								for cutoff in range(2,5):
									curr_path_between_local_peering1_and_local_peering2 = list(nx.all_simple_paths(G, local_peering_p1, local_peering_p2, cutoff=cutoff))
									for curr_path in curr_path_between_local_peering1_and_local_peering2:
										all_paths_between_peering1_and_peering2.append(curr_path)
									if len(all_paths_between_peering1_and_peering2) > 5:
										break 

							if local_peering_p1 == local_peering_p2:
								all_paths_between_peering1_and_peering2.append([local_peering_p1,local_peering_p2])

							for curr_path_ in all_paths_between_peering1_and_peering2:

								global_curr_path_ = []
								for node_in_path in curr_path_:
									global_curr_path_.append(dict_local_to_global_node[isp + '_node_' + str(node_in_path)])

								considered_peering = global_curr_path_[-1]
								for peering_link in peering_links_of_curr_isp:
									if peering_link[0] == considered_peering:

										global_curr_path_1 = copy.copy(global_curr_path_)
										global_curr_path_1.append(peering_link[1])

										suitable_path_when_neither_vf1_nor_vf2_are_in_isp.append(global_curr_path_1)

										curr_path = ''
										for el in global_curr_path_:
											curr_path += str(el) + '_'
										curr_path = curr_path[:-1]

										dict_mapping_between_row_id_and_path_when_neither_vf1_nor_vf2_are_in_isp[row_id_when_neither_vf1_nor_vf2_are_in_isp] = curr_path
										row_id_when_neither_vf1_nor_vf2_are_in_isp += 1

					# define Q/states for case 1 
					number_of_paths = len(list(dict_mapping_between_row_id_and_path_when_vf1_and_vf2_are_in_isp.keys()))

					dict_mapping_between_row_id_and_path_when_vf1_and_vf2_are_in_isp_per_pair_u_v = {}
					num_paths_per_pair_uv = {}
					for node_u in range(num_physical_nodes_for_curr_isp):
						global_phy_u = dict_local_to_global_node[isp + '_node_' + str(node_u)]
						for node_v in range(num_physical_nodes_for_curr_isp):
							global_phy_v = dict_local_to_global_node[isp + '_node_' + str(node_v)]

							num_paths_per_pair_uv[str(global_phy_u) + '_' + str(global_phy_v)] = 0

							row_id_uv = 0 
							for row_id__ in dict_mapping_between_row_id_and_path_when_vf1_and_vf2_are_in_isp:
								path___ = dict_mapping_between_row_id_and_path_when_vf1_and_vf2_are_in_isp[row_id__]
								pp = path___.split('_')
								uuu = int(pp[0])
								vvv = int(pp[-1])

								if uuu == global_phy_u and vvv == global_phy_v:
									num_paths_per_pair_uv[str(global_phy_u) + '_' + str(global_phy_v)] += 1 
									dict_mapping_between_row_id_and_path_when_vf1_and_vf2_are_in_isp_per_pair_u_v[row_id_uv] = path___	
									row_id_uv += 1

							number_of_paths = 

							if number_of_paths > 0:				
								state_vector = np.zeros(number_of_paths)
								curr_state = np.random.choice(np.arange(number_of_paths))
								state_vector[curr_state] = 1 
								Q_matrix = np.zeros([number_of_paths,num_of_possible_actions])
							else:
								state_vector = [] 
								Q_matrix = []

					Q_dict[isp][curr_virtual_path]['case_1'] = Q_matrix
					states[isp][curr_virtual_path]['case_1'] = state_vector 
					all_mapping_dictionaries[isp][curr_virtual_path]['case_1'] = dict_mapping_between_row_id_and_path_when_vf1_and_vf2_are_in_isp

					# define Q/states for case 2
					number_of_paths = len(list(dict_mapping_between_row_id_and_path_when_only_vf1_is_in_isp.keys()))
					
					if number_of_paths > 0:				
						state_vector = np.zeros(number_of_paths)
						curr_state = np.random.choice(np.arange(number_of_paths))
						state_vector[curr_state] = 1 
						Q_matrix = np.zeros([number_of_paths,num_of_possible_actions])
					else:
						state_vector = [] 
						Q_matrix = []

					Q_dict[isp][curr_virtual_path]['case_2'] = Q_matrix
					states[isp][curr_virtual_path]['case_2'] = state_vector 
					all_mapping_dictionaries[isp][curr_virtual_path]['case_2'] = dict_mapping_between_row_id_and_path_when_only_vf1_is_in_isp

					# define Q/states for case 3
					number_of_paths = len(list(dict_mapping_between_row_id_and_path_when_only_vf2_is_in_isp.keys()))
					
					if number_of_paths > 0:				
						state_vector = np.zeros(number_of_paths)
						curr_state = np.random.choice(np.arange(number_of_paths))
						state_vector[curr_state] = 1 
						Q_matrix = np.zeros([number_of_paths,num_of_possible_actions])
					else:
						state_vector = [] 
						Q_matrix = []

					Q_dict[isp][curr_virtual_path]['case_3'] = Q_matrix
					states[isp][curr_virtual_path]['case_3'] = state_vector 
					all_mapping_dictionaries[isp][curr_virtual_path]['case_3'] = dict_mapping_between_row_id_and_path_when_only_vf2_is_in_isp

					# define Q/states for case 4
					number_of_paths = len(list(dict_mapping_between_row_id_and_path_when_neither_vf1_nor_vf2_are_in_isp.keys()))
												
					state_vector = np.zeros(number_of_paths+1)
					curr_state = np.random.choice(np.arange(number_of_paths))
					state_vector[curr_state] = 1 
					Q_matrix = np.zeros([number_of_paths+1,num_of_possible_actions])

					last_id = sorted(list(dict_mapping_between_row_id_and_path_when_neither_vf1_nor_vf2_are_in_isp.keys()))[-1]
					dict_mapping_between_row_id_and_path_when_neither_vf1_nor_vf2_are_in_isp[last_id+1] = 'no_path'

					Q_dict[isp][curr_virtual_path]['case_4'] = Q_matrix
					states[isp][curr_virtual_path]['case_4'] = state_vector 
					all_mapping_dictionaries[isp][curr_virtual_path]['case_4'] = dict_mapping_between_row_id_and_path_when_neither_vf1_nor_vf2_are_in_isp

	return states, Q_dict, all_mapping_dictionaries

def go_to_the_next_state_for_node(current_Q,curr_state,eps):

	actions_of_curr_state = current_Q[curr_state]
	maximum_state = current_Q.shape[0] - 1

	rv = np.random.binomial(1,eps)
	if rv > 0: #i.e., exploitation
		curr_action = np.argmax(actions_of_curr_state)
	else: #i.e., exploration 
		curr_action = np.random.choice(np.arange(len(actions_of_curr_state)))

	if curr_action == 0: # i.e., stay in the same place 
		new_state = curr_state
		#print("summing 0:",curr_state,new_state,'over a max state of ',maximum_state)
	else:
		if curr_action%2 == 0:

			new_state = curr_state + curr_action 
			new_state = min(new_state,maximum_state)			

			#print("summing",curr_action,":",curr_state,new_state,'over a max state of ',maximum_state)

		else:

			new_state = curr_state - curr_action 
			new_state = max(new_state,0)

			#print("substracting",curr_action,":",curr_state,new_state,'over a max state of ',maximum_state)

	return new_state, curr_action

def go_to_the_next_state_for_path(current_Q,curr_state,eps,feasibile_paths):

	# action 0 -> stay, action in [1,501] -> subtract, action in [502,1001] -> sum
	num_of_possible_actions = current_Q.shape[1]
	th_subtract = (num_of_possible_actions+3)/2 

	actions_of_curr_state = current_Q[curr_state]
	maximum_state = current_Q.shape[0] - 1

	all_possible_actions = [] 
	for path in feasibile_paths:
		if path == curr_state:
			all_possible_actions.append(0)
		elif path > curr_state:
			all_possible_actions.append(np.abs(curr_state - path)*2)
		elif path < curr_state:
			all_possible_actions.append(np.abs(curr_state - path)*2 - 1)

	all_possible_actions = [int(el) for el in all_possible_actions]
	#print(all_possible_actions)
	#print(num_of_possible_actions)
	#print(feasibile_paths,curr_state)

	values_possible_actions = [current_Q[curr_state][el] for el in all_possible_actions]

	rv = np.random.binomial(1,eps)
	if rv > 0: #i.e., exploitation
		curr_action = np.argmax(values_possible_actions)
	else: #i.e., exploration 
		curr_action = np.random.choice(len(values_possible_actions))
	
	curr_action = all_possible_actions[curr_action]

	if curr_action == 0:
		new_state = curr_state 
	elif curr_action%2 == 0:
		new_state = curr_state + curr_action
		new_state = min(new_state,int(np.max(feasibile_paths)))
	elif curr_action %2 == 1:
		new_state = curr_state - curr_action 
		new_state = max(new_state,int(np.min(feasibile_paths)))


	return new_state, curr_action

def go_to_the_next_state_for_path_(current_Q,curr_state,eps):

	actions_of_curr_state = current_Q[curr_state]
	maximum_state = current_Q.shape[0] - 1

	rv = np.random.binomial(1,eps)
	if rv > 0: #i.e., exploitation
		curr_action = np.argmax(actions_of_curr_state)
	else: #i.e., exploration 
		curr_action = np.random.choice(np.arange(len(actions_of_curr_state)))

	if curr_action == 0: # i.e., stay in the same place 
		new_state = curr_state
		#print("summing 0:",curr_state,new_state,'over a max state of ',maximum_state)
	else:
		if curr_action%2 == 0:
			new_state = curr_state + curr_action 
			new_state = min(new_state,maximum_state)			
			#print("summing",curr_action,":",curr_state,new_state,'over a max state of ',maximum_state)
		else:
			new_state = curr_state - curr_action 
			new_state = max(new_state,0)
			#print("substracting",curr_action,":",curr_state,new_state,'over a max state of ',maximum_state)

	return new_state, curr_action

def update_Qs_and_states(Q_dict,current_states,all_adjacency_matrices,overall_mapping_dict,eps,infinity_value,dict_of_capacities_of_physical_nodes,demand_per_virtual_node,dict_local_to_global_node,bandwidth_requests_between_virtual_nodes,capacity_of_all_links,dict_local_to_peering,learning_rate,discount_factor,iteration,feasibility_matrix,global_position_of_nodes,info_per_physical_node,cost_per_link_matrix,cost_per_vnf):

	where_is_infeasibility = []

	old_Q_dict = Q_dict 
	old_current_states = current_states

	#print(all_adjacency_matrices.keys())
	number_of_isps = len(all_adjacency_matrices.keys()) - 3
	all_isps_ids = ['ISP_' + str(el) for el in range(number_of_isps)]

	# getting the vfs for each isp according to the current state of the orchestrator
	vfs_for_isp = {}
	for isp_id in all_isps_ids:
		vfs_for_isp[isp_id] = []
		for vf_id in Q_dict['orchestrator']:
			state_vf_id = current_states['orchestrator'][vf_id]
			assigned_isp = np.where(state_vf_id > 0)[0][0]
			assigned_isp = 'ISP_' + str(assigned_isp)

			if assigned_isp == isp_id:
				vfs_for_isp[isp_id].append(vf_id)

	number_of_virtual_nodes = all_adjacency_matrices['virtual_graph'].shape[0]
	# updating Qs and states for all the isps 
	all_virtual_paths_managed_per_isps = {}
	current_states_actions_next_state_for_virtual_path_per_isp = {}

	cost_per_vpath_per_isp = {}
	cost_per_vf_per_isp = {}

	phy_nodes_per_vfs_for_all_isps = {}
	for isp_id in all_isps_ids:

		phy_nodes_per_vfs_for_all_isps[isp_id] = {}

		cost_per_vpath_per_isp[isp_id] = {}
		cost_per_vf_per_isp[isp_id] = {}

		vfs_assigned_to_this_isp = vfs_for_isp[isp_id]
		current_states_actions_next_state_for_virtual_path = {}
		current_states_actions_next_state_for_virtual_node = {}

		for vf1 in range(number_of_virtual_nodes):
			vf_id1 = 'vf_' + str(vf1)
			if vf_id1 in vfs_assigned_to_this_isp:
				
				curr_Q = Q_dict[isp_id][vf_id1]
				curr_state = current_states[isp_id][vf_id1]

				if len(curr_state) < 1:
					#cost_per_vpath_per_isp[isp_id][curr_virtual_path] += infinity_value
					if vf_id1 not in cost_per_vf_per_isp[isp_id]:
						cost_per_vf_per_isp[isp_id][vf_id1] = 0
					cost_per_vf_per_isp[isp_id][vf_id1] += infinity_value 
					where_is_infeasibility.append('infeasible_node')
				else:
					curr_state = np.where(curr_state > 0)[0][0]
					new_state, curr_action = go_to_the_next_state_for_node(curr_Q ,curr_state,eps)

					current_states_actions_next_state_for_virtual_node[vf_id1] = {}
					current_states_actions_next_state_for_virtual_node[vf_id1]['curr_state'] = curr_state 
					current_states_actions_next_state_for_virtual_node[vf_id1]['curr_action'] = curr_action 
					current_states_actions_next_state_for_virtual_node[vf_id1]['next_state'] = new_state				

		all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path = {}
		for vf1 in range(number_of_virtual_nodes):
			for vf2 in range(number_of_virtual_nodes):
				vf_id1 = 'vf_' + str(vf1)
				vf_id2 = 'vf_' + str(vf2) 

				curr_virtual_path = vf_id1 + '_' + vf_id2
				if vf1 != vf2:

					if curr_virtual_path not in cost_per_vpath_per_isp[isp_id]:
						cost_per_vpath_per_isp[isp_id][curr_virtual_path] = 0
					if vf_id1 not in cost_per_vf_per_isp[isp_id]:
						cost_per_vf_per_isp[isp_id][vf_id1] = 0
					if vf_id2 not in cost_per_vf_per_isp[isp_id]:
						cost_per_vf_per_isp[isp_id][vf_id2] = 0

					current_states_actions_next_state_for_virtual_path[curr_virtual_path] = {}
					current_states_actions_next_state_for_virtual_path[curr_virtual_path]['curr_state'] = 'no_curr_state'
					current_states_actions_next_state_for_virtual_path[curr_virtual_path]['curr_action'] = 'no_action' 
					current_states_actions_next_state_for_virtual_path[curr_virtual_path]['next_state'] = 'no_next_state'

					if vf_id1 in vfs_assigned_to_this_isp and vf_id2 in vfs_assigned_to_this_isp:
						# case 1
						curr_Q = Q_dict[isp_id][curr_virtual_path]['case_1']
						curr_state = current_states[isp_id][curr_virtual_path]['case_1']

						#print(current_states_actions_next_state_for_virtual_node)

						"""
						new_state_for_vf1 = current_states_actions_next_state_for_virtual_node[vf_id1]['next_state']
						new_phy_node_for_vf1 = overall_mapping_dict[isp_id][vf_id1][new_state_for_vf1]
						new_state_for_vf2 = current_states_actions_next_state_for_virtual_node[vf_id2]['next_state']
						new_phy_node_for_vf2 = overall_mapping_dict[isp_id][vf_id2][new_state_for_vf2]

						print(new_phy_node_for_vf1,new_state_for_vf2)
						"""

						if len(curr_state) < 1:
							cost_per_vf_per_isp[isp_id][vf_id1] += infinity_value 
							cost_per_vf_per_isp[isp_id][vf_id2] += infinity_value 
							where_is_infeasibility.append('infeasible_node')
							where_is_infeasibility.append('infeasible_node')
						else:

							xx = current_states_actions_next_state_for_virtual_node[vf_id1]['next_state'] 
							phy_node_for_vf1 = overall_mapping_dict[isp_id][vf_id1][xx]

							xx = current_states_actions_next_state_for_virtual_node[vf_id2]['next_state'] 
							phy_node_for_vf2 = overall_mapping_dict[isp_id][vf_id2][xx]

							feasibile_paths = []
							for iii in range(curr_Q.shape[0]):
								#print(overall_mapping_dict[isp_id][curr_virtual_path]['case_1'])
								c_path = overall_mapping_dict[isp_id][curr_virtual_path]['case_1'][iii].split('_')
								uuu_node = int(c_path[0])
								vvv_node = int(c_path[-1])

								if uuu_node == phy_node_for_vf1 and vvv_node == phy_node_for_vf2:
									#print(c_path,phy_node_for_vf1,phy_node_for_vf2)
									feasibile_paths.append(iii)

							curr_state = np.where(curr_state > 0)[0][0]
							new_state, curr_action = go_to_the_next_state_for_path(curr_Q ,curr_state,eps,feasibile_paths)

							current_states_actions_next_state_for_virtual_path[curr_virtual_path]['curr_state'] = curr_state 
							current_states_actions_next_state_for_virtual_path[curr_virtual_path]['curr_action'] = curr_action 
							current_states_actions_next_state_for_virtual_path[curr_virtual_path]['next_state'] = new_state

							physical_path___ = overall_mapping_dict[isp_id][curr_virtual_path]['case_1'][new_state]
							physical_path___ = physical_path___.split('_')
							node_uuu = int(physical_path___[0])
							node_vvv = int(physical_path___[-1])

							if vf_id1 not in phy_nodes_per_vfs_for_all_isps[isp_id]:
								phy_nodes_per_vfs_for_all_isps[isp_id][vf_id1] = []

							phy_nodes_per_vfs_for_all_isps[isp_id][vf_id1].append(node_uuu)

							if vf_id2 not in phy_nodes_per_vfs_for_all_isps[isp_id]:
								phy_nodes_per_vfs_for_all_isps[isp_id][vf_id2] = []

							phy_nodes_per_vfs_for_all_isps[isp_id][vf_id2].append(node_vvv)

							#current_states[isp_id][curr_virtual_path]['case_1'][:] = 0
							#current_states[isp_id][curr_virtual_path]['case_1'][new_state] = 1

							all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path[curr_virtual_path] = {}
							all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path[curr_virtual_path]['path'] = new_state
							all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path[curr_virtual_path]['case'] = 1

					elif vf_id1 in vfs_assigned_to_this_isp and vf_id2 not in vfs_assigned_to_this_isp:
						# case 2

						if vf_id1 not in cost_per_vf_per_isp[isp_id]:
							cost_per_vf_per_isp[isp_id][vf_id1] = 0

						curr_Q = Q_dict[isp_id][curr_virtual_path]['case_2']
						curr_state = current_states[isp_id][curr_virtual_path]['case_2']

						if len(curr_state) < 1:
							#cost_per_vpath_per_isp[isp_id][curr_virtual_path] += infinity_value
							cost_per_vf_per_isp[isp_id][vf_id1] += infinity_value 
							where_is_infeasibility.append('infeasible_node')
						else:

							xx = current_states_actions_next_state_for_virtual_node[vf_id1]['next_state'] 
							phy_node_for_vf1 = overall_mapping_dict[isp_id][vf_id1][xx]

							feasibile_paths = []
							for iii in range(curr_Q.shape[0]):
								#print(overall_mapping_dict[isp_id][curr_virtual_path]['case_1'])
								c_path = overall_mapping_dict[isp_id][curr_virtual_path]['case_2'][iii].split('_')
								uuu_node = int(c_path[0])

								if uuu_node == phy_node_for_vf1:
									#print(c_path,phy_node_for_vf1,phy_node_for_vf2)
									feasibile_paths.append(iii)

							curr_state = np.where(curr_state > 0)[0][0]
							new_state, curr_action = go_to_the_next_state_for_path(curr_Q ,curr_state,eps,feasibile_paths)

							current_states_actions_next_state_for_virtual_path[curr_virtual_path]['curr_state'] = curr_state 
							current_states_actions_next_state_for_virtual_path[curr_virtual_path]['curr_action'] = curr_action 
							current_states_actions_next_state_for_virtual_path[curr_virtual_path]['next_state'] = new_state

							physical_path___ = overall_mapping_dict[isp_id][curr_virtual_path]['case_2'][new_state]
							physical_path___ = physical_path___.split('_')
							node_uuu = int(physical_path___[0])

							if vf_id1 not in phy_nodes_per_vfs_for_all_isps[isp_id]:
								phy_nodes_per_vfs_for_all_isps[isp_id][vf_id1] = []

							phy_nodes_per_vfs_for_all_isps[isp_id][vf_id1].append(node_uuu)

							#current_states[isp_id][curr_virtual_path]['case_2'][:] = 0
							#current_states[isp_id][curr_virtual_path]['case_2'][new_state] = 1

							all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path[curr_virtual_path] = {}
							all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path[curr_virtual_path]['path'] = new_state
							all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path[curr_virtual_path]['case'] = 2

					if vf_id1 not in vfs_assigned_to_this_isp and vf_id2 in vfs_assigned_to_this_isp:						
						# case 3

						if vf_id2 not in cost_per_vf_per_isp[isp_id]:
							cost_per_vf_per_isp[isp_id][vf_id2] = 0

						curr_Q = Q_dict[isp_id][curr_virtual_path]['case_3']
						curr_state = current_states[isp_id][curr_virtual_path]['case_3']

						if len(curr_state) < 1:
							#cost_per_vpath_per_isp[isp_id][curr_virtual_path] += infinity_value
							cost_per_vf_per_isp[isp_id][vf_id2] += infinity_value 
							where_is_infeasibility.append('infeasible_node')
						else:

							xx = current_states_actions_next_state_for_virtual_node[vf_id2]['next_state'] 
							phy_node_for_vf2 = overall_mapping_dict[isp_id][vf_id2][xx]

							feasibile_paths = []
							for iii in range(curr_Q.shape[0]):
								#print(overall_mapping_dict[isp_id][curr_virtual_path]['case_1'])
								c_path = overall_mapping_dict[isp_id][curr_virtual_path]['case_3'][iii].split('_')
								vvv_node = int(c_path[-1])

								if vvv_node == phy_node_for_vf2:
									#print(c_path,phy_node_for_vf1,phy_node_for_vf2)
									feasibile_paths.append(iii)

							curr_state = np.where(curr_state > 0)[0][0]
							new_state, curr_action = go_to_the_next_state_for_path(curr_Q ,curr_state,eps,feasibile_paths)

							current_states_actions_next_state_for_virtual_path[curr_virtual_path]['curr_state'] = curr_state 
							current_states_actions_next_state_for_virtual_path[curr_virtual_path]['curr_action'] = curr_action 
							current_states_actions_next_state_for_virtual_path[curr_virtual_path]['next_state'] = new_state

							physical_path___ = overall_mapping_dict[isp_id][curr_virtual_path]['case_3'][new_state]
							physical_path___ = physical_path___.split('_')
							node_vvv = int(physical_path___[-1])

							if vf_id2 not in phy_nodes_per_vfs_for_all_isps[isp_id]:
								phy_nodes_per_vfs_for_all_isps[isp_id][vf_id2] = []
							phy_nodes_per_vfs_for_all_isps[isp_id][vf_id2].append(node_vvv)

							#current_states[isp_id][curr_virtual_path]['case_3'][:] = 0
							#current_states[isp_id][curr_virtual_path]['case_3'][new_state] = 1	

							all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path[curr_virtual_path] = {}
							all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path[curr_virtual_path]['path'] = new_state
							all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path[curr_virtual_path]['case'] = 3

					if vf_id1 not in vfs_assigned_to_this_isp and vf_id2 not in vfs_assigned_to_this_isp:
						
						# case 4
						curr_Q = Q_dict[isp_id][curr_virtual_path]['case_4']
						curr_state = current_states[isp_id][curr_virtual_path]['case_4']

						if len(curr_state) < 1:
							pass 
						else:
							curr_state = np.where(curr_state > 0)[0][0]
							new_state, curr_action = go_to_the_next_state_for_path_(curr_Q ,curr_state,eps)

							current_states_actions_next_state_for_virtual_path[curr_virtual_path]['curr_state'] = curr_state 
							current_states_actions_next_state_for_virtual_path[curr_virtual_path]['curr_action'] = curr_action 
							current_states_actions_next_state_for_virtual_path[curr_virtual_path]['next_state'] = new_state

							#current_states[isp_id][curr_virtual_path]['case_4'][:] = 0
							#current_states[isp_id][curr_virtual_path]['case_4'][new_state] = 1	

							link_passes_in_isp = 1
							path_corresponding_to_new_state = overall_mapping_dict[isp_id][curr_virtual_path]['case_4'][new_state]
							
							"""
							if path_corresponding_to_new_state == 'no_path':
								new_state = curr_Q.shape[0] - 1
							"""

							all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path[curr_virtual_path] = {}
							all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path[curr_virtual_path]['path'] = new_state
							all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path[curr_virtual_path]['case'] = 4

		all_virtual_paths_managed_per_isps[isp_id] = all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path

		current_states_actions_next_state_for_virtual_path_per_isp[isp_id] = current_states_actions_next_state_for_virtual_path

	feasibility_per_vfs = {}
	cases_per_isp = {}
	physical_nodes_traversed_for_each_virtual_path = {}
	embedding_cost_per_isp = {}

	for isp_id in all_virtual_paths_managed_per_isps:
		embedding_cost_per_isp[isp_id] = 0
		current_states_actions_next_state_for_virtual_path = current_states_actions_next_state_for_virtual_path_per_isp[isp_id]
		cases_per_isp[isp_id] = {}
		all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path = all_virtual_paths_managed_per_isps[isp_id]		
		
		# node consistency constraint: we have to ensure that there are no replicas of the same vfs 			
		phy_nodes_per_vf = phy_nodes_per_vfs_for_all_isps[isp_id] 

		for vf_iii in phy_nodes_per_vf:
			phy_nodes_per_vf[vf_iii] = list(set(phy_nodes_per_vf[vf_iii]))


		vfs_per_phy_nodes = {} 
		for vf_id__ in phy_nodes_per_vf:
			for phy_node_of_this_vf in phy_nodes_per_vf[vf_id__]:
				if phy_node_of_this_vf not in vfs_per_phy_nodes:
					vfs_per_phy_nodes[phy_node_of_this_vf] = []
				vfs_per_phy_nodes[phy_node_of_this_vf].append(vf_id__)

		virtual_path_per_phy_link = {}

		for curr_virtual_path in all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path:
			vf_i = 'vf_' + str(curr_virtual_path.split('_')[1])
			vf_j = 'vf_' + str(curr_virtual_path.split('_')[3])

			phy_path = all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path[curr_virtual_path]
			case_ = phy_path['case']
			phy_path_id = phy_path['path']

			if case_ == 1:
				dict_ = overall_mapping_dict[isp_id][curr_virtual_path]['case_1']
			elif case_ == 2:
				dict_ = overall_mapping_dict[isp_id][curr_virtual_path]['case_2']
			elif case_ == 3:
				dict_ = overall_mapping_dict[isp_id][curr_virtual_path]['case_3']
			elif case_ == 4:
				dict_ = overall_mapping_dict[isp_id][curr_virtual_path]['case_4']

			phy_path = dict_[phy_path_id]

			if phy_path != 'no_path':

				phy_path = phy_path.split('_')
				# now let's consider the links 
				for path_index in range(len(phy_path)-1):
					phy_link = str(phy_path[path_index]) + '_' + str(phy_path[path_index+1])
					if phy_link not in virtual_path_per_phy_link:
						virtual_path_per_phy_link[phy_link] = []

					virtual_path_per_phy_link[phy_link].append(curr_virtual_path)
		
		# check consistency
		for curr_virtual_path in all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path:
			vf_i = 'vf_' + str(curr_virtual_path.split('_')[1])
			vf_j = 'vf_' + str(curr_virtual_path.split('_')[3])

			case_ = all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path[curr_virtual_path]['case']
			phy_path_id = all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path[curr_virtual_path]['path']
			physical_path = overall_mapping_dict[isp_id][curr_virtual_path]['case_' + str(case_)][phy_path_id]

			if physical_path != 'no_path':
				if curr_virtual_path not in physical_nodes_traversed_for_each_virtual_path:
					physical_nodes_traversed_for_each_virtual_path[curr_virtual_path] = []

				phy_path = physical_path.split('_')
				#print(phy_path)
				physical_nodes_traversed_for_each_virtual_path[curr_virtual_path].append([int(phy_path[0]),int(phy_path[-1])])				

			if case_ == 1:
				if vf_i in phy_nodes_per_vf:
					if len(phy_nodes_per_vf[vf_i]) > 1:
						#cost_per_vpath_per_isp[isp_id][curr_virtual_path] += infinity_value				
						cost_per_vf_per_isp[isp_id][vf_i] += len(phy_nodes_per_vf[vf_i])*infinity_value
						#print('isp ',isp_id,vf_i,phy_nodes_per_vf[vf_i])
						for iii in range(len(phy_nodes_per_vf[vf_i])):
							where_is_infeasibility.append('not_single_phy')
					if len(phy_nodes_per_vf[vf_i]) < 1:
						#cost_per_vpath_per_isp[isp_id][curr_virtual_path] += infinity_value					
						cost_per_vf_per_isp[isp_id][vf_i] += infinity_value 
						where_is_infeasibility.append('no_phy')

				if vf_j in phy_nodes_per_vf:
					if len(phy_nodes_per_vf[vf_j]) > 1:
						#cost_per_vpath_per_isp[isp_id][curr_virtual_path] += infinity_value
						cost_per_vf_per_isp[isp_id][vf_j] += len(phy_nodes_per_vf[vf_j])*infinity_value
						#print('isp ',isp_id,vf_j,phy_nodes_per_vf[vf_j])
						for iii in range(len(phy_nodes_per_vf[vf_j])):
							where_is_infeasibility.append('not_single_phy')		

					if len(phy_nodes_per_vf[vf_j]) < 1:
						#cost_per_vpath_per_isp[isp_id][curr_virtual_path] += infinity_value
						cost_per_vf_per_isp[isp_id][vf_j] += infinity_value 
						where_is_infeasibility.append('no_phy')	

			elif case_ == 2:
				if vf_i in phy_nodes_per_vf:
					if len(phy_nodes_per_vf[vf_i]) > 1:
						#cost_per_vpath_per_isp[isp_id][curr_virtual_path] += infinity_value
						cost_per_vf_per_isp[isp_id][vf_i] += len(phy_nodes_per_vf[vf_i])*infinity_value
						#print('isp ',isp_id,vf_i,phy_nodes_per_vf[vf_i])
						for iii in range(len(phy_nodes_per_vf[vf_i])):
							where_is_infeasibility.append('not_single_phy')

					if len(phy_nodes_per_vf[vf_i]) < 1:
						#cost_per_vpath_per_isp[isp_id][curr_virtual_path] += infinity_value
						cost_per_vf_per_isp[isp_id][vf_i] += infinity_value
						where_is_infeasibility.append('no_phy') 

			elif case_ == 3:
				if vf_j in phy_nodes_per_vf:
					if len(phy_nodes_per_vf[vf_j]) > 1:
						cost_per_vpath_per_isp[isp_id][curr_virtual_path] += len(phy_nodes_per_vf[vf_j])*infinity_value
						#cost_per_vf_per_isp[isp_id][vf_j] += infinity_value 
						#print('isp ',isp_id,vf_j,phy_nodes_per_vf[vf_j])
						for iii in range(len(phy_nodes_per_vf[vf_j])):
							where_is_infeasibility.append('not_single_phy')
					if len(phy_nodes_per_vf[vf_j]) < 1:
						cost_per_vpath_per_isp[isp_id][curr_virtual_path] += infinity_value
						#cost_per_vf_per_isp[isp_id][vf_j] += infinity_value 
						where_is_infeasibility.append('no_phy')


		# check node capacity constraint  
		for phy_node in vfs_per_phy_nodes:
			vfs_per_phy_nodes[phy_node] = list(set(vfs_per_phy_nodes[phy_node]))		
			overall_cpu_capacity_of_node = dict_of_capacities_of_physical_nodes[phy_node]
			#print(overall_cpu_capacity_of_node)
			occupied_capacity = 0
			for vf in vfs_per_phy_nodes[phy_node]:
				vf_number_id = int(vf.split('_')[1])
				curr_cpu_demand = demand_per_virtual_node[vf_number_id]
				occupied_capacity += curr_cpu_demand

				"""
				if vf not in feasibility_per_vfs:
					feasibility_per_vfs[vf] = feasibility_matrix[phy_node,vf_number_id]
				if feasibility_per_vfs[vf] > 2:
					cost_per_vf_per_isp[isp_id][vf] += infinity_value
				"""
				cost_per_vf_per_isp[isp_id][vf] += demand_per_virtual_node[vf_number_id]

			if occupied_capacity > overall_cpu_capacity_of_node:
				for curr_virtual_path in cost_per_vpath_per_isp[isp_id]:
					for vfs in vfs_per_phy_nodes[phy_node]:
						if vfs in curr_virtual_path:
							cost_per_vf_per_isp[isp_id][vfs] += infinity_value
							where_is_infeasibility.append('node_capacity')
		# check link capacity constraint 
		all_peering_nodes = []
		for peering_node in range(all_adjacency_matrices['peering_network'].shape[0]):
			gl_p = dict_local_to_peering[peering_node]
			all_peering_nodes.append(gl_p)

		for phy_link in virtual_path_per_phy_link:

			n1,n2 = phy_link.split('_')
			n1 = int(n1)
			n2 = int(n2)
			
			if n1 != n2:
				total_bw_of_link = capacity_of_all_links[n1,n2]

				virtual_paths_on_curr_link = virtual_path_per_phy_link[phy_link]
				#print(virtual_paths_on_curr_link)
				occupied_bw_of_link = 0
				for virtual_path_on_curr_link in virtual_paths_on_curr_link:
					vf1 = int(virtual_path_on_curr_link.split('_')[1])
					vf2 = int(virtual_path_on_curr_link.split('_')[3])

					curr_bw_demand = bandwidth_requests_between_virtual_nodes[vf1,vf2]
					occupied_bw_of_link += curr_bw_demand

				if occupied_bw_of_link > total_bw_of_link:
					for virtual_path_passing_on_this_link in virtual_paths_on_curr_link:

						vf_id_x = 'vf_' + str(virtual_path_passing_on_this_link.split('_')[1])
						vf_id_y = 'vf_' + str(virtual_path_passing_on_this_link.split('_')[3])

						cost_per_vpath_per_isp[isp_id][virtual_path_passing_on_this_link] += infinity_value
					#where_is_infeasibility.append('link_capacity')
					print('capac',occupied_bw_of_link,total_bw_of_link)
					from time import sleep 
					sleep(10)

		# add the actual cost of the chosen embedding strategy
		cases = {}
		for curr_virtual_path in all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path:
			
			vf1 = int(curr_virtual_path.split('_')[1])
			vf2 = int(curr_virtual_path.split('_')[3])

			vf1_id = 'vf_' + str(vf1)
			vf2_id = 'vf_' + str(vf2)

			cpu_requirement_vf1 = demand_per_virtual_node[vf1]
			cpu_requirement_vf2 = demand_per_virtual_node[vf2]

			phy_path = all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path[curr_virtual_path]
			case_ = phy_path['case']
			phy_path_id = phy_path['path']

			if case_ == 1:				
				physical_path = overall_mapping_dict[isp_id][curr_virtual_path]['case_1'][phy_path_id]
			elif case_ == 2:
				physical_path = overall_mapping_dict[isp_id][curr_virtual_path]['case_2'][phy_path_id]
			elif case_ == 3:
				physical_path = overall_mapping_dict[isp_id][curr_virtual_path]['case_3'][phy_path_id]
			elif case_ == 4:
				physical_path = overall_mapping_dict[isp_id][curr_virtual_path]['case_4'][phy_path_id]

			#print(cost_per_vnf)
			if physical_path != 'no_path':
			
				physical_path__ = physical_path.split('_')
				node_uu = int(physical_path[0])
				node_vv = int(physical_path[-1])
				
				cost_per_vf_per_isp[isp_id][vf_id1] += cpu_requirement_vf1*cost_per_vnf[int(vf_id1.split('_')[-1])][node_uu]
				cost_per_vf_per_isp[isp_id][vf_id2] += cpu_requirement_vf2*cost_per_vnf[int(vf_id2.split('_')[-1])][node_vv]

				embedding_cost_per_isp[isp_id] += cpu_requirement_vf1*cost_per_vnf[int(vf_id1.split('_')[-1])][node_uu] + cpu_requirement_vf2*cost_per_vnf[int(vf_id2.split('_')[-1])][node_vv]

			nodes_in_the_path = physical_path.split('_')
			if nodes_in_the_path[0] != nodes_in_the_path[1]:
				N_link_in_the_path = len(nodes_in_the_path) - 1 
				#cost_per_vpath_per_isp[isp_id][curr_virtual_path] += N_link_in_the_path*bandwidth_requests_between_virtual_nodes[vf1,vf2]
				for index_node in range(N_link_in_the_path):
					curr_link_id = (nodes_in_the_path[index_node],nodes_in_the_path[index_node+1])
					if 'path' in curr_link_id:
						pass
					else:
						cost_per_this_link = bandwidth_requests_between_virtual_nodes[vf1,vf2]*cost_per_link_matrix[int(curr_link_id[0]),int(curr_link_id[1])]
						cost_per_vpath_per_isp[isp_id][curr_virtual_path] += cost_per_this_link
						embedding_cost_per_isp[isp_id] += cost_per_this_link

			#cost_for_this_isp += cosfeasibilt_per_virtual_path[curr_virtual_path]	
			cases[curr_virtual_path] = case_ 

		cases_per_isp[isp_id] = cases

	#np.save('physical_nodes_traversed_for_each_virtual_path',physical_nodes_traversed_for_each_virtual_path)

	# check inter-domain flow consistency
	for curr_virtual_path in physical_nodes_traversed_for_each_virtual_path:
		#print(physical_nodes_traversed_for_each_virtual_path[curr_virtual_path])
		edges = physical_nodes_traversed_for_each_virtual_path[curr_virtual_path]
		owner_isps = []
		for edge in edges:
			for node_ in edge:
				owner_isp = info_per_physical_node[node_]['ISP']
				owner_isps.append('ISP_' + str(owner_isp))

		g = nx.Graph()
		g.add_edges_from(edges)

		try:
		    cycle = nx.find_cycle(g)
		    cycle = set(map(frozenset, cycle))
		    edges = set(map(frozenset, edges))
		    if cycle == edges:
		        pass#print("The edges form a loop")
		    else: 
		    	#print("The edges don't form a loop")
		    	for isp_id in owner_isps:
		    		cost_per_vpath_per_isp[isp_id][curr_virtual_path] += infinity_value
		    		where_is_infeasibility.append('flow_consistency')

		except nx.exception.NetworkXNoCycle: 
		    pass#print("No cycle found")

	all_network = all_adjacency_matrices['multidomain']
	all_network_G = nx.from_numpy_matrix(all_network)

	edge_labels = {}
	reward_per_all_isps = {}

	for isp_id in all_virtual_paths_managed_per_isps:
		all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path = all_virtual_paths_managed_per_isps[isp_id]

		reward_per_this_isp = 0
		for curr_virtual_path in all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path:

			phy_path = all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path[curr_virtual_path]
			case_ = phy_path['case']

			ppp = curr_virtual_path.split('_')
			vf1 = 'vf_' + str(ppp[1])
			vf2 = 'vf_' + str(ppp[3])

			if case_ == 1:
				reward_per_this_isp += cost_per_vpath_per_isp[isp_id][curr_virtual_path] + cost_per_vf_per_isp[isp_id][vf1] + cost_per_vf_per_isp[isp_id][vf2]
				cost_per_vpath_per_isp[isp_id][curr_virtual_path] += cost_per_vf_per_isp[isp_id][vf1] + cost_per_vf_per_isp[isp_id][vf2]

			elif case_ == 2:
				reward_per_this_isp += cost_per_vpath_per_isp[isp_id][curr_virtual_path] + cost_per_vf_per_isp[isp_id][vf1]
				cost_per_vpath_per_isp[isp_id][curr_virtual_path] += cost_per_vf_per_isp[isp_id][vf1]
			elif case_ == 3:
				reward_per_this_isp += cost_per_vpath_per_isp[isp_id][curr_virtual_path] + cost_per_vf_per_isp[isp_id][vf2]
				cost_per_vpath_per_isp[isp_id][curr_virtual_path] += cost_per_vf_per_isp[isp_id][vf2]
			elif case_ == 4:
				reward_per_this_isp += cost_per_vpath_per_isp[isp_id][curr_virtual_path]

			#reward_per_this_isp += cost_per_vf_per_isp[isp_id][vf1] + cost_per_vf_per_isp[isp_id][vf2]
		#print(isp_id,cost_per_isps[isp_id])
		current_states_actions_next_state_for_virtual_path = current_states_actions_next_state_for_virtual_path_per_isp[isp_id]
		reward_per_this_isp = -reward_per_this_isp

		for curr_virtual_path in all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path:

			curr_state = current_states_actions_next_state_for_virtual_path[curr_virtual_path]['curr_state']  
			curr_action = current_states_actions_next_state_for_virtual_path[curr_virtual_path]['curr_action']  
			next_state = current_states_actions_next_state_for_virtual_path[curr_virtual_path]['next_state']

			phy_path = all_virtual_paths_managed_by_curr_isp_and_corresponding_physical_path[curr_virtual_path]
			case_ = phy_path['case']

			phy_path = overall_mapping_dict[isp_id][curr_virtual_path]['case_' + str(case_)][next_state]
			phy_nodes = phy_path.split('_')
			if 'path' in phy_nodes:
				pass 
			else:
				phy_links = []
				for ii in range(len(phy_nodes)-1):
					phy_links.append([int(phy_nodes[ii]),int(phy_nodes[ii+1])])

			rv = np.random.binomial(1,1)
			if rv > 0:
		
				curr_vfs = curr_virtual_path.split('_')
				vf1 = curr_vfs[0] + '_' + curr_vfs[1]
				vf2 = curr_vfs[2] + '_' + curr_vfs[3]

				discount_term = np.max(Q_dict[isp_id][curr_virtual_path]['case_' + str(case_)][next_state])
				
				#curr_cost = cost_per_vpath_per_isp[isp_id][curr_virtual_path]
				#reward = -curr_cost
		
				if case_ == 1:
					#reward += -(cost_per_vf_per_isp[isp_id][vf1] + cost_per_vf_per_isp[isp_id][vf2])
					#curr_cost += (cost_per_vf_per_isp[isp_id][vf1] + cost_per_vf_per_isp[isp_id][vf2])
					cost_per_vpath_per_isp[isp_id][curr_virtual_path] += (cost_per_vf_per_isp[isp_id][vf1] + cost_per_vf_per_isp[isp_id][vf2])
				elif case_ == 2:
					#reward += -(cost_per_vf_per_isp[isp_id][vf1])
					#curr_cost += (cost_per_vf_per_isp[isp_id][vf1])
					cost_per_vpath_per_isp[isp_id][curr_virtual_path] += cost_per_vf_per_isp[isp_id][vf1]
				elif case_ == 3:
					#reward += -(cost_per_vf_per_isp[isp_id][vf2])
					#curr_cost += (cost_per_vf_per_isp[isp_id][vf2])
					cost_per_vpath_per_isp[isp_id][curr_virtual_path] += cost_per_vf_per_isp[isp_id][vf2]

				curr_cost = cost_per_vpath_per_isp[isp_id][curr_virtual_path]
				if curr_cost < infinity_value:
					reward = 0
				else:
					reward = -curr_cost

				reward_per_all_isps[isp_id] = reward

				#cost_of_this_isp = cost_per_isps[isp_id] + additional_cost

				Q_dict[isp_id][curr_virtual_path]['case_' + str(case_)][curr_state,curr_action] = Q_dict[isp_id][curr_virtual_path]['case_' + str(case_)][curr_state,curr_action] + learning_rate*(reward + discount_factor*discount_term - old_Q_dict[isp_id][curr_virtual_path]['case_' + str(case_)][curr_state,curr_action])
				
				current_states[isp_id][curr_virtual_path]['case_'+str(case_)][:] = 0
				current_states[isp_id][curr_virtual_path]['case_'+str(case_)][next_state] = 1
	
	curr_Q_orchestrator = Q_dict['orchestrator']
	curr_state_orchestrator = current_states['orchestrator']

	overall_cost = 0
	for isp_id in cost_per_vpath_per_isp:
		for curr_virtual_path in cost_per_vpath_per_isp[isp_id]:
			overall_cost += cost_per_vpath_per_isp[isp_id][curr_virtual_path]
		for vf_id in cost_per_vf_per_isp[isp_id]:
			overall_cost += cost_per_vf_per_isp[isp_id][vf_id]

	
	if overall_cost > infinity_value:
		eps_overall = 0.7
		for vf_id1 in curr_Q_orchestrator:
			for vf_id2 in curr_Q_orchestrator:
				#overall_reward_ = overall_reward #cost_per_vfs[vf_id]

				if vf_id1 != vf_id2:
					curr_virtual_path = vf_id1 + '_' + vf_id2
					
					curr_Q_orchestrator_vf_id1 = curr_Q_orchestrator[vf_id1]
					curr_state_orchestrator_vf_id1 = curr_state_orchestrator[vf_id1]

					curr_state1 = np.where(curr_state_orchestrator_vf_id1 > 0)[0][0]
					next_state1, curr_action1 = go_to_the_next_state_for_path_(curr_Q_orchestrator_vf_id1,curr_state1,eps_overall)
					next_isp_in_charge_of_this_vf1 = 'ISP_' + str(next_state1)

					curr_Q_orchestrator_vf_id2 = curr_Q_orchestrator[vf_id2]
					curr_state_orchestrator_vf_id2 = curr_state_orchestrator[vf_id2]

					curr_state2 = np.where(curr_state_orchestrator_vf_id2 > 0)[0][0]
					next_state2, curr_action2 = go_to_the_next_state_for_path_(curr_Q_orchestrator_vf_id2,curr_state2,eps_overall)
					next_isp_in_charge_of_this_vf2 = 'ISP_' + str(next_state2)

					current_states['orchestrator'][vf_id1][:] = 0
					current_states['orchestrator'][vf_id1][next_state1] = 1 
					discount_term1 = np.max(Q_dict['orchestrator'][vf_id1][next_state1])
					Q_dict['orchestrator'][vf_id1][curr_state1,curr_action1] = Q_dict['orchestrator'][vf_id1][curr_state1,curr_action1] + learning_rate*(-overall_cost + discount_factor*discount_term1 - old_Q_dict['orchestrator'][vf_id1][curr_state1,curr_action1])

					current_states['orchestrator'][vf_id2][:] = 0
					current_states['orchestrator'][vf_id2][next_state2] = 1 
					discount_term2 = np.max(Q_dict['orchestrator'][vf_id2][next_state2])
					Q_dict['orchestrator'][vf_id2][curr_state2,curr_action2] = Q_dict['orchestrator'][vf_id2][curr_state2,curr_action2] + learning_rate*(-overall_cost + discount_factor*discount_term2 - old_Q_dict['orchestrator'][vf_id2][curr_state2,curr_action2])
	

	total_embedding_cost = 0
	for isp_id in embedding_cost_per_isp:
		total_embedding_cost += embedding_cost_per_isp[isp_id]

	for vf_id1 in curr_Q_orchestrator:
		for vf_id2 in curr_Q_orchestrator:
			#overall_reward_ = overall_reward #cost_per_vfs[vf_id]

			if vf_id1 != vf_id2:
				curr_virtual_path = vf_id1 + '_' + vf_id2
				
				curr_Q_orchestrator_vf_id1 = curr_Q_orchestrator[vf_id1]
				curr_state_orchestrator_vf_id1 = curr_state_orchestrator[vf_id1]

				curr_state1 = np.where(curr_state_orchestrator_vf_id1 > 0)[0][0]
				next_state1, curr_action1 = go_to_the_next_state_for_path_(curr_Q_orchestrator_vf_id1,curr_state1,eps)

				next_isp_in_charge_of_this_vf1 = 'ISP_' + str(next_state1)
				overall_reward_1 = -cost_per_vpath_per_isp[next_isp_in_charge_of_this_vf1][curr_virtual_path]

				curr_Q_orchestrator_vf_id2 = curr_Q_orchestrator[vf_id2]
				curr_state_orchestrator_vf_id2 = curr_state_orchestrator[vf_id2]

				curr_state2 = np.where(curr_state_orchestrator_vf_id2 > 0)[0][0]
				next_state2, curr_action2 = go_to_the_next_state_for_path_(curr_Q_orchestrator_vf_id2,curr_state2,eps)

				next_isp_in_charge_of_this_vf2 = 'ISP_' + str(next_state2)
				overall_reward_2 = -cost_per_vpath_per_isp[next_isp_in_charge_of_this_vf2][curr_virtual_path]
				
				update = 0
				update_1 = 0
				update_2 = 0 
				eps1 = eps
				eps2 = eps  
				if overall_reward_1 < -infinity_value:
					update_1 = 1 
					eps1 = 0.2
					overall_reward_1 = overall_reward_1
				else:
					update_1 = 1 
					eps1 = eps 
					overall_reward_1 = 10**3

				if overall_reward_2 < -infinity_value:
					update_2 = 1 
					eps2 = 0.2
					overall_reward_2 = overall_reward_2
				else:
					update_2 = 1 
					eps2 = eps		
					overall_reward_2 = 10**3

				rv = np.random.binomial(1,1)
				if rv > 0:
					if update_1 > 0: #iteration%1000 == 0 or update > 0: 

						curr_state1 = np.where(curr_state_orchestrator_vf_id1 > 0)[0][0]
						next_state1, curr_action1 = go_to_the_next_state_for_path_(curr_Q_orchestrator_vf_id1,curr_state1,eps1)

						next_isp_in_charge_of_this_vf1 = 'ISP_' + str(next_state1)
						overall_reward_1 = -cost_per_vpath_per_isp[next_isp_in_charge_of_this_vf1][curr_virtual_path]

						current_states['orchestrator'][vf_id1][:] = 0
						current_states['orchestrator'][vf_id1][next_state1] = 1 
						discount_term1 = np.max(Q_dict['orchestrator'][vf_id1][next_state1])
						Q_dict['orchestrator'][vf_id1][curr_state1,curr_action1] = Q_dict['orchestrator'][vf_id1][curr_state1,curr_action1] + learning_rate*(overall_reward_1 + discount_factor*discount_term1 - old_Q_dict['orchestrator'][vf_id1][curr_state1,curr_action1])

					if update_2 > 0: #iteration%1000 == 0 or update > 0: 

						curr_state2 = np.where(curr_state_orchestrator_vf_id2 > 0)[0][0]
						next_state2, curr_action2 = go_to_the_next_state_for_path_(curr_Q_orchestrator_vf_id2,curr_state2,eps2)

						next_isp_in_charge_of_this_vf2 = 'ISP_' + str(next_state2)
						overall_reward_2 = -cost_per_vpath_per_isp[next_isp_in_charge_of_this_vf2][curr_virtual_path]

						current_states['orchestrator'][vf_id2][:] = 0
						current_states['orchestrator'][vf_id2][next_state2] = 1 
						discount_term2 = np.max(Q_dict['orchestrator'][vf_id2][next_state2])
						Q_dict['orchestrator'][vf_id2][curr_state2,curr_action2] = Q_dict['orchestrator'][vf_id2][curr_state2,curr_action2] + learning_rate*(overall_reward_2 + discount_factor*discount_term2 - old_Q_dict['orchestrator'][vf_id2][curr_state2,curr_action2])

				#print(Q_dict['orchestrator'][vf_id][curr_state,curr_action],'aa')


	if overall_cost < 2000:
		print("overall cost ",overall_cost)
		print(cost_per_vf_per_isp)
		from time import sleep 
		sleep(20)

	from collections import Counter 
	xx = Counter(where_is_infeasibility).most_common()
	print(xx)

	return current_states, Q_dict, overall_mapping_dict, overall_cost, total_embedding_cost, iteration

def learning_function_(all_adjacency_matrices,all_isps,virtual_graph,dict_local_to_peering,info_per_physical_node,eps,bandwidth_requests_between_virtual_nodes,feasibility_matrix,demand_per_virtual_node,dict_local_to_global_node,alpha_cost,beta_cost,dict_of_capacities_of_physical_nodes,capacities_of_physical_links,learning_rate,discount_factor,infinity_value,n_epochs,global_position_of_nodes,cost_per_link_matrix,cost_per_vnf):
	
	num_of_possible_actions = 601

	current_states, Q_dict, overall_mapping_dict = get_initial_states_and_Q(all_adjacency_matrices,all_isps,virtual_graph,num_of_possible_actions,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix)
	np.save('current_states',current_states)
	np.save('q_dict',Q_dict)

	#from time import sleep 
	#print(current_states['ISP_0']['nodes'].keys())
	#sleep(10000)

	overall_cost_list = [] 
	embedding_cost_list = []
	for iteration in range(n_epochs):

		current_states, Q_dict, overall_mapping_dict, overall_cost, total_embedding_cost, iteration_ = update_Qs_and_states(Q_dict,current_states,all_adjacency_matrices,overall_mapping_dict,eps,infinity_value,dict_of_capacities_of_physical_nodes,demand_per_virtual_node,dict_local_to_global_node,bandwidth_requests_between_virtual_nodes,capacities_of_physical_links,dict_local_to_peering,learning_rate,discount_factor,iteration,feasibility_matrix,global_position_of_nodes,info_per_physical_node,cost_per_link_matrix,cost_per_vnf)
		overall_cost_list.append(overall_cost)#,total_embedding_cost)
		embedding_cost_list.append(total_embedding_cost)

		print(iteration,overall_cost,total_embedding_cost)

	from matplotlib import pyplot as plt 
	plt.plot(overall_cost_list,label='overall')
	plt.plot(embedding_cost_list,label='embedding')
	plt.legend()
	plt.show()



