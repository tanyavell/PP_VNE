import numpy as np
import copy 
import networkx as nx 
from matplotlib import pyplot as plt 
from collections import Counter
import time

def get_rounding(x):

	nearest_integer = int(round(x))
	lower_bound = int(np.floor(x))
	upper_bound = int(np.ceil(x))

	prob_teta = np.abs(nearest_integer - x)
	rv_teta = np.random.binomial(1,1 - prob_teta)

	if nearest_integer == lower_bound:
		if rv_teta > 0:
			y = lower_bound
		else:
			y = upper_bound
	elif nearest_integer == upper_bound:
		if rv_teta > 0:
			y = upper_bound
		else:
			y = lower_bound

	return y 

def get_initial_states_and_Q(all_adjacency_matrices,all_isps,virtual_graph,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix,min_cutoff,max_cutoff, num_of_possible_actions_internal_node, num_of_possible_actions_internal_path, num_of_possible_actions_external_node, num_of_possible_actions_external_path):

	N_ISPs = len(list(all_adjacency_matrices.keys())) -3 
	number_of_virtual_nodes = all_adjacency_matrices['virtual_graph'].shape[0]

	adj_matrix_peering = all_adjacency_matrices['peering_network']
	adj_global = all_adjacency_matrices['multidomain']
	tot_num_of_phy_nodes = adj_global.shape[0]

	peering_nodes_for_each_isp = {}
	for peering_p1 in range(adj_matrix_peering.shape[0]):
		for peering_p2 in range(adj_matrix_peering.shape[0]):

			if adj_matrix_peering[peering_p1,peering_p2] > 0:

				global_peering_p1 = dict_local_to_peering[peering_p1]
				global_peering_p2 = dict_local_to_peering[peering_p2]

				owner_isp_p1 = info_per_physical_node[global_peering_p1]['ISP']
				owner_isp_p1 = 'ISP_' + str(owner_isp_p1)

				owner_isp_p2 = info_per_physical_node[global_peering_p2]['ISP']
				owner_isp_p2 = 'ISP_' + str(owner_isp_p2)

				if owner_isp_p1 not in peering_nodes_for_each_isp:
					peering_nodes_for_each_isp[owner_isp_p1] = []
				peering_nodes_for_each_isp[owner_isp_p1].append(global_peering_p1)

				if owner_isp_p2 not in peering_nodes_for_each_isp:
					peering_nodes_for_each_isp[owner_isp_p2] = []
				peering_nodes_for_each_isp[owner_isp_p2].append(global_peering_p2)

	# peering_nodes_for_each_isp: key: e.g., "ISP_0"; value: list of peerings nodes (with global IDs) belonging to that ISP

	source_isp_peerings = []
	for isp_source in range(N_ISPs):
		isp_source_id = 'ISP_' + str(isp_source)
		pp_sources = peering_nodes_for_each_isp[isp_source_id]
		for el in pp_sources:
			source_isp_peerings.append(el)

	destination_isp_peerings = []
	for isp_dest in range(N_ISPs):
		isp_dest_id = 'ISP_' + str(isp_dest)
		pp_dests = peering_nodes_for_each_isp[isp_dest_id]
		for el in pp_dests:
			destination_isp_peerings.append(el)

	source_isp_peerings = list(set(source_isp_peerings))
	destination_isp_peerings = list(set(destination_isp_peerings))

	# source_isp_peerings and destination_isp_peerings are the same list, it is just needed to perform iteration with for cycles
	# defining vectors and dictionaries 
	
	states = {}
	Q_dict = {}
	overall_mapping_dict = {} 

	states['orchestrator'] = {}
	Q_dict['orchestrator'] = {}
	overall_mapping_dict['orchestrator'] = {}

	states['orchestrator']['vf'] = {}
	Q_dict['orchestrator']['vf'] = {}
	overall_mapping_dict['orchestrator']['vf'] = {}	

	states['orchestrator']['v_path'] = {}
	Q_dict['orchestrator']['v_path'] = {}
	overall_mapping_dict['orchestrator']['v_path'] = {}	

	# initializing Q/states for vfs 
	for vf_id_number in range(number_of_virtual_nodes):
		vf_id = 'vf_' + str(vf_id_number)
		states['orchestrator']['vf'][vf_id] = np.zeros(N_ISPs)
		curr_state = np.random.choice(np.arange(len(states['orchestrator']['vf'][vf_id]))) #np.argmax(states['orchestrator'][vf_id])
		states['orchestrator']['vf'][vf_id][curr_state] = 1
		Q_dict['orchestrator']['vf'][vf_id] = np.zeros([N_ISPs,num_of_possible_actions_external_node])
		overall_mapping_dict['orchestrator']['vf'][vf_id] = {}
		for row_id in range(N_ISPs):
			overall_mapping_dict['orchestrator']['vf'][vf_id][row_id] = 'ISP_' + str(row_id)


	dict_peering_to_local = {}
	for local_peering in dict_local_to_peering:
		global_peering = dict_local_to_peering[local_peering]
		if global_peering not in dict_peering_to_local:
			dict_peering_to_local[global_peering] = local_peering

	# initializing Q/states for vpaths for the orchestrator
	G_peering_nodes = nx.from_numpy_matrix(adj_matrix_peering)
	
	#min_cutoff = 3
	#max_cutoff_ = 10
	min_paths = 1
	for vf1 in range(number_of_virtual_nodes):
		for vf2 in range(number_of_virtual_nodes):
			vf_id1 = 'vf_' + str(vf1)
			vf_id2 = 'vf_' + str(vf2)

			if vf_id1 != vf_id2:
				curr_virtual_path = vf_id1 + '_' + vf_id2
				if curr_virtual_path not in Q_dict['orchestrator']['v_path']:
					Q_dict['orchestrator']['v_path'][curr_virtual_path] = {}
					states['orchestrator']['v_path'][curr_virtual_path] = {}
					overall_mapping_dict['orchestrator']['v_path'][curr_virtual_path] = {}

				for source_isp in all_isps:
					for dest_isp in all_isps:
						owner_isps = source_isp + '_' + dest_isp 
						if owner_isps not in Q_dict['orchestrator']['v_path'][curr_virtual_path]:
							Q_dict['orchestrator']['v_path'][curr_virtual_path][owner_isps] = []
							states['orchestrator']['v_path'][curr_virtual_path][owner_isps] = []
							overall_mapping_dict['orchestrator']['v_path'][curr_virtual_path][owner_isps] = {}	
			
						if source_isp == dest_isp:
							Q_dict['orchestrator']['v_path'][curr_virtual_path][owner_isps] = np.zeros([1,num_of_possible_actions_external_path])
							states['orchestrator']['v_path'][curr_virtual_path][owner_isps] = np.ones(1)
							overall_mapping_dict['orchestrator']['v_path'][curr_virtual_path][owner_isps][0] = 'same_isp'
						else:
							peerings_of_source_isp = peering_nodes_for_each_isp[source_isp]
							peerings_of_dest_isp = peering_nodes_for_each_isp[dest_isp]

							local_peerings_of_source_isp = [dict_peering_to_local[el] for el in peerings_of_source_isp]
							local_peerings_of_dest_isp = [dict_peering_to_local[el] for el in peerings_of_dest_isp]

							all_paths_between_source_and_dest_ips = []
							for local_peering_in_source_isp in local_peerings_of_source_isp:
								for local_peering_in_dest_isp in local_peerings_of_dest_isp:

									paths_between_node_local_peering_in_source_isp_and_node_local_peering_in_dest_isp = []
									for cutoff in range(min_cutoff,max_cutoff):
										paths_ = nx.all_simple_paths(G_peering_nodes, local_peering_in_source_isp, local_peering_in_dest_isp, cutoff=cutoff)
										for path___ in paths_:
											global_paths___ = [dict_local_to_peering[elel] for elel in path___]
											paths_between_node_local_peering_in_source_isp_and_node_local_peering_in_dest_isp.append(global_paths___)
										if len(paths_between_node_local_peering_in_source_isp_and_node_local_peering_in_dest_isp) > min_paths:
											break 

									for path_xy in paths_between_node_local_peering_in_source_isp_and_node_local_peering_in_dest_isp:
										all_paths_between_source_and_dest_ips.append(path_xy)

							n_paths = len(all_paths_between_source_and_dest_ips)
							Q_dict['orchestrator']['v_path'][curr_virtual_path][owner_isps] = np.zeros([n_paths,num_of_possible_actions_external_path])					
							states['orchestrator']['v_path'][curr_virtual_path][owner_isps] = np.zeros(n_paths)
							curr_state = np.random.choice(np.arange(n_paths))
							states['orchestrator']['v_path'][curr_virtual_path][owner_isps][curr_state] = 1 
							for row_id in range(n_paths):
								overall_mapping_dict['orchestrator']['v_path'][curr_virtual_path][owner_isps][row_id] = all_paths_between_source_and_dest_ips[row_id]	

	# now initialize the states/Q/dictionary for the isps
	for isp_id in range(N_ISPs):
		isp = 'ISP_' + str(isp_id)
		print(isp)

		Q_dict[isp] = {}
		states[isp] = {} 
		overall_mapping_dict[isp] = {}

		Q_dict[isp]['vf'] = {}
		states[isp]['vf'] = {}
		overall_mapping_dict[isp]['vf'] = {}

		Q_dict[isp]['v_path'] = {}
		states[isp]['v_path'] = {} 
		overall_mapping_dict[isp]['v_path'] = {}

		adj_matrix_curr_isp = all_adjacency_matrices[isp_id]
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

		peering_nodes_of_curr_isp = list(set(peering_nodes_of_curr_isp))
		peering_links_of_curr_isp_ = [] 
		for el in peering_links_of_curr_isp:
			if el not in peering_links_of_curr_isp_:
				peering_links_of_curr_isp_.append(el)

		peering_links_of_curr_isp = peering_links_of_curr_isp_

		# defining vectors for the vfs
		for vf1 in range(number_of_virtual_nodes):
			vf_id1 = 'vf_' + str(vf1)

			if vf_id1 not in overall_mapping_dict[isp]['vf']:
				overall_mapping_dict[isp]['vf'][vf_id1] = {}

			id_row = 0
			for phy_node_id in range(tot_num_of_phy_nodes):
				owner_isp = info_per_physical_node[phy_node_id]['ISP']
				if owner_isp == isp_id:
					overall_mapping_dict[isp]['vf'][vf_id1][id_row] = phy_node_id
					id_row += 1 
				
			num_physical_nodes_for_curr_isp = len(list(overall_mapping_dict[isp]['vf'][vf_id1].keys()))

			states[isp]['vf'][vf_id1] = np.zeros(num_physical_nodes_for_curr_isp)
			curr_state = np.random.choice(np.arange(num_physical_nodes_for_curr_isp))
			states[isp]['vf'][vf_id1][curr_state] = 1 
			Q_dict[isp]['vf'][vf_id1] = np.zeros([num_physical_nodes_for_curr_isp,num_of_possible_actions_internal_node])

		G = nx.from_numpy_matrix(adj_matrix_curr_isp)
		# defining vectors for the vpaths
		for vf1 in range(number_of_virtual_nodes):
			for vf2 in range(number_of_virtual_nodes):
				vf_id1 = 'vf_' + str(vf1)
				vf_id2 = 'vf_' + str(vf2)

				if vf1 != vf2:

					curr_virtual_path = vf_id1 + '_' + vf_id2 
					Q_dict[isp]["v_path"][curr_virtual_path] = {}
					states[isp]["v_path"][curr_virtual_path] = {} 
					overall_mapping_dict[isp]["v_path"][curr_virtual_path] = {}
	
					for node_u in range(num_physical_nodes_for_curr_isp):
						global_phy_u = dict_local_to_global_node[isp + '_node_' + str(node_u)]

						for node_v in range(num_physical_nodes_for_curr_isp):
							global_phy_v = dict_local_to_global_node[isp + '_node_' + str(node_v)]	

							curr_phy_path = str(global_phy_u) + '_' + str(global_phy_v)	
							if curr_phy_path not in Q_dict[isp]["v_path"][curr_virtual_path]:
								Q_dict[isp]["v_path"][curr_virtual_path][curr_phy_path] = []
								states[isp]["v_path"][curr_virtual_path][curr_phy_path] = []
								overall_mapping_dict[isp]["v_path"][curr_virtual_path][curr_phy_path] = {}				
								
								# let's compute the physical paths between each pair of (local) nodes
								paths_between_node_u_and_node_v = []

								if node_u != node_v:
									for cutoff in range(min_cutoff,max_cutoff):
										paths_ = nx.all_simple_paths(G, node_u, node_v, cutoff=cutoff)
										for path___ in paths_:
											paths_between_node_u_and_node_v.append(path___)
										if len(paths_between_node_u_and_node_v) > min_paths:
											break 
								else:
									paths_between_node_u_and_node_v.append([node_u,node_u])

							# and then convert those nodes to global 
							global_paths_between_node_u_and_node_v = []
							for path___ in paths_between_node_u_and_node_v:
								global_path = [dict_local_to_global_node[isp + '_node_' + str(node_k)] for node_k in path___]
								global_paths_between_node_u_and_node_v.append(global_path)

							n_paths = len(global_paths_between_node_u_and_node_v)

							Q_dict[isp]["v_path"][curr_virtual_path][curr_phy_path] = np.zeros([n_paths,num_of_possible_actions_internal_path])
							states[isp]["v_path"][curr_virtual_path][curr_phy_path] = np.zeros(n_paths)
							curr_state = np.random.choice(np.arange(n_paths))
							states[isp]["v_path"][curr_virtual_path][curr_phy_path][curr_state] = 1 

							for row_id in range(n_paths):
								overall_mapping_dict[isp]["v_path"][curr_virtual_path][curr_phy_path][row_id] = global_paths_between_node_u_and_node_v[row_id]

					# now I am considering in the states/Qs also the interconnection between peerings going out from the considered isp 
					for peering_node_u in range(adj_matrix_peering.shape[0]):
						global_peering_u = dict_local_to_peering[peering_node_u]

						for peering_node_v in range(adj_matrix_peering.shape[0]):
							global_peering_v = dict_local_to_peering[peering_node_v]

							curr_phy_path = str(global_peering_u) + '_' + str(global_peering_v)	
							owner_isp_of_global_peering_u = 'ISP_' + str(info_per_physical_node[global_peering_u]['ISP'])

							if owner_isp_of_global_peering_u == isp:

								if curr_phy_path not in Q_dict[isp]["v_path"][curr_virtual_path]:
									Q_dict[isp]["v_path"][curr_virtual_path][curr_phy_path] = []
									states[isp]["v_path"][curr_virtual_path][curr_phy_path] = []
									overall_mapping_dict[isp]["v_path"][curr_virtual_path][curr_phy_path] = {}				

									# let's compute the physical paths between each pair of (local) nodes
									paths_between_peering_node_u_and_peering_node_v = []

									if peering_node_u != peering_node_v:
										if adj_matrix_peering[peering_node_u,peering_node_v] > 0:
											paths_between_peering_node_u_and_peering_node_v.append([peering_node_u,peering_node_v])

									# and then convert those nodes to global 
									global_paths_between_node_u_and_node_v = [[dict_local_to_peering[peering_node_u],dict_local_to_peering[peering_node_v]]]
									n_paths = len(global_paths_between_node_u_and_node_v)

									Q_dict[isp]["v_path"][curr_virtual_path][curr_phy_path] = np.zeros([n_paths,num_of_possible_actions_internal_path])
									states[isp]["v_path"][curr_virtual_path][curr_phy_path] = np.zeros(n_paths)
									curr_state = np.random.choice(np.arange(n_paths))
									states[isp]["v_path"][curr_virtual_path][curr_phy_path][curr_state] = 1 

									for row_id in range(n_paths):
										overall_mapping_dict[isp]["v_path"][curr_virtual_path][curr_phy_path][row_id] = global_paths_between_node_u_and_node_v[row_id]

	return states, Q_dict, overall_mapping_dict

def go_to_the_next_state(current_Q,curr_state,eps):

	actions_of_curr_state = current_Q[curr_state]
	maximum_state = current_Q.shape[0] - 1

	rv = np.random.binomial(1,eps)
	if rv > 0: #i.e., exploitation
		curr_action = np.argmax(actions_of_curr_state)
	else: #i.e., exploration 
		curr_action = np.random.choice(np.arange(len(actions_of_curr_state)))

	if curr_action == 0: # i.e., stay in the same place 
		new_state = curr_state
	else:
		if curr_action%2 == 0:
			actual_action = +int(curr_action)/2 
			new_state = curr_state + actual_action
		else:
			actual_action = -(int(curr_action)/2 + 1)
			new_state = curr_state + actual_action

	new_state = int(new_state)
	new_state = new_state%current_Q.shape[0]
	curr_action = int(curr_action)

	return new_state, curr_action

def choose_actions_Q_orch_vfs(current_states, Q_dict, overall_mapping_dict,eps):

	decisions = {}
	decisions['orchestrator'] = {}
	decisions['orchestrator']['vf'] = {}

	for vf_id in current_states['orchestrator']['vf']:

		decisions['orchestrator']['vf'][vf_id] = {}

		curr_state = np.where(current_states['orchestrator']['vf'][vf_id] > 0)[0][0]
		current_Q = Q_dict['orchestrator']['vf'][vf_id]
		new_state, curr_action = go_to_the_next_state(current_Q,curr_state,eps)

		current_states['orchestrator']['vf'][vf_id][:] = 0
		current_states['orchestrator']['vf'][vf_id][new_state] = 1

		decisions['orchestrator']['vf'][vf_id]['curr_state'] = curr_state
		decisions['orchestrator']['vf'][vf_id]['curr_action'] = curr_action
		decisions['orchestrator']['vf'][vf_id]['next_state'] = new_state

	return current_states, decisions

def choose_actions_Q_orch_vpaths(current_states, Q_dict, overall_mapping_dict, eps_external_path,info_per_physical_node,bandwidth_requests_between_virtual_nodes):

	decisions = {}
	for curr_vpath in current_states['orchestrator']['v_path']:
		vfs = curr_vpath.split('_')
		vf_source = 'vf_' + str(vfs[1])
		vf_dest = 'vf_' + str(vfs[3])

		v1 = int(vfs[1])
		v2 = int(vfs[3])

		if bandwidth_requests_between_virtual_nodes[v1,v2] > 0:

			source_isp = current_states['orchestrator']['vf'][vf_source]
			source_isp = np.where(source_isp > 0)[0][0]
			source_isp = overall_mapping_dict['orchestrator']['vf'][vf_source][source_isp]

			dest_isp = current_states['orchestrator']['vf'][vf_dest]
			dest_isp = np.where(dest_isp > 0)[0][0]
			dest_isp = overall_mapping_dict['orchestrator']['vf'][vf_dest][dest_isp]

			curr_owners = source_isp + '_' + dest_isp 
			
			if 'orchestrator' not in decisions:
				decisions['orchestrator'] = {}
				decisions['orchestrator']['v_path'] = {}

			if curr_vpath not in decisions['orchestrator']['v_path']:
				decisions['orchestrator']['v_path'][curr_vpath] = {}

			if curr_owners not in decisions['orchestrator']['v_path'][curr_vpath]:
				decisions['orchestrator']['v_path'][curr_vpath][curr_owners] = {}

			curr_path = current_states['orchestrator']['v_path'][curr_vpath][curr_owners]
			curr_state = np.where(curr_path > 0)[0][0]

			current_Q = Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners]
			new_state, curr_action = go_to_the_next_state(current_Q,curr_state,eps_external_path)

			decisions['orchestrator']['v_path'][curr_vpath][curr_owners]['curr_state'] = curr_state 
			decisions['orchestrator']['v_path'][curr_vpath][curr_owners]['curr_action'] = curr_action 
			decisions['orchestrator']['v_path'][curr_vpath][curr_owners]['next_state'] = new_state 

			current_states['orchestrator']['v_path'][curr_vpath][curr_owners][:] = 0
			current_states['orchestrator']['v_path'][curr_vpath][curr_owners][new_state] = 1

	return current_states, decisions

def choose_actions_Q_isp_vfs(current_states, Q_dict, overall_mapping_dict, eps_internal_node):

	decisions_isp_vfs = {}
	for vf_id in current_states['orchestrator']['vf']:
		curr_state = np.where(current_states['orchestrator']['vf'][vf_id] > 0)[0][0]
		curr_isp = overall_mapping_dict['orchestrator']['vf'][vf_id][curr_state]

		if curr_isp not in decisions_isp_vfs:
			decisions_isp_vfs[curr_isp] = {}
			decisions_isp_vfs[curr_isp]['vf'] = {}

		decisions_isp_vfs[curr_isp]['vf'][vf_id] = {}

		curr_phy_node = current_states[curr_isp]['vf'][vf_id]
		curr_state = np.where(curr_phy_node>0)[0][0]
		current_Q = Q_dict[curr_isp]['vf'][vf_id]

		new_state, curr_action = go_to_the_next_state(current_Q,curr_state,eps_internal_node)

		decisions_isp_vfs[curr_isp]['vf'][vf_id]['curr_state'] = curr_state 
		decisions_isp_vfs[curr_isp]['vf'][vf_id]['curr_action'] = curr_action 
		decisions_isp_vfs[curr_isp]['vf'][vf_id]['next_state'] = new_state

		current_states[curr_isp]['vf'][vf_id][:] = 0
		current_states[curr_isp]['vf'][vf_id][new_state] = 1 

	return current_states, decisions_isp_vfs					

def choose_actions_Q_isp_vpath(current_states, Q_dict, overall_mapping_dict,eps_internal_path,info_per_physical_node,bandwidth_requests_between_virtual_nodes):

	decisions = {}
	for curr_vpath in current_states['orchestrator']['v_path']:
		vfs = curr_vpath.split('_')
		vf_source = 'vf_' + str(vfs[1])
		vf_dest = 'vf_' + str(vfs[3])

		v1 = int(vfs[1])
		v2 = int(vfs[3])

		if bandwidth_requests_between_virtual_nodes[v1,v2] > 0: 

			source_isp = current_states['orchestrator']['vf'][vf_source]
			source_isp = np.where(source_isp > 0)[0][0]
			source_isp = overall_mapping_dict['orchestrator']['vf'][vf_source][source_isp]

			dest_isp = current_states['orchestrator']['vf'][vf_dest]
			dest_isp = np.where(dest_isp > 0)[0][0]
			dest_isp = overall_mapping_dict['orchestrator']['vf'][vf_dest][dest_isp]

			owner_isps = source_isp + '_' + dest_isp
			curr_peering_path = current_states['orchestrator']['v_path'][curr_vpath][owner_isps]
			curr_state = np.where(curr_peering_path>0)[0][0]
			curr_peering_path = overall_mapping_dict['orchestrator']['v_path'][curr_vpath][owner_isps][curr_state]

			if curr_peering_path != 'same_isp':
			
				# what you called before "case 2", i.e., only the source vf is in the isp
				node_u = current_states[source_isp]['vf'][vf_source]
				node_u = np.where(node_u >0)[0][0]
				node_u = overall_mapping_dict[source_isp]['vf'][vf_source][node_u]
				node_v = curr_peering_path[0]

				path_ = current_states[source_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]
				#print(path_)
				curr_state_internal_path = np.where(path_>0)[0][0]
				path_ = overall_mapping_dict[source_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)][curr_state_internal_path]

				current_Q = Q_dict[source_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]
				new_state, curr_action = go_to_the_next_state(current_Q,curr_state_internal_path,eps_internal_path)

				if source_isp not in decisions:
					decisions[source_isp] = {}
					decisions[source_isp]['v_path'] = {}
				if curr_vpath not in decisions[source_isp]['v_path']:
					decisions[source_isp]['v_path'][curr_vpath] = {}
				if str(node_u) + '_' + str(node_v) not in decisions[source_isp]['v_path'][curr_vpath]:
					decisions[source_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)] = {}

				decisions[source_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]['curr_state'] = curr_state_internal_path	
				decisions[source_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]['curr_action'] = curr_action
				decisions[source_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]['next_state'] = new_state

				current_states[source_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)][:] = 0
				current_states[source_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)][new_state] = 1

				# what you called before "case 3", i.e., only the source vf is in the isp			
				node_u = curr_peering_path[-1]

				node_v = current_states[dest_isp]['vf'][vf_dest]
				node_v = np.where(node_v >0)[0][0]
				node_v = overall_mapping_dict[dest_isp]['vf'][vf_dest][node_v]
				
				path_ = current_states[dest_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]
				#print(path_)
				curr_state_internal_path = np.where(path_>0)[0][0]
				path_ = overall_mapping_dict[dest_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)][curr_state_internal_path]

				current_Q = Q_dict[dest_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]
				new_state, curr_action = go_to_the_next_state(current_Q,curr_state_internal_path,eps_internal_path)

				if dest_isp not in decisions:
					decisions[dest_isp] = {}
					decisions[dest_isp]['v_path'] = {}
				if curr_vpath not in decisions[dest_isp]['v_path']:
					decisions[dest_isp]['v_path'][curr_vpath] = {}
				if str(node_u) + '_' + str(node_v) not in decisions[dest_isp]['v_path'][curr_vpath]:
					decisions[dest_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)] = {}

				decisions[dest_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]['curr_state'] = curr_state_internal_path	
				decisions[dest_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]['curr_action'] = curr_action
				decisions[dest_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]['next_state'] = new_state

				current_states[dest_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)][:] = 0
				current_states[dest_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)][new_state] = 1

				# what you called "case 4", i.e., the isp does not have any vf assinged to it but it has to transport the virtual flow
				intermediate_peerings = curr_peering_path[1:-1]
				#intermediate_peerings = ['ISP_' + str(info_per_physical_node[el]['ISP']) for el in intermediate_peerings]

				if len(intermediate_peerings) == 1:
					node_u = intermediate_peerings[0]
					node_v = intermediate_peerings[0]

					considered_isp = 'ISP_' + str(info_per_physical_node[node_u]['ISP'])

					path_ = current_states[considered_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]
					#print(path_)
					curr_state_internal_path = np.where(path_>0)[0][0]
					path_ = overall_mapping_dict[considered_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)][curr_state_internal_path]

					current_Q = Q_dict[considered_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]
					new_state, curr_action = go_to_the_next_state(current_Q,curr_state_internal_path,eps_internal_path)

					if considered_isp not in decisions:
						decisions[considered_isp] = {}
						decisions[considered_isp]['v_path'] = {}
					if curr_vpath not in decisions[considered_isp]['v_path']:
						decisions[considered_isp]['v_path'][curr_vpath] = {}
					if str(node_u) + '_' + str(node_v) not in decisions[considered_isp]['v_path'][curr_vpath]:
						decisions[considered_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)] = {}

					decisions[considered_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]['curr_state'] = curr_state_internal_path	
					decisions[considered_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]['curr_action'] = curr_action
					decisions[considered_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]['next_state'] = new_state

					current_states[considered_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)][:] = 0
					current_states[considered_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)][new_state] = 1				

				else:
					for index_ in range(len(intermediate_peerings)-1):
						node_u = intermediate_peerings[index_]
						node_v = intermediate_peerings[index_+1]	

						considered_isp_u = 'ISP_' + str(info_per_physical_node[node_u]['ISP'])	
						considered_isp_v = 'ISP_' + str(info_per_physical_node[node_v]['ISP'])
						considered_isp = considered_isp_u

						path_ = current_states[considered_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]
						#print(path_)
						curr_state_internal_path = np.where(path_>0)[0][0]
						path_ = overall_mapping_dict[considered_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)][curr_state_internal_path]

						current_Q = Q_dict[considered_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]
						new_state, curr_action = go_to_the_next_state(current_Q,curr_state_internal_path,eps_internal_path)

						if considered_isp not in decisions:
							decisions[considered_isp] = {}
							decisions[considered_isp]['v_path'] = {}
						if curr_vpath not in decisions[considered_isp]['v_path']:
							decisions[considered_isp]['v_path'][curr_vpath] = {}
						if str(node_u) + '_' + str(node_v) not in decisions[considered_isp]['v_path'][curr_vpath]:
							decisions[considered_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)] = {}

						decisions[considered_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]['curr_state'] = curr_state_internal_path	
						decisions[considered_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]['curr_action'] = curr_action
						decisions[considered_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]['next_state'] = new_state

						current_states[considered_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)][:] = 0
						current_states[considered_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)][new_state] = 1						
			
			elif curr_peering_path == 'same_isp':
				node_u = current_states[source_isp]['vf'][vf_source]
				node_u = np.where(node_u >0)[0][0]
				node_u = overall_mapping_dict[source_isp]['vf'][vf_source][node_u]

				node_v = current_states[dest_isp]['vf'][vf_dest]
				node_v = np.where(node_v >0)[0][0]
				node_v = overall_mapping_dict[dest_isp]['vf'][vf_dest][node_v]

				path_ = current_states[source_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]
				#print(path_)
				curr_state_internal_path = np.where(path_>0)[0][0]
				path_ = overall_mapping_dict[source_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)][curr_state_internal_path]

				current_Q = Q_dict[source_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]
				new_state, curr_action = go_to_the_next_state(current_Q,curr_state_internal_path,eps_internal_path)

				if source_isp not in decisions:
					decisions[source_isp] = {}
					decisions[source_isp]['v_path'] = {}
				if curr_vpath not in decisions[source_isp]['v_path']:
					decisions[source_isp]['v_path'][curr_vpath] = {}
				if str(node_u) + '_' + str(node_v) not in decisions[source_isp]['v_path'][curr_vpath]:
					decisions[source_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)] = {}

				decisions[source_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]['curr_state'] = curr_state_internal_path	
				decisions[source_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]['curr_action'] = curr_action
				decisions[source_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)]['next_state'] = new_state

				current_states[source_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)][:] = 0
				current_states[source_isp]['v_path'][curr_vpath][str(node_u) + '_' + str(node_v)][new_state] = 1

	return current_states, decisions

def update_Q_isp_vpaths(Q_dict,decisions_isp_vpath,learning_rate,discount_factor,overall_mapping_dict,bandwidth_requests_between_virtual_nodes,cost_per_link_matrix,capacity_of_all_links,infinity_value):

	old_Q_dict = copy.copy(Q_dict)
	#print(cost_per_link_matrix,'aaaaaaaaaaaaaaa',cost_per_link_matrix.shape,np.max(cost_per_link_matrix))
	bw_on_phy_links = np.zeros(cost_per_link_matrix.shape)
	#print(capacity_of_all_links,'aaaaaaaaaaaaaaaaaaaaa',capacity_of_all_links.shape,cost_per_link_matrix.shape)

	# let us collect the information about the be passing on each pysical link
	for isp in decisions_isp_vpath:
		for curr_vpath in decisions_isp_vpath[isp]['v_path']:

			curr_vpath_ids = curr_vpath.split("_")
			vf_id1 = int(curr_vpath_ids[1])
			vf_id2 = int(curr_vpath_ids[3])

			if bandwidth_requests_between_virtual_nodes[vf_id1,vf_id2] > 0:

				for pair_of_nodes in decisions_isp_vpath[isp]['v_path'][curr_vpath]:
					curr_state = decisions_isp_vpath[isp]['v_path'][curr_vpath][pair_of_nodes]['curr_state']
					curr_action = decisions_isp_vpath[isp]['v_path'][curr_vpath][pair_of_nodes]['curr_action']
					next_state = decisions_isp_vpath[isp]['v_path'][curr_vpath][pair_of_nodes]['next_state']
					next_phy_path = overall_mapping_dict[isp]['v_path'][curr_vpath][pair_of_nodes][next_state]
					
					cost = 0 
					for phy_node_id in range(len(next_phy_path)-1):
						n1 = next_phy_path[phy_node_id]
						n2 = next_phy_path[phy_node_id+1]

						if n1 != n2 and vf_id1 != vf_id2:
							bw_on_phy_links[n1,n2] += bandwidth_requests_between_virtual_nodes[vf_id1,vf_id2]

	cost_of_all_vpaths_per_isp = {}
	overall_cost = 0 
	for isp in decisions_isp_vpath:
		if isp not in cost_of_all_vpaths_per_isp:
			cost_of_all_vpaths_per_isp[isp] = {}
		for curr_vpath in decisions_isp_vpath[isp]['v_path']:

			curr_vpath_ids = curr_vpath.split("_")
			vf_id1 = int(curr_vpath_ids[1])
			vf_id2 = int(curr_vpath_ids[3])

			if bandwidth_requests_between_virtual_nodes[vf_id1,vf_id2] > 0:

				if curr_vpath not in cost_of_all_vpaths_per_isp:
					cost_of_all_vpaths_per_isp[isp][curr_vpath] = 0
				
				cost_curr_vpath = 0
				for pair_of_nodes in decisions_isp_vpath[isp]['v_path'][curr_vpath]:

					curr_state = decisions_isp_vpath[isp]['v_path'][curr_vpath][pair_of_nodes]['curr_state']
					curr_action = decisions_isp_vpath[isp]['v_path'][curr_vpath][pair_of_nodes]['curr_action']
					next_state = decisions_isp_vpath[isp]['v_path'][curr_vpath][pair_of_nodes]['next_state']

					next_phy_path = overall_mapping_dict[isp]['v_path'][curr_vpath][pair_of_nodes][next_state]
					
					cost = 0 
					for phy_node_id in range(len(next_phy_path)-1):
						n1 = next_phy_path[phy_node_id]
						n2 = next_phy_path[phy_node_id+1]

						if n1 != n2 and vf_id1 != vf_id2:
							cost += cost_per_link_matrix[n1,n2]*bandwidth_requests_between_virtual_nodes[vf_id1,vf_id2]#*300
							if bw_on_phy_links[n1,n2] > capacity_of_all_links[n1,n2]:
								#print("links' capacity violated")
								cost_curr_vpath += infinity_value
								cost += infinity_value

					cost_curr_vpath += cost 				

					overall_cost += cost
					reward = -cost #+ 10**6

					#reward/=1000. 
					#reward = int(reward)
					#reward*=10000. 

					max_Q = np.max(old_Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes][next_state])

					"""
					max_ = np.max(old_Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes])
					min_ = np.min(old_Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes])

					if max_ > min_:
						old_Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes] = (old_Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes] - min_)/(max_ - min_)
					"""

					#to_add1 = get_rounding(discount_factor*max_Q)
					to_add1 = discount_factor*max_Q
					to_add = learning_rate*(reward + to_add1 - old_Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes][curr_state,curr_action])
					#to_add = get_rounding(to_add)

					Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes][curr_state,curr_action] += to_add
					#Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes] = np.around(Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes],decimals=0)
					#print(Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes])

				cost_of_all_vpaths_per_isp[isp][curr_vpath] += cost_curr_vpath

	#cost_of_all_vpaths_per_isp[isp][curr_vpath] = cost_of_all_vpaths_per_isp[isp][curr_vpath]**3

	return Q_dict, overall_cost, cost_of_all_vpaths_per_isp

def update_Q_isp_vpaths_private(Q_dict,decisions_isp_vpath,learning_rate,discount_factor,overall_mapping_dict,bandwidth_requests_between_virtual_nodes,cost_per_link_matrix,capacity_of_all_links,infinity_value):

	old_Q_dict = copy.copy(Q_dict)
	#print(cost_per_link_matrix,'aaaaaaaaaaaaaaa',cost_per_link_matrix.shape,np.max(cost_per_link_matrix))
	bw_on_phy_links = np.zeros(cost_per_link_matrix.shape)
	#print(capacity_of_all_links,'aaaaaaaaaaaaaaaaaaaaa',capacity_of_all_links.shape,cost_per_link_matrix.shape)

	# let us collect the information about the be passing on each pysical link
	for isp in decisions_isp_vpath:
		for curr_vpath in decisions_isp_vpath[isp]['v_path']:

			curr_vpath_ids = curr_vpath.split("_")
			vf_id1 = int(curr_vpath_ids[1])
			vf_id2 = int(curr_vpath_ids[3])

			if bandwidth_requests_between_virtual_nodes[vf_id1,vf_id2] > 0:

				for pair_of_nodes in decisions_isp_vpath[isp]['v_path'][curr_vpath]:
					curr_state = decisions_isp_vpath[isp]['v_path'][curr_vpath][pair_of_nodes]['curr_state']
					curr_action = decisions_isp_vpath[isp]['v_path'][curr_vpath][pair_of_nodes]['curr_action']
					next_state = decisions_isp_vpath[isp]['v_path'][curr_vpath][pair_of_nodes]['next_state']
					next_phy_path = overall_mapping_dict[isp]['v_path'][curr_vpath][pair_of_nodes][next_state]
					
					cost = 0 
					for phy_node_id in range(len(next_phy_path)-1):
						n1 = next_phy_path[phy_node_id]
						n2 = next_phy_path[phy_node_id+1]

						if n1 != n2 and vf_id1 != vf_id2:
							bw_on_phy_links[n1,n2] += bandwidth_requests_between_virtual_nodes[vf_id1,vf_id2]

	cost_of_all_vpaths_per_isp = {}
	overall_cost = 0 
	for isp in decisions_isp_vpath:
		if isp not in cost_of_all_vpaths_per_isp:
			cost_of_all_vpaths_per_isp[isp] = {}
		for curr_vpath in decisions_isp_vpath[isp]['v_path']:

			curr_vpath_ids = curr_vpath.split("_")
			vf_id1 = int(curr_vpath_ids[1])
			vf_id2 = int(curr_vpath_ids[3])

			if bandwidth_requests_between_virtual_nodes[vf_id1,vf_id2] > 0:

				if curr_vpath not in cost_of_all_vpaths_per_isp:
					cost_of_all_vpaths_per_isp[isp][curr_vpath] = 0
				
				cost_curr_vpath = 0
				for pair_of_nodes in decisions_isp_vpath[isp]['v_path'][curr_vpath]:

					curr_state = decisions_isp_vpath[isp]['v_path'][curr_vpath][pair_of_nodes]['curr_state']
					curr_action = decisions_isp_vpath[isp]['v_path'][curr_vpath][pair_of_nodes]['curr_action']
					next_state = decisions_isp_vpath[isp]['v_path'][curr_vpath][pair_of_nodes]['next_state']

					next_phy_path = overall_mapping_dict[isp]['v_path'][curr_vpath][pair_of_nodes][next_state]
					
					cost = 0 
					for phy_node_id in range(len(next_phy_path)-1):
						n1 = next_phy_path[phy_node_id]
						n2 = next_phy_path[phy_node_id+1]

						if n1 != n2 and vf_id1 != vf_id2:
							cost += cost_per_link_matrix[n1,n2]*bandwidth_requests_between_virtual_nodes[vf_id1,vf_id2]#*300
							if bw_on_phy_links[n1,n2] > capacity_of_all_links[n1,n2]:
								#print("links' capacity violated")
								cost_curr_vpath += infinity_value
								cost += infinity_value

					cost_curr_vpath += cost 				

					overall_cost += cost
					reward = -cost #+ 10**6

					#reward/=1000. 
					#reward = int(reward)
					#reward*=10000. 

					max_Q = np.max(old_Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes][next_state])

					"""
					max_ = np.max(old_Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes])
					min_ = np.min(old_Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes])

					if max_ > min_:
						old_Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes] = (old_Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes] - min_)/(max_ - min_)
					"""

					to_add1 = get_rounding(discount_factor*max_Q)
					to_add = learning_rate*(reward + to_add1 - old_Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes][curr_state,curr_action])
					to_add = get_rounding(to_add)

					Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes][curr_state,curr_action] += to_add
					Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes] = np.around(Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes],decimals=0)
					#print(Q_dict[isp]['v_path'][curr_vpath][pair_of_nodes])

				cost_of_all_vpaths_per_isp[isp][curr_vpath] += cost_curr_vpath

	#cost_of_all_vpaths_per_isp[isp][curr_vpath] = cost_of_all_vpaths_per_isp[isp][curr_vpath]**3

	return Q_dict, overall_cost, cost_of_all_vpaths_per_isp

def update_Q_isp_vfs(Q_dict,decisions_isp_vpath,decisions_isp_vfs,cost_of_all_vpaths_per_isp,cost_of_all_vfs_per_isp,learning_rate_internal_node,discount_factor_internal_node,overall_mapping_dict,cost_per_virtual_node,demand_per_virtual_node,dict_of_capacities_of_physical_nodes,infinity_value,feasibility_matrix,bandwidth_requests_between_virtual_nodes,prob_node_capacity_verification):

	old_Q_dict = copy.copy(Q_dict)

	cost_of_all_vfs_per_isp_embedding_plus_vpath = {}
	for isp in decisions_isp_vpath:
		cost_of_all_vfs_per_isp_embedding_plus_vpath[isp] = {}
		for vpath in decisions_isp_vpath[isp]['v_path']:

			vfs = vpath.split('_')
			vf1 = 'vf_' + str(vfs[1])
			vf2 = 'vf_' + str(vfs[3])

			#print(bandwidth_requests_between_virtual_nodes,vf1,vf2)

			if bandwidth_requests_between_virtual_nodes[int(vfs[1]),int(vfs[3])] > 0:

				if vf1 not in cost_of_all_vfs_per_isp_embedding_plus_vpath[isp]:
					cost_of_all_vfs_per_isp_embedding_plus_vpath[isp][vf1] = 0

				if vf2 not in cost_of_all_vfs_per_isp_embedding_plus_vpath[isp]:
					cost_of_all_vfs_per_isp_embedding_plus_vpath[isp][vf2] = 0

				cost_of_all_vfs_per_isp_embedding_plus_vpath[isp][vf1] += 1*cost_of_all_vpaths_per_isp[isp][vpath]
				cost_of_all_vfs_per_isp_embedding_plus_vpath[isp][vf2] += 1*cost_of_all_vpaths_per_isp[isp][vpath]

	for isp in decisions_isp_vfs:
		if isp not in cost_of_all_vfs_per_isp_embedding_plus_vpath:
			cost_of_all_vfs_per_isp_embedding_plus_vpath[isp] = {}
		for vf_id in decisions_isp_vfs[isp]['vf']:
			if vf_id not in cost_of_all_vfs_per_isp_embedding_plus_vpath[isp]:
				cost_of_all_vfs_per_isp_embedding_plus_vpath[isp][vf_id] = 0

	# cost_of_all_vfs_per_isp_embedding_plus_vpath takes into consideration also the cost of the vpaths of which a vf is endpoint
	# whereas cost_of_all_vfs_per_isp only considers the cost of embeddig a vf into a physical node
	cost_of_all_vfs_per_isp = {}

	occupied_capacity_per_phy_nodes = {}
	for isp_id in decisions_isp_vfs:
		cost_of_all_vfs_per_isp[isp_id] = {}
		for vf_id in decisions_isp_vfs[isp_id]['vf']:
			curr_state = decisions_isp_vfs[isp_id]['vf'][vf_id]['curr_state']
			curr_action = decisions_isp_vfs[isp_id]['vf'][vf_id]['curr_action']
			new_state = decisions_isp_vfs[isp_id]['vf'][vf_id]['next_state']

			vf_id_number = int(vf_id.split('_')[1])
			#print(cost_per_virtual_node.shape,vf_id_number,cost_per_virtual_node[vf_id_number],'aa')
			global_phy_node_of_new_state = overall_mapping_dict[isp_id]['vf'][vf_id][new_state]
			if global_phy_node_of_new_state not in occupied_capacity_per_phy_nodes:
				occupied_capacity_per_phy_nodes[global_phy_node_of_new_state] = []
			occupied_capacity_per_phy_nodes[global_phy_node_of_new_state].append(vf_id)

	capacity_violated_for_vf = {} #key: vf, value: either infinity (if capacity of the phy node where the vf is embedded) or 0 otherwise
	for global_phy_node_of_new_state in occupied_capacity_per_phy_nodes:
		tot_capacity_occupied_for_this_node = 0 
		for vf___ in occupied_capacity_per_phy_nodes[global_phy_node_of_new_state]:
			vf_id_number = int(vf___.split('_')[-1])
			tot_capacity_occupied_for_this_node += demand_per_virtual_node[vf_id_number] 

		for vf___ in occupied_capacity_per_phy_nodes[global_phy_node_of_new_state]:
			if tot_capacity_occupied_for_this_node > dict_of_capacities_of_physical_nodes[global_phy_node_of_new_state]:
				rv = np.random.binomial(1,prob_node_capacity_verification)
				if rv > 0:
					capacity_violated_for_vf[vf___] = infinity_value
				else:
					capacity_violated_for_vf[vf___] = 0
			else:
				capacity_violated_for_vf[vf___] = 0

	overall_cost = 0	
	for isp_id in decisions_isp_vfs:
		cost_of_all_vfs_per_isp[isp_id] = {}
		for vf_id in decisions_isp_vfs[isp_id]['vf']:

			if vf_id not in cost_of_all_vfs_per_isp_embedding_plus_vpath[isp]:
				cost_of_all_vfs_per_isp_embedding_plus_vpath[isp][vf_id] = 0

			curr_state = decisions_isp_vfs[isp_id]['vf'][vf_id]['curr_state']
			curr_action = decisions_isp_vfs[isp_id]['vf'][vf_id]['curr_action']
			new_state = decisions_isp_vfs[isp_id]['vf'][vf_id]['next_state']

			vf_id_number = int(vf_id.split('_')[1])
			#print(cost_per_virtual_node.shape,vf_id_number,cost_per_virtual_node[vf_id_number],'aa')

			global_phy_node_of_new_state = overall_mapping_dict[isp_id]['vf'][vf_id][new_state]
			cost = cost_per_virtual_node[global_phy_node_of_new_state,vf_id_number]*feasibility_matrix[global_phy_node_of_new_state,vf_id_number] #*demand_per_virtual_node[vf_id_number]
			cost += capacity_violated_for_vf[vf___]

			cost_of_all_vfs_per_isp[isp_id][vf_id] = cost
			cost_of_all_vfs_per_isp_embedding_plus_vpath[isp_id][vf_id] += cost 			

			overall_cost += cost
			#cost += cost_of_all_vfs_per_isp_embedding_plus_vpath[isp][vf_id]

			reward = -cost #+ len(occupied_capacity_per_phy_nodes[global_phy_node_of_new_state])*50
			#reward = -cost #+ 10**6 
			#reward/=1000. 
			#reward = int(reward)
			#reward*=10000. 

			"""
			max_ = np.max(old_Q_dict[isp_id]['vf'][vf_id][new_state])
			min_ = np.min(old_Q_dict[isp_id]['vf'][vf_id][new_state])

			if max_ > min_:
				old_Q_dict[isp_id]['vf'][vf_id][new_state] = (old_Q_dict[isp_id]['vf'][vf_id][new_state] - min_)/(max_ - min_)
			"""
			max_Q = np.max(old_Q_dict[isp_id]['vf'][vf_id][new_state])
			#to_add1 = get_rounding(discount_factor_internal_node*max_Q)
			to_add1 = discount_factor_internal_node*max_Q
			to_add = learning_rate_internal_node*(reward + to_add1 - old_Q_dict[isp_id]['vf'][vf_id][curr_state,curr_action])
			#to_add = get_rounding(to_add)

			Q_dict[isp_id]['vf'][vf_id][curr_state,curr_action] += to_add
			#Q_dict[isp_id]['vf'][vf_id] = np.around(Q_dict[isp_id]['vf'][vf_id],decimals=0)
	
	return Q_dict, cost_of_all_vfs_per_isp_embedding_plus_vpath

def update_Q_isp_vfs_private(Q_dict,decisions_isp_vpath,decisions_isp_vfs,cost_of_all_vpaths_per_isp,cost_of_all_vfs_per_isp,learning_rate_internal_node,discount_factor_internal_node,overall_mapping_dict,cost_per_virtual_node,demand_per_virtual_node,dict_of_capacities_of_physical_nodes,infinity_value,feasibility_matrix,bandwidth_requests_between_virtual_nodes,prob_node_capacity_verification):

	old_Q_dict = copy.copy(Q_dict)

	cost_of_all_vfs_per_isp_embedding_plus_vpath = {}
	for isp in decisions_isp_vpath:
		cost_of_all_vfs_per_isp_embedding_plus_vpath[isp] = {}
		for vpath in decisions_isp_vpath[isp]['v_path']:

			vfs = vpath.split('_')
			vf1 = 'vf_' + str(vfs[1])
			vf2 = 'vf_' + str(vfs[3])

			#print(bandwidth_requests_between_virtual_nodes,vf1,vf2)

			if bandwidth_requests_between_virtual_nodes[int(vfs[1]),int(vfs[3])] > 0:

				if vf1 not in cost_of_all_vfs_per_isp_embedding_plus_vpath[isp]:
					cost_of_all_vfs_per_isp_embedding_plus_vpath[isp][vf1] = 0

				if vf2 not in cost_of_all_vfs_per_isp_embedding_plus_vpath[isp]:
					cost_of_all_vfs_per_isp_embedding_plus_vpath[isp][vf2] = 0

				cost_of_all_vfs_per_isp_embedding_plus_vpath[isp][vf1] += 1*cost_of_all_vpaths_per_isp[isp][vpath]
				cost_of_all_vfs_per_isp_embedding_plus_vpath[isp][vf2] += 1*cost_of_all_vpaths_per_isp[isp][vpath]

	for isp in decisions_isp_vfs:
		if isp not in cost_of_all_vfs_per_isp_embedding_plus_vpath:
			cost_of_all_vfs_per_isp_embedding_plus_vpath[isp] = {}
		for vf_id in decisions_isp_vfs[isp]['vf']:
			if vf_id not in cost_of_all_vfs_per_isp_embedding_plus_vpath[isp]:
				cost_of_all_vfs_per_isp_embedding_plus_vpath[isp][vf_id] = 0

	# cost_of_all_vfs_per_isp_embedding_plus_vpath takes into consideration also the cost of the vpaths of which a vf is endpoint
	# whereas cost_of_all_vfs_per_isp only considers the cost of embeddig a vf into a physical node
	cost_of_all_vfs_per_isp = {}

	occupied_capacity_per_phy_nodes = {}
	for isp_id in decisions_isp_vfs:
		cost_of_all_vfs_per_isp[isp_id] = {}
		for vf_id in decisions_isp_vfs[isp_id]['vf']:
			curr_state = decisions_isp_vfs[isp_id]['vf'][vf_id]['curr_state']
			curr_action = decisions_isp_vfs[isp_id]['vf'][vf_id]['curr_action']
			new_state = decisions_isp_vfs[isp_id]['vf'][vf_id]['next_state']

			vf_id_number = int(vf_id.split('_')[1])
			#print(cost_per_virtual_node.shape,vf_id_number,cost_per_virtual_node[vf_id_number],'aa')
			global_phy_node_of_new_state = overall_mapping_dict[isp_id]['vf'][vf_id][new_state]
			if global_phy_node_of_new_state not in occupied_capacity_per_phy_nodes:
				occupied_capacity_per_phy_nodes[global_phy_node_of_new_state] = []
			occupied_capacity_per_phy_nodes[global_phy_node_of_new_state].append(vf_id)

	capacity_violated_for_vf = {} #key: vf, value: either infinity (if capacity of the phy node where the vf is embedded) or 0 otherwise
	for global_phy_node_of_new_state in occupied_capacity_per_phy_nodes:
		tot_capacity_occupied_for_this_node = 0 
		for vf___ in occupied_capacity_per_phy_nodes[global_phy_node_of_new_state]:
			vf_id_number = int(vf___.split('_')[-1])
			tot_capacity_occupied_for_this_node += demand_per_virtual_node[vf_id_number] 

		for vf___ in occupied_capacity_per_phy_nodes[global_phy_node_of_new_state]:
			if tot_capacity_occupied_for_this_node > dict_of_capacities_of_physical_nodes[global_phy_node_of_new_state]:
				rv = np.random.binomial(1,prob_node_capacity_verification)
				if rv > 0:
					capacity_violated_for_vf[vf___] = infinity_value
				else:
					capacity_violated_for_vf[vf___] = 0
			else:
				capacity_violated_for_vf[vf___] = 0

	overall_cost = 0	
	for isp_id in decisions_isp_vfs:
		cost_of_all_vfs_per_isp[isp_id] = {}
		for vf_id in decisions_isp_vfs[isp_id]['vf']:

			if vf_id not in cost_of_all_vfs_per_isp_embedding_plus_vpath[isp]:
				cost_of_all_vfs_per_isp_embedding_plus_vpath[isp][vf_id] = 0

			curr_state = decisions_isp_vfs[isp_id]['vf'][vf_id]['curr_state']
			curr_action = decisions_isp_vfs[isp_id]['vf'][vf_id]['curr_action']
			new_state = decisions_isp_vfs[isp_id]['vf'][vf_id]['next_state']

			vf_id_number = int(vf_id.split('_')[1])
			#print(cost_per_virtual_node.shape,vf_id_number,cost_per_virtual_node[vf_id_number],'aa')

			global_phy_node_of_new_state = overall_mapping_dict[isp_id]['vf'][vf_id][new_state]
			cost = cost_per_virtual_node[global_phy_node_of_new_state,vf_id_number]*feasibility_matrix[global_phy_node_of_new_state,vf_id_number] #*demand_per_virtual_node[vf_id_number]
			cost += capacity_violated_for_vf[vf___]

			cost_of_all_vfs_per_isp[isp_id][vf_id] = cost
			cost_of_all_vfs_per_isp_embedding_plus_vpath[isp_id][vf_id] += cost 			

			overall_cost += cost
			cost += cost_of_all_vfs_per_isp_embedding_plus_vpath[isp][vf_id]

			reward = -cost #+ len(occupied_capacity_per_phy_nodes[global_phy_node_of_new_state])*50
			#reward = -cost #+ 10**6 
			#reward/=1000. 
			#reward = int(reward)
			#reward*=10000. 

			"""
			max_ = np.max(old_Q_dict[isp_id]['vf'][vf_id][new_state])
			min_ = np.min(old_Q_dict[isp_id]['vf'][vf_id][new_state])

			if max_ > min_:
				old_Q_dict[isp_id]['vf'][vf_id][new_state] = (old_Q_dict[isp_id]['vf'][vf_id][new_state] - min_)/(max_ - min_)
			"""
			max_Q = np.max(old_Q_dict[isp_id]['vf'][vf_id][new_state])
			to_add1 = get_rounding(discount_factor_internal_node*max_Q)
			to_add = learning_rate_internal_node*(reward + to_add1 - old_Q_dict[isp_id]['vf'][vf_id][curr_state,curr_action])
			to_add = get_rounding(to_add)

			Q_dict[isp_id]['vf'][vf_id][curr_state,curr_action] += to_add
			Q_dict[isp_id]['vf'][vf_id] = np.around(Q_dict[isp_id]['vf'][vf_id],decimals=0)
	
	return Q_dict, cost_of_all_vfs_per_isp_embedding_plus_vpath

def update_Q_orch_vpaths(Q_dict, decisions_orch_vpath, cost_of_all_vpaths_per_isp, learning_rate_external_path, discount_factor_external_path,bandwidth_requests_between_virtual_nodes):

	cumulative_cost_of_all_vpaths = {}
	for isp in cost_of_all_vpaths_per_isp:
		for curr_vpath in cost_of_all_vpaths_per_isp[isp]:
			if curr_vpath not in cumulative_cost_of_all_vpaths:
				cumulative_cost_of_all_vpaths[curr_vpath] = 0 
			cumulative_cost_of_all_vpaths[curr_vpath] += cost_of_all_vpaths_per_isp[isp][curr_vpath]

	old_Q_dict = copy.copy(Q_dict)
	#print(decisions_orch_vpath['orchestrator']['v_path'].keys())
	if 'orchestrator' in decisions_orch_vpath: 
		for curr_vpath in decisions_orch_vpath['orchestrator']['v_path']:

			vfs = curr_vpath.split('_')
			vf1 = int(vfs[1])
			vf2 = int(vfs[3])

			if bandwidth_requests_between_virtual_nodes[vf1,vf2] > 0:

				for curr_owners in decisions_orch_vpath['orchestrator']['v_path'][curr_vpath]:
					curr_state = decisions_orch_vpath['orchestrator']['v_path'][curr_vpath][curr_owners]['curr_state']
					curr_action = decisions_orch_vpath['orchestrator']['v_path'][curr_vpath][curr_owners]['curr_action']
					new_state = decisions_orch_vpath['orchestrator']['v_path'][curr_vpath][curr_owners]['next_state']

					isps = curr_owners.split('_')
					isp_x = 'ISP_' + str(isps[1]) 
					isp_y = 'ISP_' + str(isps[3]) 

					cost = cumulative_cost_of_all_vpaths[curr_vpath]
					reward = -cost #+10**6 #np.random.random(1)*10**6 #10**6

					#reward/=1000. 
					#reward = int(reward)

					#reward/=1000. 
					#reward*=10000. 

					"""
					max_ = np.max(old_Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners])
					min_ = np.min(old_Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners])

					if max_ > min_:
						old_Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners] = (old_Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners] - min_)/(max_ - min_)				
					"""
					max_Q = np.max(old_Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners][new_state])
					to_add1 = discount_factor_external_path*max_Q
					#to_add1 = get_rounding(discount_factor_external_path*max_Q)

					to_add = learning_rate_external_path*(reward + to_add1 - old_Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners][curr_state,curr_action])
					#to_add = get_rounding(to_add)

					
					Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners][curr_state,curr_action] += to_add
					#Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners] = np.around(Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners],decimals=0)


	return Q_dict

def update_Q_orch_vpaths_private(Q_dict, decisions_orch_vpath, cost_of_all_vpaths_per_isp, learning_rate_external_path, discount_factor_external_path,bandwidth_requests_between_virtual_nodes):

	cumulative_cost_of_all_vpaths = {}
	for isp in cost_of_all_vpaths_per_isp:
		for curr_vpath in cost_of_all_vpaths_per_isp[isp]:
			if curr_vpath not in cumulative_cost_of_all_vpaths:
				cumulative_cost_of_all_vpaths[curr_vpath] = 0 
			cumulative_cost_of_all_vpaths[curr_vpath] += cost_of_all_vpaths_per_isp[isp][curr_vpath]

	old_Q_dict = copy.copy(Q_dict)
	#print(decisions_orch_vpath['orchestrator']['v_path'].keys())
	for curr_vpath in decisions_orch_vpath['orchestrator']['v_path']:

		vfs = curr_vpath.split('_')
		vf1 = int(vfs[1])
		vf2 = int(vfs[3])

		if bandwidth_requests_between_virtual_nodes[vf1,vf2] > 0:

			for curr_owners in decisions_orch_vpath['orchestrator']['v_path'][curr_vpath]:
				curr_state = decisions_orch_vpath['orchestrator']['v_path'][curr_vpath][curr_owners]['curr_state']
				curr_action = decisions_orch_vpath['orchestrator']['v_path'][curr_vpath][curr_owners]['curr_action']
				new_state = decisions_orch_vpath['orchestrator']['v_path'][curr_vpath][curr_owners]['next_state']

				isps = curr_owners.split('_')
				isp_x = 'ISP_' + str(isps[1]) 
				isp_y = 'ISP_' + str(isps[3]) 

				cost = cumulative_cost_of_all_vpaths[curr_vpath]
				reward = -cost #+10**6 #np.random.random(1)*10**6 #10**6

				#reward/=1000. 
				#reward = int(reward)

				#reward/=1000. 
				#reward*=10000. 

				"""
				max_ = np.max(old_Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners])
				min_ = np.min(old_Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners])

				if max_ > min_:
					old_Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners] = (old_Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners] - min_)/(max_ - min_)				
				"""
				max_Q = np.max(old_Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners][new_state])
				to_add1 = get_rounding(discount_factor_external_path*max_Q)

				to_add = learning_rate_external_path*(reward + to_add1 - old_Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners][curr_state,curr_action])
				to_add = get_rounding(to_add)

				
				Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners][curr_state,curr_action] += to_add
				Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners] = np.around(Q_dict['orchestrator']['v_path'][curr_vpath][curr_owners],decimals=0)


	return Q_dict

def update_Q_orch_vfs(Q_dict, decisions_orch_vfs, cost_of_all_vfs_per_isp_embedding_plus_vpath,learning_rate_external_node, discount_factor_external_node):

	cost_per_vfs = {}
	for isp in cost_of_all_vfs_per_isp_embedding_plus_vpath:
		for vf in cost_of_all_vfs_per_isp_embedding_plus_vpath[isp]:
			if vf not in cost_per_vfs:
				cost_per_vfs[vf] = 0
			cost_per_vfs[vf] += cost_of_all_vfs_per_isp_embedding_plus_vpath[isp][vf]

	old_Q_dict = copy.copy(Q_dict)

	for vf_id in decisions_orch_vfs['orchestrator']['vf']:

		curr_state = decisions_orch_vfs['orchestrator']['vf'][vf_id]['curr_state']
		curr_action = decisions_orch_vfs['orchestrator']['vf'][vf_id]['curr_action']
		new_state = decisions_orch_vfs['orchestrator']['vf'][vf_id]['next_state']

		reward = -cost_per_vfs[vf_id] #+ 10**6

		#reward/=1000. 
		#reward = int(reward)
		#reward*=10000. 

		"""
		max_ = np.max(old_Q_dict['orchestrator']['vf'][vf_id])
		min_ = np.min(old_Q_dict['orchestrator']['vf'][vf_id])

		if max_ > min_:
			old_Q_dict['orchestrator']['vf'][vf_id] = (old_Q_dict['orchestrator']['vf'][vf_id] - min_)/(max_ - min_) 
		"""

		max_Q = np.max(old_Q_dict['orchestrator']['vf'][vf_id][new_state])
		#to_add1 = int(discount_factor_external_node*max_Q)
		#to_add1 = get_rounding(discount_factor_external_node*max_Q)
		to_add1 = discount_factor_external_node*max_Q
		to_add = learning_rate_external_node*(reward + to_add1 - old_Q_dict['orchestrator']['vf'][vf_id][curr_state,curr_action])
		#to_add = get_rounding(to_add)

		Q_dict['orchestrator']['vf'][vf_id][curr_state,curr_action] += to_add
		#Q_dict['orchestrator']['vf'][vf_id][curr_state,curr_action] = np.around(Q_dict['orchestrator']['vf'][vf_id][curr_state,curr_action],decimals=0)

	return Q_dict

def update_Q_orch_vfs_private(Q_dict, decisions_orch_vfs, cost_of_all_vfs_per_isp_embedding_plus_vpath,learning_rate_external_node, discount_factor_external_node):

	cost_per_vfs = {}
	for isp in cost_of_all_vfs_per_isp_embedding_plus_vpath:
		for vf in cost_of_all_vfs_per_isp_embedding_plus_vpath[isp]:
			if vf not in cost_per_vfs:
				cost_per_vfs[vf] = 0
			cost_per_vfs[vf] += cost_of_all_vfs_per_isp_embedding_plus_vpath[isp][vf]

	old_Q_dict = copy.copy(Q_dict)

	for vf_id in decisions_orch_vfs['orchestrator']['vf']:

		curr_state = decisions_orch_vfs['orchestrator']['vf'][vf_id]['curr_state']
		curr_action = decisions_orch_vfs['orchestrator']['vf'][vf_id]['curr_action']
		new_state = decisions_orch_vfs['orchestrator']['vf'][vf_id]['next_state']

		reward = -cost_per_vfs[vf_id] #+ 10**6

		#reward/=1000. 
		#reward = int(reward)
		#reward*=10000. 

		"""
		max_ = np.max(old_Q_dict['orchestrator']['vf'][vf_id])
		min_ = np.min(old_Q_dict['orchestrator']['vf'][vf_id])

		if max_ > min_:
			old_Q_dict['orchestrator']['vf'][vf_id] = (old_Q_dict['orchestrator']['vf'][vf_id] - min_)/(max_ - min_) 
		"""

		max_Q = np.max(old_Q_dict['orchestrator']['vf'][vf_id][new_state])
		#to_add1 = int(discount_factor_external_node*max_Q)
		to_add1 = get_rounding(discount_factor_external_node*max_Q)
		to_add = learning_rate_external_node*(reward + to_add1 - old_Q_dict['orchestrator']['vf'][vf_id][curr_state,curr_action])
		to_add = get_rounding(to_add)

		Q_dict['orchestrator']['vf'][vf_id][curr_state,curr_action] += to_add
		Q_dict['orchestrator']['vf'][vf_id][curr_state,curr_action] = np.around(Q_dict['orchestrator']['vf'][vf_id][curr_state,curr_action],decimals=0)

	return Q_dict

def compute_embedding_cost(current_states,overall_mapping_dict,cost_per_virtual_node,demand_per_virtual_node,dict_of_capacities_of_physical_nodes,infinity_value,feasibility_matrix):

	#print(feasibility_matrix,feasibility_matrix.shape)
	occupied_capacity_per_phy_nodes = {}
	for vf in current_states['orchestrator']['vf']:
		where_is_vf = current_states['orchestrator']['vf'][vf]
		where_is_vf = np.where(where_is_vf > 0)[0][0]
		where_is_vf = overall_mapping_dict['orchestrator']['vf'][vf][where_is_vf]
		#print(where_is_vf,vf)

		where_is_vf_within_isp = current_states[where_is_vf]['vf'][vf]
		where_is_vf_within_isp = np.where(where_is_vf_within_isp > 0)[0][0]
		where_is_vf_within_isp = overall_mapping_dict[where_is_vf]['vf'][vf][where_is_vf_within_isp]

		vf_id_number = int(vf.split('_')[-1])

		if where_is_vf_within_isp not in occupied_capacity_per_phy_nodes:
			occupied_capacity_per_phy_nodes[where_is_vf_within_isp] = 0

		occupied_capacity_per_phy_nodes[where_is_vf_within_isp] += demand_per_virtual_node[vf_id_number]				

	cost_of_embedding = 0
	vf_per_phy_nodes = {}
	for vf in current_states['orchestrator']['vf']:
		where_is_vf = current_states['orchestrator']['vf'][vf]
		where_is_vf = np.where(where_is_vf > 0)[0][0]
		where_is_vf = overall_mapping_dict['orchestrator']['vf'][vf][where_is_vf]
		#print(where_is_vf,vf)

		where_is_vf_within_isp = current_states[where_is_vf]['vf'][vf]
		where_is_vf_within_isp = np.where(where_is_vf_within_isp > 0)[0][0]
		where_is_vf_within_isp = overall_mapping_dict[where_is_vf]['vf'][vf][where_is_vf_within_isp]

		vf_id_number = int(vf.split('_')[-1])
		cost_of_embedding_curr_vf = cost_per_virtual_node[where_is_vf_within_isp,vf_id_number] #this includes the cost and the demand #*demand_per_virtual_node[vf_id_number]			
		cost_of_embedding_curr_vf*=feasibility_matrix[where_is_vf_within_isp,vf_id_number]

		cost_of_embedding += cost_of_embedding_curr_vf

		if occupied_capacity_per_phy_nodes[where_is_vf_within_isp] > dict_of_capacities_of_physical_nodes[where_is_vf_within_isp]:
			cost_of_embedding += infinity_value

		if where_is_vf_within_isp not in vf_per_phy_nodes:
			vf_per_phy_nodes[where_is_vf_within_isp] = []
		vf_per_phy_nodes[where_is_vf_within_isp].append(vf)

	return cost_of_embedding, vf_per_phy_nodes

def compute_vpath_cost(current_states,overall_mapping_dict,cost_per_link_matrix,bandwidth_requests_between_virtual_nodes,info_per_physical_node,capacity_of_all_links,infinity_value):

	traversed_links = []
	bw_on_phy_links = np.zeros(capacity_of_all_links.shape)
	
	for curr_vpath in current_states['orchestrator']['v_path']:

		vfs = curr_vpath.split("_")
		vf_id1 = 'vf_' + str(vfs[1])
		vf_id2 = 'vf_' + str(vfs[3])

		vf_id1_number = int(vfs[1])
		vf_id2_number = int(vfs[3])

		owner_isp_vf_id1 = current_states['orchestrator']['vf'][vf_id1]
		owner_isp_vf_id1 = np.where(owner_isp_vf_id1>0)[0][0]
		owner_isp_vf_id1 = overall_mapping_dict['orchestrator']['vf'][vf_id1][owner_isp_vf_id1]

		owner_isp_vf_id2 = current_states['orchestrator']['vf'][vf_id2]
		owner_isp_vf_id2 = np.where(owner_isp_vf_id2>0)[0][0]
		owner_isp_vf_id2 = overall_mapping_dict['orchestrator']['vf'][vf_id2][owner_isp_vf_id2]

		curr_owners = owner_isp_vf_id1 + '_' + owner_isp_vf_id2

		curr_peering_path = current_states['orchestrator']['v_path'][curr_vpath][curr_owners]
		curr_peering_path = np.where(curr_peering_path > 0)[0][0]
		curr_peering_path = overall_mapping_dict['orchestrator']['v_path'][curr_vpath][curr_owners][curr_peering_path]

		if curr_peering_path == 'same_isp':
			# case 1
			global_phy_u = current_states[owner_isp_vf_id1]['vf'][vf_id1]
			global_phy_u = np.where(global_phy_u>0)[0][0]
			global_phy_u = overall_mapping_dict[owner_isp_vf_id1]['vf'][vf_id1][global_phy_u]

			global_phy_v = current_states[owner_isp_vf_id2]['vf'][vf_id2]
			global_phy_v = np.where(global_phy_v>0)[0][0]
			global_phy_v = overall_mapping_dict[owner_isp_vf_id2]['vf'][vf_id2][global_phy_v]

			owner_isp = owner_isp_vf_id1
			curr_phy_path = current_states[owner_isp]['v_path'][curr_vpath][str(global_phy_u) + '_' + str(global_phy_v)]
			curr_phy_path = np.where(curr_phy_path > 0)[0][0]
			curr_phy_path = overall_mapping_dict[owner_isp]['v_path'][curr_vpath][str(global_phy_u) + '_' + str(global_phy_v)][curr_phy_path]

			for node_index in range(len(curr_phy_path)-1):
				n1 = curr_phy_path[node_index]
				n2 = curr_phy_path[node_index+1]

				if n1 != n2 and vf_id1 != vf_id2:
					bw_on_phy_links[n1,n2] += bandwidth_requests_between_virtual_nodes[vf_id1_number,vf_id2_number]
					if bandwidth_requests_between_virtual_nodes[vf_id1_number,vf_id2_number] > 0:
						traversed_links.append([n1,n2])

		else:
			# case 2 
			source_peering_node = curr_peering_path[0]
			global_phy_v = source_peering_node

			global_phy_u = current_states[owner_isp_vf_id1]['vf'][vf_id1]
			global_phy_u = np.where(global_phy_u>0)[0][0]
			global_phy_u = overall_mapping_dict[owner_isp_vf_id1]['vf'][vf_id1][global_phy_u]

			curr_phy_path = current_states[owner_isp_vf_id1]['v_path'][curr_vpath][str(global_phy_u) + '_' + str(global_phy_v)]
			curr_phy_path = np.where(curr_phy_path > 0)[0][0]
			curr_phy_path = overall_mapping_dict[owner_isp_vf_id1]['v_path'][curr_vpath][str(global_phy_u) + '_' + str(global_phy_v)][curr_phy_path]
	
			for node_index in range(len(curr_phy_path)-1):
				n1 = curr_phy_path[node_index]
				n2 = curr_phy_path[node_index+1]

				if n1 != n2 and vf_id1 != vf_id2:
					bw_on_phy_links[n1,n2] += bandwidth_requests_between_virtual_nodes[vf_id1_number,vf_id2_number]
					
					if bandwidth_requests_between_virtual_nodes[vf_id1_number,vf_id2_number] > 0:
						traversed_links.append([n1,n2])

			# case 3
			source_peering_node = curr_peering_path[-1]
			global_phy_u = source_peering_node

			global_phy_v= current_states[owner_isp_vf_id2]['vf'][vf_id2]
			global_phy_v = np.where(global_phy_v>0)[0][0]
			global_phy_v = overall_mapping_dict[owner_isp_vf_id2]['vf'][vf_id2][global_phy_v]

			curr_phy_path = current_states[owner_isp_vf_id2]['v_path'][curr_vpath][str(global_phy_u) + '_' + str(global_phy_v)]
			curr_phy_path = np.where(curr_phy_path > 0)[0][0]
			curr_phy_path = overall_mapping_dict[owner_isp_vf_id2]['v_path'][curr_vpath][str(global_phy_u) + '_' + str(global_phy_v)][curr_phy_path]
	
			for node_index in range(len(curr_phy_path)-1):
				n1 = curr_phy_path[node_index]
				n2 = curr_phy_path[node_index+1]

				if n1 != n2 and vf_id1 != vf_id2:
					bw_on_phy_links[n1,n2] += bandwidth_requests_between_virtual_nodes[vf_id1_number,vf_id2_number]
					if bandwidth_requests_between_virtual_nodes[vf_id1_number,vf_id2_number] > 0:
						traversed_links.append([n1,n2])

			# case 4 
			intermediate_peerings_paths = curr_peering_path[1:-1]
			#print(intermediate_peerings_paths)
			for intermediate_peering_node_index in range(len(intermediate_peerings_paths)-1):
				n1 = intermediate_peerings_paths[intermediate_peering_node_index]
				n2 = intermediate_peerings_paths[intermediate_peering_node_index+1]

				owner_isp_n1 = 'ISP_' + str(info_per_physical_node[n1]['ISP'])
				owner_isp_n2 = 'ISP_' + str(info_per_physical_node[n2]['ISP'])

				curr_phy_path = current_states[owner_isp_n1]['v_path'][curr_vpath][str(n1) + '_' + str(n2)]
				curr_phy_path = np.where(curr_phy_path > 0)[0][0]
				curr_phy_path = overall_mapping_dict[owner_isp_n1]['v_path'][curr_vpath][str(n1) + '_' + str(n2)][curr_phy_path]
	
				for node_index in range(len(curr_phy_path)-1):
					n1 = curr_phy_path[node_index]
					n2 = curr_phy_path[node_index+1]

					if n1 != n2 and vf_id1 != vf_id2:
						bw_on_phy_links[n1,n2] += bandwidth_requests_between_virtual_nodes[vf_id1_number,vf_id2_number]
						if bandwidth_requests_between_virtual_nodes[vf_id1_number,vf_id2_number] > 0:
							traversed_links.append([n1,n2])

	cost_of_all_vpaths_per_isp = {}
	for curr_vpath in current_states['orchestrator']['v_path']:

		vfs = curr_vpath.split("_")
		vf_id1 = 'vf_' + str(vfs[1])
		vf_id2 = 'vf_' + str(vfs[3])

		vf_id1_number = int(vfs[1])
		vf_id2_number = int(vfs[3])

		owner_isp_vf_id1 = current_states['orchestrator']['vf'][vf_id1]
		owner_isp_vf_id1 = np.where(owner_isp_vf_id1>0)[0][0]
		owner_isp_vf_id1 = overall_mapping_dict['orchestrator']['vf'][vf_id1][owner_isp_vf_id1]

		owner_isp_vf_id2 = current_states['orchestrator']['vf'][vf_id2]
		owner_isp_vf_id2 = np.where(owner_isp_vf_id2>0)[0][0]
		owner_isp_vf_id2 = overall_mapping_dict['orchestrator']['vf'][vf_id2][owner_isp_vf_id2]

		curr_owners = owner_isp_vf_id1 + '_' + owner_isp_vf_id2

		curr_peering_path = current_states['orchestrator']['v_path'][curr_vpath][curr_owners]
		curr_peering_path = np.where(curr_peering_path > 0)[0][0]
		curr_peering_path = overall_mapping_dict['orchestrator']['v_path'][curr_vpath][curr_owners][curr_peering_path]

		cost_of_traversing_peerings = 0 
		if len(curr_peering_path) > 0 and 'isp' not in curr_peering_path:
			for index__ in range(len(curr_peering_path)-1):
				n1 = curr_peering_path[index__]
				n2 = curr_peering_path[index__+1]
				traversed_links.append([n1,n2])

				cost_of_traversing_peerings += cost_per_link_matrix[n1,n2]*bandwidth_requests_between_virtual_nodes[vf_id1_number,vf_id2_number]

		if curr_peering_path == 'same_isp':
			# case 1
			global_phy_u = current_states[owner_isp_vf_id1]['vf'][vf_id1]
			global_phy_u = np.where(global_phy_u>0)[0][0]
			global_phy_u = overall_mapping_dict[owner_isp_vf_id1]['vf'][vf_id1][global_phy_u]

			global_phy_v = current_states[owner_isp_vf_id2]['vf'][vf_id2]
			global_phy_v = np.where(global_phy_v>0)[0][0]
			global_phy_v = overall_mapping_dict[owner_isp_vf_id2]['vf'][vf_id2][global_phy_v]

			owner_isp = owner_isp_vf_id1
			curr_phy_path = current_states[owner_isp]['v_path'][curr_vpath][str(global_phy_u) + '_' + str(global_phy_v)]
			curr_phy_path = np.where(curr_phy_path > 0)[0][0]
			curr_phy_path = overall_mapping_dict[owner_isp]['v_path'][curr_vpath][str(global_phy_u) + '_' + str(global_phy_v)][curr_phy_path]

			if owner_isp not in cost_of_all_vpaths_per_isp:
				cost_of_all_vpaths_per_isp[owner_isp] = {}
			if curr_vpath not in cost_of_all_vpaths_per_isp[owner_isp]:
				cost_of_all_vpaths_per_isp[owner_isp][curr_vpath] = 0 

			for node_index in range(len(curr_phy_path)-1):
				n1 = curr_phy_path[node_index]
				n2 = curr_phy_path[node_index+1]

				if n1 != n2 and vf_id1 != vf_id2:
					cost_of_all_vpaths_per_isp[owner_isp][curr_vpath] += cost_per_link_matrix[n1,n2]*bandwidth_requests_between_virtual_nodes[vf_id1_number,vf_id2_number]

					if bw_on_phy_links[n1,n2] > capacity_of_all_links[n1,n2]:
						pass#print("links' capacity violated")
						cost_of_all_vpaths_per_isp[owner_isp][curr_vpath] += infinity_value						

		else:
			# case 2 
			source_peering_node = curr_peering_path[0]
			global_phy_v = source_peering_node

			global_phy_u = current_states[owner_isp_vf_id1]['vf'][vf_id1]
			global_phy_u = np.where(global_phy_u>0)[0][0]
			global_phy_u = overall_mapping_dict[owner_isp_vf_id1]['vf'][vf_id1][global_phy_u]

			curr_phy_path = current_states[owner_isp_vf_id1]['v_path'][curr_vpath][str(global_phy_u) + '_' + str(global_phy_v)]
			curr_phy_path = np.where(curr_phy_path > 0)[0][0]
			curr_phy_path = overall_mapping_dict[owner_isp_vf_id1]['v_path'][curr_vpath][str(global_phy_u) + '_' + str(global_phy_v)][curr_phy_path]
	
			if owner_isp_vf_id1 not in cost_of_all_vpaths_per_isp:
				cost_of_all_vpaths_per_isp[owner_isp_vf_id1] = {}
			if curr_vpath not in cost_of_all_vpaths_per_isp[owner_isp_vf_id1]:
				cost_of_all_vpaths_per_isp[owner_isp_vf_id1][curr_vpath] = 0 

			for node_index in range(len(curr_phy_path)-1):
				n1 = curr_phy_path[node_index]
				n2 = curr_phy_path[node_index+1]

				if n1 != n2 and vf_id1 != vf_id2:
					cost_of_all_vpaths_per_isp[owner_isp_vf_id1][curr_vpath] += cost_per_link_matrix[n1,n2]*bandwidth_requests_between_virtual_nodes[vf_id1_number,vf_id2_number]

					if bw_on_phy_links[n1,n2] > capacity_of_all_links[n1,n2]:
						cost_of_all_vpaths_per_isp[owner_isp_vf_id1][curr_vpath] += infinity_value	
						print("links' capacity violated")
	

			# case 3
			source_peering_node = curr_peering_path[-1]
			global_phy_u = source_peering_node

			global_phy_v= current_states[owner_isp_vf_id2]['vf'][vf_id2]
			global_phy_v = np.where(global_phy_v>0)[0][0]
			global_phy_v = overall_mapping_dict[owner_isp_vf_id2]['vf'][vf_id2][global_phy_v]

			curr_phy_path = current_states[owner_isp_vf_id2]['v_path'][curr_vpath][str(global_phy_u) + '_' + str(global_phy_v)]
			curr_phy_path = np.where(curr_phy_path > 0)[0][0]
			curr_phy_path = overall_mapping_dict[owner_isp_vf_id2]['v_path'][curr_vpath][str(global_phy_u) + '_' + str(global_phy_v)][curr_phy_path]
	
			if owner_isp_vf_id2 not in cost_of_all_vpaths_per_isp:
				cost_of_all_vpaths_per_isp[owner_isp_vf_id2] = {}
			if curr_vpath not in cost_of_all_vpaths_per_isp[owner_isp_vf_id2]:
				cost_of_all_vpaths_per_isp[owner_isp_vf_id2][curr_vpath] = 0 

			for node_index in range(len(curr_phy_path)-1):
				n1 = curr_phy_path[node_index]
				n2 = curr_phy_path[node_index+1]

				if n1 != n2 and vf_id1 != vf_id2:
					cost_of_all_vpaths_per_isp[owner_isp_vf_id2][curr_vpath] += cost_per_link_matrix[n1,n2]*bandwidth_requests_between_virtual_nodes[vf_id1_number,vf_id2_number]

					if bw_on_phy_links[n1,n2] > capacity_of_all_links[n1,n2]:
						cost_of_all_vpaths_per_isp[owner_isp_vf_id2][curr_vpath] += infinity_value	
						print("links' capacity violated")
	
			# case 4 
			intermediate_peerings_paths = curr_peering_path[1:-1]
			#print(intermediate_peerings_paths)
			for intermediate_peering_node_index in range(len(intermediate_peerings_paths)-1):
				n1 = intermediate_peerings_paths[intermediate_peering_node_index]
				n2 = intermediate_peerings_paths[intermediate_peering_node_index+1]

				owner_isp_n1 = 'ISP_' + str(info_per_physical_node[n1]['ISP'])
				owner_isp_n2 = 'ISP_' + str(info_per_physical_node[n2]['ISP'])

				curr_phy_path = current_states[owner_isp_n1]['v_path'][curr_vpath][str(n1) + '_' + str(n2)]
				curr_phy_path = np.where(curr_phy_path > 0)[0][0]
				curr_phy_path = overall_mapping_dict[owner_isp_n1]['v_path'][curr_vpath][str(n1) + '_' + str(n2)][curr_phy_path]
	
				#print(curr_phy_path)
				if owner_isp_n1 not in cost_of_all_vpaths_per_isp:
					cost_of_all_vpaths_per_isp[owner_isp_n1] = {}
				if curr_vpath not in cost_of_all_vpaths_per_isp[owner_isp_n1]:
					cost_of_all_vpaths_per_isp[owner_isp_n1][curr_vpath] = 0 

				for node_index in range(len(curr_phy_path)-1):
					n1 = curr_phy_path[node_index]
					n2 = curr_phy_path[node_index+1]

					if n1 != n2 and vf_id1 != vf_id2:
						cost_of_all_vpaths_per_isp[owner_isp_n1][curr_vpath] += cost_per_link_matrix[n1,n2]*bandwidth_requests_between_virtual_nodes[vf_id1_number,vf_id2_number]

						if bw_on_phy_links[n1,n2] > capacity_of_all_links[n1,n2]:
							cost_of_all_vpaths_per_isp[owner_isp_n1][curr_vpath] += infinity_value	
							print("links' capacity violated")
	

	cost_of_allocating_vpaths = 0
	for isp in cost_of_all_vpaths_per_isp:
		for curr_vpath in cost_of_all_vpaths_per_isp[isp]:
			cost_of_allocating_vpaths += cost_of_all_vpaths_per_isp[isp][curr_vpath]

	cost_of_allocating_vpaths += cost_of_traversing_peerings

	if cost_of_allocating_vpaths < 1:
		pass#print(cost_of_all_vpaths_per_isp,'cost of allocating vpaths ')

	traversed_links_ = []
	for el in traversed_links:
		if el not in traversed_links_:
			traversed_links_.append(el)

	traversed_links = traversed_links_

	return cost_of_allocating_vpaths, traversed_links


def compute_links_traversed(current_states,overall_mapping_dict,cost_per_link_matrix,bandwidth_requests_between_virtual_nodes,info_per_physical_node,capacity_of_all_links,infinity_value):

	links_traversed  = []
	for curr_vpath in current_states['orchestrator']['v_path']:

		vfs = curr_vpath.split("_")
		vf_id1 = 'vf_' + str(vfs[1])
		vf_id2 = 'vf_' + str(vfs[3])

		vf_id1_number = int(vfs[1])
		vf_id2_number = int(vfs[3])

		owner_isp_vf_id1 = current_states['orchestrator']['vf'][vf_id1]
		owner_isp_vf_id1 = np.where(owner_isp_vf_id1>0)[0][0]
		owner_isp_vf_id1 = overall_mapping_dict['orchestrator']['vf'][vf_id1][owner_isp_vf_id1]

		owner_isp_vf_id2 = current_states['orchestrator']['vf'][vf_id2]
		owner_isp_vf_id2 = np.where(owner_isp_vf_id2>0)[0][0]
		owner_isp_vf_id2 = overall_mapping_dict['orchestrator']['vf'][vf_id2][owner_isp_vf_id2]

		curr_owners = owner_isp_vf_id1 + '_' + owner_isp_vf_id2

		curr_peering_path = current_states['orchestrator']['v_path'][curr_vpath][curr_owners]
		curr_peering_path = np.where(curr_peering_path > 0)[0][0]
		curr_peering_path = overall_mapping_dict['orchestrator']['v_path'][curr_vpath][curr_owners][curr_peering_path]

		if 'isp' not in curr_peering_path:
			for index_ in range(len(curr_peering_path)-1):
				n1 = curr_peering_path[index_]
				n2 = curr_peering_path[index_+1]
				links_traversed.append(str(n1) + '_' + str(n2))

		if curr_peering_path == 'same_isp':
			# case 1
			global_phy_u = current_states[owner_isp_vf_id1]['vf'][vf_id1]
			global_phy_u = np.where(global_phy_u>0)[0][0]
			global_phy_u = overall_mapping_dict[owner_isp_vf_id1]['vf'][vf_id1][global_phy_u]

			global_phy_v = current_states[owner_isp_vf_id2]['vf'][vf_id2]
			global_phy_v = np.where(global_phy_v>0)[0][0]
			global_phy_v = overall_mapping_dict[owner_isp_vf_id2]['vf'][vf_id2][global_phy_v]

			owner_isp = owner_isp_vf_id1
			curr_phy_path = current_states[owner_isp]['v_path'][curr_vpath][str(global_phy_u) + '_' + str(global_phy_v)]
			curr_phy_path = np.where(curr_phy_path > 0)[0][0]
			curr_phy_path = overall_mapping_dict[owner_isp]['v_path'][curr_vpath][str(global_phy_u) + '_' + str(global_phy_v)][curr_phy_path]

			for node_index in range(len(curr_phy_path)-1):
				n1 = curr_phy_path[node_index]
				n2 = curr_phy_path[node_index+1]
				
				if n1 != n2:
					links_traversed.append(str(n1) + '_' + str(n2))

		else:
			
			# case 2 
			source_peering_node = curr_peering_path[0]
			global_phy_v = source_peering_node

			global_phy_u = current_states[owner_isp_vf_id1]['vf'][vf_id1]
			global_phy_u = np.where(global_phy_u>0)[0][0]
			global_phy_u = overall_mapping_dict[owner_isp_vf_id1]['vf'][vf_id1][global_phy_u]

			curr_phy_path = current_states[owner_isp_vf_id1]['v_path'][curr_vpath][str(global_phy_u) + '_' + str(global_phy_v)]
			curr_phy_path = np.where(curr_phy_path > 0)[0][0]
			curr_phy_path = overall_mapping_dict[owner_isp_vf_id1]['v_path'][curr_vpath][str(global_phy_u) + '_' + str(global_phy_v)][curr_phy_path]

			print(curr_phy_path,source_peering_node)
			for node_index in range(len(curr_phy_path)-1):
				n1 = curr_phy_path[node_index]
				n2 = curr_phy_path[node_index+1]
				if n1 != n2:
					links_traversed.append(str(n1) + '_' + str(n2))
			
			# case 3
			source_peering_node = curr_peering_path[-1]
			global_phy_u = source_peering_node

			global_phy_v= current_states[owner_isp_vf_id2]['vf'][vf_id2]
			global_phy_v = np.where(global_phy_v>0)[0][0]
			global_phy_v = overall_mapping_dict[owner_isp_vf_id2]['vf'][vf_id2][global_phy_v]

			curr_phy_path = current_states[owner_isp_vf_id2]['v_path'][curr_vpath][str(global_phy_u) + '_' + str(global_phy_v)]
			curr_phy_path = np.where(curr_phy_path > 0)[0][0]
			curr_phy_path = overall_mapping_dict[owner_isp_vf_id2]['v_path'][curr_vpath][str(global_phy_u) + '_' + str(global_phy_v)][curr_phy_path]
	
			#print(source_peering_node,curr_phy_path)
			for node_index in range(len(curr_phy_path)-1):
				n1 = curr_phy_path[node_index]
				n2 = curr_phy_path[node_index+1]

				if n1 != n2:
					links_traversed.append(str(n1) + '_' + str(n2))
			
			# case 4 
			intermediate_peerings_paths = curr_peering_path[1:-1]
			#print(intermediate_peerings_paths)
			for intermediate_peering_node_index in range(len(intermediate_peerings_paths)-1):
				n1 = intermediate_peerings_paths[intermediate_peering_node_index]
				n2 = intermediate_peerings_paths[intermediate_peering_node_index+1]

				owner_isp_n1 = 'ISP_' + str(info_per_physical_node[n1]['ISP'])
				owner_isp_n2 = 'ISP_' + str(info_per_physical_node[n2]['ISP'])

				curr_phy_path = current_states[owner_isp_n1]['v_path'][curr_vpath][str(n1) + '_' + str(n2)]
				curr_phy_path = np.where(curr_phy_path > 0)[0][0]
				curr_phy_path = overall_mapping_dict[owner_isp_n1]['v_path'][curr_vpath][str(n1) + '_' + str(n2)][curr_phy_path]
	
				for node_index in range(len(curr_phy_path)-1):
					n1 = curr_phy_path[node_index]
					n2 = curr_phy_path[node_index+1]

					if n1 != n2:
						links_traversed.append(str(n1) + '_' + str(n2))

	links_traversed = list(set(links_traversed))

	return links_traversed

def learning_function_(current_states, Q_dict, overall_mapping_dict,all_results_RL, round_external_node, round_external_path, round_internal_node,round_internal_link,all_adjacency_matrices,all_isps,virtual_graph,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix,min_cutoff,max_cutoff, num_of_possible_actions_internal_node, num_of_possible_actions_internal_path, num_of_possible_actions_external_node, num_of_possible_actions_external_path, n_epochs,eps_external_node,eps_external_path,eps_internal_node,eps_internal_path,learning_rate_internal_node, learning_rate_external_node,learning_rate_internal_path,learning_rate_external_path,discount_factor_internal_node, discount_factor_external_node, discount_factor_internal_path, discount_factor_external_path,bandwidth_requests_between_virtual_nodes,cost_per_link_matrix,cost_per_virtual_node,demand_per_virtual_node,dict_of_capacities_of_physical_nodes,infinity_value,capacity_of_all_links,patience,prob_node_capacity_verification):
				
	#print(demand_per_virtual_node)
	#print("...")
	#print(cost_per_virtual_node)
	#print("expected minimum")
	#print(np.sum(cost_per_virtual_node,axis=1))
	#print(np.min(np.sum(cost_per_virtual_node,axis=1)))
	print(np.min(feasibility_matrix,axis=0),'aa',feasibility_matrix.shape)

	#current_states, Q_dict, overall_mapping_dict = get_initial_states_and_Q(all_adjacency_matrices,all_isps,virtual_graph,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix,min_cutoff,max_cutoff, num_of_possible_actions_internal_node, num_of_possible_actions_internal_path, num_of_possible_actions_external_node, num_of_possible_actions_external_path)

	all_costs = []
	cost_of_all_vpaths_per_isp = {}
	cost_of_all_vfs_per_isp = {}

	minimum_cost = 10**10
	minimum_cost_dict = {}
	minimum_cost_dict['vf'] = 10**10
	minimum_cost_dict['vpath'] = 10**10

	tot_time = {}
	tot_time['orch'] = []
	tot_time['isp'] = []

	N_isps = len(current_states['orchestrator']['vf'].keys())

	#for iteration in range(n_epochs):
	iteration = 0 
	while 1:
		pass
		#print(iteration)
		cost_per_iteration = 0 

		# take actions and change states
		if iteration%round_external_node == 0:
			t_in = time.time()
			current_states, decisions_orch_vfs = choose_actions_Q_orch_vfs(current_states, Q_dict, overall_mapping_dict, eps_external_node)		
			t_fin = time.time()
			tot_time['orch'].append(t_fin-t_in)

		if iteration%round_external_path == 0:
			t_in = time.time()
			current_states, decisions_orch_vpath = choose_actions_Q_orch_vpaths(current_states, Q_dict, overall_mapping_dict, eps_external_path,info_per_physical_node,bandwidth_requests_between_virtual_nodes)
			t_fin = time.time()
			tot_time['orch'].append(t_fin-t_in)
		
		if iteration%round_internal_node == 0:
			t_in = time.time()
			current_states, decisions_isp_vfs = choose_actions_Q_isp_vfs(current_states, Q_dict, overall_mapping_dict,eps_internal_node)
			t_fin = time.time()
			tot_time['isp'].append((t_fin-t_in)/float(N_isps))

		#if iteration >= 0:
		if iteration < 1 or iteration%round_internal_link == 0:	
			t_in = time.time()	
			current_states, decisions_isp_vpath = choose_actions_Q_isp_vpath(current_states, Q_dict, overall_mapping_dict, eps_internal_path,info_per_physical_node,bandwidth_requests_between_virtual_nodes)
			t_fin = time.time()
			tot_time['isp'].append((t_fin-t_in)/float(N_isps))

			# update the Qs 
			t_in = time.time()
			Q_dict, cost_isp_vpaths, cost_of_all_vpaths_per_isp = update_Q_isp_vpaths(Q_dict,decisions_isp_vpath,learning_rate_internal_path,discount_factor_internal_path,overall_mapping_dict,bandwidth_requests_between_virtual_nodes,cost_per_link_matrix,capacity_of_all_links,infinity_value)
			t_fin = time.time()
			#print('isp vpath',t_fin - t_in)
			tot_time['isp'].append((t_fin-t_in)/float(N_isps))

		if iteration < 1 or iteration%round_internal_node == round_internal_node - 1:
			t_in = time.time()
			Q_dict, cost_of_all_vfs_per_isp_embedding_plus_vpath = update_Q_isp_vfs(Q_dict,decisions_isp_vpath,decisions_isp_vfs,cost_of_all_vpaths_per_isp,cost_of_all_vfs_per_isp,learning_rate_internal_node,discount_factor_internal_node,overall_mapping_dict,cost_per_virtual_node,demand_per_virtual_node,dict_of_capacities_of_physical_nodes,infinity_value,feasibility_matrix,bandwidth_requests_between_virtual_nodes,prob_node_capacity_verification)
			t_fin = time.time()
			#print('isp vf',t_fin - t_in)
			tot_time['isp'].append((t_fin-t_in)/float(N_isps))

			#print(Q_dict['ISP_0']['vf']['vf_0'])

		if iteration < 1 or iteration%round_external_path == round_external_path -1:
			t_in = time.time()
			Q_dict = update_Q_orch_vpaths(Q_dict, decisions_orch_vpath, cost_of_all_vpaths_per_isp,learning_rate_external_path, discount_factor_external_path,bandwidth_requests_between_virtual_nodes)
			t_fin = time.time()
			#print('orch vpath',t_fin - t_in)
			tot_time['orch'].append((t_fin-t_in))

		if iteration < 1 or iteration%round_external_node == round_external_node -1:
			t_in = time.time()
			Q_dict = update_Q_orch_vfs(Q_dict, decisions_orch_vfs, cost_of_all_vfs_per_isp_embedding_plus_vpath,learning_rate_external_node, discount_factor_external_node)			
			t_fin = time.time()
			#print('orch vf',t_fin - t_in)
			tot_time['orch'].append((t_fin-t_in))

			#print(Q_dict['orchestrator']['vf']['vf_0'])

		cost_of_embedding, vf_per_phy_nodes = compute_embedding_cost(current_states,overall_mapping_dict,cost_per_virtual_node,demand_per_virtual_node,dict_of_capacities_of_physical_nodes,infinity_value,feasibility_matrix)
		cost_of_allocating_vpaths, links_traversed = compute_vpath_cost(current_states,overall_mapping_dict,cost_per_link_matrix,bandwidth_requests_between_virtual_nodes,info_per_physical_node,capacity_of_all_links,infinity_value)

		cost_per_iteration = cost_of_embedding + cost_of_allocating_vpaths

		#print(cost_of_embedding,cost_of_allocating_vpaths,cost_per_iteration)
		if cost_per_iteration < minimum_cost:
			minimum_cost = cost_per_iteration 
			minimum_cost_dict['vf'] = cost_of_embedding 
			minimum_cost_dict['vpath'] = cost_of_allocating_vpaths
			minimum_cost_dict['vf_per_phy_nodes'] = vf_per_phy_nodes
			best_states = current_states 
			best_links_traversed = links_traversed

			consecutive_non_improvements = 0
		else:
			consecutive_non_improvements += 1 

		all_results_RL.append(minimum_cost)
		#all_results_RL.append(cost_per_iteration)

		if iteration%1000 == 0: #20000
			print(iteration,cost_per_iteration,'minimum',minimum_cost)		

		if iteration > 0:
			all_costs.append(cost_per_iteration)

		iteration += 1 

		if consecutive_non_improvements > patience:
			min_cost = all_results_RL[-1]
			for el in range(iteration,n_epochs):
				all_results_RL.append(min_cost)

			break 


	#links_traversed = compute_links_traversed(best_states,overall_mapping_dict,cost_per_link_matrix,bandwidth_requests_between_virtual_nodes,info_per_physical_node,capacity_of_all_links,infinity_value)
	#print(links_traversed_by_all_vpath)

	links_traversed_ = []
	for edge in best_links_traversed:
		
		e1 = (int(edge[0]),int(edge[1]))
		e2 = (int(edge[1]),int(edge[0]))

		if e1 not in links_traversed_:
			links_traversed_.append(e1)
		if e2 not in links_traversed_:
			links_traversed_.append(e2)

	peering_nodes = np.arange(all_adjacency_matrices['peering_network'].shape[0])
	peering_nodes = [dict_local_to_peering[peering_el] for peering_el in peering_nodes]

	multidomain_adj_matrix = all_adjacency_matrices['multidomain']
	G = nx.from_numpy_matrix(multidomain_adj_matrix)
	all_edges = G.edges

	links_non_traversed_ = []
	for e in all_edges:
		if e not in links_traversed_:
			links_non_traversed_.append(e)

	
	pos = nx.spring_layout(G)  # positions for all nodes
	non_peering_nodes = [el for el in G.nodes if el not in peering_nodes]
	# nodes
	nx.draw_networkx_nodes(G, pos,
	                       nodelist=non_peering_nodes,
	                       node_color='black',
	                       node_size=50,
	                       alpha=0.8)

	nx.draw_networkx_nodes(G, pos,
	                       nodelist=peering_nodes,
	                       node_color='black',
	                       node_size=50,
	                       alpha=0.8)

	# edges
	
	nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
	nx.draw_networkx_edges(G, pos,
	                       edgelist=links_traversed_,
	                       width=6, alpha=0.5, edge_color='r')
	nx.draw_networkx_edges(G, pos,
	                       edgelist=links_non_traversed_,
	                       width=2, alpha=0.5, edge_color='b')
	

	#plt.show()
		
	#print(cost_per_virtual_node)
	print('min cost dict',minimum_cost_dict)
	print(minimum_cost)
	print(best_links_traversed)


	plt.plot(all_costs)
	#plt.show()
	

	return minimum_cost, all_results_RL, tot_time

def learning_function_private(current_states, Q_dict, overall_mapping_dict,all_results_RL, round_external_node, round_external_path, round_internal_node,round_internal_link,all_adjacency_matrices,all_isps,virtual_graph,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix,min_cutoff,max_cutoff, num_of_possible_actions_internal_node, num_of_possible_actions_internal_path, num_of_possible_actions_external_node, num_of_possible_actions_external_path, n_epochs,eps_external_node,eps_external_path,eps_internal_node,eps_internal_path,learning_rate_internal_node, learning_rate_external_node,learning_rate_internal_path,learning_rate_external_path,discount_factor_internal_node, discount_factor_external_node, discount_factor_internal_path, discount_factor_external_path,bandwidth_requests_between_virtual_nodes,cost_per_link_matrix,cost_per_virtual_node,demand_per_virtual_node,dict_of_capacities_of_physical_nodes,infinity_value,capacity_of_all_links,patience,prob_node_capacity_verification):
				
	#print(demand_per_virtual_node)
	#print("...")
	#print(cost_per_virtual_node)
	#print("expected minimum")
	#print(np.sum(cost_per_virtual_node,axis=1))
	#print(np.min(np.sum(cost_per_virtual_node,axis=1)))
	print(np.min(feasibility_matrix,axis=0),'aa',feasibility_matrix.shape)

	#current_states, Q_dict, overall_mapping_dict = get_initial_states_and_Q(all_adjacency_matrices,all_isps,virtual_graph,dict_local_to_peering,info_per_physical_node,dict_local_to_global_node,feasibility_matrix,min_cutoff,max_cutoff, num_of_possible_actions_internal_node, num_of_possible_actions_internal_path, num_of_possible_actions_external_node, num_of_possible_actions_external_path)

	all_costs = []
	cost_of_all_vpaths_per_isp = {}
	cost_of_all_vfs_per_isp = {}

	minimum_cost = 10**10
	minimum_cost_dict = {}
	minimum_cost_dict['vf'] = 10**10
	minimum_cost_dict['vpath'] = 10**10

	tot_time = {}
	tot_time['orch'] = []
	tot_time['isp'] = []

	N_isps = len(current_states['orchestrator']['vf'].keys())

	#for iteration in range(n_epochs):
	iteration = 0 
	while 1:
		pass
		#print(iteration)
		cost_per_iteration = 0 

		# take actions and change states
		if iteration%round_external_node == 0:
			t_in = time.time()
			current_states, decisions_orch_vfs = choose_actions_Q_orch_vfs(current_states, Q_dict, overall_mapping_dict, eps_external_node)		
			t_fin = time.time()
			tot_time['orch'].append(t_fin-t_in)

		if iteration%round_external_path == 0:
			t_in = time.time()
			current_states, decisions_orch_vpath = choose_actions_Q_orch_vpaths(current_states, Q_dict, overall_mapping_dict, eps_external_path,info_per_physical_node,bandwidth_requests_between_virtual_nodes)
			t_fin = time.time()
			tot_time['orch'].append(t_fin-t_in)
		
		if iteration%round_internal_node == 0:
			t_in = time.time()
			current_states, decisions_isp_vfs = choose_actions_Q_isp_vfs(current_states, Q_dict, overall_mapping_dict,eps_internal_node)
			t_fin = time.time()
			tot_time['isp'].append((t_fin-t_in)/float(N_isps))

		#if iteration >= 0:
		if iteration < 1 or iteration%round_internal_link == 0:	
			t_in = time.time()	
			current_states, decisions_isp_vpath = choose_actions_Q_isp_vpath(current_states, Q_dict, overall_mapping_dict, eps_internal_path,info_per_physical_node,bandwidth_requests_between_virtual_nodes)
			t_fin = time.time()
			tot_time['isp'].append((t_fin-t_in)/float(N_isps))

			# update the Qs 
			t_in = time.time()
			Q_dict, cost_isp_vpaths, cost_of_all_vpaths_per_isp = update_Q_isp_vpaths_private(Q_dict,decisions_isp_vpath,learning_rate_internal_path,discount_factor_internal_path,overall_mapping_dict,bandwidth_requests_between_virtual_nodes,cost_per_link_matrix,capacity_of_all_links,infinity_value)
			t_fin = time.time()
			#print('isp vpath',t_fin - t_in)
			tot_time['isp'].append((t_fin-t_in)/float(N_isps))

		if iteration < 1 or iteration%round_internal_node == round_internal_node - 1:
			t_in = time.time()
			Q_dict, cost_of_all_vfs_per_isp_embedding_plus_vpath = update_Q_isp_vfs_private(Q_dict,decisions_isp_vpath,decisions_isp_vfs,cost_of_all_vpaths_per_isp,cost_of_all_vfs_per_isp,learning_rate_internal_node,discount_factor_internal_node,overall_mapping_dict,cost_per_virtual_node,demand_per_virtual_node,dict_of_capacities_of_physical_nodes,infinity_value,feasibility_matrix,bandwidth_requests_between_virtual_nodes,prob_node_capacity_verification)
			t_fin = time.time()
			#print('isp vf',t_fin - t_in)
			tot_time['isp'].append((t_fin-t_in)/float(N_isps))

			#print(Q_dict['ISP_0']['vf']['vf_0'])

		if iteration < 1 or iteration%round_external_path == round_external_path -1:
			t_in = time.time()
			Q_dict = update_Q_orch_vpaths_private(Q_dict, decisions_orch_vpath, cost_of_all_vpaths_per_isp,learning_rate_external_path, discount_factor_external_path,bandwidth_requests_between_virtual_nodes)
			t_fin = time.time()
			#print('orch vpath',t_fin - t_in)
			tot_time['orch'].append((t_fin-t_in))

		if iteration < 1 or iteration%round_external_node == round_external_node -1:
			t_in = time.time()
			Q_dict = update_Q_orch_vfs_private(Q_dict, decisions_orch_vfs, cost_of_all_vfs_per_isp_embedding_plus_vpath,learning_rate_external_node, discount_factor_external_node)			
			t_fin = time.time()
			#print('orch vf',t_fin - t_in)
			tot_time['orch'].append((t_fin-t_in))

			#print(Q_dict['orchestrator']['vf']['vf_0'])

		cost_of_embedding, vf_per_phy_nodes = compute_embedding_cost(current_states,overall_mapping_dict,cost_per_virtual_node,demand_per_virtual_node,dict_of_capacities_of_physical_nodes,infinity_value,feasibility_matrix)
		cost_of_allocating_vpaths, links_traversed = compute_vpath_cost(current_states,overall_mapping_dict,cost_per_link_matrix,bandwidth_requests_between_virtual_nodes,info_per_physical_node,capacity_of_all_links,infinity_value)

		cost_per_iteration = cost_of_embedding + cost_of_allocating_vpaths

		#print(cost_of_embedding,cost_of_allocating_vpaths,cost_per_iteration)
		if cost_per_iteration < minimum_cost:
			minimum_cost = cost_per_iteration 
			minimum_cost_dict['vf'] = cost_of_embedding 
			minimum_cost_dict['vpath'] = cost_of_allocating_vpaths
			minimum_cost_dict['vf_per_phy_nodes'] = vf_per_phy_nodes
			best_states = current_states 
			best_links_traversed = links_traversed

			consecutive_non_improvements = 0
		else:
			consecutive_non_improvements += 1 

		all_results_RL.append(minimum_cost)
		#all_results_RL.append(cost_per_iteration)

		if iteration%1000 == 0: #20000
			print(iteration,cost_per_iteration,'minimum',minimum_cost)		

		if iteration > 0:
			all_costs.append(cost_per_iteration)

		iteration += 1 

		if consecutive_non_improvements > patience:
			min_cost = all_results_RL[-1]
			for el in range(iteration,n_epochs):
				all_results_RL.append(min_cost)

			break 


	#links_traversed = compute_links_traversed(best_states,overall_mapping_dict,cost_per_link_matrix,bandwidth_requests_between_virtual_nodes,info_per_physical_node,capacity_of_all_links,infinity_value)
	#print(links_traversed_by_all_vpath)

	links_traversed_ = []
	for edge in best_links_traversed:
		
		e1 = (int(edge[0]),int(edge[1]))
		e2 = (int(edge[1]),int(edge[0]))

		if e1 not in links_traversed_:
			links_traversed_.append(e1)
		if e2 not in links_traversed_:
			links_traversed_.append(e2)

	peering_nodes = np.arange(all_adjacency_matrices['peering_network'].shape[0])
	peering_nodes = [dict_local_to_peering[peering_el] for peering_el in peering_nodes]

	multidomain_adj_matrix = all_adjacency_matrices['multidomain']
	G = nx.from_numpy_matrix(multidomain_adj_matrix)
	all_edges = G.edges

	links_non_traversed_ = []
	for e in all_edges:
		if e not in links_traversed_:
			links_non_traversed_.append(e)

	
	pos = nx.spring_layout(G)  # positions for all nodes
	non_peering_nodes = [el for el in G.nodes if el not in peering_nodes]
	# nodes
	nx.draw_networkx_nodes(G, pos,
	                       nodelist=non_peering_nodes,
	                       node_color='black',
	                       node_size=50,
	                       alpha=0.8)

	nx.draw_networkx_nodes(G, pos,
	                       nodelist=peering_nodes,
	                       node_color='black',
	                       node_size=50,
	                       alpha=0.8)

	# edges
	
	nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
	nx.draw_networkx_edges(G, pos,
	                       edgelist=links_traversed_,
	                       width=6, alpha=0.5, edge_color='r')
	nx.draw_networkx_edges(G, pos,
	                       edgelist=links_non_traversed_,
	                       width=2, alpha=0.5, edge_color='b')
	

	#plt.show()
		
	#print(cost_per_virtual_node)
	print('min cost dict',minimum_cost_dict)
	print(minimum_cost)
	print(best_links_traversed)


	plt.plot(all_costs)
	#plt.show()
	

	return minimum_cost, all_results_RL, tot_time