from config import *
import numpy as np 
import networkx as nx 
from matplotlib import pyplot as plt 
from pulp import *
import copy

def draw_graph(all_adjacency_matrices,phy_node_per_vf_node,traversed_links):

	multidomain_adj = all_adjacency_matrices['multidomain']
	G=nx.from_numpy_matrix(multidomain_adj)				
	pos = nx.spring_layout(G)  # positions for all nodes
	nx.draw_networkx_nodes(G, pos,
	                       nodelist=G.nodes,
	                       node_color='black',
	                       node_size=50,
	                       alpha=0.8)

	vf_node_per_phy_node = {}
	for gl_vf in phy_node_per_vf_node:
		gl_phy = phy_node_per_vf_node[gl_vf]
		vf_node_per_phy_node[gl_phy] = gl_vf

	labels = {}
	for i in range(multidomain_adj.shape[0]):
		#labels[i] = i
		if i in vf_node_per_phy_node:
			labels[i] = vf_node_per_phy_node[i]
		else:
			labels[i] = ''

	nx.draw_networkx_labels(G, pos, labels=labels,font_size=50)

	edges_non_traversed = [el for el in G.edges if el not in traversed_links]
	edges_travesed = traversed_links

	# edges
	#nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
	nx.draw_networkx_edges(G, pos,
	                       edgelist=edges_travesed,
	                       width=6, alpha=0.5, edge_color='r')
	nx.draw_networkx_edges(G, pos,
	                       edgelist=edges_non_traversed,
	                       width=2, alpha=0.5, edge_color='b')
	
	plt.show()

def define_virtual_graph(n_virtual_nodes, prob_link,types_of_virtual_nodes):

	adj_matrix = np.zeros([n_virtual_nodes,n_virtual_nodes])
	for node_i in range(n_virtual_nodes):
		for node_j in range(node_i + 1,n_virtual_nodes):
			rv = np.random.binomial(1,prob_link)
			if rv > 0:
				adj_matrix[node_i,node_j] = 1
				adj_matrix[node_j,node_i] = 1

	connected = 1
	iteration = 0 
	repeat = 0
	while connected == 0:
		
		iteration += 1 

		for node_i in range(n_virtual_nodes):
			if np.sum(adj_matrix[node_i]) == 0:
				other_nodes = list(np.arange(n_virtual_nodes))
				other_nodes.remove(node_i)
				other_nodes = np.asarray(other_nodes)

				node_j = np.random.choice(other_nodes)
				adj_matrix[node_i,node_j] = 1

		for node_i in range(n_virtual_nodes):
			for node_j in range(n_virtual_nodes):
				if adj_matrix[node_i,node_j] == 1:
					adj_matrix[node_j,node_i] = 1

		G=nx.from_numpy_matrix(adj_matrix)
		is_connected = nx.is_connected(G)

		if is_connected == 1:
			connected = 1 
		else:
			connected = 0

		if iteration > 100:
			repeat = 1 
			break
		"""
		print("is connected?",connected)
		#G=nx.from_numpy_matrix(adj_isp_i)
		nx.draw(G,pos=nx.spring_layout(G)) # use spring layout
		plt.draw()
		plt.show()
		"""
	dict_virtual_node_types = {}
	for v_node in range(len(adj_matrix)):
		type_ = np.random.choice(types_of_virtual_nodes)
		dict_virtual_node_types[v_node] = type_


	return adj_matrix, dict_virtual_node_types,repeat

def preprocess_data_for_RL(cost_per_virtual_node,info_per_physical_node,number_of_ISPs,all_adjacency_matrices):

	# this module is aimed to make the data generated for the benchmarks consistent with the data needed by the RL 

	cost_per_vnf = {} 
	n_virtual_nodes = cost_per_virtual_node.shape[1]
	n_physical_nodes = cost_per_virtual_node.shape[0]

	for vnf_id in range(n_virtual_nodes):
		if vnf_id not in cost_per_vnf:
			cost_per_vnf[vnf_id] = {}
			for phy_node in range(n_physical_nodes):
				if phy_node not in cost_per_vnf[vnf_id]:
					cost_per_vnf[vnf_id][phy_node] = cost_per_virtual_node[phy_node,vnf_id]

	all_isps = {} 
	N_nodes_dict = {} #this dictionary has key: isp_id , value: number of physical node of that isp 
	for isp_id in range(number_of_ISPs):
		isp_id_ = 'ISP_' + str(isp_id)
		if isp_id_ not in all_isps:
			all_isps[isp_id_] = all_adjacency_matrices[isp_id]

		N_nodes_dict[isp_id_] = all_isps[isp_id_].shape[0]			

	mapping_multidomain_to_singledomain_node = {}
	for phy_node in range(n_physical_nodes):
		corr_local_node = info_per_physical_node[phy_node]['single_domain_node_identifier']
		corr_isp = info_per_physical_node[phy_node]['ISP'] 

		v = (corr_isp,corr_local_node)
		mapping_multidomain_to_singledomain_node[phy_node] = v 

	mapping_singledomain_to_multidomain_node = {}
	for phy_id in mapping_multidomain_to_singledomain_node:
		v = mapping_multidomain_to_singledomain_node[phy_id]
		mapping_singledomain_to_multidomain_node[v] = phy_id 

	return mapping_singledomain_to_multidomain_node, mapping_multidomain_to_singledomain_node, all_isps, N_nodes_dict, cost_per_vnf

def get_feasible_association_vfs_isps(dict_virtual_node_types,feasibile_vfs_types_per_isp):

	where_our_vfs_can_go = {}
	for vf in dict_virtual_node_types:
		where_our_vfs_can_go[vf] = []
		type_of_vf = dict_virtual_node_types[vf]
		for isp__ in feasibile_vfs_types_per_isp:
			if type_of_vf in feasibile_vfs_types_per_isp[isp__]:
				where_our_vfs_can_go[vf].append(isp__)

	return where_our_vfs_can_go

def get_ISP_network(curr_isp, global_position_of_nodes,dict_local_to_global_node,info_per_physical_node,global_node_identifier,ISP_id,nodes,radius,types_of_physical_nodes,alpha,beta):

	repeat = 1
	while repeat:

		x_centroid_of_isp = np.random.uniform(5*10**5)
		y_centroid_of_isp = np.random.uniform(5*10**5)

		position_of_nodes = {}
		for node_id in range(nodes):
			#print(node_id)

			position_of_nodes[node_id] = {}

			l = np.random.uniform(0,radius)
			angle = np.random.uniform(2*np.pi)

			x_node = l*np.cos(angle)
			y_node = l*np.sin(angle)

			position_of_nodes[node_id]['x'] = x_node 
			position_of_nodes[node_id]['y'] = y_node 

		adjacency_matrix = np.zeros([nodes,nodes])
		L = np.sqrt(2)*radius
		for node_i in range(nodes):

			x_node_i = position_of_nodes[node_i]['x']
			y_node_i = position_of_nodes[node_i]['y']

			for node_j in range(node_i+1,nodes):

				x_node_j = position_of_nodes[node_j]['x']
				y_node_j = position_of_nodes[node_j]['y']

				dist_ij = np.sqrt((x_node_i - x_node_j)**2 + (y_node_i - y_node_j)**2)
				prob_link = alpha*np.exp((1/float(beta))*(-dist_ij/float(L)))

				rv = np.random.binomial(1,prob_link)
				if rv > 0:
					adjacency_matrix[node_i,node_j] = 1 
					adjacency_matrix[node_j,node_i] = 1 

		position_of_nodes_ = {} 
		for el in position_of_nodes:
			position_of_nodes_[el] = [position_of_nodes[el]['x'],position_of_nodes[el]['y']]
			position_of_nodes_[el][0] += x_centroid_of_isp 
			position_of_nodes_[el][1] += y_centroid_of_isp

		position_of_nodes = position_of_nodes_


		labels = {} 
		for i in range(adjacency_matrix.shape[0]):
			labels[i] = i 

		"""
		G = nx.from_numpy_matrix(adjacency_matrix)
		nx.draw(G,pos=nx.spring_layout(G),labels=labels)
		plt.draw()
		plt.show()	
		"""
		go = 1 
		while go:
			G = nx.from_numpy_matrix(adjacency_matrix)
			to_remove = list(nx.isolates(G))

			adjacency_matrix = np.delete(adjacency_matrix, to_remove, axis=0)
			adjacency_matrix = np.delete(adjacency_matrix,to_remove, axis=1)
			
			go = 0
			n_to_remove = len(to_remove)
			if n_to_remove > 0:
				go = 1

		connected = nx.is_connected(G)
		repeat = 0
		if not connected:
			repeat = 1

	"""
	labels = {} 
	for i in range(adjacency_matrix.shape[0]):
		labels[i] = i
	G = nx.from_numpy_matrix(adjacency_matrix)
	nx.draw(G,pos=nx.spring_layout(G),labels=labels)
	plt.draw()
	plt.show()	
	"""

	nodes = len(adjacency_matrix)
	for node_id in range(nodes):

		info_per_physical_node[global_node_identifier] = {}
		info_per_physical_node[global_node_identifier]['single_domain_node_identifier'] = node_id  
		info_per_physical_node[global_node_identifier]['ISP'] = ISP_id 
		info_per_physical_node[global_node_identifier]['type_of_physical_node'] = np.random.choice(types_of_physical_nodes)


		dict_local_to_global_node['ISP_' + str(ISP_id) + '_node_' + str(node_id)] = global_node_identifier
		
		global_position_of_nodes[global_node_identifier] = {}
		global_position_of_nodes[global_node_identifier] = (position_of_nodes[node_id][0],position_of_nodes[node_id][1])

		global_node_identifier += 1 

	"""
	G=nx.from_numpy_matrix(adjacency_matrix)
	nx.draw(G,pos=position_of_nodes,node_size=50) # use spring layout
	plt.draw()
	plt.show()
	"""
	
	number_of_internal_links = 0 
	for node_i in range(adjacency_matrix.shape[0]):
		for node_j in range(node_i+1,adjacency_matrix.shape[0]):
			if adjacency_matrix[node_i,node_j] == 1:
				number_of_internal_links +=1 

	connected = 1
	for el in adjacency_matrix:
		if np.sum(el) < 1:
			connected = 0 

	#print("isp ",curr_isp,'connected',connected)

	return global_node_identifier, info_per_physical_node, dict_local_to_global_node, adjacency_matrix, number_of_internal_links, global_position_of_nodes

def get_multidomain_graph(global_node_identifier, all_adjacency_matrices, dict_local_to_global_node,info_per_physical_node,average_peering_nodes_per_ISP,number_of_ISPs):

	total_number_of_nodes = global_node_identifier  
	global_adj_matrix = np.zeros([total_number_of_nodes,total_number_of_nodes])
	for isp_id in all_adjacency_matrices:

		curr_adj_matrix = all_adjacency_matrices[isp_id]
		for node_i in range(len(curr_adj_matrix)):
			for node_j in range(len(curr_adj_matrix)):

				entry = curr_adj_matrix[node_i,node_j]
				global_node_i = dict_local_to_global_node['ISP_' + str(isp_id) + '_node_' + str(node_i)]
				global_node_j = dict_local_to_global_node['ISP_' + str(isp_id) + '_node_' + str(node_j)]

				global_adj_matrix[global_node_i,global_node_j] = entry
	"""
	G=nx.from_numpy_matrix(global_adj_matrix)
	nx.draw(G,pos=global_position_of_nodes,node_size=50) # use spring layout
	plt.draw()
	plt.title("multidomain")
	plt.show()
	"""

	stop = 0
	go = 1 
	from time import time 
	t0 = time() 
	while go:

		total_number_of_peering_nodes = int(average_peering_nodes_per_ISP*number_of_ISPs)
		total_number_of_physical_nodes = global_node_identifier 

		all_physical_nodes = np.arange(total_number_of_physical_nodes)
		np.random.shuffle(all_physical_nodes)
		all_physical_nodes = all_physical_nodes[0:total_number_of_peering_nodes]

		#print(info_per_physical_node.keys())

		peering_nodes_per_ISP = {}
		for p1 in all_physical_nodes:
			isp = info_per_physical_node[p1]['ISP']
			#print(p1,isp)
			if isp not in peering_nodes_per_ISP:
				peering_nodes_per_ISP[isp] = []
			peering_nodes_per_ISP[isp].append(p1)

		go = 0 
		for isp in range(NUMBER_OF_ISPS):
			if isp not in peering_nodes_per_ISP:
				go = 1 

		tf = time()
		if (tf - t0) > 2:
			stop = 1
			break

	if stop:
		return 1,1,1	

	else:

		peering_links = []
		for isp in range(NUMBER_OF_ISPS):	

			number_of_peerings_in_curr_isp = len(peering_nodes_per_ISP[isp])
			if number_of_peerings_in_curr_isp >= 2:

				for peering_id in range(number_of_peerings_in_curr_isp - 1):
					p1 = peering_nodes_per_ISP[isp][peering_id]
					p2 = peering_nodes_per_ISP[isp][peering_id + 1]

					peering_links.append([p1,p2])

		for isp in range(NUMBER_OF_ISPS - 1):	
			
			p1 = peering_nodes_per_ISP[isp][-1]
			p2 = peering_nodes_per_ISP[isp+1][0]

			peering_links.append([p1,p2])

		#print(peering_links)

		p1 = peering_nodes_per_ISP[NUMBER_OF_ISPS - 1][-1]
		p2 = peering_nodes_per_ISP[0][0]

		peering_links.append([p1,p2])	


		#print(global_position_of_nodes)
		#print(peering_links)

		go = 1 
		while go:

			peering_nodes = []
			for peering_link in peering_links:
				p1 = peering_link[0]
				p2 = peering_link[1]

				peering_nodes.append(p1)
				peering_nodes.append(p2)

			peering_nodes = list(set(peering_nodes))

			p1 = np.random.choice(peering_nodes)
			p2 = np.random.choice(peering_nodes)

			if p1 != p2:

				isp1 = info_per_physical_node[p1]['ISP']
				isp2 = info_per_physical_node[p2]['ISP']

				if isp1 != isp2:
					peering_links.append([p1,p2])

			for peering_link in peering_links:
				p1 = peering_link[0]
				p2 = peering_link[1]
				global_adj_matrix[p1,p2] = 1 
				global_adj_matrix[p2,p1] = 1 

			number_of_external_links_per_ISP = {}
			for ISP_id in range(NUMBER_OF_ISPS):
				number_of_external_links_per_ISP[ISP_id] = 0 

			for peering_link in peering_links:
				p1 = peering_link[0]
				p2 = peering_link[1]

				isp1 = info_per_physical_node[p1]['ISP']
				isp2 = info_per_physical_node[p2]['ISP']

				if isp1 == isp2:
					number_of_external_links_per_ISP[isp1] += 1
				else:
					number_of_external_links_per_ISP[isp1] += 1
					number_of_external_links_per_ISP[isp2] += 1

			average_number_of_peering_links = np.mean(list(number_of_external_links_per_ISP.values()))
			#print("external ",average_number_of_peering_links)

			if average_number_of_peering_links > MEAN_NUMBER_OF_PEERING_LINKS:
				go = 0

		return global_adj_matrix, peering_links, 0


def get_adjacencies_matrices(number_of_ISPs,average_peering_nodes_per_ISP,alpha,beta,radius,types_of_physical_nodes, nodes,virtual_graph,capacities_of_physical_nodes,capacities_of_physical_links,capacities_of_peering_links,types_of_virtual_nodes,prob_of_feasibility_isp):

	info_per_physical_node = {}
	dict_local_to_global_node = {}

	all_adjacency_matrices = {} 
	global_node_identifier = 0 

	all_number_of_internal_links = []
	global_position_of_nodes = {} 
	feasibile_vfs_types_per_isp = {}
	for ISP_id in range(number_of_ISPs): 

		res = get_ISP_network(ISP_id,global_position_of_nodes,dict_local_to_global_node,info_per_physical_node,global_node_identifier,ISP_id,nodes,radius,types_of_physical_nodes,alpha,beta)
		global_node_identifier, info_per_physical_node, dict_local_to_global_node, adjacency_matrix, number_of_internal_links, global_position_of_nodes = res 

		all_adjacency_matrices[ISP_id] = adjacency_matrix
		all_number_of_internal_links.append(number_of_internal_links)

		feasibile_vfs_types_per_isp[ISP_id] = []
		for vnf_type in types_of_virtual_nodes:
			rv = np.random.binomial(1,prob_of_feasibility_isp)
			if rv > 0:
				feasibile_vfs_types_per_isp[ISP_id].append(vnf_type)

	for curr_isp in range(number_of_ISPs):
		curr_adj = all_adjacency_matrices[curr_isp]

		labels = {}
		for i in range(curr_adj.shape[0]):
			labels[i] = i

		"""
		G = nx.from_numpy_matrix(curr_adj)
		nx.draw(G,pos=nx.spring_layout(G),labels=labels)
		plt.draw()
		plt.show()	
		"""

	global_adj_matrix, peering_links, stop = get_multidomain_graph(global_node_identifier, all_adjacency_matrices,dict_local_to_global_node,info_per_physical_node,average_peering_nodes_per_ISP,number_of_ISPs)

	for curr_isp in range(number_of_ISPs):
		curr_adj = all_adjacency_matrices[curr_isp]

		labels = {}
		for i in range(curr_adj.shape[0]):
			labels[i] = i

		"""
		G = nx.from_numpy_matrix(curr_adj)
		nx.draw(G,pos=nx.spring_layout(G),labels=labels)
		plt.draw()
		plt.show()	
		"""

	if stop:
		return 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

	else:

		all_adjacency_matrices['multidomain'] = global_adj_matrix

		capacity_of_all_links = np.zeros(global_adj_matrix.shape)
		for physical_node_u in range(capacity_of_all_links.shape[0]):
			for physical_node_v in range(capacity_of_all_links.shape[1]):

				if global_adj_matrix[physical_node_u,physical_node_v] > 0:

					if (physical_node_u,physical_node_v) in peering_links:
						value_ = np.random.choice(capacities_of_peering_links)  
						print("peering links here") 
					else:
						value_ = np.random.choice(capacities_of_physical_links)   
					
					capacity_of_all_links[physical_node_u,physical_node_v] = value_
		
		peering_nodes = []
		for peering_link in peering_links:
			
			p1 = peering_link[0]
			p2 = peering_link[1]

			peering_nodes.append(p1)
			peering_nodes.append(p2)

		peering_nodes = list(set(peering_nodes))
		dict_peering_to_local = {} # this dictionary maps the global id of the peering nodes to a local id
		dict_local_to_peering = {}

		c_local = 0
		for node in peering_nodes:
			dict_peering_to_local[node] = c_local 
			dict_local_to_peering[c_local] = node 
			c_local += 1 

		#print(dict_peering_to_local)

		peering_adj = np.zeros([len(peering_nodes),len(peering_nodes)])
		for node_i in range(len(peering_nodes)):
			for node_j in range(len(peering_nodes)):

				node_i_global = dict_local_to_peering[node_i]
				node_j_global = dict_local_to_peering[node_j]

				if [node_i_global,node_j_global] in peering_links or [node_j_global,node_i_global] in peering_links:
					peering_adj[node_i,node_j] = 1 
					peering_adj[node_j,node_i] = 1 

		all_adjacency_matrices['peering_network'] = peering_adj
		all_adjacency_matrices['virtual_graph'] = virtual_graph

		dict_of_capacities_of_physical_nodes = {}
		for global_physical_node_id in range(global_adj_matrix.shape[0]):
			dict_of_capacities_of_physical_nodes[global_physical_node_id] = np.random.choice(capacities_of_physical_nodes)

		all_peerings_nodes = []
		for node_i in range(len(peering_nodes)):
			all_peerings_nodes.append(dict_local_to_peering[node_i])

		for peering_node in dict_of_capacities_of_physical_nodes:
			if peering_node in all_peerings_nodes:
				dict_of_capacities_of_physical_nodes[peering_node] = 0

		return feasibile_vfs_types_per_isp, all_adjacency_matrices, info_per_physical_node, dict_local_to_global_node, global_position_of_nodes, peering_links, dict_peering_to_local, dict_local_to_peering, peering_nodes, dict_of_capacities_of_physical_nodes, capacity_of_all_links, 0

def define_feasibility_matrix(all_adjacency_matrices,info_per_physical_node,prob_of_feasibility_isp,prob_feasibility_node,where_vfs_can_be_hosted,infinity_value):

	multidomain_adj = all_adjacency_matrices['multidomain']
	virtual_adj = all_adjacency_matrices['virtual_graph']

	n_physical_nodes = multidomain_adj.shape[0]
	n_virtual_nodes = virtual_adj.shape[0]

	feasibility_matrix = np.ones([n_physical_nodes,n_virtual_nodes])*infinity_value
	for phy_id in range(n_physical_nodes):
		for vf_id in range(n_virtual_nodes):
			feasible_rv = np.random.binomial(1,prob_feasibility_node)
			if feasible_rv > 0:
				feasibility_matrix[phy_id,vf_id] = 1 

	for phy_id in range(n_physical_nodes):
		owner_isp = info_per_physical_node[phy_id]['ISP']
		for vf in range(n_virtual_nodes):
			if owner_isp not in where_vfs_can_be_hosted[vf]:
				feasibility_matrix[phy_id,vf] = infinity_value

	return feasibility_matrix

def define_virtual_node_cost_matrix(all_adjacency_matrices,virtual_node_types,number_of_ISPs,info_per_physical_node,max_demand_per_node,n_virtual_nodes,dict_virtual_node_types,max_cost_per_virtual_node):

	adj_matrix = all_adjacency_matrices['multidomain']
	n_physical_nodes = len(adj_matrix)

	demand_per_virtual_node = np.random.randint(1,max_demand_per_node+1,size=n_virtual_nodes)
	
	# we assume costs of virtual nodes are assinged randomly to the ISPs. We could play on this parameter, because now the model 
	# is not very good... we should consider that there is a nominal cost for each VF (e.g., firewall could cost more than a normal server) 
	# and that the ISPs will offer different costs within a specific range
	
	num_of_types_of_vfs = len(virtual_node_types)
	cost_per_virtual_node = np.random.randint(1,max_cost_per_virtual_node+1,[number_of_ISPs,num_of_types_of_vfs])

	cost_matrix = np.zeros([n_physical_nodes,n_virtual_nodes])
	for node_id in range(n_physical_nodes):
		for virtual_node_id in range(n_virtual_nodes):

			curr_isp = info_per_physical_node[node_id]['ISP']
			virtual_node_type = dict_virtual_node_types[virtual_node_id]

			cost_for_curr_virtual_node_in_curr_isp = cost_per_virtual_node[curr_isp,virtual_node_type]
			demand_for_curr_virtual_node = demand_per_virtual_node[virtual_node_id]

			cost_matrix[node_id,virtual_node_id] = demand_for_curr_virtual_node*cost_for_curr_virtual_node_in_curr_isp  

	return cost_matrix, demand_per_virtual_node

def define_physical_links_cost_matrix(all_adjacency_matrices,info_per_physical_node,inter_links_costs, intra_links_costs):

	adj_matrix = all_adjacency_matrices['multidomain']
	link_matrix = np.zeros(adj_matrix.shape)

	for node_i in range(adj_matrix.shape[0]):
		for node_j in range(node_i+1,adj_matrix.shape[1]):

			if adj_matrix[node_i,node_j] == 1:

				ispi = info_per_physical_node[node_i]['ISP']
				ispj = info_per_physical_node[node_j]['ISP']

				if ispi == ispj:
					cost = np.random.choice(intra_links_costs)
				else:
					cost = np.random.choice(inter_links_costs)

				link_matrix[node_i,node_j] = cost 
				link_matrix[node_j,node_i] = cost 

	return link_matrix

def define_bandwidth_cost_for_all_possible_assignments_in_peering_network(cost_per_link_matrix,all_adjacency_matrices,virtual_node_types,max_bandwidth_demand_per_pairs,dict_peering_to_local,dict_local_to_peering,num_of_virtual_nodes):

	# this function computes the accumulated costs over the shortest path between each pair of PEERING nodes for each possible pair of virtual nodes.
	# a similar module is defined considering the overall network (i.e., not only that composed of peering nodes)

	adj_matrix = all_adjacency_matrices['multidomain']
	num_of_physical_nodes = adj_matrix.shape[0]
	#num_of_virtual_nodes = len(virtual_node_types)

	bandwidth_requests = np.random.randint(0,max_bandwidth_demand_per_pairs+1,[num_of_virtual_nodes,num_of_virtual_nodes])
	for i in range(bandwidth_requests.shape[0]):
		bandwidth_requests[i,i] = 0

	peering_network = all_adjacency_matrices['peering_network']
	G_peering = nx.from_numpy_matrix(peering_network)

	num_peering_nodes = peering_network.shape[0]
	cost_per_virtual_links = np.zeros([num_peering_nodes,num_peering_nodes,num_of_virtual_nodes,num_of_virtual_nodes])
	shortest_path_per_each_pair_of_peering = {}

	for phy_node_p in range(peering_network.shape[0]):
		for phy_node_pp in range(peering_network.shape[0]):

			for virtual_node_i in range(num_of_virtual_nodes):
				for virtual_node_j in range(num_of_virtual_nodes):

					curr_pair_of_peerings_and_virtual_nodes = str(phy_node_p) + '_' + str(phy_node_pp) + str(virtual_node_i) + '_' + str(virtual_node_j)
					shortest_path_per_each_pair_of_peering[curr_pair_of_peerings_and_virtual_nodes] = None 

					curr_req = bandwidth_requests[virtual_node_i,virtual_node_j]
					if phy_node_p != phy_node_pp:
						all_paths = nx.all_simple_paths(G_peering,source=phy_node_p,target=phy_node_pp)
					else:
						all_paths = [[phy_node_p,phy_node_pp]]

					minimum_cost = 10**6
					for curr_path in all_paths:
						curr_cost = 0
						for ii in range(len(curr_path)-1):
							n1 = curr_path[ii]
							n2 = curr_path[ii+1]
							#print(n1,n2)
							global_id_for_n1 = dict_local_to_peering[n1]
							global_id_for_n2 = dict_local_to_peering[n2]

							cost_of_link = cost_per_link_matrix[global_id_for_n1,global_id_for_n2]
							curr_cost += cost_of_link*curr_req  

						if curr_cost < minimum_cost:
							minimum_cost = curr_cost 
							shortest_path = curr_path

					cost_per_virtual_links[phy_node_p,phy_node_pp,virtual_node_i,virtual_node_j] = minimum_cost
					shortest_path_per_each_pair_of_peering[curr_pair_of_peerings_and_virtual_nodes] = shortest_path

	"""
	for phy_node_p in range(cost_per_virtual_links.shape[0]):
		for phy_node_pp in range(cost_per_virtual_links.shape[1]):
			print(cost_per_virtual_links[phy_node_p,phy_node_pp])
	"""

	return cost_per_virtual_links, bandwidth_requests, shortest_path_per_each_pair_of_peering

