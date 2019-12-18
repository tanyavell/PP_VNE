from config import *
import numpy as np 
import networkx as nx 
from matplotlib import pyplot as plt 
from pulp import *
import copy

def define_x_matrix_for_ilp1(n_peering_nodes,n_virtual_nodes):

	x_matrix = np.zeros([n_peering_nodes, n_virtual_nodes]).astype(LpVariable)

	for physical_node in range(n_peering_nodes):
		for virtual_node in range(n_virtual_nodes):
			x_matrix[physical_node,virtual_node] = LpVariable('x_{}_{}'.format(physical_node, virtual_node), lowBound=0, cat='Continuous')
			#x_matrix[physical_node,virtual_node] = LpVariable('x_{}_{}'.format(physical_node, virtual_node), lowBound=0, cat='Binary')

	return x_matrix

def define_y_matrix_for_ilp1(n_peering_nodes,n_virtual_nodes):

	y_matrix = np.zeros([n_peering_nodes, n_peering_nodes, n_virtual_nodes, n_virtual_nodes]).astype(LpVariable)
	for physical_node in range(n_peering_nodes):
		for physical_node_ in range(n_peering_nodes):
			for virtual_node in range(n_virtual_nodes):
				for virtual_node_ in range(n_virtual_nodes):
					y_matrix[physical_node,physical_node_,virtual_node,virtual_node_] = LpVariable('y_{}_{}_{}_{}'.format(physical_node, physical_node_, virtual_node,virtual_node_), lowBound=0, cat='Continuous')
					#y_matrix[physical_node,physical_node_,virtual_node,virtual_node_] = LpVariable('y_{}_{}_{}_{}'.format(physical_node, physical_node_, virtual_node,virtual_node_), lowBound=0, cat='Binary')
	

	return y_matrix

def define_x_matrix_for_ilp2(n_physical_nodes_of_curr_isp,n_virtual_nodes_of_curr_isp,peering_nodes_of_curr_isp_working_as_gateways):

	n_peering_nodes = len(peering_nodes_of_curr_isp_working_as_gateways)
	x_matrix = np.zeros([n_physical_nodes_of_curr_isp, n_virtual_nodes_of_curr_isp + n_peering_nodes]).astype(LpVariable)

	for physical_node in range(n_physical_nodes_of_curr_isp):
		for virtual_node in range(n_virtual_nodes_of_curr_isp + n_peering_nodes):
			x_matrix[physical_node,virtual_node] = LpVariable('x_{}_{}'.format(physical_node, virtual_node), lowBound=0, cat='Continuous')
			#x_matrix[physical_node,virtual_node] = LpVariable('x_{}_{}'.format(physical_node, virtual_node), lowBound=0, cat='Binary')

	return x_matrix

def define_y_matrix_for_ilp2(n_physical_nodes_of_curr_isp,n_virtual_nodes_of_curr_isp,peering_nodes_of_curr_isp_working_as_gateways):

	n_peering_nodes = len(peering_nodes_of_curr_isp_working_as_gateways)
	y_matrix = np.zeros([n_physical_nodes_of_curr_isp, n_physical_nodes_of_curr_isp, n_virtual_nodes_of_curr_isp + n_peering_nodes, n_virtual_nodes_of_curr_isp + n_peering_nodes]).astype(LpVariable)
	for physical_node in range(n_physical_nodes_of_curr_isp):
		for physical_node_ in range(n_physical_nodes_of_curr_isp):
			for virtual_node in range(n_virtual_nodes_of_curr_isp + n_peering_nodes):
				for virtual_node_ in range(n_virtual_nodes_of_curr_isp + n_peering_nodes):
					y_matrix[physical_node,physical_node_,virtual_node,virtual_node_] = LpVariable('y_{}_{}_{}_{}'.format(physical_node, physical_node_, virtual_node,virtual_node_), lowBound=0, cat='Continuous')
					#y_matrix[physical_node,physical_node_,virtual_node,virtual_node_] = LpVariable('y_{}_{}_{}_{}'.format(physical_node, physical_node_, virtual_node,virtual_node_), lowBound=0, cat='Binary')
	

	return y_matrix

def define_x_matrix_for_FID(n_physical_nodes,n_virtual_nodes):

	x_matrix = np.zeros([n_physical_nodes, n_virtual_nodes]).astype(LpVariable)

	for physical_node in range(n_physical_nodes):
		for virtual_node in range(n_virtual_nodes):
			x_matrix[physical_node,virtual_node] = LpVariable('x_{}_{}'.format(physical_node, virtual_node), lowBound=0, cat='Continuous')
			#x_matrix[physical_node,virtual_node] = LpVariable('x_{}_{}'.format(physical_node, virtual_node), lowBound=0, cat='Binary')
	
	return x_matrix

def define_y_matrix_for_FID(n_physical_nodes,n_virtual_nodes):

	y_matrix = np.zeros([n_physical_nodes, n_physical_nodes, n_virtual_nodes, n_virtual_nodes]).astype(LpVariable)
	for physical_node in range(n_physical_nodes):
		for physical_node_ in range(n_physical_nodes):
			for virtual_node in range(n_virtual_nodes):
				for virtual_node_ in range(n_virtual_nodes):
					y_matrix[physical_node,physical_node_,virtual_node,virtual_node_] = LpVariable('y_{}_{}_{}_{}'.format(physical_node, physical_node_, virtual_node,virtual_node_), lowBound=0, cat='Continuous')
					#y_matrix[physical_node,physical_node_,virtual_node,virtual_node_] = LpVariable('y_{}_{}_{}_{}'.format(physical_node, physical_node_, virtual_node,virtual_node_), lowBound=0, cat='Binary')
	
	return y_matrix

def compute_cost_ilp1_replaced(x_solution_ilp1,cost_per_virtual_node,cost_per_virtual_links,dict_local_to_peering,info_per_physical_node,where_vfs_can_be_hosted,infinity_value,vfs_assigned_per_isp,all_adjacency_matrices,bandwidth_requests_between_virtual_nodes):

	isp_per_vf = {}
	for owner_isp in vfs_assigned_per_isp:
		vfs = vfs_assigned_per_isp[owner_isp]
		for vf in vfs:
			if vf not in isp_per_vf:
				isp_per_vf[vf] = []
			isp_per_vf[vf].append(owner_isp)

	multidomain_adj = all_adjacency_matrices['multidomain']
	G_multidomain = nx.from_numpy_matrix(multidomain_adj)

	traversed_links = []
	# cost of putting flows on physical links
	cost = 0 
	for vf1 in range(x_solution_ilp1.shape[1]):
		where_is_vf1 = np.where(x_solution_ilp1[:,vf1] > 0)[0][0]
		where_is_vf1_global = dict_local_to_peering[where_is_vf1]
		owner_of_vf1 = info_per_physical_node[where_is_vf1_global]['ISP']

		f1 = 1 
		if owner_of_vf1 not in where_vfs_can_be_hosted[vf1]:
			f1 = infinity_value

		for vf2 in range(x_solution_ilp1.shape[1]):
			where_is_vf2 = np.where(x_solution_ilp1[:,vf2] > 0)[0][0]
			where_is_vf2_global = dict_local_to_peering[where_is_vf2]
			owner_of_vf2 = info_per_physical_node[where_is_vf2_global]['ISP']

			f2 = 1 
			if owner_of_vf2 not in where_vfs_can_be_hosted[vf2]:
				f2 = infinity_value

			bw = bandwidth_requests_between_virtual_nodes[vf1,vf2]
			if bw > 0:

				shortest_path = nx.shortest_path(G_multidomain,where_is_vf1_global,where_is_vf2_global)
				if len(shortest_path) > 1:
					for index_ in range(len(shortest_path)-1):
						n1 = shortest_path[index_]
						n2 = shortest_path[index_+1]
						traversed_links.append([n1,n2])
				#cost_of_curr_link = cost_per_virtual_links[where_is_vf1,where_is_vf2,vf1,vf2]*f1*f2
				cost_of_curr_link = cost_per_virtual_links[where_is_vf1,where_is_vf2,vf1,vf2]*f1*f2
				cost += cost_of_curr_link

	return cost, traversed_links 


def compute_cost_ilp1(x_solution_ilp1,y_solution_ilp1,cost_per_virtual_node,cost_per_virtual_links,dict_local_to_peering,info_per_physical_node,where_vfs_can_be_hosted,infinity_value):

	# cost of putting flows on physical links
	cost = 0 
	for vf1 in range(x_solution_ilp1.shape[1]):
		where_is_vf1 = np.where(x_solution_ilp1[:,vf1] > 0)[0][0]
		where_is_vf1_global = dict_local_to_peering[where_is_vf1]
		owner_of_vf1 = info_per_physical_node[where_is_vf1_global]['ISP']

		f1 = 1 
		if owner_of_vf1 not in where_vfs_can_be_hosted[vf1]:
			f1 = infinity_value

		for vf2 in range(x_solution_ilp1.shape[1]):
			where_is_vf2 = np.where(x_solution_ilp1[:,vf2] > 0)[0][0]
			where_is_vf2_global = dict_local_to_peering[where_is_vf2]
			owner_of_vf2 = info_per_physical_node[where_is_vf2_global]['ISP']

			f2 = 1 
			if owner_of_vf2 not in where_vfs_can_be_hosted[vf2]:
				f2 = infinity_value

			cost_of_curr_link = cost_per_virtual_links[where_is_vf1,where_is_vf2,vf1,vf2]*f1*f2
			cost += cost_of_curr_link

	return cost 

def compute_cost_ilp2(x_solution,f_solution,curr_isp,feasibility_matrix,vfs_assigned_per_isp,dict_local_to_global_node,cpu_demand_per_virtual_node,cost_per_virtual_node,virtual_traffic_matrix_inside_curr_isp,cost_per_link_matrix):

	n_virtual_nodes_of_curr_isp = len(vfs_assigned_per_isp[curr_isp])
	# cost of embedding vnodes on phy nodes
	cost = 0  
	cost_embedding = 0 
	for vf in range(n_virtual_nodes_of_curr_isp): 
		which_phy_node = np.where(x_solution[:,vf] > 0)[0][0]
		global_phy_node = dict_local_to_global_node['ISP_' + str(curr_isp) + '_node_' + str(which_phy_node)]

		global_virtual_node = vfs_assigned_per_isp[curr_isp][vf]
		cost += cost_per_virtual_node[global_phy_node,global_virtual_node]#cpu_demand_per_virtual_node[global_virtual_node]*

	cost_embedding = cost 

	cost_links = 0 
	traversed_links = []
	for vf1 in range(virtual_traffic_matrix_inside_curr_isp.shape[0]):
		for vf2 in range(virtual_traffic_matrix_inside_curr_isp.shape[1]):
			if vf1 != vf2:
				curr_f = f_solution[:,:,vf1,vf2]
				for local_phy_node1 in range(curr_f.shape[0]):
					for local_phy_node2 in range(curr_f.shape[1]):
						global_phy_node1 = dict_local_to_global_node['ISP_' + str(curr_isp) + '_node_' + str(local_phy_node1)]
						global_phy_node2 = dict_local_to_global_node['ISP_' + str(curr_isp) + '_node_' + str(local_phy_node2)]

						curr_cost = f_solution[local_phy_node1,local_phy_node2,vf1,vf2]*cost_per_link_matrix[global_phy_node1,global_phy_node2]
						cost += curr_cost 
						cost_links += curr_cost

						if f_solution[local_phy_node1,local_phy_node2,vf1,vf2] > 0:
							edge = [global_phy_node1,global_phy_node2]
							if edge not in traversed_links:
								traversed_links.append(edge)

	return cost, cost_embedding, cost_links, traversed_links

def compute_cost_ILP_FID(x_solution,f_solution,feasibility_matrix,cpu_demand_per_virtual_node,cost_per_virtual_node,bandwidth_requests_between_virtual_nodes,cost_per_link_matrix):

	n_virtual_nodes = x_solution.shape[1]

	# cost of embedding vnodes on phy nodes
	cost = 0  
	for vf in range(n_virtual_nodes): 
		phy_node = np.where(x_solution[:,vf] > 0)[0][0]
		cost += cost_per_virtual_node[phy_node,vf]#cpu_demand_per_virtual_node[vf]*

	print("embedding cost FID",cost)

	for vf1 in range(n_virtual_nodes):
		for vf2 in range(n_virtual_nodes):
			if vf1 != vf2:
				curr_f = f_solution[:,:,vf1,vf2]
				for phy_node1 in range(curr_f.shape[0]):
					for phy_node2 in range(curr_f.shape[1]):

						cost += f_solution[phy_node1,phy_node2,vf1,vf2]*cost_per_link_matrix[phy_node1,phy_node2]

	return cost 

def solve_ilp1(where_vfs_can_be_hosted,cost_per_virtual_node,cost_per_virtual_links,all_adjacency_matrices,feasibility_matrix,dict_local_to_global_peering,infinity_value,info_per_physical_node):

	go = 1 
	additional_constraints = [] 
	while go:

		n_peering_nodes = all_adjacency_matrices['peering_network'].shape[0]
		n_virtual_nodes = all_adjacency_matrices['virtual_graph'].shape[0]

		x_matrix = define_x_matrix_for_ilp1(n_peering_nodes,n_virtual_nodes)
		y_matrix = define_y_matrix_for_ilp1(n_peering_nodes,n_virtual_nodes)

		# Set objective
		# objective relative to nodes 
		nodes_cost = []
		for virtual_node_i in range(n_virtual_nodes):
			for peering_p in range(n_peering_nodes):
				global_id_for_peering_p = dict_local_to_global_peering[peering_p]
				curr_isp = info_per_physical_node[global_id_for_peering_p]['ISP']

				if curr_isp in where_vfs_can_be_hosted[virtual_node_i]:
					f1 = 1 
				else:
					f1 = infinity_value

				f2 = cost_per_virtual_node[global_id_for_peering_p,virtual_node_i]
				f3 = x_matrix[peering_p,virtual_node_i]

				c = f1*f2*f3 

				nodes_cost.append(c)

		nodes_cost = lpSum(nodes_cost)

		#objective relative to links 
		links_cost = []
		for peering_p in range(n_peering_nodes):
			for peering_pp in range(n_peering_nodes):
				for virtual_node_i in range(n_virtual_nodes):
					for virtual_node_j in range(n_virtual_nodes):

						if virtual_node_i != virtual_node_j:

							global_id_for_peering_p = dict_local_to_global_peering[peering_p]
							global_id_for_peering_pp = dict_local_to_global_peering[peering_pp]
							
							isp_peering_p = info_per_physical_node[global_id_for_peering_p]['ISP']
							isp_peering_pp = info_per_physical_node[global_id_for_peering_pp]['ISP']

							if isp_peering_p in where_vfs_can_be_hosted[virtual_node_i]:
								f1 = 1 
							else:
								f1 = infinity_value

							if isp_peering_pp in where_vfs_can_be_hosted[virtual_node_j]:
								f2 = 1 
							else:
								f2 = infinity_value

							f3 = cost_per_virtual_links[peering_p,peering_pp,virtual_node_i,virtual_node_j]
							f4 = y_matrix[peering_p,peering_pp,virtual_node_i,virtual_node_j]

							c = f1*f2*f3*f4
							links_cost.append(c)

		links_cost = lpSum(links_cost)
			
		problem = LpProblem("Cost Minimization", LpMinimize) 
		problem += nodes_cost + links_cost #(eq. 6)

		# now we set constraints 
		for virtual_node_i in range(n_virtual_nodes):
			assignment_for_this_virtual_node = x_matrix[:,virtual_node_i]
			problem += lpSum(assignment_for_this_virtual_node) == 1

		# integrity of the VNF request (8)
		for virtual_node_i in range(n_virtual_nodes):
			for peering_p in range(n_peering_nodes):
				constraint = [] 
				for peering_pp in range(n_peering_nodes):
					for virtual_node_j in range(n_virtual_nodes):
						if virtual_node_i != virtual_node_j:
							constraint.append((y_matrix[peering_p,peering_pp,virtual_node_i,virtual_node_j] + y_matrix[peering_pp,peering_p,virtual_node_j,virtual_node_i]))

				problem += lpSum(constraint) == 2*(n_virtual_nodes-1)*x_matrix[peering_p,virtual_node_i]

		# consistency of nodes' assignment 
		for virtual_node_i in range(n_virtual_nodes):
			for virtual_node_j in range(n_virtual_nodes):
				constraint = [] 
				for peering_p in range(n_peering_nodes):
					for peering_pp in range(n_peering_nodes):
						constraint.append(y_matrix[peering_p,peering_pp,virtual_node_i,virtual_node_j]) 

				problem += lpSum(constraint) == 1

		for constraint in additional_constraints:
			best_p, best_i = constraint
			problem += lpSum(x_matrix[best_p,best_i]) == 1

		problem.writeLP("CheckLpProgram.lp")
		problem.solve()
		problem.roundSolution()

		x_solution = np.zeros(x_matrix.shape)
		y_solution = np.zeros(y_matrix.shape)

		for v in problem.variables():
			var = str(v)
			if 'x' in var:
				info_var = var.split('_')
				physical_node = int(info_var[1])
				virtual_node = int(info_var[2])

				value = v.varValue
				#print(physical_node, virtual_node, value)
				x_solution[physical_node,virtual_node] = value 

			if 'y' in var:
				info_var = var.split('_')
				physical_node_u = int(info_var[1])
				physical_node_v = int(info_var[2])
				virtual_node_i = int(info_var[3])
				virtual_node_j = int(info_var[4])

				value = v.varValue

				y_solution[physical_node_u,physical_node_v,virtual_node_i,virtual_node_j] = value 

		N_v_prime = []
		for vf in range(n_virtual_nodes):
			arg_max_ = np.argmax(x_solution[:,vf])
			max_ = x_solution[arg_max_,vf]
			if max_ < 1:
				N_v_prime.append(vf)

		if len(N_v_prime) > 0:

			X_dict = {}
			for vf_i in N_v_prime:
				X_i = x_solution[:,vf_i]
				#sorted(X_i)
				X_i = np.sort(X_i)[::-1]
				X_dict[vf_i] = X_i#[::-1]

			ratio_per_vf = {}
			for vf_i in N_v_prime:
				ratio_per_vf[vf_i] = X_dict[vf_i][0]/X_dict[vf_i][1]

			max_ =-10**6
			for vf_i in ratio_per_vf:
				if ratio_per_vf[vf_i] > max_:
					max_ = ratio_per_vf[vf_i]
					best_i = vf_i

			best_p = np.argmax(x_solution[:,best_i])
			additional_constraints.append([best_p,best_i])
		
		else:
			go = 0


	ok = 1
	for virtual_node_i in range(n_virtual_nodes):
		for peering_p in range(n_peering_nodes):
			constraint = [] 
			for peering_pp in range(n_peering_nodes):
				for virtual_node_j in range(n_virtual_nodes):
					if virtual_node_i != virtual_node_j:
						constraint.append((y_solution[peering_p,peering_pp,virtual_node_i,virtual_node_j] + y_matrix[peering_pp,peering_p,virtual_node_j,virtual_node_i]))

			s_left = np.sum(constraint)
			s_right = 2*(n_virtual_nodes-1)*x_solution[peering_p,virtual_node_i]

			if s_left != s_right:
				ok = 0

	if not ok:
		pass#print("inconsitency")

	return x_solution, y_solution

def solve_ilp2(all_adjacency_matrices,vfs_assigned_per_isp,curr_isp,x_solution_ilp1,dict_local_to_peering,info_per_physical_node,bandwidth_requests_between_virtual_nodes,dict_local_to_global_node,feasibility_matrix,cpu_demand_per_virtual_node,dict_of_capacities_of_physical_nodes,capacity_of_all_links):

	#print(dict_local_to_global_node)
	
	dict_global_to_local_node = {}
	curr_isp_id = 'ISP_' + str(curr_isp)
	for el in dict_local_to_global_node:
		if curr_isp_id in el:
			gl_node = dict_local_to_global_node[el]
			dict_global_to_local_node[gl_node] = int(el.split('_')[-1])

	peering_nodes_of_curr_isp_working_as_gateways = []
	n_physical_nodes_of_curr_isp = all_adjacency_matrices[curr_isp].shape[0]
	n_virtual_nodes = all_adjacency_matrices['virtual_graph'].shape[0]

	n_virtual_nodes_of_curr_isp = len(vfs_assigned_per_isp[curr_isp])
	for vf in vfs_assigned_per_isp[curr_isp]:
		peering_p = np.where(x_solution_ilp1[:,vf]>0)[0][0]
		peering_p = dict_local_to_peering[peering_p]
		peering_nodes_of_curr_isp_working_as_gateways.append(peering_p)

	dict_global_virtual_id_to_local_virtual_id = {}
	c_id = 0
	for vf in vfs_assigned_per_isp[curr_isp]:
		dict_global_virtual_id_to_local_virtual_id[vf] = c_id 
		c_id += 1 

	dict_local_virtual_id_to_global_virtual_id = {}
	for vf in dict_global_virtual_id_to_local_virtual_id:
		el = dict_global_virtual_id_to_local_virtual_id[vf]
		dict_local_virtual_id_to_global_virtual_id[el] = vf			

	peering_nodes_of_curr_isp_working_as_gateways = list(set(peering_nodes_of_curr_isp_working_as_gateways))
	n_peering_nodes_curr_isp = len(peering_nodes_of_curr_isp_working_as_gateways)
	
	dict_peering_id_to_its_virtual_alias = {}
	c_id = n_virtual_nodes_of_curr_isp
	for pp in peering_nodes_of_curr_isp_working_as_gateways:
		dict_peering_id_to_its_virtual_alias[pp] = c_id 
		c_id += 1 

	dict_virtual_alias_to_its_peering_id = {}
	for pp in dict_peering_id_to_its_virtual_alias:
		el = dict_peering_id_to_its_virtual_alias[pp]
		dict_virtual_alias_to_its_peering_id[el] = pp 


	virtual_traffic_matrix_inside_curr_isp = np.zeros([n_virtual_nodes_of_curr_isp+n_peering_nodes_curr_isp,n_virtual_nodes_of_curr_isp+n_peering_nodes_curr_isp])
	for vf_i in range(n_virtual_nodes):
		for vf_j in range(n_virtual_nodes):
			if vf_i in vfs_assigned_per_isp[curr_isp] and vf_j in vfs_assigned_per_isp[curr_isp]:
				virtual_traffic_matrix_inside_curr_isp[dict_global_virtual_id_to_local_virtual_id[vf_i],dict_global_virtual_id_to_local_virtual_id[vf_j]] = bandwidth_requests_between_virtual_nodes[vf_i,vf_j]

			elif vf_i in vfs_assigned_per_isp[curr_isp] and vf_j not in vfs_assigned_per_isp[curr_isp]:
				peering_node_of_alias_of_vf_i = np.where(x_solution_ilp1[:,vf_i]>0)[0][0]
				peering_node_of_alias_of_vf_i = dict_local_to_peering[peering_node_of_alias_of_vf_i]
				peering_node_of_alias_of_vf_i = dict_peering_id_to_its_virtual_alias[peering_node_of_alias_of_vf_i]
				virtual_traffic_matrix_inside_curr_isp[dict_global_virtual_id_to_local_virtual_id[vf_i],peering_node_of_alias_of_vf_i] += bandwidth_requests_between_virtual_nodes[vf_i,vf_j]

			elif vf_i not in vfs_assigned_per_isp[curr_isp] and vf_j in vfs_assigned_per_isp[curr_isp]:
				peering_node_of_alias_of_vf_j = np.where(x_solution_ilp1[:,vf_j]>0)[0][0]
				peering_node_of_alias_of_vf_j = dict_local_to_peering[peering_node_of_alias_of_vf_j]
				peering_node_of_alias_of_vf_j = dict_peering_id_to_its_virtual_alias[peering_node_of_alias_of_vf_j]
				virtual_traffic_matrix_inside_curr_isp[peering_node_of_alias_of_vf_j,dict_global_virtual_id_to_local_virtual_id[vf_j]] += bandwidth_requests_between_virtual_nodes[vf_i,vf_j]

			elif vf_i not in vfs_assigned_per_isp[curr_isp] and vf_j not in vfs_assigned_per_isp[curr_isp]:
				pass 

	x_matrix = define_x_matrix_for_ilp2(n_physical_nodes_of_curr_isp,n_virtual_nodes_of_curr_isp,peering_nodes_of_curr_isp_working_as_gateways)
	f_matrix = define_y_matrix_for_ilp2(n_physical_nodes_of_curr_isp,n_virtual_nodes_of_curr_isp,peering_nodes_of_curr_isp_working_as_gateways)	

	# Set objective
	nodes_cost = []
	for physical_node_u_local_id in range(x_matrix.shape[0]):
		physical_node_u_global_id = dict_local_to_global_node['ISP_' + str(curr_isp) + '_node_' + str(physical_node_u_local_id)]
		for virtual_node_i_local_id in range(n_virtual_nodes_of_curr_isp):
			virtual_node_i_global_id = dict_local_virtual_id_to_global_virtual_id[virtual_node_i_local_id]

			f1 = feasibility_matrix[physical_node_u_global_id,virtual_node_i_global_id]
			f2 = cpu_demand_per_virtual_node[virtual_node_i_local_id]
			f3 = x_matrix[physical_node_u_local_id,virtual_node_i_local_id]

			c = f1*f2*f3 
			nodes_cost.append(c)

	nodes_cost = lpSum(nodes_cost)

	links_cost = []
	for physical_node_u_local_id in range(n_physical_nodes_of_curr_isp):
		physical_node_u_global_id = dict_local_to_global_node['ISP_' + str(curr_isp) + '_node_' + str(physical_node_u_local_id)]	
		
		for physical_node_v_local_id in range(n_physical_nodes_of_curr_isp):
			physical_node_v_global_id = dict_local_to_global_node['ISP_' + str(curr_isp) + '_node_' + str(physical_node_v_local_id)]
			
			if physical_node_u_local_id != physical_node_v_local_id:

				for virtual_node_i_local_id in range(f_matrix.shape[-1]):
					for virtual_node_j_local_id in range(f_matrix.shape[-1]):
				
						if virtual_node_i_local_id != virtual_node_j_local_id:

							f1 = 1 #there is no point in setting this constraint, why should all the traversed node be feasibile? In my opinion this requirement should hold only for the hosting nodes, not for the traversed ones! 
							f2 = 1#feasibility_matrix[physical_node_v_global_id,virtual_node_j_global_id]
							f3 = f_matrix[physical_node_u_local_id,physical_node_v_local_id,virtual_node_i_local_id,virtual_node_j_local_id]

							c = f1*f2*f3 
							links_cost.append(c)

	links_cost = lpSum(links_cost)
	problem = LpProblem("Cost Minimization", LpMinimize) 
	problem += nodes_cost + links_cost #(eq. 11)

	#print(n_virtual_nodes_of_curr_isp,dict_virtual_alias_to_its_peering_id)
	# forcing the alias of the peering nodes to stay exactly on that peering 	
	
	for vf_id_alias in range(n_virtual_nodes_of_curr_isp,x_matrix.shape[-1]):
		phy_node_of_this_alias = dict_virtual_alias_to_its_peering_id[vf_id_alias]
		#print(vf_id_alias,phy_node_of_this_alias)
		local_phy_node_of_this_alias = dict_global_to_local_node[phy_node_of_this_alias]
		#print(vf_id_alias,local_phy_node_of_this_alias)
		for local_phy_node in range(x_matrix.shape[0]):
			s = x_matrix[local_phy_node,vf_id_alias]
			if local_phy_node == local_phy_node_of_this_alias:
				problem += lpSum(s) == 1
			else:
				problem += lpSum(s) == 0
	
	
	# constraints 
	# each vf should be placed in exactly one physical node (12)	
	for virtual_node_i_local_id in range(n_virtual_nodes_of_curr_isp):
		s = x_matrix[:,virtual_node_i_local_id]
		problem += lpSum(s) == 1
	

	# flow conservation constraint (13)
	for physical_node_u_local_id in range(n_physical_nodes_of_curr_isp):
		for virtual_node_i_local_id in range(f_matrix.shape[-1]):
			for virtual_node_j_local_id in range(f_matrix.shape[-1]):

				if virtual_node_i_local_id != virtual_node_j_local_id:

					flow_conservation_constraint = []
					for physical_node_v_local_id in range(n_physical_nodes_of_curr_isp):

						if physical_node_u_local_id != physical_node_v_local_id:

							curr_constraint1 = f_matrix[physical_node_u_local_id,physical_node_v_local_id,virtual_node_i_local_id,virtual_node_j_local_id]
							curr_constraint2 = f_matrix[physical_node_v_local_id,physical_node_u_local_id,virtual_node_i_local_id,virtual_node_j_local_id]

							flow_conservation_constraint.append(curr_constraint1 - curr_constraint2)

					d_ij = virtual_traffic_matrix_inside_curr_isp[virtual_node_i_local_id,virtual_node_j_local_id]
					problem += lpSum(flow_conservation_constraint) == d_ij*(x_matrix[physical_node_u_local_id,virtual_node_i_local_id] - x_matrix[physical_node_u_local_id,virtual_node_j_local_id])

	# node capacity constraint 
	for physical_node_u_local_id in range(n_physical_nodes_of_curr_isp):
		physical_node_u_global_id = dict_local_to_global_node['ISP_' + str(curr_isp) + '_node_' + str(physical_node_u_local_id)]	
		capacity_of_physical_node_u = dict_of_capacities_of_physical_nodes[physical_node_u_global_id]

		curr_capacity_constraint = []
		for virtual_node_i_local_id in range(n_virtual_nodes_of_curr_isp):

			virtual_node_i_global_id = dict_local_virtual_id_to_global_virtual_id[virtual_node_i_local_id]
			d_i = cpu_demand_per_virtual_node[virtual_node_i_global_id]

			curr_capacity_constraint.append(d_i*x_matrix[physical_node_u_local_id,virtual_node_i_local_id])

		problem += lpSum(curr_capacity_constraint) <= capacity_of_physical_node_u

	# link capacity constraint 
	for physical_node_u_local_id in range(n_physical_nodes_of_curr_isp):
		physical_node_u_global_id = dict_local_to_global_node['ISP_' + str(curr_isp) + '_node_' + str(physical_node_u_local_id)]

		for physical_node_v_local_id in range(n_physical_nodes_of_curr_isp):
			physical_node_v_global_id = dict_local_to_global_node['ISP_' + str(curr_isp) + '_node_' + str(physical_node_v_local_id)]

			capacity_of_the_link = capacity_of_all_links[physical_node_u_global_id,physical_node_v_global_id]

			capacity_of_the_link_constraint = []
			for virtual_node_i_local_id in range(virtual_traffic_matrix_inside_curr_isp.shape[0]):
				for virtual_node_j_local_id in range(virtual_traffic_matrix_inside_curr_isp.shape[0]):				
					capacity_of_the_link_constraint.append(f_matrix[physical_node_u_local_id,physical_node_v_local_id,virtual_node_i_local_id,virtual_node_j_local_id])

			problem += lpSum(capacity_of_the_link_constraint) <= capacity_of_the_link

	# additional constraints to obtain feasible solutions 
	#print(additional_constraints,'const')

	go = 1 
	additional_constraints = [] 
	while go:

		for additional_constraint in additional_constraints:
			candidate_physical,candidate_virtual = additional_constraint
			problem += lpSum(x_matrix[candidate_physical,candidate_virtual]) == 1
			additional_constraints = []

		problem.writeLP("CheckLpProgram.lp")
		status = problem.solve()
		problem.roundSolution()

		x_solution = np.zeros(x_matrix.shape)
		f_solution = np.zeros(f_matrix.shape)

		for v in problem.variables():
			var = str(v)
			if 'x' in var:
				info_var = var.split('_')
				physical_node = int(info_var[1])
				virtual_node = int(info_var[2])

				value = v.varValue
				#print(physical_node, virtual_node, value)
				x_solution[physical_node,virtual_node] = value 

			if 'y' in var:
				info_var = var.split('_')
				physical_node_u = int(info_var[1])
				physical_node_v = int(info_var[2])
				virtual_node_i = int(info_var[3])
				virtual_node_j = int(info_var[4])

				value = v.varValue
				f_solution[physical_node_u,physical_node_v,virtual_node_i,virtual_node_j] = value 

		max_ = -10**6 
		set_X = []
		for physical_node_ in range(x_solution.shape[0]):
			for virtual_node in range(x_solution.shape[1]):
				value = x_solution[physical_node_,virtual_node]
				if value >= max_ and value != 0 and value != 1:
					max_ = value 

					candidates = [physical_node_,virtual_node]
					set_X.append(candidates)

		if len(set_X) > 0:
			v = np.random.randint(len(set_X))
			candidates = set_X[v]
			additional_constraints.append(candidates)
		else:
			go = 0 

	return x_solution, f_solution, virtual_traffic_matrix_inside_curr_isp, dict_global_virtual_id_to_local_virtual_id

def solve_ILP_FID(all_adjacency_matrices,bandwidth_requests_between_virtual_nodes,feasibility_matrix,cpu_demand_per_virtual_node,dict_of_capacities_of_physical_nodes,capacity_of_all_links):

	multidomain_adj = all_adjacency_matrices['multidomain']
	n_physical_nodes = multidomain_adj.shape[0]

	virtual_graph = all_adjacency_matrices['virtual_graph']
	n_virtual_nodes = virtual_graph.shape[0]

	x_matrix = define_x_matrix_for_FID(n_physical_nodes,n_virtual_nodes)
	f_matrix = define_y_matrix_for_FID(n_physical_nodes,n_virtual_nodes)	

	# Set objective
	nodes_cost = []
	for physical_node_u in range(n_physical_nodes):
		for virtual_node_i in range(n_virtual_nodes):

			f1 = feasibility_matrix[physical_node_u,virtual_node_i]
			f2 = cpu_demand_per_virtual_node[virtual_node_i]
			f3 = x_matrix[physical_node_u,virtual_node_i]

			c = f1*f2*f3 
			nodes_cost.append(c)

	nodes_cost = lpSum(nodes_cost)

	links_cost = []
	for physical_node_u in range(n_physical_nodes):
		for physical_node_v in range(n_physical_nodes):

			if physical_node_u != physical_node_v:

				for virtual_node_i in range(f_matrix.shape[-1]):
					for virtual_node_j in range(f_matrix.shape[-1]):
				
						if virtual_node_i != virtual_node_j:

							f1 = 1 #there is no point in setting this constraint, why should all the traversed node be feasibile? In my opinion this requirement should hold only for the hosting nodes, not for the traversed ones! 
							f2 = 1#feasibility_matrix[physical_node_v_global_id,virtual_node_j_global_id]
							f3 = f_matrix[physical_node_u,physical_node_v,virtual_node_i,virtual_node_j]

							c = f1*f2*f3 
							links_cost.append(c)

	links_cost = lpSum(links_cost)
	problem = LpProblem("Cost Minimization", LpMinimize) 
	problem += nodes_cost + links_cost #(eq. 11)

	#print(n_virtual_nodes_of_curr_isp,dict_virtual_alias_to_its_peering_id)
	# forcing the alias of the peering nodes to stay exactly on that peering 	
	
	# constraints 
	# each vf should be placed in exactly one physical node (12)	
	for virtual_node_i in range(n_virtual_nodes):
		s = x_matrix[:,virtual_node_i]
		problem += lpSum(s) == 1
	

	# flow conservation constraint (13)
	for physical_node_u in range(n_physical_nodes):
		for virtual_node_i in range(n_virtual_nodes):
			for virtual_node_j in range(n_virtual_nodes):

				if virtual_node_i != virtual_node_j:

					flow_conservation_constraint = []
					for physical_node_v in range(n_physical_nodes):

						if physical_node_u != physical_node_v:

							curr_constraint1 = f_matrix[physical_node_u,physical_node_v,virtual_node_i,virtual_node_j]
							curr_constraint2 = f_matrix[physical_node_v,physical_node_u,virtual_node_i,virtual_node_j]

							flow_conservation_constraint.append(curr_constraint1 - curr_constraint2)

					d_ij = bandwidth_requests_between_virtual_nodes[virtual_node_i,virtual_node_j]
					problem += lpSum(flow_conservation_constraint) == d_ij*(x_matrix[physical_node_u,virtual_node_i] - x_matrix[physical_node_u,virtual_node_j])

	# node capacity constraint 
	for physical_node_u in range(n_physical_nodes):
		capacity_of_physical_node_u = dict_of_capacities_of_physical_nodes[physical_node_u]
		curr_capacity_constraint = []
		for virtual_node_i in range(n_virtual_nodes):
			d_i = cpu_demand_per_virtual_node[virtual_node_i]

			curr_capacity_constraint.append(d_i*x_matrix[physical_node_u,virtual_node_i])

		problem += lpSum(curr_capacity_constraint) <= capacity_of_physical_node_u

	# link capacity constraint 
	for physical_node_u in range(n_physical_nodes):
		for physical_node_v in range(n_physical_nodes):

			capacity_of_the_link = capacity_of_all_links[physical_node_u,physical_node_v]

			capacity_of_the_link_constraint = []
			for virtual_node_i in range(n_virtual_nodes):
				for virtual_node_j in range(n_virtual_nodes):				
					capacity_of_the_link_constraint.append(f_matrix[physical_node_u,physical_node_v,virtual_node_i,virtual_node_j])

			problem += lpSum(capacity_of_the_link_constraint) <= capacity_of_the_link

	# additional constraints to obtain feasible solutions 
	#print(additional_constraints,'const')

	go = 1 
	additional_constraints = [] 
	while go:

		for additional_constraint in additional_constraints:
			candidate_physical,candidate_virtual = additional_constraint
			problem += lpSum(x_matrix[candidate_physical,candidate_virtual]) == 1
			additional_constraints = []

		problem.writeLP("CheckLpProgram.lp")
		status = problem.solve()
		problem.roundSolution()

		x_solution = np.zeros(x_matrix.shape)
		f_solution = np.zeros(f_matrix.shape)

		for v in problem.variables():
			var = str(v)
			if 'x' in var:
				info_var = var.split('_')
				physical_node = int(info_var[1])
				virtual_node = int(info_var[2])

				value = v.varValue
				#print(physical_node, virtual_node, value)
				x_solution[physical_node,virtual_node] = value 

			if 'y' in var:
				info_var = var.split('_')
				physical_node_u = int(info_var[1])
				physical_node_v = int(info_var[2])
				virtual_node_i = int(info_var[3])
				virtual_node_j = int(info_var[4])

				value = v.varValue

				f_solution[physical_node_u,physical_node_v,virtual_node_i,virtual_node_j] = value 

		max_ = -10**6 
		set_X = []
		for physical_node_ in range(x_solution.shape[0]):
			for virtual_node in range(x_solution.shape[1]):
				value = x_solution[physical_node_,virtual_node]
				if value >= max_ and value != 0 and value != 1:
					max_ = value 

					candidates = [physical_node_,virtual_node]
					set_X.append(candidates)

		if len(set_X) > 0:
			v = np.random.randint(len(set_X))
			candidates = set_X[v]
			additional_constraints.append(candidates)
		else:
			go = 0 

	return x_solution, f_solution
