from matplotlib import pyplot as plt 
import numpy as np 
from config import * 

def compute_percentile(x):

	percentile_vec = []
	for i in range(101):
		percentile_vec.append(np.percentile(x,i))

	return percentile_vec

#r = np.load('results_simulations.npy').item()
r = np.load('results_simulations.npy').item()
last_samples = N_EPOCHS

tot_lid = [0 for el in range(last_samples)]
tot_fid = [0 for el in range(last_samples)]
tot_rl = [0 for el in range(last_samples)]
tot_rl_private_o1 = [0 for el in range(last_samples)]
tot_rl_private_o2 = [0 for el in range(last_samples)]
tot_rl_private_o3 = [0 for el in range(last_samples)]
tot_rl_private_o4 = [0 for el in range(last_samples)]

good_sims = 0
average_lid_time = 0
average_fid_time = 0 
average_rl_time = 0 

average_rl_private_time = 0
average_rl_private_o1_time = 0
average_rl_private_o2_time = 0
average_rl_private_o3_time = 0
average_rl_private_o4_time = 0

final_lids = []
final_fids = []
final_rls = []
final_rls_private_o1 = []
final_rls_private_o2 = []
final_rls_private_o3 = []
final_rls_private_o4 = []

for n_sim in r:
	curr_res = r[n_sim]
	L = len(curr_res['RL']['cost'][-last_samples:])
	
	if L > 1 and curr_res['LID']['cost'] < 10000 and curr_res['LID']['cost'] < 10000 and curr_res['RL_private_o2']['cost'][-1] < 10000:

		good_sims += 1 
		print(good_sims)

		res_lid = np.asarray([curr_res['LID']['cost'] for el in range(L)])
		res_fid = np.asarray([curr_res['FID']['cost'] for el in range(L)])
		res_rl = np.asarray(curr_res['RL']['cost'][-last_samples:])
		res_rl_private_o1 = np.asarray(curr_res['RL_private_o1']['cost'][-last_samples:])
		res_rl_private_o2 = np.asarray(curr_res['RL_private_o2']['cost'][-last_samples:])
		res_rl_private_o3 = np.asarray(curr_res['RL_private_o3']['cost'][-last_samples:])
		res_rl_private_o4 = np.asarray(curr_res['RL_private_o4']['cost'][-last_samples:])

		#print(res_fid[-1],res_rl[-1],res_rl_private_o1[-1],res_rl_private_o2[-1],res_rl_private_o3[-1])

		#min_rl = np.min(curr_res['RL']['cost'])
		#res_rl = [min_rl for el in range(L)]
		#res_rl = np.asarray(res_rl)
		"""
		for ii in range(L-1):
			min_ = np.min(curr_res['RL']['cost'][:ii+1])
			res_rl.append(min_)
		"""	
		
		#res_rl = np.divide((res_rl - res_fid),res_fid)
		#res_lid = np.divide((res_lid - res_fid),res_fid)
		#res_fid = np.divide((res_fid - res_fid),res_fid)

		final_lids.append(res_lid[-1])
		final_fids.append(res_fid[-1])
		final_rls.append(res_rl[-1])
		final_rls_private_o1.append(res_rl_private_o1[-1])
		final_rls_private_o2.append(res_rl_private_o2[-1])
		final_rls_private_o3.append(res_rl_private_o3[-1])
		final_rls_private_o4.append(res_rl_private_o4[-1])

		tot_lid += res_lid 
		tot_fid += res_fid 
		tot_rl += res_rl
		tot_rl_private_o1 += res_rl_private_o1
		tot_rl_private_o2 += res_rl_private_o2
		tot_rl_private_o3 += res_rl_private_o3
		tot_rl_private_o4 += res_rl_private_o4

		time_rl = 1#np.sum(curr_res['RL']['time']['orch']) + np.sum(curr_res['RL']['time']['isp'])
		time_rl_private = 1

		average_lid_time += curr_res['LID']['time']
		average_fid_time += curr_res['FID']['time']
		average_rl_time += time_rl
		average_rl_private_time += time_rl_private

		"""
		plt.plot(res_lid,label=label_lid)
		plt.plot(res_fid,label=label_fid)
		plt.plot(res_rl,label=label_rl)

		plt.legend()
		plt.show()
		"""

print(good_sims,'go')

tot_lid = np.asarray(tot_lid)/good_sims
tot_fid = np.asarray(tot_fid)/good_sims
tot_rl = np.asarray(tot_rl)/good_sims
tot_rl_private_o1 = np.asarray(tot_rl_private_o1)/good_sims
tot_rl_private_o2 = np.asarray(tot_rl_private_o2)/good_sims
tot_rl_private_o3 = np.asarray(tot_rl_private_o3)/good_sims
tot_rl_private_o4 = np.asarray(tot_rl_private_o4)/good_sims

#tot_rl = np.divide(tot_rl,tot_lid)
#tot_lid = np.divide(tot_lid,tot_lid)
#tot_fid = np.divide(tot_fid,tot_fid)

print('ratio rl',(tot_rl[-1] - tot_fid[-1])/float(tot_fid[-1]))
print('ratio rl private overhead 1',(tot_rl_private_o1[-1] - tot_fid[-1])/float(tot_fid[-1]))
print('ratio rl private overhead 2',(tot_rl_private_o2[-1] - tot_fid[-1])/float(tot_fid[-1]))
print('ratio rl private overhead 3',(tot_rl_private_o3[-1] - tot_fid[-1])/float(tot_fid[-1]))
print('ratio rl private overhead 4',(tot_rl_private_o4[-1] - tot_fid[-1])/float(tot_fid[-1]))
print('ration lid',(tot_lid[-1] - tot_fid[-1])/float(tot_fid[-1]))

average_lid_time/=float(good_sims)
average_fid_time/=float(good_sims)
average_rl_time/=float(good_sims)
average_rl_private_time/=float(good_sims)

#average_rl_time = average_rl_time/average_lid_time
#average_fid_time = average_fid_time/average_lid_time
#average_lid_time = 1 

label_lid = 'lid_' + str(average_lid_time)
label_fid = 'fid_' + str(average_fid_time)
label_rl = 'rl_' + str(average_rl_time)
label_rl_private_o1 = 'rl_private_o1' + str(average_rl_private_time)
label_rl_private_o2 = 'rl_private_o2' + str(average_rl_private_time)
label_rl_private_o3 = 'rl_private_o3' + str(average_rl_private_time)
label_rl_private_o4 = 'rl_private_o4' + str(average_rl_private_time)

plt.plot(tot_lid,label=label_lid)
plt.plot(tot_fid,label=label_fid)
plt.plot(tot_rl,label=label_rl)
plt.plot(tot_rl_private_o1,label=label_rl_private_o1)
plt.plot(tot_rl_private_o2,label=label_rl_private_o2)
plt.plot(tot_rl_private_o3,label=label_rl_private_o3)
plt.plot(tot_rl_private_o4,label=label_rl_private_o4)
plt.legend()
plt.show()

percentile_vec_lid = compute_percentile(final_lids)
percentile_vec_fid = compute_percentile(final_fids)
percentile_vec_rl = compute_percentile(final_rls)
percentile_vec_rl_private = compute_percentile(final_rls_private)

plt.plot(percentile_vec_lid,label='LID')
plt.plot(percentile_vec_fid,label='FID')
plt.plot(percentile_vec_rl,label='RL')
plt.plot(percentile_vec_rl_private,label='RL_private')
plt.legend()
plt.show()

counter_lid = 0
counter_fid = 0
counter_rl = 0
counter_rl_private = 0

for el_index in range(good_sims):
	els = [final_lids[el_index],final_fids[el_index],final_rls[el_index],final_rls_private[el_index]]
	#els = [final_lids[el_index],final_rls[el_index]]
	min_ = np.argmin(els)

	if min_ == 0:
		# min with LID
		counter_lid += 1
	elif min_ == 1:
		# min with FID
		counter_fid += 1
	elif min_ == 2:#2:
		# min with RL
		counter_rl += 1 

	elif min_ == 3:
		counter_rl_private += 1

values = np.asarray([counter_lid,counter_fid,counter_rl,counter_rl_private])
s = np.sum(values)
values = values/float(s)
values/=float(np.sum(values))

x = np.arange(4)
plt.bar(x, height= values)
plt.xticks(x+.5, ['FID','LID','RL','RL_private'])
plt.show()


