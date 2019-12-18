import numpy as np 

r = np.load('results_simulations.npy').item()
print(1)
r2 = np.load('results_simulations_2.npy').item()
print(2)
r3 = np.load('results_simulations_3.npy').item()
print(3)
r4 = np.load('results_simulations_4.npy').item()
print(4)

counter = 0 
merged_r = {} 
for el in r:
	merged_r[counter] = r[el]
	counter += 1 

for el in r2:
	merged_r[counter] = r2[el]
	counter += 1 

for el in r3:
	merged_r[counter] = r3[el]
	counter += 1 

for el in r4:
	merged_r[counter] = r4[el]
	counter += 1 

np.save('merged_results',merged_r)