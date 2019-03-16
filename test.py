import numpy as np

population= np.arange(30)

for i in range(len(population[:10])):
    for j in range(i+1, len(population[:10])):
        print(population[i],population[j])