import scipy as sc
import pandas as pd
import scipy.io as sio 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA as sklearnPCA


test = sio.loadmat('kmeans_data.mat')
data = np.array(test['data'])
frame = pd.DataFrame(data)

number_of_clusters = []
objective_function = []

for k_clusters in range(2,11,1):
    clusters = np.array(frame.sample(k_clusters))
    tempC = np.zeros(clusters.shape)
    count = np.zeros((clusters.shape[0]))
    iterations =0
    original_clusters = clusters

    instances = np.array(frame)

    assigned_cluster = np.zeros((instances.shape[0],1))
    objective_fn = 0

    #print(clusters)

    z = np.zeros(clusters.shape)
    while True:
        instance_index = 0
        count = np.zeros((clusters.shape[0]))
        tempC = np.zeros(clusters.shape)
        objective_fn = 0
        iterations+=1
        for k in instances:
            #print(k)
            #print(clusters)
            z = np.subtract(clusters,k)
            #print(z)
            z = np.square(z)
            #print(z)
            z = np.sum(z, axis =1)
            #print(z)
            objective_fn+=np.amin(z)
            dist_index = np.argmin(z)
            assigned_cluster[instance_index] = dist_index
            instance_index+=1
            #print(dist_index)
            tempC[dist_index] = tempC[dist_index] + k
            count[dist_index]+=1
        z1 = tempC / count.reshape(count.shape[0],1)
        #print(z1)
        #print(clusters)
        #clusters = z1
        if((clusters == z1).all()):
            break
        else:
            clusters = z1


    print(k_clusters)
    print(objective_fn)
    #print (count)

    np.round((clusters - original_clusters),2)
    number_of_clusters.append(k_clusters)
    objective_function.append(objective_fn)



plt.plot(number_of_clusters,objective_function)
plt.title("KMeans Algorithm")
plt.xlabel('Number of Clusters')
plt.ylabel('Objective Function Values(KMean)')
plt.show()

# Visualization of clusters using PCA

# pca = sklearnPCA(n_components=2)
# transformed = pd.DataFrame(pca.fit_transform(instances))
# df = pd.DataFrame(np.zeros(assigned_cluster.shape))
# fig, ax = plt.subplots()
# colors = {0:'red', 1:'blue',2:'yellow',3:'green',4:'black',5:''}
# df['color'] = pd.DataFrame(assigned_cluster)
# ax.scatter(transformed[0], transformed[1], c=df['color'].apply(lambda x: colors[x]))
# #plt.scatter(,, c = "'red' if 1")
# plt.show()





