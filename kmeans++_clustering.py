import scipy as sc
import pandas as pd
import scipy.io as sio 
import matplotlib.pyplot as plt
import numpy as np

test = sio.loadmat('kmeans_data.mat')
data = np.array(test['data'])
frame = pd.DataFrame(data)
z = np.zeros(data.shape)
def Clusters(num_of_clusters,clusters):
    temp = np.zeros(data.shape)
    for i in range(0,num_of_clusters-1):
        z = np.subtract(clusters[i],data)
        z = np.square(z)
        z = np.sum(z, axis =1)
        if(i > 0):
             z = np.minimum(temp,z)
        #print(np.argmax(z))
        index = np.argmax(z)
        #Indexs.append(index)
        clusters = np.concatenate((clusters,data[index].reshape(1,21)))
        temp = z
    
    #print(np.max(z))
    return clusters        

number_of_clusters = []
objective_function = []

for k_clusters in range(2,11,1):
    clusters = np.array(frame.sample(1))
    clusters = Clusters(k_clusters-1,clusters)


    tempC = np.zeros(clusters.shape)
    count = np.zeros((clusters.shape[0]))
    iterations =0
    original_clusters = clusters
    instances = np.array(frame)
    instances.shape
    assigned_cluster = np.zeros((instances.shape[0],1))
    objective_fn = 0

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
    number_of_clusters.append(k_clusters)
    objective_function.append(objective_fn)

plt.plot(number_of_clusters,objective_function)
plt.title("KMeans ++ Algorithm")
plt.xlabel('Number of Clusters')
plt.ylabel('Objective Function Values(KMeans++)')
plt.show()


# Visualization of clusters using PCA


# from sklearn.decomposition import PCA as sklearnPCA
# pca = sklearnPCA(n_components=2)
# transformed = pd.DataFrame(pca.fit_transform(instances))
# A = []
# B = []
# for index in Indexs:
#     A.append(round(float(transformed.get_value(index=index,col = 0)),3))
#     B.append(round(float(transformed.get_value(index=index,col = 1)),3))
#
# print(A, '   ',B)
#
#
# fig, ax = plt.subplots()
#
# #A = round(float(transformed.get_value(index=718,col = 0)),3),round(float(transformed.get_value(index=6950,col = 0)),3),round(float(transformed.get_value(index=2448,col = 0)),3),
# #B = round(float(transformed.get_value(index=718,col = 1)),3),round(float(transformed.get_value(index=6950,col = 1)),3),round(float(transformed.get_value(index=2448,col = 1)),3),
# # B = 0.73, 0.97, 1.0, 0.97, 0.88, 0.73, 0.54
# plt.plot(A,B)
# for xy in zip(A, B):                                       # <--
#     ax.annotate('Centers', xy=(xy), textcoords='data',color='green', xytext=(0.05, 0.05),
#                 horizontalalignment='left',
#             verticalalignment='bottom',arrowprops=dict(facecolor='black', shrink=0.05),) # <--
#
#
# df = pd.DataFrame(np.zeros(assigned_cluster.shape))
#
# colors = {0:'red', 1:'blue',2:'yellow',3:'green',4:'black'}
# df['color'] = pd.DataFrame(assigned_cluster)
# ax.scatter(transformed[0], transformed[1], c=df['color'].apply(lambda x: colors[x]),s = 3)
# #plt.scatter(,, c = "'red' if 1")
# plt.show()

