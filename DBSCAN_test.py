from sklearn import datasets
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import scipy.io as sio
import h5py
def findNeighbor(j,X,eps):
    N=[]
    for p in range(X.shape[0]):   #找到所有领域内对象
        temp=np.sqrt(np.sum(np.square(X[j]-X[p])))   #欧氏距离
        if(temp<=eps):
            N.append(p)
    return N


def dbscan(X,eps,min_Pts):
    k=-1
    NeighborPts=[]      #array,某点领域内的对象
    Ner_NeighborPts=[]
    fil=[]                                      #初始时已访问对象列表为空
    gama=[x for x in range(len(X))]            #初始时将所有点标记为未访问
    cluster=[-1 for y in range(len(X))]
    while len(gama)>0:
        j=random.choice(gama)
        gama.remove(j)  #未访问列表中移除
        fil.append(j)   #添加入访问列表
        NeighborPts=findNeighbor(j,X,eps)
        if len(NeighborPts) < min_Pts:
            cluster[j]=-1   #标记为噪声点
        else:
            k=k+1
            cluster[j]=k
            for i in NeighborPts:
                if i not in fil:
                    gama.remove(i)
                    fil.append(i)
                    Ner_NeighborPts=findNeighbor(i,X,eps)
                    if len(Ner_NeighborPts) >= min_Pts:
                        for a in Ner_NeighborPts:
                            if a not in NeighborPts:
                                NeighborPts.append(a)
                    if (cluster[i]==-1):
                        cluster[i]=k
    print(k)
    return cluster


# X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
#                                        noise=.05)
# X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
#                 random_state=9)
# X = np.concatenate((X1, X2))

#引入mat文件的数据
mat = h5py.File(r"pt.mat")
# 可以用keys方法查看cell的名字
print(mat.keys())
# 可以用values方法查看各个cell的信息
# print(mat.values())
# 可以用shape查看维度信息
print(mat['pt'].shape)
# 注意，这里看到的shape信息与你在matlab打开的不同
# 这里的矩阵是matlab打开时矩阵的转置
# 所以，我们需要将它转置回来
mat_t = np.transpose(mat['pt'])
# mat_t 是numpy.ndarray格式
# 再将其存为npy格式文件
np.save('pt.npy', mat_t)
matrix = np.load('pt.npy')
X=matrix


eps=0.08
min_Pts=10
begin=time.time()
C=dbscan(X,eps,min_Pts)
end=time.time()
dbscan_time=end-begin
print (dbscan_time)
plt.figure(figsize=(12, 9), dpi=80)
plt.scatter(X[:,0],X[:,1],c=C)
plt.show()

