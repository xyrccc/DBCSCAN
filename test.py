import numpy as np
X = np.load('pt.npy')
temp=np.sqrt(np.sum(np.square(X[1]-X[0])))
temp1=X[1]-X[0]
temp2=np.square(X[1]-X[0])
temp3=np.sum(np.square(X[1]-X[0]))
print(X[1])
print(X[0])
print(temp1)
print(temp2)
print(temp3)