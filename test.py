import numpy as np

a=np.zeros([128])
b=np.zeros([523,24,30])
# c=np.concatenate((a,b))
b=b[:,-12:,:]
print(b.shape)

