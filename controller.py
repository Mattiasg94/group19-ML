import matplotlib.pyplot as plt

import numpy as np
# ! -- Setup -- ! #
dt=10

B=np.array([[1/dt,0],[0,1/dt]])
P=np.array([[1,0],[0,1]])

l=10
rd1=[3 for i in range(l)]
r01=[2 for i in range(l)]

rd2=[2 for i in range(l)]
r02=[1 for i in range(l)]

rd3=[2 for i in range(l)]
r03=[1 for i in range(l)]

rd4=[4 for i in range(l)]
r04=[3 for i in range(l)]

rd_lst=[rd1,rd2,rd3,rd4]
r0_lst=[r01,r02,r03,r04]

x=np.array([[3],[2]])

# ! -- Setup -- ! #
T=[]
rd=rd1
r0=r01
for i,r_state in enumerate(rd_lst):
    rd=np.concatenate((rd,rd_lst[i+1]))
    if i==len(rd_lst)-2:
        break
for i,r_state in enumerate(r0_lst):
    r0=np.concatenate((r0,r0_lst[i+1]))
    if i==len(r0_lst)-2:
        break

r=np.array([rd,r0])

n=int(np.size(r)/2)

for t in range(n):
    rk=np.reshape(r[:,t],(2,1))
    xk=np.reshape(x[:,t],(2,1))
    x_delta=np.dot(np.dot(B,P),(rk-xk))
    x_k1=xk+x_delta
    x=np.append(x,x_k1,axis=1)
    T.append(t)
T=np.array(T)

plt.style.use("ggplot")
plt.figure()
plt.plot(x[0,:],label='d')
plt.plot(x[1,:],label='theta')
plt.title("Controller output")
plt.xlabel("time t="+str(n))
plt.ylabel("distance/angle")
plt.legend(loc="upper left")
plt.show()