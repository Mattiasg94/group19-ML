import matplotlib.pyplot as plt

import numpy as np
# ! -- Setup -- ! #
dt=10

B=np.array([[1/dt,0],[0,1/dt]])
P=np.array([[10,0],[0,5]])
rd=[4,3,1]
r0=[3,2,1]
x=np.array([[3],[2]])
n=20
test_controller=False
# ! -- Setup -- ! #
T=[]
i=0
if test_controller:
    for t in range(n):
        if t==10:
            i=1
        elif t==20:
            i==2
        rk=np.reshape([rd[i],r0[i]],(2,1))
        xk=np.reshape(x[:,t],(2,1))
        dx=np.dot(np.dot(B,P),(rk-xk))
        x_k1=xk+dx
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


def P_controller(d,theta,p_station,states_lst):
    global x
    if p_station>0.8:
        xk=np.reshape([d,theta],(2,1))
    else:
        xk=np.reshape(states_lst[:,-1],(2,1))
    rk=np.reshape([0,0],(2,1))
    
    dx=np.dot(np.dot(B,P),(rk-xk))
    x_k1=xk+dx
    x=np.append(x,x_k1,axis=1)
    return x,dx