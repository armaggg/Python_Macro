
# coding: utf-8

# In[ ]:


# stochastic/non-stochastic value function iteration

import numpy as np

import scipy

import scipy.optimize as sciopt

from scipy.optimize import minimize


from scipy import interpolate
from scipy import optimize

from numpy import linalg as LA

import matplotlib.pyplot as plt

import numpy as np

import pylab

import matplotlib.pyplot as plt

#value function of social planner's problem and its interpolation

def valuefunction(k):

    f = scipy.interpolate.interp1d(kmatrix, v_0,fill_value="extrapolate")

    gg=f(k)

#labor augmented per capita consumption function

    c = k_0**alpha - k + (1-dep)*k_0

#if c is negative we penalize it

    if c<=0:

        val = -9999999 - 999*abs(c)

    else:

      val=(1/(1-s))*(c**(1-s)-1) + beta*gg*shock

 

    return -val

#paramaters of the first batch without shock

shock=1 

alpha = 0.33 

beta = 0.95

dep= 0.1

s=2

tol = 0.01

#number of iterations

maxits = 10

dif = tol+1000

its=0

kgrid = 99

#steady state values (below)

kstar = (alpha/((1/beta)-(1-dep)))**(1/(1-alpha))

cstar = kstar**(alpha)-dep*kstar

istar = dep*kstar

ystar = kstar**(alpha)

#minimum and maximum capital

kmin = 0.25*kstar

kmax = 1.75*kstar

kgrid = 99

grid = (kmax-kmin)/kgrid

kmatrix=[kmin+i*(kmax-kmin)/kgrid for i in  range(int(kgrid))]

#initializing 

k_1_1=[i  for i in range(99)]

v_1=[i  for i in range(99)]

v_0=[0  for i in range(99)]

#we didn't use np.array to avoid the problems we have faced when we used it

#we get in a while loop until the value function converges or maximum number of iteration has reached. This is the  objective of this project: to see the value function converges 

 

while its<maxits and dif>tol:

    for i in range(99):

        k_0=kmatrix[i]

        k_1=optimize.fminbound(valuefunction,kmin,kmax)

        v_1[i]=valuefunction(k_1)

        k_1_1[i]=k_1

        

    a=np.asarray(v_1,dtype='f')

    b=np.asarray(v_0,dtype='f')

    

    dif=LA.norm(a-b)

        

    for i in range(99):

         v_0[i]=v_1[i]    

        

    its+=1

#paramaters of the second batch with a different discount factor without shock

shock=1 

alpha = 0.33 

beta = 0.85

dep= 0.1

s=2

tol = 0.01

maxits = 10

dif = tol+1000

its=0

kgrid = 99

 

kstar = (alpha/((1/beta)-(1-dep)))**(1/(1-alpha))

cstar = kstar**(alpha)-dep*kstar

istar = dep*kstar

ystar = kstar**(alpha)

 

kmin = 0.25*kstar

kmax = 1.75*kstar

kgrid = 99

grid = (kmax-kmin)/kgrid

kmatrix=[kmin+i*(kmax-kmin)/kgrid for i in  range(int(kgrid))]

 

k_1_1=[i  for i in range(99)]

v_2=[i  for i in range(99)]

v_0=[0  for i in range(99)]

 

while its<maxits and dif>tol:

    for i in range(99):

        k_0=kmatrix[i]

        k_1=optimize.fminbound(valuefunction,kmin,kmax)

        v_2[i]=valuefunction(k_1)

        k_1_1[i]=k_1

 

    a=np.asarray(v_2,dtype='f')

    b=np.asarray(v_0,dtype='f')

 

    dif=LA.norm(a-b)

 

    for i in range(99):

        v_0[i]=v_2[i]    

 

    its+=1

#paramaters of the third batch with a different discount factor without shock

shock=1 

alpha = 0.33 

beta = 0.75

dep= 0.1

s=2

tol = 0.01

maxits = 10

dif = tol+1000

its=0

kgrid = 99

 

kstar = (alpha/((1/beta)-(1-dep)))**(1/(1-alpha))

cstar = kstar**(alpha)-dep*kstar

istar = dep*kstar

ystar = kstar**(alpha)

 

kmin = 0.25*kstar

kmax = 1.75*kstar

kgrid = 99

grid = (kmax-kmin)/kgrid

kmatrix=[kmin+i*(kmax-kmin)/kgrid for i in  range(int(kgrid))]

 

k_1_1=[i  for i in range(99)]

v_3=[i  for i in range(99)]

v_0=[0  for i in range(99)]

 

while its<maxits and dif>tol:

    for i in range(99):

        k_0=kmatrix[i]

        k_1=optimize.fminbound(valuefunction,kmin,kmax)

        v_3[i]=valuefunction(k_1)

        k_1_1[i]=k_1

 

    a=np.asarray(v_3,dtype='f')

    b=np.asarray(v_0,dtype='f')

 
    dif=LA.norm(a-b)

 

    for i in range(99):

       v_0[i]=v_3[i]    

 

    its+=1

#paramaters of the first batch with realtively higher shock

shock=1.05 

alpha = 0.33 

beta = 0.95

dep= 0.1

s=2

tol = 0.01

maxits = 10

dif = tol+1000

its=0

kgrid = 99

 

kstar = (alpha/((1/beta)-(1-dep)))**(1/(1-alpha))

cstar = kstar**(alpha)-dep*kstar

istar = dep*kstar

ystar = kstar**(alpha)

 

kmin = 0.25*kstar

kmax = 1.75*kstar

kgrid = 99

grid = (kmax-kmin)/kgrid

kmatrix=[kmin+i*(kmax-kmin)/kgrid for i in  range(int(kgrid))]

 

k_1_1=[i  for i in range(99)]

v_4=[i  for i in range(99)]

v_0=[0  for i in range(99)]

 

while its<maxits and dif>tol:

    for i in range(99):

        k_0=kmatrix[i]

        k_1=optimize.fminbound(valuefunction,kmin,kmax)

        v_4[i]=valuefunction(k_1)

        k_1_1[i]=k_1

        

    a=np.asarray(v_4,dtype='f')

    b=np.asarray(v_0,dtype='f')

    

    dif=LA.norm(a-b)

        

    for i in range(99):

        v_0[i]=v_4[i]    

        

    its+=1

#paramaters of the first batch with relatively lower shock

shock=0.3 

alpha = 0.33 

beta = 0.95

dep= 0.1

s=2

tol = 0.01

maxits = 10

dif = tol+1000

its=0

kgrid = 99

 

kstar = (alpha/((1/beta)-(1-dep)))**(1/(1-alpha))

cstar = kstar**(alpha)-dep*kstar

istar = dep*kstar

ystar = kstar**(alpha)

 

kmin = 0.25*kstar

kmax = 1.75*kstar

kgrid = 99

grid = (kmax-kmin)/kgrid

kmatrix=[kmin+i*(kmax-kmin)/kgrid for i in  range(int(kgrid))]

 

k_1_1=[i  for i in range(99)]

v_5=[i  for i in range(99)]

v_0=[0  for i in range(99)]

 

while its<maxits and dif>tol:

    for i in range(99):

        k_0=kmatrix[i]

        k_1=optimize.fminbound(valuefunction,kmin,kmax)

        v_5[i]=valuefunction(k_1)

        k_1_1[i]=k_1

        

    a=np.asarray(v_5,dtype='f')

    b=np.asarray(v_0,dtype='f')

    

    dif=LA.norm(a-b)

        

    for i in range(99):

       v_0[i]=v_5[i]    

        

    its+=1

 

#we make muptiply them by -1 since we want to maximize and code is to minimize

v_1=[-v_1[i] for i in range(99)]

v_2=[-v_2[i] for i in range(99)]

v_3=[-v_3[i] for i in range(99)]

v_4=[-v_4[i] for i in range(99)]

v_5=[-v_5[i] for i in range(99)]

 

 

#consumption matrix

con=[kmatrix[i]**alpha-k_1_1[i]+(1-dep)*kmatrix[i] for i in range(99)]

 

pylab.plot(kmatrix, v_1, '-b', label='Beta=.95')

pylab.plot(kmatrix, v_2, '-r', label='Beta=.85')

pylab.plot(kmatrix, v_3, '-y', label='Beta=.75')

pylab.legend(loc='lower right')

plt.xlabel('k')

plt.ylabel('V(k)')

plt.draw()

 

plt.figure()

plt.plot(kmatrix,con)

plt.xlabel('k')

plt.ylabel('c')

plt.draw()

 

plt.figure()

pylab.plot(kmatrix, v_1, '-b', label='No shock')

pylab.plot(kmatrix, v_4, '-r', label='Shock=1.05')

pylab.plot(kmatrix, v_5, '-y', label='Shock=.3')

pylab.legend(loc='lower right')

plt.xlabel('k')

plt.ylabel('V(k)')

plt.figtext(.15, 0.8, "Beta=0.95")

plt.draw()

plt.show()

