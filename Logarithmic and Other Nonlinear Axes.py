#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import matplotlib.pyplot as plt
x = np.linspace (-np.pi , np.pi, 256 , endpoint = True)
c, s = np.cos(x) , np.sin(x)

plt.plot()


# In[3]:


import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi , np.pi, 256 , endpoint= True)
c,s = np.cos(x), np.sin(x)
plt.plot(x,c)
plt.plot(x,s)
plt.show()


# In[3]:


import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.ylabel('Some Numbers')
plt.show()


# In[4]:


plt.plot([1,2,3,4] , [4,9,14,19])


# In[6]:


plt.plot([1,2,3,4] , [9,11,25,32] , 'ro')
plt.axis([0,6,0,20])
plt.show()


# In[9]:


import matplotlib.pyplot as plt
import numpy as np
#evenl sample times at 200ms intervals
t = np.arange(0. , 5. , 0.2)
# red dash, blue square and green triangle
plt.plot(t,t,'r--' , t,t**2, 'bs' , t, t**3, 'g^')
plt.show()


# In[7]:


import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t)*np.cos(2*np.pi*t)
t1 = np.arange(0.0,5.0,0.1)
t2 = np.arange(0.0,5.0,0.02)

plt.figure(1)
plt.subplot(211)

plt.plot(t1 , f(t1) , 'bo' , t2 , f(t2) , 'k') 
plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()


# In[6]:


import matplotlib.pyplot as plt
import numpy as np
#create some data to use for the plot
dt = 0.001
t = np.arange(0.0 , 10.0 , dt)
r = np.exp(-t[:1000]/0.05)
x = np.random.randn(len(t))
s = np.convolve(x,r) [:len(x)]*dt
#color noised
#the main axes is subject (111) by default
plt.plot(t,s)
plt.axis ([0,1,1.1*np.amin(s) , 2*np.amax(s)])
plt.xlabel('time(s)')
plt.ylabel('current(rA)')
plt.title('Gaussian colored noise')

#This is an inset axes over the main axes
a = plt.axes([.65 , .6 , .2 , .2] , facecolor = 'y')
#Changed from normed = 1 to density = 1
n, bins, patches = plt.hist(s,400, density=1)
plt.title('Probability')
plt.xticks([])
plt.yticks([])
#this is another inset axes over the main axes
a = plt.axes ([0.2 , 0.6 , .2 , .2], facecolor = 'y')
plt.plot(t[:len(r)],r)
plt.title('Impulse response')
plt.xlim(0,0.2)
plt.xticks([])
plt.yticks([])


# In[9]:


import matplotlib.pyplot as plt
import numpy as np
# fixiing random state for reproductability 
np.random.seed(19680801)
mu , sigma = 100,15
x = mu + sigma*np.random.randn(1000)
#The histogram of the date
# Change from normed = 1 to density = 1
n, bins, patches = plt.hist(x, 50 , density=1, facecolor = 'g' , alpha = 0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma = 15$')
plt.axis([40 , 160 , 0 , 0.03])
plt.grid(True)
plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

ax = plt.subplot(111)

t = np.arange(0.0 , 5.0 , 0.01)
s = np.cos(2*np.pi*t)
line , = plt.plot(t, s, lw=2)

plt.annotate('local max' , xy =(2,1) , xytext = (3 , 1.5), 
            arrowprops=dict(facecolor='black' , shrink = 0.05),
            )
plt.ylim(-2,2)
plt.show()


# In[1]:


import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter #useful for 'logit' scale

#fixing random state for reproducibility
np.random.seed(19680801)
#make up some data in the interval ]0,1[
y = np.random.normal(loc = 0.5 , scale = 0.4 , size = 1000)
y = y [(y > 0) & (y<1)]
y.sort()
x = np.arange(len(y))

#plot with various axes scales
plt.figure(1)

#linear
plt.subplot(221)
plt.plot(x,y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)

#log
plt.subplot(222)
plt.plot(x,y)
plt.yscale('log')
plt.title('log')
plt.grid(True)

#symmetric log
plt.subplot(223)
plt.plot(x , y - y.mean())
plt.yscale('symlog' , linthreshy=0.01)
plt.title('symlog')
plt.grid(True)

#logit
plt.subplot(224)
plt.plot(x,y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)
#Format the minor tick labels of the y-axis into empty strings sith 
#'NullFormatter' , to avoid cumbering the axis with too many labels. 
plt.gca().yaxis.set_minor_formatter(NullFormatter())
#Adjust the subplot layout, because the logit one may take more space
#than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10 , right=0.95 , hspace=0.25, wspace=0.35)
plt.show()


# In[ ]:




