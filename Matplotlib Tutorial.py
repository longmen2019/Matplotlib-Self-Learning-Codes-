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


# In[10]:


line = plt.plot(x,y,'-')
#turned off antialiasing
line.set_antialiased(False)


# In[ ]:




