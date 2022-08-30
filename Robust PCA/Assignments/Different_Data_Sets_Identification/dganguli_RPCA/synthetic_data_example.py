#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dganguli import *

# Generate low rank synthetic data:
N = 100
num_groups = 3
num_values_per_group = 40
p_missing = 0.2

Ds = []
for k in range(num_groups):
    d = np.ones((N, num_values_per_group)) * (k + 1) * 10
    Ds.append(d)

D = np.hstack(Ds) # Dimensions of D: 100x120
                  # 10s: (100, 40), 20s: (100, 40), 30s: (100, 40)

# Considers some data missing and scraps it:
n1, n2 = D.shape
S = np.random.rand(n1, n2)
D[S < p_missing] = 0

# Use R_pca to estimate the degraded data as L + S,
# where L is low rank, and S is sparse:
rpca = R_pca(D)
L, S = rpca.fit(max_iter=10000, iter_print=100)


# In[2]:


# To find out the missing data:

D2 = D.copy()

D2.resize((D2.size)) # D2.shape --> (12000,)

from collections import Counter

dict(Counter(D2.tolist()))[0.0]


# In[3]:


# Visually inspect results:
rpca.plot_fit()
# plt.show()

