#!/usr/bin/env python
# coding: utf-8

# In[1]:


import loss, rpca
from rpca import data

# Load "Sleep in Mammals" database
X = rpca.data.load_sleep()

# Transform it using Robust PCA
huber_loss = loss.HuberLoss(delta=1)
rpca_transformer = rpca.MRobustPCA(2, huber_loss)
X_rpca = rpca_transformer.fit_transform(X)


# In[2]:


import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')

# Plot progress during iterations
plt.figure(figsize=(8, 5))
plt.plot(range(1, rpca_transformer.n_iterations_ + 1), rpca_transformer.errors_, '-o')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

