import sys, numpy as np; sys.path.insert(0,'.')
from cadence import Brain
b=Brain.load('brains0/Livy_ck0.npz')
pre=np.asarray(b.connectome.pre); hist=np.asarray(b.connectome.populations['history'])
r=np.ones(len(pre)); m=np.isin(pre,hist); r[m]=0.125
b.learner.synapse_rate=r; b.save('brains0/Livy_ck0.npz')
b2=Brain.load('brains0/Livy_ck0.npz'); print('rate kept', b2.learner.synapse_rate is not None and float(b2.learner.synapse_rate[m].mean()), int(m.sum()))
