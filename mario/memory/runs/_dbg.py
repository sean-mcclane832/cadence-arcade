import sys, numpy as np
sys.path.insert(0,'.')
import mem
pl=mem.MemPlayer('brains0/Agrippa_ck0.npz','retry',0.05)
b=pl.b
print(b.describe()['genes']['actor_eta'], b.describe()['genes']['actor_eta_bias'], b.describe()['genes']['actor_eta_critic'], b.describe()['genes'].get('learning'))
g=b.learner.brain
s0=dict(eff=g.efficacy.copy(), bias=g.bias.copy(), lg=g.log_gain.copy(), wc=b.basal_ganglia.w_critic.copy(), bc=np.copy(b.basal_ganglia.b_critic))
r=mem.play_attempt(pl,'1-2',35,seed=1,)
g=b.learner.brain
s1=dict(eff=g.efficacy, bias=g.bias, lg=g.log_gain, wc=b.basal_ganglia.w_critic, bc=b.basal_ganglia.b_critic)
for k in s0:
    d=np.asarray(s1[k])-np.asarray(s0[k]); print(k, 'changed', int((d!=0).sum()), 'maxabs', float(np.abs(d).max()))
for k in s0:
    a=np.ascontiguousarray(np.asarray(s0[k],dtype=np.float64)); c=np.ascontiguousarray(np.asarray(s1[k],dtype=np.float64))
    print(k,'bytes equal', a.tobytes()==c.tobytes(), 'nan', np.isnan(a).sum(), 'negzero', int((np.signbit(a)!=np.signbit(c)).sum()))
