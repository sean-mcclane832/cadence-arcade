"""Records-only check for AgrippaX: 10 retry lives (keep, + our blame window) on 1-2/32, weights hashed before and after."""
import sys, hashlib, numpy as np
sys.path.insert(0, '.')
import mem
def h(b):
    g = b.learner.brain; s = hashlib.sha1()
    for a in (g.efficacy, g.bias, g.log_gain, b.basal_ganglia.w_critic, b.basal_ganglia.b_critic):
        s.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return s.hexdigest()
pl = mem.MemPlayer('brains0/AgrippaX_ck0.npz', 'retry', 0.05, (), (5, 30, -1.0))
h0 = h(pl.b); xs = []
for i in range(10):
    r = mem.play_attempt(pl, '1-2', 32 + mem.RETRY[i], seed=1000 * 32 + i); xs.append(r['max_x'])
print('lives x', xs, 'place writes', getattr(pl.ep.place, 'writes', '?'))
print('weights byte-identical after 10 lives:', h(pl.b) == h0)
