"""Check (11/2 end of night): what does `gauntletm.py --mode act` (used for m05, m06) actually do?
MemPlayer only special-cases 'greedy' and 'retry', so 'act' falls into the sampled branch (temperature 0.05) and calls
learn() every decision WITHOUT the frozen actor rates. Test: replay m06's first two attempts (1-1 start 0, 1-2 start 0)
the gauntlet way, hash the weights, and compare with greedy play."""
import sys, hashlib, numpy as np
sys.path.insert(0, '.')
import mem
def h(b):
    g = b.learner.brain; s = hashlib.sha1()
    for a in (g.efficacy, g.bias, g.log_gain, b.basal_ganglia.w_critic, b.basal_ganglia.b_critic):
        s.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return s.hexdigest()[:12]
P = 'runs/gauntlet_night/m06_Cicero_joint.npz'
pl = mem.MemPlayer(P, 'act', 0.05); h0 = h(pl.b)
r1 = mem.play_attempt(pl, '1-1', 0, seed=1000); h1 = h(pl.b)
r2 = mem.play_attempt(pl, '1-2', 0, seed=1001); h2 = h(pl.b)
print('mode act: 1-1', r1['max_x'], r1['flag'], r1['decisions'], '| 1-2', r2['max_x'], r2['flag'], r2['decisions'])
print('weights hash before / after 1-1 / after 1-2:', h0, h1, h2, '-> changed' if h0 != h2 else '-> unchanged')
g = mem.MemPlayer(P, 'greedy'); a = mem.play_attempt(g, '1-1', 0); b = mem.play_attempt(g, '1-2', 0)
print('greedy: 1-1', a['max_x'], a['flag'], a['decisions'], '| 1-2', b['max_x'], b['flag'], b['decisions'], 'hash', h(g.b))
print('m06 log: 1-1 flag 306 decisions | 1-2 died x 1344, 150 decisions')
