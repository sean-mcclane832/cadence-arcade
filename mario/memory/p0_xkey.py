"""AgrippaX (exploratory, Sean's idea): place memory keyed on x from game RAM. Build, parity vs Agrippa, key cosine."""
import sys, numpy as np
sys.path.insert(0, '.')
import mem
info = mem.transplant('AgrippaX', 'brains0/AgrippaX_ck0.npz', quiet=0.0); print(info)
par = {}
for L, s in (('1-1', 32), ('1-2', 35), ('1-2', 80), ('1-3', 85)):
    a = mem.play_attempt(mem.MemPlayer('brains0/Agrippa_ck0.npz', 'greedy'), L, s)
    b = mem.play_attempt(mem.MemPlayer('brains0/AgrippaX_ck0.npz', 'greedy'), L, s)
    par[f'{L}/{s}'] = (a['actions'] == b['actions'], a['max_x'], b['max_x'])
print('parity (same actions, x Agrippa, x AgrippaX):', par)
# key cosine on one real attempt (1-2/35), decisions g apart
obs = []
pl = mem.MemPlayer('brains0/AgrippaX_ck0.npz', 'greedy')
mem.play_attempt(pl, '1-2', 35, cb=lambda g, o, a: obs.append(o.copy()))
O = np.array(obs)
for nm in ('Agrippa', 'AgrippaSparse', 'AgrippaX'):
    p = mem.MemPlayer(f'brains0/{nm}_ck0.npz', 'greedy')
    X = O if nm == 'AgrippaX' else O[:, :2000]
    c = p.ep.place.separator.code(X); c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    print(nm, {g: round(float(np.mean([c[i] @ c[i + g] for i in range(len(c) - g)])), 3) for g in (1, 5, 20, 60)})
