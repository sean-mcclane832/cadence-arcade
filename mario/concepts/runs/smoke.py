import sys, time; sys.path.insert(0, '.'); sys.path.insert(0, '..')
import numpy as np, cplay as C
from cadence import Brain
b = Brain.compose(C.n_inputs('ram'), 7, modules=(256,), seed=0, teaching='normalized', learning_surprise=False)
b.save('runs/smoke_ram.npz')
t = time.time()
r = C.episode(C.task('runs/smoke_ram.npz', 'ramscreen', '1-1', 0, mode='dagger', pol_senses='ram', last_all=5, every=20, tiew=True, keep_obs=True))
print(r['end'], r['max_x'], r['obs'].shape, r['labels'], r['own_obs'].shape, time.time() - t)
e = C.episode(C.task('runs/smoke_ram.npz', 'ram', '1-1', 0))
print(e['end'], e['max_x'], e['actions'][:10] == r['actions'][:10])
