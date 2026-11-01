import sys, numpy as np; sys.path.insert(0,'.')
import mem
mem.transplant('AgrippaSparse','brains0/AgrippaSparse_ck0.npz',quiet=0.0)
z=np.load('runs/seq/seq_1-2_coach.npz'); O=z['O'][:z['off'][1]].astype(float)/255
for nm in ('Agrippa','AgrippaSparse'):
    pl=mem.MemPlayer(f'brains0/{nm}_ck0.npz','greedy'); sep=pl.ep.place.separator
    c=sep.code(O); c=c/np.maximum(np.linalg.norm(c,axis=1,keepdims=True),1e-12)
    n=len(c); out={}
    for gap in (1,5,20,60):
        out[gap]=round(float(np.mean([c[i]@c[i+gap] for i in range(n-gap)])),3)
    print(nm, 'mean key cosine at decision gaps', out)
