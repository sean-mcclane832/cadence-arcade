"""How spatially smooth are the learned input weights? Neighbour correlation of each unit's weight map
(1 = smooth blobs, 0 = salt and pepper). From scratch/weights_<run>_mats.npz."""
import sys, json, numpy as np
out = {}
for run in sys.argv[1:]:
    W = np.load(f"scratch/weights_{run}_mats.npz")["Wsa"]
    def nb(a, b, h, w):
        m = W[a:b].T.reshape(-1, h, w)
        cx = np.mean([np.corrcoef(u[:, :-1].ravel(), u[:, 1:].ravel())[0, 1] for u in m])
        cy = np.mean([np.corrcoef(u[:-1].ravel(), u[1:].ravel())[0, 1] for u in m])
        return round(float(cx), 3), round(float(cy), 3)
    now, prev = W[0:832], W[832:1664]
    out[run] = dict(periph_now=nb(0, 832, 26, 32), fov_R=nb(2000, 2384, 16, 24), fov_prev=nb(3152, 3536, 16, 24),
                    now_vs_prev_same_cell=round(float(np.mean([np.corrcoef(now[:, u], prev[:, u])[0, 1] for u in range(W.shape[1])])), 3),
                    fovR_vs_fovG=round(float(np.mean([np.corrcoef(W[2000:2384, u], W[2384:2768, u])[0, 1] for u in range(W.shape[1])])), 3))
print(json.dumps(out, indent=1)); json.dump(out, open("scratch/smooth.json", "w"), indent=1)
