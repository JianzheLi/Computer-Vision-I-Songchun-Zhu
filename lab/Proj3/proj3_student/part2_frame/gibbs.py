'''
This is the tool file of Part 2: gibbs.py
'''


import numpy as np
from tqdm import tqdm


def gibbs_sample(img_syn, filter_list, lambdas, bounds, num_bins, sweep, T):
    '''
    Gibbs sampling for FRAME model.
    Uses energy function: -sum(lambda_n * phi_n(x))
    '''
    H, W = img_syn.shape
    num_filters = len(filter_list)
    
    print(" ---- GIBBS SAMPLING FOR FRAME ---- ")
    
    # Precompute bin widths
    bin_widths = np.zeros(num_filters)
    bin_mins = np.zeros(num_filters)
    for fi in range(num_filters):
        max_resp, min_resp = bounds[fi]
        rng = max_resp - min_resp
        if rng == 0:
            rng = 1e-6
        bin_widths[fi] = rng / num_bins
        bin_mins[fi] = min_resp
    
    def resp_to_bin(fi, resp):
        idx = int((resp - bin_mins[fi]) / bin_widths[fi])
        return np.clip(idx, 0, num_bins - 1)
    
    def compute_response_at(fi, rh, rw, img):
        filt = filter_list[fi]
        Kh, Kw = filt.shape
        ctr_h = Kh // 2
        ctr_w = Kw // 2
        resp = 0.0
        for u in range(Kh):
            for v in range(Kw):
                ih = rh + (u - ctr_h)
                iw = rw + (v - ctr_w)
                if 0 <= ih < H and 0 <= iw < W:
                    resp += filt[u, v] * img[ih, iw]
        return resp
    
    current_T = T
    
    for s in tqdm(range(sweep)):
        for pos_h in range(H):
            for pos_w in range(W):
                old_val = img_syn[pos_h, pos_w]
                candidates = np.arange(8, dtype=np.float32)
                
                energies = np.zeros(len(candidates))
                
                for ci, cand in enumerate(candidates):
                    delta = float(cand - old_val)
                    if delta == 0:
                        energies[ci] = 0.0
                        continue
                    
                    energy = 0.0
                    
                    # Compute energy change for each filter
                    for fi in range(num_filters):
                        filt = filter_list[fi]
                        Kh, Kw = filt.shape
                        ctr_h = Kh // 2
                        ctr_w = Kw // 2
                        
                        for u in range(Kh):
                            for v in range(Kw):
                                wuv = filt[u, v]
                                if wuv == 0:
                                    continue
                                
                                rh = pos_h - (u - ctr_h)
                                rw = pos_w - (v - ctr_w)
                                
                                if not (0 <= rh < H and 0 <= rw < W):
                                    continue
                                
                                # Temporarily set pixel to candidate
                                img_syn[pos_h, pos_w] = cand
                                resp_new = compute_response_at(fi, rh, rw, img_syn)
                                img_syn[pos_h, pos_w] = old_val
                                resp_old = compute_response_at(fi, rh, rw, img_syn)
                                
                                bin_new = resp_to_bin(fi, resp_new)
                                bin_old = resp_to_bin(fi, resp_old)
                                
                                # Energy contribution: -lambda * phi (where phi is indicator)
                                if bin_old != bin_new:
                                    energy -= lambdas[fi, bin_old]  # Remove old contribution
                                    energy += lambdas[fi, bin_new]  # Add new contribution
                    
                    energies[ci] = energy
                
                # Convert to probabilities
                probs = np.exp(-energies / current_T)
                probs = probs / (probs.sum() + 1e-12)
                
                # Sample new value
                new_val = np.random.choice(candidates, p=probs)
                img_syn[pos_h, pos_w] = new_val
        
        current_T *= 0.96
        
        if (s + 1) % 10 == 0:
            print(f'Gibbs sweep {s+1}/{sweep} completed, T={current_T:.4f}')
        
    return img_syn
