import numpy as np
from tqdm import tqdm
import gibbs_optimized as cy
def gibbs_sample(img_syn, hists_syn,
                 img_ori, hists_ori,
                 filter_list, sweep, bounds,
                 T, weight, num_bins):
    H,W = img_syn.shape
    num_chosen_filters = len(filter_list)
    print(" ---- GIBBS SAMPLING ---- ")
    ###cython 
    bin_widths = (bounds[:,0] - bounds[:,1]) / num_bins   # 你 Cython 优化版本需要的
    img_syn = np.ascontiguousarray(img_syn, dtype=np.float32)
    img_ori = np.ascontiguousarray(img_ori, dtype=np.float32)

    # histograms: shape should be (num_filters_selected, num_bins)
    hists_syn = np.ascontiguousarray(hists_syn, dtype=np.float32)
    hists_ori = np.ascontiguousarray(hists_ori, dtype=np.float32)

    # bounds: (num_filters_selected, 2) where each row is (max, min)
    bounds = np.ascontiguousarray(bounds, dtype=np.float32)

    # weight: (num_bins,)
    weight = np.ascontiguousarray(weight, dtype=np.float32)

    # filters: make sure each kernel is float32 and contiguous
    filter_list = [np.ascontiguousarray(f, dtype=np.float32) for f in filter_list]

    # bin widths required by cython (float32)
    # bounds[:,0] is max, bounds[:,1] is min as your code uses
    bin_widths = np.ascontiguousarray((bounds[:, 0] - bounds[:, 1]) / float(num_bins), dtype=np.float32)

    #################Cython 
    for s in tqdm(range(sweep)):
        for pos_h in range(H):
            for pos_w in range(W):
                
                cy.pos_gibbs_sample_update_py(
                    img_syn,
                    hists_syn,
                    hists_ori,
                    filter_list,
                    bounds,
                    weight,
                    bin_widths,
                    pos_h, pos_w,
                    T
                )
        """for s in tqdm(range(sweep)):
        for pos_h in range(H):
            for pos_w in range(W):
                pos = [pos_h,pos_w]
                img_syn,hists_syn = pos_gibbs_sample_update(img_syn,hists_syn,img_ori,hists_ori,filter_list,bounds,weight,pos,num_bins,T)"""
        # compute current error summary
        errors = np.abs(hists_syn - hists_ori) * weight  # shape: [num_filters, num_bins]
        max_error = errors.sum(axis=1).max()
        print(f'Gibbs iteration {s+1}: mean_error = {errors.sum(axis=1).mean()} max_error: {max_error}')
        T = T * 0.96
        if max_error < 0.1:
            print(f"Gibbs iteration {s+1}: max_error: {max_error} < 0.1, stop!")
            break
    return img_syn, hists_syn

def pos_gibbs_sample_update(img_syn, hists_syn,
                            img_ori, hists_ori,
                            filter_list, bounds,
                            weight, pos,
                            num_bins, T):
    '''
    See your docstring above.
    Implementation uses local updates described in the explanation.
    '''
    H = img_syn.shape[0]
    W = img_syn.shape[1]
    pos_h = pos[0]
    pos_w = pos[1]

    # candidates: use unique intensities from original image (faster and sensible)
    candidates = np.unique(img_ori).astype(img_syn.dtype)
    old_val = img_syn[pos_h, pos_w]

    # precompute bin widths/ranges for each filter
    num_filters = len(filter_list)
    bin_mins = np.zeros(num_filters, dtype=float)
    bin_widths = np.zeros(num_filters, dtype=float)
    for fi in range(num_filters):
        max_r = bounds[fi][0]
        min_r = bounds[fi][1]
        # guard against degenerate range
        rng = max_r - min_r
        if rng == 0:
            rng = 1e-6
        bin_mins[fi] = min_r
        bin_widths[fi] = rng / num_bins

    # helper: map response value -> bin index
    def resp_to_bin(fi, resp):
        idx = int((resp - bin_mins[fi]) / bin_widths[fi])
        if idx < 0:
            return 0
        if idx >= num_bins:
            return num_bins - 1
        return idx

    # helper: compute filter response at response-location (rh, rw)
    # with zero-padding when patch goes out of bounds
    def compute_response_at(fi, rh, rw):
        filt = filter_list[fi]
        Kh, Kw = filt.shape
        # center offset: assume filter is centered at floor(Kh/2), floor(Kw/2)
        ctr_h = Kh // 2
        ctr_w = Kw // 2
        resp = 0.0
        # iterate filter
        for u in range(Kh):
            for v in range(Kw):
                ih = rh + (u - ctr_h)
                iw = rw + (v - ctr_w)
                if 0 <= ih < H and 0 <= iw < W:
                    resp += filt[u, v] * img_syn[ih, iw]
                else:
                    # zero padding: contributes 0
                    pass
        return resp

    # For each candidate, compute energy if we set pixel -> candidate
    energies = np.zeros(len(candidates), dtype=float)

    # For candidate evaluation we will create temporary hist copies only for filters
    # that are affected; we will update those copies, compute the weighted diff,
    # and compute energy = sum(abs(h_tmp - h_ori) @ weight)
    for ci, cand in enumerate(candidates):
        delta = float(cand - old_val)
        # if no change at pixel, energy is simply current energy
        if delta == 0:
            # compute current energy quickly
            # compute weighted difference per filter and sum
            total_e = 0.0
            for fi in range(num_filters):
                total_e += np.abs(hists_syn[fi] - hists_ori[fi]) @ weight
            energies[ci] = total_e
            continue

        # copy histograms for affected filters (we will modify these copies)
        # For simplicity, copy all filters (num_filters typically small). If you want to micro-opt,
        # copy only those filters where at least one filter weight overlapping the pixel is non-zero.
        h_tmp = hists_syn.copy()

        # For each filter, iterate over filter element positions (u,v) and find the
        # response location (rh,rw) that includes the pixel at pos_h,pos_w.
        for fi in range(num_filters):
            filt = filter_list[fi]
            Kh, Kw = filt.shape
            ctr_h = Kh // 2
            ctr_w = Kw // 2
            # For each (u,v) in filter kernel, the response location that uses pixel
            # pos = (pos_h,pos_w) is (rh = pos_h - (u-ctr_h), rw = pos_w - (v-ctr_w))
            for u in range(Kh):
                for v in range(Kw):
                    wuv = filt[u, v]
                    if wuv == 0:
                        continue
                    rh = pos_h - (u - ctr_h)
                    rw = pos_w - (v - ctr_w)
                    # ensure response location is within image (valid response positions)
                    if not (0 <= rh < H and 0 <= rw < W):
                        continue
                    # compute old response at (rh,rw)
                    resp_old = compute_response_at(fi, rh, rw)
                    resp_new = resp_old + wuv * delta
                    bin_old = resp_to_bin(fi, resp_old)
                    bin_new = resp_to_bin(fi, resp_new)
                    if bin_old != bin_new:
                        # update temporary histogram counts
                        h_tmp[fi, bin_old] -= 1.0
                        h_tmp[fi, bin_new] += 1.0
                        # note: histogram counts are floats here; ensure consistency elsewhere

        # compute energy from temporary histograms
        total_e = 0.0
        for fi in range(num_filters):
            total_e += np.abs(h_tmp[fi] - hists_ori[fi]) @ weight
        energies[ci] = total_e

    # now convert energies to probabilities
    probs = np.exp(-energies / T)
    eps = 1e-12
    probs = probs + eps
    probs = probs / probs.sum()

    # sample new intensity from candidates according to probs
    new_val = np.random.choice(candidates, p=probs)

    # if new_val equals old_val, nothing changes; otherwise apply updates to img_syn and hists_syn
    if new_val != old_val:
        delta = float(new_val - old_val)
        # update image
        img_syn[pos_h, pos_w] = new_val
        # update histograms in place using same logic as in candidate loop
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
                    resp_old = compute_response_at(fi, rh, rw) - wuv * delta  # because we already changed img_syn; compute_response_at reads new img_syn
                    resp_new = resp_old + wuv * delta
                    bin_old = resp_to_bin(fi, resp_old)
                    bin_new = resp_to_bin(fi, resp_new)
                    if bin_old != bin_new:
                        hists_syn[fi, bin_old] -= 1.0
                        hists_syn[fi, bin_new] += 1.0

    return img_syn, hists_syn