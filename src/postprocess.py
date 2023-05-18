def lr_check(ldisp, rdisp, lr_thresh=1.0):
    h, w = ldisp.shape
    for v in range(h):
        for u in range(w):
            d = ldisp[v, u]
            ur = round(u - d)
            if ur < 0 or abs(rdisp[v, ur] - d) > lr_thresh:
                ldisp[v, u] = -1
