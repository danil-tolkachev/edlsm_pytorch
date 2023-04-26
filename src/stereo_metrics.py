from collections import Counter, OrderedDict
import numpy as np

INVALID_DISP = 0


class StereoMetrics:
    def __init__(self, gt=None, disp=None, frame=None):
        self.counter = Counter()
        self.ngt = 0
        self.nboth = 0
        if gt is not None and disp is not None:
            self.update(gt, disp, frame)

    def update(self, gt, disp, frame):
        mask_gt = gt != INVALID_DISP
        mask_disp = disp != INVALID_DISP
        mask_both = mask_gt & mask_disp

        self.frame = frame
        self.ngt = np.count_nonzero(mask_gt)
        self.nboth = np.count_nonzero(mask_both)

        err = np.abs(gt[mask_both] - disp[mask_both])
        self.counter = Counter(err)

    def __iadd__(self, other):
        self.ngt += other.ngt
        self.nboth += other.nboth
        self.counter += other.counter
        return self

    def metrics(self):
        res = OrderedDict()
        res['Frame'] = self.frame
        res['NGT'] = self.ngt
        res['Density'] = self.nboth / self.ngt

        err = np.array(list(self.counter.elements()))
        res['Std'] = np.sqrt((err**2).mean())
        res['Avg'] = np.mean(err)

        for thr in [1, 2, 3, 4, 5]:
            key = 'Err%dpx' % thr
            res[key] = 100 * np.count_nonzero(err > thr) / self.nboth

        for perc in [50, 90, 95, 99]:
            res['Perc%d' % perc] = np.quantile(err, perc / 100)

        return res
