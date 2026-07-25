import torch
import csv
from itertools import combinations


class GradientCosineTracker:
    """
    Computes per-component gradients via autograd.grad, flattens them, logs
    pairwise cosine similarity to CSV.  Called through IntervalPenalization
    as an optional side-effect; zero overhead when not set.
    """

    loss_names = ["ce", "var", "output_reg", "interval_drift", "align"]

    def __init__(self, optimizer, csv_path):
        self.csv_path = csv_path
        self._step = 0
        self._fieldnames = None

        # collect all trainable parameters from every param_group
        self._params: list[torch.Tensor] = []
        for pg in optimizer.param_groups:
            self._params.extend(pg["params"])

    def record(self, loss_dict):
        """loss_dict : {name: scalar_tensor}.  Must be called before backward."""
        params = [p for p in self._params if p.requires_grad]
        if not params:
            return

        grad_buffers = {}
        for name in self.loss_names:
            t = loss_dict.get(name)
            if t is None or not t.requires_grad:
                grad_buffers[name] = None
                continue
            grads = torch.autograd.grad(
                t, params, only_inputs=True, retain_graph=True, allow_unused=True
            )
            grad_buffers[name] = [g.detach().cpu() if g is not None else None for g in grads]

        filled_names = [n for n, v in grad_buffers.items() if v is not None and any(g is not None for g in v)]
        if len(filled_names) < 2:
            return

        names = sorted(filled_names)
        pairs = list(combinations(names, 2))
        row: list[float] = [float(self._step)]
        for a, b in pairs:
            ga_list, gb_list = grad_buffers[a], grad_buffers[b]
            parts_a, parts_b = [], []
            for ga, gb in zip(ga_list, gb_list):
                if ga is not None and gb is not None:
                    parts_a.append(ga.flatten())
                    parts_b.append(gb.flatten())
            if not parts_a:
                row.append(float("nan"))
            else:
                ca = torch.cat(parts_a, dim=0)
                cb = torch.cat(parts_b, dim=0)
                denom = torch.norm(ca) * torch.norm(cb)
                cos = float(torch.dot(ca, cb) / (denom + 1e-12)) if denom > 0 else float("nan")
                row.append(cos)

        header_ok = self._fieldnames is None
        if header_ok:
            self._fieldnames = ["step"] + [f"{a}__{b}" for a, b in pairs]

        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if header_ok:
                w.writerow(self._fieldnames)
            w.writerow(row)

        self._step += 1
