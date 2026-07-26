import torch
import csv


class GradientCosineTracker:
    """Logs cosine similarity between interval_drift and align loss gradients."""

    def __init__(self, optimizer, csv_path):
        self.csv_path = csv_path
        self._step = 0
        self._fieldnames = None

        self._params: list[torch.Tensor] = []
        for pg in optimizer.param_groups:
            self._params.extend(pg["params"])

    def record(self, loss_dict):
        params = [p for p in self._params if p.requires_grad]
        if not params:
            return

        drift_t = loss_dict.get("interval_drift")
        align_t = loss_dict.get("align")

        if drift_t is None or not drift_t.requires_grad:
            drift_grads = None
        else:
            drift_grads = torch.autograd.grad(
                drift_t, params, only_inputs=True, retain_graph=True, allow_unused=True
            )

        if align_t is None or not align_t.requires_grad:
            align_grads = None
        else:
            align_grads = torch.autograd.grad(
                align_t, params, only_inputs=True, retain_graph=True, allow_unused=True
            )

        parts_d, parts_a = [], []
        if drift_grads is not None and align_grads is not None:
            for g1, g2 in zip(drift_grads, align_grads):
                if g1 is not None and g2 is not None:
                    parts_d.append(g1.flatten().cpu())
                    parts_a.append(g2.flatten().cpu())

        if not parts_d:
            cos = float("nan")
        else:
            v_d, v_a = torch.cat(parts_d), torch.cat(parts_a)
            denom = torch.norm(v_d) * torch.norm(v_a)
            cos = float(torch.dot(v_d, v_a) / (denom + 1e-12)) if denom > 0 else float("nan")

        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if self._fieldnames is None:
                w.writerow(["step", "interval_drift__align"])
            w.writerow([float(self._step), cos])

        self._fieldnames = ["step", "interval_drift__align"]
        self._step += 1
