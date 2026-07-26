import torch
import csv


class GradientCosineTracker:
    """Logs cosine similarity between interval_drift and align loss gradients.

    Uses per-component backward() calls instead of autograd.grad() to avoid
    tracing failures through stored intermediate activations.
    """

    def __init__(self, optimizer, csv_path):
        self.csv_path = csv_path
        self._step = 0
        self._header_written = False

        self._params: list[torch.Tensor] = []
        for pg in optimizer.param_groups:
            self._params.extend(pg["params"])

    def record(self, loss_dict):
        params = [p for p in self._params if p.requires_grad]
        if not params:
            return

        drift_t = loss_dict.get("interval_drift")
        align_t = loss_dict.get("align")

        drift_ok = drift_t is not None and drift_t.requires_grad
        align_ok = align_t is not None and align_t.requires_grad

        cos = float("nan")

        if drift_ok and align_ok:
            self._zero_grads(params)
            drift_t.backward(retain_graph=True)
            drift_grads = {p: p.grad.detach().cpu().clone() for p in params if p.grad is not None}

            self._zero_grads(params)
            align_t.backward(retain_graph=True)
            align_grads = {p: p.grad.detach().cpu().clone() for p in params if p.grad is not None}

            self._zero_grads(params)

            shared = [p for p in params if p in drift_grads and p in align_grads]
            if shared:
                v_d = torch.cat([drift_grads[p].flatten() for p in shared])
                v_a = torch.cat([align_grads[p].flatten() for p in shared])
                denom = torch.norm(v_d) * torch.norm(v_a)
                cos = float(torch.dot(v_d, v_a) / (denom + 1e-12)) if denom > 0 else float("nan")

        else:
            self._zero_grads(params)

        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if not self._header_written:
                w.writerow(["step", "interval_drift__align"])
                self._header_written = True
            w.writerow([float(self._step), cos])

        self._step += 1

    @staticmethod
    def _zero_grads(params):
        for p in params:
            p.grad = None