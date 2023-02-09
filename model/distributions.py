import torch


class BinaryConcrete(torch.distributions.relaxed_bernoulli.RelaxedBernoulli):
    def __init__(self, temperature, logits):
        super().__init__(temperature=temperature, logits=logits)
        self.device = self.temperature.device

    def cdf(self, value):
        return torch.sigmoid(
            (torch.log(value) - torch.log(1.0 - value)) * self.temperature - self.logits
        )

    def log_prob(self, value):
        return torch.where(
            (value > 0) & (value < 1),
            super().log_prob(value),
            torch.full_like(value, -float("inf")),
        )

    def log_expected_L0(self, value):
        return -torch.nn.functional.softplus(
            (torch.log(value) - torch.log(1 - value)) * self.temperature - self.logits
        )


class Streched(torch.distributions.TransformedDistribution):
    def __init__(self, base_dist, l=-0.1, r=1.1):
        super().__init__(
            base_dist, torch.distributions.AffineTransform(loc=l, scale=r - l)
        )

    def log_expected_L0(self):
        value = torch.tensor(0.0, device=self.base_dist.device)
        for transform in self.transforms[::-1]:
            value = transform.inv(value)
        if self._validate_args:
            self.base_dist._validate_sample(value)
        value = self.base_dist.log_expected_L0(value)
        value = self._monotonize_cdf(value)
        return value

    def expected_L0(self):
        return self.log_expected_L0().exp()


class RectifiedStreched(Streched):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, sample_shape=torch.Size([])):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size([])):
        x = super().rsample(sample_shape)
        return x.clamp(0, 1)


def confusion_matrix(y_pred, y_true):
    device = y_pred.device
    labels = max(y_pred.max().item() + 1, y_true.max().item() + 1)

    return (
        (
            torch.stack((y_true, y_pred), -1).unsqueeze(-2).unsqueeze(-2)
            == torch.stack(
                (
                    torch.arange(labels, device=device).unsqueeze(-1).repeat(1, labels),
                    torch.arange(labels, device=device).unsqueeze(-2).repeat(labels, 1),
                ),
                -1,
            )
        )
        .all(-1)
        .sum(-3)
    )

def accuracy_precision_recall_f1(y_pred, y_true, average=True):
    M = confusion_matrix(y_pred, y_true)

    tp = M.diagonal(dim1=-2, dim2=-1).float()

    precision_den = M.sum(-2)
    precision = torch.where(
        precision_den == 0, torch.zeros_like(tp), tp / precision_den
    )

    recall_den = M.sum(-1)
    recall = torch.where(recall_den == 0, torch.ones_like(tp), tp / recall_den)

    f1_den = precision + recall
    f1 = torch.where(
        f1_den == 0, torch.zeros_like(tp), 2 * (precision * recall) / f1_den
    )

    return ((y_pred == y_true).float().mean(-1),) + (
        tuple(e.mean(-1) for e in (precision, recall, f1))
        if average
        else (precision, recall, f1)
    )
