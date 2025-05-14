import numpy as np
from dataclasses import dataclass
from scipy.stats import beta

@dataclass(frozen=True)
class BetaParams:
    a: float
    b: float

    def __post_init__(self):
        if self.a < 0.:
            raise ValueError(f"parameter `a` of Beta distribution need to be > 0")
        if self.b < 0.:
            raise ValueError(f"parameter `b` of Beta distribution need to be > 0")

class Beta():
    params: BetaParams

    def __init__(self, samples=None, params=None):
        self.params = params if params is not None else self.fit(samples)

    def fit(self, sampels: np.ndarray[float]) -> BetaParams:
        mu = np.mean(sampels)
        var = np.var(sampels)

        total_evidence = ((mu * (1. - mu)) / var) - 1.  # a + b

        a = float(mu * total_evidence)
        b = float((1. - mu) * total_evidence)

        return BetaParams(a=a, b=b)

    def stats(self):
        return beta.stats(self.params.a, self.params.b, moments='mvsk')
