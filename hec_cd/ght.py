"""GHT Eq. (13) with an Otsu-pooled empirical variance prior.

Core mathematics: Barron, ECCV 2020, arXiv:2007.07350.
Reference code: jonbarron/hist_thresh, experiments.ipynb, commit
0861e3d2e96f65fbbd92c06cd242e9d1e980b8ae. Adapted from the Apache-2.0
reference; modified to add the empirical prior, input guards, and inclusive
uint8 cutoff. The Otsu-pooled prior and fixed rho=10 are the HEC-CD adaptation.

Copyright 2020 Google LLC (reference implementation).
Licensed under the Apache License, Version 2.0 (the "License"); you may not
use the reference code except in compliance with the License. You may obtain
a copy at https://www.apache.org/licenses/LICENSE-2.0 . Unless required by
applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the specific language
governing permissions and limitations under the License.
"""

import math

import numpy as np
from skimage.filters import threshold_otsu


def ght(n, x=None, nu=0.0, tau=0.0, kappa=0.0, omega=0.5):
    """Return the GHT split and objective, excluding empty candidate classes."""
    n = np.asarray(n, dtype=np.float64)
    x = np.arange(len(n), dtype=float) if x is None else np.asarray(x, dtype=float)
    if (n.ndim != 1 or len(n) < 2 or x.shape != n.shape or not np.isfinite(n).all()
            or not np.isfinite(x).all() or (n < 0).any() or n.sum() <= 0
            or (np.diff(x) < 0).any()):
        raise ValueError("Expected sorted finite locations and nonnegative nonempty weights")
    if not np.isfinite([nu, tau, kappa, omega]).all() or min(nu, tau, kappa) < 0 or not 0 <= omega <= 1:
        raise ValueError("Invalid GHT prior parameters")
    csum = lambda a: np.cumsum(a)[:-1]
    dsum = lambda a: np.cumsum(a[::-1])[-2::-1]
    w0, w1 = csum(n), dsum(n)
    valid = (w0 > 0) & (w1 > 0)
    if not valid.any():
        raise ValueError("GHT requires two nonempty candidate classes")
    w0, w1 = np.maximum(w0, 1e-30), np.maximum(w1, 1e-30)
    p0, p1 = w0 / (w0 + w1), w1 / (w0 + w1)
    mu0, mu1 = csum(n * x) / w0, dsum(n * x) / w1
    d0 = np.maximum(0.0, csum(n * x * x) - w0 * mu0 * mu0)
    d1 = np.maximum(0.0, dsum(n * x * x) - w1 * mu1 * mu1)
    v0 = np.maximum(1e-30, (p0 * nu * tau * tau + d0) / (p0 * nu + w0))
    v1 = np.maximum(1e-30, (p1 * nu * tau * tau + d1) / (p1 * nu + w1))
    f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa * omega) * np.log(w0)
    f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
    objective = f0 + f1
    objective[~valid] = -np.inf
    best = np.max(objective)
    return float(np.mean(x[:-1][objective == best])), objective


def nonzero_histogram(histogram):
    h = np.asarray(histogram, dtype=np.float64).copy()
    if h.shape != (256,) or not np.isfinite(h).all() or (h < 0).any():
        raise ValueError("Expected finite nonnegative 256-bin histogram")
    h[0] = 0
    return h


def empirical_parameters(histogram, rho):
    if not np.isfinite(rho) or rho < 0:
        raise ValueError("Prior strength must be finite and nonnegative")
    h = nonzero_histogram(histogram)
    occupied = np.flatnonzero(h)
    n = float(h.sum())
    if len(occupied) < 2:
        split = int(occupied[0]) if len(occupied) else 256
        pooled = 0.0
    else:
        split = int(threshold_otsu(hist=(h[1:], np.arange(1, 256))))
        pooled = 0.0
        for idx in [slice(1, split + 1), slice(split + 1, 256)]:
            weights = h[idx]
            values = np.arange(256)[idx]
            mean = np.average(values, weights=weights)
            pooled += float(np.dot(weights, (values - mean) ** 2)) / n
    return dict(rho=float(rho), nonzero_count=n, nu=float(rho * n),
                tau_squared_u8=max(pooled, 1 / 12), raw_pooled_variance_u8=pooled,
                kappa=0.0, omega=0.5, otsu_split_u8=split,
                shrinkage_weight=float(rho / (1 + rho)))


def inclusive_cutoff(split):
    if not np.isfinite(split) or not 0 <= split < 255:
        raise ValueError("Expected a finite split below the maximum uint8 bin")
    return int(math.floor(split)) + 1


def regularized_spec(histogram, rho):
    h = nonzero_histogram(histogram)
    params = empirical_parameters(h, rho)
    occupied = np.flatnonzero(h)
    if len(occupied) < 2:
        cut = int(occupied[0]) if len(occupied) else 256
        split = None
    else:
        split, _ = ght(h[1:] / h.sum(), np.arange(1, 256), nu=rho,
                       tau=math.sqrt(params["tau_squared_u8"]), kappa=0.0, omega=0.5)
        cut = inclusive_cutoff(split)
    return dict(kind="single", low_u8=cut, high_u8=cut, comparison=">=",
                ght_split_u8=split, ght_parameters=params)
