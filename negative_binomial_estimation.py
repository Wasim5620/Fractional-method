import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln


def nb_negative_loglikelihood(observed, predicted_mean, dispersion):
    observed = np.asarray(observed, dtype=float)
    predicted_mean = np.asarray(predicted_mean, dtype=float)
    dispersion = float(dispersion)

    if dispersion <= 0:
        return np.inf

    predicted_mean = np.clip(predicted_mean, 1e-12, None)
    k = 1.0 / dispersion

    logpmf = (
        gammaln(observed + k)
        - gammaln(k)
        - gammaln(observed + 1.0)
        + k * np.log(k / (k + predicted_mean))
        + observed * np.log(predicted_mean / (k + predicted_mean))
    )
    return -np.sum(logpmf)


def fit_nb_dispersion(observed, predicted_mean, initial_dispersion=0.2, bounds=(1e-8, 10.0)):
    observed = np.asarray(observed, dtype=float)
    predicted_mean = np.asarray(predicted_mean, dtype=float)

    def objective(x):
        return nb_negative_loglikelihood(observed, predicted_mean, x[0])

    result = minimize(
        objective,
        x0=np.array([initial_dispersion], dtype=float),
        bounds=[bounds],
        method="L-BFGS-B",
    )

    return {
        "dispersion": float(result.x[0]),
        "nll": float(result.fun),
        "success": bool(result.success),
        "message": result.message,
    }


def compute_aic(nll, n_parameters):
    return 2.0 * float(n_parameters) + 2.0 * float(nll)
