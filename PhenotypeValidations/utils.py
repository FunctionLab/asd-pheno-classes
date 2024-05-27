import numpy as np


def sabic(model, X, Y=None):
    """Sample-Sized Adjusted BIC.

    References
    ----------
    Sclove SL. Application of model-selection criteria to some problems in multivariate analysis. Psychometrika. 1987;52(3):333–343.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    Y : array-like of shape (n_samples, n_features_structural), default=None

    Returns
    -------
    ssa_bic : float
    """
    n = X.shape[0]

    return -2 * model.score(X, Y) * n + model.n_parameters * np.log(
        n * ((n + 2) / 24)
    )


def c_aic(model, X, Y=None):
    """Consistent AIC.

    References
    ----------
    Bozdogan, H. 1987. Model selection and Akaike’s information criterion (AIC):
    The general theory and its analytical extensions. Psychometrika 52: 345–370.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    Y : array-like of shape (n_samples, n_features_structural), default=None

    Returns
    -------
    caic : float
        The lower the better.
    """
    n = X.shape[0]
    return -2 * model.score(X, Y) * n + model.n_parameters * (np.log(n) + 1)


def awe(model, X, Y=None):
        """Approximate weight of evidence. (Banfield & Raftery (1993))

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_features_structural), default=None

        Returns
        -------
        awe : float
        """
        n = X.shape[0]
        return -2 * model.score(X, Y) * n + model.n_parameters * (np.log(n) + 1.5)


def scramble_column(column):
    return np.random.permutation(column)
