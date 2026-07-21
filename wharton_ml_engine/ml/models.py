"""Lightweight, self-contained ML models (numpy only).

Kept deliberately simple and *interpretable* — linear models expose per-signal
coefficients the team can explain in the IPS ("the model learned to weight
quality at +0.31, momentum at +0.18 ...").  No scikit-learn dependency, so the
whole engine stays reproducible and portable.

Models:
* ``StandardScaler``   — feature standardisation (fit on train only).
* ``RidgeRegressor``   — L2-regularised linear regression (closed form) for
  expected forward return.
* ``LogisticRegressor``— L2-regularised logistic regression (gradient descent)
  for probability of out-performing the cross-section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class StandardScaler:
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, float) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict[str, list]:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, list]) -> "StandardScaler":
        return cls(mean_=np.asarray(d["mean"], float), std_=np.asarray(d["std"], float))


@dataclass
class RidgeRegressor:
    """Ridge regression via the closed-form normal equations.

    Solves ``w = (XᵀX + αI)⁻¹ Xᵀy`` with an unpenalised intercept.
    """

    alpha: float = 1.0
    coef_: Optional[np.ndarray] = None
    intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n, d = X.shape
        Xb = np.hstack([np.ones((n, 1)), X])            # bias column
        reg = self.alpha * np.eye(d + 1)
        reg[0, 0] = 0.0                                  # do not penalise intercept
        w = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
        self.intercept_ = float(w[0])
        self.coef_ = w[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, float) @ self.coef_ + self.intercept_

    def to_dict(self) -> Dict[str, object]:
        return {"type": "ridge", "alpha": self.alpha,
                "coef": self.coef_.tolist(), "intercept": self.intercept_}

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "RidgeRegressor":
        m = cls(alpha=float(d["alpha"]))
        m.coef_ = np.asarray(d["coef"], float)
        m.intercept_ = float(d["intercept"])
        return m


@dataclass
class LogisticRegressor:
    """L2-regularised logistic regression trained by batch gradient descent."""

    l2: float = 1.0
    lr: float = 0.1
    epochs: int = 400
    coef_: Optional[np.ndarray] = None
    intercept_: float = 0.0
    loss_history_: List[float] = field(default_factory=list)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressor":
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        self.loss_history_ = []
        for _ in range(self.epochs):
            p = self._sigmoid(X @ w + b)
            err = p - y
            grad_w = X.T @ err / n + (self.l2 / n) * w
            grad_b = float(err.mean())
            w -= self.lr * grad_w
            b -= self.lr * grad_b
            eps = 1e-12
            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            self.loss_history_.append(float(loss))
        self.coef_ = w
        self.intercept_ = float(b)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(np.asarray(X, float) @ self.coef_ + self.intercept_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def to_dict(self) -> Dict[str, object]:
        return {"type": "logistic", "l2": self.l2, "lr": self.lr,
                "epochs": self.epochs, "coef": self.coef_.tolist(),
                "intercept": self.intercept_}

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "LogisticRegressor":
        m = cls(l2=float(d["l2"]), lr=float(d["lr"]), epochs=int(d["epochs"]))
        m.coef_ = np.asarray(d["coef"], float)
        m.intercept_ = float(d["intercept"])
        return m


def model_from_dict(d: Dict[str, object]):
    return {"ridge": RidgeRegressor, "logistic": LogisticRegressor}[d["type"]].from_dict(d)
