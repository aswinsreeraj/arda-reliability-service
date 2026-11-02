"""Domain models, ports, and implementations for confidence calibration.

Confidence calibration transforms raw reliability scores into calibrated probability
estimates representing empirical decision correctness:
    P(decision is correct | raw_reliability_score = s)

Crucially, raw scores and calibrated confidence values are distinct:
- raw_score: Decomposed heuristic agreement * density * coverage in [0.0, 1.0].
- calibrated_confidence: Empirical probability estimate in [0.0, 1.0].
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from arda_reliability_service.domain.exceptions import DomainValidationError


class CalibrationError(Exception):
    """Base exception for calibration failures."""


class CalibrationUnavailableError(CalibrationError):
    """Raised when calibration is requested but no fitted calibration model is available."""


class UnfittedCalibratorError(CalibrationError):
    """Raised when attempting to calibrate with an unfitted model."""


class InvalidCalibrationInputError(CalibrationError):
    """Raised when raw score input is outside the valid [0.0, 1.0] domain."""


@dataclass(frozen=True)
class CalibratedConfidence:
    """Value object encapsulating raw score and its calibrated confidence estimate.

    Guarantees explicit separation between raw scores and calibrated probabilities.
    """

    raw_score: float
    confidence: float | None
    is_calibrated: bool
    calibrator_version: str | None = None
    calibrator_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.raw_score <= 1.0):
            raise DomainValidationError(
                f"raw_score must be in [0.0, 1.0], got {self.raw_score}"
            )
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise DomainValidationError(
                f"confidence must be in [0.0, 1.0] or None, got {self.confidence}"
            )
        if self.is_calibrated and self.confidence is None:
            raise DomainValidationError(
                "is_calibrated cannot be True when confidence is None."
            )


class CalibratorPort(ABC):
    """Port defining the calibration model contract."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Version string identifying the calibrator artifact and training lineage."""

    @property
    @abstractmethod
    def calibrator_type(self) -> str:
        """Type of calibrator (e.g. 'platt', 'isotonic', 'unavailable')."""

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether this calibrator has been fitted with empirical evidence."""

    @property
    @abstractmethod
    def metrics(self) -> dict[str, Any]:
        """Evaluation metrics recorded during calibrator fitting (e.g. ECE, Brier score)."""

    @abstractmethod
    def calibrate(self, raw_score: float) -> float:
        """Map a single raw reliability score in [0.0, 1.0] to calibrated confidence in [0.0, 1.0].

        Args:
            raw_score: Raw score in [0.0, 1.0].

        Returns:
            Calibrated probability estimate in [0.0, 1.0].

        Raises:
            InvalidCalibrationInputError: If raw_score is out of bounds or NaN.
            UnfittedCalibratorError: If model is not fitted.
        """

    def calibrate_batch(self, raw_scores: Sequence[float]) -> list[float]:
        """Calibrate a sequence of raw reliability scores."""
        return [self.calibrate(s) for s in raw_scores]

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize calibrator state to dictionary."""

    @abstractmethod
    def to_json(self) -> str:
        """Serialize calibrator state to JSON string."""


class UnavailableCalibrator(CalibratorPort):
    """Explicit Null Object representing an unavailable, disabled, or uncalibrated state.

    Ensures the platform never silently treats uncalibrated raw scores as calibrated confidence.
    """

    def __init__(self, reason: str = "No calibrated model configured") -> None:
        self._reason = reason

    @property
    def version(self) -> str:
        return "none"

    @property
    def calibrator_type(self) -> str:
        return "unavailable"

    @property
    def is_fitted(self) -> bool:
        return False

    @property
    def metrics(self) -> dict[str, Any]:
        return {"status": "unavailable", "reason": self._reason}

    def calibrate(self, raw_score: float) -> float:
        raise CalibrationUnavailableError(
            f"Confidence calibration is unavailable: {self._reason}. "
            "Raw reliability score cannot be used as calibrated confidence."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibrator_type": self.calibrator_type,
            "version": self.version,
            "reason": self._reason,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class PlattCalibrator(CalibratorPort):
    """Parametric Platt Scaling calibrator using a logistic sigmoid.

    Formula:
        P(correct=1 | s) = 1 / (1 + exp(-(a * s + b)))
    where s is the raw reliability score.
    """

    def __init__(
        self,
        a: float,
        b: float,
        version: str = "v1.0.0",
        metrics: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> None:
        self.a = float(a)
        self.b = float(b)
        self._version = version
        self._metrics = metrics or {}
        self._created_at = created_at or datetime.now(UTC).isoformat()

    @property
    def version(self) -> str:
        return self._version

    @property
    def calibrator_type(self) -> str:
        return "platt"

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def metrics(self) -> dict[str, Any]:
        return self._metrics

    @property
    def created_at(self) -> str:
        return self._created_at

    def calibrate(self, raw_score: float) -> float:
        if math.isnan(raw_score) or not (0.0 <= raw_score <= 1.0):
            raise InvalidCalibrationInputError(
                f"raw_score must be a float in [0.0, 1.0], got {raw_score}"
            )

        logit = self.a * raw_score + self.b
        # Clamp logit to prevent math overflow in exp
        logit_clamped = max(-50.0, min(50.0, logit))
        prob = 1.0 / (1.0 + math.exp(-logit_clamped))
        return min(1.0, max(0.0, round(prob, 4)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibrator_type": self.calibrator_type,
            "version": self.version,
            "a": self.a,
            "b": self.b,
            "metrics": self.metrics,
            "created_at": self.created_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlattCalibrator:
        if data.get("calibrator_type") != "platt":
            raise DomainValidationError(
                f"Expected calibrator_type 'platt', got {data.get('calibrator_type')}"
            )
        return cls(
            a=data["a"],
            b=data["b"],
            version=data.get("version", "v1.0.0"),
            metrics=data.get("metrics", {}),
            created_at=data.get("created_at"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> PlattCalibrator:
        return cls.from_dict(json.loads(json_str))


class IsotonicCalibrator(CalibratorPort):
    """Non-parametric Isotonic Regression calibrator.

    Fits a non-decreasing piecewise-constant or piecewise-linear mapping from knots:
        (x_0, y_0), (x_1, y_1), ..., (x_k, y_k)
    where x_i in [0.0, 1.0] are sorted raw score thresholds and y_i in [0.0, 1.0] are empirical probabilities.
    """

    def __init__(
        self,
        x_knots: Sequence[float],
        y_knots: Sequence[float],
        version: str = "v1.0.0",
        metrics: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> None:
        if len(x_knots) != len(y_knots):
            raise DomainValidationError(
                f"x_knots (len {len(x_knots)}) and y_knots (len {len(y_knots)}) must have identical lengths"
            )
        if len(x_knots) == 0:
            raise DomainValidationError("Isotonic knots must not be empty.")

        self.x_knots = [float(x) for x in x_knots]
        self.y_knots = [float(y) for y in y_knots]
        self._version = version
        self._metrics = metrics or {}
        self._created_at = created_at or datetime.now(UTC).isoformat()

        # Validate sorting
        for i in range(len(self.x_knots) - 1):
            if self.x_knots[i] > self.x_knots[i + 1]:
                raise DomainValidationError("x_knots must be monotonically sorted.")

    @property
    def version(self) -> str:
        return self._version

    @property
    def calibrator_type(self) -> str:
        return "isotonic"

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def metrics(self) -> dict[str, Any]:
        return self._metrics

    @property
    def created_at(self) -> str:
        return self._created_at

    def calibrate(self, raw_score: float) -> float:
        if math.isnan(raw_score) or not (0.0 <= raw_score <= 1.0):
            raise InvalidCalibrationInputError(
                f"raw_score must be a float in [0.0, 1.0], got {raw_score}"
            )

        # Clip if out of bounds of knots
        if raw_score <= self.x_knots[0]:
            return min(1.0, max(0.0, round(self.y_knots[0], 4)))
        if raw_score >= self.x_knots[-1]:
            return min(1.0, max(0.0, round(self.y_knots[-1], 4)))

        # Linear interpolation between adjacent knots
        for i in range(len(self.x_knots) - 1):
            x0, x1 = self.x_knots[i], self.x_knots[i + 1]
            if x0 <= raw_score <= x1:
                y0, y1 = self.y_knots[i], self.y_knots[i + 1]
                if abs(x1 - x0) < 1e-9:
                    prob = y0
                else:
                    t = (raw_score - x0) / (x1 - x0)
                    prob = y0 + t * (y1 - y0)
                return min(1.0, max(0.0, round(prob, 4)))

        return min(1.0, max(0.0, round(self.y_knots[-1], 4)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibrator_type": self.calibrator_type,
            "version": self.version,
            "x_knots": self.x_knots,
            "y_knots": self.y_knots,
            "metrics": self.metrics,
            "created_at": self.created_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IsotonicCalibrator:
        if data.get("calibrator_type") != "isotonic":
            raise DomainValidationError(
                f"Expected calibrator_type 'isotonic', got {data.get('calibrator_type')}"
            )
        return cls(
            x_knots=data["x_knots"],
            y_knots=data["y_knots"],
            version=data.get("version", "v1.0.0"),
            metrics=data.get("metrics", {}),
            created_at=data.get("created_at"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> IsotonicCalibrator:
        return cls.from_dict(json.loads(json_str))
