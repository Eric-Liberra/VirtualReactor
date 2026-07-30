import numpy as np

from virtualreactor.units import (
    to_magnitude,
    ureg,
)


class PropertyModel:
    """Calculate thermophysical properties from the current state."""

    def __init__(
        self,
        evaluation_function,
        composition_evaluation_function=None,
        model_name=None,
    ):
        if not callable(evaluation_function):
            raise TypeError(
                "evaluation_function must be callable."
            )

        if (
            composition_evaluation_function is not None
            and not callable(
                composition_evaluation_function
            )
        ):
            raise TypeError(
                "composition_evaluation_function "
                "must be callable or None."
            )

        self._evaluation_function = (
            evaluation_function
        )

        self._composition_evaluation_function = (
            composition_evaluation_function
        )

        self.model_name = model_name

    @classmethod
    def constant(
        cls,
        density,
        heat_capacity,
    ):
        """Create a model with constant thermophysical properties."""

        density = to_magnitude(
            density,
            ureg.kilogram / ureg.meter**3,
            "density",
        )

        heat_capacity = to_magnitude(
            heat_capacity,
            ureg.joule
            / (
                ureg.kilogram
                * ureg.kelvin
            ),
            "heat_capacity",
        )

        if density <= 0.0:
            raise ValueError(
                "density must be positive."
            )

        if heat_capacity <= 0.0:
            raise ValueError(
                "heat_capacity must be positive."
            )

        def evaluate_constant(
            composition,
            temperature,
            pressure,
        ):
            return density, heat_capacity

        return cls(
            evaluation_function=evaluate_constant,
            composition_evaluation_function=(
                evaluate_constant
            ),
            model_name="constant",
        )

    def evaluate(
        self,
        concentrations,
        temperature,
        pressure,
    ):
        """Evaluate properties from species concentrations."""

        concentrations = np.asarray(
            concentrations,
            dtype=float,
        )

        if concentrations.ndim != 1:
            raise ValueError(
                "concentrations must be one-dimensional."
            )

        density, heat_capacity = (
            self._evaluation_function(
                concentrations,
                temperature,
                pressure,
            )
        )

        return self._validate_properties(
            density,
            heat_capacity,
        )

    def evaluate_from_composition(
        self,
        mole_fractions,
        temperature,
        pressure,
    ):
        """Evaluate properties from mole fractions."""

        mole_fractions = np.asarray(
            mole_fractions,
            dtype=float,
        )

        if mole_fractions.ndim != 1:
            raise ValueError(
                "mole_fractions must be one-dimensional."
            )

        total = np.sum(mole_fractions)

        if not np.isfinite(total):
            raise ValueError(
                "Mole fractions must be finite."
            )

        if total <= 0.0:
            raise ValueError(
                "Mole fractions must have a positive sum."
            )

        mole_fractions = (
            mole_fractions / total
        )

        if (
            self._composition_evaluation_function
            is None
        ):
            raise NotImplementedError(
                f"Property model "
                f"{self.model_name!r} does not support "
                "evaluation from composition."
            )

        density, heat_capacity = (
            self._composition_evaluation_function(
                mole_fractions,
                temperature,
                pressure,
            )
        )

        return self._validate_properties(
            density,
            heat_capacity,
        )

    @staticmethod
    def _validate_properties(
        density,
        heat_capacity,
    ):
        """Validate calculated SI property values."""

        density = float(density)
        heat_capacity = float(
            heat_capacity
        )

        if not np.isfinite(density):
            raise ValueError(
                "Calculated density must be finite."
            )

        if not np.isfinite(
            heat_capacity
        ):
            raise ValueError(
                "Calculated heat capacity must be finite."
            )

        if density <= 0.0:
            raise ValueError(
                "Calculated density must be positive."
            )

        if heat_capacity <= 0.0:
            raise ValueError(
                "Calculated heat capacity must be positive."
            )

        return density, heat_capacity

    def __repr__(self):
        return (
            "PropertyModel("
            f"model_name={self.model_name!r})"
        )


class ConstantVolumetricFlow:
    """Constant volumetric flow model for incompressible fluids."""

    def __init__(
        self,
        volumetric_flow,
    ):
        self.volumetric_flow = to_magnitude(
            volumetric_flow,
            ureg.meter**3 / ureg.second,
            "volumetric_flow",
        )

        if not np.isfinite(
            self.volumetric_flow
        ):
            raise ValueError(
                "volumetric_flow must be finite."
            )

        if self.volumetric_flow <= 0.0:
            raise ValueError(
                "volumetric_flow must be positive."
            )

    def evaluate(
        self,
        molar_flows,
        temperature,
        pressure,
    ):
        """Return the local volumetric flow in cubic metres per second."""

        return self.volumetric_flow

    def __repr__(self):
        return (
            "ConstantVolumetricFlow("
            f"volumetric_flow="
            f"{self.volumetric_flow!r})"
        )