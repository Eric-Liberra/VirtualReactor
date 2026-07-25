import numpy as np

class PropertyModel:
    """Calculate thermophysical properties from the current state."""

    def __init__(
        self,
        evaluation_function,
        model_name,
        composition_evaluation_function=None,
    ):
        self._evaluation_function = evaluation_function
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
        density = float(density)
        heat_capacity = float(heat_capacity)

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
            composition_evaluation_function=evaluate_constant,
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

        # if np.any(concentrations < 0.0):
        #     raise ValueError(
        #         "concentrations must not be negative."
        #     )

        density, heat_capacity = self._evaluation_function(
            concentrations,
            temperature,
            pressure,
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

        # if np.any(mole_fractions < 0.0):
        #     raise ValueError(
        #         "mole_fractions must not be negative."
        #     )

        total = np.sum(mole_fractions)

        if total <= 0.0:
            raise ValueError(
                "Mole fractions must have a positive sum."
            )

        mole_fractions = mole_fractions / total

        if self._composition_evaluation_function is None:
            raise NotImplementedError(
                f"Property model {self.model_name!r} does not support "
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
        """Validate and return calculated properties."""
        density = float(density)
        heat_capacity = float(heat_capacity)

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
            f"PropertyModel("
            f"model_name={self.model_name!r})"
        )



# class PropertyModel:
#     """Calculate thermophysical properties from the current state."""

#     def __init__(self, evaluation_function, model_name):
#         self._evaluation_function = evaluation_function
#         self.model_name = model_name

#     @classmethod
#     def constant(
#         cls,
#         density,
#         heat_capacity,
#     ):
#         """Create a model with constant thermophysical properties."""
#         density = float(density)
#         heat_capacity = float(heat_capacity)

#         if density <= 0.0:
#             raise ValueError("density must be positive.")

#         if heat_capacity <= 0.0:
#             raise ValueError("heat_capacity must be positive.")

#         def evaluate_constant(
#             concentrations,
#             temperature,
#             pressure,
#         ):
#             return density, heat_capacity

#         return cls(
#             evaluation_function=evaluate_constant,
#             model_name="constant",
#         )

#     def evaluate(
#         self,
#         concentrations,
#         temperature,
#         pressure,
#     ):
#         """Return density and mass-specific heat capacity."""
#         density, heat_capacity = self._evaluation_function(
#             concentrations,
#             temperature,
#             pressure,
#         )

#         if density <= 0.0:
#             raise ValueError(
#                 "Calculated density must be positive."
#             )

#         if heat_capacity <= 0.0:
#             raise ValueError(
#                 "Calculated heat capacity must be positive."
#             )

#         return float(density), float(heat_capacity)

#     def __repr__(self):
#         return (
#             f"PropertyModel("
#             f"model_name={self.model_name!r})"
#         )