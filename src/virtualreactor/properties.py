class PropertyModel:
    """Calculate thermophysical properties from the current state."""

    def __init__(self, evaluation_function, model_name):
        self._evaluation_function = evaluation_function
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
            raise ValueError("density must be positive.")

        if heat_capacity <= 0.0:
            raise ValueError("heat_capacity must be positive.")

        def evaluate_constant(
            concentrations,
            temperature,
            pressure,
        ):
            return density, heat_capacity

        return cls(
            evaluation_function=evaluate_constant,
            model_name="constant",
        )

    def evaluate(
        self,
        concentrations,
        temperature,
        pressure,
    ):
        """Return density and mass-specific heat capacity."""
        density, heat_capacity = self._evaluation_function(
            concentrations,
            temperature,
            pressure,
        )

        if density <= 0.0:
            raise ValueError(
                "Calculated density must be positive."
            )

        if heat_capacity <= 0.0:
            raise ValueError(
                "Calculated heat capacity must be positive."
            )

        return float(density), float(heat_capacity)

    def __repr__(self):
        return (
            f"PropertyModel("
            f"model_name={self.model_name!r})"
        )