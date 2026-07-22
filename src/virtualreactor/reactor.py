import numpy as np


class BatchReactor:
    """Ideal closed batch reactor.

    The reactor contributes no inlet or outlet terms. In the simplest
    adiabatic and constant-volume formulation, all state changes originate
    from the reaction mechanism.
    """

    def derivative_contribution(
        self,
        state,
        density,
        heat_capacity,
    ):
        """Return the reactor contribution to the full state derivative."""
        state = np.asarray(state, dtype=float)

        if state.ndim != 1:
            raise ValueError("state must be a one-dimensional vector.")

        if density <= 0:
            raise ValueError("density must be positive.")

        if heat_capacity <= 0:
            raise ValueError("heat_capacity must be positive.")

        return np.zeros_like(state, dtype=float)

    def __repr__(self):
        return "BatchReactor(mode='closed, adiabatic, constant-volume')"