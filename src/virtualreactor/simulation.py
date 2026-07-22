import numpy as np
from scipy.integrate import solve_ivp


class Simulation:
    """Couple a reaction mechanism and a reactor model."""

    def __init__(
        self,
        mechanism,
        reactor,
        properties
    ):
        self.mechanism = mechanism
        self.reactor = reactor
        self.properties=properties

    def rhs(self, time, state):
        """Return the complete time derivative of the state vector."""
        n_species = len(self.mechanism.species)

        concentrations = state[:n_species]
        temperature = state[n_species]
        pressure = state[n_species + 1]

        density, heat_capacity = self.properties.evaluate(
            concentrations,
            temperature,
            pressure,
        )

        mechanism_term = self.mechanism.derivative_contribution(
            state=state,
            density=density,
            heat_capacity=heat_capacity,
        )

        reactor_term = self.reactor.derivative_contribution(
            state=state,
            density=density,
            heat_capacity=heat_capacity,
        )

        return mechanism_term + reactor_term

    def solve(
        self,
        initial_state,
        time_span,
        evaluation_times=None,
        method="BDF",
        relative_tolerance=1e-6,
        absolute_tolerance=1e-9,
    ):
        """Integrate the coupled reactor model."""
        initial_state = np.asarray(initial_state, dtype=float)

        solution = solve_ivp(
            fun=self.rhs,
            t_span=time_span,
            y0=initial_state,
            t_eval=evaluation_times,
            method=method,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )

        if not solution.success:
            raise RuntimeError(
                f"Simulation failed: {solution.message}"
            )

        return solution