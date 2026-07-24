import numpy as np

import numpy as np


class BatchReactor:
    """Ideal closed batch reactor.

    The reactor contributes no inlet or outlet terms. In the simplest
    adiabatic and constant-volume formulation, all state changes originate
    from the reaction mechanism.
    """

    def chemical_state(
        self,
        state,
        properties,
    ):
        """Return concentrations, temperature, and pressure."""
        state = np.asarray(
            state,
            dtype=float,
        )

        n_species = state.size - 2

        concentrations = state[:n_species]
        temperature = state[n_species]
        pressure = state[n_species + 1]

        density, heat_capacity = properties.evaluate(
            concentrations,
            temperature,
            pressure,
        )

        context = {
            "density": density,
            "heat_capacity": heat_capacity,
        }

        return state, context

    def transformation_operator(
        self,
        chemical_contribution,
        context,
    ):
        """Apply the batch-reactor identity transformation."""
        return np.asarray(
            chemical_contribution,
            dtype=float,
        ).copy()

    def derivative_contribution(
        self,
        xi,
        state,
        context,
    ):
        """Return non-chemical batch-reactor contributions."""
        return np.zeros_like(
            state,
            dtype=float,
        )
# class BatchReactor:
#     """Ideal closed batch reactor.

#     The reactor contributes no inlet or outlet terms. In the simplest
#     adiabatic and constant-volume formulation, all state changes originate
#     from the reaction mechanism.
#     """
#     def derivative_contribution(
#         self,
#         state,
#         density,
#         heat_capacity,
#     ):
#         """Return the reactor contribution to the full state derivative."""
#         state = np.asarray(state, dtype=float)

#         if state.ndim != 1:
#             raise ValueError("state must be a one-dimensional vector.")

#         if density <= 0:
#             raise ValueError("density must be positive.")

#         if heat_capacity <= 0:
#             raise ValueError("heat_capacity must be positive.")

#         return np.zeros_like(state, dtype=float)

#     def transformation_operator(
#         self,
#         chemical_contribution,
#     ):
#         """Return the unchanged chemical contribution vector.

#         For a batch reactor, the reactor transformation operator is the
#         identity:

#             T_batch(s_chemical) = s_chemical
#         """
#         return np.asarray(
#             chemical_contribution,
#             dtype=float,
#         )

#     def __repr__(self):
#         return "BatchReactor(mode='closed, adiabatic, constant-volume')"

class PlugFlowReactor:
    """Steady-state plug-flow reactor using molar flows."""

    def __init__(
        self,
        molar_masses,
    ):
        self.molar_masses = np.asarray(
            molar_masses,
            dtype=float,
        )

        if self.molar_masses.ndim != 1:
            raise ValueError(
                "molar_masses must be one-dimensional."
            )

        if np.any(self.molar_masses <= 0):
            raise ValueError(
                "All molar masses must be positive."
            )

    def chemical_state(
        self,
        state,
        properties,
    ):
        """Convert molar flows into the local chemical state."""
        state = np.asarray(
            state,
            dtype=float,
        )

        n_species = len(self.molar_masses)
        expected_length = n_species + 2

        if state.shape != (expected_length,):
            raise ValueError(
                f"State must have length {expected_length}."
            )

        molar_flows = state[:n_species]
        temperature = state[n_species]
        pressure = state[n_species + 1]

        if np.any(molar_flows < 0):
            raise ValueError(
                "Molar flows must not be negative."
            )

        total_molar_flow = np.sum(
            molar_flows
        )

        if total_molar_flow <= 0:
            raise ValueError(
                "Total molar flow must be positive."
            )

        mole_fractions = (
            molar_flows
            / total_molar_flow
        )

        density, heat_capacity = (
            properties.evaluate_from_composition(
                mole_fractions,
                temperature,
                pressure,
            )
        )

        mass_flow = np.dot(
            molar_flows,
            self.molar_masses,
        )

        volumetric_flow = (
            mass_flow
            / density
        )

        concentrations = (
            molar_flows
            / volumetric_flow
        )

        chemical_state = np.concatenate(
            [
                concentrations,
                [
                    temperature,
                    pressure,
                ],
            ]
        )

        context = {
            "density": density,
            "heat_capacity": heat_capacity,
            "volumetric_flow": volumetric_flow,
            "mass_flow": mass_flow,
            "mole_fractions": mole_fractions,
        }

        return chemical_state, context

    def transformation_operator(
        self,
        chemical_contribution,
        context,
    ):
        """Transform chemical rates into derivatives over reactor volume."""
        chemical_contribution = np.asarray(
            chemical_contribution,
            dtype=float,
        )

        transformed = chemical_contribution.copy()

        volumetric_flow = context[
            "volumetric_flow"
        ]

        # Species:
        #
        #     dF_i/dV = R_i
        #
        # The species source terms remain unchanged.

        # Temperature:
        #
        #     dT/dV = (dT/dt) / volumetric_flow
        transformed[-2] /= volumetric_flow

        # Chemistry does not directly determine the pressure gradient.
        transformed[-1] = 0.0

        return transformed

    def derivative_contribution(
        self,
        xi,
        state,
        context,
    ):
        """Return non-chemical PFR contributions."""
        return np.zeros_like(
            state,
            dtype=float,
        )