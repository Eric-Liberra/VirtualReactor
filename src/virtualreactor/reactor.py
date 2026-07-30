import numpy as np
from virtualreactor.units import to_magnitude, ureg


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


class PlugFlowReactor:
    """Steady-state plug-flow reactor using molar flows.

    The reactor state is

        [F_1, ..., F_n, T, p]

    where F_i are molar flows, T is temperature, and p is pressure.

    All physical reactor parameters are converted to SI units during
    initialization.
    """

    def __init__(
        self,
        flow_model,
        jacket_temperature=None,
        overall_heat_transfer_coefficient=None,
        heat_transfer_area_per_volume=None,
    ):
        if not hasattr(
            flow_model,
            "evaluate",
        ):
            raise TypeError(
                "flow_model must provide an evaluate() method."
            )

        self.flow_model = flow_model

        self.jacket_temperature = (
            None
            if jacket_temperature is None
            else to_magnitude(
                jacket_temperature,
                ureg.kelvin,
                "jacket_temperature",
            )
        )

        self.overall_heat_transfer_coefficient = (
            0.0
            if overall_heat_transfer_coefficient is None
            else to_magnitude(
                overall_heat_transfer_coefficient,
                ureg.watt
                / (
                    ureg.meter**2
                    * ureg.kelvin
                ),
                "overall_heat_transfer_coefficient",
            )
        )

        self.heat_transfer_area_per_volume = (
            0.0
            if heat_transfer_area_per_volume is None
            else to_magnitude(
                heat_transfer_area_per_volume,
                1 / ureg.meter,
                "heat_transfer_area_per_volume",
            )
        )

        if (
            self.jacket_temperature is not None
            and self.jacket_temperature <= 0.0
        ):
            raise ValueError(
                "jacket_temperature must be positive."
            )

        if self.overall_heat_transfer_coefficient < 0.0:
            raise ValueError(
                "overall_heat_transfer_coefficient "
                "must not be negative."
            )

        if self.heat_transfer_area_per_volume < 0.0:
            raise ValueError(
                "heat_transfer_area_per_volume "
                "must not be negative."
            )

        jacket_enabled = (
            self.jacket_temperature is not None
        )

        heat_transfer_enabled = (
            self.overall_heat_transfer_coefficient > 0.0
            or self.heat_transfer_area_per_volume > 0.0
        )

        if jacket_enabled != heat_transfer_enabled:
            raise ValueError(
                "Jacket heat transfer requires "
                "jacket_temperature, "
                "overall_heat_transfer_coefficient, and "
                "heat_transfer_area_per_volume."
            )

        if jacket_enabled:
            if self.overall_heat_transfer_coefficient <= 0.0:
                raise ValueError(
                    "overall_heat_transfer_coefficient must be "
                    "positive when jacket heat transfer is enabled."
                )

            if self.heat_transfer_area_per_volume <= 0.0:
                raise ValueError(
                    "heat_transfer_area_per_volume must be "
                    "positive when jacket heat transfer is enabled."
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

        if state.ndim != 1:
            raise ValueError(
                "state must be one-dimensional."
            )

        if state.size < 3:
            raise ValueError(
                "state must contain at least one species, "
                "temperature, and pressure."
            )

        n_species = state.size - 2

        molar_flows = state[:n_species]
        temperature = state[n_species]
        pressure = state[n_species + 1]

        volumetric_flow = self.flow_model.evaluate(
            molar_flows=molar_flows,
            temperature=temperature,
            pressure=pressure,
        )

        volumetric_flow = float(
            volumetric_flow
        )

        if volumetric_flow <= 0.0:
            raise ValueError(
                "Calculated volumetric flow must be positive."
            )

        concentrations = (
            molar_flows
            / volumetric_flow
        )

        density, heat_capacity = properties.evaluate(
            concentrations=concentrations,
            temperature=temperature,
            pressure=pressure,
        )

        mass_flow = (
            density
            * volumetric_flow
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

        if volumetric_flow <= 0.0:
            raise ValueError(
                "volumetric_flow must be positive."
            )

        # Species balance:
        #
        #     dF_i / dV = R_i
        #
        # R_i already has units of mol m^-3 s^-1,
        # equivalent to (mol s^-1) / m^3.
        #
        # Therefore, the species contributions do not need
        # to be transformed.

        # Chemical temperature contribution:
        #
        #     dT / dV = (dT / dt) / volumetric_flow
        transformed[-2] /= volumetric_flow

        # Chemistry does not directly determine pressure loss.
        transformed[-1] = 0.0

        return transformed

    def derivative_contribution(
        self,
        xi,
        state,
        context,
    ):
        """Return non-chemical PFR contributions."""
        state = np.asarray(
            state,
            dtype=float,
        )

        derivative = np.zeros_like(
            state,
            dtype=float,
        )

        n_species = state.size - 2
        temperature = state[n_species]

        mass_flow = context[
            "mass_flow"
        ]

        heat_capacity = context[
            "heat_capacity"
        ]

        if mass_flow <= 0.0:
            raise ValueError(
                "mass_flow must be positive."
            )

        if heat_capacity <= 0.0:
            raise ValueError(
                "heat_capacity must be positive."
            )

        external_heat_source = self.heat_transfer_source(
            temperature=temperature,
        )

        derivative[n_species] = (
            external_heat_source
            / (
                mass_flow
                * heat_capacity
            )
        )

        return derivative

    def heat_transfer_source(
        self,
        temperature,
    ):
        """Return the external volumetric heat source in W/m³."""
        if self.jacket_temperature is None:
            return 0.0

        return (
            self.overall_heat_transfer_coefficient
            * self.heat_transfer_area_per_volume
            * (
                self.jacket_temperature
                - temperature
            )
        )

    @classmethod
    def adiabatic(
        cls,
        flow_model,
    ):
        """Create an adiabatic plug-flow reactor."""
        return cls(
            flow_model=flow_model,
        )

    @classmethod
    def with_constant_jacket_temperature(
        cls,
        flow_model,
        jacket_temperature,
        overall_heat_transfer_coefficient,
        heat_transfer_area_per_volume,
    ):
        """Create a PFR with a constant jacket temperature."""
        return cls(
            flow_model=flow_model,
            jacket_temperature=jacket_temperature,
            overall_heat_transfer_coefficient=(
                overall_heat_transfer_coefficient
            ),
            heat_transfer_area_per_volume=(
                heat_transfer_area_per_volume
            ),
        )

    def __repr__(self):
        lines = [
            "PlugFlowReactor(",
            f"    flow_model={self.flow_model!r},",
        ]

        if self.jacket_temperature is None:
            lines.append(
                "    heat_transfer='adiabatic',"
            )

        else:
            lines.extend(
                [
                    "    heat_transfer='constant_jacket_temperature',",
                    (
                        "    jacket_temperature="
                        f"{self.jacket_temperature:g} K,"
                    ),
                    (
                        "    overall_heat_transfer_coefficient="
                        f"{self.overall_heat_transfer_coefficient:g} "
                        "W/(m²·K),"
                    ),
                    (
                        "    heat_transfer_area_per_volume="
                        f"{self.heat_transfer_area_per_volume:g} 1/m,"
                    ),
                ]
            )

        lines.append(")")
        return "\n".join(lines)