import numpy as np
from scipy.integrate import solve_ivp
from .results import SimulationResult
import h5py

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

    def to_hdf5_group(
        self,
        group,
    ):
        """Write the complete simulation model to an HDF5 group."""

        mechanism_group = group.create_group(
            "mechanism"
        )
        self.mechanism.to_hdf5_group(
            mechanism_group
        )

        reactor_group = group.create_group(
            "reactor"
        )
        self.reactor.to_hdf5_group(
            reactor_group
        )

        properties_group = group.create_group(
            "properties"
        )
        self.properties.to_hdf5_group(
            properties_group
        )

    def save_hdf5(
        self,
        path,
    ):
        """Write the simulation model to an HDF5 file."""

        with h5py.File(
            path,
            "w",
        ) as file:
            simulation_group = file.create_group(
                "simulation"
            )

            self.to_hdf5_group(
                simulation_group
            )

    @classmethod
    def from_hdf5_group(
        cls,
        group,
    ):
        """Create a simulation model from an HDF5 group."""

        mechanism = Mechanism.from_hdf5_group(
            group["mechanism"]
        )

        reactor = Reactor.from_hdf5_group(
            group["reactor"]
        )

        properties = PropertyModel.from_hdf5_group(
            group["properties"]
        )

        return cls(
            mechanism=mechanism,
            reactor=reactor,
            properties=properties,
        )

    def rhs(self, xi, state):
        """Evaluate the governing reactor equations.

        The state derivative is computed as

            dy/dξ = T_reactor(s_chemical) + s_reactor

        where

            s_chemical = [
                species contribution,
                temperature contribution,
                pressure contribution,
            ]

        ξ denotes the independent variable:

            ξ = t    for transient reactors (Batch, CSTR)
            ξ = V    for steady plug-flow reactors.
        """
        chemical_state, reactor_context = (
            self.reactor.chemical_state(
                state=state,
                properties=self.properties,
            )
        )

        n_species = len(self.mechanism.species)

        concentrations = chemical_state[:n_species]
        temperature = chemical_state[n_species]
        pressure = chemical_state[n_species + 1]

        density = reactor_context["density"]
        heat_capacity = reactor_context["heat_capacity"]

        chemical_contribution = self.mechanism.chemical_contribution(
            state=chemical_state,
            density=density,
            heat_capacity=heat_capacity,
        )

        transformed_chemical_contribution = (
            self.reactor.transformation_operator(
                chemical_contribution=chemical_contribution,
                context=reactor_context,
            )
        )

        reactor_contribution = self.reactor.derivative_contribution(
            xi=xi,
            state=state,
            context=reactor_context,
        )

        return (
            transformed_chemical_contribution
            + reactor_contribution
        )

    def solve(
        self,
        initial_state,
        xi_span,
        evaluation_times=None,
        method="BDF",
        relative_tolerance=1e-6,
        absolute_tolerance=1e-9,
    ):
        """Integrate the coupled reactor model."""
        initial_state = np.asarray(initial_state, dtype=float)

        solution = solve_ivp(
            fun=self.rhs,
            t_span=xi_span,
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

        return SimulationResult(
            coordinates=solution.t,
            states=solution.y.T,
            species=self.mechanism.species,
            coordinate_name="time",
            species_state_name="concentration",
            success=solution.success,
            message=solution.message,
        )