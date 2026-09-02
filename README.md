# VirtualReactor

VirtualReactor is a modular Python framework for the simulation of chemical reactor systems with mechanistic reaction kinetics.

The framework separates reaction chemistry, reactor behavior, thermophysical properties, and numerical simulation into reusable components.

Beyond reactor simulation, VirtualReactor is being developed toward data-driven process optimization, with Bayesian optimization as a planned approach for identifying optimal reactor designs and operating conditions.

## Features

- Generic reaction mechanisms based on stoichiometric matrices
- Arrhenius kinetics
- Reversible reactions derived from thermodynamic data
- Batch reactor simulation
- Plug-flow reactor simulation
- Heat-transfer models for non-isothermal reactors
- Pint-based physical units
- HDF5 serialization of simulation results
- Structured result containers for downstream analysis and optimization

## Architecture

VirtualReactor follows a modular architecture that separates chemical kinetics, reactor behavior, thermophysical properties, and numerical simulation.

```text
Mechanism ────────┐
Reactor ──────────┼──▶ Simulation ──▶ SimulationResult ──▶ HDF5
PropertyModel ────┘
```

The main components are:

- `Mechanism` — defines reaction networks, kinetics, and thermodynamics.
- `Reactor` — defines the reactor-specific state representation and physical behavior.
- `PropertyModel` — provides thermophysical properties such as density and heat capacity.
- `Simulation` — combines the individual components and performs the numerical integration.
- `SimulationResult` — provides structured access to simulation results, visualization, and serialization.

## Examples

The repository contains example notebooks demonstrating different aspects of VirtualReactor:

- **Van de Vusse reaction** — simulation of a multi-reaction kinetic network
- **Plug-flow reactor** — steady-state reactor simulation
- **Chemical equilibrium** — reversible kinetics and thermodynamic equilibrium

The notebooks are available in the [`examples/`](examples/) directory.

## Mathematical Framework

VirtualReactor describes reactor systems through a general state equation

$$
\frac{d\mathbf{y}}{d\xi}
=
\mathcal{T}_{\mathrm{reactor}}
\left(\mathbf{s}_{\mathrm{chemical}}\right)
+
\mathbf{s}_{\mathrm{reactor}},
$$

where the chemical source terms are determined by the reaction mechanism and transformed according to the selected reactor model.

This separation allows the same reaction mechanism to be reused across different reactor configurations.

A more detailed description of the mathematical framework is provided in the project documentation.

## Documentation

A detailed description of the mathematical framework and software architecture is available in the project manual:

[`docs/manual.pdf`](docs/manual.pdf)

The corresponding LaTeX source files are included in the [`docs/`](docs/) directory.

## Roadmap

VirtualReactor is being developed toward a general simulation and optimization framework for chemical reactor systems.

Planned extensions include:

- Bayesian optimization of reactor and process parameters
- Multi-parameter optimization of operating conditions such as temperature, residence time, flow rate, and reactor geometry
- Optimization of product yield and selectivity
- Automated generation of structured simulation datasets
- Integration of surrogate models and data-driven methods
- Self-optimizing reactor workflows combining mechanistic simulation with experimental or synthetic data

A long-term objective is to use VirtualReactor as a virtual environment for developing and testing optimization strategies for chemical processes before transferring them to experimental reactor systems.

## Project Status

VirtualReactor is currently under active development.

The current implementation focuses on modular mechanistic reactor simulation. Future development will extend the framework toward automated parameter studies, data-driven optimization, and self-optimizing reactor workflows.



## Quick Start

The following example demonstrates the basic VirtualReactor workflow for a simple irreversible reaction

$$
A \rightarrow B.
$$

```python
import numpy as np
import virtualreactor as vr
from virtualreactor.units import ureg

# Define the reaction mechanism
mechanism = vr.Mechanism(
    species=["A", "B"],
    stoichiometry=[
        [-1],
        [ 1],
    ],
    reaction_orders=[
        [1],
        [0],
    ],
    pre_exponential_factors=(
        np.array([1.0e3]) / ureg.second
    ),
    activation_energies=(
        np.array([20.0e3])
        * ureg.joule
        / ureg.mole
    ),
)

# Define the reactor
reactor = vr.BatchReactor()

# Define constant thermophysical properties
properties = vr.PropertyModel.constant(
    density=1000.0 * ureg.kilogram / ureg.meter**3,
    heat_capacity=(
        4180.0
        * ureg.joule
        / (ureg.kilogram * ureg.kelvin)
    ),
)

# Build the simulation
simulation = vr.Simulation(
    mechanism=mechanism,
    reactor=reactor,
    properties=properties,
)

# Initial state: [A, B, temperature, pressure]
initial_state = np.array([
    1.0,
    0.0,
    300.0,
    0.0,
])

# Run the simulation
result = simulation.solve(
    initial_state=initial_state,
    xi_span=(0.0, 10.0),
)
```

The same reaction mechanism can be combined with different reactor models without redefining the chemistry. This modularity is a central design principle of VirtualReactor.

More detailed examples, including plug-flow reactor simulations and reversible reaction networks, are available in the [`examples/`](examples/) directory.

## Installation

### Requirements

- Python 3.11 or newer
- `pip`

### Install from source

Clone the repository and install VirtualReactor in editable mode:

```bash
git clone <repository-url>
cd VirtualReactor
pip install -e .
```

The package can then be imported from Python:

```python
import virtualreactor as vr
```

For development and testing, install the optional development dependencies:

```bash
pip install -e ".[dev]"
```

## License

VirtualReactor is released under the MIT License. See [`LICENSE`](LICENSE) for details.