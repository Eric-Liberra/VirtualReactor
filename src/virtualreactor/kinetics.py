import numpy as np

R_GAS_CONSTANT = 8.314462618  # J mol^-1 K^-1


class Mechanism:
    """Chemical reaction mechanism based on matrix notation.

    One column of ``stoichiometry`` represents one chemical reaction.
    Reversible reactions are evaluated as net rates:

        r_net = r_forward - r_reverse

    For reversible reactions, the reverse rate constant is derived from
    thermodynamics at the current temperature:

        delta_G(T) = delta_H - T * delta_S
        K(T) = exp(-delta_G(T) / (R * T))
        k_reverse(T) = k_forward(T) / K(T)

    This implementation assumes constant reaction enthalpies and reaction
    entropies over the simulated temperature range.
    """

    def __init__(
        self,
        species,
        stoichiometry,
        reaction_orders=None,
        pre_exponential_factors=None,
        activation_energies=None,
        reaction_enthalpies=None,
        reaction_entropies=None,
        reversible=None,
        reverse_reaction_orders=None,
    ):
        self.species = list(species)

        self.stoichiometry = np.asarray(
            stoichiometry,
            dtype=float,
        )

        if self.stoichiometry.ndim != 2:
            raise ValueError(
                "stoichiometry must be a two-dimensional matrix."
            )

        n_species, n_reactions = self.stoichiometry.shape

        # Elementary forward reaction:
        # reactants are represented by negative stoichiometric coefficients.
        if reaction_orders is None:
            self.reaction_orders = np.clip(
                -self.stoichiometry,
                0.0,
                None,
            )
        else:
            self.reaction_orders = np.asarray(
                reaction_orders,
                dtype=float,
            )

        self.pre_exponential_factors = np.asarray(
            pre_exponential_factors,
            dtype=float,
        )

        self.activation_energies = np.asarray(
            activation_energies,
            dtype=float,
        )

        if reaction_enthalpies is None:
            self.reaction_enthalpies = None
        else:
            self.reaction_enthalpies = np.asarray(
                reaction_enthalpies,
                dtype=float,
            )

        if reaction_entropies is None:
            self.reaction_entropies = None
        else:
            self.reaction_entropies = np.asarray(
                reaction_entropies,
                dtype=float,
            )

        if reversible is None:
            self.reversible = np.zeros(
                n_reactions,
                dtype=bool,
            )
        else:
            self.reversible = np.asarray(
                reversible,
                dtype=bool,
            )

        # Elementary reverse reaction:
        # products of the forward reaction become reverse reactants.
        if reverse_reaction_orders is None:
            self.reverse_reaction_orders = np.clip(
                self.stoichiometry,
                0.0,
                None,
            )
        else:
            self.reverse_reaction_orders = np.asarray(
                reverse_reaction_orders,
                dtype=float,
            )

        self._validate_dimensions()
        self._validate_values()

    def _validate_dimensions(self):
        """Validate dimensions of all mechanism arrays."""
        n_species = len(self.species)
        n_reactions = self.stoichiometry.shape[1]
        expected_matrix_shape = (n_species, n_reactions)

        if self.stoichiometry.shape != expected_matrix_shape:
            raise ValueError(
                "Number of species must match the rows of stoichiometry."
            )

        if self.reaction_orders.shape != expected_matrix_shape:
            raise ValueError(
                "reaction_orders must have the same shape as "
                "stoichiometry."
            )

        if self.reverse_reaction_orders.shape != expected_matrix_shape:
            raise ValueError(
                "reverse_reaction_orders must have the same shape as "
                "stoichiometry."
            )

        if self.pre_exponential_factors.shape != (n_reactions,):
            raise ValueError(
                "pre_exponential_factors must contain one value "
                "per reaction."
            )

        if self.activation_energies.shape != (n_reactions,):
            raise ValueError(
                "activation_energies must contain one value "
                "per reaction."
            )

        if self.reversible.shape != (n_reactions,):
            raise ValueError(
                "reversible must contain one Boolean value "
                "per reaction."
            )

        if self.reaction_enthalpies is not None:
            if self.reaction_enthalpies.shape != (n_reactions,):
                raise ValueError(
                    "reaction_enthalpies must contain one value "
                    "per reaction."
                )

        if self.reaction_entropies is not None:
            if self.reaction_entropies.shape != (n_reactions,):
                raise ValueError(
                    "reaction_entropies must contain one value "
                    "per reaction."
                )

    def _validate_values(self):
        """Validate numerical values and required thermodynamic data."""
        if not np.all(np.isfinite(self.stoichiometry)):
            raise ValueError("stoichiometry must contain finite values.")

        if not np.all(np.isfinite(self.reaction_orders)):
            raise ValueError("reaction_orders must contain finite values.")

        if not np.all(np.isfinite(self.reverse_reaction_orders)):
            raise ValueError(
                "reverse_reaction_orders must contain finite values."
            )

        if np.any(self.reaction_orders < 0):
            raise ValueError("reaction_orders cannot be negative.")

        if np.any(self.reverse_reaction_orders < 0):
            raise ValueError(
                "reverse_reaction_orders cannot be negative."
            )

        if not np.all(np.isfinite(self.pre_exponential_factors)):
            raise ValueError(
                "pre_exponential_factors must contain finite values."
            )

        if np.any(self.pre_exponential_factors < 0):
            raise ValueError(
                "pre_exponential_factors cannot be negative."
            )

        if not np.all(np.isfinite(self.activation_energies)):
            raise ValueError(
                "activation_energies must contain finite values."
            )

        if np.any(self.reversible):
            if self.reaction_enthalpies is None:
                raise ValueError(
                    "Reversible reactions require reaction_enthalpies."
                )

            if self.reaction_entropies is None:
                raise ValueError(
                    "Reversible reactions require reaction_entropies."
                )

        if self.reaction_enthalpies is not None:
            if not np.all(np.isfinite(self.reaction_enthalpies)):
                raise ValueError(
                    "reaction_enthalpies must contain finite values."
                )

        if self.reaction_entropies is not None:
            if not np.all(np.isfinite(self.reaction_entropies)):
                raise ValueError(
                    "reaction_entropies must contain finite values."
                )

    @staticmethod
    def _validate_temperature(temperature):
        """Return a validated scalar temperature in kelvin."""
        temperature = float(temperature)

        if not np.isfinite(temperature):
            raise ValueError("Temperature must be finite.")

        if temperature <= 0:
            raise ValueError(
                "Temperature must be positive and given in Kelvin."
            )

        return temperature

    def reaction_gibbs_energies(self, temperature):
        """Return reaction Gibbs energies at the current temperature.

        Parameters
        ----------
        temperature : float
            Temperature in K.

        Returns
        -------
        numpy.ndarray
            Reaction Gibbs energies in J mol^-1.
        """
        temperature = self._validate_temperature(temperature)

        if self.reaction_enthalpies is None:
            raise ValueError(
                "reaction_enthalpies were not provided."
            )

        if self.reaction_entropies is None:
            raise ValueError(
                "reaction_entropies were not provided."
            )

        return (
            self.reaction_enthalpies
            - temperature * self.reaction_entropies
        )

    def equilibrium_constants(self, temperature):
        """Return thermodynamic equilibrium constants at temperature."""
        temperature = self._validate_temperature(temperature)

        delta_g = self.reaction_gibbs_energies(
            temperature
        )

        exponent = (
            -delta_g
            / (R_GAS_CONSTANT * temperature)
        )

        # Avoid silent floating-point overflow in exp for extreme inputs.
        max_exponent = np.log(np.finfo(float).max)
        exponent = np.clip(
            exponent,
            -max_exponent,
            max_exponent,
        )

        return np.exp(exponent)

    def forward_rate_constants(self, temperature):
        """Return forward Arrhenius rate constants."""
        temperature = self._validate_temperature(temperature)

        return (
            self.pre_exponential_factors
            * np.exp(
                -self.activation_energies
                / (R_GAS_CONSTANT * temperature)
            )
        )

    def reverse_rate_constants(
        self,
        temperature,
        forward_rate_constants=None,
    ):
        """Return reverse rate constants.

        Irreversible reactions receive a reverse rate constant of zero.
        """
        temperature = self._validate_temperature(temperature)

        if forward_rate_constants is None:
            forward_rate_constants = self.forward_rate_constants(
                temperature
            )
        else:
            forward_rate_constants = np.asarray(
                forward_rate_constants,
                dtype=float,
            )

        reverse_rate_constants = np.zeros_like(
            forward_rate_constants
        )

        if not np.any(self.reversible):
            return reverse_rate_constants

        equilibrium_constants = self.equilibrium_constants(
            temperature
        )

        reverse_rate_constants[self.reversible] = (
            forward_rate_constants[self.reversible]
            / equilibrium_constants[self.reversible]
        )

        return reverse_rate_constants

    def rate_constants(self, temperature):
        """Return forward and reverse rate constants.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(k_forward, k_reverse)``.
        """
        forward_rate_constants = self.forward_rate_constants(
            temperature
        )

        reverse_rate_constants = self.reverse_rate_constants(
            temperature,
            forward_rate_constants=forward_rate_constants,
        )

        return (
            forward_rate_constants,
            reverse_rate_constants,
        )

    def directional_reaction_rates(
        self,
        concentrations,
        temperature,
    ):
        """Return forward and reverse reaction-rate vectors."""
        concentrations = np.asarray(
            concentrations,
            dtype=float,
        )

        if concentrations.shape != (len(self.species),):
            raise ValueError(
                "Concentration vector length must match "
                "the number of species."
            )

        if not np.all(np.isfinite(concentrations)):
            raise ValueError(
                "Concentrations must contain finite values."
            )

        # if np.any(concentrations < 0):
        #     raise ValueError(
        #         "Concentrations cannot be negative."
        #     )

        (
            forward_rate_constants,
            reverse_rate_constants,
        ) = self.rate_constants(temperature)

        forward_concentration_terms = np.prod(
            concentrations[:, None]
            ** self.reaction_orders,
            axis=0,
        )

        reverse_concentration_terms = np.prod(
            concentrations[:, None]
            ** self.reverse_reaction_orders,
            axis=0,
        )

        forward_rates = (
            forward_rate_constants
            * forward_concentration_terms
        )

        reverse_rates = (
            reverse_rate_constants
            * reverse_concentration_terms
        )

        return forward_rates, reverse_rates

    def reaction_rates(
        self,
        concentrations,
        temperature,
    ):
        """Return net reaction-rate vector.

        Each entry corresponds to one chemical reaction:

            r_net = r_forward - r_reverse
        """
        (
            forward_rates,
            reverse_rates,
        ) = self.directional_reaction_rates(
            concentrations,
            temperature,
        )

        return forward_rates - reverse_rates

    def reaction_source(self, rates):
        """Return chemical concentration source term."""
        rates = np.asarray(
            rates,
            dtype=float,
        )

        n_reactions = self.stoichiometry.shape[1]

        if rates.shape != (n_reactions,):
            raise ValueError(
                "rates must contain one value per reaction."
            )

        return self.stoichiometry @ rates

    def reaction_heat(self, rates):
        """Return volumetric reaction heat source.

        Positive values mean heat is released by the reaction system.
        The supplied rates must be net rates.
        """
        if self.reaction_enthalpies is None:
            raise ValueError(
                "reaction_enthalpies were not provided."
            )

        rates = np.asarray(
            rates,
            dtype=float,
        )

        n_reactions = self.stoichiometry.shape[1]

        if rates.shape != (n_reactions,):
            raise ValueError(
                "rates must contain one value per reaction."
            )

        return -np.dot(
            self.reaction_enthalpies,
            rates,
        )

    def chemical_contribution(
        self,
        state,
        density,
        heat_capacity,
    ):
        """Return the chemical contribution vector.

        The vector contains the species production rates, the hypothetical
        temperature rate caused by reaction heat, and a zero pressure
        contribution.
        """
        state = np.asarray(
            state,
            dtype=float,
        )

        n_species = len(self.species)
        expected_length = n_species + 2

        if state.shape != (expected_length,):
            raise ValueError(
                f"State must have length {expected_length}: "
                f"{n_species} concentrations, temperature, "
                "and pressure."
            )

        concentrations = state[:n_species]
        temperature = state[n_species]

        # reaction_rates() recalculates delta_G(T), K(T), k_forward(T),
        # and k_reverse(T) using the current state temperature on every
        # RHS evaluation.
        rates = self.reaction_rates(
            concentrations,
            temperature,
        )

        chemical_concentration_contribution = self.reaction_source(
            rates
        )

        if density <= 0:
            raise ValueError(
                "Density must be positive."
            )

        if heat_capacity <= 0:
            raise ValueError(
                "Heat capacity must be positive."
            )

        if self.reaction_enthalpies is None:
            chemical_temperature_contribution = 0.0
        else:
            heat_source = self.reaction_heat(
                rates
            )

            chemical_temperature_contribution = (
                heat_source*1000
                / (density * heat_capacity)
            )

        chemical_pressure_contribution = 0.0

        return np.concatenate(
            [
                chemical_concentration_contribution,
                [
                    chemical_temperature_contribution,
                    chemical_pressure_contribution,
                ],
            ]
        )

    def __repr__(self):
        n_reactions = self.stoichiometry.shape[1]
        n_reversible = int(np.count_nonzero(self.reversible))

        return (
            f"Mechanism("
            f"species={self.species}, "
            f"n_reactions={n_reactions}, "
            f"n_reversible={n_reversible}, "
            f"has_reaction_enthalpies="
            f"{self.reaction_enthalpies is not None}, "
            f"has_reaction_entropies="
            f"{self.reaction_entropies is not None}"
            f")"
        )