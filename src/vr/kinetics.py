import numpy as np

R_GAS_CONSTANT = 8.314462618  # J mol^-1 K^-1


class Mechanism:
    """Chemical reaction mechanism based on matrix notation."""

    def __init__(
        self,
        species,
        stoichiometry,
        reaction_orders,
        pre_exponential_factors,
        activation_energies,
        reaction_enthalpies=None,
    ):
        self.species = list(species)
        self.stoichiometry = np.asarray(stoichiometry, dtype=float)
        self.reaction_orders = np.asarray(reaction_orders, dtype=float)
        self.pre_exponential_factors = np.asarray(pre_exponential_factors, dtype=float)
        self.activation_energies = np.asarray(activation_energies, dtype=float)

        if reaction_enthalpies is None:
            self.reaction_enthalpies = None
        else:
            self.reaction_enthalpies = np.asarray(reaction_enthalpies, dtype=float)

        self._validate_dimensions()

    def _validate_dimensions(self):
        n_species = len(self.species)
        n_reactions = self.stoichiometry.shape[1]

        if self.stoichiometry.shape != self.reaction_orders.shape:
            raise ValueError("stoichiometry and reaction_orders must have the same shape.")

        if self.stoichiometry.shape[0] != n_species:
            raise ValueError("Number of species must match rows of stoichiometry.")

        if self.pre_exponential_factors.shape[0] != n_reactions:
            raise ValueError("pre_exponential_factors must contain one value per reaction.")

        if self.activation_energies.shape[0] != n_reactions:
            raise ValueError("activation_energies must contain one value per reaction.")

        if self.reaction_enthalpies is not None:
            if self.reaction_enthalpies.shape[0] != n_reactions:
                raise ValueError("reaction_enthalpies must contain one value per reaction.")

    def rate_constants(self, temperature):
        """Return Arrhenius rate constants at temperature in K."""
        if temperature <= 0:
            raise ValueError("Temperature must be positive and given in Kelvin.")

        return self.pre_exponential_factors * np.exp(
            -self.activation_energies / (R_GAS_CONSTANT * temperature)
        )

    def reaction_rates(self, concentrations, temperature):
        """Return reaction rate vector r."""
        concentrations = np.asarray(concentrations, dtype=float)

        if concentrations.shape[0] != len(self.species):
            raise ValueError("Concentration vector length must match number of species.")

        if np.any(concentrations < 0):
            raise ValueError("Concentrations must be non-negative.")

        k = self.rate_constants(temperature)

        concentration_terms = np.prod(
            concentrations[:, None] ** self.reaction_orders,
            axis=0,
        )

        return k * concentration_terms

    def reaction_source(self, concentrations, temperature):
        """Return chemical source term dc/dt caused only by reactions."""
        rates = self.reaction_rates(concentrations, temperature)
        return self.stoichiometry @ rates

    def reaction_heat(self, concentrations, temperature):
        """Return volumetric reaction heat source.

        Positive values mean heat is released by the reaction system.

        reaction_enthalpies should be given in J/mol.
        reaction_rates are assumed to be in mol/(L s) or consistent units.
        """
        if self.reaction_enthalpies is None:
            raise ValueError("reaction_enthalpies were not provided.")

        rates = self.reaction_rates(concentrations, temperature)

        return -np.dot(self.reaction_enthalpies, rates)

    def __repr__(self):
        return (
            f"Mechanism("
            f"species={self.species}, "
            f"n_reactions={self.stoichiometry.shape[1]}, "
            f"has_reaction_enthalpies={self.reaction_enthalpies is not None}"
            f")"
        )