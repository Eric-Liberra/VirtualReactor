from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


class SimulationResult:
    """Store, inspect, plot, and serialize one simulation result."""

    FILE_FORMAT = "VirtualReactor SimulationResult"
    SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        coordinates,
        states,
        species,
        coordinate_name="time",
        species_state_name="concentration",
        success=True,
        message="",
    ):
        self.coordinates = np.asarray(
            coordinates,
            dtype=float,
        )

        self.states = np.asarray(
            states,
            dtype=float,
        )

        self.species = [
            str(name)
            for name in species
        ]

        self.coordinate_name = str(
            coordinate_name
        )

        self.species_state_name = str(
            species_state_name
        )

        self.success = bool(success)
        self.message = str(message)

        self._validate()

    def _validate(self):
        """Validate the stored simulation data."""

        if self.coordinates.ndim != 1:
            raise ValueError(
                "coordinates must be one-dimensional."
            )

        if self.states.ndim != 2:
            raise ValueError(
                "states must be two-dimensional."
            )

        if self.states.shape[0] != self.coordinates.size:
            raise ValueError(
                "states and coordinates must contain "
                "the same number of points."
            )

        if len(self.species) == 0:
            raise ValueError(
                "species must contain at least one species."
            )

        if len(set(self.species)) != len(self.species):
            raise ValueError(
                "species names must be unique."
            )

        expected_state_size = len(self.species) + 2

        if self.states.shape[1] != expected_state_size:
            raise ValueError(
                f"Each state must have length "
                f"{expected_state_size}: "
                f"{len(self.species)} species values, "
                "temperature, and pressure."
            )

        if not np.all(
            np.isfinite(self.coordinates)
        ):
            raise ValueError(
                "coordinates must contain only finite values."
            )

        if not np.all(
            np.isfinite(self.states)
        ):
            raise ValueError(
                "states must contain only finite values."
            )

    @property
    def n_points(self):
        """Return the number of stored coordinate points."""
        return self.coordinates.size

    @property
    def n_species(self):
        """Return the number of chemical species."""
        return len(self.species)

    @property
    def species_states(self):
        """Return the species-related state values."""
        return self.states[:, :self.n_species]

    @property
    def concentrations(self):
        """Return the species states as concentrations.

        This alias should only be used when species_state_name is
        'concentration'.
        """

        if self.species_state_name != "concentration":
            raise AttributeError(
                "The stored species states are not concentrations. "
                f"They are marked as "
                f"{self.species_state_name!r}."
            )

        return self.species_states

    @property
    def molar_flows(self):
        """Return the species states as molar flows.

        This alias should only be used when species_state_name is
        'molar_flow'.
        """

        # if self.species_state_name != "molar_flow":
        #     raise AttributeError(
        #         "The stored species states are not molar flows. "
        #         f"They are marked as "
        #         f"{self.species_state_name!r}."
        #     )

        return self.species_states

    @property
    def temperature(self):
        """Return the temperature profile."""
        return self.states[:, self.n_species]

    @property
    def pressure(self):
        """Return the pressure profile."""
        return self.states[:, self.n_species + 1]

    def species_index(
        self,
        name,
    ):
        """Return the state-column index of a species."""

        try:
            return self.species.index(name)

        except ValueError as error:
            available_species = ", ".join(
                self.species
            )

            raise ValueError(
                f"Unknown species {name!r}. "
                f"Available species: "
                f"{available_species}."
            ) from error

    def plot_species_vs_coordinate(
        self,
        species=None,
        ax=None,
    ):
        """Plot species states against the simulation coordinate."""

        if ax is None:
            _, ax = plt.subplots(
                figsize=(7.0, 4.5),
                constrained_layout=True,
            )

        selected_species = self._normalize_species_selection(
            species
        )

        for name in selected_species:
            index = self.species_index(name)

            ax.plot(
                self.coordinates,
                self.species_states[:, index],
                linewidth=2,
                label=name,
            )

        ax.set_xlabel(
            self.coordinate_name.replace("_", " ").title()
        )

        ax.set_ylabel(
            self.species_state_name.replace("_", " ").title()
        )

        ax.legend(frameon=False)
        ax.grid(alpha=0.25)

        return ax

    def plot_species_vs_temperature(
        self,
        species=None,
        ax=None,
    ):
        """Plot species states against temperature."""

        if ax is None:
            _, ax = plt.subplots(
                figsize=(7.0, 4.5),
                constrained_layout=True,
            )

        selected_species = self._normalize_species_selection(
            species
        )

        for name in selected_species:
            index = self.species_index(name)

            ax.plot(
                self.temperature,
                self.species_states[:, index],
                linewidth=2,
                label=name,
            )

        ax.set_xlabel("Temperature")
        ax.set_ylabel(
            self.species_state_name.replace("_", " ").title()
        )

        ax.legend(frameon=False)
        ax.grid(alpha=0.25)

        return ax

    def _normalize_species_selection(
        self,
        species,
    ):
        """Normalize a species selection to a list."""

        if species is None:
            return list(self.species)

        if isinstance(species, str):
            selected_species = [species]

        else:
            selected_species = list(species)

        for name in selected_species:
            self.species_index(name)

        return selected_species

    def append_to_hdf5(
        self,
        path,
        *,
        group_name="result",
        overwrite=False,
        compression="gzip",
        compression_level=4,
    ):
        """Append this result to an existing HDF5 file."""

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"HDF5 file does not exist: {path}."
            )

        with h5py.File(
            path,
            mode="a",
        ) as file:
            if group_name in file:
                if not overwrite:
                    raise FileExistsError(
                        f"HDF5 group {group_name!r} "
                        f"already exists in {path}. "
                        "Pass overwrite=True to replace it."
                    )

                del file[group_name]

            result_group = file.create_group(
                group_name
            )

            result_group.attrs["file_format"] = (
                self.FILE_FORMAT
            )
            result_group.attrs["schema_version"] = (
                self.SCHEMA_VERSION
            )

            self.to_hdf5_group(
                result_group,
                compression=compression,
                compression_level=compression_level,
            )

        return path

    def save_hdf5(
        self,
        path,
        *,
        overwrite=False,
        compression="gzip",
        compression_level=4,
    ):
        """Save this result to a standalone HDF5 file."""

        path = Path(path)

        if path.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {path}. "
                "Pass overwrite=True to replace it."
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with h5py.File(
            path,
            mode="w",
        ) as file:
            file.attrs["file_format"] = self.FILE_FORMAT
            file.attrs["schema_version"] = (
                self.SCHEMA_VERSION
            )

            self.to_hdf5_group(
                file,
                compression=compression,
                compression_level=compression_level,
            )

        return path

    @classmethod
    def load_hdf5(
        cls,
        path,
    ):
        """Load a standalone SimulationResult HDF5 file."""

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"File does not exist: {path}."
            )

        with h5py.File(
            path,
            mode="r",
        ) as file:
            file_format = cls._read_string_attribute(
                file,
                "file_format",
                default="",
            )

            if file_format != cls.FILE_FORMAT:
                raise ValueError(
                    "The file is not a supported "
                    "VirtualReactor SimulationResult file."
                )

            schema_version = cls._read_string_attribute(
                file,
                "schema_version",
                default="",
            )

            if schema_version != cls.SCHEMA_VERSION:
                raise ValueError(
                    "Unsupported SimulationResult schema "
                    f"version: {schema_version!r}."
                )

            return cls.from_hdf5_group(file)

    def to_hdf5_group(
        self,
        group,
        *,
        compression="gzip",
        compression_level=4,
    ):
        """Write this result into an existing HDF5 group."""

        group.attrs["coordinate_name"] = (
            self.coordinate_name
        )

        group.attrs["species_state_name"] = (
            self.species_state_name
        )

        group.attrs["success"] = self.success
        group.attrs["message"] = self.message

        group.create_dataset(
            "coordinates",
            data=self.coordinates,
            compression=compression,
            compression_opts=compression_level,
            shuffle=True,
        )

        group.create_dataset(
            "states",
            data=self.states,
            compression=compression,
            compression_opts=compression_level,
            shuffle=True,
        )

        string_dtype = h5py.string_dtype(
            encoding="utf-8"
        )

        group.create_dataset(
            "species",
            data=np.asarray(
                self.species,
                dtype=object,
            ),
            dtype=string_dtype,
        )

    @classmethod
    def from_hdf5_group(
        cls,
        group,
    ):
        """Create a SimulationResult from an HDF5 group."""

        required_datasets = {
            "coordinates",
            "states",
            "species",
        }

        missing_datasets = (
            required_datasets - set(group.keys())
        )

        if missing_datasets:
            missing = ", ".join(
                sorted(missing_datasets)
            )

            raise ValueError(
                f"Missing HDF5 datasets: {missing}."
            )

        coordinates = group["coordinates"][...]
        states = group["states"][...]

        species = [
            name.decode("utf-8")
            if isinstance(name, bytes)
            else str(name)
            for name in group["species"][...]
        ]

        coordinate_name = cls._read_string_attribute(
            group,
            "coordinate_name",
            default="time",
        )

        species_state_name = cls._read_string_attribute(
            group,
            "species_state_name",
            default="concentration",
        )

        success = bool(
            group.attrs.get(
                "success",
                True,
            )
        )

        message = cls._read_string_attribute(
            group,
            "message",
            default="",
        )

        return cls(
            coordinates=coordinates,
            states=states,
            species=species,
            coordinate_name=coordinate_name,
            species_state_name=species_state_name,
            success=success,
            message=message,
        )

    @staticmethod
    def _read_string_attribute(
        group,
        name,
        default="",
    ):
        """Read and decode a string-valued HDF5 attribute."""

        value = group.attrs.get(
            name,
            default,
        )

        if isinstance(value, bytes):
            return value.decode("utf-8")

        return str(value)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"n_points={self.n_points}, "
            f"n_species={self.n_species}, "
            f"coordinate={self.coordinate_name!r}, "
            f"species_state={self.species_state_name!r}, "
            f"success={self.success})"
        )


class SimulationSeries:
    """Store a collection of parametrized simulation results."""

    FILE_FORMAT = "VirtualReactor SimulationSeries"
    SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        name,
        results=None,
        metadata=None,
    ):
        self.name = str(name)

        self.results = []

        self.metadata = (
            {}
            if metadata is None
            else dict(metadata)
        )

        if results is not None:
            for entry in results:
                self.add_result(
                    parameters=entry["parameters"],
                    result=entry["result"],
                )

    def add_result(
        self,
        parameters,
        result,
    ):
        """Add one parametrized simulation result."""

        if not isinstance(parameters, dict):
            raise TypeError(
                "parameters must be a dictionary."
            )

        if not isinstance(result, SimulationResult):
            raise TypeError(
                "result must be a SimulationResult."
            )

        normalized_parameters = {
            str(name): self._normalize_parameter_value(value)
            for name, value in parameters.items()
        }

        self.results.append(
            {
                "parameters": normalized_parameters,
                "result": result,
            }
        )

    def __len__(self):
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(
        self,
        index,
    ):
        return self.results[index]

    @property
    def parameter_names(self):
        """Return all parameter names used in the series."""

        names = set()

        for entry in self.results:
            names.update(
                entry["parameters"].keys()
            )

        return sorted(names)

    def save_hdf5(
        self,
        path,
        *,
        overwrite=False,
        compression="gzip",
        compression_level=4,
    ):
        """Save the complete simulation series to one HDF5 file."""

        path = Path(path)

        if path.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {path}. "
                "Pass overwrite=True to replace it."
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with h5py.File(
            path,
            mode="w",
        ) as file:
            file.attrs["file_format"] = self.FILE_FORMAT
            file.attrs["schema_version"] = (
                self.SCHEMA_VERSION
            )
            file.attrs["name"] = self.name
            file.attrs["n_results"] = len(self)

            metadata_group = file.create_group(
                "metadata"
            )

            self._write_attributes(
                metadata_group,
                self.metadata,
            )

            simulations_group = file.create_group(
                "simulations"
            )

            for index, entry in enumerate(self.results):
                run_group = simulations_group.create_group(
                    f"{index:06d}"
                )

                run_group.attrs["run_index"] = index

                parameters_group = run_group.create_group(
                    "parameters"
                )

                self._write_attributes(
                    parameters_group,
                    entry["parameters"],
                )

                result_group = run_group.create_group(
                    "result"
                )

                entry["result"].to_hdf5_group(
                    result_group,
                    compression=compression,
                    compression_level=compression_level,
                )

            self._write_run_table(file)

        return path

    @classmethod
    def load_hdf5(
        cls,
        path,
    ):
        """Load a SimulationSeries from an HDF5 file."""

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"File does not exist: {path}."
            )

        with h5py.File(
            path,
            mode="r",
        ) as file:
            file_format = (
                SimulationResult._read_string_attribute(
                    file,
                    "file_format",
                    default="",
                )
            )

            if file_format != cls.FILE_FORMAT:
                raise ValueError(
                    "The file is not a supported "
                    "VirtualReactor SimulationSeries file."
                )

            schema_version = (
                SimulationResult._read_string_attribute(
                    file,
                    "schema_version",
                    default="",
                )
            )

            if schema_version != cls.SCHEMA_VERSION:
                raise ValueError(
                    "Unsupported SimulationSeries schema "
                    f"version: {schema_version!r}."
                )

            name = (
                SimulationResult._read_string_attribute(
                    file,
                    "name",
                    default="Unnamed simulation series",
                )
            )

            metadata = {}

            if "metadata" in file:
                metadata = cls._read_attributes(
                    file["metadata"]
                )

            series = cls(
                name=name,
                metadata=metadata,
            )

            if "simulations" not in file:
                return series

            simulations_group = file["simulations"]

            run_names = sorted(
                simulations_group.keys()
            )

            for run_name in run_names:
                run_group = simulations_group[run_name]

                if "parameters" not in run_group:
                    raise ValueError(
                        f"Run {run_name!r} has no "
                        "'parameters' group."
                    )

                if "result" not in run_group:
                    raise ValueError(
                        f"Run {run_name!r} has no "
                        "'result' group."
                    )

                parameters = cls._read_attributes(
                    run_group["parameters"]
                )

                result = SimulationResult.from_hdf5_group(
                    run_group["result"]
                )

                series.add_result(
                    parameters=parameters,
                    result=result,
                )

        return series

    def select(
        self,
        **parameter_filters,
    ):
        """Return entries matching exact parameter values.

        Example
        -------
        series.select(
            jacket_temperature=300.0,
            reaction_enthalpy_1=-50000.0,
        )
        """

        matches = []

        for entry in self.results:
            parameters = entry["parameters"]

            is_match = all(
                name in parameters
                and parameters[name] == value
                for name, value in parameter_filters.items()
            )

            if is_match:
                matches.append(entry)

        return matches

    def _write_run_table(
        self,
        file,
    ):
        """Write a compact summary table for fast inspection."""

        run_table_group = file.create_group(
            "run_table"
        )

        run_table_group.create_dataset(
            "run_index",
            data=np.arange(
                len(self),
                dtype=int,
            ),
        )

        run_table_group.create_dataset(
            "success",
            data=np.asarray(
                [
                    entry["result"].success
                    for entry in self.results
                ],
                dtype=bool,
            ),
        )

        for parameter_name in self.parameter_names:
            values = [
                entry["parameters"].get(
                    parameter_name,
                    np.nan,
                )
                for entry in self.results
            ]

            if all(
                cls_value_is_numeric(value)
                for value in values
            ):
                run_table_group.create_dataset(
                    parameter_name,
                    data=np.asarray(
                        values,
                        dtype=float,
                    ),
                )

            else:
                string_dtype = h5py.string_dtype(
                    encoding="utf-8"
                )

                run_table_group.create_dataset(
                    parameter_name,
                    data=np.asarray(
                        [
                            str(value)
                            for value in values
                        ],
                        dtype=object,
                    ),
                    dtype=string_dtype,
                )

    @staticmethod
    def _normalize_parameter_value(
        value,
    ):
        """Normalize values before storing them as HDF5 attributes."""

        if isinstance(
            value,
            np.generic,
        ):
            return value.item()

        if isinstance(
            value,
            (
                str,
                bool,
                int,
                float,
            ),
        ):
            return value

        raise TypeError(
            "Parameter values must currently be strings, "
            "booleans, integers, floats, or NumPy scalars. "
            f"Received {type(value).__name__}."
        )

    @staticmethod
    def _write_attributes(
        group,
        values,
    ):
        """Write a dictionary as HDF5 attributes."""

        for name, value in values.items():
            normalized_value = (
                SimulationSeries._normalize_parameter_value(
                    value
                )
            )

            group.attrs[str(name)] = normalized_value

    @staticmethod
    def _read_attributes(
        group,
    ):
        """Read all attributes of an HDF5 group."""

        values = {}

        for name, value in group.attrs.items():
            if isinstance(value, bytes):
                value = value.decode("utf-8")

            elif isinstance(value, np.generic):
                value = value.item()

            values[str(name)] = value

        return values

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"n_results={len(self)}, "
            f"parameters={self.parameter_names})"
        )


def cls_value_is_numeric(
    value,
):
    """Return whether a value can be stored in a numeric run table."""

    return isinstance(
        value,
        (
            bool,
            int,
            float,
            np.integer,
            np.floating,
        ),
    )