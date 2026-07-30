"""Central unit registry for VirtualReactor."""

import pint
import numbers
import warnings
import numpy as np

ureg = pint.UnitRegistry(
    autoconvert_offset_to_baseunit=True,
)

Q_ = ureg.Quantity


def to_magnitude(
    value,
    unit,
    name="value",
):
    """Return a float in the requested SI unit."""

    if isinstance(value, numbers.Real):
        warnings.warn(
            f"{name} has no unit. "
            f"Interpreting {value} as {unit}.",
            stacklevel=2,
        )
        return float(value)

    if isinstance(value, pint.Quantity):
        try:
            return float(
                value.to(unit).magnitude
            )

        except pint.DimensionalityError as error:
            raise ValueError(
                f"{name} must have units compatible with {unit}."
            ) from error

    raise TypeError(
        f"{name} must be a real number or a Pint quantity."
    )


def to_magnitude_array(
    values,
    unit,
    name="values",
):
    """Return a NumPy array in the requested SI unit."""

    if isinstance(values, pint.Quantity):
        try:
            return np.asarray(
                values.to(unit).magnitude,
                dtype=float,
            )

        except pint.DimensionalityError as error:
            raise ValueError(
                f"{name} must have units compatible with {unit}."
            ) from error

    try:
        array = np.asarray(
            values,
            dtype=float,
        )

    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must be an array of real numbers "
            "or a Pint quantity."
        ) from error

    warnings.warn(
        f"{name} has no unit. "
        f"Interpreting all values as {unit}.",
        stacklevel=2,
    )

    return array