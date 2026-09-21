#!/usr/bin/env python3
"""
Measure a bond distance, angle, or dihedral-angle distribution with MDTraj.

Examples
--------
Bond:
    python measure.py \
        --prmtop system.prmtop \
        --dcd trajectory.dcd \
        --bond 10 25

Angle:
    python measure.py \
        --prmtop system.prmtop \
        --dcd trajectory.dcd \
        --angle 10 25 31

Dihedral:
    python measure.py \
        --prmtop system.prmtop \
        --dcd trajectory.dcd \
        --dihedral 10 25 31 44

MDTraj uses zero-based atom indices. The first atom is atom 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np


def positive_integer(value: str) -> int:
    """Argparse validator for positive integers."""
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected an integer, received {value!r}."
        ) from exc

    if result <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")

    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a bond distance, angle, or dihedral over an MDTraj "
            "trajectory and plot its distribution."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--prmtop", required=True, type=Path,
        help="AMBER topology file.",
    )
    parser.add_argument("--dcd", required=True, type=Path,
        help="DCD trajectory file.",
    )
    
    measurement_group = parser.add_mutually_exclusive_group(required=True)

    measurement_group.add_argument("--bond", nargs=2, type=int, metavar=("ATOM_1", "ATOM_2"),
        help="Measure a bond distance using two zero-based atom indices.",
    )
    measurement_group.add_argument("--angle", nargs=3, type=int, metavar=("ATOM_1", "ATOM_2", "ATOM_3"),
        help="Measure an angle using three zero-based atom indices.",
    )
    measurement_group.add_argument("--dihedral", nargs=4, type=int, metavar=("ATOM_1", "ATOM_2", "ATOM_3", "ATOM_4"),
        help="Measure a dihedral using four zero-based atom indices.",
    )
    parser.add_argument("--bins", type=positive_integer, default=72,
        help="Number of histogram bins.",
    )
    parser.add_argument("--stride", type=positive_integer, default=1,
        help="Read every Nth trajectory frame.",
    )
    parser.add_argument("--output", type=Path, default=None,
        help="Output plot filename. A descriptive filename is used by default.",
    )
    parser.add_argument("--csv", type=Path, default=None,
        help="Output CSV filename. A descriptive filename is used by default.",
    )
    parser.add_argument("--no-periodic", action="store_true",
        help="Disable minimum-image periodic-boundary handling.",
    )
    parser.add_argument("--show", action="store_true",
        help="Display the plot interactively after saving it.",
    )

    return parser.parse_args()


def validate_input_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def validate_atom_indices(
    atom_indices: list[int],
    number_of_atoms: int,
) -> None:
    for index in atom_indices:
        if index < 0:
            raise ValueError(
                f"Atom index {index} is negative. Atom indices must be zero-based "
                "non-negative integers."
            )

        if index >= number_of_atoms:
            raise IndexError(
                f"Atom index {index} is outside the topology. "
                f"The topology contains {number_of_atoms} atoms, with valid "
                f"indices from 0 to {number_of_atoms - 1}."
            )


def describe_atoms(
    topology: md.Topology,
    atom_indices: list[int],
) -> str:
    """Create a readable description of the selected atoms."""
    descriptions = []

    for index in atom_indices:
        atom = topology.atom(index)
        descriptions.append(f"{index}:{atom}")

    return " - ".join(descriptions)


def calculate_measurement(
    trajectory: md.Trajectory,
    measurement_type: str,
    atom_indices: list[int],
    periodic: bool,
) -> tuple[np.ndarray, str, str]:
    """
    Calculate the requested measurement.

    Returns
    -------
    values
        One measurement per trajectory frame.
    unit
        Display unit.
    column_name
        Name used in the output CSV file.
    """
    index_array = np.asarray([atom_indices], dtype=np.int32)

    if measurement_type == "bond":
        # MDTraj returns distances in nanometers.
        values_nm = md.compute_distances(
            trajectory,
            index_array,
            periodic=periodic,
        )[:, 0]

        # Convert nanometers to angstroms.
        values = values_nm * 10.0
        return values, "Å", "distance_angstrom"

    if measurement_type == "angle":
        # MDTraj returns angles in radians.
        values_radians = md.compute_angles(
            trajectory,
            index_array,
            periodic=periodic,
        )[:, 0]

        values = np.degrees(values_radians)
        return values, "degrees", "angle_degrees"

    if measurement_type == "dihedral":
        # MDTraj returns dihedral angles in radians.
        values_radians = md.compute_dihedrals(
            trajectory,
            index_array,
            periodic=periodic,
        )[:, 0]

        values = np.degrees(values_radians)
        return values, "degrees", "dihedral_degrees"

    raise ValueError(f"Unsupported measurement type: {measurement_type}")


def determine_measurement(
    arguments: argparse.Namespace,
) -> tuple[str, list[int]]:
    if arguments.bond is not None:
        return "bond", arguments.bond

    if arguments.angle is not None:
        return "angle", arguments.angle

    if arguments.dihedral is not None:
        return "dihedral", arguments.dihedral

    raise RuntimeError("No measurement was selected.")


def print_statistics(
    values: np.ndarray,
    measurement_type: str,
    atom_indices: list[int],
    atom_description: str,
    unit: str,
) -> None:
    print()
    print(f"Measurement : {measurement_type}")
    print(f"Atom indices: {' '.join(map(str, atom_indices))}")
    print(f"Atoms       : {atom_description}")
    print(f"Frames      : {values.size}")
    print(f"Minimum     : {np.min(values):.6f} {unit}")
    print(f"Maximum     : {np.max(values):.6f} {unit}")
    print(f"Mean        : {np.mean(values):.6f} {unit}")
    print(f"Median      : {np.median(values):.6f} {unit}")
    print(f"Std. dev.   : {np.std(values, ddof=0):.6f} {unit}")
    print()


def save_csv(
    output_path: Path,
    values: np.ndarray,
    trajectory: md.Trajectory,
    column_name: str,
    stride: int,
) -> None:
    frame_numbers = np.arange(values.size, dtype=int) * stride

    if trajectory.time is not None and len(trajectory.time) == len(values):
        data = np.column_stack((frame_numbers, trajectory.time, values))
        header = f"frame,time_ps,{column_name}"
        format_string = ["%d", "%.8f", "%.8f"]
    else:
        data = np.column_stack((frame_numbers, values))
        header = f"frame,{column_name}"
        format_string = ["%d", "%.8f"]

    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt=format_string,
    )


def plot_distribution(
    values: np.ndarray,
    measurement_type: str,
    atom_indices: list[int],
    unit: str,
    number_of_bins: int,
    output_path: Path,
    show_plot: bool,
) -> None:
    atom_text = "-".join(map(str, atom_indices))

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.hist(
        values,
        bins=number_of_bins,
        density=True,
        edgecolor="black",
        alpha=0.75,
    )

    mean_value = np.mean(values)
    median_value = np.median(values)

    axis.axvline(
        mean_value,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_value:.3f} {unit}",
    )
    axis.axvline(
        median_value,
        linestyle=":",
        linewidth=1.5,
        label=f"Median: {median_value:.3f} {unit}",
    )

    if measurement_type == "bond":
        axis.set_xlabel(f"Distance ({unit})")
    else:
        axis.set_xlabel(f"{measurement_type.capitalize()} ({unit})")

    axis.set_ylabel("Probability density")
    axis.set_title(
        f"{measurement_type.capitalize()} distribution\n"
        f"Atom indices: {atom_text}"
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()

    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot written to: {output_path}")

    if show_plot:
        plt.show()

    plt.close(figure)


def main() -> int:
    arguments = parse_arguments()

    try:
        validate_input_file(arguments.prmtop, "Topology file")
        validate_input_file(arguments.dcd, "Trajectory file")

        measurement_type, atom_indices = determine_measurement(arguments)

        print(f"Loading topology:   {arguments.prmtop}")
        print(f"Loading trajectory: {arguments.dcd}")

        trajectory = md.load(
            str(arguments.dcd),
            top=str(arguments.prmtop),
            stride=arguments.stride,
        )

        if trajectory.n_frames == 0:
            raise ValueError("The trajectory contains no readable frames.")

        validate_atom_indices(atom_indices, trajectory.n_atoms)

        periodic = not arguments.no_periodic

        values, unit, column_name = calculate_measurement(
            trajectory=trajectory,
            measurement_type=measurement_type,
            atom_indices=atom_indices,
            periodic=periodic,
        )

        atom_description = describe_atoms(
            trajectory.topology,
            atom_indices,
        )

        index_text = "_".join(map(str, atom_indices))
        default_stem = f"{measurement_type}_{index_text}"

        plot_path = (
            arguments.output
            if arguments.output is not None
            else Path(f"{default_stem}_distribution.png")
        )
        csv_path = (
            arguments.csv
            if arguments.csv is not None
            else Path(f"{default_stem}_values.csv")
        )

        print_statistics(
            values=values,
            measurement_type=measurement_type,
            atom_indices=atom_indices,
            atom_description=atom_description,
            unit=unit,
        )

        save_csv(
            output_path=csv_path,
            values=values,
            trajectory=trajectory,
            column_name=column_name,
            stride=arguments.stride,
        )
        print(f"Values written to: {csv_path}")

        plot_distribution(
            values=values,
            measurement_type=measurement_type,
            atom_indices=atom_indices,
            unit=unit,
            number_of_bins=arguments.bins,
            output_path=plot_path,
            show_plot=arguments.show,
        )

    except (FileNotFoundError, IndexError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            f"Unexpected error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
