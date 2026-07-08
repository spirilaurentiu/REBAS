"""Reusable trajectory observable extractors."""

from __future__ import annotations

from typing import Sequence

import mdtraj as md
import numpy as np


class DihedralGeometryError(ValueError):
    """Raised when a dihedral angle is undefined for degenerate geometry."""


def angle(v1, v2):
    """Return the angle between vectors in radians."""
    norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_prod == 0.0:
        return np.nan
    cos_theta = np.dot(v1, v2) / norm_prod
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


# Keep these aliases so dihedral_mine can remain exactly as written.
numpy = np
cross = np.cross
pi = np.pi


class Observables:
    """Collection of static extractors for trajectory observables."""

    ALA1_BASINS = {
        "C5": {
            "phi_min": np.deg2rad(-180),
            "phi_max": np.deg2rad(-95),
            "psi_min": np.deg2rad(105),
            "psi_max": np.deg2rad(180),
        },
        "PPII": {
            "phi_min": np.deg2rad(-96),
            "phi_max": np.deg2rad(-45),
            "psi_min": np.deg2rad(105),
            "psi_max": np.deg2rad(180),
        },
        "C7_eq": {
            "phi_min": np.deg2rad(-96),
            "phi_max": np.deg2rad(-45),
            "psi_min": np.deg2rad(-25),
            "psi_max": np.deg2rad(104),
        },
        "alpha_eq": {
            "phi_min": np.deg2rad(35),
            "phi_max": np.deg2rad(85),
            "psi_min": np.deg2rad(-180),
            "psi_max": np.deg2rad(25),
        },
    }

    TRPCH_BASINS = {
        "basin1": {
            "psi_min": -1.5,
            "psi_max": 0.5,
            "ee_dist_min": 0.0,
            "ee_dist_max": 1.27,
        },
        "basin2": {
            "psi_min": -1.5,
            "psi_max": 0.5,
            "ee_dist_min": 1.27,
            "ee_dist_max": 5.0,
        },
        "basin3": {
            "psi_min": 2.0,
            "psi_max": 3.0,
            "ee_dist_min": 0.0,
            "ee_dist_max": 1.27,
        },
        "basin4": {
            "psi_min": 2.0,
            "psi_max": 3.0,
            "ee_dist_min": 1.27,
            "ee_dist_max": 5.0,
        },
    }

    @staticmethod
    def distances(
        traj,
        pairs: Sequence[Sequence[int]] | None = None,
    ):
        """Compute pairwise distances and return shape (n_pairs, n_frames)."""
        if pairs is None:
            pairs = [[8, 298], [100, 200]]

        result = md.compute_distances(traj, pairs)
        return result.T

    @staticmethod
    def dihedral_a1_a2_a3_a4(traj, a1=4, a2=6, a3=8, a4=14):
        """Calculate dihedral angle defined by four atoms across all frames."""
        result = md.compute_dihedrals(traj, [[a1, a2, a3, a4]])
        return result.T

    @staticmethod
    def dihedral_a1_a2_a3_a4_explicit(traj, a1=4, a2=6, a3=8, a4=14, degrees=False):
        """Calculate a dihedral explicitly from coordinates without md.compute_dihedrals."""
        xyz = traj.xyz

        p0 = xyz[:, a1, :]
        p1 = xyz[:, a2, :]
        p2 = xyz[:, a3, :]
        p3 = xyz[:, a4, :]

        b0 = p1 - p0
        b1 = p2 - p1
        b2 = p3 - p2

        b1_norm = np.linalg.norm(b1, axis=1, keepdims=True)
        b1_norm = np.where(b1_norm == 0.0, np.nan, b1_norm)
        b1_unit = b1 / b1_norm

        v = b0 - np.sum(b0 * b1_unit, axis=1, keepdims=True) * b1_unit
        w = b2 - np.sum(b2 * b1_unit, axis=1, keepdims=True) * b1_unit

        x = np.sum(v * w, axis=1)
        y = np.sum(np.cross(b1_unit, v) * w, axis=1)

        angles = np.arctan2(y, x)
        if degrees:
            angles = np.rad2deg(angles)

        return angles[np.newaxis, :]

    @staticmethod
    def dihedral_mine(v1, v2, v3, v4):
        """
        Returns a float value for the dihedral angle between
        the four vectors. They define the bond for which the
        torsion is calculated (~) as:
        V1 - V2 ~ V3 - V4
        The vectors vec1 .. vec4 can be array objects, lists or tuples of length
        three containing floats.
        For Scientific.geometry.Vector objects the behavior is different
        on Windows and Linux. Therefore, the latter is not a featured input type
        even though it may work.

        If the dihedral angle cant be calculated (because vectors are collinear),
        the function raises a DihedralGeometryError
        """
        all_vecs = [v1,v2,v3,v4]

        # rule out that two of the atoms are identical
        # except the first and last, which may be.
        for i in range(len(all_vecs)-1):
            for j in range(i+1,len(all_vecs)):
                if i>0 or j<3: # exclude the (1,4) pair
                    equals = all_vecs[i]==all_vecs[j]
                    if equals.all():
                        raise DihedralGeometryError(\
                            "Vectors #%i and #%i may not be identical!"%(i,j))

        # calculate vectors representing bonds
        v12 = v2-v1
        v23 = v3-v2
        v34 = v4-v3

        # calculate vectors perpendicular to the bonds
        normal1 = cross(v12,v23)
        normal2 = cross(v23,v34)

        # check for linearity
        if numpy.linalg.norm(normal1) == 0 or numpy.linalg.norm(normal2)== 0:
            raise DihedralGeometryError(\
                "Vectors are in one line; cannot calculate normals!")

        # normalize them to length 1.0
        normal1 = normal1/numpy.linalg.norm(normal1)
        normal2 = normal2/numpy.linalg.norm(normal2)

        # calculate torsion and convert to degrees
        torsion = angle(normal1,normal2) * 180.0/pi

        # take into account the determinant
        # (the determinant is a scalar value distinguishing
        # between clockwise and counter-clockwise torsion.
        if np.dot(normal1,v34) >= 0:
            return torsion
        else:
            torsion = 360-torsion
            if torsion == 360:
                torsion = 0.0
            return torsion

    @staticmethod
    def dihedral_a1_a2_a3_a4_using_mine(traj, a1=4, a2=6, a3=8, a4=14, degrees=False):
        """Calculate trajectory dihedral using dihedral_mine frame-by-frame."""
        torsions_deg = []
        for frame_xyz in traj.xyz:
            v1 = frame_xyz[a1]
            v2 = frame_xyz[a2]
            v3 = frame_xyz[a3]
            v4 = frame_xyz[a4]
            try:
                t_deg = Observables.dihedral_mine(v1, v2, v3, v4)
                if t_deg > 180.0:
                    t_deg -= 360.0
            except DihedralGeometryError:
                t_deg = np.nan
            torsions_deg.append(t_deg)

        torsions_deg = np.asarray(torsions_deg, dtype=float)
        if degrees:
            return torsions_deg[np.newaxis, :]
        return np.deg2rad(torsions_deg)[np.newaxis, :]
    
    @staticmethod
    def dihedral_adj_a1_a2_a3_a4_a5(traj, a1=4, a2=6, a3=8, a4=14, a5=16):
        dihedrals = md.compute_dihedrals(traj, [[a1, a2, a3, a4], [a2, a3, a4, a5]])
        return 1 - np.cos(dihedrals[:, 0] - dihedrals[:, 1])

    @staticmethod
    def quaternion_multiply(q1, q2):
        """Vectorized quaternion multiplication."""
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        res = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        ).T
        return res

    @staticmethod
    def dihedral_quat_a1_a2_a3_a4_a5(traj, a1=4, a2=6, a3=8, a4=14, a5=16):
        """Compute phi/psi quaternions and return their frame-wise inner products."""
        phi = md.compute_dihedrals(traj, [[a1, a2, a3, a4]])
        psi = md.compute_dihedrals(traj, [[a2, a3, a4, a5]])
        q_phi = np.stack([np.cos(phi / 2), np.sin(phi / 2), np.zeros_like(phi), np.zeros_like(phi)], axis=-1)
        q_psi = np.stack([np.cos(psi / 2), np.zeros_like(psi), np.sin(psi / 2), np.zeros_like(psi)], axis=-1)
        inner_q = np.dot(q_phi, q_psi)
        return inner_q

    @staticmethod
    def dihedral_phi_psi(traj, phi_psi="psi", resid=0):
        """Calculate phi or psi for a given residue index."""
        mdtraj_result = None
        if phi_psi == "phi":
            mdtraj_result = md.compute_phi(traj, periodic=False)
        elif phi_psi == "psi":
            mdtraj_result = md.compute_psi(traj, periodic=False)
        else:
            raise ValueError(f"Invalid phi_psi value: {phi_psi}. Must be 'phi' or 'psi'.")

        torsions_all = mdtraj_result[1]

        if resid >= torsions_all.shape[1] or resid < 0:
            raise ValueError(f"Invalid resid {resid}. Must be between 0 and {torsions_all.shape[1] - 1}")

        torsions = torsions_all[:, resid].ravel()

        from batana import BATStats

        batStats = BATStats()
        torsions_mean = batStats.dihedralMean(torsions)
        torsions_std = batStats.dihedralStd(torsions)
        print(
            f"\nMean {phi_psi} for residue {resid}: {torsions_mean:.3f} radians "
            f"({np.rad2deg(torsions_mean):.1f} degrees)"
        )
        print(
            f"Std {phi_psi} for residue {resid}: {torsions_std:.3f} radians "
            f"({np.rad2deg(torsions_std):.1f} degrees)"
        )

        return torsions

    @staticmethod
    def ala_PMF_indicator(traj, a1=4, a2=6, a3=8, a4=14, a5=16):
        phi = md.compute_dihedrals(traj, [[a1, a2, a3, a4]]).ravel()
        psi = md.compute_dihedrals(traj, [[a2, a3, a4, a5]]).ravel()

        is_c5 = (
            (phi >= np.deg2rad(-180))
            & (phi <= np.deg2rad(-95))
            & (psi >= np.deg2rad(105))
            & (psi <= np.deg2rad(180))
        )
        is_ppii = (
            (phi >= np.deg2rad(-96))
            & (phi <= np.deg2rad(-45))
            & (psi >= np.deg2rad(105))
            & (psi <= np.deg2rad(180))
        )
        is_c7eq = (
            (phi >= np.deg2rad(-96))
            & (phi <= np.deg2rad(-45))
            & (psi >= np.deg2rad(-25))
            & (psi <= np.deg2rad(104))
        )
        is_alpha = (
            (phi >= np.deg2rad(35))
            & (phi <= np.deg2rad(85))
            & (psi >= np.deg2rad(-180))
            & (psi <= np.deg2rad(25))
        )

        state_zero_mask = is_c5 | is_ppii | is_c7eq
        conditions = [state_zero_mask, is_alpha]
        choices = [0.0, 1.0]
        states = np.select(conditions, choices, default=0.5)
        return states

    @staticmethod
    def dist_atom1_atom2(traj, a1=8, a2=298):
        """Distance timeseries between two atoms as shape (n_frames,)."""
        return md.compute_distances(traj, [[a1, a2]]).ravel()

    @staticmethod
    def trpch_PMF_indicator(traj, phi_psi="psi", resid=0):
        psi = Observables.dihedral_phi_psi(traj, phi_psi=phi_psi, resid=resid)
        ee_dist = Observables.dist_atom1_atom2(traj, a1=8, a2=298)

        psi_range1 = (psi >= -1.5) & (psi <= 0.5)
        psi_range2 = (psi >= 2.0) & (psi <= 3.0)
        dist_short = ee_dist <= 1.27
        dist_long = (ee_dist > 1.27) & (ee_dist <= 5.0)

        is_basin1 = psi_range1 & dist_short
        is_basin2 = psi_range1 & dist_long
        is_basin3 = psi_range2 & dist_short
        is_basin4 = psi_range2 & dist_long

        conditions = [is_basin1, is_basin2, is_basin3, is_basin4]
        choices = [0.0, 0.25, 0.5, 0.75]
        states = np.select(conditions, choices, default=1.0)
        return states
