"""trajectory_mapper.py: Maps an all-atom (AA) topology to a coarse grained (CG) topology"""
__all__ = ['map_trajectory']

import numpy as np
import networkx as nx

import mdtraj as md

from cgtools.mapping import TopologyMapper


def map_trajectory(topology_mapper: TopologyMapper, aa_traj: md.Trajectory) -> md.Trajectory:
    # initialize array of coordinates
    cg_xyz = np.empty((aa_traj.n_frames, topology_mapper.n_cg_beads, 3))

    # find the maximum number of atoms that any one bead is mapped to
    _max_atoms_per_bead = max([len(bead.aa_indices) for bead in topology_mapper.cg_beads])

    # determine atomistic indices and masses per bead
    n_atoms_per_bead = np.empty(topology_mapper.n_cg_beads, dtype=int)
    aa_indices_per_bead = np.empty((topology_mapper.n_cg_beads, _max_atoms_per_bead), dtype=int)
    aa_masses_per_bead = np.empty((topology_mapper.n_cg_beads, _max_atoms_per_bead), dtype=int)
    for i, bead in enumerate(topology_mapper.cg_beads):
        n_atoms = len(bead.aa_indices)
        n_atoms_per_bead[i] = n_atoms
        aa_indices_per_bead[i, :n_atoms] = bead.aa_indices
        aa_masses_per_bead[i, :n_atoms] = bead.aa_masses

    # call Numba-accelerated helper function to map coordinates
    cg_xyz = _map_coordinates(aa_traj.xyz, cg_xyz, n_atoms_per_bead, aa_indices_per_bead, aa_masses_per_bead)

    # return md.Trajectory
    return md.Trajectory(cg_xyz, topology_mapper.cg_topology,
                         unitcell_lengths=aa_traj.unitcell_lengths,
                         unitcell_angles=aa_traj.unitcell_angles)


def _map_coordinates(aa_xyz, cg_xyz, n_atoms_per_bead, aa_indices_per_bead, aa_masses_per_bead):
    for i, (n_atoms, aa_indices, aa_masses) in enumerate(zip(n_atoms_per_bead, aa_indices_per_bead, aa_masses_per_bead)):
        aa_indices = aa_indices[:n_atoms]
        aa_masses = aa_masses[:n_atoms]
        cg_xyz[:, i, :] = np.average(aa_xyz[:, aa_indices, :], axis=1, weights=aa_masses)
    return cg_xyz
