"""mapping.py: Maps an all-atom (AA) topology to a coarse grained (CG) topology"""
__all__ = ['CenterOfMass', 'Centroid', 'TopologyMapper']

import warnings
from collections import defaultdict
from enum import Enum, auto

import numpy as np

import networkx as nx
import networkx.algorithms.isomorphism as iso

import mdtraj as md

try:
    import sim
except ModuleNotFoundError:
    sim = None


class CenterType(Enum):
    """Enumerate mapping methods for AA atoms to a CG bead."""

    CenterOfMass = auto()
    """A center-of-mass mapping is used."""

    Centroid = auto()
    """Masses of atoms are ignored and when mapping AA positions to CG."""


# set enums equal to exported variables
CenterOfMass = CenterType.CenterOfMass
Centroid = CenterType.Centroid


def _get_unique_residue_names_from_topology(topology: md.Topology):
    """Utility method used to get all unique residue names from a mdtraj.Topology.

    Parameters
    ----------
    topology : mdtraj.Topology
        Stores the topological information about an atomistic system.
    """
    residue_names = []
    for residue in topology.residues:
        residue_names.append(residue.name)
    return set(residue_names)


class TopologyMapper(object):
    """TopologyMapper manages mapping from an AA topology to a CG version.

    Attributes
    ----------
    center_type : CenterType
        Specifies the method of mapping atomistic positions to a CG position.
    unmapped_residues : set of str
        Set of all residues missing a specified mapping.
    """

    def __init__(self, topology, center_type=CenterOfMass):
        """Initializes a TopologyMapper object.

        Parameters
        ----------
        topology : mdtraj.Topology
            Stores the topological information about an atomistic system.
        center_type : CenterType, {CenterOfMass, Centroid}
            Specifies the method of mapping atomistic positions to a CG position.
        """
        self._topology = topology
        self._residue_maps = {}
        self._unique_residue_names = _get_unique_residue_names_from_topology(topology)
        self._center_type = center_type

        # these will be set after create_map() is called
        self._cg_topology_graph = None
        self._unique_cg_bead_names = None
        self._aa_to_cg_bead_map = None

    @classmethod
    def from_file(cls, filename):
        """Initializes a TopologyMapper object from a file.

        Parameters
        ----------
        filename : str
            Path for a file that contains topology information for an atomistic
            system.
        """
        # load mdtraj.Trajectory object from file
        t = md.load(filename)

        # create TopologyMapper object using the topology
        return TopologyMapper(t.topology)

    @property
    def center_type(self):
        """Specifies the method of mapping atomistic positions to a CG position."""
        return self._center_type

    @center_type.setter
    def center_type(self, value):
        """Setter for `center_type` attribute."""
        if not isinstance(value, CenterType):
            raise TypeError("`center_type` must be `CenterOfMass` or `Centroid`")
        self._center_type = value

    def add_residue_map(
            self, residue_name, cg_bead_names,
            n_heavy_atoms_per_cg_bead=None, heavy_atom_masses=None
    ):
        """Adds a CG mapping for an atomistic residue.

        Parameters
        ----------
        residue_name : str
            Name of the atomistic residue.
        cg_bead_names : str or list of str
            List of CG bead names that this residue maps to. If the whole
            residue is mapped to a single CG bead, only the bead name itself
            neads to be specified.
        n_heavy_atoms_per_cg_bead : list of int, optional
            Number of heavy (e.g., non-H) atoms per CG bead. The length of
            this argument should be the same as the length of `cg_bead_names`
            sum of this list should be equal to the number of heavy atoms in
            the residue. For example, if there are 9 heavy atoms in the residue
            and [5, 4] is passed as the value of this argument, then the first
            five heavy atoms are mapped to the first CG bead in `cg_bead_names`
            and the next 4 heavy atoms are mapped to the second CG bead in
            `cg_bead_names`. This argument only needs to be specified if the
            residue is being mapped to more than one bead
        heavy_atom_masses : list of float, optional
            List of masses of each heavy atom in the residue. If not specified,
            the masses of the atoms will be inferred from their elements. This
            option should only be used for topologies with atoms that have
            non-standard elements (i.e., united-atom systems). Not needed if
            using a centroid mapping instead of a center-of-mass mapping.
        """
        # convert arguments to lists if needed
        if isinstance(cg_bead_names, str):
            cg_bead_names = [cg_bead_names]

        # residue should exist in atomistic topology
        if residue_name not in self._unique_residue_names:
            raise ValueError(
                f"no residue with name '{residue_name}' in the atomistic topology"
            )

        # check that the length of arguments are consistent
        if len(cg_bead_names) > 1 and len(cg_bead_names) != len(n_heavy_atoms_per_cg_bead):
            raise ValueError(
                f"if mapping residue to more than one bead, `cg_bead_names` and "
                f"`n_heavy_atoms_per_cg_bead` should be the same length"
            )

        # check that the number of heavy atom masses passed is consistent
        if heavy_atom_masses is not None and len(heavy_atom_masses) != np.sum(n_heavy_atoms_per_cg_bead):
            raise ValueError(
                f"the number of heavy atom masses provided does not match the "
                f"total number of heavy atoms in `n_heavy_atoms_per_cg_bead`"
            )

        # store residue map
        self._residue_maps[residue_name] = (cg_bead_names, n_heavy_atoms_per_cg_bead, heavy_atom_masses)

    @property
    def unmapped_residues(self):
        """All residues missing a specified mapping."""
        return set(self._unique_residue_names) - set(self._residue_maps.keys())

    def create_map(self):
        """Creates the atomistic-to-CG mapping. Only """
        # initialize graph of CG topology
        cg_top_graph = nx.Graph()

        # initialize map for atomistic indices to CG beads
        aa_to_cg_map = {}

        # initialize list of unique cg bead names
        unique_cg_bead_names = []

        # find where all hydrogen atoms are
        h_atom_map = defaultdict(list)  # maps heavy atom indices to a list of attached hydrogen indices
        h_atom_indices = self._topology.select('element == H')  # indices of hydrogen atoms
        for bond in self._topology.bonds:
            atom0, atom1 = bond
            if atom0.index in h_atom_indices:
                h_atom_map[atom1.index].append(atom0.index)
            elif atom1.index in h_atom_indices:
                h_atom_map[atom0.index].append(atom1.index)

        # iterate through residues to add beads to the CG topology
        cg_bead_index = 0
        for residue in self._topology.residues:
            # get corresponding data from the residue map
            try:
                _residue_map = self._residue_maps[residue.name]
                cg_bead_names_in_res, n_heavy_atoms_per_cg_bead, heavy_atom_masses_in_res = _residue_map
            except KeyError:
                raise ValueError(f"no mapping found for residue '{residue.name}'")

            # get indices of heavy atoms in residue
            _selection_string = f"resid {residue.index} and element != H"
            heavy_atom_indices_in_res = self._topology.select(_selection_string)

            # check that the number of heavy atoms provided in the mapping is consistent
            if np.sum(n_heavy_atoms_per_cg_bead) != len(heavy_atom_indices_in_res):
                raise ValueError(
                    f"sum of `n_heavy_atoms_per_cg_bead` does not match the "
                    f"number of heavy atoms in residue {residue.index} ({residue.name})"
                )

            # if providing masses, check that the number of heavy atom masses
            # provided matches the total number of heavy atoms in the residue
            # NOTE: only check if mapping type is center-of-mass
            if self.center_type is CenterOfMass and heavy_atom_masses_in_res is not None:
                if len(heavy_atom_masses_in_res) != len(heavy_atom_indices_in_res):
                    raise ValueError(
                        f"number of heavy atom masses provided does not match "
                        f"the number of heavy atoms in residue {residue.index} ({residue.name})"
                    )

            # all masses set to 1 if centroid mapping
            if self.center_type is Centroid:
                heavy_atom_masses_in_res = np.ones_like(heavy_atom_indices_in_res)

            # if COM mapping and masses not provided, find using elements of the atoms
            if self.center_type is CenterOfMass and heavy_atom_masses_in_res is None:
                heavy_atom_masses_in_res = []
                for i in heavy_atom_masses_in_res:
                    mass_i = self._topology.atom(i).element.mass
                    heavy_atom_masses_in_res.append(mass_i)
                heavy_atom_masses_in_res = np.array(heavy_atom_masses_in_res)

            # determine split points and split heavy atom indices and masses between beads
            split_points = np.cumsum(n_heavy_atoms_per_cg_bead)[:-1]
            heavy_atom_indices_split = np.split(heavy_atom_indices_in_res, split_points)
            heavy_atom_masses_split = np.split(heavy_atom_masses_in_res, split_points)

            # iterate through CG beads in this residue
            for i_bead_in_res in range(len(cg_bead_names_in_res)):
                # get bead name lists of heavy atom indices and masses in bead
                cg_bead_name = cg_bead_names_in_res[i_bead_in_res]
                heavy_atom_indices_in_bead = heavy_atom_indices_split[i_bead_in_res]
                heavy_atom_masses_in_bead = heavy_atom_masses_split[i_bead_in_res]

                # add bead name if it hasn't been recorded already
                if cg_bead_name not in unique_cg_bead_names:
                    unique_cg_bead_names.append(cg_bead_name)

                # add indices and masses of hydrogen atoms
                atom_indices_in_bead = []
                atom_masses_in_bead = []
                for i_atom, mass_i in zip(heavy_atom_indices_in_bead, heavy_atom_masses_in_bead):
                    atom_indices_in_bead.append(i_atom)
                    atom_masses_in_bead.append(mass_i)
                    for i_hydrogen in h_atom_map[i_atom]:
                        atom_indices_in_bead.append(i_hydrogen)
                        if self.center_type is CenterOfMass:
                            mass_hydrogen = self._topology.atom(i_hydrogen).element.mass
                            atom_masses_in_bead.append(mass_hydrogen)
                        else:
                            atom_masses_in_bead.append(1.0)
                atom_indices_in_bead = np.array(atom_indices_in_bead, dtype=int)
                atom_masses_in_bead = np.array(atom_masses_in_bead, dtype=float)

                # add bead to CG topology
                cg_bead = CGBead(cg_bead_name, cg_bead_index, atom_indices_in_bead, atom_masses_in_bead)
                cg_top_graph.add_node(cg_bead, pair=cg_bead_name)

                # map each atom to its respective CG bead
                for i_atom in atom_indices_in_bead:
                    aa_to_cg_map[i_atom] = cg_bead

                cg_bead_index += 1

        # add bonds to CG topology
        # first raise warning if there are no bonds in the atomistic topology
        if self._topology.n_bonds == 0:
            warnings.warn(
                "no bonds found in the atomistic topology; unable to determine"
                "bonds in the CG topology"
            )

        for bond in self._topology.bonds:
            cg_bead_0 = aa_to_cg_map[bond[0].index]
            cg_bead_1 = aa_to_cg_map[bond[1].index]
            if cg_bead_0 is not cg_bead_1:
                cg_bead_name_0 = cg_bead_0.name
                cg_bead_name_1 = cg_bead_1.name
                name_pair = tuple(sorted([cg_bead_name_0, cg_bead_name_1]))
                cg_top_graph.add_edge(cg_bead_0, cg_bead_1, name_pair=name_pair)

        self._cg_topology_graph = cg_top_graph
        self._aa_to_cg_bead_map = aa_to_cg_map
        self._unique_cg_bead_names = unique_cg_bead_names

    @property
    def cg_beads(self):
        if self._cg_topology_graph is None:
            raise AttributeError("must call `create_map` method before accessing this attribute")
        return iter(self._cg_topology_graph.nodes(data=False))

    @property
    def cg_bonds(self):
        if self._cg_topology_graph is None:
            raise AttributeError("must call `create_map` method before accessing this attribute")
        return iter(self._cg_topology_graph.edges(data=False))

    def to_sim(self):
        """Create a representation of the CG topology in the format of sim, a
        relative-entropy coarse graining package developed by the Shell Group
        at UCSB.
        """
        if sim is None:
            raise ModuleNotFoundError("no module named 'sim'")

        # create sim AtomTypes
        sim_AtomTypes = {}
        for cg_bead_name in self._unique_cg_bead_names:
            sim_AtomTypes[cg_bead_name] = sim.chem.AtomType(cg_bead_name)

        # create the position map
        sim_PosMap = sim.atommap.PosMap()
        for bead in sorted(self.cg_beads, key=lambda bead: bead.index):
            sim_AtomMap = sim.atommap.AtomMap(list(bead.aa_indices), bead.index, Mass1=bead.aa_masses, Atom2Name=bead.name)
            sim_PosMap.append(sim_AtomMap)

        # separate the CG topology into molecules
        molecule_graphs = [self._cg_topology_graph.subgraph(c) for c in nx.connected_components(self._cg_topology_graph)]

        # iterate through molecules and find different molecule types
        nm = iso.categorical_node_match(["name"], [None])
        em = iso.categorical_edge_match(["name_pair"], [None])
        molecule_types = []
        molecule_type_index_in_topology = []
        for m_graph in molecule_graphs:
            mol_type_recorded = False
            i_m_type = 0
            for m_type in molecule_types:
                if nx.is_isomorphic(m_graph, m_type, node_match=nm, edge_match=em):
                    mol_type_recorded = True
                    break
                i_m_type += 1
            if not mol_type_recorded:
                molecule_types.append(m_graph)
            molecule_type_index_in_topology.append(i_m_type)

        # convert molecule types to sim
        sim_MolTypes = []
        for i_mol_type, mol_type in enumerate(molecule_types):
            sim_MolTypes.append(_convert_molecule_graph_to_sim_MolType(i_mol_type, mol_type, sim_AtomTypes))

        # create sim World and System
        sim_World = sim.chem.World(sim_MolTypes, Dim=3, Units=sim.units.DimensionlessUnits)
        sim_System = sim.system.System(sim_World, Name="system")

        # add molecules to the System
        for i_mol in molecule_type_index_in_topology:
            sim_System += sim_MolTypes[i_mol].New()

        # gather together everything into a topology
        sim_topology = SimTopology(sim_AtomTypes, sim_PosMap, sim_MolTypes, sim_World, sim_System)

        return sim_topology


def _convert_molecule_graph_to_sim_MolType(molecule_index, molecule_graph, sim_AtomTypes):
    """Utility function that converts a graph representation of a CG molecule
    to a sim MolType.

    Parameters
    ----------
    molecule_index : int
        Index of the molecule type in the topology. Will be used as the default
        name of the MolType.
    molecule_graph : networkx.Graph
        CG molecule represented as a graph.
    sim_AtomTypes : dict
        Dictionary of sim AtomTypes.
    """
    # initialize list of sim AtomTypes
    sim_AtomTypes_in_MolType = []
    min_index = None    # will need to subtract off this number from indices when adding bonds

    # get sim AtomTypes from the molecule graph
    for bead in sorted(molecule_graph, key=lambda bead: bead.index):
        # get the minimum bead index
        if min_index is None:
            min_index = bead.index

        # add sim AtomType to the list
        sim_AtomTypes_in_MolType.append(sim_AtomTypes[bead.name])

    # create sim MolType
    sim_MolType = sim.chem.MolType(str(molecule_index), sim_AtomTypes_in_MolType)

    # add bonds to MolType
    for bond in molecule_graph.edges:
        sim_MolType.Bond(bond[0].index-min_index, bond[1].index-min_index)

    return sim_MolType


class CGBead(object):
    """A CGBead object represents a bead in a CG mapping.

    Attributes
    ----------
    name : str
        The name of the CGBead.
    index : int
        The index of the CGBead in the CG topology.
    aa_indices : array_like
        List of atom indices from the atomistic topology that map to this CGBead.
    aa_masses : array_like
        List of atom masses from the atomistic topology used to map atomistic
        positions to the position of this CGBead.
    """

    def __init__(self, name, index, aa_indices, aa_masses):
        self.name = name
        self.index = index
        self.aa_indices = aa_indices
        self.aa_masses = aa_masses


class SimTopology(object):
    """Sim representation of the topology of a system.

    Attributes
    ----------
    sim_AtomTypes : dict
        sim AtomTypes in the system.
    sim_PosMap : sim.atommap.PosMap
        Maps atomistic positions to CG positions.
    sim_MolTypes : list of sim.chem.MolType
        sim MolTypes in the System.
    sim_World:
        Contains chemical details.
    sim_System:
        Contains information about the box, integrator, etc.
    """

    def __init__(self, sim_AtomTypes, sim_PosMap, sim_MolTypes, sim_World, sim_System):
        self.sim_AtomTypes = sim_AtomTypes
        self.sim_PosMap = sim_PosMap
        self.sim_MolTypes = sim_MolTypes
        self.sim_World = sim_World
        self.sim_System = sim_System
