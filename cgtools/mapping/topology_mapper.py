"""topology_mapper.py: Maps an all-atom (AA) topology to a coarse grained (CG) topology"""
__all__ = ['CenterOfMass', 'Centroid', 'TopologyMapper']

import warnings
from collections import defaultdict
import itertools
from enum import Enum, auto
from typing import (
    Iterator,
    List,
    Set
)

import numpy as np

import networkx as nx
import networkx.algorithms.isomorphism as iso

import mdtraj as md

try:
    import sim
except ModuleNotFoundError:
    sim = None


class MappingType(Enum):
    """Enumerate mapping methods for AA atoms to a CG bead."""

    CenterOfMass = auto()
    """A center-of-mass mapping is used."""

    Centroid = auto()
    """Masses of atoms are ignored when mapping AA positions to CG."""


# set enums equal to exported variables
CenterOfMass = MappingType.CenterOfMass
Centroid = MappingType.Centroid


def _get_unique_residue_names_from_topology(topology: md.Topology):
    """Utility method used to get all unique residue names and their number of
    heavy atoms from a mdtraj.Topology.

    Parameters
    ----------
    topology : mdtraj.Topology
        Stores the topological information about an atomistic system.

    Returns
    -------
    dict
        Gives the number of heavy atoms in each residue type.
    """
    n_heavy_atoms_per_residue = {}
    for residue in topology.residues:
        if residue.name not in n_heavy_atoms_per_residue.keys():
            _selection_string = f"resid {residue.index} and element != H"
            n_heavy_atoms_per_residue[residue.name] = len(topology.select(_selection_string))
    return n_heavy_atoms_per_residue


class _ResidueMap(object):
    """Container class used to store information about a residue mapping.

    Attributes
    ----------
    name : str
        Name of residue in the AA topology.
    cg_bead_names : list of str
        List of CG bead names that the residue maps to.
    heavy_atom_indices_per_cg_bead : list of arraylike
        List of heavy atoms indices that map to each CG bead
    heavy_atom_masses : list of float

    """

    def __init__(
            self,
            name: str,
            cg_bead_names: List[str] = None,
            heavy_atom_indices_per_cg_bead: List[List[int]] = None,
            heavy_atom_masses: List[float] = None,
            mapping_type: MappingType = None
    ):
        self.name = name
        self.cg_bead_names = cg_bead_names
        self.heavy_atom_indices_per_cg_bead = heavy_atom_indices_per_cg_bead
        self.heavy_atom_masses = heavy_atom_masses
        self.mapping_type = mapping_type


class TopologyMapper(object):
    """TopologyMapper manages mapping from an AA topology to a CG version.

    Attributes
    ----------
    mapping_type : MappingType
        Specifies the method of mapping atomistic positions to a CG position.
    unmapped_residues : set of str
        Set of all residues missing a specified mapping.
    """

    def __init__(self, topology: md.Topology, mapping_type: MappingType = CenterOfMass):
        """Initializes a TopologyMapper object.

        Parameters
        ----------
        topology : mdtraj.Topology or str
            Stores the topological information about an atomistic system.
        mapping_type : MappingType, {CenterOfMass, Centroid}
            Specifies the method of mapping atomistic positions to a CG position.
        """
        if isinstance(topology, str):
            self._aa_topology = md.load(topology).topology
        else:
            self._aa_topology = topology
        self._residue_maps = {}
        self._unique_residues = _get_unique_residue_names_from_topology(self._aa_topology)
        self._mapping_type = mapping_type
        self._name_to_mass = {}

        # these will be set after create_map() is called
        self._cg_topology = None
        self._cg_topology_graph = None
        self._unique_cg_bead_names = None
        self._aa_to_cg_bead_map = None

    @property
    def aa_topology(self) -> md.Topology:
        """MDTraj topology."""
        return self._aa_topology

    @property
    def mapping_type(self) -> MappingType:
        """Specifies the method of mapping atomistic positions to a CG position."""
        return self._mapping_type

    @mapping_type.setter
    def mapping_type(self, value: MappingType) -> None:
        """Setter for `mapping_type` attribute."""
        if not isinstance(value, MappingType):
            raise TypeError("`center_type` must be `CenterOfMass` or `Centroid`")
        self._mapping_type = value

    def set_atom_mass(self, atom_name: str, mass: float) -> None:
        """Set the mass of an atomistic particle. Needed when using a
        `CenterOfMass` mapping with particles that are of non-conventional
        elements. This method can be also used to override masses of atoms that
        have a defined element.

        Parameters
        ----------
        atom_name : str
            Name of the atom.
        mass : float
            Mass of the atom.
        """
        self._name_to_mass[atom_name] = mass

    def remove_atom_mass(self, atom_name: str) -> None:
        """Remove the mass of an atomistic particle that has already previously
        set.

        Parameters
        ----------
        atom_name : str
            Name of the atom.
        """
        try:
            del self._name_to_mass[atom_name]
        except KeyError:
            pass

    def add_residue_map(
            self,
            residue_name: str,
            cg_bead_names: List[str],
            n_heavy_atoms_per_cg_bead: List[int] = None,
            heavy_atom_indices_per_cg_bead: List[List[int]] = None,
            heavy_atom_masses: List[float] = None,
            mapping_type: MappingType = None,
            replace: float = False
    ) -> None:
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
            residue is being mapped to more than one bead. If
            `heavy_atom_indices_per_cg_bead` is specified, then this argument
            should not be specified.
        heavy_atom_indices_per_cg_bead : list of array_like, optional
            Relative indices of heavy atoms in the residues per CG bead. Should
            be a nested list where the number of inner lists is equal to the
            number of CG beads and each inner list contains the relative
            indices of heavy atoms in the residue. For example, if we wanted to
            map AA dodecane to a two-bead model, the value of this argument
            would be [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]] assuming the
            carbons are in sequential order. This is also equivalent to passing
            [6, 6] to the `n_heavy_atoms_per_cg_bead` argument. Note that the
            maximum index should be equal to the number of heavy atoms in the
            residue minus one. If `n_heavy_atoms_per_cg_bead` is specified,
            then this argument should not be specified.
        heavy_atom_masses : list of float, optional
            List of masses of each heavy atom in the residue. If not specified,
            the masses of the atoms will be inferred from their elements or
            from the specified atom-to-mass mappings. This option should only
            be used if you want to override the default atom masses.
        mapping_type : MappingType, {CenterOfMass, Centroid}, optional
            Specifies the method of mapping atomistic positions to a CG
            position. Passing a value for this argument will override the
            overall mapping type specified in the TopologyMapper object.
        replace : bool, default=False
            If True, then replace a mapping for the same residue if it already
            exists in the TopologyMapper object.
        """
        # residue should exist in atomistic topology
        if residue_name not in self._unique_residues.keys():
            raise ValueError(
                f"no residue with name '{residue_name}' in the atomistic topology"
            )

        # check if mapping for the residue already exists
        if not replace:
            if residue_name in self._residue_maps.keys():
                raise ValueError(
                    f"mapping for the '{residue_name}' residue already exists"
                )

        # if residue is mapped to a single bead, then convert `cg_bead_names`
        # to a single-element list
        if isinstance(cg_bead_names, str):
            cg_bead_names = [cg_bead_names]

        # check that `n_heavy_atoms_per_cg_bead` or `heavy_atoms_indices_per_cg_bead`
        # is specified when mapping residue to more than one CG bead
        if len(cg_bead_names) > 1 and \
                n_heavy_atoms_per_cg_bead is None and \
                heavy_atom_indices_per_cg_bead is None:
            raise ValueError(
                f"`n_heavy_atoms_per_cg_bead` or `heavy_atom_indices_per_cg_bead` "
                f"must be specified if mapping residue '{residue_name}' to more "
                f"than one CG bead"
            )

        # check that `n_heavy_atoms_per_cg_bead` or `heavy_atoms_indices_per_cg_bead`
        # are not both specified
        if n_heavy_atoms_per_cg_bead is not None and \
                heavy_atom_indices_per_cg_bead is not None:
            raise ValueError(
                f"`n_heavy_atoms_per_cg_bead` or `heavy_atom_indices_per_cg_bead` "
                f"cannot both be specified"
            )

        # if mapping residue to one CG bead, set `n_heavy_atoms_per_cg_bead`
        # equal to a single-item list with the number of heavy atoms as the
        # only element
        if len(cg_bead_names) == 1:
            n_heavy_atoms_per_cg_bead = [self._unique_residues[residue_name]]

        # convert `n_heavy_atoms_per_cg_bead` to `heavy_atom_indices_per_cg_bead`
        if n_heavy_atoms_per_cg_bead is not None:
            # check that the length of `n_heavy_atoms_per_cg_bead` is the same
            # length as `cg_bead_names`
            if len(cg_bead_names) != len(n_heavy_atoms_per_cg_bead):
                raise ValueError(
                    f"length of `n_heavy_atoms_per_cg_bead` "
                    f"({len(n_heavy_atoms_per_cg_bead)}) doesn't match the "
                    f"length of `cg_bead_names` ({len(cg_bead_names)})"
                )
            # check that the total number of heavy atoms is correct
            _n_total_atoms = sum(n_heavy_atoms_per_cg_bead)
            if _n_total_atoms != self._unique_residues[residue_name]:
                raise ValueError(
                    f"total number of heavy atoms provided in "
                    f"`n_heavy_atoms_per_cg_bead` is {_n_total_atoms}, but "
                    f"{self._unique_residues[residue_name]} heavy atoms were "
                    f"found in residue type {residue_name}"
                )
            # carry out conversion
            heavy_atom_indices_per_cg_bead = []
            n_total = 0
            for n in n_heavy_atoms_per_cg_bead:
                heavy_atom_indices_per_cg_bead.append(list(range(n_total, n_total+n)))
                n_total += n

        # check that the length of arguments are consistent
        if len(cg_bead_names) != len(heavy_atom_indices_per_cg_bead):
            raise ValueError(
                f"if mapping residue to more than one CG bead, `cg_bead_names` "
                f"and `heavy_atom_indices_per_cg_bead` should be the same length"
            )

        # check that the number of indices provided matches the number of heavy
        # atoms in the residue
        _n_indices_provided = sum(len(i) for i in heavy_atom_indices_per_cg_bead)
        if _n_indices_provided != self._unique_residues[residue_name]:
            raise ValueError(
                f"total number of indices provided ({_n_indices_provided}) "
                f"doesn't match the number of heavy atoms in residue type "
                f"{residue_name} ({self._unique_residues[residue_name]})"
            )

        # check that there aren't any out of bound indices
        _set_indices = set(list(itertools.chain.from_iterable(heavy_atom_indices_per_cg_bead)))
        if set(range(self._unique_residues[residue_name])) != _set_indices:
            raise ValueError(
                "out-of-bound indices in `heavy_atom_indices_per_cg_bead` "
                "argument"
            )

        # check that the number of heavy atom masses passed is consistent
        if heavy_atom_masses is not None and \
                len(heavy_atom_masses) != self._unique_residues[residue_name]:
            raise ValueError(
                f"the number of heavy atom masses provided does not match the "
                f"total number of heavy atoms in `n_heavy_atoms_per_cg_bead`"
            )

        # store residue map
        self._residue_maps[residue_name] = _ResidueMap(
            residue_name, cg_bead_names=cg_bead_names,
            heavy_atom_indices_per_cg_bead=heavy_atom_indices_per_cg_bead,
            heavy_atom_masses=heavy_atom_masses,
            mapping_type=mapping_type
        )

    def get_residue_map(self, residue_name: str) -> _ResidueMap:
        """Retrieves the residue map for the specified residue name.

        Parameters
        ----------
        residue_name : str
            Name of the residue

        Returns
        -------
        _ResidueMap
        """
        return self._residue_maps[residue_name]

    @property
    def residue_maps(self) -> Iterator[_ResidueMap]:
        """Iterates through all the specified residue maps."""
        return iter(self._residue_maps.values())

    @property
    def unmapped_residues(self) -> Set[str]:
        """All residues missing a specified mapping."""
        return set(self._unique_residues.keys()) - set(self._residue_maps.keys())

    def create_map(self, ignore_unmatched_residues: bool = False) -> None:
        """Creates the atomistic-to-CG mapping. Only can be called when
        mappings for all residues have been specified.

        Parameters
        ----------
        ignore_unmatched_residues : bool, optional, default=False
            Flag that specifies whether or not to ignore residues that don't
            have a provided residue map.
        """
        # raise error if there are unmatched residues and not ignoring
        if not ignore_unmatched_residues and len(self.unmapped_residues) > 0:
            raise ValueError(
                f"there are unmatched residues: {', '.join(list(self.unmapped_residues))}"
            )

        # initialize graph of CG topology
        cg_top_graph = nx.Graph()

        # initialize CG topology in mdtraj format
        cg_top = md.Topology()

        # initialize map for atomistic indices to CG beads
        aa_to_cg_map = {}

        # initialize list of unique cg bead names
        unique_cg_bead_names = []

        # find where all hydrogen atoms are
        h_atom_map = defaultdict(list)  # maps heavy atom indices to a list of attached hydrogen indices
        h_atom_indices = self._aa_topology.select('element == H')  # indices of hydrogen atoms
        for bond in self._aa_topology.bonds:
            atom0, atom1 = bond
            if atom0.index in h_atom_indices:
                h_atom_map[atom1.index].append(atom0.index)
            elif atom1.index in h_atom_indices:
                h_atom_map[atom0.index].append(atom1.index)

        # iterate through residues to add beads to the CG topology
        cg_bead_index = 0
        for chain in self._aa_topology.chains:
            cg_chain = cg_top.add_chain()
            for residue in chain.residues:
                # skip if no mapping available
                if residue.name not in self._residue_maps.keys():
                    continue

                # add residue to cg topology
                cg_residue = cg_top.add_residue(residue.name, cg_chain)

                # get corresponding data from the residue map
                _residue_map = self._residue_maps[residue.name]

                # get indices of heavy atoms in residue
                _selection_string = f"resid {residue.index} and element != H"
                heavy_atom_indices_in_res = self._aa_topology.select(_selection_string)

                # check that the number of heavy atoms provided in the mapping is consistent
                if sum([len(i) for i in _residue_map.heavy_atom_indices_per_cg_bead]) != len(heavy_atom_indices_in_res):
                    raise ValueError(
                        f"number of heavy atoms specified does not match the "
                        f"number of heavy atoms in residue {residue.name}-{residue.index}"
                    )

                # check to see if residue has a different mapping type
                if _residue_map.mapping_type is not None:
                    mapping_type = _residue_map.mapping_type
                else:
                    mapping_type = self.mapping_type

                # if providing masses, check that the number of heavy atom masses
                # provided matches the total number of heavy atoms in the residue
                # NOTE: only check if mapping type is center-of-mass
                if mapping_type is CenterOfMass and _residue_map.heavy_atom_masses is not None:
                    if len(_residue_map.heavy_atom_masses) != len(heavy_atom_indices_in_res):
                        raise ValueError(
                            f"number of heavy atom masses provided does not match "
                            f"the number of heavy atoms in residue {residue.name}-{residue.index}"
                        )

                # all masses set to 1 if centroid mapping
                if mapping_type is Centroid:
                    heavy_atom_masses_in_res = np.ones_like(heavy_atom_indices_in_res)
                # if COM mapping and masses not provided, find using elements of the atoms
                elif mapping_type is CenterOfMass and _residue_map.heavy_atom_masses is None:
                    heavy_atom_masses_in_res = []
                    for i in heavy_atom_indices_in_res:
                        try:
                            mass_i = self._name_to_mass[self._aa_topology.atom(i).name]
                        except KeyError:
                            element_i = self._aa_topology.atom(i).element
                            if element_i is md.element.virtual:
                                name_i = self._aa_topology.atom(i).name
                                raise ValueError(
                                    f"no mass found for atom {name_i} (index={i})"
                                )
                            mass_i = self._aa_topology.atom(i).element.mass
                        heavy_atom_masses_in_res.append(mass_i)
                    heavy_atom_masses_in_res = np.array(heavy_atom_masses_in_res)
                else:
                    heavy_atom_masses_in_res = _residue_map.heavy_atom_masses

                # determine split points and split heavy atom indices and masses between beads
                heavy_atom_indices_split = [heavy_atom_indices_in_res[i] for i in _residue_map.heavy_atom_indices_per_cg_bead]
                heavy_atom_masses_split = [heavy_atom_masses_in_res[i] for i in _residue_map.heavy_atom_indices_per_cg_bead]

                # iterate through CG beads in this residue
                for i_bead_in_res in range(len(_residue_map.cg_bead_names)):
                    # get bead name lists of heavy atom indices and masses in bead
                    cg_bead_name = _residue_map.cg_bead_names[i_bead_in_res]
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
                            if self.mapping_type is CenterOfMass:
                                mass_hydrogen = self._aa_topology.atom(i_hydrogen).element.mass
                                atom_masses_in_bead.append(mass_hydrogen)
                            else:
                                atom_masses_in_bead.append(1.0)
                    atom_indices_in_bead = np.array(atom_indices_in_bead, dtype=int)
                    atom_masses_in_bead = np.array(atom_masses_in_bead, dtype=float)

                    # add bead to CG topology
                    cg_bead = CGBead(cg_bead_name, cg_bead_index, atom_indices_in_bead, atom_masses_in_bead)
                    cg_top_graph.add_node(cg_bead, pair=cg_bead_name)
                    cg_top.add_atom(cg_bead.name, md.element.virtual, cg_residue)

                    # map each atom to its respective CG bead
                    for i_atom in atom_indices_in_bead:
                        aa_to_cg_map[i_atom] = cg_bead

                    cg_bead_index += 1

        # before adding bonds, first raise warning if there are no bonds in the
        # atomistic topology
        if self._aa_topology.n_bonds == 0:
            warnings.warn(
                "no bonds found in the atomistic topology; unable to determine"
                "bonds in the CG topology"
            )

        # add bonds to CG topology
        for bond in self._aa_topology.bonds:
            try:
                cg_bead_0 = aa_to_cg_map[bond[0].index]
                cg_bead_1 = aa_to_cg_map[bond[1].index]
            except KeyError:
                warnings.warn(
                    f"{repr(bond)} contains an atom not mapped to a CG bead - "
                    f"skipping this bond"
                )
                continue
            if cg_bead_0 is not cg_bead_1:
                cg_bead_name_0 = cg_bead_0.name
                cg_bead_name_1 = cg_bead_1.name
                name_pair = tuple(sorted([cg_bead_name_0, cg_bead_name_1]))
                cg_top_graph.add_edge(cg_bead_0, cg_bead_1, name_pair=name_pair)
                cg_top.add_bond(cg_top.atom(cg_bead_0.index), cg_top.atom(cg_bead_1.index))

        # save to private attributes
        self._cg_topology = cg_top
        self._cg_topology_graph = cg_top_graph
        self._aa_to_cg_bead_map = aa_to_cg_map
        self._unique_cg_bead_names = unique_cg_bead_names

    @property
    def cg_beads(self) -> Iterator['CGBead']:
        if self._cg_topology_graph is None:
            raise AttributeError("must call `create_map` method before accessing this attribute")
        return iter(self._cg_topology_graph.nodes(data=False))

    @property
    def n_cg_beads(self) -> int:
        if self._cg_topology_graph is None:
            raise AttributeError("must call `create_map` method before accessing this attribute")
        return len(list(self.cg_beads))

    @property
    def cg_bonds(self):
        if self._cg_topology_graph is None:
            raise AttributeError("must call `create_map` method before accessing this attribute")
        return iter(self._cg_topology_graph.edges(data=False))

    @property
    def cg_topology(self) -> md.Topology:
        if self._cg_topology_graph is None:
            raise AttributeError("must call `create_map` method before accessing this attribute")
        return self._cg_topology

    @property
    def cg_topology_graph(self) -> nx.Graph:
        if self._cg_topology_graph is None:
            raise AttributeError("must call `create_map` method before accessing this attribute")
        return self._cg_topology_graph

    def to_sim(self) -> 'SimTopology':
        """Create a representation of the CG topology in the format of sim, a
        relative-entropy coarse graining package developed by the Shell Group
        at UCSB.
        """
        if sim is None:
            raise ModuleNotFoundError("no module named 'sim' found")

        # create sim AtomTypes
        sim_AtomTypes = {}
        for cg_bead_name in self._unique_cg_bead_names:
            sim_AtomTypes[cg_bead_name] = sim.chem.AtomType(cg_bead_name)

        # create sim AtomNames
        sim_AtomNames = [bead.name for bead in self.cg_beads]

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
        sim_topology = SimTopology(sim_AtomTypes, sim_AtomNames, sim_PosMap, sim_MolTypes, sim_World, sim_System)

        return sim_topology


def _convert_molecule_graph_to_sim_MolType(
        molecule_index: int,
        molecule_graph: nx.Graph,
        sim_AtomTypes: dict
):
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

    def __init__(self, name: str, index: int, aa_indices: List[int], aa_masses: List[float]):
        self.name = name
        self.index = index
        self.aa_indices = aa_indices
        self.aa_masses = aa_masses


class SimTopology(object):
    """Sim representation of the topology of a system.

    Attributes
    ----------
    AtomTypes : dict
        sim AtomTypes in the system.
    PosMap : sim.atommap.PosMap
        Maps atomistic positions to CG positions.
    MolTypes : list of sim.chem.MolType
        sim MolTypes in the System.
    World:
        Contains chemical details.
    System:
        Contains information about the box, integrator, etc.
    """

    def __init__(self, AtomTypes, AtomNames, PosMap, MolTypes, World, System):
        self.AtomTypes = AtomTypes
        self.AtomNames = AtomNames
        self.PosMap = PosMap
        self.MolTypes = MolTypes
        self.World = World
        self.System = System
