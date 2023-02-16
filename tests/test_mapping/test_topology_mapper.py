import unittest

from cgtools.mapping import TopologyMapper

try:
    import sim
except ModuleNotFoundError:
    sim = None


class TestMapping(unittest.TestCase):

    def test_add_residue_map_errors(self):
        """Tests that errors are thrown properly when add residue maps."""
        # create TopologyMapper object
        tm = TopologyMapper("5A4_2dod_ua.pdb")

        # error when adding map for residue that doesn't exist in the AA topology
        with self.assertRaises(ValueError):
            tm.add_residue_map("asdf", "A")

        # map butyl acrylate to one bead
        tm.add_residue_map("A4", "asdf")

        # error when try to add map for butyl acrylate with replace=False
        with self.assertRaises(ValueError):
            tm.add_residue_map("A4", "j;kl")

        # error when adding residue map with no atom splittings specified
        with self.assertRaises(ValueError):
            tm.add_residue_map("A4", ["Bpba", "D4"], replace=True)

        # error when splitting is inconsistent with number of CG beads
        with self.assertRaises(ValueError):
            tm.add_residue_map("A4", ["Bpba", "D4"],
                               n_heavy_atoms_per_cg_bead=[5, 3, 1],
                               replace=True)

        # error when total number of heavy atoms specified doesn't match the
        # number of heavy atoms detected in the residue
        with self.assertRaises(ValueError):
            tm.add_residue_map("A4", ["Bpba", "D4"],
                               n_heavy_atoms_per_cg_bead=[5, 3],
                               replace=True)

        # error when specifying both splitting arguments
        with self.assertRaises(ValueError):
            tm.add_residue_map("A4", ["Bpba", "D4"],
                               n_heavy_atoms_per_cg_bead=[5, 4],
                               heavy_atom_indices_per_cg_bead=[[0, 1, 2, 3, 4], [5, 6, 7, 8]],
                               replace=True)

        # error when heavy atom splitting is of different length than the CG bead names
        with self.assertRaises(ValueError):
            tm.add_residue_map("A4", ["Bpba", "D4"],
                               heavy_atom_indices_per_cg_bead=[[0, 1, 2, 3, 4], [5, 6, 7], [8]],
                               replace=True)

        # error when number of indices doesn't match number of heavy atoms
        with self.assertRaises(ValueError):
            tm.add_residue_map("A4", ["Bpba", "D4"],
                               heavy_atom_indices_per_cg_bead=[[0, 1, 2, 3], [4, 5, 6, 7]],
                               replace=True)

        # error when out-of-bound indices are provided
        with self.assertRaises(ValueError):
            tm.add_residue_map("A4", ["Bpba", "D4"],
                               heavy_atom_indices_per_cg_bead=[[0, 1, 2, 3, 4], [5, 6, 7, 9]],
                               replace=True)

    def test_5A4_2dod_UA(self):
        """Tests TopologyMapper on a system containing a poly butyl acrylate
        5-mer and 2 dodecane molecules in a united-atom representation."""
        # create TopologyMapper object
        tm = TopologyMapper("5A4_2dod_ua.pdb")

        # add masses for atoms
        tm.set_atom_mass("_CH", 13.019)
        tm.set_atom_mass("_CH2", 14.027)
        tm.set_atom_mass("_CH3", 15.035)

        # first add residue maps for butyl acrylate and dodecane
        tm.add_residue_map("A4", ["Bpba", "D4"], n_heavy_atoms_per_cg_bead=[5, 4])
        tm.add_residue_map("DOD", ["D4", "D4", "D4"], n_heavy_atoms_per_cg_bead=[4, 4, 4])

        # errors should be raised when accessing CG mappings before create_map is called
        with self.assertRaises(AttributeError):
            _ = list(tm.cg_beads)
        with self.assertRaises(AttributeError):
            _ = list(tm.cg_bonds)

        # create map
        tm.create_map()

        # verify number of CG beads and bonds
        self.assertEqual(26, len(list(tm.cg_beads)))
        self.assertEqual(22, len(list(tm.cg_bonds)))

        # check that output is correct
        cg_bead = list(tm.cg_beads)[2]
        self.assertEqual("Bpba", cg_bead.name)
        self.assertEqual(2, cg_bead.index)
        self.assertCountEqual([9, 10, 11, 12, 13], cg_bead.aa_indices)

        # check sim if module exists
        if sim is not None:
            sim_topology = tm.to_sim()
            sim_AtomTypes = sim_topology.AtomTypes
            self.assertCountEqual(["Bpba", "D4"], sim_AtomTypes.keys())
            sim_MolTypes = sim_topology.MolTypes
            self.assertEqual(9, len(sim_MolTypes[0].Bonds))
            self.assertEqual(2, len(sim_MolTypes[1].Bonds))
            sim_System = sim_topology.System
            self.assertEqual(4, sim_System.NMol)


if __name__ == '__main__':
    unittest.main()
