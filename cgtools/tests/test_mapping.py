import unittest

from cgtools import TopologyMapper

try:
    import sim
except ModuleNotFoundError:
    sim = None


class TestMapping(unittest.TestCase):

    def test_5A4_dod(self):
        # create TopologyMapper object
        tm = TopologyMapper.from_file("5A4_dod.pdb")

        # add residue map for butyl acrylate
        a4_masses = [14.027, 13.019, 12.011, 15.999, 15.999, 14.027, 14.027, 14.027, 15.035]
        tm.add_residue_map("A4", ["Bpba", "D4"], n_heavy_atoms_per_cg_bead=[5, 4], heavy_atom_masses=a4_masses)
        # dodecane should still be unmatched
        self.assertCountEqual({"DOD"}, tm.unmapped_residues)
        # trying to create the mapping should raise a ValueError due to the unmatched residue
        with self.assertRaises(ValueError):
            tm.create_map()

        # add residue map for dodecane
        dod_masses = [15.035] + [14.027]*10 + [15.035]
        tm.add_residue_map("DOD", ["D4"]*3, n_heavy_atoms_per_cg_bead=[4, 4, 4], heavy_atom_masses=dod_masses)
        # no unmatched residues should remain
        self.assertFalse(tm.unmapped_residues)

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

        # check sim if module exists
        if sim is not None:
            sim_topology = tm.to_sim()
            sim_AtomTypes = sim_topology.sim_AtomTypes
            self.assertCountEqual(["Bpba", "D4"], sim_AtomTypes.keys())
            sim_MolTypes = sim_topology.sim_MolTypes
            self.assertEqual(9, len(sim_MolTypes[0].Bonds))
            self.assertEqual(2, len(sim_MolTypes[1].Bonds))
            sim_System = sim_topology.sim_System
            self.assertEqual(4, sim_System.NMol)


if __name__ == '__main__':
    unittest.main()
