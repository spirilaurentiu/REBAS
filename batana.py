#region Imports
import numpy as np
import mdtraj as md
from observables import compute_bat_values, compute_bat_rmsf
#endregion

# -----------------------------------------------------------------------------
#                           BAT Utilities
#region -----------------------------------------------------------------------
class BATIndexBuilder:
    """Build BAT index arrays from topology bond connectivity."""

    @staticmethod
    def getAnglesFromBonds(bonds):
        """Get angle indices from bond indices.
        :param bonds: list/array of shape (N, 2)
        :return: np.array of shape (M, 3)
        """
        angle_indices = []
        for i in range(len(bonds)):
            for j in range(i + 1, len(bonds)):
                if bonds[i][1] == bonds[j][0]:
                    angle_indices.append([bonds[i][0], bonds[i][1], bonds[j][1]])
                elif bonds[i][0] == bonds[j][1]:
                    angle_indices.append([bonds[i][0], bonds[i][1], bonds[j][0]])
        return np.array(angle_indices, dtype=int)

    @staticmethod
    def getDihedralsFromBonds(bonds):
        """Get dihedral indices from bond indices.
        :param bonds: list/array of shape (N, 2)
        :return: np.array of shape (K, 4)
        """
        dihedral_indices = []
        for i in range(len(bonds)):
            for j in range(i + 1, len(bonds)):
                if bonds[i][1] == bonds[j][0]:
                    for k in range(j + 1, len(bonds)):
                        if bonds[j][1] == bonds[k][0]:
                            dihedral_indices.append([bonds[i][0], bonds[i][1], bonds[j][1], bonds[k][1]])
                elif bonds[i][0] == bonds[j][1]:
                    for k in range(j + 1, len(bonds)):
                        if bonds[j][0] == bonds[k][1]:
                            dihedral_indices.append([bonds[i][0], bonds[i][1], bonds[j][0], bonds[k][1]])
        return np.array(dihedral_indices, dtype=int)

    @staticmethod
    def from_topology(topology):
        """Build BAT bond/angle/dihedral index arrays from mdtraj topology."""
        bond_rows = list(topology.to_bondgraph().edges)
        boIxs = np.array([[row[0].index, row[1].index] for row in bond_rows], dtype=int)
        angIxs = BATIndexBuilder.getAnglesFromBonds(boIxs)
        dihIxs = BATIndexBuilder.getDihedralsFromBonds(boIxs)
        return boIxs, angIxs, dihIxs


#endregion

# -----------------------------------------------------------------------------
#                               BAT Class
#region -----------------------------------------------------------------------
class BAT:
    """ Calculates Bond-Angle-Torsion coordinates.
    Attributes:
        dcd: trajectory filename
        prmtop: topology filename
        bonds, bondsList: NetworkX molecular graph
        boIxs, angIxs, dihIxs: BAT indexes
        bos, angs, dihs: BAT values
    """

    def __init__(self, dcd, prmtop):
        """ Inits SampleClass with blah.
        :param prmtop: topology filename
        :param dcd: trajectory filename
        """
        self.dcd = dcd
        self.prmtop = prmtop

        self.mdtrajObj = md.load(self.dcd, top = self.prmtop)
 
        #self.mdtrajObj.unitcell_lengths[:] = 100.0
        #self.mdtrajObj.unitcell_angles[:] = 90.0

        self.bonds = None
        self.bondsList = None

        self.boIxs = None
        self.angIxs = None
        self.dihIxs = None

        self.bos = None
        self.angs = None
        self.dihs = None
        self.rmsf = None

    # Get angle indexes from bond indices
    def getAnglesFromBonds(self, bonds):
        """ Get angle indexes from bond indices (ChatGPT) 
        :param bonds: list of indexes shape (N, 2)
        :return: np.array of angles of shape (N, 3)
        """
        return BATIndexBuilder.getAnglesFromBonds(bonds)

    # Get dihedrals from bond indices
    def getDihedralsFromBonds(self, bonds):
        """ Get dihedral indexes from bond indices (ChatGPT) 
        :param bonds: list of indexes shape (N, 2)
        :return: np.array of dihedrals of shape (N, 4)
        """
        return BATIndexBuilder.getDihedralsFromBonds(bonds)

    # Calculate BAT indexes
    def calcBATIndexes(self):
        """ Get bonds, angles and torsions indexes.
        """
        self.bonds = self.mdtrajObj.topology.to_bondgraph().edges
        self.bondsList = list(self.mdtrajObj.topology.to_bondgraph().edges)
        self.boIxs, self.angIxs, self.dihIxs = BATIndexBuilder.from_topology(self.mdtrajObj.topology)


        # Ethane
        # self.boIxs = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [1, 6], [1, 7]])
        # self.angIxs = np.array([[0, 1, 5], [0, 1, 6], [0, 1, 7], [1, 0, 2], [1, 0, 3], [1, 0, 4]])
        # self.dihIxs = np.array([[2, 0, 1, 5]])

        #print("self.boIxs", self.boIxs)
        #print("self.angIxs", self.angIxs)
        #print("self.dihIxs", self.dihIxs)

    # Calculate BAT values
    def calcBAT(self):
        """ Get bonds, angles and torsions values.
        """
        if self.boIxs is None or self.angIxs is None or self.dihIxs is None:
            self.calcBATIndexes()
        self.bos, self.angs, self.dihs = compute_bat_values(
            self.mdtrajObj,
            self.boIxs,
            self.angIxs,
            self.dihIxs,
        )

    def calcBATRMSF(self):
        """Compute BAT-space RMS fluctuations for the current trajectory."""
        if self.bos is None or self.angs is None or self.dihs is None:
            self.calcBAT()
        self.rmsf = compute_bat_rmsf(self.bos, self.angs, self.dihs)
        return self.rmsf
#endregion


