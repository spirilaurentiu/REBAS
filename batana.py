#region Imports
import numpy as np
import mdtraj as md
import math
import matplotlib
import matplotlib.pyplot as plt
#endregion

# -----------------------------------------------------------------------------
#                           BATStats Class
#region -----------------------------------------------------------------------
# Spherical statistics
class BATStats:
    """ Circular and spherical statistics 
    Ref: Jammalamadaka, S. R. & Sengupta, A. Topics in Circular Statistics
    World Scientific Publishing Company Incorporated (2001).
    """

    def __init__(self):
        """"""
        pass

    # Mean dihedral
    def dihedralMean(self, dihs):
        """ Mean dihedral
        :param dihs: list of dihedrals
        :return: mean dihedral
        """
        
        dihSinSum = np.sum(np.sin(dihs))
        dihCosSum = np.sum(np.cos(dihs))        
        
        return np.arctan2( dihSinSum, dihCosSum )

    # Circular variance of dihedrals
    def dihedralVar(self, dihs):
        """ Circular variance
        :param dihs: list of dihedrals
        :return: circular variance
        """
        N = len(dihs)
        if N == 0:
            return 0.0
        
        dihSinSum = np.sum(np.sin(dihs))
        dihCosSum = np.sum(np.cos(dihs))
        
        # Calculate the mean resultant length (R)
        # We divide the magnitude of the vector sum by the number of elements
        R = np.sqrt(dihSinSum**2 + dihCosSum**2) / N

        return (1 - R)
    
    def dihedralStd(self, dihs):
        """ Circular standard deviation
        :param dihs: list of dihedrals
        :return: circular standard deviation
        """
        N = len(dihs)
        if N == 0: return 0.0
        
        # Calculate R directly to avoid redundant math
        dihSinSum = np.sum(np.sin(dihs))
        dihCosSum = np.sum(np.cos(dihs))
        R = np.sqrt(dihSinSum**2 + dihCosSum**2) / N
        
        # Clip R to avoid log(0) or log of negative due to precision
        R = np.clip(R, 1e-12, 1.0)
        
        return np.sqrt(-2 * np.log(R))
    #

    # Circular correlation between two dihedral series
    def dihedralsCorrelation(self, dihs1, dihs2):
        """ Circular correlation between two dihedral series
        :param dihs1: first list of dihedrals (radians)
        :param dihs2: second list of dihedrals (radians)
        :return: circular correlation coefficient (-1 to 1)
        """
        x, y = np.array(dihs1), np.array(dihs2)
        
        # Ensure these methods return the circular mean
        x_bar = self.dihedralMean(x)
        y_bar = self.dihedralMean(y)
        
        # Calculate sines of the deviations
        x_diff = np.sin(x - x_bar)
        y_diff = np.sin(y - y_bar)

        numerator = np.sum(x_diff * y_diff)
        denominator = np.sqrt(np.sum(x_diff**2) * np.sum(y_diff**2))
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator
    
    #  Compute all-vs-all dihedral correlations
    def compute_all_vs_all_correlations(self, dihs):
        """ Compute all-vs-all dihedral correlations using NumPy vectorization.
        ChatGPT
        :param dihs: 2D array of dihedrals, shape (frames, dihedrals)
        :return: 2D array of correlation values, shape (dihedrals, dihedrals)
        """
        n_frames, n_dihs = dihs.shape

        # Step 1: Calculate means for each dihedral series
        means = np.arctan2(
            np.sum(np.sin(dihs), axis=0),
            np.sum(np.cos(dihs), axis=0)
        )

        # Step 2: Calculate sine differences for all dihedrals
        sin_diffs = np.sin(dihs - means)

        # Step 3: Compute variances for normalization
        variances = np.sum(sin_diffs**2, axis=0)

        # Step 4: Compute numerator for correlation matrix
        # This uses the outer product to calculate pairwise correlations
        numerators = sin_diffs.T @ sin_diffs  # Shape (n_dihs, n_dihs)

        # Step 5: Compute denominator for normalization
        norm_factors = np.sqrt(np.outer(variances, variances))  # Shape (n_dihs, n_dihs)

        # Step 6: Compute the correlation matrix
        correlations = numerators / norm_factors

        # Handle possible NaN values in the diagonal (from zero variance)
        correlations = np.nan_to_num(correlations)

        return correlations

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

    # Get angle indexes from bond indices
    def getAnglesFromBonds(self, bonds):
        """ Get angle indexes from bond indices (ChatGPT) 
        :param bonds: list of indexes shape (N, 2)
        :return: np.array of angles of shape (N, 3)
        """
        angle_indices = []
        for i in range(len(bonds)):
            for j in range(i + 1, len(bonds)):
                # Check if the atoms in the bond pair share a common atom
                if bonds[i][1] == bonds[j][0]:  # middle atom is shared
                    angle_indices.append([bonds[i][0], bonds[i][1], bonds[j][1]])
                elif bonds[i][0] == bonds[j][1]:  # middle atom is shared
                    angle_indices.append([bonds[i][0], bonds[i][1], bonds[j][0]])
        return np.array(angle_indices)

    # Get dihedrals from bond indices
    def getDihedralsFromBonds(self, bonds):
        """ Get dihedral indexes from bond indices (ChatGPT) 
        :param bonds: list of indexes shape (N, 2)
        :return: np.array of dihedrals of shape (N, 4)
        """
        dihedral_indices = []
        for i in range(len(bonds)):
            for j in range(i + 1, len(bonds)):
                # Check if the atoms in the bond pair share a common atom
                if bonds[i][1] == bonds[j][0]:  # middle atom is shared
                    for k in range(j + 1, len(bonds)):
                        # Find the third bond that shares a common atom
                        if bonds[j][1] == bonds[k][0]:
                            dihedral_indices.append([bonds[i][0], bonds[i][1], bonds[j][1], bonds[k][1]])
                elif bonds[i][0] == bonds[j][1]:  # middle atom is shared
                    for k in range(j + 1, len(bonds)):
                        # Find the third bond that shares a common atom
                        if bonds[j][0] == bonds[k][1]:
                            dihedral_indices.append([bonds[i][0], bonds[i][1], bonds[j][0], bonds[k][1]])
        return np.array(dihedral_indices)

    # Calculate BAT indexes
    def calcBATIndexes(self):
        """ Get bonds, angles and torsions indexes.
        """

        # Get bonds indexes
        self.bonds = self.mdtrajObj.topology.to_bondgraph().edges
        self.bondsList = list(self.mdtrajObj.topology.to_bondgraph().edges)
        self.boIxs = []
        for row in self.bondsList:
            self.boIxs.append([row[0].index, row[1].index])
        
        # Get angles indexes
        self.angIxs = self.getAnglesFromBonds(self.boIxs)

        # Get dihedral indexes
        self.dihIxs = self.getDihedralsFromBonds(self.boIxs)


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
        # Get dihedrals
        self.bos = md.compute_distances(self.mdtrajObj, self.boIxs)
        self.bos = np.array(self.bos, dtype=float)

        # Get dihedrals
        self.angs = md.compute_angles(self.mdtrajObj, self.angIxs)
        self.angs = np.array(self.angs, dtype=float)

        # Get dihedrals
        self.dihs = md.compute_dihedrals(self.mdtrajObj, self.dihIxs)
        self.dihs = np.array(self.dihs, dtype=float)
#endregion


