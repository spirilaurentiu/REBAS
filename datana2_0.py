# Imports
import os, sys
import copy, itertools
import numpy as np
import glob


# region ######################################################################
npHistIx_data = 0
npHistIx_edge = 1
HistIx_delta = 2
kB = 0.00831447086363 
# endregion ###################################################################



# Print Numpy array the way we want
def PrintNumpyArray(arr__, precision=3):
    """ Print Numpy array the way we want
    :param arr__: Numpy array
    :param precision: Number of decimal places to print
    """
    with np.printoptions(precision=precision, suppress=True):
        print(np.array2string(arr__, precision=precision, suppress_small=True))


# -----------------------------------------------------------------------------
#                           Class: Data Reader
#region -----------------------------------------------------------------------
class DataReader:
    """ Read data from a list of list of files [inFNRoot][suffix].
    Attributes:
        inFNRoots: List of Robosample output names
    """

    # Reads data
    def __init__(self, directory, inFNRoots, skipheaderrows):
        """ Read data from files.
        :param directory: Directory containing the files
        :param inFNRoots: List of Robosample output names
        :param skipheaderrows: Number of header rows to skip in each file
        """
        self.inFNRoots = inFNRoots
        self.nroots = len(inFNRoots)
        self.allData = [None for _ in range(self.nroots)]
        self.allSeeds = [None for _ in range(self.nroots)]

        # Iterate each file root
        self.maxNofFiles = 0
        self.maxNofRows = 0
        for inFNRootIx, inFNRoot in enumerate(inFNRoots):

            # Get all filenames
            FNList = [entry.path for entry in os.scandir(directory) if entry.is_file() and entry.name.startswith(inFNRoot)]
            print("Root", inFNRoot, "files:\n", FNList)

            if(len(FNList) == 0):
                sys.stderr.write("No files with root " + inFNRoot + "\n")
                sys.exit(1)

            if(len(FNList) > self.maxNofFiles):
                self.maxNofFiles = len(FNList)

            self.allData[inFNRootIx] = []
            self.allSeeds[inFNRootIx] = []

            # Iterate each file
            for inFNIx, inFN in enumerate(FNList):

                # Get data from a file
                self.allData[inFNRootIx].append(np.genfromtxt(inFN, skip_header=skipheaderrows))

                # Get data's shape
                nrows = self.allData[inFNRootIx][-1].shape[0]

                if (inFNRootIx == 0) and (inFNIx == 0):
                    if len(self.allData[inFNRootIx][-1].shape) == 2:
                        ncols = self.allData[inFNRootIx][-1].shape[1]
                    else:
                        ncols = 1

                if len(self.allData[inFNRootIx][-1].shape) == 2:
                    if ncols != self.allData[inFNRootIx][-1].shape[1]:
                        sys.stderr.write("Input data doesn't have the same number of columns\n")
                        sys.exit(1)

                self.allData[inFNRootIx][-1] = (self.allData[inFNRootIx][-1]).reshape((nrows, ncols))

                # Get data's maximum size
                if nrows > self.maxNofRows:
                    self.maxNofRows = nrows

                # Get seed
                FNparts = (inFN.split("/")[-1]).split(".")
                self.allSeeds[inFNRootIx].append(int(FNparts[-1]))

        self.data = np.empty((self.nroots, self.maxNofFiles, ncols, self.maxNofRows)) * np.nan
        for Ix, Root in enumerate(self.inFNRoots):
            for Jx in range(self.maxNofFiles):

                if Jx >= len(self.allData[Ix][Jx]): continue
                
                for Kx  in range(ncols):

                    nofRows = len(self.allData[Ix][Jx][:, Kx])
                    restArr = np.empty( (self.maxNofRows - nofRows) ) * np.nan

                    self.data[Ix, Jx, Kx] = np.concatenate( (self.allData[Ix][Jx][:, Kx], restArr) )

                    # print("data", Ix, Jx, Kx, self.data[Ix, Jx, Kx])
                    # print("allData", Ix, Jx, Kx, self.allData[Ix][Jx][:, Kx])

        self.nroots = self.data.shape[0]
        self.nfiles = self.data.shape[1]
        self.ncols = self.data.shape[2]
        self.nrows = self.data.shape[3]

        print("self.data.shape", self.data.shape)

    # Get read data
    def getData(self):
        """ Get read data
        :return: data
        """
        return self.data

    # Get indexes
    def getSeeds(self):
        """ Get indexes read from filenames
        :return: indexes
        """
        return self.allSeeds

#endregion --------------------------------------------------------------------

# -----------------------------------------------------------------------------
#                           Class: Matrix Data Reader
#region -----------------------------------------------------------------------
class MatrixDataReader(DataReader):
    """ Reads data from a list of lists of files [inFNRoot][suffix].
    Attributes:
         inFNRoots: List of filename roots
    """
    def __init__(self, dir, inFNRoots, nrepl):
        super().__init__(dir, inFNRoots, 0)

        self.nrepl = nrepl
        self.nroundsAllRepl = self.nrows
        self.nrounds = self.nroundsAllRepl // self.nrepl

        self.data = np.transpose(self.data, axes=(0, 1, 3, 2))
        self.data = self.data.reshape((self.nroots, self.nfiles, self.nrounds, self.nrepl, self.nrepl))

    # Calculate exchange rates between replicas
    def exchangeRate(self,):
        """ Calculate exchange rates between replicas
        :return: exchange rates between replicas
        """
        pass

#endregion --------------------------------------------------------------------

# -----------------------------------------------------------------------------
#                           Class: Statistics
#region -----------------------------------------------------------------------
class Statistics:
    """ Statistics on a list of list of files [inFNRoot][suffix].
    Attributes:
    """

    # 
    def __init__(self, data = None):
        """ Desc.
        """
        self.data = data

    # Basic statistics
    def basic(self,):
        """ Compute average, std
        :return: (average, std)
        """
        print("Basic statistics under construction.")
        self.means = np.empty(self.data.shape[:-1], dtype=float) * np.nan
        self.stds = np.empty(self.data.shape[:-1], dtype=float) * np.nan

        self.meanOfMeans = np.empty((self.data.shape[0]), dtype=float) * np.nan

        for Ix, inFNRoot in enumerate(range(self.data.shape[0])):
            for Jx, inFN in enumerate(range(self.data.shape[1])):
                self.means[Ix, Jx] = np.nanmean(self.data[Ix, Jx])
                self.stds[Ix, Jx] = np.nanstd(self.data[Ix, Jx])

            self.meanOfMeans[Ix] = np.nanmean(self.means[Ix])
        
        return self.means, self.stds, self.meanOfMeans

    # Histograms
    def calcHistograms(self, data=None, nbins=10, way="counts", rangePerFile=False):
        """ Histograms
        :param nbins: number of bins
        :return: histograms
        """

        data = None
        if (data is None):
            data = self.data

        if(len(data.shape) != 4):
            sys.stderr.write("Data should be 4-dim: root, file, columnInFile, rowInFile\n")

        #region Get all ranges ===============================================
        self.minAll = np.nanmin(self.data)
        self.maxAll = np.nanmax(self.data)
        #print("minAll maxAll", self.minAll, self.maxAll)

        self.minCol = np.empty(self.data.shape[2]) * np.nan
        self.maxCol = np.empty(self.data.shape[2]) * np.nan
        for Kx_Col in range(self.data.shape[2]):
            self.minCol[Kx_Col] = np.nanmin(self.data[:, :, Kx_Col, :])
            self.maxCol[Kx_Col] = np.nanmax(self.data[:, :, Kx_Col, :])
        #print("self.minCol", self.minCol)
        #print("self.maxCol", self.maxCol)

        minEntries = np.zeros((self.data.shape[0], self.data.shape[2]), dtype=float)
        maxEntries = np.zeros((self.data.shape[0], self.data.shape[2]), dtype=float)
        #print("data.shape", data.shape)

        for Ix, inFNRoot in enumerate(range(self.data.shape[0])):
            for Jx, inFN in enumerate(range(self.data.shape[1])):
                for Kx, colData in enumerate(range(self.data.shape[2])):        
                    minEntries[Ix, Kx] = np.nanmin(data[Ix, :, Kx])
                    maxEntries[Ix, Kx] = np.nanmax(data[Ix, :, Kx])
        #endregion ------------------------------------------------------------

        histShape = (self.data.shape[0], self.data.shape[1], self.data.shape[2]) + (3, nbins,)
        hists = np.empty(histShape, dtype = float) * np.nan

        #region Calculate histograms ==========================================
        for Ix, inFNRoot in enumerate(range(self.data.shape[0])):
            for Jx, inFN in enumerate(range(self.data.shape[1])):
                for Kx, colData in enumerate(range(self.data.shape[2])):

                    # Get range for histogram
                    if rangePerFile:
                        minMaxRange = [minEntries[Ix, Kx], maxEntries[Ix, Kx]]
                    else:
                        minMaxRange = [self.minCol[Kx], self.maxCol[Kx]]

                    if np.abs(minMaxRange[1] - minMaxRange[0]) < 0.00001:
                         print("Range limits for histograms are the same. Dirac Delta situation.")
                         minMaxRange[0] -= 0.0001
                         minMaxRange[1] += 0.0001

                    delta_X_ = (minMaxRange[1] - minMaxRange[0]) / nbins

                    # Calculate histograms
                    hist_vals, bin_edges = np.histogram(self.data[Ix, Jx, Kx], bins=nbins, range=minMaxRange, density=False) # Counts

                    hists[Ix, Jx, Kx][0] = hist_vals

                    hists[Ix, Jx, Kx][1] = [(bin_edge + bin_edges[ix + 1]) / 2 for ix, bin_edge in enumerate(bin_edges[:-1])] # Put the centers of the edges

                    hists[Ix, Jx, Kx][2] = np.zeros(hist_vals.shape) + delta_X_

                    if way == "probability":
                        hists[Ix, Jx, Kx][0] /= float(np.sum(~np.isnan(self.data[Ix, Jx, Kx]))) # Probability
                    elif way == "density":
                        hists[Ix, Jx, Kx][0] /= float(np.sum(~np.isnan(self.data[Ix, Jx, Kx]))) # Probability first
                        hists[Ix, Jx, Kx][0] /= hists[Ix, Jx, Kx][2] # Density
                    #print("Sum hist", np.sum(hists[Ix, Jx, Kx][0]))

                    # Check delta_X_
                    if(delta_X_ == 0.0):
                        print("Dirac delta")

                    if np.sum((bin_edges[1:] - bin_edges[0:-1]) - delta_X_) > 0.00001:
                        print("Error", "delta Observable in histograms is wrong")
                        print(bin_edges[1:] - bin_edges[0:-1])
                        print(delta_X_)
                    

        #endregions -----------------------------------------------------------

        return hists


#endregion --------------------------------------------------------------------

