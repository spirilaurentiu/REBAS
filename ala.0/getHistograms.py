# Imports
import os, sys, glob
import numpy as np
import numpy.ma as ma

import RobosampleAnalysis as RA

# Filter data
def filterData(outData, Index, indicatorIndex, filterFunc):
    """
    Filter data by data[indicatorIndex] and return data[Index]
    """

    # Checks
    assert len(outData.shape) > 1, """
        filterData: data should be multidimensional
        """       
    assert type(filterFunc( np.array([1, 2]) )[0]) is np.bool_, """
        filter data should return array of bool
        """

    # Convenient names
    data              = outData[:, Index]
    dataMaskIndicator = outData[:, indicatorIndex]

    # Mask the data
    mask = filterFunc(dataMaskIndicator)
    dataMasked = ma.array(data, mask = mask)
    
    # Return
    return dataMasked[dataMasked.mask == False]


# Get a particular histogram
def get_1DHistogram(rawData, Index, rangeLeft, rangeRight, nbins, variant = 0,
    filterFunc = None,  indicatorIndex = 0):
    """
    Histogram of data[Index] by data[indicatorIndex]
    """

    # Use masked array to filter data
    if variant == "filterMask":
        procDataForHistogram = filterData(rawData,
            Index, indicatorIndex, filterFunc)

    # Use Numpy where to filter data
    elif variant == "filterWhere":
        procDataForHistogram = np.where(filterFunc(rawData[:, indicatorIndex]),
            rawData[:, Index], np.NaN)

    # Don't filter data
    elif variant == "byReplica":
        procDataForHistogram = rawData[:, Index]

    else:
        print("Warning: no filtering method provided. Using byREplica")
        procDataForHistogram = rawData[:, Index]

    # Get histogram
    hist = np.histogram( procDataForHistogram, bins = nbins, density = True,
        range = [rangeLeft, rangeRight] )

    # Return values
    return hist

# Get all histograms at once
def getAll_1DHistograms(rawData, nbins,
    wantTypes, wantSubTypes, repeats, wantReplicas, wantWorlds,
    filterFunc = None, indicatorIndex = 0):
    """
    Get all 1D histograms from data
    """

    (nofTypes, nofSubTypes, nofRepeats, nofRounds, nofReplicas,
        nofWorldsPerReplica, nofIndexes) = rawData.shape

    hists = np.zeros((nofTypes, nofSubTypes, nofRepeats, #nofRounds, 
                    nofReplicas, nofWorldsPerReplica, nofIndexes, nbins),
                    dtype = float) * np.nan

    # Histogram ranges
    rangeLeft  = np.ones((nofSubTypes, nofIndexes), dtype = float) * (-1.0 * np.Inf)
    rangeRight = np.ones((nofSubTypes, nofIndexes), dtype = float) * (+1.0 * np.Inf)


    for subTypeIx_i, subTypeIx in enumerate(wantSubTypes):                  # Subtypes: BA
        for Index_i, Index in enumerate(range(nofIndexes)):                 # Indexes: T, wIx, ...
        
            # Set ranges for this subtype histograms
            rangeData = rawData[:, subTypeIx, :, :, :, :, Index]
            if not np.isnan(rangeData).all():
                rangeLeft[subTypeIx, Index]  = np.nanmin(rangeData)
                rangeRight[subTypeIx, Index] = np.nanmax(rangeData)

            for TypeIx_i, TypeIx in enumerate(wantTypes):                   # Types: HMC, ...
                for repIx_i, repIx in enumerate(repeats):                   # Repeats
                    for replica_i, replicaIx in enumerate(wantReplicas):    # Replicas
                        for world_i, worldIx in enumerate(wantWorlds):      # Worlds

                                # Get data
                                singleOutData = rawData[TypeIx, subTypeIx, repIx, :, replicaIx, worldIx]

                                # Get histogram
                                hist = get_1DHistogram(singleOutData, Index,
                                    rangeLeft[subTypeIx, Index], rangeRight[subTypeIx, Index], nbins,
                                    variant = "filterMask",
                                    filterFunc = filterFunc,  indicatorIndex = indicatorIndex)

                                # Store values
                                hists[TypeIx, subTypeIx, repIx, replicaIx, worldIx, Index] = hist[0]

    return hists, rangeLeft, rangeRight



# Get a particular histogram
def get_2DHistogram(rawData, Indexes,
    range, nbins,
    variant = "", filterFunc = None,  indicatorIndex = 0
):
    """
    2D Histogram of data[Index] by data[indicatorIndex]
    """

    # Use masked array to filter data
    if variant == "filterMask":
        procDataForHistogram = np.array(
            [filterData(rawData, Indexes[0], indicatorIndex, filterFunc),
             filterData(rawData, Indexes[1], indicatorIndex, filterFunc)]
        )

    # Use Numpy where to filter data
    elif variant == "filterWhere":
        procDataForHistogram = np.array(
            [np.where(
                filterFunc(rawData[:, indicatorIndex]),
                rawData[:, Indexes[0]], np.NaN),
             np.where(
                filterFunc(rawData[:, indicatorIndex]),
                rawData[:, Indexes[1]], np.NaN)]
        )

    # Don't filter data
    elif variant == "byReplica":
        procDataForHistogram = rawData[:, (Indexes[0], Indexes[1])]

    else:
        print("Warning: no filtering method provided. Using byREplica")
        procDataForHistogram = np.transpose(rawData[:, (Indexes[0], Indexes[1])])

    # Get histogram
    hist = np.histogram2d( procDataForHistogram[0], procDataForHistogram[1],
        bins = nbins, density = True,
        range = range )

    # Return values
    return hist

# Get all histograms at once
def getAll_2DHistograms(rawData, Indexes, nbins, range,
    wantTypes, wantSubTypes, repeats, wantReplicas, wantWorlds,
    filterFunc = None, indicatorIndex = 0):
    """
    Get all 2D histograms from data
    """

    (nofTypes, nofSubTypes, nofRepeats, nofRounds, nofReplicas,
        nofWorldsPerReplica, nofIndexes) = rawData.shape

    hists = np.zeros((nofTypes, nofSubTypes, nofRepeats, #nofRounds, 
                    nofReplicas, nofWorldsPerReplica, nbins, nbins),
                    dtype = float) * np.nan

    # Histogram ranges
    rangeLeft  = np.ones((nofSubTypes, 2), dtype = float) * (-1.0 * np.Inf)
    rangeRight = np.ones((nofSubTypes, 2), dtype = float) * (+1.0 * np.Inf)

    for subTypeIx_i, subTypeIx in enumerate(wantSubTypes):                  # Subtypes: BA

            if np.isnan(range[0, 0]):

                # Set ranges for this subtype histograms
                rangeData = [None] * 2
                rangeData[0] = rawData[:, subTypeIx, :, :, :, :, Indexes[0]]
                rangeData[1] = rawData[:, subTypeIx, :, :, :, :, Indexes[1]]

                if not np.isnan(rangeData).all():
                    rangeLeft[subTypeIx, 0]  = np.nanmin(rangeData[0])
                    rangeLeft[subTypeIx, 1]  = np.nanmax(rangeData[1])

                    rangeRight[subTypeIx, 0] = np.nanmin(rangeData[0])
                    rangeRight[subTypeIx, 1] = np.nanmax(rangeData[1])

                    range = [[rangeLeft[subTypeIx, 0], rangeLeft[subTypeIx,1]],
                            [rangeRight[subTypeIx, 0], rangeRight[subTypeIx, 1]]]


            for TypeIx_i, TypeIx in enumerate(wantTypes):                   # Types: HMC, ...
                for repIx_i, repIx in enumerate(repeats):                   # Repeats
                    for replica_i, replicaIx in enumerate(wantReplicas):    # Replicas
                        for world_i, worldIx in enumerate(wantWorlds):      # Worlds

                                # Get data
                                singleOutData = rawData[TypeIx, subTypeIx, repIx, :, replicaIx, worldIx]

                                # Get histogram
                                hist = get_2DHistogram(singleOutData, Indexes,
                                    range, nbins,
                                    variant = "filterMask",
                                    filterFunc = filterFunc,  indicatorIndex = indicatorIndex)

                                # Store values
                                hists[TypeIx, subTypeIx, repIx, replicaIx, worldIx] = hist[0]

    return hists, rangeLeft, rangeRight