/**
 * @file        BaseOutputHDF5Stream.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 *
 * @brief       The header file of the class saving RealMatrix data into
 *              the output HDF5 file.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        11 July      2012, 10:30 (created) \n
 *              18 February  2015, 14:04 (revised)
 *
 * @section License
 * This file is part of the C++ extension of the k-Wave Toolbox
 * (http://www.k-wave.org).\n Copyright (C) 2014 Jiri Jaros, Beau Johnston
 * and Bradley Treeby
 *
 * This file is part of the k-Wave. k-Wave is free software: you can
 * redistribute it and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation, either version
 * 3 of the License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with k-Wave. If not, see http://www.gnu.org/licenses/.
 */

#ifndef BASE_OUTPUT_HDF5_STREAM_H
#define BASE_OUTPUT_HDF5_STREAM_H

#include <string>
#include <cstring>
#include <vector>
#include <stdexcept>

#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/IndexMatrix.h>
#include <HDF5/HDF5_File.h>



using namespace std;

/**
 * @class TBaseOutputHDF5Stream
 * @brief   Abstract base class for output data streams (sampled data).
 * @details Abstract base class for output data streams (sampled data).
 *
 */
class TBaseOutputHDF5Stream
{
  public:
    /**
     * @enum TOutputHDF5StreamReductionOperator
     * @brief How to aggregate data \n
     *        roNONE - store actual data (time series)
     *        roRMS  - calculate root mean square \n
     *        roMAX  - store maximum
     *        roMIN  - store minimum
     */
    enum TReductionOperator
    {
        roNONE, roRMS, roMAX, roMIN
    };

    /**
     * Constructor - it links the HDF5 dataset, source (sampled matrix) and the reduction
     * operator together.
     */
    TBaseOutputHDF5Stream(THDF5_File &             HDF5_File,
                          const char *             HDF5_RootObjectName,
                          const TRealMatrix &      SourceMatrix,
                          const TReductionOperator ReductionOp);

    /**
     * @brief Destructor.
     * @details Destructor.
     */
    virtual ~TBaseOutputHDF5Stream()
    {
      delete [] HDF5_RootObjectName;
    };

    /// Create a HDF5 stream and allocate data for it.
    virtual void Create() = 0;

    /// Reopen the output stream after restart.
    virtual void Reopen() = 0;

    /// Sample data into buffer, apply reduction - based on a sensor mask (no data flushed to disk).
    virtual void Sample() = 0;

    /// Flush raw data to disk.
    virtual void FlushRaw() = 0;

    /// Apply post-processing on the buffer and flush it to the file.
    virtual void PostProcess();

    /// Checkpoint the stream.
    virtual void Checkpoint() = 0;

    /// Close stream (apply post-processing if necessary, flush data and close).
    virtual void Close() = 0;

  protected:
    /// Default constructor not allowed.
    TBaseOutputHDF5Stream();
    /// Copy constructor not allowed.
    TBaseOutputHDF5Stream(const TBaseOutputHDF5Stream & src);
    /// Operator = not allowed (we don't want any data movements).
    TBaseOutputHDF5Stream & operator = (const TBaseOutputHDF5Stream & src);

    /// A generic function to allocate memory - not used in the base class.
    virtual void AllocateMemory();
    /// A generic function to free memory - not used in the base class.
    virtual void FreeMemory();

    /// Copy data  HostStoreBuffer -> DeviceStoreBuffer
    virtual void CopyDataToDevice();
    /// Copy data  DeviceStoreBuffer -> HostStoreBuffer
    virtual void CopyDataFromDevice();

    /// HDF5 file handle.
    THDF5_File& HDF5_File;

    /// Dataset name.
    char* HDF5_RootObjectName;

    /// Source matrix to be sampled.
    const TRealMatrix& SourceMatrix;

    /// HDF5 dataset handle.
    hid_t HDF5_Dataset_id;
    /// Reduction operator.
    const TReductionOperator ReductionOp;

    /// Position in the dataset
    TDimensionSizes Position;

    /// Buffer size
    size_t  BufferSize;

    /// Temporary buffer for store on the GPU side
    float * HostStoreBuffer;
    /// Temporary buffer on the GPU side - only for aggregated quantities
    float * DeviceStoreBuffer;

    /// chunk size of 4MB in number of float elements
    static const size_t ChunkSize_4MB = 1048576;

    /// The minimum number of elements to start sampling in parallel (4MB)
    static const size_t MinGridpointsToSampleInParallel = 1048576;

};// end of TBaseOutputHDF5Stream

#endif /* BASE_OUTPUT_HDF5_STREAM_H */

