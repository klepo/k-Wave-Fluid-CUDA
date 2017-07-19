/**
 * @file        BaseOutputStream.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file of the class saving RealMatrix data into the output HDF5 file.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        11 July      2012, 10:30 (created) \n
 *              19 July      2017, 15:21 (revised)
 *
 * @section License
 * This file is part of the C++ extension of the k-Wave Toolbox
 * (http://www.k-wave.org).\n Copyright (C) 2016 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see http://www.gnu.org/licenses/.
 */

#ifndef BaseOutputStreamH
#define BaseOutputStreamH


#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/IndexMatrix.h>
#include <HDF5/HDF5_File.h>

/**
 * @class   BaseOutputStream
 * @brief   Abstract base class for output data streams (sampled data).
 *
 * Data are sampled based on the the sensor mask and the reduction operator. The sampled data is stored in the output
 * HDF5 file.
 *
 */
class BaseOutputStream
{
  public:
    /**
     * @enum  ReduceOperator
     * @brief How to aggregate data.
     */
    enum class ReduceOperator
    {
      /// Store actual data (time series).
      kNone,
       /// Calculate root mean square
      kRms,
      /// Store maximum
      kMax,
      /// Store minimum
      kMin
    };


    /// Default constructor not allowed.
    BaseOutputStream() = delete;
    /**
     * @brief Constructor - there is no sensor mask by default!
     *
     * The constructor links the HDF5 dataset, source (sampled matrix) and the reduce operator together.
     * The constructor DOES NOT allocate memory because the size of the sensor mask is not known at the time the
     * instance of the class is being created.
     *
     * @param [in] file           - Handle to the HDF5 (output) file.
     * @param [in] rootObjectName - The root object that stores the sample  data (dataset or group).
     * @param [in] sourceMatrix   - The source matrix (only real matrices are supported).
     * @param [in] reduceOp       - Reduction operator.
     */
    BaseOutputStream(THDF5_File&          file,
                     MatrixName&          rootObjectName,
                     const RealMatrix&    sourceMatrix,
                     const ReduceOperator reduceOp);

    /// Copy constructor not allowed.
    BaseOutputStream(const BaseOutputStream&) = delete;
    /// Destructor
    virtual ~BaseOutputStream() {};

    /// Operator= is not allowed.
    BaseOutputStream& operator=(const BaseOutputStream&) = delete;

    /// Create a HDF5 stream and allocate data for it.
    virtual void create() = 0;

    /// Reopen the output stream after restart.
    virtual void reopen() = 0;

    /// Sample data into buffer, apply reduction - based on a sensor mask (no data flushed to disk).
    virtual void sample() = 0;

    /// Flush raw data to disk.
    virtual void flushRaw() = 0;

    /// Apply post-processing on the buffer and flush it to the file.
    virtual void postProcess();

    /// Checkpoint the stream.
    virtual void checkpoint() = 0;

    /// Close stream (apply post-processing if necessary, flush data and close).
    virtual void close() = 0;

  protected:
    /**
     * @brief    Allocate memory using proper memory alignment.
     * @throw    std::bad_alloc - If there's not enough memory.
     * @warning  This can routine is not used in the base class (should be used in derived ones).
     */
    virtual void allocateMemory();
    /**
     * @brief   Free memory.
     * @warning This can routine is not used in the base class (should be used in derived ones).
     */
    virtual void freeMemory();

    /// Copy data Host -> Device
    virtual void copyToDevice();
    /// Copy data Device -> Host
    virtual void copyFromDevice();

    /// Handle to HDF5 output file
    THDF5_File& mFile;

    /// HDF5 group in the output file where to store data in.
    std::string mRootObjectName;

    /// Source matrix to be sampled.
    const RealMatrix& mSourceMatrix;

    /// HDF5 dataset handle.
    hid_t mDataset;
    /// Reduction operator.
    const ReduceOperator mReduceOp;

    /// Position in the dataset
    DimensionSizes mPosition;

    /// Buffer size
    size_t  mSize;

    /// Temporary buffer for store on the GPU side
    float* mHostBuffer;
    /// Temporary buffer on the GPU side - only for aggregated quantities
    float* mDeviceBuffer;

    /// chunk size of 4MB in number of float elements
    static constexpr size_t kChunkSize4MB = 1048576;
    /// The minimum number of elements to start sampling in parallel (4MB)
    static constexpr size_t kMinGridpointsToSampleInParallel = 1048576;

};// end of BaseOutputStream
//----------------------------------------------------------------------------------------------------------------------
#endif /* BaseOutputStreamH */

