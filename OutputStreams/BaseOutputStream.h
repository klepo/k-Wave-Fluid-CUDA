/**
 * @file      BaseOutputStream.h
 *
 * @author    Jiri Jaros, Petr Kleparnik \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file of the class saving RealMatrix data into the output HDF5 file.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      11 July      2012, 10:30 (created) \n
 *            08 February  2023, 12:00 (revised)
 *
 * @copyright Copyright (C) 2019 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the C++ extension of the [k-Wave Toolbox](http://www.k-wave.org).
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify it under the terms
 * of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 */

#ifndef BASE_OUTPUT_STREAM_H
#define BASE_OUTPUT_STREAM_H

#include <limits>

// Windows build needs to undefine macro MINMAX to support std::limits
#ifdef _WIN64
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#endif

#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/IndexMatrix.h>
#include <Hdf5/Hdf5File.h>
#include <Compression/CompressHelper.h>
#include <Logger/Logger.h>

class OutputStreamContainer;

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
      /// Store compressed data (time series).
      kC,
      /// Calculate root mean square.
      kRms,
      /// Store maximum.
      kMax,
      /// Store minimum.
      kMin,
      /// Store average intensity
      kIAvg,
      /// Store average intensity using compression
      kIAvgC,
      /// Store Q term
      kQTerm,
      /// Store Q term using compression
      kQTermC
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
    BaseOutputStream(Hdf5File& file,
      MatrixName& rootObjectName,
      const RealMatrix& sourceMatrix,
      const ReduceOperator reduceOp,
      OutputStreamContainer* outputStreamContainer,
      bool doNotSaveFlag);

    /// Copy constructor not allowed.
    BaseOutputStream(const BaseOutputStream&) = delete;
    /// Destructor.
    virtual ~BaseOutputStream(){};

    /// Operator= is not allowed.
    BaseOutputStream& operator=(const BaseOutputStream&) = delete;

    /// Create a HDF5 stream and allocate data for it.
    virtual void create() = 0;

    /// Reopen the output stream after restart.
    virtual void reopen() = 0;

    /// Sample data into buffer, apply reduction - based on a sensor mask (no data flushed to disk).
    virtual void sample() = 0;

    /// Post sampling step, can work with other filled stream buffers.
    virtual void postSample();

    /// Post sampling step 2, can work with other filled stream buffers.
    virtual void postSample2();

    /// Apply post-processing on the buffer and flush it to the file.
    virtual void postProcess();

    /// Apply post-processing 2 on the buffer and flush it to the file.
    virtual void postProcess2();

    /// Flush raw data to disk.
    virtual void flushRaw() = 0;

    /// Checkpoint the stream.
    virtual void checkpoint() = 0;

    /// Close stream (apply post-processing if necessary, flush data and close).
    virtual void close() = 0;

    /// Get current store buffer.
    float* getCurrentStoreBuffer();

    /// Zero current store buffer.
    void zeroCurrentStoreBuffer();

  protected:
    /**
     * @struct ReducedValue
     * @brief  This structure is for minimal and maximal value and their indices.
     */
    struct ReducedValue
    {
        /// Min or max value.
        float value;
        /// 1D index.
        hsize_t index;
    };

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

    /**
     * @brief Check or set local minimal and maximal value and their indices.
     * @param [out] minValue - Minimal reduced local value
     * @param [out] maxValue - Maximal reduced local value
     * @param [in] value     - Value to check
     * @param [in] index     - Index of value
     */
    virtual void checkOrSetMinMaxValue(ReducedValue& minValue, ReducedValue& maxValue, float value, hsize_t index);
    /**
     * @brief Check or set global (#pragma omp critical) minimal and maximal value and their indices.
     * @param [out] minValue     - Minimal reduced global value
     * @param [out] maxValue     - Maximal reduced global value
     * @param [in] minValueLocal - Minimal reduced local value
     * @param [in] maxValueLocal - Maximal reduced local value
     */
    virtual void checkOrSetMinMaxValueGlobal(
      ReducedValue& minValue, ReducedValue& maxValue, ReducedValue minValueLocal, ReducedValue maxValueLocal);
    /**
     * @brief Load minimal and maximal values from dataset attributes.
     * @param [in, out] file   - File handle.
     * @param [in] group       - Group id
     * @param [in] datasetName - Dataset name
     * @param [out] minValue   - Minimal reduced value
     * @param [out] maxValue   - Maximal reduced value
     */
    virtual void loadMinMaxValues(
      Hdf5File& file, hid_t group, std::string datasetName, ReducedValue& minValue, ReducedValue& maxValue);
    /**
     * @brief Store minimal and maximal values as dataset attributes.
     * @param [in, out] file   - File handle.
     * @param [in] group       - Group id
     * @param [in] datasetName - Dataset name
     * @param [in] minValue    - Minimal reduced value
     * @param [in] maxValue    - Maximal reduced value
     */
    virtual void storeMinMaxValues(
      Hdf5File& file, hid_t group, std::string datasetName, ReducedValue minValue, ReducedValue maxValue);
    /**
     * @brief Load checkpoint compression coefficients and average intensity.
     */
    virtual void loadCheckpointCompressionCoefficients();
    /**
     * @brief Store checkpoint compression coefficients and average intensity.
     */
    virtual void storeCheckpointCompressionCoefficients();

    /// Copy data Host -> Device
    virtual void copyToDevice();
    /// Copy data Device -> Host
    virtual void copyFromDevice();

    /// Handle to HDF5 output file.
    Hdf5File& mFile;
    /// HDF5 group/dataset in the output file where to store data in.
    std::string mRootObjectName;
    /// Source matrix to be sampled.
    const RealMatrix& mSourceMatrix;
    /// Reduction operator.
    const ReduceOperator mReduceOp;

    /// Buffer size
    size_t mSize;
    /// Temporary buffer for store on the GPU side.
    float* mHostBuffer;
    /// Temporary buffer on the GPU side - only for aggregated quantities.
    float* mDeviceBuffer;

    /// Compression helper variables
    /// Output stream container for getting pressure compression coefficients
    /// from velocity output streams.
    OutputStreamContainer* mOutputStreamContainer = nullptr;
    /// Flag for not saving of compression coefficients or average intensity,
    /// if we want save e.g. only Q term.
    bool mDoNotSaveFlag                           = false;
    /// Compression buffer size
    size_t mOSize;
    /// First store buffer for compression coefficients.
    float* mHostBuffer1             = nullptr;
    /// Second store buffer for compression coefficients.
    float* mHostBuffer2             = nullptr;
    /// Odd or even buffer with compression coefficients.
    float* mCurrentStoreBuffer      = nullptr;
    /// Velocity output stream index
    int mVelocityOutputStreamIdx    = -1;
    /// Compression helper object.
    CompressHelper* mCompressHelper = nullptr;
    /// Local step index.
    hsize_t mStepLocal              = 0;
    /// Saving flag.
    bool mSavingFlag                = false;
    /// Odd frame flag.
    bool mOddFrameFlag              = false;
    /// Number of compressed coefficients.
    hsize_t mCompressedTimeStep     = 0;
    /// Compression complex exponential window basis
    const FloatComplex* mBE         = nullptr;
    /// Compression inverted complex exponential window basis
    const FloatComplex* mBE_1       = nullptr;
    /// Complex size
    float mComplexSize              = 2.0f;
    /// Max exponent
    int mE                          = 124;
    /// Shift flag.
    bool mShiftFlag                 = false;
    /// Mirror first "half" frame flag.
    bool mMirrorFirstHalfFrameFlag  = false;

    /// Chunk size of 4MB in number of float elements.
    static constexpr size_t kChunkSize4MB = 1048576;

}; // end of BaseOutputStream
//----------------------------------------------------------------------------------------------------------------------
#endif /* BASE_OUTPUT_STREAM_H */
