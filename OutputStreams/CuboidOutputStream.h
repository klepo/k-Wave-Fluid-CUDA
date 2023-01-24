/**
 * @file      CuboidOutputStream.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file of classes responsible for storing output quantities based on the
 *            cuboid sensor mask into the output HDF5 file.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      13 February  2015, 12:51 (created) \n
 *            06 March     2019, 13:19 (revised)
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

#ifndef CUBOID_OUTPUT_STREAM_H
#define CUBOID_OUTPUT_STREAM_H

#include <vector>
#include <cuda_runtime.h>

#include <OutputStreams/BaseOutputStream.h>

/**
 * @class   CuboidOutputStream
 * @brief   Output stream for quantities sampled by a cuboid corner sensor mask.
 *
 * This class writes data into separated datasets (one per cuboid) under a given dataset in the HDF5 file
 * (time-series as well as aggregations).
 *
 */
class CuboidOutputStream : public BaseOutputStream {
public:
  /// Default constructor not allowed
  CuboidOutputStream() = delete;

  /**
    * @brief Constructor.
    *
    * Constructor - links the HDF5 dataset, source (sampled matrix), sensor mask and the reduce
    * operator together. The constructor DOES NOT allocate memory because the size of the sensor mask
    * is not known at the time the instance of the class is being created.
    *
    * @param [in] file         - HDF5 file to write the output to
    * @param [in] groupName    - The name of the HDF5 group with datasets for particular cuboids
    * @param [in] sourceMatrix - Source matrix to be sampled
    * @param [in] sensorMask   - Sensor mask with the cuboid coordinates
    * @param [in] reduceOp     - Reduce operator
    */
  CuboidOutputStream(Hdf5File& file,
                     MatrixName& groupName,
                     const RealMatrix& sourceMatrix,
                     const IndexMatrix& sensorMask,
                     const ReduceOperator reduceOp,
                     OutputStreamContainer* outputStreamContainer,
                     bool doNotSaveFlag);

  /// Copy constructor is not allowed.
  CuboidOutputStream(const CuboidOutputStream&) = delete;

  /**
    * @brief Destructor.
    *
    * If the file is still opened, it applies the post processing and flush the data.
    * Then, the object memory is freed and the object destroyed.
    */
  virtual ~CuboidOutputStream();

  /// operator= is not allowed.
  CuboidOutputStream& operator=(const CuboidOutputStream&) = delete;

  /// Create the stream, allocate data for it and open necessary datasets.
  virtual void create();

  /// Reopen the output stream after restart and reload data.
  virtual void reopen();

  /**
    * @brief Sample grid points, line them up in the buffer, if necessary a reduce operator is applied.
    * @warning data is not flushed, there is no sync.
    */
  virtual void sample();

  /// Post sampling step, can work with other filled stream buffers
  virtual void postSample();

  /// Flush data to disk (from raw streams only).
  virtual void flushRaw();

  /// Apply post-processing on the buffer and flush it to the file.
  virtual void postProcess();

  /// Apply post-processing 2 on the buffer and flush it to the file.
  virtual void postProcess2();

  /// Checkpoint the stream and close.
  virtual void checkpoint();

  /// Close stream (apply post-processing if necessary, flush data and close).
  virtual void close();

protected:
  /**
    * @struct CuboidInfo
    * @brief  This structure holds information about one cuboid. Namely, its HDF5_ID,
    *         starting position in a lineup buffer.
    */
  struct CuboidInfo {
    /// Index of the dataset storing the given cuboid.
    hid_t cuboidIdx;
    /// Having a single buffer for all cuboids, where this one starts.
    size_t startingPossitionInBuffer;
    /// Maximal value
    ReducedValue maxValue;
    /// Minimal value
    ReducedValue minValue;
  };

  /**
    * @brief  Create a new dataset for a given cuboid specified by index (order).
    * @param  [in] cuboidIdx - Index of the cuboid in the sensor mask.
    * @return HDF5 handle to the dataset.
    */
  virtual hid_t createCuboidDataset(const size_t cuboidIdx);

  /// Flush the buffer to the file.
  virtual void flushBufferToFile(float* bufferToFlush = nullptr);

  /// Sensor mask to sample data.
  const IndexMatrix& mSensorMask;

  /// Handle to a HDF5 dataset.
  hid_t mGroup;

  /// vector keeping handles and positions of all cuboids.
  std::vector<CuboidInfo> mCuboidsInfo;

  /// Time step to store (N/A for aggregated).
  size_t mSampledTimeStep;

  /// Has the sampling finished?
  cudaEvent_t mEventSamplingFinished;
}; // end of CuboidOutputStream
//----------------------------------------------------------------------------------------------------------------------

#endif /* CUBOID_OUTPUT_STREAM_H */
