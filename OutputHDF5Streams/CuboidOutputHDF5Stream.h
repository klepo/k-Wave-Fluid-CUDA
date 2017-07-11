/**
 * @file        CuboidOutputHDF5Stream.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file of classes responsible for storing output quantities based on the
 *              cuboid sensor mask into the output HDF5 file.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        13 February  2015, 12:51 (created)
 *              11 June      2017, 15:44 (revised)
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

#ifndef CUBOID_OUTPUT_HDF5_STREAM_H
#define	CUBOID_OUTPUT_HDF5_STREAM_H

#include <vector>
#include <cuda_runtime.h>

#include <OutputHDF5Streams/BaseOutputHDF5Stream.h>

/**
 * @class   TCuboidOutputHDF5Stream
 * @brief   Output stream for quantities sampled by a cuboid corner sensor mask.
 * @details Output stream for quantities sampled by a cuboid corner sensor mask.
 *          This class writes data into separated datasets (one per cuboid) under a given dataset
 *          in the HDF5 file (time-series as well as aggregations).
 *
 */
class TCuboidOutputHDF5Stream : public TBaseOutputHDF5Stream
{
  public:
    /// Default constructor not allowed
    TCuboidOutputHDF5Stream() = delete;
    /// Constructor.
    TCuboidOutputHDF5Stream(THDF5_File&           file,
                            MatrixName&          groupName,
                            const TRealMatrix&    sourceMatrix,
                            const TIndexMatrix&   sensorMask,
                            const TReduceOperator reduceOp);
    /// Destructor.
    virtual ~TCuboidOutputHDF5Stream();

    /// operator= is not allowed.
    TCuboidOutputHDF5Stream& operator=(const TCuboidOutputHDF5Stream&) = delete;

    /// Create a HDF5 stream and allocate data for it.
    virtual void Create();

    /// Reopen the output stream after restart and reload data.
    virtual void Reopen();

    /// Sample data into buffer, apply reduce or copy to the CPU side
    virtual void Sample();

    /// Flush data to disk (from raw streams only).
    virtual void FlushRaw();

    /// Apply post-processing on the buffer and flush it to the file.
    virtual void PostProcess();

    /// Checkpoint the stream and close.
    virtual void Checkpoint();

    /// Close stream (apply post-processing if necessary, flush data and close).
    virtual void Close();

 protected:
    /**
     * @struct TCuboidInfo
     * @brief   This structure information about one cuboid. Namely, its HDF5_ID,
     *          starting position in a lineup buffer.
     * @details This structure information about one cuboid. Namely, its HDF5_ID,
     *          starting position in a lineup buffer.
     */
    struct TCuboidInfo
    {
      /// Idx of the dataset storing the given cuboid.
      hid_t  cuboidIdx;
      /// Having a single buffer for all cuboids, where this one starts.
      size_t startingPossitionInBuffer;
    };

    /// Create a new dataset for a given cuboid specified by index (order).
    virtual hid_t CreateCuboidDataset(const size_t cuboidIdx);

    /// Flush the buffer to the file.
    virtual void FlushBufferToFile();

    /// Sensor mask to sample data.
    const TIndexMatrix& sensorMask;

    /// Handle to a HDF5 dataset.
    hid_t group;

    /// vector keeping handles and positions of all cuboids
    std::vector<TCuboidInfo> cuboidsInfo;

    /// Timestep to store (N/A for aggregated).
    size_t sampledTimeStep;

    /// Has the sampling finished?
    cudaEvent_t eventSamplingFinished;
};// end of TCuboidOutputHDF5Stream
//--------------------------------------------------------------------------------------------------

#endif	/* CUBOID_OUTPUT_HDF5_STREAM_H */

