/**
 * @file      OutputStreamContainer.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file defining the output stream container.
 *
 * @version   kspaceFirstOrder3D 3.5
 *
 * @date      04 December  2014, 11:00 (created) \n
 *            04 September 2017, 08:42 (revised)
 *
 * @copyright Copyright (C) 2017 Jiri Jaros and Bradley Treeby.
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

#ifndef OUTPUT_STREAM_CONTAINER_H
#define OUTPUT_STREAM_CONTAINER_H

#include <map>

#include <Containers/MatrixContainer.h>
#include <OutputStreams/BaseOutputStream.h>

#include <Utils/MatrixNames.h>
#include <Utils/DimensionSizes.h>


/**
 * @class   OutputStreamContainer
 * @brief   A container for output streams.
 * @details The output stream container maintains matrices used to sample data.
 * These may or may not require some scratch place or reuse temp matrices.
 */
class OutputStreamContainer
{
  public:
    /**
      * @enum    OutputStreamIdx
      * @brief   Output streams identifiers in k-Wave.
      * @details Output streams identifiers in k-Wave.
      * @warning The order of Idxs is mandatory! it determines the order of sampling and is used for
      *          hiding the PCI-E latency
      */
    enum class OutputStreamIdx
    {
      /// Pressure time series.
      kPressureRaw,
      /// Velocity x time series.
      kVelocityXRaw,
      /// Velocity y time series.
      kVelocityYRaw,
      /// Velocity z time series.
      kVelocityZRaw,
      /// Non staggered velocity x time series.
      kVelocityXNonStaggeredRaw,
      /// Non staggered velocity y time series.
      kVelocityYNonStaggeredRaw,
      /// Non staggered velocity z time series.
      kVelocityZNonStaggeredRaw,

      /// RMS of pressure over sensor mask.
      kPressureRms,
      /// Max of pressure over sensor mask.
      kPressureMax,
      /// Min of pressure over sensor mask.
      kPressureMin,
      /// Max of pressure over all domain.
      kPressureMaxAll,
      /// Min of pressure over all domain.
      kPressureMinAll,

      /// RMS of velocity x over sensor mask.
      kVelocityXRms,
      /// RMS of velocity y over sensor mask.
      kVelocityYRms,
      /// RMS of velocity z over sensor mask.
      kVelocityZRms,
      /// Max of velocity x over sensor mask.
      kVelocityXMax,
      /// Max of velocity y over sensor mask.
      kVelocityYMax,
      /// Max of velocity z over sensor mask.
      kVelocityZMax,
      /// Min of velocity x over sensor mask.
      kVelocityXMin,
      /// Min of velocity y over sensor mask.
      kVelocityYMin,
      /// Min of velocity z over sensor mask.
      kVelocityZMin,

      /// Max of velocity x over all domain.
      kVelocityXMaxAll,
      /// Max of velocity y over all domain.
      kVelocityYMaxAll,
      /// Max of velocity z over all domain.
      kVelocityZMaxAll,
      /// Min of velocity x over all domain.
      kVelocityXMinAll,
      /// Min of velocity y over all domain.
      kVelocityYMinAll,
      /// Min of velocity z over all domain.
      kVelocityZMinAll,
    };// end of OutputStreamIdx


    /// Constructor.
    OutputStreamContainer();
    /// Copy constructor not allowed.
    OutputStreamContainer(const OutputStreamContainer&) = delete;
    /// Destructor.
    ~OutputStreamContainer();

    /// Operator = not allowed.
    OutputStreamContainer& operator=(OutputStreamContainer&) = delete;

    /**
     * @brief  Get size of the container.
     * @return the size of the container
     */
    inline size_t size() const
    {
      return mContainer.size();
    };

    /**
     * @brief   Is the container empty?
     * @return  true - If the container is empty.
     */
    inline bool empty() const
    {
      return mContainer.empty();
    };

    /**
     * @brief operator []
     * @param [in] outputStreamIdx - Id of the output stream.
     * @return An element of the container.
     */
    BaseOutputStream& operator[](const OutputStreamIdx outputStreamIdx)
    {
      return (* (mContainer[outputStreamIdx]));
    };

    /**
     * @brief Add all streams in the simulation to the container, set all streams records here!
     *
     * Please note, the Matrix container has to be populated before calling this routine.
     *
     * @param [in] matrixContainer - Matrix container to link the steams with sampled matrices and
     *                               sensor masks.
     */
    void addStreams(MatrixContainer& matrixContainer);

    /// Create all streams - opens the datasets.
    void createStreams();
    /// Reopen streams after checkpoint file (datasets).
    void reopenStreams();

    /**
     * @brief   Sample all streams.
     * @warning In the GPU implementation, no data is flushed on disk (just data is sampled).
     */
    void sampleStreams();
    /// Flush streams to disk - only raw streams.
    void flushRawStreams();

    /// Post-process all streams and flush them to the file.
    void postProcessStreams();
    /// Checkpoint streams.
    void checkpointStreams();

    /// Close all streams,  apply post-processing if necessary, flush data and close.
    void closeStreams();
    /// Free all streams - destroy them.
    void freeStreams();

  protected:

   private:
    /**
     * @brief Create a new output stream.
     *
     * @param [in] matrixContainer  - Name of the HDF5 dataset or group
     * @param [in] sampledMatrixIdx - Code id of the matrix
     * @param [in] fileObjectName   - Name of the HDF5 dataset or group
     * @param [in] reduceOp         - Reduce operator
     *
     * @return New output stream with defined links
     *
     */
    BaseOutputStream* createOutputStream(MatrixContainer&                       matrixContainer,
                                         const MatrixContainer::MatrixIdx       sampledMatrixIdx,
                                         const MatrixName&                      fileObjectName,
                                         const BaseOutputStream::ReduceOperator reduceOp);

    /// Map with output streams.
    std::map<OutputStreamIdx, BaseOutputStream*> mContainer;
}; // end of OutputStreamContainer
//----------------------------------------------------------------------------------------------------------------------

#endif	/* OUTPUT_STREAM_CONTAINER_H */
