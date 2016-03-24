/**
 * @file        OutputStreamContainer.h
 * @author      Jiri Jaros & Beau Johnston \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file defining the output stream container.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        04 December  2014, 11:00 (created)
 *              24 March     2016, 17:06 (revised)
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

#ifndef OUTPUT_STREAM_CONTAINER_H
#define	OUTPUT_STREAM_CONTAINER_H

#include <string.h>
#include <map>

#include <Containers/MatrixContainer.h>
#include <OutputHDF5Streams/BaseOutputHDF5Stream.h>

#include <Utils/MatrixNames.h>
#include <Utils/DimensionSizes.h>


/**
 * @enum TOutputStreamID
 * @brief Output streams identifiers in k-Wave
 * @warning the order of IDs is mandatory! it determines the order of sampling
 *          and is used for hiding the PCI-E latency
 */
enum TOutputStreamID
{
  // raw quantities are sampled at first - to allow some degree of asynchronous copies
  p_sensor_raw,  ux_sensor_raw, uy_sensor_raw, uz_sensor_raw,
  ux_shifted_sensor_raw, uy_shifted_sensor_raw, uz_shifted_sensor_raw,

  // then we sample aggregated quantities
  p_sensor_rms, p_sensor_max, p_sensor_min,
  p_sensor_max_all, p_sensor_min_all,

  ux_sensor_rms, uy_sensor_rms, uz_sensor_rms,
  ux_sensor_max, uy_sensor_max, uz_sensor_max,
  ux_sensor_min, uy_sensor_min, uz_sensor_min,

  ux_sensor_max_all, uy_sensor_max_all, uz_sensor_max_all,
  ux_sensor_min_all, uy_sensor_min_all, uz_sensor_min_all,
};


/**
 * @class TOutputStreamContainer
 * @brief A container for output streams.
 * @details The output stream container maintains matrices used for sampling data.
 * These may or may not require some scratch place or reuse temp matrices.
 */
class TOutputStreamContainer
{
  public:
    /// Constructor.
    TOutputStreamContainer() {};
    /// Destructor - no need for virtual destructor (no polymorphism).
    ~TOutputStreamContainer();

    /// Get size of the container.
    size_t size() const
    {
      return OutputStreamContainer.size();
    };

    /// Is the container empty?
    bool empty() const
    {
      return OutputStreamContainer.empty();
    };

    /**
     * @brief Operator []
     * @details Operator []
     * @param MatrixID
     * @return
     */
    TBaseOutputHDF5Stream& operator [] (const TOutputStreamID OutputStreamID)
    {
      return (* (OutputStreamContainer[OutputStreamID]));
    };

    /// Create all streams in container (no file manipulation).
    void AddStreamsIntoContainer(TMatrixContainer & MatrixContainer);

    /// Create all streams - opens the datasets.
    void CreateStreams();
    /// Reopen streams after checkpoint file (datasets).
    void ReopenStreams();

    /// Sample all streams (only sample, no disk operations).
    void SampleStreams();
    /// Flush streams to disk - only raw streams!.
    void FlushRawStreams();

    /// Post-process all streams and flush them to the file.
    void PostProcessStreams();
    /// Checkpoint streams.
    void CheckpointStreams();

    /// Close all streams.
    void CloseStreams();
    /// Free all streams - destroy them.
    void FreeStreams();

  protected:
    /// Create a new output stream
    TBaseOutputHDF5Stream* CreateNewOutputStream(TMatrixContainer& MatrixContainer,
                                                 const TMatrixID   SampledMatrixID,
                                                 const char*      HDF5_DatasetName,
                                                 const TBaseOutputHDF5Stream::TReductionOperator ReductionOp);

    /// Copy constructor not allowed for public.
    TOutputStreamContainer(const TOutputStreamContainer&);
    /// Operator = not allowed for public.
    TOutputStreamContainer & operator = (TOutputStreamContainer&);

  private:
    /// Output stream map.
    typedef map<TOutputStreamID, TBaseOutputHDF5Stream*> TOutputStreamMap;

    /// Map with output streams.
    TOutputStreamMap OutputStreamContainer;

}; // end of TOutputStreamContainer
//------------------------------------------------------------------------------

#endif	/*  OUTPUTSTREAMCONTAINER_H */
