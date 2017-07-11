/**
 * @file        WholeDomainOutputHDF5Stream.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file of the class saving whole RealMatrix into the output HDF5 file,
 *              e.g. p_max_all.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        28 August   2014, 10:20 (created)
 *              07 July     2017, 18:58 (revised)
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

#ifndef WHOLE_DOMAIN_OUTPUT_HDF5_STREAM_H
#define WHOLE_DOMAIN_OUTPUT_HDF5_STREAM_H

#include <OutputHDF5Streams/BaseOutputHDF5Stream.h>

/**
 * @class TWholeDomainOutputHDF5Stream
 * @brief   Output stream for quantities sampled in the whole domain.
 * @details Output stream for quantities sampled in the whole domain.
 *          The data is stored in a single dataset (aggregated quantities only).
 */
class TWholeDomainOutputHDF5Stream : public TBaseOutputHDF5Stream
{
  public:
    /// Default constructor not allowed.
    TWholeDomainOutputHDF5Stream() = delete;
    /// Constructor.
    TWholeDomainOutputHDF5Stream(THDF5_File&           file,
                                 TMatrixName&          datasetName,
                                 TRealMatrix&          sourceMatrix,
                                 const TReduceOperator reduceOp);
    /// Destructor.
    virtual ~TWholeDomainOutputHDF5Stream();

     /// operator= is not allowed.
    TWholeDomainOutputHDF5Stream& operator=(const TWholeDomainOutputHDF5Stream&) = delete;


    /// Create a HDF5 stream and allocate data for it.
    virtual void Create();

    /// Reopen the output stream after restart and reload data.
    virtual void Reopen();

    /// Sample data (copy from GPU memory and then flush - no overlapping implemented!)
    virtual void Sample();

    /// Flush data to disk (from raw streams only) - empty routine (no overlapping implemented)
    virtual void FlushRaw() {};

    /// Apply post-processing on the buffer and flush it to the file.
    virtual void PostProcess();

    //Checkpoint the stream and close.
    virtual void Checkpoint();

    /// Close stream (apply post-processing if necessary, flush data and close).
    virtual void Close();

  protected:
      /// Flush the buffer to the file.
    virtual void FlushBufferToFile();

    /// Handle to a HDF5 dataset.
    hid_t  dataset;

    /// Time step to store (N/A for aggregated).
    size_t sampledTimeStep;
};// end of TWholeDomainOutputHDF5Stream
//--------------------------------------------------------------------------------------------------

#endif /* WHOLEDOMAINOUTPUTHDF5STREAM_H */

