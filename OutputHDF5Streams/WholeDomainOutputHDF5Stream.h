/**
 * @file        WholeDomainOutputHDF5Stream.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file of the class saving index data into
 *              the output HDF5 file.
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        28 August   2014, 10:20 (created)
 *              04 November 2014, 17:05 (revised)
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
#ifndef WHOLEDOMAINOUTPUTHDF5STREAM_H
#define WHOLEDOMAINOUTPUTHDF5STREAM_H

#include "./BaseOutputHDF5Stream.h"

/**
 * @class TWholeDomainOutputHDF5Stream
 * @brief Output stream for quantities sampled in the whole domain.
 *        The data is stored in a single dataset (aggregated quantities only)
 */
class TWholeDomainOutputHDF5Stream : public TBaseOutputHDF5Stream
{

  public:
    /// Constructor - links the HDF5 File, SourceMatrix, and SensorMask together
    TWholeDomainOutputHDF5Stream(THDF5_File&              HDF5_File,
                                 const char*              HDF5_DatasetName,
                                 TRealMatrix&             SourceMatrix,
                                 const TReductionOperator ReductionOp,
                                 float*                   BufferToReuse = NULL);

    /// Destructor
    virtual ~TWholeDomainOutputHDF5Stream();

    /// Create a HDF5 stream and allocate data for it
    virtual void Create();

    /// Reopen the output stream after restart and reload data
    virtual void Reopen();

    /// Sample data into buffer and apply reduction, or flush to disk - based on a sensor mask
    virtual void Sample();

    /// Apply post-processing on the buffer and flush it to the file
    virtual void PostProcess();

    //Checkpoint the stream and close
    virtual void Checkpoint();

    /// Close stream (apply post-processing if necessary, flush data and close)
    virtual void Close();

  protected:
      /// Flush the buffer to the file
    virtual void FlushBufferToFile();

    /// Handle to a HDF5 dataset
    hid_t  HDF5_DatasetId;

    /// Time step to store (N/A for aggregated)
    size_t SampledTimeStep;
};// end of TWholeDomainOutputHDF5Stream

#endif /* WHOLEDOMAINOUTPUTHDF5STREAM_H */

