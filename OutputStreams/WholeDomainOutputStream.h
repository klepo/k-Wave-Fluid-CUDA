/**
 * @file      WholeDomainOutputStream.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file of the class saving whole RealMatrix into the output HDF5 file, e.g. p_max_all.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      28 August    2014, 10:20 (created) \n
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

#ifndef WHOLE_DOMAIN_OUTPUT_STREAM_H
#define WHOLE_DOMAIN_OUTPUT_STREAM_H

#include <OutputStreams/BaseOutputStream.h>

/**
 * @class   WholeDomainOutputStream
 * @brief   Output stream for quantities sampled in the whole domain.
 *
 * Output stream for quantities sampled in the whole domain. The data is stored in a single dataset
 * (aggregated quantities only).
 */
class WholeDomainOutputStream : public BaseOutputStream
{
  public:
    /// Default constructor not allowed.
    WholeDomainOutputStream() = delete;
    /**
     * @brief Constructor - links the HDF5 dataset and SourceMatrix
     * @param [in] file         - HDF5 file to write the output to.
     * @param [in] datasetName  - The name of the HDF5 group containing datasets for particular cuboids.
     * @param [in] sourceMatrix - Source matrix to be sampled.
     * @param [in] reduceOp     - Reduce operator.
     */
    WholeDomainOutputStream(Hdf5File&            file,
                            MatrixName&          datasetName,
                            RealMatrix&          sourceMatrix,
                            const ReduceOperator reduceOp);

    /// Copy constructor not allowed.
    WholeDomainOutputStream(const WholeDomainOutputStream&) = delete;

    /**
     * @brief Destructor.
     *
     * If the file is still opened, it applies the post processing and flush the data.
     * Then, the object memory is freed and the object destroyed.
     */
    virtual ~WholeDomainOutputStream();

     /// operator= is not allowed.
    WholeDomainOutputStream& operator=(const WholeDomainOutputStream&) = delete;


    /// Create a HDF5 stream and allocate data for it.
    virtual void create();

    /// Reopen the output stream after restart and reload data.
    virtual void reopen();

    /// Sample data (copy from GPU memory and then flush - no overlapping implemented!)
    virtual void sample();

    /// Flush data to disk (from raw streams only) - empty routine (no overlapping implemented).
    virtual void flushRaw() {};

    /// Apply post-processing on the buffer and flush it to the file.
    virtual void postProcess();

    //Checkpoint the stream and close.
    virtual void checkpoint();

    /// Close stream (apply post-processing if necessary, flush data and close).
    virtual void close();

  protected:
      /// Flush the buffer to the file.
    virtual void flushBufferToFile();

    /// Handle to a HDF5 dataset.
    hid_t  mDataset;

    /// Time step to store (N/A for aggregated).
    size_t mSampledTimeStep;
};// end of WholeDomainOutputHDF5Stream
//----------------------------------------------------------------------------------------------------------------------

#endif /* WHOLE_DOMAIN_OUTPUT_STREAM_H */

