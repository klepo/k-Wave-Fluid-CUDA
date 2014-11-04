/**
 * @file        OutputStreamContainer.h
 * @author      Jiri Jaros & Beau Johnston \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file defining the output stream container.
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        27 August   2014, 10:22 (created)
 *              04 November 2014, 17:20 (revised)
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

#ifndef OUTPUTSTREAMCONTAINER_H
#define	OUTPUTSTREAMCONTAINER_H

#include <string.h>
#include <map>

#include "./MatrixContainer.h"
#include "./OutputHDF5Stream/BaseOutputHDF5Stream.h"

#include "../Utils/MatrixNames.h"
#include "../Utils/DimensionSizes.h"

/**
 * @class TOutputStreamContainer
 * @brief A container for output streams
 */
class TOutputStreamContainer
{
  public:
    /// Constructor
    TOutputStreamContainer() {};
    /// Destructor
    virtual ~TOutputStreamContainer();

    /// Get size of the container
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
    TBaseOutputHDF5Stream & operator [] (const TMatrixID MatrixID)
    {
      return (* (OutputStreamContainer[MatrixID]));
    };

    /// Create all streams
    void AddStreamsIntoContainer(TMatrixContainer & MatrixContainer);

    /// Create all streams
    void CreateStreams();
    /// Reopen streams after checkpoint file
    void ReopenStreams();

    /// Sample all streams
    void SampleStreams();
    /// Post-process all streams and flush them to the file
    void PostProcessStreams();
    /// Checkpoint streams
    void CheckpointStreams();

    /// Close all streams
    void CloseStreams();

    /// Free all streams - destroy them
    void FreeAllStreams();

  protected:
    // Create a new output stream
    TBaseOutputHDF5Stream* CreateNewOutputStream(
            TMatrixContainer& MatrixContainer,
            const TMatrixID SampledMatrixID,
            const char* HDF5_DatasetName,
            const TBaseOutputHDF5Stream::TReductionOperator ReductionOp,
            float* BufferToReuse = NULL);

    /// Copy constructor not allowed for public
    TOutputStreamContainer(const TOutputStreamContainer &);
    /// Operator = not allowed for public
    TOutputStreamContainer & operator = (TOutputStreamContainer &);

  private:
    /// Output stream map
    typedef map < TMatrixID, TBaseOutputHDF5Stream * > TOutputStreamMap;
    /// Map with output streams
    TOutputStreamMap OutputStreamContainer;

}; // end of TOutputStreamContainer
//------------------------------------------------------------------------------

#endif	/*  OUTPUTSTREAMCONTAINER_H */
