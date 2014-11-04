/**
 * @file        BaseOutputHDF5Stream.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 *
 * @brief       The header file of the class saving RealMatrix data into
 *              the output HDF5 file.
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        11 July      2012, 10:30 (created) \n
 *              04 November  2014, 17:01 (revised)
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

#ifndef BASEOUTPUTHDF5STREAM_H
#define BASEOUTPUTHDF5STREAM_H

#include <string>
#include <cstring>
#include <vector>
#include <stdexcept>

#include "../RealMatrix.h"
#include "../LongMatrix.h"
#include "../../HDF5/HDF5_File.h"


#include "../../CUDA/CUDAImplementations.h"




using namespace std;

/**
 * @class TBaseOutputHDF5Stream
 * @brief Output stream for sensor data. It stores time series every timestep
 */
class TBaseOutputHDF5Stream {
    public:
        /**
         * @enum TOutputHDF5StreamReductionOperator
         * @brief How to aggregate data \n
         *        roNONE - store actual data (time series)
         *        roRMS  - calculate root mean square \n
         *        roMAX  - store maximum
         *        roMIN  - store minimum
         */
        enum TReductionOperator
        {
            roNONE, roRMS, roMAX, roMIN
        };

        /**
         * Constructor - there is no sensor mask by default!
         * it links the HDF5 dataset, source (sampled matrix) and the reduction
         * operator together. The constructor DOES NOT allocate memory because the
         * size of the sensor mask is not known at the time the instance of
         * the class is being created.
         *
         * @param [in] HDF5_File           - Handle to the HDF5 (output) file
         * @param [in] HDF5_RootObjectName - The root object that stores the sample
         *                                   data (dataset or group)
         * @param [in] SourceMatrix        - The source matrix (only real matrices
         *                                   are supported)
         * @param [in] ReductionOp         - Reduction operator
         * @param [in] BufferToReuse       - An external buffer can be used to line
         *                                   up the grid points
         */
        TBaseOutputHDF5Stream(THDF5_File&              HDF5_File,
                              const char*              HDF5_RootObjectName,
                              TRealMatrix&             SourceMatrix,
                              const TReductionOperator ReductionOp,
                              float*                   BufferToReuse = NULL)
            : HDF5_File          (HDF5_File),
              HDF5_RootObjectName(NULL),
              SourceMatrix       (SourceMatrix),
              ReductionOp        (ReductionOp),
              BufferReuse        (BufferToReuse != NULL),
              BufferSize         (0),
              StoreBuffer        (BufferToReuse)
    {
      // copy the dataset name (just for sure)
      this->HDF5_RootObjectName = new char[strlen(HDF5_RootObjectName)];
      strcpy(this->HDF5_RootObjectName, HDF5_RootObjectName);

            cuda_implementations = CUDAImplementations::GetInstance();

        };

        /// Destructor
        virtual ~TBaseOutputHDF5Stream()
        {
            delete [] HDF5_RootObjectName;
        };


        /// Create a HDF5 stream and allocate data for it - a virtual method
        virtual void Create() = 0;

        /// Reopen the output stream after restart
        virtual void Reopen() = 0;

        /// Sample data into buffer, apply reduction or flush to disk - based on a sensor mask
        virtual void Sample() = 0;

        /// Apply post-processing on the buffer and flush it to the file
        virtual void PostProcess();

        /// Checkpoint the stream
        virtual void Checkpoint() = 0;

        /// Close stream (apply post-processing if necessary, flush data and close)
        virtual void Close() = 0;

    protected:
        /// Default constructor not allowed
        TBaseOutputHDF5Stream();
        /// Copy constructor not allowed
        TBaseOutputHDF5Stream(const TBaseOutputHDF5Stream & src);
        /// Operator = not allowed (we don't want any data movements)
        TBaseOutputHDF5Stream & operator = (const TBaseOutputHDF5Stream & src);

        /// A generic function to allocate memory - not used in the base class
        virtual void AllocateMemory();
        /// A generic function to free memory - not used in the base class
        virtual void FreeMemory();

        /// HDF5 file handle
        THDF5_File& HDF5_File;

        /// Dataset name
        char* HDF5_RootObjectName;

        /// Source matrix to be sampled
        TRealMatrix& SourceMatrix;

        /// HDF5 dataset handle
        hid_t HDF5_Dataset_id;
        /// Reduction operator
        const TReductionOperator ReductionOp;

        /// Position in the dataset
        TDimensionSizes Position;

        /// if true, the container reuses e.g. Temp_1_RS3D, Temp_2_RS3D, Temp_3_RS3D
        bool    BufferReuse;
        /// Buffer size
        size_t  BufferSize;
        /// Temporary buffer for store - only if Buffer Reuse = false!
        float * StoreBuffer;

        /// chunk size of 4MB in number of float elements
        static const size_t ChunkSize_4MB = 1048576;

        /// The minimum number of elements to start sampling in parallel (4MB)
        static const size_t MinGridpointsToSampleInParallel = 1048576;


        CUDAImplementations* cuda_implementations;

};// end of TBaseOutputHDF5Stream

#endif /* BASEOUTPUTHDF5STREAM_H */

