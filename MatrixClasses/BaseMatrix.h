/**
 * @file        BaseMatrix.h
 * @author      Jiri Jaros              \n
 *              CECS, ANU, Australia    \n
 *              jiri.jaros@anu.edu.au   \n
 *
 * @brief       The header file of the common ancestor of all matrix classes.
 *              A pure abstract class
 *
 * @version     kspaceFirstOrder3D 2.13
 * @date        11 July 2012, 11:34      (created) \n
 *              17 September 2012, 15:35 (revised)
 *
 * @section License
 * This file is part of the C++ extension of the k-Wave Toolbox
 * (http://www.k-wave.org).\n
 * Copyright (C) 2012 Jiri Jaros and Bradley Treeby
 *
 * This file is part of the k-Wave. k-Wave is free software: you can
 * redistribute it and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation, either version
 * 3 of the License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with k-Wave. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef BASEMATRIX_H
#define BASEMATRIX_H

#include "../Utils/DimensionSizes.h"
#include "../HDF5/HDF5_File.h"

/**
 * @class TBaseMatrix
 * @brief Abstract base class, the common ancestor defining the common interface
 *      and allowing derived classes to be allocated, freed and loaded from the file
 *      using the Matrix container
 *
 */
class TBaseMatrix {
    public:
        /// Default constructor
        TBaseMatrix() {};

        /// Get dimension sizes of the matrix
        virtual struct TDimensionSizes GetDimensionSizes() const  = 0;

        /// Get total element count of the matrix
        virtual size_t GetTotalElementCount()              const = 0;

        /// Get total allocated element count (might differ from total element
        // count used for the simulation because of padding).
        virtual size_t GetTotalAllocatedElementCount()     const  = 0;


        /**
         * Read matrix from the HDF5 file
         * @param [in] HDF5_File    - Handle to the HDF5 file
         * @param [in] MatrixName   - HDF5 dataset name to read from
         */
        virtual void ReadDataFromHDF5File(THDF5_File& HDF5_File,
                                          const char* MatrixName) {};

        /**
         * Write data into the HDF5 file
         * @param HDF5_File         - Handle to the HDF5 file
         * @param MatrixName        - HDF5 dataset name to write to
         * @param CompressionLevel  - Compression level for the HDF5 dataset
         */
        virtual void WriteDataToHDF5File(THDF5_File& HDF5_File,
                                         const char* MatrixName,
                                         const int CompressionLevel) {};

#if (defined (CUDA_VERSION) || defined(OPENCL_VERSION))
        virtual void SyncroniseToGPUDevice() {};
        virtual void SyncroniseToCPUHost() {};
#endif

        /// Destructor
        virtual ~TBaseMatrix() {};

    protected:

};// end of TBaseMatrix

#endif /* BASEMATRIX_H */

