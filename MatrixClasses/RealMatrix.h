/**
 * @file        RealMatrix.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the class for real matrices.
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        11 July      2011, 10:30 (created) \n
 *              04 November  2014, 13:45 (revised)
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

#ifndef REALMATRIXDATA_H
#define REALMATRIXDATA_H

#include "../MatrixClasses/BaseFloatMatrix.h"
#include "../MatrixClasses/LongMatrix.h"

#include "../Utils/DimensionSizes.h"

class TComplexMatrix;

/**
 * @class TRealMatrix
 * @brief The class for real matrices
 */
class TRealMatrix : public TBaseFloatMatrix{
    public:

        /// Constructor
        TRealMatrix(struct TDimensionSizes DimensionSizes);

        /// Destructor
        virtual ~TRealMatrix() { FreeMemory(); };

        /// Read data from the HDF5 file
        virtual void ReadDataFromHDF5File(THDF5_File& HDF5_File,
                                          const char* MatrixName);

        /// Write data into the HDF5 file
        virtual void WriteDataToHDF5File(THDF5_File& HDF5_File,
                                         const char* MatrixName,
                                         const int CompressionLevel);

        /**
         * @brief operator []
         * @param index - 1D index
         * @return an element
         */
        float& operator [](const size_t& index) {
            return pMatrixData[index];
        };

        /**
         * @brief Get element from 3D matrix
         * @param X - X dimension
         * @param Y - Y dimension
         * @param Z - Z dimension
         * @return  an element
         */
        inline float&  GetElementFrom3D(const size_t X,
                                        const size_t Y,
                                        const size_t Z)
        {
            return pMatrixData[Z * p2DDataSliceSize + Y * pDataRowSize +  X];
        };

    protected:

        /// Init dimension
        virtual void InitDimensions(struct TDimensionSizes DimensionSizes);

        /// Default constructor is not allowed for public
        TRealMatrix() : TBaseFloatMatrix() {};

        /// Copy constructor not allowed for public
        TRealMatrix(const TRealMatrix& orig);

    private:
        /// Number of elements to get 4MB block of data
        static const size_t ChunkSize_1D_4MB   = 1048576; //(4MB)
        /// Number of elements to get 1MB block of data
        static const size_t ChunkSize_1D_1MB   =  262144; //(1MB)
        /// Number of elements to get 256KB block of data
        static const size_t ChunkSize_1D_256KB =   65536; //(256KB)

};// end of class TRealMatrix
//----------------------------------------------------------------------------

#endif	/* REALMATRIXDATA_H */



