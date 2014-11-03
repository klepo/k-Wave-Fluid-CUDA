/**
 * @file        LongMatrix.h
 * @author      Jiri Jaros & Beau Johnston \n
 *              CECS, ANU, Australia       \n
 *              beau.johnston@anu.edu.au
 *
 * @brief       The header file containing the class for 64b integer matrices
 *
 * @version     kspaceFirstOrder3D 2.13
 * @date        26 July 2011, 15:16      (created) \n
 *              17 September 2012, 15:35 (revised)
 *
 * @section License This file is part of the C++ extension of the k-Wave
 * Toolbox (http://www.k-wave.org).\n Copyright (C) 2012 Jiri Jaros and Bradley
 * Treeby
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
 * along with k-Wave. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LONGMATRIXDATA_H
#define	LONGMATRIXDATA_H

#include "../MatrixClasses/BaseLongMatrix.h"
#include "../Utils/DimensionSizes.h"

/**
 * @class TLongMatrix
 * @brief The class for 64b integers. It is used for index mask into
 * float matrices
 */
class TLongMatrix : public TBaseLongMatrix{
    public:

        /// Constructor
        TLongMatrix(struct TDimensionSizes DimensionSizes);

        /// Destructor
        virtual ~TLongMatrix() { FreeMemory(); };

        /// Read data from the HDF5 file
        virtual void ReadDataFromHDF5File(THDF5_File& HDF5_File,
                                          const char* MatrixName);
        /// Write data into the HDF5 file
        virtual void WriteDataToHDF5File(THDF5_File& HDF5_File,
                                         const char* MatrixName,
                                         const int CompressionLevel);

        /**
         * Operator []
         * @param index - 1D index into the matrix
         * @return  Value of the index
         */
        size_t& operator [](const size_t& index)
        {
            return pMatrixData[index];
        };

        /**
         * Get element form the 3D matrix
         * @param X - X dimension
         * @param Y - Y dimension
         * @param Z - Z dimension
         * @return an alement
         */
        inline size_t&  GetElementFrom3D(const size_t X,
                                       const size_t Y,
                                       const size_t Z)
        {
            return pMatrixData[Z * p2DDataSliceSize + Y * pDataRowSize +  X];
        };

        ///  Recompute indices MATALAB->C++
        void RecomputeIndices();

    protected:
        /// Default constructor not allowed for public
        TLongMatrix()  : TBaseLongMatrix() {};

        /// Copy constructor not allowed for public
        TLongMatrix(const TLongMatrix& orig);

    private:
        /// Number of elements to get 4MB block of data
        static const size_t ChunkSize_1D_4MB   = 1048576; //(4MB)
        /// Number of elements to get 1MB block of data
        static const size_t ChunkSize_1D_1MB   =  262144; //(1MB)
        /// Number of elements to get 256KB block of data
        static const size_t ChunkSize_1D_256KB =   65536; //(256KB)

};// end of TLongMatrixData
//------------------------------------------------------------------------------
#endif /* LONGMATRIXDATA_H */

