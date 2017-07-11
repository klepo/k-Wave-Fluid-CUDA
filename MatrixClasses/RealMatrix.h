/**
 * @file        RealMatrix.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the class for real matrices.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        11 July      2011, 10:30 (created) \n
 *              11 July      2017, 14:43 (revised)
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

#ifndef REAL_MATRIX_H
#define REAL_MATRIX_H

#include <MatrixClasses/BaseFloatMatrix.h>
#include <Utils/DimensionSizes.h>

// Forward declaration
class TComplexMatrix;

/**
 * @class TRealMatrix
 * @brief   The class for real matrices
 * @details The class for real matrices (floats) on both CPU and GPU side
 */
class TRealMatrix : public TBaseFloatMatrix
{
  public:
    /// Default constructor is not allowed.
    TRealMatrix() = delete;
    /// Constructor.
    TRealMatrix(const TDimensionSizes& dimensionSizes);
    /// Copy constructor not allowed.
    TRealMatrix(const TRealMatrix&) = delete;
    /// Destructor.
    virtual ~TRealMatrix();

    /// Operator= is not allowed.
    TRealMatrix& operator=(const TRealMatrix&);

    /// Read data from the HDF5 file - only from the root group.
    virtual void ReadDataFromHDF5File(THDF5_File&  file,
                                      MatrixName& matrixName);

    /// Write data into the HDF5 file.
    virtual void WriteDataToHDF5File(THDF5_File&  file,
                                     MatrixName& matrixName,
                                     const size_t compressionLevel);

    /**
     * @brief  Operator [].
     * @details Operator [].
     * @param [in] index - 1D index
     * @return An element
     */
    inline float& operator[](const size_t& index)
    {
      return hostData[index];
    };

    /**
     * @brief   Operator [], constant version.
     * @details Operator [], constant version.
     * @param [in] index - 1D index
     * @return An element
     */
    inline const float& operator[](const size_t& index) const
    {
      return hostData[index];
    };

    /// Init dimension sizes.
    virtual void InitDimensions(const TDimensionSizes& dimensionSizes);

private:

   /// Number of elements to get 4MB block of data.
   static constexpr size_t CHUNK_SIZE_1D_4MB   = 1048576; //(4MB)
   /// Number of elements to get 1MB block of data.
   static constexpr size_t CHUNK_SIZE_1D_1MB   =  262144; //(1MB)
   /// Number of elements to get 256KB block of data.
   static constexpr size_t CHUNK_SIZE_1D_256KB =   65536; //(256KB)
};// end of class TRealMatrix
//--------------------------------------------------------------------------------------------------

#endif	/* REAL_MATRIX_H */