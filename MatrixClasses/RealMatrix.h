/**
 * @file      RealMatrix.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file containing the class for real matrices.
 *
 * @version   kspaceFirstOrder3D 3.5
 * @date      11 July      2011, 10:30 (created) \n
 *            16 August    2017, 13:54 (revised)
 *
 * @copyright Copyright (C) 2017 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the C++ extension of the k-Wave Toolbox [k-Wave Toolbox](http://www.k-wave.org).
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

#ifndef REAL_MATRIX_H
#define REAL_MATRIX_H

#include <MatrixClasses/BaseFloatMatrix.h>
#include <Utils/DimensionSizes.h>

// Forward declaration
class ComplexMatrix;

/**
 * @class   RealMatrix
 * @brief   The class for real matrices.
 * @details The class for real matrices (floats) on both CPU and GPU side.
 */
class RealMatrix : public BaseFloatMatrix
{
  public:
    /// Default constructor is not allowed.
    RealMatrix() = delete;
    /**
     * @brief Constructor.
     * @param [in] dimensionSizes - Dimension sizes of the matrix.
     */
    RealMatrix(const DimensionSizes& dimensionSizes);
    /// Copy constructor not allowed.
    RealMatrix(const RealMatrix&) = delete;
    /// Destructor.
    virtual ~RealMatrix();

    /// Operator= is not allowed.
    RealMatrix& operator=(const RealMatrix&);

    /**
     * @brief   Read matrix from HDF5 file.
     * @details Read matrix from HDF5 file.
     * @param [in] file       - Handle to the HDF5 file
     * @param [in] matrixName - HDF5 dataset name to read from
     * @throw ios::failure    - If error occurred.
     */
    virtual void readData(Hdf5File&   file,
                          MatrixName& matrixName);
    /**
     * @brief   Write data into HDF5 file.
     * @details Write data into HDF5 file.
     * @param [in] file             - Handle to the HDF5 file
     * @param [in] matrixName       - HDF5 dataset name to write to
     * @param [in] compressionLevel - Compression level for the HDF5 dataset
     * @throw ios::failure          - If an error occurred.
     */
    virtual void writeData(Hdf5File&    file,
                           MatrixName&  matrixName,
                           const size_t compressionLevel);

    /**
     * @brief  operator[].
     * @param [in] index - 1D index into the matrix.
     * @return An element of the matrix.
     */
    inline float&       operator[](const size_t& index)       { return mHostData[index]; };
    /**
     * @brief   Operator [], constant version.
     * @param [in] index - 1D index into the matrix.
     * @return An element of the matrix.
     */
    inline const float& operator[](const size_t& index) const { return mHostData[index]; };

  private:
    /**
     * @brief Initialize dimension sizes.
     * @param [in] dimensionSizes - Dimension sizes of the matrix.
     */
    void initDimensions(const DimensionSizes& dimensionSizes);

     /// Number of elements to get 4MB block of data.
     static constexpr size_t kChunkSize1D4MB   = 1048576; //(4MB)
     /// Number of elements to get 1MB block of data.
      static constexpr size_t kChunkSize1D1MB   =  262144; //(1MB)
      /// Number of elements to get 256KB block of data.
      static constexpr size_t kChunkSize1D256kB =   65536; //(256KB)
};// end of class RealMatrix
//----------------------------------------------------------------------------------------------------------------------

#endif	/* REAL_MATRIX_H */