/**
 * @file        IndexMatrix.h
 *
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the class for 64b integer matrices.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        26 July     2011, 15:16 (created) \n
 *              11 July     2017, 16:45 (revised)
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

#ifndef INDEX_MATRIX_H
#define	INDEX_MATRIX_H

#include <MatrixClasses/BaseIndexMatrix.h>
#include <Utils/DimensionSizes.h>

/**
 * @class TIndexMatrix
 * @brief The class for 64b unsigned integers (indices). It is used for  sensor_mask_index or
 * sensor_corners_mask to get the address of sampled voxels.
 *
 * @details The class for 64b unsigned integers (indices). It is used for sensor_mask_index or
 * sensor_corners_mask to get the address of sampled voxels. Stores data both GPU and CPU side.
 */
class TIndexMatrix : public TBaseIndexMatrix
{
  public:
    /// Default constructor not allowed.
    TIndexMatrix() = delete;
    /// Constructor allocating memory.
    TIndexMatrix(const DimensionSizes& dimensionSizes);
    /// Copy constructor not allowed.
    TIndexMatrix(const TIndexMatrix&);
    /// Destructor.
    virtual ~TIndexMatrix();

    /// Operator= is not allowed.
    TIndexMatrix& operator= (const TIndexMatrix&);

    /// Read data from the HDF5 file.
    virtual void ReadDataFromHDF5File(THDF5_File&  file,
                                      MatrixName& matrixName);
    /// Write data into the HDF5 file.
    virtual void WriteDataToHDF5File(THDF5_File&  file,
                                     MatrixName& matrixName,
                                     const size_t compressionLevel);

    /**
     * @brief   Operator [].
     * @details Operator [].
     * @param [in] index - 1D index into the matrix
     * @return Value of the index
     */
    inline size_t& operator[](const size_t& index)
    {
      return hostData[index];
    };

    /**
     * @brief Operator [], constant version
     * @details Operator [], constant version
     * @param [in] index - 1D index into the matrix
     * @return Value of the index
     */
    inline const size_t& operator[](const size_t& index) const
    {
      return hostData[index];
    };

    /// Get the top left corner of the index-th cuboid.
    DimensionSizes GetTopLeftCorner(const size_t& index) const;
    /// Get the bottom right corner of the index-th cuboid
    DimensionSizes GetBottomRightCorner(const size_t& index) const;

    ///  Recompute indices MATALAB->C++.
    void RecomputeIndicesToCPP();
    ///  Recompute indices C++ -> MATLAB.
    void RecomputeIndicesToMatlab();

    /// Get the total number of elements to be sampled within all cuboids.
    size_t GetTotalNumberOfElementsInAllCuboids() const;

  protected:

  private:
    /// Number of elements to get 4MB block of data.
    static constexpr size_t CHUNK_SIZE_1D_4MB   = 1048576; //(4MB)
    /// Number of elements to get 1MB block of data.
    static constexpr size_t CHUNK_SIZE_1D_1MB   =  262144; //(1MB)
    /// Number of elements to get 256KB block of data.
    static constexpr size_t CHUNK_SIZE_1D_256KB =   65536; //(256KB)

};// end of TIndexMatrix
//--------------------------------------------------------------------------------------------------
#endif /* 	INDEX_MATRIX_H */

