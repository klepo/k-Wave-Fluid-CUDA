/**
 * @file        IndexMatrix.h
 * @author      Jiri Jaros & Beau Johnston \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the class for 64b integer matrices.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        26 July     2011, 15:16 (created) \n
 *              13 November 2014, 15:16 (revised)
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

#ifndef INDEX_MATRIX_DATA_H
#define	INDEX_MATRIX_DATA_H

#include <MatrixClasses/BaseIndexMatrix.h>
#include <Utils/DimensionSizes.h>

/**
 * @class TIndexMatrix
 * @brief The class for 64b unsigned integers (indices). It is used for
 *  sensor_mask_index or sensor_corners_mask to get the address of sampled voxels.
 *
 * @details The class for 64b unsigned integers (indices). It is used for
 *  sensor_mask_index or sensor_corners_mask to get the address of sampled voxels.
 *  Stores data both GPU and CPU side.
 *
 */
class TIndexMatrix : public TBaseIndexMatrix
{
  public:
    /// Constructor allocating memory.
    TIndexMatrix(const TDimensionSizes & DimensionSizes);

    /// Destructor
    virtual ~TIndexMatrix()
    {
      FreeMemory();
    };

    /// Read data from the HDF5 file.
    virtual void ReadDataFromHDF5File(THDF5_File& HDF5_File,
                                      const char* MatrixName);
    /// Write data into the HDF5 file.
    virtual void WriteDataToHDF5File(THDF5_File& HDF5_File,
                                     const char* MatrixName,
                                     const size_t CompressionLevel);

   /**
     * Operator [].
     * @param [in] index - 1D index into the matrix
     * @return Value of the index
     */
    size_t& operator [](const size_t& index)
    {
      return pMatrixData[index];
    };

    /**
     * Operator [], constant version
     * @param [in] index - 1D index into the matrix
     * @return Value of the index
     */
    const size_t & operator [](const size_t& index) const
    {
      return pMatrixData[index];
    };

    /**
     * @brief Get the top left corner of the index-th cuboid.
     * @details Get the top left corner of the index-th cuboid. Cuboids are
     *          stored as 6-tuples (two 3D coordinates).
     *          This gives the first three coordinates
     * @param [in] index - Id of the corner
     * @return the top left corner
     */
    TDimensionSizes GetTopLeftCorner(const size_t& index) const
    {
      size_t X =  pMatrixData[6 * index   ];
      size_t Y =  pMatrixData[6 * index +1];
      size_t Z =  pMatrixData[6 * index +2];

      return TDimensionSizes(X, Y, Z);
    };

    /**
     * @brief   Get the bottom right corner of the index-th cuboid
     * @details Get the top bottom right of the index-th cuboid. Cuboids are
     *          stored as 6-tuples (two 3D coordinates).
     *          This gives the first three coordinates.
     *          This routine works only on the CPU side.
     * @param [in] index -Id of the corner
     * @return the bottom right corner
     */
    TDimensionSizes GetBottomRightCorner(const size_t & index) const
    {
      size_t X =  pMatrixData[6 * index + 3];
      size_t Y =  pMatrixData[6 * index + 4];
      size_t Z =  pMatrixData[6 * index + 5];

      return TDimensionSizes(X, Y, Z);
    };

    ///  Recompute indices MATALAB->C++.
    void RecomputeIndicesToCPP();

    ///  Recompute indices C++ -> MATLAB.
    void RecomputeIndicesToMatlab();

    /// Get the total number of elements to be sampled within all cuboids.
    size_t GetTotalNumberOfElementsInAllCuboids() const;


  protected:
    /// Default constructor not allowed for public.
    TIndexMatrix() {};

    /// Copy constructor not allowed for public.
    TIndexMatrix(const TIndexMatrix& src);

    /// Operator =  not allowed for public.
    TIndexMatrix& operator = (const TIndexMatrix& src);

  private:
    /// Number of elements to get 4MB block of data.
    static const size_t ChunkSize_1D_4MB   = 1048576; //(4MB)
    /// Number of elements to get 1MB block of data.
    static const size_t ChunkSize_1D_1MB   =  262144; //(1MB)
    /// Number of elements to get 256KB block of data.
    static const size_t ChunkSize_1D_256KB =   65536; //(256KB)

};// end of TIndexMatrixData
//------------------------------------------------------------------------------
#endif /* INDEX_MATRIX_DATA_H */

