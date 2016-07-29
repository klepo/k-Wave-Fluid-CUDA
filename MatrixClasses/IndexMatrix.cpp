/**
 * @file        IndexMatrix.cpp
 *
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the class for 64b integer matrices.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        26 July     2011, 15:16 (created) \n
 *              29 July     2016, 16:53 (revised)
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

#include <iostream>

#include <MatrixClasses/IndexMatrix.h>
#include <Logger/Logger.h>

//------------------------------------------------------------------------------------------------//
//------------------------------------------ CONSTANTS -------------------------------------------//
//------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Constructor allocating memory.
 *
 * @param [in] dimensionSizes - Dimension sizes
 */
TIndexMatrix::TIndexMatrix(const TDimensionSizes& dimensionSizes)
        : TBaseIndexMatrix()
{
  this->dimensionSizes = dimensionSizes;

  nElements = dimensionSizes.nx * dimensionSizes.ny * dimensionSizes.nz;

  nAllocatedElements = nElements;

  rowSize  = dimensionSizes.nx;
  slabSize = dimensionSizes.nx * dimensionSizes.ny;

  AllocateMemory();
}// end of TIndexMatrix
//--------------------------------------------------------------------------------------------------

/**
 * Destructor.
 */
TIndexMatrix::~TIndexMatrix()
{
  FreeMemory();
}
//--------------------------------------------------------------------------------------------------

/**
 * Read data from HDF5 file (only from the root group).
 *
 * @param [in] file       - HDF5 file handle
 * @param [in] matrixName - HDF5 dataset name
 *
 * @throw ios:failure if error occurs.
 */
void TIndexMatrix::ReadDataFromHDF5File(THDF5_File&  file,
                                        TMatrixName& matrixName)
{
  if (file.ReadMatrixDataType(file.GetRootGroup(), matrixName) != THDF5_File::LONG)
  {
    throw std::ios::failure(TLogger::FormatMessage(ERR_FMT_MATRIX_NOT_INDEX, matrixName.c_str()));
  }

  if (file.ReadMatrixDomainType(file.GetRootGroup(),matrixName) != THDF5_File::REAL)
  {
    throw std::ios::failure(TLogger::FormatMessage(ERR_FMT_MATRIX_NOT_REAL,matrixName.c_str()));
  }

  file.ReadCompleteDataset(file.GetRootGroup(), matrixName, dimensionSizes, hostData);
}// end of LoadDataFromMatlabFile
//--------------------------------------------------------------------------------------------------

/**
 * Write data to HDF5 file.
 *
 * @param [in] file             - HDF5 file handle
 * @param [in] matrixName       - HDF5 dataset name
 * @param [in] compressionLevel - compression level
 *
 * @throw ios:failure if error occurs.
 */
void TIndexMatrix::WriteDataToHDF5File(THDF5_File&  file,
                                       TMatrixName&  matrixName,
                                       const size_t compressionLevel)
{
  // set chunks - may be necessary for long index based sensor masks
  TDimensionSizes chunks = dimensionSizes;
  chunks.nz = 1;

  //1D matrices
  if ((dimensionSizes.ny == 1) && (dimensionSizes.nz == 1))
  {
    // Chunk = 4MB
    if (dimensionSizes.nx > (4 * CHUNK_SIZE_1D_4MB))
    {
      chunks.nx = CHUNK_SIZE_1D_4MB;
    }
    else if (dimensionSizes.nx > (4 * CHUNK_SIZE_1D_1MB))
    {
      chunks.nx = CHUNK_SIZE_1D_1MB;
    }
    else if (dimensionSizes.nx > (4 * CHUNK_SIZE_1D_256KB))
    {
      chunks.nx = CHUNK_SIZE_1D_256KB;
    }
  }

  // create dataset and write a slab
  hid_t dataset = file.CreateIndexDataset(file.GetRootGroup(),
                                          matrixName,
                                          dimensionSizes,
                                          chunks,
                                          compressionLevel);

  file.WriteHyperSlab(dataset, TDimensionSizes(0, 0, 0), dimensionSizes, hostData);

  file.CloseDataset(dataset);

  // write data and domain types
  file.WriteMatrixDataType(file.GetRootGroup(),   matrixName, THDF5_File::LONG);
  file.WriteMatrixDomainType(file.GetRootGroup(), matrixName, THDF5_File::REAL);
}// end of WriteDataToHDF5File
//--------------------------------------------------------------------------------------------------


/**
 * Get the top left corner of the index-th cuboid. Cuboids are stored as 6-tuples (two 3D
 * coordinates). This gives the first three coordinates.
 *
 * @param [in] index - Index of the cuboid
 * @return The top left corner
 */
TDimensionSizes TIndexMatrix::GetTopLeftCorner(const size_t& index) const
{
  size_t x =  hostData[6 * index    ];
  size_t y =  hostData[6 * index + 1];
  size_t z =  hostData[6 * index + 2];

  return TDimensionSizes(x, y, z);
}// end of GetTopLeftCorner
//--------------------------------------------------------------------------------------------------

/**
 * Get the top bottom right of the index-th cuboid. Cuboids are stored as 6-tuples (two 3D
 * coordinates). This gives the first three coordinates. This routine works only on the CPU side.
 *
 * @param [in] index -Index of the cuboid
 * @return The bottom right corner
*/
TDimensionSizes TIndexMatrix::GetBottomRightCorner(const size_t& index) const
{
  size_t x =  hostData[6 * index + 3];
  size_t y =  hostData[6 * index + 4];
  size_t z =  hostData[6 * index + 5];

  return TDimensionSizes(x, y, z);
}// end of GetBottomRightCorner
//-------------------------------------------------------------------------------------------------


/**
 * Recompute indeces, MATLAB -> C++.
 */
void TIndexMatrix::RecomputeIndicesToCPP()
{
  #pragma omp parallel for if (nElements > 1e5)
  for (size_t i = 0; i < nElements; i++)
  {
    hostData[i]--;
  }
}// end of RecomputeIndices
//--------------------------------------------------------------------------------------------------

/**
 * Recompute indeces, C++ -> MATLAB.
 */
void TIndexMatrix::RecomputeIndicesToMatlab()
{
  #pragma omp parallel for if (nElements > 1e5)
  for (size_t i = 0; i < nElements; i++)
  {
    hostData[i]++;
  }
}// end of RecomputeIndicesToMatlab
//--------------------------------------------------------------------------------------------------

/**
 * Get total number of elements in all cuboids to be able to allocate output file.
 *
 * @return Total sampled grid points
 */
size_t TIndexMatrix::GetTotalNumberOfElementsInAllCuboids() const
{
  size_t elementSum = 0;
  for (size_t cuboidIdx = 0; cuboidIdx < dimensionSizes.ny; cuboidIdx++)
  {
    elementSum += (GetBottomRightCorner(cuboidIdx) - GetTopLeftCorner(cuboidIdx)).GetElementCount();
  }

  return elementSum;
}// end of GetTotalNumberOfElementsInAllCuboids
//--------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//
