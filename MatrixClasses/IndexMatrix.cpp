/**
 * @file      IndexMatrix.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file containing the class for 64b integer index matrices.
 *
 * @version   kspaceFirstOrder3D 3.5
 *
 * @date      26 July      2011, 15:16 (created) \n
 *            16 August    2017, 13:54 (revised)
 *
 * @copyright Copyright (C) 2017 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the C++ extension of the [k-Wave Toolbox](http://www.k-wave.org).
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

#include <MatrixClasses/IndexMatrix.h>
#include <Logger/Logger.h>

//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Constants -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


/**
 * Constructor allocating memory.
 */
IndexMatrix::IndexMatrix(const DimensionSizes& dimensionSizes)
  : BaseIndexMatrix()
{
  mDimensionSizes = dimensionSizes;

  mSize = dimensionSizes.nx * dimensionSizes.ny * dimensionSizes.nz;

  mCapacity = mSize;

  mRowSize  = dimensionSizes.nx;
  mSlabSize = dimensionSizes.nx * dimensionSizes.ny;

  allocateMemory();
}// end of IndexMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor.
 */
IndexMatrix::~IndexMatrix()
{
  freeMemory();
}// end of ~IndexMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Read data from HDF5 file (only from the root group).
 */
void IndexMatrix::readData(Hdf5File&   file,
                           MatrixName& matrixName)
{
  if (file.readMatrixDataType(file.getRootGroup(), matrixName) != Hdf5File::MatrixDataType::kLong)
  {
    throw std::ios::failure(Logger::formatMessage(kErrFmtMatrixNotIndex, matrixName.c_str()));
  }

  if (file.readMatrixDomainType(file.getRootGroup(),matrixName) != Hdf5File::MatrixDomainType::kReal)
  {
    throw std::ios::failure(Logger::formatMessage(kErrFmtMatrixNotReal,matrixName.c_str()));
  }

  file.readCompleteDataset(file.getRootGroup(), matrixName, mDimensionSizes, mHostData);
}// end of readData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write data to HDF5 file.
 */
void IndexMatrix::writeData(Hdf5File&    file,
                            MatrixName&  matrixName,
                            const size_t compressionLevel)
{
  // set chunks - may be necessary for long index based sensor masks
  DimensionSizes chunks = mDimensionSizes;
  chunks.nz = 1;

  //1D matrices
  if ((mDimensionSizes.ny == 1) && (mDimensionSizes.nz == 1))
  {
    // Chunk = 4MB
    if (mDimensionSizes.nx > (4 * kChunkSize1D4MB))
    {
      chunks.nx = kChunkSize1D4MB;
    }
    else if (mDimensionSizes.nx > (4 * kChunkSize1D1MB))
    {
      chunks.nx = kChunkSize1D1MB;
    }
    else if (mDimensionSizes.nx > (4 * kChunkSize1D256kB))
    {
      chunks.nx = kChunkSize1D256kB;
    }
  }

  // create dataset and write a slab
  hid_t dataset = file.createDataset(file.getRootGroup(),
                                     matrixName,
                                     mDimensionSizes,
                                     chunks,
                                     Hdf5File::MatrixDataType::kLong,
                                     compressionLevel);

  file.writeHyperSlab(dataset, DimensionSizes(0, 0, 0), mDimensionSizes, mHostData);

  file.closeDataset(dataset);

  // write data and domain types
  file.writeMatrixDataType(file.getRootGroup(),   matrixName, Hdf5File::MatrixDataType::kLong);
  file.writeMatrixDomainType(file.getRootGroup(), matrixName, Hdf5File::MatrixDomainType::kReal);
}// end of writeData
//----------------------------------------------------------------------------------------------------------------------


/**
 * Get the top left corner of the index-th cuboid. Cuboids are stored as 6-tuples (two 3D
 * coordinates). This gives the first three coordinates.
 */
DimensionSizes IndexMatrix::getTopLeftCorner(const size_t& index) const
{
  size_t x =  mHostData[6 * index    ];
  size_t y =  mHostData[6 * index + 1];
  size_t z =  mHostData[6 * index + 2];

  return DimensionSizes(x, y, z);
}// end of getTopLeftCorner
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get the top bottom right of the index-th cuboid. Cuboids are stored as 6-tuples (two 3D
 * coordinates). This gives the first three coordinates. This routine works only on the CPU side.
*/
DimensionSizes IndexMatrix::getBottomRightCorner(const size_t& index) const
{
  size_t x =  mHostData[6 * index + 3];
  size_t y =  mHostData[6 * index + 4];
  size_t z =  mHostData[6 * index + 5];

  return DimensionSizes(x, y, z);
}// end of GetBottomRightCorner
//----------------------------------------------------------------------------------------------------------------------


/**
 * Recompute indeces, MATLAB -> C++.
 */
void IndexMatrix::recomputeIndicesToCPP()
{
  #pragma omp parallel for if (mSize > 1e5)
  for (size_t i = 0; i < mSize; i++)
  {
    mHostData[i]--;
  }
}// end of recomputeIndicesToCPP
//----------------------------------------------------------------------------------------------------------------------

/**
 * Recompute indeces, C++ -> MATLAB.
 */
void IndexMatrix::recomputeIndicesToMatlab()
{
  #pragma omp parallel for if (mSize > 1e5)
  for (size_t i = 0; i < mSize; i++)
  {
    mHostData[i]++;
  }
}// end of recomputeIndicesToMatlab
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get total number of elements in all cuboids to be able to allocate output file.
 */
size_t IndexMatrix::getSizeOfAllCuboids() const
{
  size_t elementSum = 0;
  for (size_t cuboidIdx = 0; cuboidIdx < mDimensionSizes.ny; cuboidIdx++)
  {
    elementSum += (getBottomRightCorner(cuboidIdx) - getTopLeftCorner(cuboidIdx)).nElements();
  }

  return elementSum;
}// end of getSizeOfAllCuboids
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//
