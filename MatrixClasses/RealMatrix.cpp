/**
 * @file        RealMatrix.cpp
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the class for real matrices.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        11 July      2011, 10:30 (created) \n
 *              21 July      2017, 13:05 (revised)
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

#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>

#include <Logger/Logger.h>

//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Constants -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor.
 */
RealMatrix::RealMatrix(const DimensionSizes& dimensionSizes)
  : BaseFloatMatrix()
{
  initDimensions(dimensionSizes);
  allocateMemory();
}// end of RealMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor.
 */
RealMatrix::~RealMatrix()
{
  freeMemory();
}// end of ~RealMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Read data data from HDF5 file (only from the root group).
 */
void RealMatrix::readData(Hdf5File&   file,
                          MatrixName& matrixName)
{
  // test matrix datatype
  if (file.readMatrixDataType(file.getRootGroup(), matrixName) != Hdf5File::MatrixDataType::kFloat)
  {
    throw std::ios::failure(Logger::formatMessage(kErrFmtMatrixNotFloat, matrixName.c_str()));
  }
  // read matrix domain type
  if (file.readMatrixDomainType(file.getRootGroup(), matrixName) != Hdf5File::MatrixDomainType::kReal)
  {
    throw std::ios::failure(Logger::formatMessage(kErrFmtMatrixNotReal, matrixName.c_str()));
  }

  // Read matrix
  file.readCompleteDataset(file.getRootGroup(), matrixName, mDimensionSizes, mHostData);
}// end of readData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write data to HDF5 file (only from the root group).
 */
void RealMatrix::writeData(Hdf5File&    file,
                           MatrixName&  matrixName,
                           const size_t compressionLevel)
{
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

  hid_t dataset = file.createDataset(file.getRootGroup(),
                                     matrixName,
                                     mDimensionSizes,
                                     chunks,
                                     Hdf5File::MatrixDataType::kFloat,
                                     compressionLevel);

  file.writeHyperSlab(dataset, DimensionSizes(0, 0, 0), mDimensionSizes, mHostData);

  file.closeDataset(dataset);

  // Write data and domain type
  file.writeMatrixDataType  (file.getRootGroup(), matrixName, Hdf5File::MatrixDataType::kFloat);
  file.writeMatrixDomainType(file.getRootGroup(), matrixName, Hdf5File::MatrixDomainType::kReal);
}// end of writeData
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Set necessary dimensions and auxiliary variables.
 */
void RealMatrix::initDimensions(const DimensionSizes& dimensionSizes)
{
  mDimensionSizes = dimensionSizes;

  mSize = dimensionSizes.nx * dimensionSizes.ny * dimensionSizes.nz;

  mCapacity = mSize;

  mRowSize  = dimensionSizes.nx;
  mSlabSize = dimensionSizes.nx * dimensionSizes.ny;
}// end of initDimensions
//----------------------------------------------------------------------------------------------------------------------

