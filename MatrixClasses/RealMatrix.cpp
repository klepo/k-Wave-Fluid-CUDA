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
 *              19 July      2017, 12:15 (revised)
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
RealMatrix::RealMatrix(const DimensionSizes& dimensionSizes) :
  BaseFloatMatrix()
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
void RealMatrix::readData(THDF5_File&  file,
                          MatrixName& matrixName)
{
  // test matrix datatype
  if (file.ReadMatrixDataType(file.GetRootGroup(), matrixName) != THDF5_File::TMatrixDataType::FLOAT)
  {
    throw std::ios::failure(Logger::formatMessage(kErrFmtMatrixNotFloat, matrixName.c_str()));
  }
  // read matrix domain type
  if (file.ReadMatrixDomainType(file.GetRootGroup(), matrixName) != THDF5_File::TMatrixDomainType::REAL)
  {
    throw std::ios::failure(Logger::formatMessage(kErrFmtMatrixNotReal, matrixName.c_str()));
  }

  // Read matrix
  file.ReadCompleteDataset(file.GetRootGroup(), matrixName, mDimensionSizes, mHostData);
}// end of readData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write data to HDF5 file (only from the root group).
 */
void RealMatrix::writeData(THDF5_File&  file,
                           MatrixName& matrixName,
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

  hid_t dataset = file.CreateFloatDataset(file.GetRootGroup(),
                                          matrixName,
                                          mDimensionSizes,
                                          chunks,
                                          compressionLevel);

  file.WriteHyperSlab(dataset, DimensionSizes(0, 0, 0), mDimensionSizes, mHostData);

  file.CloseDataset(dataset);

  // Write data and domain type
  file.WriteMatrixDataType  (file.GetRootGroup(), matrixName, THDF5_File::TMatrixDataType::FLOAT);
  file.WriteMatrixDomainType(file.GetRootGroup(), matrixName, THDF5_File::TMatrixDomainType::REAL);
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

