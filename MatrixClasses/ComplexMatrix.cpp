/**
 * @file        ComplexMatrix.cpp
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file with the class for complex matrices.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        11 July     2011, 14:02 (created) \n
 *              18 July     2017, 15:20 (revised)
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

ComplexMatrix::ComplexMatrix(const DimensionSizes& dimensionSizes)
  : BaseFloatMatrix()
{
  initDimensions(dimensionSizes);
  allocateMemory();
} // end of ComplexMatrixData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor.
 */
ComplexMatrix::~ComplexMatrix()
{
  freeMemory();
}// end of ComplexMatrix
//----------------------------------------------------------------------------------------------------------------------


/**
 * Read data from HDF5 file (do some basic checks). Only from the root group.
 */
void ComplexMatrix::readData(THDF5_File& file,
                             MatrixName& matrixName)
{
  // check data type
  if (file.ReadMatrixDataType(file.GetRootGroup(), matrixName) != THDF5_File::TMatrixDataType::FLOAT)
  {
    throw std::ios::failure(Logger::formatMessage(kErrFmtMatrixNotFloat, matrixName.c_str()));
  }

  // check domain type
  if (file.ReadMatrixDomainType(file.GetRootGroup(), matrixName) != THDF5_File::TMatrixDomainType::COMPLEX)
  {
    throw std::ios::failure(Logger::formatMessage(kErrFmtMatrixNotComplex, matrixName.c_str()));
  }

  // Initialise dimensions
  DimensionSizes complexDims = mDimensionSizes;
  complexDims.nx = 2 * complexDims.nx;

  // Read data from the file
  file.ReadCompleteDataset(file.GetRootGroup(), matrixName, complexDims, mHostData);
}// end of readData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write data to HDF5 file (only from the root group).
 */
void ComplexMatrix::writeData(THDF5_File&  file,
                              MatrixName& matrixName,
                              const size_t compressionLevel)
{
  // set dimensions and chunks
  DimensionSizes complexDims = mDimensionSizes;
  complexDims.nx = 2 * complexDims.nx;

  DimensionSizes chunks = complexDims;
  complexDims.nz = 1;

  // create a dataset
  hid_t dataset = file.CreateFloatDataset(file.GetRootGroup(),
                                          matrixName,
                                          complexDims,
                                          chunks,
                                          compressionLevel);
 // Write write the matrix at once.
  file.WriteHyperSlab(dataset, DimensionSizes(0, 0, 0), mDimensionSizes, mHostData);
  file.CloseDataset(dataset);

 // Write data and domain type
  file.WriteMatrixDataType(file.GetRootGroup()  , matrixName, THDF5_File::TMatrixDataType::FLOAT);
  file.WriteMatrixDomainType(file.GetRootGroup(), matrixName, THDF5_File::TMatrixDomainType::COMPLEX);
}// end of writeData
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------- Private methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Initialize matrix dimension sizes.
 */
void ComplexMatrix::initDimensions(const DimensionSizes& dimensionSizes)
{
  mDimensionSizes = dimensionSizes;

  mSize = dimensionSizes.nx * dimensionSizes.ny * dimensionSizes.nz;

  mRowSize  = 2 * dimensionSizes.nx;
  mSlabSize = 2 * dimensionSizes.nx * dimensionSizes.ny;
  // compute actual necessary memory sizes
  mCapacity = 2 * mSize;

}// end of initDimensions
//----------------------------------------------------------------------------------------------------------------------
