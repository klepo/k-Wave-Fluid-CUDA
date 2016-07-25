
/**
 * @file        ComplexMatrix.cpp
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
 *              21 July     2016, 15:55 (revised)
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
#include <Logger/ErrorMessages.h>

//------------------------------------------------------------------------------------------------//
//------------------------------------------ CONSTANTS -------------------------------------------//
//------------------------------------------------------------------------------------------------//


//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Constructor.
 * @param [in] dimensionSizes - Dimension sizes of the matrix
 */

TComplexMatrix::TComplexMatrix(const TDimensionSizes& dimensionSizes)
        : TBaseFloatMatrix()
{
  InitDimensions(dimensionSizes);
  AllocateMemory();
} // end of TComplexMatrixData
//--------------------------------------------------------------------------------------------------

/**
 * Destructor.
 */
TComplexMatrix::~TComplexMatrix()
{
  FreeMemory();
}// end of TComplexMatrix
//--------------------------------------------------------------------------------------------------


/**
 * Read data from HDF5 file (do some basic checks). Only from the root group.
 *
 * @param [in] file   - HDF5 file
 * @param [in] matrixName  - HDF5 dataset name
 *
 * * @throw ios::failure when there is a problem
 */
void TComplexMatrix::ReadDataFromHDF5File(THDF5_File& file,
                                          const char* matrixName)
{
  // check data type
  if (file.ReadMatrixDataType(file.GetRootGroup(), matrixName) != THDF5_File::FLOAT)
  {
    char errMsg[256];
    snprintf(errMsg, 256, ERR_FMT_MATRIX_NOT_FLOAT, matrixName);
    throw ios::failure(errMsg);
  }

  // check domain type
  if (file.ReadMatrixDomainType(file.GetRootGroup(), matrixName) != THDF5_File::COMPLEX)
  {
    char errMsg[256];
    snprintf(errMsg, 256, ERR_FMT_MATRIX_NOT_COMPLEX, matrixName);
    throw ios::failure(errMsg);
  }

  // Initialise dimensions
  TDimensionSizes complexDims = dimensionSizes;
  complexDims.nx = 2 * complexDims.nx;

  // Read data from the file
  file.ReadCompleteDataset(file.GetRootGroup(), matrixName, complexDims, matrixData);
}// end of LoadDataFromMatlabFile
//--------------------------------------------------------------------------------------------------

/**
 * Write data to HDF5 file (only from the root group).
 *
 * @param [in] file             - HDF5 file handle
 * @param [in] matrixName       - HDF5 dataset name
 * @param [in] compressionLevel - Compression level for the dataset
 *
 * @throw ios::failure an exception what the operation fails
 */
void TComplexMatrix::WriteDataToHDF5File(THDF5_File& file,
                                         const char* matrixName,
                                         const size_t compressionLevel)
{
  // set dimensions and chunks
  TDimensionSizes complexDims = dimensionSizes;
  complexDims.nx = 2 * complexDims.nx;

  TDimensionSizes chunks = complexDims;
  complexDims.nz = 1;

  // create a dataset
  hid_t dataset = file.CreateFloatDataset(file.GetRootGroup(),
                                          matrixName,
                                          complexDims,
                                          chunks,
                                          compressionLevel);
 // Write write the matrix at once.
  file.WriteHyperSlab(dataset, TDimensionSizes(0, 0, 0), dimensionSizes, matrixData);
  file.CloseDataset(dataset);

 // Write data and domain type
  file.WriteMatrixDataType(file.GetRootGroup()  , matrixName, THDF5_File::FLOAT);
  file.WriteMatrixDomainType(file.GetRootGroup(), matrixName, THDF5_File::COMPLEX);
}// end of WriteDataToHDF5File
//--------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Initialize matrix dimension sizes.
 *
 * @param [in] dimensionSizes - Dimension sizes of the matrix
 */
void TComplexMatrix::InitDimensions(const TDimensionSizes& dimensionSizes)
{

  this->dimensionSizes = dimensionSizes;

  totalElementCount = dimensionSizes.nx * dimensionSizes.ny * dimensionSizes.nz;

  dataRowSize  = 2 * dimensionSizes.nx;
  dataSlabSize = 2 * dimensionSizes.nx * dimensionSizes.ny;
  // compute actual necessary memory sizes
  totalAllocatedElementCount = 2 * totalElementCount;

}// end of InitDimensions
//--------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//

