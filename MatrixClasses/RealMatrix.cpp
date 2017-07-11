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
 *              28 June      2017, 15:15 (revised)
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

//------------------------------------------------------------------------------------------------//
//------------------------------------------ Constants -------------------------------------------//
//------------------------------------------------------------------------------------------------//


//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//
/**
 * Constructor.
 * @param [in] dimensionSizes - Dimension sizes of the matrix
 */
TRealMatrix::TRealMatrix(const TDimensionSizes& dimensionSizes)
         : TBaseFloatMatrix()
{
  InitDimensions(dimensionSizes);
  AllocateMemory();
}// end of TRealMatrix
//--------------------------------------------------------------------------------------------------

/**
 * Destructor.
 */
TRealMatrix::~TRealMatrix()
{
  FreeMemory();
}// end of ~TRealMatrix
//--------------------------------------------------------------------------------------------------

/**
 * Read data data from HDF5 file (only from the root group).
 *
 * @param [in] file      - HDF5 file
 * @param [in] matrixName - HDF5 dataset name
 *
 * @throw ios::failure if error occurred.
 */
void TRealMatrix::ReadDataFromHDF5File(THDF5_File&  file,
                                       TMatrixName& matrixName)
{
  // test matrix datatype
  if (file.ReadMatrixDataType(file.GetRootGroup(), matrixName) != THDF5_File::TMatrixDataType::FLOAT)
  {
    throw std::ios::failure(TLogger::FormatMessage(ERR_FMT_MATRIX_NOT_FLOAT, matrixName.c_str()));
  }
  // read matrix domain type
  if (file.ReadMatrixDomainType(file.GetRootGroup(), matrixName) != THDF5_File::TMatrixDomainType::REAL)
  {
    throw std::ios::failure(TLogger::FormatMessage(ERR_FMT_MATRIX_NOT_REAL, matrixName.c_str()));
  }

  // Read matrix
  file.ReadCompleteDataset(file.GetRootGroup(), matrixName, dimensionSizes, hostData);
}// end of LoadDataFromMatlabFile
//--------------------------------------------------------------------------------------------------

/**
 * Write data to HDF5 file (only from the root group).
 *
 * @param [in] file             - HDF5 file
 * @param [in] matrixName       - HDF5 Matrix name
 * @param [in] compressionLevel - Compression level
 *
 * @throw ios::failure if an error occurred.
 */
void TRealMatrix::WriteDataToHDF5File(THDF5_File&  file,
                                      TMatrixName& matrixName,
                                      const size_t compressionLevel)
{
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

  hid_t dataset = file.CreateFloatDataset(file.GetRootGroup(),
                                          matrixName,
                                          dimensionSizes,
                                          chunks,
                                          compressionLevel);

  file.WriteHyperSlab(dataset, TDimensionSizes(0, 0, 0), dimensionSizes, hostData);

  file.CloseDataset(dataset);

  // Write data and domain type
  file.WriteMatrixDataType  (file.GetRootGroup(), matrixName, THDF5_File::TMatrixDataType::FLOAT);
  file.WriteMatrixDomainType(file.GetRootGroup(), matrixName, THDF5_File::TMatrixDomainType::REAL);
}// end of WriteDataToHDF5File
//--------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Set necessary dimensions and auxiliary variables.
 *
 * @param [in] dimensionSizes - 3D Dimension sizes
 */
void TRealMatrix::InitDimensions(const TDimensionSizes& dimensionSizes)
{
  this->dimensionSizes = dimensionSizes;

  nElements = dimensionSizes.nx * dimensionSizes.ny * dimensionSizes.nz;

  nAllocatedElements = nElements;

  dataRowSize  = dimensionSizes.nx;
  dataSlabSize = dimensionSizes.nx * dimensionSizes.ny;
}// end of SetDimensions
//-------------------------------------------------------------------------------------------------/

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//

