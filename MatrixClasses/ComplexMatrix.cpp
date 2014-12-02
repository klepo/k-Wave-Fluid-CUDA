
/**
 * @file        ComplexMatrix.cpp
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file with the class for complex matrices.
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        11 July     2011, 14:02 (created) \n
 *              13 November 2014, 15:47 (revised)
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

#include <iostream>

#include <MatrixClasses/ComplexMatrix.h>
#include <Utils/ErrorMessages.h>

//--------------------------------------------------------------------------//
//                            Constants                                     //
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
//                            Definitions                                   //
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
//                            Implementation                                //
//                            public methods                                //
//--------------------------------------------------------------------------//

/**
 * Constructor.
 * @param [in] DimensionSizes - Dimension sizes
 */

TComplexMatrix::TComplexMatrix(const TDimensionSizes & DimensionSizes)
                      : TBaseFloatMatrix()
{
  InitDimensions(DimensionSizes);

  AllocateMemory();
} // end of TComplexMatrixData
//-----------------------------------------------------------------------------


/**
 * Read data from HDF5 file (do some basic checks). Only from the root group.
 * \throw ios::failure when there is a problem
 *
 * @param [in] HDF5_File   - HDF5 file
 * @param [in] MatrixName  - HDF5 dataset name
 */
void TComplexMatrix::ReadDataFromHDF5File(THDF5_File & HDF5_File,
                                          const char * MatrixName)
{
  // check data type
  if (HDF5_File.ReadMatrixDataType(HDF5_File.GetRootGroup(), MatrixName) != THDF5_File::hdf5_mdt_float)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, Matrix_ERR_FMT_MatrixNotFloat, MatrixName);
    throw ios::failure(ErrorMessage);
  }

  // check domain type
  if (HDF5_File.ReadMatrixDomainType(HDF5_File.GetRootGroup(), MatrixName) != THDF5_File::hdf5_mdt_complex)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, Matrix_ERR_FMT_MatrixNotComplex, MatrixName);
    throw ios::failure(ErrorMessage);
  }

  // Initialise dimensions
  TDimensionSizes ComplexDims = pDimensionSizes;
  ComplexDims.X = 2 * ComplexDims.X;

  // Read data from the file
  HDF5_File.ReadCompleteDataset(HDF5_File.GetRootGroup(),
                                MatrixName,
                                ComplexDims,
                                pMatrixData);
}// end of LoadDataFromMatlabFile
//------------------------------------------------------------------------------

/**
 * Write data to HDF5 file (only from the root group).
 * \throw ios::failure an exception what the operation fails
 *
 * @param [in] HDF5_File             - HDF5 file handle
 * @param [in] MatrixName            - HDF5 dataset name
 * @param [in] CompressionLevel      - Compression level for the dataset
 */
void TComplexMatrix::WriteDataToHDF5File(THDF5_File & HDF5_File,
                                         const char * MatrixName,
                                         const size_t CompressionLevel)
{
  // set dimensions and chunks
  TDimensionSizes ComplexDims = pDimensionSizes;
  ComplexDims.X = 2 * ComplexDims.X;

  TDimensionSizes Chunks = ComplexDims;
  ComplexDims.Z = 1;

  // create a dataset
  hid_t HDF5_Dataset_id = HDF5_File.CreateFloatDataset(HDF5_File.GetRootGroup(),
                                                       MatrixName,
                                                       ComplexDims,
                                                       Chunks,
                                                       CompressionLevel);
 // Write write the matrix at once.
  HDF5_File.WriteHyperSlab(HDF5_Dataset_id,
                           TDimensionSizes(0, 0, 0),
                           pDimensionSizes,
                           pMatrixData);
  HDF5_File.CloseDataset(HDF5_Dataset_id);

 // Write data and domain type
  HDF5_File.WriteMatrixDataType(HDF5_File.GetRootGroup(),
                                MatrixName,
                                THDF5_File::hdf5_mdt_float);

  HDF5_File.WriteMatrixDomainType(HDF5_File.GetRootGroup(),
                                  MatrixName,
                                  THDF5_File::hdf5_mdt_complex);
}// end of WriteDataToHDF5File
//---------------------------------------------------------------------------


//--------------------------------------------------------------------------//
//                            Implementation                                //
//                           protected methods                              //
//--------------------------------------------------------------------------//

/**
 * Initialize matrix dimension sizes.
 * @param [in] DimensionSizes
 */
void TComplexMatrix::InitDimensions(const TDimensionSizes & DimensionSizes)
{

  pDimensionSizes = DimensionSizes;

  pTotalElementCount = pDimensionSizes.X *
                       pDimensionSizes.Y *
                       pDimensionSizes.Z;

  pDataRowSize = (pDimensionSizes.X << 1);

  p2DDataSliceSize = (pDimensionSizes.X *
                      pDimensionSizes.Y) << 1;

  // compute actual necessary memory sizes
  pTotalAllocatedElementCount = pTotalElementCount << 1;

}// end of InitDimensions
//------------------------------------------------------------------------------


//--------------------------------------------------------------------------//
//                            Implementation                                //
//                            private methods                               //
//--------------------------------------------------------------------------//

