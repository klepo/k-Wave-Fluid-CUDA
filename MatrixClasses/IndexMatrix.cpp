/**
 * @file        LongMatrix.cpp
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the class for 64b integer
 *              matrices.
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        26 July     2011, 15:16 (created) \n
 *              04 November 2014, 17:14 (revised)
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

#include "../MatrixClasses/IndexMatrix.h"
#include "../Utils/ErrorMessages.h"

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
 * Constructor
 * @param [in] DimensionSizes - Dimension sizes
 */
TLongMatrix::TLongMatrix(struct TDimensionSizes DimensionSizes)
    : TBaseIndexMatrix()
{

    pDimensionSizes = DimensionSizes;

    pTotalElementCount = pDimensionSizes.X *
                         pDimensionSizes.Y *
                         pDimensionSizes.Z;

    pTotalAllocatedElementCount = pTotalElementCount;

    pDataRowSize       = pDimensionSizes.X;

    p2DDataSliceSize   = pDimensionSizes.X * pDimensionSizes.Y;

    AllocateMemory();
}// end of TRealMatrixData
//---------------------------------------------------------------------------

/**
 * Read data from HDF5 file
 * @throw ios:failure if there's an error
 *
 * @param HDF5_File - HDF5 file handle
 * @param MatrixName  - HDF5 dataset name
 */
void TLongMatrix::ReadDataFromHDF5File(THDF5_File& HDF5_File,
                                       const char* MatrixName)
{

    if (HDF5_File.ReadMatrixDataType(HDF5_File.GetRootGroup(), MatrixName) !=
            THDF5_File::hdf5_mdt_long){
        char ErrorMessage[256];
        sprintf(ErrorMessage,Matrix_ERR_FMT_MatrixNotLong,MatrixName);
        throw ios::failure(ErrorMessage);
    }

    if (HDF5_File.ReadMatrixDomainType(HDF5_File.GetRootGroup(),MatrixName) !=
            THDF5_File::hdf5_mdt_real){
        char ErrorMessage[256];
        sprintf(ErrorMessage,Matrix_ERR_FMT_MatrixNotReal,MatrixName);
        throw ios::failure(ErrorMessage);
    }

    HDF5_File.ReadCompleteDataset(HDF5_File.GetRootGroup(),
                                  MatrixName,
                                  pDimensionSizes,
                                  pMatrixData);

}// end of LoadDataFromMatlabFile
//----------------------------------------------------------------------------

/**
 * Recompute indeces, MATLAB -> C++
 */
void TLongMatrix::RecomputeIndices()
{

    for (size_t i = 0; i< pTotalElementCount; i++){
        pMatrixData[i]--;
    }

}// end of RecomputeIndices
//----------------------------------------------------------------------------

/**
 * Write data to HDF5 file
 * @throw ios:failure
 *
 * @param [in] HDF5_File - HDF5 file handle
 * @param [in] MatrixName  - HDF5 Dataset name
 * @param [in] CompressionLevel - Compression level
 */
void TLongMatrix::WriteDataToHDF5File(THDF5_File& HDF5_File,
                                      const char* MatrixName,
                                      const int CompressionLevel)
{
    // set chunks - may be necessary for long index based sensor masks
    TDimensionSizes Chunks = pDimensionSizes;
    Chunks.Z = 1;

    //1D matrices
    if ((pDimensionSizes.Y == 1) && (pDimensionSizes.Z == 1))
    {
        // Chunk = 4MB
        if (pDimensionSizes.X > 4 * ChunkSize_1D_4MB)
        {
            Chunks.X = ChunkSize_1D_4MB;
        }
        else if (pDimensionSizes.X > 4 * ChunkSize_1D_1MB)
        {
            Chunks.X = ChunkSize_1D_1MB;
        }
        else if (pDimensionSizes.X > 4 * ChunkSize_1D_256KB)
        {
            Chunks.X = ChunkSize_1D_256KB;
        }
    }

    hid_t HDF5_Dataset_id = HDF5_File.CreateIndexDataset(HDF5_File.GetRootGroup(),
                                                        MatrixName,
                                                        pDimensionSizes,
                                                        Chunks,
                                                        CompressionLevel);

    HDF5_File.WriteHyperSlab(HDF5_Dataset_id,
                             TDimensionSizes(0, 0, 0),
                             pDimensionSizes,
                             pMatrixData);

    HDF5_File.CloseDataset(HDF5_Dataset_id);

    HDF5_File.WriteMatrixDataType(HDF5_File.GetRootGroup(),
                                  MatrixName,
                                  THDF5_File::hdf5_mdt_long);

  HDF5_File.WriteMatrixDomainType(HDF5_File.GetRootGroup(),
                                  MatrixName,
                                  THDF5_File::hdf5_mdt_real);
}// end of WriteDataToHDF5File
//----------------------------------------------------------------------------

//--------------------------------------------------------------------------//
//                            Implementation                                //
//                           protected methods                              //
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
//                            Implementation                                //
//                            private methods                               //
//--------------------------------------------------------------------------//

