/**
 * @file        BaseIndexMatrix.cpp
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the base class for 64b-wide integers implemented
 *              as size_t datatype.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        26 July     2011, 14:17 (created) \n
 *              22 July     2016, 13:49 (revised)
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

#include <immintrin.h>

#include <MatrixClasses/BaseIndexMatrix.h>
#include <Utils/DimensionSizes.h>
#include <Logger/ErrorMessages.h>


//------------------------------------------------------------------------------------------------//
//------------------------------------------ CONSTANTS -------------------------------------------//
//------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//


/**
 * Default constructor
 */
TBaseIndexMatrix::TBaseIndexMatrix() : TBaseMatrix(),
                                       totalElementCount(0),
                                       totalAllocatedElementCount(0),
                                       dimensionSizes(),
                                       dataRowSize(0),
                                       dataSlabSize(0),
                                       matrixData(nullptr),
                                       deviceMatrixData(nullptr)
{

}// end of TBaseIndexMatrix
//--------------------------------------------------------------------------------------------------

/**
 *  Zero all allocated elements.
 */
void TBaseIndexMatrix::ZeroMatrix()
{
  #pragma omp parallel for schedule (static)
  for (size_t i = 0; i < totalAllocatedElementCount; i++)
  {
    matrixData[i] = size_t(0);
  }
}// end of ZeroMatrix
//--------------------------------------------------------------------------------------------------

/**
 * Copy data from CPU -> GPU (Host -> Device).
 */
void TBaseIndexMatrix::CopyToDevice()
{
  checkCudaErrors(cudaMemcpy(deviceMatrixData,
                             matrixData,
                             totalAllocatedElementCount * sizeof(size_t),
                             cudaMemcpyHostToDevice));

}// end of CopyToDevice
//--------------------------------------------------------------------------------------------------

/**
 * Copy data from GPU -> CPU (Device -> Host).
 */
void TBaseIndexMatrix::CopyFromDevice()
{
  checkCudaErrors(cudaMemcpy(matrixData,
                             deviceMatrixData,
                             totalAllocatedElementCount * sizeof(size_t),
                             cudaMemcpyDeviceToHost));
}// end of CopyFromDevice
//--------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Memory allocation based on the total number of elements. \n
 *
 * CPU memory is aligned by the DATA_ALIGNMENT and then registered as pinned and zeroed.
 * The GPU memory is allocated on GPU but not zeroed (no reason).
 */
void TBaseIndexMatrix::AllocateMemory()
{
  //size of memory to allocate
  size_t sizeInBytes = totalAllocatedElementCount * sizeof(size_t);

  matrixData = static_cast<size_t*>(_mm_malloc(sizeInBytes, DATA_ALIGNMENT));

  if (!matrixData)
  {
    throw bad_alloc();
  }

  // Register Host memory (pin in memory)
  checkCudaErrors(cudaHostRegister(matrixData, sizeInBytes, cudaHostRegisterPortable));

  checkCudaErrors(cudaMalloc<size_t>(&deviceMatrixData, sizeInBytes));
  if (!deviceMatrixData)
  {
    throw bad_alloc();
  }
}// end of AllocateMemory
//--------------------------------------------------------------------------------------------------

/**
 * Free memory.
 */
void TBaseIndexMatrix::FreeMemory()
{
  if (matrixData)
  {
    cudaHostUnregister(matrixData);
    _mm_free(matrixData);
  }
  matrixData = nullptr;

  // Free GPU memory
  if (deviceMatrixData)
  {
    checkCudaErrors(cudaFree(deviceMatrixData));
  }
  deviceMatrixData = nullptr;
}// end of FreeMemory
//--------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//
