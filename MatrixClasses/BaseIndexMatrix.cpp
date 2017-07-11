/**
 * @file        BaseIndexMatrix.cpp
 *
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
 *              11 July     2017, 16:44 (revised)
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
#include <Logger/Logger.h>


//------------------------------------------------------------------------------------------------//
//------------------------------------------ Constants -------------------------------------------//
//------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//


/**
 * Default constructor
 */
TBaseIndexMatrix::TBaseIndexMatrix() : TBaseMatrix(),
                                       nElements(0),
                                       nAllocatedElements(0),
                                       dimensionSizes(),
                                       rowSize(0),
                                       slabSize(0),
                                       hostData(nullptr),
                                       deviceData(nullptr)
{

}// end of TBaseIndexMatrix
//--------------------------------------------------------------------------------------------------

/**
 *  Zero all allocated elements.
 */
void TBaseIndexMatrix::ZeroMatrix()
{
  #pragma omp parallel for schedule (static)
  for (size_t i = 0; i < nAllocatedElements; i++)
  {
    hostData[i] = size_t(0);
  }
}// end of ZeroMatrix
//--------------------------------------------------------------------------------------------------

/**
 * Copy data from CPU -> GPU (Host -> Device).
 */
void TBaseIndexMatrix::CopyToDevice()
{
  checkCudaErrors(cudaMemcpy(deviceData,
                             hostData,
                             nAllocatedElements * sizeof(size_t),
                             cudaMemcpyHostToDevice));

}// end of CopyToDevice
//--------------------------------------------------------------------------------------------------

/**
 * Copy data from GPU -> CPU (Device -> Host).
 */
void TBaseIndexMatrix::CopyFromDevice()
{
  checkCudaErrors(cudaMemcpy(hostData,
                             deviceData,
                             nAllocatedElements * sizeof(size_t),
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
  size_t sizeInBytes = nAllocatedElements * sizeof(size_t);

  hostData = static_cast<size_t*>(_mm_malloc(sizeInBytes, kDataAlignment));

  if (!hostData)
  {
    throw std::bad_alloc();
  }

  // Register Host memory (pin in memory)
  checkCudaErrors(cudaHostRegister(hostData, sizeInBytes, cudaHostRegisterPortable));

  if ((cudaMalloc<size_t>(&deviceData, sizeInBytes) != cudaSuccess) || (!deviceData))
  {
    throw std::bad_alloc();
  }
}// end of AllocateMemory
//--------------------------------------------------------------------------------------------------

/**
 * Free memory.
 */
void TBaseIndexMatrix::FreeMemory()
{
  if (hostData)
  {
    cudaHostUnregister(hostData);
    _mm_free(hostData);
  }
  hostData = nullptr;

  // Free GPU memory
  if (deviceData)
  {
    checkCudaErrors(cudaFree(deviceData));
  }
  deviceData = nullptr;
}// end of FreeMemory
//--------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//
