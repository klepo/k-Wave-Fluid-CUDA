/**
 * @file        BaseFloatMatrix.cpp
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the base class for single precisions floating
 *              point numbers (floats).
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        11 July      2011, 12:13 (created) \n
 *              10 August    2016, 11:54 (revised)
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


#include <cstring>
#include <immintrin.h>

#include <MatrixClasses/BaseFloatMatrix.h>
#include <Utils/DimensionSizes.h>
#include <Logger/Logger.h>


using std::string;

//------------------------------------------------------------------------------------------------//
//------------------------------------------ Constants -------------------------------------------//
//------------------------------------------------------------------------------------------------//


//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Default constructor
 */
TBaseFloatMatrix::TBaseFloatMatrix(): TBaseMatrix(),
                                      nElements(0),
                                      nAllocatedElements(0),
                                      dimensionSizes(),
                                      dataRowSize(0),
                                      dataSlabSize(0),
                                      hostData(nullptr),
                                      deviceData(nullptr)
{

}// end of TBaseFloatMatrix
//--------------------------------------------------------------------------------------------------

/**
 * Zero all allocated elements in parallel. \n
 * Also work as the first touch strategy on NUMA machines.
 */
void TBaseFloatMatrix::ZeroMatrix()
{
  #pragma omp parallel for schedule (static)
  for (size_t i = 0; i < nAllocatedElements; i++)
  {
    hostData[i] = 0.0f;
  }
}// end of ZeroMatrix
//--------------------------------------------------------------------------------------------------

/**
 * Divide a scalar by the elements of matrix.
 *
 * @param [in] scalar - Scalar to be divided by evey element of the array
 */
void TBaseFloatMatrix::ScalarDividedBy(const float scalar)
{
  #pragma omp parallel for schedule (static)
  for (size_t i = 0; i < nAllocatedElements; i++)
  {
    hostData[i] = scalar / hostData[i];
  }
}// end of ScalarDividedBy
//-------------------------------------------------------------------------------------------------

/**
 * Copy data from CPU -> GPU (Host -> Device).
 * The transfer is synchronous (there is nothing to overlap with in the code)
 */
void TBaseFloatMatrix::CopyToDevice()
{
  checkCudaErrors(cudaMemcpy(deviceData,
                             hostData,
                             nAllocatedElements * sizeof(float),
                             cudaMemcpyHostToDevice));
}// end of CopyToDevice
//--------------------------------------------------------------------------------------------------

/**
 * Copy data from GPU -> CPU (Device -> Host).
 * The transfer is synchronous (there is nothing to overlap with in the code)
 */
void TBaseFloatMatrix::CopyFromDevice()
{
  checkCudaErrors(cudaMemcpy(hostData,
                             deviceData,
                             nAllocatedElements * sizeof(float),
                             cudaMemcpyDeviceToHost));
}// end of CopyFromDevice
//--------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Memory allocation based on the total number of elements. \n
 * CPU memory is aligned by the DATA_ALIGNMENT and then registered as pinned and zeroed.
 */
void TBaseFloatMatrix::AllocateMemory()
{
  // Size of memory to allocate
  size_t sizeInBytes = nAllocatedElements * sizeof(float);

  // Allocate CPU memory
  hostData = static_cast<float*> (_mm_malloc(sizeInBytes, DATA_ALIGNMENT));
  if (!hostData)
  {
    throw std::bad_alloc();
  }

  // Register Host memory (pin in memory)
  checkCudaErrors(cudaHostRegister(hostData, sizeInBytes, cudaHostRegisterPortable));

  // Allocate memory on the GPU
  if ((cudaMalloc<float>(&deviceData, sizeInBytes) != cudaSuccess) || (!deviceData))
  {
    throw std::bad_alloc();
  }
  // This has to be done for simulations based on input sources
  ZeroMatrix();
}//end of AllocateMemory
//--------------------------------------------------------------------------------------------------

/**
 * Free memory.
 * Both on the CPU and GPU side
 */
void TBaseFloatMatrix::FreeMemory()
{
  // Free CPU memory
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
}//end of FreeMemory
//--------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//

