/**
 * @file      BaseIndexMatrix.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file containing the base class for 64b-wide integers implemented
 *            as size_t datatype.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      26 July      2011, 14:17 (created) \n
 *            06 March     2019, 13:19 (revised)
 *
 * @copyright Copyright (C) 2019 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the C++ extension of the [k-Wave Toolbox](http://www.k-wave.org).
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify it under the terms
 * of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 */

#include <immintrin.h>

#include <MatrixClasses/BaseIndexMatrix.h>
#include <Utils/DimensionSizes.h>
#include <Logger/Logger.h>


//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Constants -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


/**
 * Default constructor
 */
BaseIndexMatrix::BaseIndexMatrix()
  : BaseMatrix(),
    mSize(0), mCapacity(0),
    mDimensionSizes(),
    mRowSize(0), mSlabSize(0),
    mHostData(nullptr), mDeviceData(nullptr)
{

}// end of BaseIndexMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Zero all allocated elements.
 */
void BaseIndexMatrix::zeroMatrix()
{
  #pragma omp parallel for schedule (static)
  for (size_t i = 0; i < mCapacity; i++)
  {
    mHostData[i] = size_t(0);
  }
}// end of zeroMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data from CPU -> GPU (Host -> Device).
 */
void BaseIndexMatrix::copyToDevice()
{
  cudaCheckErrors(cudaMemcpy(mDeviceData, mHostData, mCapacity * sizeof(size_t), cudaMemcpyHostToDevice));
}// end of copyToDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data from GPU -> CPU (Device -> Host).
 */
void BaseIndexMatrix::copyFromDevice()
{
  cudaCheckErrors(cudaMemcpy(mHostData, mDeviceData, mCapacity * sizeof(size_t), cudaMemcpyDeviceToHost));
}// end of copyFromDevice
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Memory allocation based on the total number of elements. \n
 *
 * CPU memory is aligned by the kDataAlignment and then registered as pinned and zeroed.
 * The GPU memory is allocated on GPU but not zeroed (no reason).
 */
void BaseIndexMatrix::allocateMemory()
{
  //size of memory to allocate
  size_t sizeInBytes = mCapacity * sizeof(size_t);

  mHostData = static_cast<size_t*>(_mm_malloc(sizeInBytes, kDataAlignment));

  if (!mHostData)
  {
    throw std::bad_alloc();
  }

  // Register Host memory (pin in memory)
  cudaCheckErrors(cudaHostRegister(mHostData, sizeInBytes, cudaHostRegisterPortable));

  if ((cudaMalloc<size_t>(&mDeviceData, sizeInBytes) != cudaSuccess) || (!mDeviceData))
  {
    throw std::bad_alloc();
  }
}// end of allocateMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Free memory.
 */
void BaseIndexMatrix::freeMemory()
{
  if (mHostData)
  {
    cudaHostUnregister(mHostData);
    _mm_free(mHostData);
  }
  mHostData = nullptr;

  // Free GPU memory
  if (mDeviceData)
  {
    cudaCheckErrors(cudaFree(mDeviceData));
  }
  mDeviceData = nullptr;
}// end of freeMemory
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Private methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//
