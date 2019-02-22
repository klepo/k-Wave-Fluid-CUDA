/**
 * @file      BaseOutputStream.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file of the class saving RealMatrix data into the output HDF5 file.
 *
 * @version   kspaceFirstOrder3D 3.6
 *
 * @date      11 July      2012, 10:30 (created) \n
 *            22 February  2019, 15:46 (revised)
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

#include <cmath>
#include <limits>
#include <immintrin.h>

#include <OutputStreams/BaseOutputStream.h>
#include <OutputStreams/OutputStreamsCudaKernels.cuh>

#include <Logger/Logger.h>
#include <Parameters/Parameters.h>


//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Constants -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor - there is no sensor mask by default!
 */
BaseOutputStream::BaseOutputStream(Hdf5File&            file,
                                   MatrixName&          rootObjectName,
                                   const RealMatrix&    sourceMatrix,
                                   const ReduceOperator reduceOp)
  : mFile(file),
    mRootObjectName(rootObjectName),
    mSourceMatrix(sourceMatrix),
    mReduceOp(reduceOp)
{

 }// end of BaseOutputStream
//----------------------------------------------------------------------------------------------------------------------


/**
 * Apply post-processing on the buffer (Done on the GPU side as well).
 */
void BaseOutputStream::postProcess()
{
  switch (mReduceOp)
  {
    case ReduceOperator::kNone:
    {
      // do nothing
      break;
    }

    case ReduceOperator::kRms:
    {
      const float scalingCoeff = 1.0f / (Parameters::getInstance().getNt() -
                                         Parameters::getInstance().getSamplingStartTimeIndex());

      OutputStreamsCudaKernels::postProcessingRms(mDeviceBuffer, scalingCoeff, mSize);
      break;
    }

    case ReduceOperator::kMax:
    {
      // do nothing
      break;
    }

    case ReduceOperator::kMin:
    {
      // do nothing
      break;
    }
  }// switch

}// end of postProcessing
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Protected methods ------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Allocate memory using proper memory alignment.
 */
void BaseOutputStream::allocateMemory()
{
  // Allocate memory on the CPU side (always)
  mHostBuffer = (float*) _mm_malloc(mSize * sizeof (float), kDataAlignment);

  if (!mHostBuffer)
  {
    throw std::bad_alloc();
  }

  // memory allocation done on core 0 - GPU is pinned to the first sockets
  // we need different initialization for different reduce ops
  switch (mReduceOp)
  {
    case ReduceOperator::kNone:
    {
      // zero the matrix - on the CPU side and lock on core 0 (gpu pinned to 1st socket)
      for (size_t i = 0; i < mSize; i++)
      {
        mHostBuffer[i] = 0.0f;
      }
      break;
    }

    case ReduceOperator::kRms:
    {
      // zero the matrix - on the CPU side and lock on core 0 (gpu pinned to 1st socket)
      for (size_t i = 0; i < mSize; i++)
      {
        mHostBuffer[i] = 0.0f;
      }
      break;
    }

    case ReduceOperator::kMax:
    {
      // set the values to the highest negative float value - on the core 0
      for (size_t i = 0; i < mSize; i++)
      {
        mHostBuffer[i] = -1 * std::numeric_limits<float>::max();
      }
      break;
    }

    case ReduceOperator::kMin:
    {
      // set the values to the highest float value - on the core 0
      for (size_t i = 0; i < mSize; i++)
      {
        mHostBuffer[i] = std::numeric_limits<float>::max();
      }
      break;
    }
  }// switch

  // Register Host memory (pin in memory only - no mapped data)
  cudaCheckErrors(cudaHostRegister(mHostBuffer,
                                   mSize * sizeof (float),
                                   cudaHostRegisterPortable | cudaHostRegisterMapped));
  // cudaHostAllocWriteCombined - cannot be used since GPU writes and CPU reads

  // Map CPU data to GPU memory (RAW data) or allocate a GPU data (aggregated)
  if (mReduceOp == ReduceOperator::kNone)
  {
    // Register CPU memory for zero-copy
    cudaCheckErrors(cudaHostGetDevicePointer<float>(&mDeviceBuffer, mHostBuffer, 0));
  }
  else
  {
    // Allocate memory on the GPU side
    if ((cudaMalloc<float>(&mDeviceBuffer, mSize * sizeof (float))!= cudaSuccess) || (!mDeviceBuffer))
    {
      throw std::bad_alloc();
    }
    // if doing aggregation copy initialised arrays on GPU
    copyToDevice();
  }
}// end of allocateMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Free memory.
 */
void BaseOutputStream::freeMemory()
{
  // free host buffer
  if (mHostBuffer)
  {
    cudaHostUnregister(mHostBuffer);
    _mm_free(mHostBuffer);
  }
  mHostBuffer = nullptr;

  // Free GPU memory
  if (mReduceOp != ReduceOperator::kNone)
  {
    cudaCheckErrors(cudaFree(mDeviceBuffer));
  }
  mDeviceBuffer = nullptr;
}// end of FreeMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 *  Copy data hostBuffer -> deviceBuffer
 */
void BaseOutputStream::copyToDevice()
{
  cudaCheckErrors(cudaMemcpy(mDeviceBuffer, mHostBuffer, mSize * sizeof(float), cudaMemcpyHostToDevice));
}// end of copyToDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data deviceBuffer -> hostBuffer
 */
void BaseOutputStream::copyFromDevice()
{
  cudaCheckErrors(cudaMemcpy(mHostBuffer, mDeviceBuffer, mSize * sizeof(float), cudaMemcpyDeviceToHost));
}// end of copyFromDevice
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Private methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//
