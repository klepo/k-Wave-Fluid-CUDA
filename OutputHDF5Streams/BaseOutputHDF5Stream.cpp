/**
 * @file        BaseOutputHDF5Stream.cpp
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file of the class saving RealMatrix data into the output
 *              HDF5 file.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        11 July      2012, 10:30 (created) \n
 *              17 June      2017, 16:16 (revised)
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

#include <cmath>
#include <limits>
#include <immintrin.h>

#include <OutputHDF5Streams/BaseOutputHDF5Stream.h>
#include <OutputHDF5Streams/OutputStreamsCUDAKernels.cuh>

#include <Logger/Logger.h>
#include <Parameters/Parameters.h>


//------------------------------------------------------------------------------------------------//
//------------------------------------------ Constants -------------------------------------------//
//------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Constructor - there is no sensor mask by default!
 * it links the HDF5 dataset, source (sampled matrix) and the reduce operator together.
 * The constructor DOES NOT allocate memory because the size of the sensor mask is not known at
 * the time the instance of the class is being created.
 *
 * @param [in] file           - Handle to the HDF5 (output) file
 * @param [in] rootObjectName - The root object that stores the sample  data (dataset or group)
 * @param [in] sourceMatrix   - The source matrix (only real matrices are supported)
 * @param [in] reduceOp       - Reduce operator
 */
TBaseOutputHDF5Stream::TBaseOutputHDF5Stream(THDF5_File&           file,
                                             MatrixName&          rootObjectName,
                                             const TRealMatrix&    sourceMatrix,
                                             const TReduceOperator reduceOp)
            : file(file),
              rootObjectName(rootObjectName),
              sourceMatrix(sourceMatrix),
              reduceOp(reduceOp)
{

 }// end of TBaseOutputHDF5Stream
//--------------------------------------------------------------------------------------------------


/**
 * Apply post-processing on the buffer (Done on the GPU side as well).
 */
void TBaseOutputHDF5Stream::PostProcess()
{
  switch (reduceOp)
  {
    case TReduceOperator::NONE:
    {
      // do nothing
      break;
    }

    case TReduceOperator::RMS:
    {
      const float scalingCoeff = 1.0f / (Parameters::getInstance().getNt() -
                                         Parameters::getInstance().getSamplingStartTimeIndex());

      OutputStreamsCUDAKernels::PostProcessingRMS(deviceBuffer, scalingCoeff, bufferSize);
      break;
    }

    case TReduceOperator::MAX:
    {
      // do nothing
      break;
    }

    case TReduceOperator::MIN:
    {
      // do nothing
      break;
    }
  }// switch

}// end of ApplyPostProcessing
//-------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Allocate memory using proper memory alignment.
 *
 * @warning - This can routine is not used in the base class (should be used in derived ones).
 */
void TBaseOutputHDF5Stream::AllocateMemory()
{
  // Allocate memory on the CPU side (always)
  hostBuffer = (float*) _mm_malloc(bufferSize * sizeof (float), kDataAlignment);

  if (!hostBuffer)
  {
    throw std::bad_alloc();
  }

  // memory allocation done on core 0 - GPU is pinned to the first sockets
  // we need different initialization for different reduce ops
  switch (reduceOp)
  {
    case TReduceOperator::NONE :
    {
      // zero the matrix - on the CPU side and lock on core 0 (gpu pinned to 1st socket)
      for (size_t i = 0; i < bufferSize; i++)
      {
        hostBuffer[i] = 0.0f;
      }
      break;
    }

    case TReduceOperator::RMS :
    {
      // zero the matrix - on the CPU side and lock on core 0 (gpu pinned to 1st socket)
      for (size_t i = 0; i < bufferSize; i++)
      {
        hostBuffer[i] = 0.0f;
      }
      break;
    }

    case TReduceOperator::MAX :
    {
      // set the values to the highest negative float value - on the core 0
      for (size_t i = 0; i < bufferSize; i++)
      {
        hostBuffer[i] = -1 * std::numeric_limits<float>::max();
      }
      break;
    }

    case TReduceOperator::MIN :
    {
      // set the values to the highest float value - on the core 0
      for (size_t i = 0; i < bufferSize; i++)
      {
        hostBuffer[i] = std::numeric_limits<float>::max();
      }
      break;
    }
  }// switch

  // Register Host memory (pin in memory only - no mapped data)
  cudaCheckErrors(cudaHostRegister(hostBuffer,
                                   bufferSize * sizeof (float),
                                   cudaHostRegisterPortable | cudaHostRegisterMapped));
  // cudaHostAllocWriteCombined - cannot be used since GPU writes and CPU reads

  // Map CPU buffer to GPU memory (RAW data) or allocate a GPU buffer (aggregated)
  if (reduceOp == TReduceOperator::NONE)
  {
    // Register CPU memory for zero-copy
    cudaCheckErrors(cudaHostGetDevicePointer<float>(&deviceBuffer, hostBuffer, 0));
  }
  else
  {
    // Allocate memory on the GPU side
    if ((cudaMalloc<float>(&deviceBuffer, bufferSize * sizeof (float))!= cudaSuccess) || (!deviceBuffer))
    {
      throw std::bad_alloc();
    }
    // if doing aggregation copy initialised arrays on GPU
    CopyDataToDevice();
  }
}// end of AllocateMemory
//--------------------------------------------------------------------------------------------------

/**
 * Free memory.
 *
 * @warning - This can routine is not used in the base class (should be used in derived ones).
 */
void TBaseOutputHDF5Stream::FreeMemory()
{
  // free host buffer
  if (hostBuffer)
  {
    cudaHostUnregister(hostBuffer);
    _mm_free(hostBuffer);
  }
  hostBuffer = nullptr;

  // Free GPU memory
  if (reduceOp != TReduceOperator::NONE)
  {
    cudaCheckErrors(cudaFree(deviceBuffer));
  }
  deviceBuffer = nullptr;
}// end of FreeMemory
//--------------------------------------------------------------------------------------------------

/**
 *  Copy data hostBuffer -> deviceBuffer
 */
void TBaseOutputHDF5Stream::CopyDataToDevice()
{

  cudaCheckErrors(cudaMemcpy(deviceBuffer,
                             hostBuffer,
                             bufferSize * sizeof(float),
                             cudaMemcpyHostToDevice));

}// end of CopyDataToDevice
//--------------------------------------------------------------------------------------------------

/**
 * Copy data deviceBuffer -> hostBuffer
 */
void TBaseOutputHDF5Stream::CopyDataFromDevice()
{
  cudaCheckErrors(cudaMemcpy(hostBuffer,
                             deviceBuffer,
                             bufferSize * sizeof(float),
                             cudaMemcpyDeviceToHost));
}// end of CopyDataFromDevice
//--------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//

