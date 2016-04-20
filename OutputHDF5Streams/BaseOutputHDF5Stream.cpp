/**
 * @file        OutputHDF5Stream.cpp
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file of the class saving RealMatrix data
 *              into the output HDF5 file.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        11 July      2012, 10:30 (created) \n
 *              20 April     2016, 10:40 (revised)
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

#include <cmath>
#include <limits>
#include <immintrin.h>

#include <OutputHDF5Streams/BaseOutputHDF5Stream.h>
#include <OutputHDF5Streams/OutputStreamsCUDAKernels.cuh>

#include <Logger/ErrorMessages.h>
#include <Parameters/Parameters.h>


using namespace std;

//----------------------------------------------------------------------------//
//--------------------------- Internal methods -------------------------------//
//----------------------------------------------------------------------------//


//--------------------------------------------------------------------------//
//                              Constants                                   //
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
//                              Definitions                                 //
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
//                              Implementation                              //
//                              public methods                              //
//--------------------------------------------------------------------------//

/**
     * Constructor - there is no sensor mask by default!
     * it links the HDF5 dataset, source (sampled matrix) and the reduction
     * operator together. The constructor DOES NOT allocate memory because the
     * size of the sensor mask is not known at the time the instance of
     * the class is being created.
     *
     * @param [in] HDF5_File           - Handle to the HDF5 (output) file
     * @param [in] HDF5_RootObjectName - The root object that stores the sample
     *                                   data (dataset or group)
     * @param [in] SourceMatrix        - The source matrix (only real matrices
     *                                   are supported)
     * @param [in] ReductionOp         - Reduction operator
     * @param [in] BufferToReuse       - An external buffer can be used to line
     *                                   up the grid points. It must be accessible
     *                                   from both CPU and GPU - used for raw series
     */
  TBaseOutputHDF5Stream::TBaseOutputHDF5Stream(THDF5_File &             HDF5_File,
                                               const char *             HDF5_RootObjectName,
                                               const TRealMatrix &      SourceMatrix,
                                               const TReductionOperator ReductionOp)
            : HDF5_File          (HDF5_File),
              HDF5_RootObjectName(NULL),
              SourceMatrix       (SourceMatrix),
              ReductionOp        (ReductionOp)
{
  // copy the dataset name (just for sure)
  this->HDF5_RootObjectName = new char[strlen(HDF5_RootObjectName)];
  strcpy(this->HDF5_RootObjectName, HDF5_RootObjectName);

 };// end of TBaseOutputHDF5Stream
//------------------------------------------------------------------------------


/**
 * Apply post-processing on the buffer (Done on the GPU side as well).
 */
void TBaseOutputHDF5Stream::PostProcess()
{
  switch (ReductionOp)
  {
    case roNONE:
    {
      // do nothing
      break;
    }

    case roRMS:
    {
      const float ScalingCoeff = 1.0f / (TParameters::GetInstance()->Get_Nt() - TParameters::GetInstance()->GetStartTimeIndex());

      OutputStreamsCUDAKernels::PostProcessingRMS(DeviceStoreBuffer,
                                                  ScalingCoeff,
                                                  BufferSize);
      break;
    }

    case roMAX:
    {
      // do nothing
      break;
    }

    case roMIN:
    {
      // do nothing
      break;
    }
  }// switch

}// end of ApplyPostProcessing
//------------------------------------------------------------------------------

//--------------------------------------------------------------------------//
//                              Implementation                              //
//                             protected methods                            //
//--------------------------------------------------------------------------//

/**
 * Allocate memory using a proper memory alignment.
 * @warning - This can routine is not used in the base class (should be used in
 *            derived ones
 *
 */
void TBaseOutputHDF5Stream::AllocateMemory()
{
  // Allocate memory on the CPU side (always)
  HostStoreBuffer = (float *) _mm_malloc(BufferSize * sizeof (float), DATA_ALIGNMENT);

  if (!HostStoreBuffer)
  {
    fprintf(stderr, Matrix_ERR_FMT_Not_Enough_Memory, "TBaseOutputHDF5Stream");
    throw bad_alloc();
  }


  // memory allocation done on core 0 - GPU is pinned to the first sockets
  // we need different initialization for different reduction ops
  switch (ReductionOp)
  {
    case roNONE :
    {
      // zero the matrix - on the CPU side and lock on core 0 (gpu pinned to 1st socket)
      for (size_t i = 0; i < BufferSize; i++)
      {
        HostStoreBuffer[i] = 0.0f;
      }
      break;
    }

    case roRMS :
    {
      // zero the matrix - on the CPU side and lock on core 0 (gpu pinned to 1st socket)
      for (size_t i = 0; i < BufferSize; i++)
      {
        HostStoreBuffer[i] = 0.0f;
      }
      break;
    }

    case roMAX :
    {
      // set the values to the highest negative float value - on the core 0
      for (size_t i = 0; i < BufferSize; i++)
      {
        HostStoreBuffer[i] = -1 * std::numeric_limits<float>::max();
      }
      break;
    }

    case roMIN :
    {
      // set the values to the highest float value - on the core 0
      for (size_t i = 0; i < BufferSize; i++)
      {
        HostStoreBuffer[i] = std::numeric_limits<float>::max();
      }
      break;
    }
  }// switch


  // Register Host memory (pin in memory only - no mapped data)
  checkCudaErrors(cudaHostRegister(HostStoreBuffer,
                                   BufferSize * sizeof (float),
                                   cudaHostRegisterPortable | cudaHostRegisterMapped));
  // cudaHostAllocWriteCombined - cannot be used since GPU writes and CPU reads

  // Map CPU buffer to GPU memory (RAW data) or allocate a GPU buffer (aggregated)
  if (ReductionOp == roNONE)
  {
    // Register CPU memory for zero-copy
    checkCudaErrors(cudaHostGetDevicePointer<float>(&DeviceStoreBuffer, HostStoreBuffer, 0));
  }
  else
  {
    // Allocate memory on the GPU side
    checkCudaErrors(cudaMalloc<float>(&DeviceStoreBuffer, BufferSize * sizeof (float)));
    // if doing aggregation copy initialised arrays on GPU
    CopyDataToDevice();
  }

}// end of AllocateMemory
//------------------------------------------------------------------------------

/**
 * Free memory.
 * @warning - This can routine is not used in the base class (should be used in
 *            derived ones).
 */
void TBaseOutputHDF5Stream::FreeMemory()
{
  // free host buffer
  if (HostStoreBuffer)
  {
    cudaHostUnregister(HostStoreBuffer);
    _mm_free(HostStoreBuffer);
  }
  HostStoreBuffer = NULL;

  // Free GPU memory
  if (ReductionOp != roNONE)
  {
    checkCudaErrors(cudaFree(DeviceStoreBuffer));
  }
  DeviceStoreBuffer = NULL;
}// end of FreeMemory
//------------------------------------------------------------------------------

/**
 *  Copy data  HostStoreBuffer -> DeviceStoreBuffer
 */
void TBaseOutputHDF5Stream::CopyDataToDevice()
{

  checkCudaErrors(cudaMemcpy(DeviceStoreBuffer,
                             HostStoreBuffer,
                             BufferSize * sizeof(float),
                             cudaMemcpyHostToDevice));

}// end of CopyDataToDevice
//------------------------------------------------------------------------------

/**
 * Copy data  DeviceStoreBuffer -> HostStoreBuffer
 */
void TBaseOutputHDF5Stream::CopyDataFromDevice()
{
  checkCudaErrors(cudaMemcpy(HostStoreBuffer,
                             DeviceStoreBuffer,
                             BufferSize * sizeof(float),
                             cudaMemcpyDeviceToHost));
}// end of CopyDataFromDevice
//------------------------------------------------------------------------------



//--------------------------------------------------------------------------//
//                              Implementation                              //
//                              private methods                             //
//--------------------------------------------------------------------------//

