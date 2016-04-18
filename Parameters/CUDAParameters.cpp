/**
 * @file        CUDAParameters.cpp
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file for the class for setting CUDA kernel parameters.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        12 November 2015, 16:49 (created) \n
 *              12 April    2016, 15:03 (revised)
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

#include <stdexcept>
#include <stdio.h>
#include <cuda_runtime.h>

#include <Parameters/CUDAParameters.h>
#include <Parameters/CUDADeviceConstants.cuh>
#include <Parameters/Parameters.h>

#include <Utils/ErrorMessages.h>

#include <KSpaceSolver/SolverCUDAKernels.cuh>



/**
 * Constructor
 */
TCUDAParameters::TCUDAParameters() :
        DeviceIdx(DefaultDeviceIdx),
        SolverBlockSize1D(UndefinedSize), SolverGridSize1D(UndefinedSize),
        SolverTransposeBlockSize(UndefinedSize), SolverTransposeGirdSize(UndefinedSize),
        SamplerBlockSize1D(UndefinedSize), SamplerGridSize1D(UndefinedSize),
        DeviceProperties()
{
}// end of default constructor
//------------------------------------------------------------------------------

//----------------------------------------------------------------------------//
//-------------------------------- Constants ---------------------------------//
//----------------------------------------------------------------------------//


//----------------------------------------------------------------------------//
//----------------------------- Initializations ------------------------------//
//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//
//---------------------------------- Public ----------------------------------//
//----------------------------------------------------------------------------//

/**
 * Select cuda device for execution. If no device is specified, the first free is
 * chosen. The routine also checks whether the CUDA runtime and driver version
 * match and whether the GPU is supported by the code.
 * If there is no free device is present, the code terminates
 * @param [in] DeviceIdx - Device index (default -1)
 */
void TCUDAParameters::SelectDevice(const int DeviceIdx)
{

  // check CUDA driver version and if not sufficient, terminate
  CheckCUDAVersion();

  this->DeviceIdx = DeviceIdx;

  //choose the GPU device with the most global memory
  int NumOfDevices;
  checkCudaErrors(cudaGetDeviceCount(&NumOfDevices));
  cudaGetLastError();

  cudaError_t lastError;
  //if the user does not provided a specific GPU, use the first one
  if (DeviceIdx == DefaultDeviceIdx)
  {
    bool DeviceFound = false;

    for (int testDevice = 0; testDevice < NumOfDevices; testDevice++)
    {
      // try to set the GPU and reset it
      lastError = cudaSetDevice(testDevice);
      //printf("\ncudaSetDevice for Idx   %d \t Error id %d \t error message \"%s\" \n", testDevice, lastError,  cudaGetErrorString(lastError));

      lastError = cudaDeviceReset();
      //printf("cudaDeviceReset for Idx %d \t Error id %d \t error message \"%s\" \n", testDevice, lastError, cudaGetErrorString(lastError));

      lastError = cudaGetLastError();
      //printf("GetLastError after Reset for Idx %d \t Error id %d \t error message \"%s\" \n", testDevice, lastError, cudaGetErrorString(lastError));

      // Reset was done properly, test CUDA code version
      if (lastError == cudaSuccess)
      {
        // Read the GPU SM version and the kernel version
        bool cudaCodeVersionOK = CheckCUDACodeVersion();
        lastError = cudaGetLastError();
        //printf("CheckCUDACodeVersion for Idx %d \t Error id %d \t error message %s \n", testDevice, lastError, cudaGetErrorString(lastError));

        if (cudaCodeVersionOK && (lastError == cudaSuccess))
        {
          // acquirte the GPU
          this->DeviceIdx = testDevice;
          DeviceFound = true;
          break;
        }
      }

      // GPU was busy, reset and continue
      lastError = cudaDeviceReset();
      //printf("cudaDeviceReset after unsuccessful allocation Idx %d \t Error id %d \t error message \"%s\" \n", testDevice, lastError, cudaGetErrorString(lastError));

      //clear last error
      cudaGetLastError();
    }

    if (!DeviceFound)
    {
      throw std::runtime_error(CUDAParameters_ERR_FMT_NoFreeDevice);
    }
  }
  else // select a device the user wants
  {
    // check if the specified device is acceptable -
    // not busy, input parameter not out of bounds
    if ((this->DeviceIdx > NumOfDevices - 1) || (this->DeviceIdx < 0))
    {
      char ErrorMessage[256];
      snprintf(ErrorMessage, 256, CUDAParameters_ERR_FMT_WrongDeviceIdx, this->DeviceIdx, NumOfDevices-1);
      // Throw exception
      throw std::runtime_error(ErrorMessage);
     }

    // set the device and copy it's properties
    lastError = cudaSetDevice(this->DeviceIdx);
    //printf("\ncudaSetDevice for Idx   %d \t Error id %d \t error message \"%s\" \n", this->DeviceIdx, lastError,  cudaGetErrorString(lastError));

    lastError = cudaDeviceReset();
    //printf("cudaDeviceReset for Idx %d \t Error id %d \t error message \"%s\" \n", this->DeviceIdx, lastError, cudaGetErrorString(lastError));

    lastError = cudaGetLastError();
    //printf("GetLastError after Reset for Idx %d \t Error id %d \t error message \"%s\" \n", this->DeviceIdx, lastError, cudaGetErrorString(lastError));

    bool cudaCodeVersionOK = CheckCUDACodeVersion();
    lastError = cudaGetLastError();
    //printf("CheckCUDACodeVersion for Idx %d \t Error id %d \t error message %s \n", this->DeviceIdx, lastError, cudaGetErrorString(lastError));

    if ((lastError != cudaSuccess) || (!cudaCodeVersionOK))
    {
      lastError = cudaDeviceReset();
      //printf("cudaDeviceReset for Idx %d \t Error id %d \t error message \"%s\" \n", this->DeviceIdx, lastError, cudaGetErrorString(lastError));

      char ErrorMessage[256];
      snprintf(ErrorMessage, 256, CUDAParameters_ERR_FMT_DeviceIsBusy, this->DeviceIdx);
      throw std::runtime_error(ErrorMessage);
    }
  }

  // Read the device that was allocated
  checkCudaErrors(cudaGetDevice(&this->DeviceIdx));
  checkCudaErrors(cudaGetLastError());

  // Reset the device to be able to set the flags
  checkCudaErrors(cudaDeviceReset());
  checkCudaErrors(cudaGetLastError());

  // Enable mapped memory
  checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));

    // Get Device name
  checkCudaErrors(cudaGetDeviceProperties(&DeviceProperties, this->DeviceIdx));

  /// Check the GPU version
  if (!CheckCUDACodeVersion())
  {
    char ErrorMessage[256];
    snprintf(ErrorMessage,256, CUDAParameters_ERR_FM_GPUNotSupported, this->DeviceIdx);
    throw std::runtime_error(ErrorMessage);
  }
}// end of SelectCUDADevice
//------------------------------------------------------------------------------


/**
 * Set kernel configuration.
 * Based on the dimension sizes, sensors masks, and the GPU architecture, adequate
 * CUDA kernel configurations are selected.
 */
void TCUDAParameters::SetKernelConfiguration()
{
  TParameters * Parameters = TParameters::GetInstance();

  TDimensionSizes FullDims(Parameters->GetFullDimensionSizes());

  // Set kernel configuration for 1D kernels
  // The goal here is to have blocks of size 256 threads and at least 8 x times
  // more blocks than SM processors - This gives us full potential on all
  // Fermi, Kepler, Maxwell still not compromising the maximum number of blocks
  // and threads.

  SolverBlockSize1D = 256;
  // Grid size is calculated based on the number of SM processors
  SolverGridSize1D  = DeviceProperties.multiProcessorCount * 8;

  // the grid size is to small, get 1 gridpoint per thread
  if ((size_t(SolverGridSize1D) * size_t(SolverBlockSize1D)) > FullDims.GetElementCount())
  {
    SolverGridSize1D  = int((FullDims.GetElementCount()  + size_t(SolverBlockSize1D) - 1 ) / size_t(SolverBlockSize1D));
  }

  // Transposition works by processing for tiles of 32x32 by 4 warps. Every block
  // is responsible for one 2D slab.
  // Block size for the transposition kernels (only 128 threads)
  SolverTransposeBlockSize = dim3(32, 4 , 1);
  // Grid size for the transposition kernels
  SolverTransposeGirdSize = dim3(DeviceProperties.multiProcessorCount * 16, 1, 1);


  // Set configuration for Streaming kernels. We always use 1D kernels of 256 threads
  // and create as many blocks as necessary to fully utilise the GPU.
  // The size of the grid is only tuned for linear sensor mask,
  // since in this execution phase, we don't
  // know how many elements there are in the cuboid sensor mask
  SamplerBlockSize1D = 256;

  SamplerGridSize1D  = DeviceProperties.multiProcessorCount * 8;

  // tune number of blocks for index based sensor mask
  if (Parameters->Get_sensor_mask_type() == TParameters::TSensorMaskType::smt_index)
  {
    // the sensor mask is smaller than 2048 * SMs than use a smaller number of blocks
    if ((size_t(SamplerGridSize1D) * size_t(SamplerBlockSize1D)) > Parameters->Get_sensor_mask_index_size())
    {
      SamplerGridSize1D  = int((Parameters->Get_sensor_mask_index_size()  + size_t(SamplerBlockSize1D) - 1 )
                               / size_t(SamplerBlockSize1D));
    }
  }

}// end of SetKernelConfiguration
//------------------------------------------------------------------------------




/**
 * Upload useful simulation constants into device constant memory
 */
void TCUDAParameters::SetUpDeviceConstants()
{
   TCUDADeviceConstants ConstantsToTransfer;

   TParameters * Params = TParameters::GetInstance();
   TDimensionSizes  FullDimensionSizes = Params->GetFullDimensionSizes();
   TDimensionSizes  ReducedDimensionSizes = Params->GetReducedDimensionSizes();

  // Set values for constant memory
  ConstantsToTransfer.Nx  = FullDimensionSizes.X;
  ConstantsToTransfer.Ny  = FullDimensionSizes.Y;
  ConstantsToTransfer.Nz  = FullDimensionSizes.Z;
  ConstantsToTransfer.TotalElementCount = FullDimensionSizes.GetElementCount();
  ConstantsToTransfer.SlabSize = FullDimensionSizes.X * FullDimensionSizes.Y;

  ConstantsToTransfer.Complex_Nx = ReducedDimensionSizes.X;
  ConstantsToTransfer.Complex_Ny = ReducedDimensionSizes.Y;
  ConstantsToTransfer.Complex_Nz = ReducedDimensionSizes.Z;
  ConstantsToTransfer.ComplexTotalElementCount = ReducedDimensionSizes.GetElementCount();
  ConstantsToTransfer.ComplexSlabSize = ReducedDimensionSizes.X * ReducedDimensionSizes.Y;

  ConstantsToTransfer.FFTDivider  = 1.0f / FullDimensionSizes.GetElementCount();
  ConstantsToTransfer.FFTDividerX = 1.0f / FullDimensionSizes.X;
  ConstantsToTransfer.FFTDividerY = 1.0f / FullDimensionSizes.Y;
  ConstantsToTransfer.FFTDividerZ = 1.0f / FullDimensionSizes.Z;



  ConstantsToTransfer.dt  = Params->Get_dt();
  ConstantsToTransfer.dt2 = Params->Get_dt() * 2.0f;
  ConstantsToTransfer.c2  = Params->Get_c0_scalar();

  ConstantsToTransfer.rho0_scalar     = Params->Get_rho0_scalar();
  ConstantsToTransfer.dt_rho0_scalar  = Params->Get_rho0_scalar() * Params->Get_dt();
  ConstantsToTransfer.rho0_sgx_scalar = Params->Get_rho0_sgx_scalar();
  ConstantsToTransfer.rho0_sgy_scalar = Params->Get_rho0_sgy_scalar(),
  ConstantsToTransfer.rho0_sgz_scalar = Params->Get_rho0_sgz_scalar(),

  ConstantsToTransfer.BonA_scalar       = Params->Get_BonA_scalar();
  ConstantsToTransfer.Absorb_tau_scalar = Params->Get_absorb_tau_scalar();
  ConstantsToTransfer.Absorb_eta_scalar = Params->Get_absorb_eta_scalar();


  /// source masks
  ConstantsToTransfer.p_source_index_size = Params->Get_p_source_index_size();
  ConstantsToTransfer.p_source_mode       = Params->Get_p_source_mode();
  ConstantsToTransfer.p_source_many       = Params->Get_p_source_many();

  ConstantsToTransfer.u_source_index_size = Params->Get_u_source_index_size();
  ConstantsToTransfer.u_source_mode       = Params->Get_u_source_mode();
  ConstantsToTransfer.u_source_many       = Params->Get_u_source_many();




  ConstantsToTransfer.SetUpCUDADeviceConstatns();
}// end of SetUpDeviceConstants
//------------------------------------------------------------------------------


//----------------------------------------------------------------------------//
//---------------------------------- Protected -------------------------------//
//----------------------------------------------------------------------------//


/**
 * Check whether the CUDA driver version installed is sufficient for the code.
 * If anything goes wrong, throw an exception and exit/
 * @return
 * @throw runtime_error when the CUDA driver is to old.
 */
void TCUDAParameters::CheckCUDAVersion()
{
  int cudaRuntimeVersion;
  int cudaDriverVersion;

  if (cudaRuntimeGetVersion(&cudaRuntimeVersion) != cudaSuccess)
  {
    throw std::runtime_error(CUDAParameters_ERR_FM_CannotReadCUDAVersion);
  }

  if (cudaDriverGetVersion(&cudaDriverVersion) != cudaSuccess)
  {
    throw std::runtime_error(CUDAParameters_ERR_FM_CannotReadCUDAVersion);
  }

  if (cudaDriverVersion < cudaRuntimeVersion)
  {
    char ErrorMessage[256];
    snprintf(ErrorMessage,
             256,
             CUDAParameters_ERR_FMT_InsufficientCUDADriver,
             cudaRuntimeVersion / 1000, (cudaRuntimeVersion % 100) / 10,
             cudaDriverVersion  / 1000, (cudaDriverVersion  % 100) / 10);
    throw std::runtime_error(ErrorMessage);
  }
}// end of CheckCUDAVersion
//------------------------------------------------------------------------------

/**
 *
 * @return The GPU version
 */
bool TCUDAParameters::CheckCUDACodeVersion()
{
  return (SolverCUDAKernels::GetCUDACodeVersion() >= 20);
}// end of CheckCUDACodeVersion
//------------------------------------------------------------------------------

//----------------------------------------------------------------------------//
//---------------------------------- Private ---------------------------------//
//----------------------------------------------------------------------------//
