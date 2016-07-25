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
 *
 * @date        12 November 2015, 16:49 (created) \n
 *              25 July     2016, 13:17 (revised)
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

#include <stdexcept>
#include <cstring>
#include <cuda_runtime.h>

#include <Parameters/CUDAParameters.h>
#include <Parameters/CUDADeviceConstants.cuh>
#include <Parameters/Parameters.h>

#include <Logger/ErrorMessages.h>

#include <KSpaceSolver/SolverCUDAKernels.cuh>


//------------------------------------------------------------------------------------------------//
//------------------------------------------ CONSTANTS -------------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Constructor.
 */
TCUDAParameters::TCUDAParameters() :
        deviceIdx(DEFAULT_DEVICE_IDX),
        solverBlockSize1D(UNDEFINDED_SIZE), solverGridSize1D(UNDEFINDED_SIZE),
        solverTransposeBlockSize(UNDEFINDED_SIZE), solverTransposeGirdSize(UNDEFINDED_SIZE),
        samplerBlockSize1D(UNDEFINDED_SIZE), samplerGridSize1D(UNDEFINDED_SIZE),
        deviceProperties()
{
}// end of default constructor
//--------------------------------------------------------------------------------------------------




//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Return the name of device used.
 *
 * @return  device name of the selected GPU
 */
std::string TCUDAParameters::GetDeviceName() const
{
  if (strcmp(deviceProperties.name, "") == 0)
  {
    return "N/A";
  }
  return deviceProperties.name;
}// end of GetDeviceName
//--------------------------------------------------------------------------------------------------

/**
 * Select cuda device for execution. If no device is specified, the first free is chosen. The
 * routine also checks whether the CUDA runtime and driver version match and whether the GPU is
 * supported by the code. If there is no free device is present, the code terminates
 *
 * @param [in] deviceIdx - Device index (default DEFAULT_DEVICE_IDX)
 */
void TCUDAParameters::SelectDevice(const int deviceIdx)
{
  // check CUDA driver version and if not sufficient, terminate
  CheckCUDAVersion();

  this->deviceIdx = deviceIdx;

  //choose the GPU device with the most global memory
  int nDevices;
  checkCudaErrors(cudaGetDeviceCount(&nDevices));
  cudaGetLastError();

  cudaError_t lastError;
  //if the user does not provided a specific GPU, use the first one
  if (deviceIdx == DEFAULT_DEVICE_IDX)
  {
    bool deviceFound = false;

    for (int testDevice = 0; testDevice < nDevices; testDevice++)
    {
      // try to set the GPU and reset it
      cudaSetDevice(testDevice);
      cudaDeviceReset();
      lastError = cudaGetLastError();

      // Reset was done properly, test CUDA code version
      if (lastError == cudaSuccess)
      {
        // Read the GPU SM version and the kernel version
        bool cudaCodeVersionOK = CheckCUDACodeVersion();
        lastError = cudaGetLastError();

        if (cudaCodeVersionOK && (lastError == cudaSuccess))
        {
          // acquire the GPU
          this->deviceIdx = testDevice;
          deviceFound = true;
          break;
        }
      }
      // GPU was busy, reset and continue
      lastError = cudaDeviceReset();

      //clear last error
      cudaGetLastError();
    }

    if (!deviceFound)
    {
      throw std::runtime_error(ERR_FMT_NO_FREE_DEVICE);
    }
  }
  else // select a device the user wants
  {
    // check if the specified device is acceptable -
    // not busy, input parameter not out of bounds
    if ((this->deviceIdx > nDevices - 1) || (this->deviceIdx < 0))
    {
      char errMsg[256];
      snprintf(errMsg, 256, ERR_FMT_BAD_DEVICE_IDX, this->deviceIdx, nDevices-1);
      // Throw exception
      throw std::runtime_error(errMsg);
     }

    // set the device and copy it's properties
    cudaSetDevice(this->deviceIdx);
    cudaDeviceReset();
    lastError = cudaGetLastError();

    bool cudaCodeVersionOK = CheckCUDACodeVersion();
    lastError = cudaGetLastError();

    if ((lastError != cudaSuccess) || (!cudaCodeVersionOK))
    {
      lastError = cudaDeviceReset();

      char ErrorMessage[256];
      snprintf(ErrorMessage, 256, ERR_FMT_DEVICE_IS_BUSY, this->deviceIdx);
      throw std::runtime_error(ErrorMessage);
    }
  }

  // Read the device that was allocated
  checkCudaErrors(cudaGetDevice(&this->deviceIdx));
  checkCudaErrors(cudaGetLastError());

  // Reset the device to be able to set the flags
  checkCudaErrors(cudaDeviceReset());
  checkCudaErrors(cudaGetLastError());

  // Enable mapped memory
  checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));

    // Get Device name
  checkCudaErrors(cudaGetDeviceProperties(&deviceProperties, this->deviceIdx));

  // Check the GPU version
  if (!CheckCUDACodeVersion())
  {
    char errMsg[256];
    snprintf(errMsg,256, ERR_FMT_GPU_NOT_SUPPORTED, this->deviceIdx);
    throw std::runtime_error(errMsg);
  }
}// end of SelectCUDADevice
//--------------------------------------------------------------------------------------------------


/**
 * Set kernel configuration.
 * Based on the dimension sizes, sensors masks, and the GPU architecture, adequate CUDA kernel
 * configurations are selected.
 */
void TCUDAParameters::SetKernelConfiguration()
{
  const TParameters& params = TParameters::GetInstance();

  TDimensionSizes fullDims(params.GetFullDimensionSizes());

  // Set kernel configuration for 1D kernels
  // The goal here is to have blocks of size 256 threads and at least 8 x times
  // more blocks than SM processors - This gives us full potential on all
  // Fermi, Kepler, Maxwell still not compromising the maximum number of blocks
  // and threads.

  solverBlockSize1D = 256;
  // Grid size is calculated based on the number of SM processors
  solverGridSize1D  = deviceProperties.multiProcessorCount * 8;

  // the grid size is to small, get 1 gridpoint per thread
  if ((size_t(solverGridSize1D) * size_t(solverBlockSize1D)) > fullDims.GetElementCount())
  {
    solverGridSize1D  = int((fullDims.GetElementCount()  + size_t(solverBlockSize1D) - 1 ) / size_t(solverBlockSize1D));
  }

  // Transposition works by processing for tiles of 32x32 by 4 warps. Every block
  // is responsible for one 2D slab.
  // Block size for the transposition kernels (only 128 threads)
  solverTransposeBlockSize = dim3(32, 4 , 1);
  // Grid size for the transposition kernels
  solverTransposeGirdSize = dim3(deviceProperties.multiProcessorCount * 16, 1, 1);


  // Set configuration for Streaming kernels. We always use 1D kernels of 256 threads
  // and create as many blocks as necessary to fully utilise the GPU.
  // The size of the grid is only tuned for linear sensor mask,
  // since in this execution phase, we don't
  // know how many elements there are in the cuboid sensor mask
  samplerBlockSize1D = 256;

  samplerGridSize1D  = deviceProperties.multiProcessorCount * 8;

  // tune number of blocks for index based sensor mask
  if (params.Get_sensor_mask_type() == TParameters::TSensorMaskType::INDEX)
  {
    // the sensor mask is smaller than 2048 * SMs than use a smaller number of blocks
    if ((size_t(samplerGridSize1D) * size_t(samplerBlockSize1D)) > params.Get_sensor_mask_index_size())
    {
      samplerGridSize1D  = int((params.Get_sensor_mask_index_size()  + size_t(samplerBlockSize1D) - 1 )
                               / size_t(samplerBlockSize1D));
    }
  }

}// end of SetKernelConfiguration
//--------------------------------------------------------------------------------------------------


/**
 * Upload useful simulation constants into device constant memory.
 */
void TCUDAParameters::SetUpDeviceConstants() const
{
  TCUDADeviceConstants constantsToTransfer;

  TParameters& params = TParameters::GetInstance();
  TDimensionSizes  fullDimSizes = params.GetFullDimensionSizes();
  TDimensionSizes  reducedDimSizes = params.GetReducedDimensionSizes();

  // Set values for constant memory
  constantsToTransfer.nx  = fullDimSizes.nx;
  constantsToTransfer.ny  = fullDimSizes.ny;
  constantsToTransfer.nz  = fullDimSizes.nz;
  constantsToTransfer.nElements = fullDimSizes.GetElementCount();
  constantsToTransfer.slabSize  = fullDimSizes.nx * fullDimSizes.ny;

  constantsToTransfer.nxComplex = reducedDimSizes.nx;
  constantsToTransfer.nyComplex = reducedDimSizes.ny;
  constantsToTransfer.nzComplex = reducedDimSizes.nz;
  constantsToTransfer.nElementsComplex = reducedDimSizes.GetElementCount();
  constantsToTransfer.slabSizeComplex = reducedDimSizes.nx * reducedDimSizes.ny;

  constantsToTransfer.fftDivider  = 1.0f / fullDimSizes.GetElementCount();
  constantsToTransfer.fftDividerX = 1.0f / fullDimSizes.nx;
  constantsToTransfer.fftDividerY = 1.0f / fullDimSizes.ny;
  constantsToTransfer.fftDividerZ = 1.0f / fullDimSizes.nz;

  constantsToTransfer.dt  = params.Get_dt();
  constantsToTransfer.dt2 = params.Get_dt() * 2.0f;
  constantsToTransfer.c2  = params.Get_c0_scalar();

  constantsToTransfer.rho0_scalar     = params.Get_rho0_scalar();
  constantsToTransfer.dt_rho0_scalar  = params.Get_rho0_scalar() * params.Get_dt();
  constantsToTransfer.rho0_sgx_scalar = params.Get_rho0_sgx_scalar();
  constantsToTransfer.rho0_sgy_scalar = params.Get_rho0_sgy_scalar(),
  constantsToTransfer.rho0_sgz_scalar = params.Get_rho0_sgz_scalar(),

  constantsToTransfer.BonA_scalar       = params.Get_BonA_scalar();
  constantsToTransfer.absorb_tau_scalar = params.Get_absorb_tau_scalar();
  constantsToTransfer.absorb_eta_scalar = params.Get_absorb_eta_scalar();


  // source masks
  constantsToTransfer.p_source_index_size = params.Get_p_source_index_size();
  constantsToTransfer.p_source_mode       = params.Get_p_source_mode();
  constantsToTransfer.p_source_many       = params.Get_p_source_many();

  constantsToTransfer.u_source_index_size = params.Get_u_source_index_size();
  constantsToTransfer.u_source_mode       = params.Get_u_source_mode();
  constantsToTransfer.u_source_many       = params.Get_u_source_many();

  constantsToTransfer.SetUpCUDADeviceConstatns();
}// end of SetUpDeviceConstants
//--------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//



/**
 * Check whether the CUDA driver version installed is sufficient for the code.
 * If anything goes wrong, throw an exception and exit/
 *
 * @throw runtime_error when the CUDA driver is too old.
 */
void TCUDAParameters::CheckCUDAVersion()
{
  int cudaRuntimeVersion;
  int cudaDriverVersion;

  if (cudaRuntimeGetVersion(&cudaRuntimeVersion) != cudaSuccess)
  {
    throw std::runtime_error(ERR_FM_CANNOT_READ_CUDA_VERSION);
  }

  if (cudaDriverGetVersion(&cudaDriverVersion) != cudaSuccess)
  {
    throw std::runtime_error(ERR_FM_CANNOT_READ_CUDA_VERSION);
  }

  if (cudaDriverVersion < cudaRuntimeVersion)
  {
    char ErrMsg[256];
    snprintf(ErrMsg,
             256,
             ERR_FMT_INSUFFICIENT_CUDA_DRIVER,
             cudaRuntimeVersion / 1000, (cudaRuntimeVersion % 100) / 10,
             cudaDriverVersion  / 1000, (cudaDriverVersion  % 100) / 10);
    throw std::runtime_error(ErrMsg);
  }
}// end of CheckCUDAVersion
//--------------------------------------------------------------------------------------------------

/**
 * Check whether the GPU has SM 2.0 at least.
 *
 * @return the GPU version
 */
bool TCUDAParameters::CheckCUDACodeVersion()
{
  return (SolverCUDAKernels::GetCUDACodeVersion() >= 20);
}// end of CheckCUDACodeVersion
//--------------------------------------------------------------------------------------------------
