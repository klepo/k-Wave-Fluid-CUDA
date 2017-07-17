/**
 * @file        SolverCUDAKernels.cu
 *
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the all CUDA kernels for the GPU implementation
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        11 March    2013, 13:10 (created) \n
 *              17 July     2017, 16:07 (revised)
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

#include <cuComplex.h>

#include <KSpaceSolver/SolverCUDAKernels.cuh>
#include <Parameters/CudaDeviceConstants.cuh>

#include <Logger/Logger.h>
#include <Utils/CudaUtils.cuh>

//------------------------------------------------------------------------------------------------//
//------------------------------------------ Constants -------------------------------------------//
//------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------------//
//------------------------------------------ Variables -------------------------------------------//
//------------------------------------------------------------------------------------------------//


/**
 * @var      cudaDeviceConstants
 * @brief    This variable holds basic simulation constants for GPU.
 * @details  This variable holds necessary simulation constants in the CUDA GPU memory.
 *           The variable is defined in TCUDADeviceConstants.cu
 */
extern __constant__ CudaDeviceConstants cudaDeviceConstants;


//------------------------------------------------------------------------------------------------//
//--------------------------------------- Global methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Get block size for 1D kernels.
 * @return 1D block size
 */
inline int GetSolverBlockSize1D()
{
  return Parameters::getInstance().getCudaParameters().getSolverBlockSize1D();
};// end of GetSolverBlockSize1D
//--------------------------------------------------------------------------------------------------

/**
 * Get grid size for 1D kernels.
 * @return 1D grid size
 */
inline int GetSolverGridSize1D()
{
  return Parameters::getInstance().getCudaParameters().getSolverGridSize1D();
};// end of GetSolverGridSize1D
//--------------------------------------------------------------------------------------------------

/**
 * Get block size for the transposition kernels.
 * @return 3D grid size
 */
inline dim3 GetSolverTransposeBlockSize()
{
  return Parameters::getInstance().getCudaParameters().getSolverTransposeBlockSize();
};//end of GetSolverTransposeBlockSize()
//--------------------------------------------------------------------------------------------------

/**
 * Get grid size for complex 3D kernels
 * @return 3D grid size
 */
inline dim3 GetSolverTransposeGirdSize()
{
  return Parameters::getInstance().getCudaParameters().getSolverTransposeGirdSize();
};// end of GetSolverTransposeGirdSize()
//--------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public routines ----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Kernel to find out the version of the code.
 * The list of GPUs can be found at https://en.wikipedia.org/wiki/CUDA
 *
 * @param [out] cudaCodeVersion
 */
__global__ void CUDAGetCUDACodeVersion(int* cudaCodeVersion)
{
  *cudaCodeVersion = -1;

  // Read __CUDA_ARCH__ only in actual kernel compilation pass.
  // NVCC does some more passes, where it isn't defined.
  #ifdef __CUDA_ARCH__
    *cudaCodeVersion = (__CUDA_ARCH__ / 10);
  #endif
}// end of CUDAGetCodeVersion
//--------------------------------------------------------------------------------------------------

/**
 * Get the CUDA architecture and GPU code version the code was compiled with.
 * @return  the CUDA code version the code was compiled for
 */
int SolverCUDAKernels::GetCUDACodeVersion()
{
  // host and device pointers, data copied over zero copy memory
  int* hCudaCodeVersion;
  int* dCudaCodeVersion;

  // returned value
  int cudaCodeVersion = 0;
  cudaError_t cudaError;

  // allocate for zero copy
  cudaError = cudaHostAlloc<int>(&hCudaCodeVersion,
                                 sizeof(int),
                                 cudaHostRegisterPortable | cudaHostRegisterMapped);

  // if the device is busy, return 0 - the GPU is not supported
  if (cudaError == cudaSuccess)
  {
    cudaCheckErrors(cudaHostGetDevicePointer<int>(&dCudaCodeVersion, hCudaCodeVersion, 0));

    // find out the CUDA code version
    CUDAGetCUDACodeVersion<<<1,1>>>(dCudaCodeVersion);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess)
    {
      // The GPU architecture is not supported
      cudaCodeVersion = 0;
    }
    else
    {
      cudaCodeVersion = *hCudaCodeVersion;
    }

    cudaCheckErrors(cudaFreeHost(hCudaCodeVersion));
  }

  return (cudaCodeVersion);
}// end of GetCodeVersion
//--------------------------------------------------------------------------------------------------


/**
 * CUDA kernel to calculate ux_sgx, uy_sgy, uz_sgz.  Default (heterogeneous case).
 *
 * @param [in, out] ux_sgx
 * @param [in, out] uy_sgy
 * @param [in, out] uz_sgz
 * @param [in] ifft_x
 * @param [in] ifft_y
 * @param [in] ifft_z
 * @param [in] dt_rho0_sgx
 * @param [in] dt_rho0_sgy
 * @param [in] dt_rho0_sgz
 * @param [in] pml_x
 * @param [in] pml_y
 * @param [in] pml_z
 */
__global__ void CUDAComputeVelocity(float*       ux_sgx,
                                    float*       uy_sgy,
                                    float*       uz_sgz,
                                    const float* ifft_x,
                                    const float* ifft_y,
                                    const float* ifft_z,
                                    const float* dt_rho0_sgx,
                                    const float* dt_rho0_sgy,
                                    const float* dt_rho0_sgz,
                                    const float* pml_x,
                                    const float* pml_y,
                                    const float* pml_z)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getReal3DCoords(i);

    const float ifft_x_el = cudaDeviceConstants.fftDivider * ifft_x[i] * dt_rho0_sgx[i];
    const float ifft_y_el = cudaDeviceConstants.fftDivider * ifft_y[i] * dt_rho0_sgy[i];
    const float ifft_z_el = cudaDeviceConstants.fftDivider * ifft_z[i] * dt_rho0_sgz[i];

    const float pml_x_data = pml_x[coords.x];
    const float pml_y_data = pml_y[coords.y];
    const float pml_z_data = pml_z[coords.z];

    ux_sgx[i] = (ux_sgx[i] * pml_x_data - ifft_x_el) * pml_x_data;
    uy_sgy[i] = (uy_sgy[i] * pml_y_data - ifft_y_el) * pml_y_data;
    uz_sgz[i] = (uz_sgz[i] * pml_z_data - ifft_z_el) * pml_z_data;
  }
}// end of CUDAComputeVelocity
//--------------------------------------------------------------------------------------------------

/**
 * Interface to the CUDA kernel computing new version of ux_sgx.  Default (heterogeneous case)
 *
 * @param [in, out] ux_sgx
 * @param [in, out] uy_sgy
 * @param [in, out] uz_sgz
 * @param [in] ifft_x
 * @param [in] ifft_y
 * @param [in] ifft_z
 * @param [in] dt_rho0_sgx
 * @param [in] dt_rho0_sgy
 * @param [in] dt_rho0_sgz
 * @param [in] pml_x
 * @param [in] pml_y
 * @param [in] pml_z
 */
void SolverCUDAKernels::ComputeVelocity(TRealMatrix&       ux_sgx,
                                        TRealMatrix&       uy_sgy,
                                        TRealMatrix&       uz_sgz,
                                        const TRealMatrix& ifft_x,
                                        const TRealMatrix& ifft_y,
                                        const TRealMatrix& ifft_z,
                                        const TRealMatrix& dt_rho0_sgx,
                                        const TRealMatrix& dt_rho0_sgy,
                                        const TRealMatrix& dt_rho0_sgz,
                                        const TRealMatrix& pml_x,
                                        const TRealMatrix& pml_y,
                                        const TRealMatrix& pml_z)
  {
    CUDAComputeVelocity<<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                       (ux_sgx.GetDeviceData(),
                        uy_sgy.GetDeviceData(),
                        uz_sgz.GetDeviceData(),
                        ifft_x.GetDeviceData(),
                        ifft_y.GetDeviceData(),
                        ifft_z.GetDeviceData(),
                        dt_rho0_sgx.GetDeviceData(),
                        dt_rho0_sgy.GetDeviceData(),
                        dt_rho0_sgz.GetDeviceData(),
                        pml_x.GetDeviceData(),
                        pml_y.GetDeviceData(),
                        pml_z.GetDeviceData());

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of ComputeVelocity
//--------------------------------------------------------------------------------------------------



/**
 * CUDA kernel to calculate ux_sgx, uy_sgy, uz_sgz.
 * This is the case for rho0 being a scalar and a uniform grid.
 *
 * @param [in, out] ux_sgx - new value of ux
 * @param [in, out] uy_sgy - new value of uy
 * @param [in, out] uz_sgz - new value of ux
 * @param [in] ifft_x - gradient for X
 * @param [in] ifft_y - gradient for Y
 * @param [in] ifft_z - gradient for Z
 * @param [in] pml_x
 * @param [in] pml_y
 * @param [in] pml_z
 */
__global__ void CUDAComputeVelocityScalarUniform(float*       ux_sgx,
                                                 float*       uy_sgy,
                                                 float*       uz_sgz,
                                                 const float* ifft_x,
                                                 const float* ifft_y,
                                                 const float* ifft_z,
                                                 const float* pml_x,
                                                 const float* pml_y,
                                                 const float* pml_z)
{
  const float Divider_X = cudaDeviceConstants.dtRho0Sgx * cudaDeviceConstants.fftDivider;
  const float Divider_Y = cudaDeviceConstants.dtRho0Sgy * cudaDeviceConstants.fftDivider;
  const float Divider_Z = cudaDeviceConstants.dtRho0Sgz * cudaDeviceConstants.fftDivider;

  for(auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getReal3DCoords(i);

    const float pml_x_el = pml_x[coords.x];
    const float pml_y_el = pml_y[coords.y];
    const float pml_z_el = pml_z[coords.z];

    ux_sgx[i] = (ux_sgx[i] * pml_x_el - Divider_X * ifft_x[i]) * pml_x_el;
    uy_sgy[i] = (uy_sgy[i] * pml_y_el - Divider_Y * ifft_y[i]) * pml_y_el;
    uz_sgz[i] = (uz_sgz[i] * pml_z_el - Divider_Z * ifft_z[i]) * pml_z_el;
  }// for
}// end of CUDAComputeVelocityScalarUniform
//--------------------------------------------------------------------------------------------------

/**
 * Interface to the CUDA kernel computing new version of ux_sgx, uy_sgy, uz_sgz.
 * This is the case for rho0 being a scalar and a uniform grid.
 *
 * @param [in, out] ux_sgx
 * @param [in, out] uy_sgy
 * @param [in, out] uz_sgz
 * @param [in] ifft_x
 * @param [in] ifft_y
 * @param [in] ifft_z
 * @param [in] pml_x
 * @param [in] pml_y
 * @param [in] pml_z
 */
void SolverCUDAKernels::ComputeVelocityScalarUniform(TRealMatrix&       ux_sgx,
                                                     TRealMatrix&       uy_sgy,
                                                     TRealMatrix&       uz_sgz,
                                                     const TRealMatrix& ifft_x,
                                                     const TRealMatrix& ifft_y,
                                                     const TRealMatrix& ifft_z,
                                                     const TRealMatrix& pml_x,
                                                     const TRealMatrix& pml_y,
                                                     const TRealMatrix& pml_z)
{
  CUDAComputeVelocityScalarUniform<<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                  (ux_sgx.GetDeviceData(),
                                   uy_sgy.GetDeviceData(),
                                   uz_sgz.GetDeviceData(),
                                   ifft_x.GetDeviceData(),
                                   ifft_y.GetDeviceData(),
                                   ifft_z.GetDeviceData(),
                                   pml_x.GetDeviceData(),
                                   pml_y.GetDeviceData(),
                                   pml_z.GetDeviceData());
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of ComputeVelocityScalarUniform
//--------------------------------------------------------------------------------------------------


/**
 * CUDA kernel to calculate ux_sgx, uy_sgy and uz_sgz.
 * This is the case for rho0 being a scalar and a non-uniform grid.
 *
 * @param [in,out] ux_sgx     - updated value of ux_sgx
 * @param [in,out] uy_sgy     - updated value of ux_sgx
 * @param [in,out] uz_sgz     - updated value of ux_sgx
 * @param [in]     ifft_x      - gradient of X
 * @param [in]     ifft_y      - gradient of X
 * @param [in]     ifft_z      - gradient of X
 * @param [in]     dxudxn_sgx - matrix dx shift
 * @param [in]     dyudyn_sgy - matrix dy shift
 * @param [in]     dzudzn_sgz - matrix dz shift
 * @param [in]     pml_x      - matrix of pml_x
 * @param [in]     pml_y       - matrix of pml_x
 * @param [in]     pml_z       - matrix of pml_x
 */
__global__ void CUDAComputeVelocityScalarNonuniform(float*       ux_sgx,
                                                    float*       uy_sgy,
                                                    float*       uz_sgz,
                                                    const float* ifft_x,
                                                    const float* ifft_y,
                                                    const float* ifft_z,
                                                    const float* dxudxn_sgx,
                                                    const float* dyudyn_sgy,
                                                    const float* dzudzn_sgz,
                                                    const float* pml_x,
                                                    const float* pml_y,
                                                    const float* pml_z)
{
  const float Divider_X = cudaDeviceConstants.dtRho0Sgx * cudaDeviceConstants.fftDivider;
  const float Divider_Y = cudaDeviceConstants.dtRho0Sgy * cudaDeviceConstants.fftDivider;;
  const float Divider_Z = cudaDeviceConstants.dtRho0Sgz * cudaDeviceConstants.fftDivider;

  for(auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getReal3DCoords(i);

    const float pml_x_el = pml_x[coords.x];
    const float pml_y_el = pml_y[coords.y];
    const float pml_z_el = pml_z[coords.z];

    const float ifft_x_el = Divider_X * dxudxn_sgx[coords.x] * ifft_x[i];
    const float ifft_y_el = Divider_Y * dyudyn_sgy[coords.y] * ifft_y[i];
    const float ifft_z_el = Divider_Z * dzudzn_sgz[coords.z] * ifft_z[i];

    ux_sgx[i] = (ux_sgx[i] * pml_x_el - ifft_x_el) * pml_x_el;
    uy_sgy[i] = (uy_sgy[i] * pml_y_el - ifft_y_el) * pml_y_el;
    uz_sgz[i] = (uz_sgz[i] * pml_z_el - ifft_z_el) * pml_z_el;
  }// for
}// end of CUDAComputeVelocityScalarNonuniform
//--------------------------------------------------------------------------------------------------

/**
 * Interface to  calculate ux_sgx, uy_sgy and uz_sgz.
 * This is the case for rho0 being a scalar and a non-uniform grid.
 * @param [in,out] ux_sgx     - updated value of ux_sgx
 * @param [in,out] uy_sgy     - updated value of ux_sgx
 * @param [in,out] uz_sgz     - updated value of ux_sgx
 * @param [in]     ifft_x      - gradient of X
 * @param [in]     ifft_y      - gradient of X
 * @param [in]     ifft_z      - gradient of X
 * @param [in]     dxudxn_sgx - matrix dx shift
 * @param [in]     dyudyn_sgy - matrix dy shift
 * @param [in]     dzudzn_sgz - matrix dz shift
 * @param [in]     pml_x      - matrix of pml_x
 * @param [in]     pml_y       - matrix of pml_x
 * @param [in]     pml_z       - matrix of pml_x
 */
void SolverCUDAKernels::ComputeVelocityScalarNonuniform(TRealMatrix&       ux_sgx,
                                                        TRealMatrix&       uy_sgy,
                                                        TRealMatrix&       uz_sgz,
                                                        const TRealMatrix& ifft_x,
                                                        const TRealMatrix& ifft_y,
                                                        const TRealMatrix& ifft_z,
                                                        const TRealMatrix& dxudxn_sgx,
                                                        const TRealMatrix& dyudyn_sgy,
                                                        const TRealMatrix& dzudzn_sgz,
                                                        const TRealMatrix& pml_x,
                                                        const TRealMatrix& pml_y,
                                                        const TRealMatrix& pml_z)
{
  CUDAComputeVelocityScalarNonuniform<<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                     (ux_sgx.GetDeviceData(),
                                      uy_sgy.GetDeviceData(),
                                      uz_sgz.GetDeviceData(),
                                      ifft_x.GetDeviceData(),
                                      ifft_y.GetDeviceData(),
                                      ifft_z.GetDeviceData(),
                                      dxudxn_sgx.GetDeviceData(),
                                      dyudyn_sgy.GetDeviceData(),
                                      dzudzn_sgz.GetDeviceData(),
                                      pml_x.GetDeviceData(),
                                      pml_y.GetDeviceData(),
                                      pml_z.GetDeviceData());

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of CUDAComputeVelocityScalarNonuniform
//--------------------------------------------------------------------------------------------------


/**
 * CUDA kernel adding transducer data to ux_sgx
 *
 * @param [in, out] ux_sgx             - Here we add the signal
 * @param [in]      u_source_index     - Where to add the signal (source)
 * @param [in, out] delay_mask         - Delay mask to push the signal in the domain (incremented per invocation)
 * @param [in]      transducer_signal  - Transducer signal
 */
__global__ void CUDAAddTransducerSource(float*        ux_sgx,
                                        const size_t* u_source_index,
                                        size_t*       delay_mask,
                                        const float*  transducer_signal)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.velocitySourceSize; i += getStride())
  {
    ux_sgx[u_source_index[i]] += transducer_signal[delay_mask[i]];
    delay_mask[i]++;
  }
}// end of CUDAAddTransducerSource
//------------------------------------------------------------------------------

/**
 * Interface to kernel adding transducer data to ux_sgx.
 *
 * @param [in, out] ux_sgx             - Here we add the signal
 * @param [in]      u_source_index     - Where to add the signal (source)
 * @param [in, out] delay_mask         - Delay mask to push the signal in the domain (incremented per invocation)
 * @param [in]      transducer_signal  - Transducer signal
 */
void SolverCUDAKernels::AddTransducerSource(TRealMatrix&        ux_sgx,
                                            const TIndexMatrix& u_source_index,
                                            TIndexMatrix&       delay_mask,
                                            const TRealMatrix&  transducer_signal)
{
  // cuda only supports 32bits anyway
  const int u_source_index_size = static_cast<int>(u_source_index.GetElementCount());

  // Grid size is calculated based on the source size
  const int gridSize  = (u_source_index_size < (GetSolverGridSize1D() *  GetSolverBlockSize1D()))
                        ? (u_source_index_size  + GetSolverBlockSize1D() - 1 ) / GetSolverBlockSize1D()
                        : GetSolverGridSize1D();

  CUDAAddTransducerSource<<<gridSize, GetSolverBlockSize1D()>>>
                         (ux_sgx.GetDeviceData(),
                          u_source_index.GetDeviceData(),
                          delay_mask.GetDeviceData(),
                          transducer_signal.GetDeviceData());
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of AddTransducerSource
//--------------------------------------------------------------------------------------------------


/**
 * CUDA kernel to add in velocity source terms.
 *
 * @param [in, out] uxyz_sgxyz          - velocity matrix to update
 * @param [in]      u_source_input      - Source input to add
 * @param [in]      u_source_index      - Index matrix
 * @param [in]      t_index             - Actual time step
 */
__global__ void CUDAAddVelocitySource(float*        uxyz_sgxyz,
                                      const float*  u_source_input,
                                      const size_t* u_source_index,
                                      const size_t  t_index)
{
  // Set 1D or 2D step for source
  auto index2D = (cudaDeviceConstants.velocitySourceMany == 0) ? t_index : t_index * cudaDeviceConstants.velocitySourceSize;

  if (cudaDeviceConstants.velocitySourceMode == 0)
  {
    for (auto i = getIndex(); i < cudaDeviceConstants.velocitySourceSize; i += getStride())
    {
      uxyz_sgxyz[u_source_index[i]]  = (cudaDeviceConstants.velocitySourceMany == 0) ? u_source_input[index2D] :
                                                                                  u_source_input[index2D + i];
    }// for
  }// end of Dirichlet

  if (cudaDeviceConstants.velocitySourceMode == 1)
  {
    for (auto i  = getIndex(); i < cudaDeviceConstants.velocitySourceSize; i += getStride())
    {
      uxyz_sgxyz[u_source_index[i]] += (cudaDeviceConstants.velocitySourceMany == 0) ? u_source_input[index2D] :
                                                                                  u_source_input[index2D + i];
    }
  }
}// end of CUDAAddVelocitySource
//------------------------------------------------------------------------------


/**
 * Interface to CUDA kernel adding in velocity source terms.
 *
 * @param [in, out] uxyz_sgxyz - Velocity matrix to update
 * @param [in] u_source_input  - Source input to add
 * @param [in] u_source_index  - Index matrix
 * @param [in] t_index         - Actual time step
 */
void SolverCUDAKernels::AddVelocitySource(TRealMatrix&        uxyz_sgxyz,
                                          const TRealMatrix&  u_source_input,
                                          const TIndexMatrix& u_source_index,
                                          const size_t        t_index)
{
  const int u_source_index_size = static_cast<int>(u_source_index.GetElementCount());

  // Grid size is calculated based on the source size
  // for small sources, a custom number of thread blocks is created,
  // otherwise, a standard number is used

  const int gridSize = (u_source_index_size < (GetSolverGridSize1D() *  GetSolverBlockSize1D()))
                       ? (u_source_index_size  + GetSolverBlockSize1D() - 1 ) / GetSolverBlockSize1D()
                       :  GetSolverGridSize1D();

  CUDAAddVelocitySource<<< gridSize, GetSolverBlockSize1D()>>>
                       (uxyz_sgxyz.GetDeviceData(),
                        u_source_input.GetDeviceData(),
                        u_source_index.GetDeviceData(),
                        t_index);

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of AddVelocitySource
//-------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to add p_source to acoustic density.
 *
 * @param [out] rhox - Acoustic density
 * @param [out] rhoy - Acoustic density
 * @param [out] rhoz - Acoustic density
 * @param [in]  p_source_input - Source input to add
 * @param [in]  p_source_index - Index matrix with source
 * @param [in]  t_index        - Actual timestep

 */
__global__ void CUDAAddPressureSource(float*        rhox,
                                      float*        rhoy,
                                      float*        rhoz,
                                      const float*  p_source_input,
                                      const size_t* p_source_index,
                                      const size_t  t_index)
{
  // Set 1D or 2D step for source
  auto index2D = (cudaDeviceConstants.presureSourceMany == 0) ? t_index : t_index * cudaDeviceConstants.presureSourceSize;

  if (cudaDeviceConstants.presureSourceMode == 0)
  {
    if (cudaDeviceConstants.presureSourceMany == 0)
    { // single signal
      for (auto i = getIndex(); i < cudaDeviceConstants.presureSourceSize; i += getStride())
      {
        rhox[p_source_index[i]] = p_source_input[index2D];
        rhoy[p_source_index[i]] = p_source_input[index2D];
        rhoz[p_source_index[i]] = p_source_input[index2D];
      }
    }
    else
    { // multiple signals
      for (auto i = getIndex(); i < cudaDeviceConstants.presureSourceSize; i += getStride())
      {
        rhox[p_source_index[i]] = p_source_input[index2D + i];
        rhoy[p_source_index[i]] = p_source_input[index2D + i];
        rhoz[p_source_index[i]] = p_source_input[index2D + i];
      }
    }
  }// end mode == 0 (Cauchy)

  if (cudaDeviceConstants.presureSourceMode == 1)
  {
    if (cudaDeviceConstants.presureSourceMany == 0)
    { // single signal
      for (auto i = getIndex(); i < cudaDeviceConstants.presureSourceSize; i += getStride())
      {
        rhox[p_source_index[i]] += p_source_input[index2D];
        rhoy[p_source_index[i]] += p_source_input[index2D];
        rhoz[p_source_index[i]] += p_source_input[index2D];
      }
    }
    else
    { // multiple signals
      for (auto i = getIndex(); i < cudaDeviceConstants.presureSourceSize; i += getStride())
      {
        rhox[p_source_index[i]] += p_source_input[index2D + i];
        rhoy[p_source_index[i]] += p_source_input[index2D + i];
        rhoz[p_source_index[i]] += p_source_input[index2D + i];
      }
    }
  }// end mode == 0 (Dirichlet)
}// end of CUDAAdd_p_source
//--------------------------------------------------------------------------------------------------

/**
 * Interface to kernel which adds in pressure source (to acoustic density).
 *
 * @param [out] rhox - Acoustic density
 * @param [out] rhoy - Acoustic density
 * @param [out] rhoz - Acoustic density
 * @param [in]  p_source_input - Source input to add
 * @param [in]  p_source_index - Index matrix with source
 * @param [in]  t_index        - Actual timestep
 */
void SolverCUDAKernels::AddPressureSource(TRealMatrix&        rhox,
                                          TRealMatrix&        rhoy,
                                          TRealMatrix&        rhoz,
                                          const TRealMatrix&  p_source_input,
                                          const TIndexMatrix& p_source_index,
                                          const size_t        t_index)
{
  const int p_source_index_size = static_cast<int>(p_source_index.GetElementCount());
  // Grid size is calculated based on the source size
  const int gridSize  = (p_source_index_size < (GetSolverGridSize1D() *  GetSolverBlockSize1D()))
                        ? (p_source_index_size  + GetSolverBlockSize1D() - 1 ) / GetSolverBlockSize1D()
                        :  GetSolverGridSize1D();

  CUDAAddPressureSource<<<gridSize,GetSolverBlockSize1D()>>>
                       (rhox.GetDeviceData(),
                        rhoy.GetDeviceData(),
                        rhoz.GetDeviceData(),
                        p_source_input.GetDeviceData(),
                        p_source_index.GetDeviceData(),
                        t_index);

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of AddPressureSource
//--------------------------------------------------------------------------------------------------

/**
 * CUDA kernel Compute u = dt ./ rho0_sgx .* u.
 *
 * @param [in, out] ux_sgx - data stored in u matrix
 * @param [in, out] uy_sgy - data stored in u matrix
 * @param [in, out] uz_sgz - data stored in u matrix
 * @param [in]      dt_rho0_sgx - inner member of the equation
 * @param [in]      dt_rho0_sgy - inner member of the equation
 * @param [in]      dt_rho0_sgz - inner member of the equation
 *
 */
template <bool Is_rho0_scalar>
__global__  void CUDACompute_p0_Velocity(float*       ux_sgx,
                                         float*       uy_sgy,
                                         float*       uz_sgz,
                                         const float* dt_rho0_sgx = nullptr,
                                         const float* dt_rho0_sgy = nullptr,
                                         const float* dt_rho0_sgz = nullptr)

{
  if (Is_rho0_scalar)
  {
    const float dividerX = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgx;
    const float dividerY = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgy;
    const float dividerZ = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgz;

    for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
    {
      ux_sgx[i] *= dividerX;
      uy_sgy[i] *= dividerY;
      uz_sgz[i] *= dividerZ;
    }
  }
  else
  { // heterogeneous
    const float divider = cudaDeviceConstants.fftDivider * 0.5f;

    for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
    {
      ux_sgx[i] *= dt_rho0_sgx[i] * divider;
      uy_sgy[i] *= dt_rho0_sgy[i] * divider;
      uz_sgz[i] *= dt_rho0_sgz[i] * divider;
    }
  }
}// end of CUDACompute_p0_Velocity
//-------------------------------------------------------------------------------------------------

/**
 * Interface to CUDA Compute u = dt ./ rho0_sgx .* ifft(FFT).
 *
 * @param [in, out] ux_sgx - data stored in u matrix
 * @param [in, out] uy_sgy - data stored in u matrix
 * @param [in, out] uz_sgz - data stored in u matrix
 * @param [in]      dt_rho0_sgx - inner member of the equation
 * @param [in]      dt_rho0_sgy - inner member of the equation
 * @param [in]      dt_rho0_sgz - inner member of the equation
 *
 */
void SolverCUDAKernels::Compute_p0_Velocity(TRealMatrix&       ux_sgx,
                                            TRealMatrix&       uy_sgy,
                                            TRealMatrix&       uz_sgz,
                                            const TRealMatrix& dt_rho0_sgx,
                                            const TRealMatrix& dt_rho0_sgy,
                                            const TRealMatrix& dt_rho0_sgz)
{
  CUDACompute_p0_Velocity<false>
                         <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                         (ux_sgx.GetDeviceData(),
                         uy_sgy.GetDeviceData(),
                         uz_sgz.GetDeviceData(),
                         dt_rho0_sgx.GetDeviceData(),
                         dt_rho0_sgy.GetDeviceData(),
                         dt_rho0_sgz.GetDeviceData());

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of Compute_p0_Velocity
//--------------------------------------------------------------------------------------------------

/**
 * Interface to CUDA Compute u = dt ./ rho0_sgx .* ifft(FFT).
 * if rho0_sgx is scalar, uniform case.
 *
 * @param [in, out] ux_sgx   - Data stored in u matrix
 * @param [in, out] uy_sgy   - Data stored in u matrix
 * @param [in, out] uz_sgz   - Data stored in u matrix
 */
void SolverCUDAKernels::Compute_p0_Velocity(TRealMatrix& ux_sgx,
                                            TRealMatrix& uy_sgy,
                                            TRealMatrix& uz_sgz)
{
  CUDACompute_p0_Velocity<true>
                         <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                         (ux_sgx.GetDeviceData(),
                          uy_sgy.GetDeviceData(),
                          uz_sgz.GetDeviceData());

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of Compute_p0_Velocity
//--------------------------------------------------------------------------------------------------



/**
 * CUDA kernel to Compute u = dt./rho0_sgy .* ifft (FFT).
 * if rho0_sg is scalar, nonuniform  non uniform grid, y component.
 *
 * @param [in, out] ux_sgx
 * @param [in, out] uy_sgy
 * @param [in, out] uz_sgz
 * @param [in] dxudxn_sgx
 * @param [in] dyudyn_sgy
 * @param [in] dzudzn_sgz
 */
__global__ void CUDACompute_p0_VelocityScalarNonUniform(float*       ux_sgx,
                                                        float*       uy_sgy,
                                                        float*       uz_sgz,
                                                        const float* dxudxn_sgx,
                                                        const float* dyudyn_sgy,
                                                        const float* dzudzn_sgz)
{
  const float dividerX = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgx;
  const float dividerY = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgy;
  const float dividerZ = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgz;

  for(auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getReal3DCoords(i);

    ux_sgx[i] *= dividerX * dxudxn_sgx[coords.x];
    uy_sgy[i] *= dividerY * dyudyn_sgy[coords.y];
    uz_sgz[i] *= dividerZ * dzudzn_sgz[coords.z];
  }
}// end of CUDACompute_p0_VelocityScalarNonUniform
//--------------------------------------------------------------------------------------------------


/**
 * Interface to CUDA kernel to Compute u = dt./rho0_sgy .* ifft (FFT).
 * if rho0_sgx is scalar, nonuniform  non uniform Compute_ddx_kappa_fft_pgrid, y component.
 *
 * @param [in, out] ux_sgx
 * @param [in, out] uy_sgy
 * @param [in, out] uz_sgz
 * @param [in] dxudxn_sgx
 * @param [in] dyudyn_sgy
 * @param [in] dzudzn_sgz
 */
  void SolverCUDAKernels::Compute_p0_VelocityScalarNonUniform(TRealMatrix&       ux_sgx,
                                                              TRealMatrix&       uy_sgy,
                                                              TRealMatrix&       uz_sgz,
                                                              const TRealMatrix& dxudxn_sgx,
                                                              const TRealMatrix& dyudyn_sgy,
                                                              const TRealMatrix& dzudzn_sgz)
{
  CUDACompute_p0_VelocityScalarNonUniform<<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                         (ux_sgx.GetDeviceData(),
                                          uy_sgy.GetDeviceData(),
                                          uz_sgz.GetDeviceData(),
                                          dxudxn_sgx.GetDeviceData(),
                                          dxudxn_sgx.GetDeviceData(),
                                          dxudxn_sgx.GetDeviceData());
// check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of Compute_p0_VelocityScalarNonUniform
//--------------------------------------------------------------------------------------------------


/**
 *  kernel which compute part of the new velocity term - gradient
 *  of p represented by:
 *  bsxfun(\@times, ddx_k_shift_pos, kappa .* p_k).
 *
 *
 * @param [in, out]    fft_x - matrix to store input for iFFT (p) /dx
 * @param [out]    fft_y - matrix to store input for iFFT (p) /dy
 * @param [out]    fft_z - matrix to store input for iFFT (p) /dz
 *
 * @param [in]     kappa - Real matrix of kappa
 *
 * @param [in]     ddx - precomputed value of ddx_k_shift_pos
 * @param [in]     ddy - precomputed value of ddy_k_shift_pos
 * @param [in]     ddz - precomputed value of ddz_k_shift_pos
 */
__global__ void CUDAComputePressurelGradient(cuFloatComplex*       fft_x,
                                             cuFloatComplex*       fft_y,
                                             cuFloatComplex*       fft_z,
                                             const float*          kappa,
                                             const cuFloatComplex* ddx,
                                             const cuFloatComplex* ddy,
                                             const cuFloatComplex* ddz)
{
  for(auto i = getIndex(); i < cudaDeviceConstants.nElementsComplex; i += getStride())
  {
    const dim3 coords = getComplex3DCoords(i);

    const cuFloatComplex p_k_el = fft_x[i] * kappa[i];

    fft_x[i] = cuCmulf(p_k_el, ddx[coords.x]);
    fft_y[i] = cuCmulf(p_k_el, ddy[coords.y]);
    fft_z[i] = cuCmulf(p_k_el, ddz[coords.z]);
  }
}// end of CUDAComputePressurelGradient
//--------------------------------------------------------------------------------------------------

/**
 *  Interface to kernel which computes the spectral part of pressure gradient calculation
 *  bsxfun(\@times, ddx_k_shift_pos, kappa .* p_k).
 *
 * @param [out]    fft_x - matrix to store input for iFFT (p) /dx
 * @param [out]    fft_y - matrix to store input for iFFT (p) /dy
 * @param [out]    fft_z - matrix to store input for iFFT (p) /dz
 *
 * @param [in]     kappa - Real matrix of kappa
 *
 * @param [in]     ddx - precomputed value of ddx_k_shift_pos
 * @param [in]     ddy - precomputed value of ddy_k_shift_pos
 * @param [in]     ddz - precomputed value of ddz_k_shift_pos
 */
void SolverCUDAKernels::ComputePressurelGradient(TCUFFTComplexMatrix& fft_x,
                                                 TCUFFTComplexMatrix& fft_y,
                                                 TCUFFTComplexMatrix& fft_z,
                                                 const TRealMatrix&    kappa,
                                                 const TComplexMatrix& ddx,
                                                 const TComplexMatrix& ddy,
                                                 const TComplexMatrix& ddz)
{
  CUDAComputePressurelGradient<<<GetSolverGridSize1D(),GetSolverBlockSize1D()>>>
                              (reinterpret_cast<cuFloatComplex*>(fft_x.GetDeviceData()),
                               reinterpret_cast<cuFloatComplex*>(fft_y.GetDeviceData()),
                               reinterpret_cast<cuFloatComplex*>(fft_z.GetDeviceData()),
                               kappa.GetDeviceData(),
                               reinterpret_cast<const cuFloatComplex*>(ddx.GetDeviceData()),
                               reinterpret_cast<const cuFloatComplex*>(ddy.GetDeviceData()),
                               reinterpret_cast<const cuFloatComplex*>(ddz.GetDeviceData()));

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of ComputePressurelGradient
//--------------------------------------------------------------------------------------------------


/**
 * Kernel calculating the inner part of du, dy, dz on uniform grid.
 * Complex numbers are passed as float2 structures.
 *
 * @param [in, out] fft_x - FFT of ux
 * @param [in, out] fft_y - FFT of uy
 * @param [in, out] fft_z - FFT of uz
 * @param [in]      kappa
 * @param [in]      ddx_neg - ddx_k_shift_neg
 * @param [in]      ddy_neg - ddy_k_shift_neg
 * @param [in]      ddz_neg - ddz_k_shift_neg
 */
__global__  void CUDAComputeVelocityGradient(cuFloatComplex*       fft_x,
                                             cuFloatComplex*       fft_y,
                                             cuFloatComplex*       fft_z,
                                             const float*          kappa,
                                             const cuFloatComplex* ddx_neg,
                                             const cuFloatComplex* ddy_neg,
                                             const cuFloatComplex* ddz_neg)
{
  for(auto i = getIndex(); i < cudaDeviceConstants.nElementsComplex; i += getStride())
  {
    const dim3 coords = getComplex3DCoords(i);

    const cuFloatComplex ddx_neg_el = ddx_neg[coords.x];
    const cuFloatComplex ddz_neg_el = ddz_neg[coords.z];
    const cuFloatComplex ddy_neg_el = ddy_neg[coords.y];

    const float kappa_el = kappa[i] * cudaDeviceConstants.fftDivider;

    const cuFloatComplex fft_x_el = fft_x[i] * kappa_el;
    const cuFloatComplex fft_y_el = fft_y[i] * kappa_el;
    const cuFloatComplex fft_z_el = fft_z[i] * kappa_el;

    fft_x[i] = cuCmulf(fft_x_el, ddx_neg_el);
    fft_y[i] = cuCmulf(fft_y_el, ddy_neg_el);
    fft_z[i] = cuCmulf(fft_z_el, ddz_neg_el);
  } // for
}// end of CUDAComputeVelocityGradient
//--------------------------------------------------------------------------------------------------

/**
 * Interface to kernel calculating the inner part of du, dy, dz on uniform grid.
 * @param [in, out] fft_x - FFT of ux
 * @param [in, out] fft_y - FFT of uy
 * @param [in, out] fft_z - FFT of uz
 * @param [in] kappa
 * @param [in] ddx_k_shift_neg
 * @param [in] ddy_k_shift_neg
 * @param [in] ddz_k_shift_neg
 */
void SolverCUDAKernels::ComputeVelocityGradient(TCUFFTComplexMatrix&  fft_x,
                                                TCUFFTComplexMatrix&  fft_y,
                                                TCUFFTComplexMatrix&  fft_z,
                                                const TRealMatrix&    kappa,
                                                const TComplexMatrix& ddx_k_shift_neg,
                                                const TComplexMatrix& ddy_k_shift_neg,
                                                const TComplexMatrix& ddz_k_shift_neg)
{
  CUDAComputeVelocityGradient<<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                        (reinterpret_cast<cuFloatComplex *>(fft_x.GetDeviceData()),
                         reinterpret_cast<cuFloatComplex *>(fft_y.GetDeviceData()),
                         reinterpret_cast<cuFloatComplex *>(fft_z.GetDeviceData()),
                         kappa.GetDeviceData(),
                         reinterpret_cast<const cuFloatComplex *>(ddx_k_shift_neg.GetDeviceData()),
                         reinterpret_cast<const cuFloatComplex *>(ddy_k_shift_neg.GetDeviceData()),
                         reinterpret_cast<const cuFloatComplex *>(ddz_k_shift_neg.GetDeviceData()));

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of ComputeVelocityGradient
//--------------------------------------------------------------------------------------------------


/**
 * CUDA kernel to shift du, dy and dz on non-uniform grid.
 *
 * @param [in,out] duxdx
 * @param [in,out] duydy
 * @param [in,out] duzdz
 * @param [in]     duxdxn
 * @param [in]     duydyn
 * @param [in]     duzdzn
 */
__global__  void CUDAComputeVelocityGradientNonuniform(float*       duxdx,
                                                       float*       duydy,
                                                       float*       duzdz,
                                                       const float* duxdxn,
                                                       const float* duydyn,
                                                       const float* duzdzn)
{
  for(auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getReal3DCoords(i);

    duxdx[i] *= duxdxn[coords.x];
    duydy[i] *= duydyn[coords.y];
    duzdz[i] *= duzdzn[coords.z];
  }
}// end of CUDAComputeVelocityGradientNonuniform
//--------------------------------------------------------------------------------------------------

/**
 * Interface to CUDA kernel which shift new values of dux, duy and duz on non-uniform grid.
 *
 * @param [in,out] duxdx
 * @param [in,out] duydy
 * @param [in,out] duzdz
 * @param [in]     dxudxn
 * @param [in]     dyudyn
 * @param [in]     dzudzn
 */
void SolverCUDAKernels::ComputeVelocityGradientNonuniform(TRealMatrix&       duxdx,
                                                          TRealMatrix&       duydy,
                                                          TRealMatrix&       duzdz,
                                                          const TRealMatrix& dxudxn,
                                                          const TRealMatrix& dyudyn,
                                                          const TRealMatrix& dzudzn)
{
  CUDAComputeVelocityGradientNonuniform<<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                       (duxdx.GetDeviceData(),
                                        duydy.GetDeviceData(),
                                        duzdz.GetDeviceData(),
                                        dxudxn.GetDeviceData(),
                                        dyudyn.GetDeviceData(),
                                        dzudzn.GetDeviceData());

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of ComputeVelocityGradientNonuniform
//--------------------------------------------------------------------------------------------------


/**
 * CUDA kernel to add initial pressure p0 into p, rhox, rhoy, rhoz.
 * c is a matrix. Heterogeneity is treated by a template
 * @param [out] p       - pressure
 * @param [out] rhox
 * @param [out] rhoy
 * @param [out] rhoz
 * @param [in]  p0       - intial pressure
 * @param [in]  c2       - sound speed
 */
template<bool Is_c0_scalar>
__global__ void CUDACompute_p0_AddInitialPressure(float*       p,
                                                  float*       rhox,
                                                  float*       rhoy,
                                                  float*       rhoz,
                                                  const float* p0,
                                                  const float* c2 = nullptr)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    float tmp = p[i] = p0[i];

    tmp = (Is_c0_scalar) ? tmp / (3.0f * cudaDeviceConstants.c2): tmp / (3.0f * c2[i]);

    rhox[i] = tmp;
    rhoy[i] = tmp;
    rhoz[i] = tmp;
  }
}// end of CUDACompute_p0_AddInitialPressure
//--------------------------------------------------------------------------------------------------


/**
 * Interface for kernel to add initial pressure p0 into p, rhox, rhoy, rhoz.
 *
 * @param [out] p            - Pressure
 * @param [out] rhox         - Density component
 * @param [out] rhoy         - Density component
 * @param [out] rhoz         - Density component
 * @param [in]  p0           - intial pressure
 * @param [in]  Is_c2_scalar - Scalar or vector?
 * @param [in]  c2           - Sound speed
 */
void SolverCUDAKernels::Compute_p0_AddInitialPressure(TRealMatrix&       p,
                                                      TRealMatrix&       rhox,
                                                      TRealMatrix&       rhoy,
                                                      TRealMatrix&       rhoz,
                                                      const TRealMatrix& p0,
                                                      const bool         Is_c2_scalar,
                                                      const float*       c2)
{
  if (Is_c2_scalar)
  {
    CUDACompute_p0_AddInitialPressure<true>
                                     <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                     (p.GetDeviceData(),
                                      rhox.GetDeviceData(),
                                      rhoy.GetDeviceData(),
                                      rhoz.GetDeviceData(),
                                      p0.GetDeviceData());
  }
  else
  {
      CUDACompute_p0_AddInitialPressure<false>
                                       <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                       (p.GetDeviceData(),
                                        rhox.GetDeviceData(),
                                        rhoy.GetDeviceData(),
                                        rhoz.GetDeviceData(),
                                        p0.GetDeviceData(),
                                        c2);
  }
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of Compute_p0_AddInitialPressure
//--------------------------------------------------------------------------------------------------

/**
 * Interface to kernel which calculate new values of rho (acoustic density).
 * Non-linear, homogenous case.
 * @param [out] rhox    - density x
 * @param [out] rhoy    - density y
 * @param [out] rhoz    - density y
 * @param [in]  pml_x   - pml x
 * @param [in]  pml_y   - pml y
 * @param [in]  pml_z   - pml z
 * @param [in]  duxdx   - gradient of velocity x
 * @param [in]  duydy   - gradient of velocity x
 * @param [in]  duzdz   - gradient of velocity z
 */
__global__ void CUDAComputeDensityNonlinearHomogeneous(float*       rhox,
                                                       float*       rhoy,
                                                       float*       rhoz,
                                                       const float* pml_x,
                                                       const float* pml_y,
                                                       const float* pml_z,
                                                       const float* duxdx,
                                                       const float* duydy,
                                                       const float* duzdz)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getReal3DCoords(i);

    const float pmlxEl = pml_x[coords.x];
    const float pmlyEl = pml_y[coords.y];
    const float pmlzEl = pml_z[coords.z];

    const float rhoxEl = rhox[i];
    const float rhoyEl = rhoy[i];
    const float rhozEl = rhoz[i];

    const float sumRhosDt = (2.0f * (rhoxEl + rhoyEl + rhozEl) + cudaDeviceConstants.rho0) *
                            cudaDeviceConstants.dt;

    rhox[i] = pmlxEl * ((pmlxEl * rhoxEl) - sumRhosDt * duxdx[i]);
    rhoy[i] = pmlyEl * ((pmlyEl * rhoyEl) - sumRhosDt * duydy[i]);
    rhoz[i] = pmlzEl * ((pmlzEl * rhozEl) - sumRhosDt * duzdz[i]);
  }
}// end of CUDAComputeDensityNonlinearHomogeneous
//--------------------------------------------------------------------------------------------------

/**
 * Interface to kernel which calculate new values of rho (acoustic density).
 * Non-linear, homogenous case.
 *
 * @param [out] rhox  - density x
 * @param [out] rhoy  - density y
 * @param [out] rhoz  - density y
 * @param [in]  pml_x - pml x
 * @param [in]  pml_y - pml y
 * @param [in]  pml_z - pml z
 * @param [in]  duxdx - gradient of velocity x
 * @param [in]  duydy - gradient of velocity x
 * @param [in]  duzdz - gradient of velocity z
 */
void SolverCUDAKernels::ComputeDensityNonlinearHomogeneous(TRealMatrix&       rhox,
                                                           TRealMatrix&       rhoy,
                                                           TRealMatrix&       rhoz,
                                                           const TRealMatrix& pml_x,
                                                           const TRealMatrix& pml_y,
                                                           const TRealMatrix& pml_z,
                                                           const TRealMatrix& duxdx,
                                                           const TRealMatrix& duydy,
                                                           const TRealMatrix& duzdz)
{
  CUDAComputeDensityNonlinearHomogeneous<<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                        (rhox.GetDeviceData(),
                                         rhoy.GetDeviceData(),
                                         rhoz.GetDeviceData(),
                                         pml_x.GetDeviceData(),
                                         pml_y.GetDeviceData(),
                                         pml_z.GetDeviceData(),
                                         duxdx.GetDeviceData(),
                                         duydy.GetDeviceData(),
                                         duzdz.GetDeviceData());
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of ComputeDensityNonlinearHomogeneous
//--------------------------------------------------------------------------------------------------

/**
 * CUDA kernel which calculate new values of rho (acoustic density).
 * Non-linear, heterogenous case.
 * @param [out] rhox  - density x
 * @param [out] rhoy  - density y
 * @param [out] rhoz  - density y
 * @param [in]  pml_x - pml x
 * @param [in]  pml_y - pml y
 * @param [in]  pml_z - pml z
 * @param [in]  duxdx - gradient of velocity x
 * @param [in]  duydy - gradient of velocity x
 * @param [in]  duzdz - gradient of velocity z
 * @param [in]  rho0  - initial density (matrix here)
 */
__global__ void CUDAComputeDensityNonlinearHeterogeneous(float*       rhox,
                                                         float*       rhoy,
                                                         float*       rhoz,
                                                         const float* pml_x,
                                                         const float* pml_y,
                                                         const float* pml_z,
                                                         const float* duxdx,
                                                         const float* duydy,
                                                         const float* duzdz,
                                                         const float* rho0)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getReal3DCoords(i);

    const float pmlxEl = pml_x[coords.x];
    const float pmlyEl = pml_y[coords.y];
    const float pmlzEl = pml_z[coords.z];

    const float rhoxEl = rhox[i];
    const float rhoyEl = rhoy[i];
    const float rhozEl = rhoz[i];

    const float sumRhosDt = (2.0f * (rhoxEl + rhoyEl + rhozEl) + rho0[i]) *
                            cudaDeviceConstants.dt;

    rhox[i] = pmlxEl * ((pmlxEl * rhoxEl) - sumRhosDt * duxdx[i]);
    rhoy[i] = pmlyEl * ((pmlyEl * rhoyEl) - sumRhosDt * duydy[i]);
    rhoz[i] = pmlzEl * ((pmlzEl * rhozEl) - sumRhosDt * duzdz[i]);
  }
}//end of CUDAComputeDensityNonlinearHeterogeneous
//--------------------------------------------------------------------------------------------------

/**
 * Interface to kernel which calculate new values of rho (acoustic density).
 * Non-linear, heterogenous case.
 * @param [out] rhox  - density x
 * @param [out] rhoy  - density y
 * @param [out] rhoz  - density y
 * @param [in]  pml_x - pml x
 * @param [in]  pml_y - pml y
 * @param [in]  pml_z - pml z
 * @param [in]  duxdx - gradient of velocity x
 * @param [in]  duydy - gradient of velocity x
 * @param [in]  duzdz - gradient of velocity z
 * @param [in]  rho0  - initial density (matrix here)
 */
void SolverCUDAKernels::ComputeDensityNonlinearHeterogeneous(TRealMatrix&       rhox,
                                                             TRealMatrix&       rhoy,
                                                             TRealMatrix&       rhoz,
                                                             const TRealMatrix& pml_x,
                                                             const TRealMatrix& pml_y,
                                                             const TRealMatrix& pml_z,
                                                             const TRealMatrix& duxdx,
                                                             const TRealMatrix& duydy,
                                                             const TRealMatrix& duzdz,
                                                             const TRealMatrix& rho0)
{
  CUDAComputeDensityNonlinearHeterogeneous<<<GetSolverGridSize1D(), GetSolverBlockSize1D() >>>
                                          (rhox.GetDeviceData(),
                                           rhoy.GetDeviceData(),
                                           rhoz.GetDeviceData(),
                                           pml_x.GetDeviceData(),
                                           pml_y.GetDeviceData(),
                                           pml_z.GetDeviceData(),
                                           duxdx.GetDeviceData(),
                                           duydy.GetDeviceData(),
                                           duzdz.GetDeviceData(),
                                           rho0.GetDeviceData());

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of ComputeDensityNonlinearHeterogeneous
//--------------------------------------------------------------------------------------------------

/**
 * Interface to kernel which calculate new values of rho (acoustic density).
 * Linear, homogenous case.
 *
 * @param [out] rhox    - Density x
 * @param [out] rhoy    - Density y
 * @param [out] rhoz    - Density y
 * @param [in]  pml_x   - pml x
 * @param [in]  pml_y   - pml y
 * @param [in]  pml_z   - pml z
 * @param [in]  duxdx   - Gradient of velocity x
 * @param [in]  duydy   - Gradient of velocity x
 * @param [in]  duzdz   - Gradient of velocity z
 */
__global__ void CUDAComputeDensityLinearHomogeneous(float*       rhox,
                                                    float*       rhoy,
                                                    float*       rhoz,
                                                    const float* pml_x,
                                                    const float* pml_y,
                                                    const float* pml_z,
                                                    const float* duxdx,
                                                    const float* duydy,
                                                    const float* duzdz)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getReal3DCoords(i);

    const float pml_x_el = pml_x[coords.x];
    const float pml_y_el = pml_y[coords.y];
    const float pml_z_el = pml_z[coords.z];

    rhox[i] = pml_x_el * (pml_x_el * rhox[i] - cudaDeviceConstants.dtRho0 * duxdx[i]);
    rhoy[i] = pml_y_el * (pml_y_el * rhoy[i] - cudaDeviceConstants.dtRho0 * duydy[i]);
    rhoz[i] = pml_z_el * (pml_z_el * rhoz[i] - cudaDeviceConstants.dtRho0 * duzdz[i]);
  }
}// end of CUDAComputeDensityLinearHomogeneous
//--------------------------------------------------------------------------------------------------

/**
 * Interface to kernel which calculate new values of rho (acoustic density).
 * Linear, homogenous case.
 * @param [out] rhox  - density x
 * @param [out] rhoy  - density y
 * @param [out] rhoz  - density y
 * @param [in]  pml_x - pml x
 * @param [in]  pml_y - pml y
 * @param [in]  pml_z - pml z
 * @param [in]  duxdx - gradient of velocity x
 * @param [in]  duydy - gradient of velocity x
 * @param [in]  duzdz - gradient of velocity z
 */
void SolverCUDAKernels::ComputeDensityLinearHomogeneous(TRealMatrix&       rhox,
                                                        TRealMatrix&       rhoy,
                                                        TRealMatrix&       rhoz,
                                                        const TRealMatrix& pml_x,
                                                        const TRealMatrix& pml_y,
                                                        const TRealMatrix& pml_z,
                                                        const TRealMatrix& duxdx,
                                                        const TRealMatrix& duydy,
                                                        const TRealMatrix& duzdz)
{
  CUDAComputeDensityLinearHomogeneous<<<GetSolverGridSize1D(), GetSolverBlockSize1D() >>>
                                     (rhox.GetDeviceData(),
                                      rhoy.GetDeviceData(),
                                      rhoz.GetDeviceData(),
                                      pml_x.GetDeviceData(),
                                      pml_y.GetDeviceData(),
                                      pml_z.GetDeviceData(),
                                      duxdx.GetDeviceData(),
                                      duydy.GetDeviceData(),
                                      duzdz.GetDeviceData());
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of ComputeDensityLinearHomogeneous
//--------------------------------------------------------------------------------------------------

/**
 * CUDA kernel which calculate new values of rho (acoustic density).
 * Linear, heterogenous case.
 *
 * @param [out] rhox  - density x
 * @param [out] rhoy  - density y
 * @param [out] rhoz  - density y
 * @param [in]  pml_x - pml x
 * @param [in]  pml_y - pml y
 * @param [in]  pml_z - pml z
 * @param [in]  duxdx - gradient of velocity x
 * @param [in]  duydy - gradient of velocity x
 * @param [in]  duzdz - gradient of velocity z
 * @param [in]  rho0  - initial density (matrix here)
 */
__global__ void CUDAComputeDensityLinearHeterogeneous(float*       rhox,
                                                      float*       rhoy,
                                                      float*       rhoz,
                                                      const float* pml_x,
                                                      const float* pml_y,
                                                      const float* pml_z,
                                                      const float* duxdx,
                                                      const float* duydy,
                                                      const float* duzdz,
                                                      const float* rho0)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getReal3DCoords(i);

    const float pml_x_el = pml_x[coords.x];
    const float pml_y_el = pml_y[coords.y];
    const float pml_z_el = pml_z[coords.z];

    const float dt_rho0  = cudaDeviceConstants.dt * rho0[i];

    rhox[i] = pml_x_el * (pml_x_el * rhox[i] - dt_rho0 * duxdx[i]);
    rhoy[i] = pml_y_el * (pml_y_el * rhoy[i] - dt_rho0 * duydy[i]);
    rhoz[i] = pml_z_el * (pml_z_el * rhoz[i] - dt_rho0 * duzdz[i]);
  }
}// end of CUDACompute_rhoxyz_linear_heterogeneous
//--------------------------------------------------------------------------------------------------

/**
 * Interface to kernel which calculate new values of rho (acoustic density).
 * Linear, heterogenous case.
 * @param [out] rhox  - Density x
 * @param [out] rhoy  - Density y
 * @param [out] rhoz  - Density y
 * @param [in]  pml_x - pml x
 * @param [in]  pml_y - pml y
 * @param [in]  pml_z - pml z
 * @param [in]  duxdx - Gradient of velocity x
 * @param [in]  duydy - Gradient of velocity x
 * @param [in]  duzdz - Gradient of velocity z
 * @param [in]  rho0  - initial density (matrix here)
 */
void SolverCUDAKernels::ComputeDensityLinearHeterogeneous(TRealMatrix&       rhox,
                                                          TRealMatrix&       rhoy,
                                                          TRealMatrix&       rhoz,
                                                          const TRealMatrix& pml_x,
                                                          const TRealMatrix& pml_y,
                                                          const TRealMatrix& pml_z,
                                                          const TRealMatrix& duxdx,
                                                          const TRealMatrix& duydy,
                                                          const TRealMatrix& duzdz,
                                                          const TRealMatrix& rho0)
{
  CUDAComputeDensityLinearHeterogeneous<<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                       (rhox.GetDeviceData(),
                                        rhoy.GetDeviceData(),
                                        rhoz.GetDeviceData(),
                                        pml_x.GetDeviceData(),
                                        pml_y.GetDeviceData(),
                                        pml_z.GetDeviceData(),
                                        duxdx.GetDeviceData(),
                                        duydy.GetDeviceData(),
                                        duzdz.GetDeviceData(),
                                        rho0.GetDeviceData());

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of ComputeDensityLinearHeterogeneous
//--------------------------------------------------------------------------------------------------


/**
 * CUDA kernel which calculates three temporary sums in the new pressure formula \n
 * non-linear absorbing case. Homogeneous and heterogenous variants are treated using templates.
 * Homogeneous variables are in constant memory.
 *
 * @param [out] rho_sum      - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [out] BonA_sum     - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [out] du_sum       - rho0* (duxdx + duydy + duzdz)
 * @param [in]  rhox,        - Acoustic density X
 * @param [in]  rhoy,        - Acoustic density Y
 * @param [in]  rhoz,        - Acoustic density Z
 * @param [in]  duxdx        - Gradient of velocity in X
 * @param [in]  duydy        - Gradient of velocity in X
 * @param [in]  duzdz        - Gradient of velocity in X
 * @param [in]  BonA_matrix  - Heterogeneous value for BonA
 * @param [in]  rho0_matrix  - Heterogeneous value for rho0
 *
 *
 */
template <bool is_BonA_scalar, bool is_rho0_scalar>
__global__ void CUDAComputePressurePartsNonLinear(float*       rho_sum,
                                                  float*       BonA_sum,
                                                  float*       du_sum,
                                                  const float* rhox,
                                                  const float* rhoy,
                                                  const float* rhoz,
                                                  const float* duxdx,
                                                  const float* duydy,
                                                  const float* duzdz,
                                                  const float* BonA_matrix,
                                                  const float* rho0_matrix)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float BonA = (is_BonA_scalar) ? cudaDeviceConstants.bOnA : BonA_matrix[i];
    const float rho0 = (is_rho0_scalar) ? cudaDeviceConstants.rho0 : rho0_matrix[i];

    const float rho_xyz_el = rhox[i] + rhoy[i] + rhoz[i];

    rho_sum[i]  = rho_xyz_el;
    BonA_sum[i] = ((BonA * rho_xyz_el * rho_xyz_el) / (2.0f * rho0)) + rho_xyz_el;
    du_sum[i]   = rho0 * (duxdx[i] + duydy[i] + duzdz[i]);
    }
}// end of CUDACalculate_SumRho_BonA_SumDu
//--------------------------------------------------------------------------------------------------

/**
 * Interface to kernel which calculates three temporary sums in the new pressure formula \n
 * non-linear absorbing case. Scalar values are in constant memory
 *
 * @param [out] rho_sum        - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [out] BonA_sum       - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [out] du_sum         - rho0* (duxdx + duydy + duzdz)
 * @param [in]  rhox,          - Acoustic density X
 * @param [in]  rhoy,          - Acoustic density Y
 * @param [in]  rhoz,          - Acoustic density Z
 * @param [in]  duxdx          - Gradient of velocity in X
 * @param [in]  duydy          - Gradient of velocity in X
 * @param [in]  duzdz          - Gradient of velocity in X
 * @param [in]  is_BonA_scalar - Is BonA a scalar value (homogeneous)
 * @param [in]  BonA_matrix    - Heterogeneous value for BonA
 * @param [in]  is_rho0_scalar - Is rho0 a scalar value (homogeneous)
 * @param [in]  rho0_matrix    - Heterogeneous value for rho0
 */
void SolverCUDAKernels::ComputePressurePartsNonLinear(TRealMatrix&       rho_sum,
                                                      TRealMatrix&       BonA_sum,
                                                      TRealMatrix&       du_sum,
                                                      const TRealMatrix& rhox,
                                                      const TRealMatrix& rhoy,
                                                      const TRealMatrix& rhoz,
                                                      const TRealMatrix& duxdx,
                                                      const TRealMatrix& duydy,
                                                      const TRealMatrix& duzdz,
                                                      const bool         is_BonA_scalar,
                                                      const float*       BonA_matrix,
                                                      const bool         is_rho0_scalar,
                                                      const float*       rho0_matrix)
{
  // all variants are treated by templates, here you can see all 4 variants
  if (is_BonA_scalar)
  {
    if (is_rho0_scalar)
    {
      CUDAComputePressurePartsNonLinear<true, true>
                                     <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                     (rho_sum.GetDeviceData(),
                                      BonA_sum.GetDeviceData(),
                                      du_sum.GetDeviceData(),
                                      rhox.GetDeviceData(),
                                      rhoy.GetDeviceData(),
                                      rhoz.GetDeviceData(),
                                      duxdx.GetDeviceData(),
                                      duydy.GetDeviceData(),
                                      duzdz.GetDeviceData(),
                                      BonA_matrix,
                                      rho0_matrix);
    }
    else
    {
      CUDAComputePressurePartsNonLinear<true, false>
                                       <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                       (rho_sum.GetDeviceData(),
                                        BonA_sum.GetDeviceData(),
                                        du_sum.GetDeviceData(),
                                        rhox.GetDeviceData(),
                                        rhoy.GetDeviceData(),
                                        rhoz.GetDeviceData(),
                                        duxdx.GetDeviceData(),
                                        duydy.GetDeviceData(),
                                        duzdz.GetDeviceData(),
                                        BonA_matrix,
                                        rho0_matrix);
    }
  }
  else // BonA is false
  {
   if (is_rho0_scalar)
    {
    CUDAComputePressurePartsNonLinear<false, true>
                                     <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                     (rho_sum.GetDeviceData(),
                                      BonA_sum.GetDeviceData(),
                                      du_sum.GetDeviceData(),
                                      rhox.GetDeviceData(),
                                      rhoy.GetDeviceData(),
                                      rhoz.GetDeviceData(),
                                      duxdx.GetDeviceData(),
                                      duydy.GetDeviceData(),
                                      duzdz.GetDeviceData(),
                                      BonA_matrix,
                                      rho0_matrix);
    }
    else
    {
    CUDAComputePressurePartsNonLinear<false, false>
                                     <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                     (rho_sum.GetDeviceData(),
                                      BonA_sum.GetDeviceData(),
                                      du_sum.GetDeviceData(),
                                      rhox.GetDeviceData(),
                                      rhoy.GetDeviceData(),
                                      rhoz.GetDeviceData(),
                                      duxdx.GetDeviceData(),
                                      duydy.GetDeviceData(),
                                      duzdz.GetDeviceData(),
                                      BonA_matrix,
                                      rho0_matrix);
    }
  }
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of ComputePressurePartsNonLinear
//--------------------------------------------------------------------------------------------------


/**
 * CUDA kernel which computes absorbing term with abosrb_nabla1 and  absorb_nabla2.
 * Calculate fft_1 = absorb_nabla1 .* fft_1 \n
 * Calculate fft_2 = absorb_nabla2 .* fft_2 \n
 *
 * @param [in,out] fft1 - Nabla1 part
 * @param [in,out] fft2 - Nabla2 part
 * @param [in]     nabla1
 * @param [in]     nabla2
 */
__global__ void CUDACompute_Absorb_nabla1_2(cuFloatComplex* fft1,
                                            cuFloatComplex* fft2,
                                            const float*    nabla1,
                                            const float*    nabla2)
{
  for(auto i = getIndex(); i < cudaDeviceConstants.nElementsComplex; i += getStride())
  {
    fft1[i] *= nabla1[i];
    fft2[i] *= nabla2[i];
  }
}// end of CUDACompute_Absorb_nabla1_2
//--------------------------------------------------------------------------------------------------

/**
 * Interface to kernel which computes absorbing term with abosrb_nabla1 and  absorb_nabla2. \n
 * Calculate fft_1 = absorb_nabla1 .* fft_1 \n
 * Calculate fft_2 = absorb_nabla2 .* fft_2 \n
 *
 * @param [in,out] fft1 - Nabla1 part
 * @param [in,out] fft2 - Nabla2 part
 * @param [in]     absorb_nabla1
 * @param [in]     absorb_nabla2
 */
void SolverCUDAKernels::ComputeAbsorbtionTerm(TCUFFTComplexMatrix& fft1,
                                              TCUFFTComplexMatrix& fft2,
                                              const TRealMatrix&   absorb_nabla1,
                                              const TRealMatrix&   absorb_nabla2)
{
  CUDACompute_Absorb_nabla1_2<<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                             (reinterpret_cast<cuFloatComplex*> (fft1.GetDeviceData()),
                              reinterpret_cast<cuFloatComplex*> (fft2.GetDeviceData()),
                              absorb_nabla1.GetDeviceData(),
                              absorb_nabla2.GetDeviceData());

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of ComputeAbsorbtionTerm
//--------------------------------------------------------------------------------------------------


/**
 * CUDA Sum sub-terms to calculate new pressure, non-linear case.
 *
 * @param [out] p           - new value of pressure
 * @param [in] BonA_temp    - rho0 * (duxdx + duydy + duzdz)
 * @param [in] c2_matrix
 * @param [in] absorb_tau
 * @param [in] tau_matrix
 * @param [in] absorb_eta   - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [in] eta_matrix
 */
template <bool is_c2_scalar, bool is_tau_eta_scalar>
__global__ void CUDASumPressureTermsNonlinear(float*       p,
                                              const float* BonA_temp,
                                              const float* c2_matrix,
                                              const float* absorb_tau,
                                              const float* tau_matrix,
                                              const float* absorb_eta,
                                              const float* eta_matrix)
{
  for(auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float c2  = (is_c2_scalar)      ? cudaDeviceConstants.c2  : c2_matrix[i];
    const float tau = (is_tau_eta_scalar) ? cudaDeviceConstants.absorbTau : tau_matrix[i];
    const float eta = (is_tau_eta_scalar) ? cudaDeviceConstants.absorbEta : eta_matrix[i];

    p[i] = c2 * (BonA_temp[i] + (cudaDeviceConstants.fftDivider *
                ((absorb_tau[i] * tau) - (absorb_eta[i] * eta))));
  }
}// end of CUDASumPressureTermsNonlinear
//--------------------------------------------------------------------------------------------------

/**
 * Interface to CUDA Sum sub-terms to calculate new pressure, non-linear case.
 * @param [in,out] p        - New value of pressure
 * @param [in] BonA_temp    - rho0 * (duxdx + duydy + duzdz)
 * @param [in] is_c2_scalar
 * @param [in] c2_matrix
 * @param [in] is_tau_eta_scalar
 * @param [in] absorb_tau
 * @param [in] tau_matrix
 * @param [in] absorb_eta   - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [in] eta_matrix
 */
void SolverCUDAKernels::SumPressureTermsNonlinear(TRealMatrix&       p,
                                                  const TRealMatrix& BonA_temp,
                                                  const bool         is_c2_scalar,
                                                  const float*       c2_matrix,
                                                  const bool         is_tau_eta_scalar,
                                                  const float*       absorb_tau,
                                                  const float*       tau_matrix,
                                                  const float*       absorb_eta,
                                                  const float*       eta_matrix)
{
  if (is_c2_scalar)
  {
    if (is_tau_eta_scalar)
    {
      CUDASumPressureTermsNonlinear<true, true>
                                   <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                   (p.GetDeviceData(),
                                    BonA_temp.GetDeviceData(),
                                    c2_matrix,
                                    absorb_tau,
                                    tau_matrix,
                                    absorb_eta,
                                    eta_matrix);
    }
    else
    {
      CUDASumPressureTermsNonlinear<true, false>
                                   <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                   (p.GetDeviceData(),
                                    BonA_temp.GetDeviceData(),
                                    c2_matrix,
                                    absorb_tau,
                                    tau_matrix,
                                    absorb_eta,
                                    eta_matrix);
    }
  }
  else
  { // c2 is matrix
     if (is_tau_eta_scalar)
    {
      CUDASumPressureTermsNonlinear<false, true>
                                   <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                   (p.GetDeviceData(),
                                    BonA_temp.GetDeviceData(),
                                    c2_matrix,
                                    absorb_tau,
                                    tau_matrix,
                                    absorb_eta,
                                    eta_matrix);
    }
    else
    {
      CUDASumPressureTermsNonlinear<false, false>
                                   <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                   (p.GetDeviceData(),
                                    BonA_temp.GetDeviceData(),
                                    c2_matrix,
                                    absorb_tau,
                                    tau_matrix,
                                    absorb_eta,
                                    eta_matrix);
    }
  }
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of SumPressureTermsNonlinear
//--------------------------------------------------------------------------------------------------


/**
 * CUDA kernel that sums sub-terms to calculate new pressure, linear case.
 *
 * @param [out] p              - new value of p
 * @param [in] absorb_tau_temp - sub-term with absorb_tau
 * @param [in] absorb_eta_temp - sub-term with absorb_eta
 * @param [in] sum_rhoxyz      - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [in] c2_matrix
 * @param [in] tau_matrix
 * @param [in] eta_matrix
 */
template <bool is_c2_scalar, bool is_tau_eta_scalar>
__global__ void CUDASumPressureTermsLinear(float*       p,
                                           const float* absorb_tau_temp,
                                           const float* absorb_eta_temp,
                                           const float* sum_rhoxyz,
                                           const float* c2_matrix,
                                           const float* tau_matrix,
                                           const float* eta_matrix)
{
  for(auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float c2  = (is_c2_scalar)      ? cudaDeviceConstants.c2                : c2_matrix[i];
    const float tau = (is_tau_eta_scalar) ? cudaDeviceConstants.absorbTau : tau_matrix[i];
    const float eta = (is_tau_eta_scalar) ? cudaDeviceConstants.absorbEta : eta_matrix[i];

    p[i] = c2 * (sum_rhoxyz[i] + (cudaDeviceConstants.fftDivider *
                (absorb_tau_temp[i] * tau - absorb_eta_temp[i] * eta)));
  }
}// end of CUDASumPressureTermsLinear
//--------------------------------------------------------------------------------------------------


/**
 * Interface to kernel that sums sub-terms to calculate new pressure, linear case.
 * @param [out] p              - New value of p
 * @param [in] absorb_tau_temp - Sub-term with absorb_tau
 * @param [in] absorb_eta_temp - Sub-term with absorb_eta
 * @param [in] sum_rhoxyz      - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [in] is_c2_scalar
 * @param [in] c2_matrix
 * @param [in] is_tau_eta_scalar
 * @param [in] tau_matrix
 * @param [in] eta_matrix
 */
void SolverCUDAKernels::SumPressureTermsLinear(TRealMatrix&       p,
                                               const TRealMatrix& absorb_tau_temp,
                                               const TRealMatrix& absorb_eta_temp,
                                               const TRealMatrix& sum_rhoxyz,
                                               const bool         is_c2_scalar,
                                               const float*       c2_matrix,
                                               const bool         is_tau_eta_scalar,
                                               const float*       tau_matrix,
                                               const float*       eta_matrix)
{
  if (is_c2_scalar)
  {
    if (is_tau_eta_scalar)
    {
      CUDASumPressureTermsLinear<true,true>
                                <<<GetSolverGridSize1D(), GetSolverBlockSize1D() >>>
                                (p.GetDeviceData(),
                                 absorb_tau_temp.GetDeviceData(),
                                 absorb_eta_temp.GetDeviceData(),
                                 sum_rhoxyz.GetDeviceData(),
                                 c2_matrix,
                                 tau_matrix,
                                 eta_matrix);
    }
    else
    {
      CUDASumPressureTermsLinear<true,false>
                                <<<GetSolverGridSize1D(), GetSolverBlockSize1D() >>>
                                (p.GetDeviceData(),
                                 absorb_tau_temp.GetDeviceData(),
                                 absorb_eta_temp.GetDeviceData(),
                                 sum_rhoxyz.GetDeviceData(),
                                 c2_matrix,
                                 tau_matrix,
                                 eta_matrix);
    }
   }
  else
  {
    if (is_tau_eta_scalar)
    {
      CUDASumPressureTermsLinear<false,true>
                                <<<GetSolverGridSize1D(), GetSolverBlockSize1D() >>>
                                (p.GetDeviceData(),
                                 absorb_tau_temp.GetDeviceData(),
                                 absorb_eta_temp.GetDeviceData(),
                                 sum_rhoxyz.GetDeviceData(),
                                 c2_matrix,
                                 tau_matrix,
                                 eta_matrix);
    }
    else
    {
      CUDASumPressureTermsLinear<false,false>
                                <<<GetSolverGridSize1D(), GetSolverBlockSize1D() >>>
                                (p.GetDeviceData(),
                                 absorb_tau_temp.GetDeviceData(),
                                 absorb_eta_temp.GetDeviceData(),
                                 sum_rhoxyz.GetDeviceData(),
                                 c2_matrix,
                                 tau_matrix,
                                 eta_matrix);
    }
  }
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of SumPressureTermsLinea
//--------------------------------------------------------------------------------------------------


/**
 * CUDA kernel that sums sub-terms for new p, non-linear lossless case.
 *
 * @param [out] p           - New value of pressure
 * @param [in]  rhox
 * @param [in]  rhoy
 * @param [in]  rhoz
 * @param [in]  c2_matrix
 * @param [in]  BonA_matrix
 * @param [in]  rho0_matrix
 */
template<bool is_c2_scalar, bool is_BonA_scalar, bool is_rho0_scalar>
__global__ void CUDASumPressureNonlinearLossless(float*       p,
                                                 const float* rhox,
                                                 const float* rhoy,
                                                 const float* rhoz,
                                                 const float* c2_matrix,
                                                 const float* BonA_matrix,
                                                 const float* rho0_matrix)
{
  for(auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float c2   = (is_c2_scalar)   ? cudaDeviceConstants.c2          : c2_matrix[i];
    const float BonA = (is_BonA_scalar) ? cudaDeviceConstants.bOnA : BonA_matrix[i];
    const float rho0 = (is_rho0_scalar) ? cudaDeviceConstants.rho0 : rho0_matrix[i];

    const float sum_rho = rhox[i] + rhoy[i] + rhoz[i];

    p[i] = c2 * (sum_rho + (BonA * (sum_rho * sum_rho) / (2.0f * rho0)));
  }
}// end of CUDASum_new_p_nonlinear_lossless
//--------------------------------------------------------------------------------------------------

/**
 * Interface to kernel that sums sub-terms for new p, non-linear lossless case.
 * @param [out] p           - New value of pressure
 * @param [in]  rhox
 * @param [in]  rhoy
 * @param [in]  rhoz
 * @param [in]  is_c2_scalar
 * @param [in]  c2_matrix
 * @param [in]  is_BonA_scalar
 * @param [in]  BonA_matrix
 * @param [in]  is_rho0_scalar
 * @param [in]  rho0_matrix
 */
void SolverCUDAKernels::SumPressureNonlinearLossless(TRealMatrix&       p,
                                                     const TRealMatrix& rhox,
                                                     const TRealMatrix& rhoy,
                                                     const TRealMatrix& rhoz,
                                                     const bool         is_c2_scalar,
                                                     const float*       c2_matrix,
                                                     const bool         is_BonA_scalar,
                                                     const float*       BonA_matrix,
                                                     const bool         is_rho0_scalar,
                                                     const float*       rho0_matrix)
{
  if (is_c2_scalar)
  {
    if (is_BonA_scalar)
    {
      if (is_rho0_scalar)
      {
        CUDASumPressureNonlinearLossless<true, true, true>
                                        <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                        (p.GetDeviceData(),
                                         rhox.GetDeviceData(),
                                         rhoy.GetDeviceData(),
                                         rhoz.GetDeviceData(),
                                         c2_matrix,
                                         BonA_matrix,
                                         rho0_matrix);
      }
      else
      {
        CUDASumPressureNonlinearLossless<true, true, false>
                                        <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                        (p.GetDeviceData(),
                                         rhox.GetDeviceData(),
                                         rhoy.GetDeviceData(),
                                         rhoz.GetDeviceData(),
                                         c2_matrix,
                                         BonA_matrix,
                                         rho0_matrix);
      }
    }// is_BonA_scalar= true
    else
    {
      if (is_rho0_scalar)
      {
        CUDASumPressureNonlinearLossless<true, false, true>
                                        <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                        (p.GetDeviceData(),
                                         rhox.GetDeviceData(),
                                         rhoy.GetDeviceData(),
                                         rhoz.GetDeviceData(),
                                         c2_matrix,
                                         BonA_matrix,
                                         rho0_matrix);
      }
      else
      {
        CUDASumPressureNonlinearLossless<true, false, false>
                                        <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                        (p.GetDeviceData(),
                                         rhox.GetDeviceData(),
                                         rhoy.GetDeviceData(),
                                         rhoz.GetDeviceData(),
                                         c2_matrix,
                                         BonA_matrix,
                                         rho0_matrix);
      }
    }
  }
  else
  { // is_c2_scalar == false
   if (is_BonA_scalar)
    {
      if (is_rho0_scalar)
      {
        CUDASumPressureNonlinearLossless<false, true, true>
                                        <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                        (p.GetDeviceData(),
                                         rhox.GetDeviceData(),
                                         rhoy.GetDeviceData(),
                                         rhoz.GetDeviceData(),
                                         c2_matrix,
                                         BonA_matrix,
                                         rho0_matrix);
      }
      else
      {
        CUDASumPressureNonlinearLossless<false, true, false>
                                        <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                        (p.GetDeviceData(),
                                         rhox.GetDeviceData(),
                                         rhoy.GetDeviceData(),
                                         rhoz.GetDeviceData(),
                                         c2_matrix,
                                         BonA_matrix,
                                         rho0_matrix);
      }
    }// is_BonA_scalar= true
    else
    {
      if (is_rho0_scalar)
      {
        CUDASumPressureNonlinearLossless<false, false, true>
                                        <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                        (p.GetDeviceData(),
                                         rhox.GetDeviceData(),
                                         rhoy.GetDeviceData(),
                                         rhoz.GetDeviceData(),
                                         c2_matrix,
                                         BonA_matrix,
                                         rho0_matrix);
      }
      else
      {
        CUDASumPressureNonlinearLossless<false, false, false>
                                        <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                        (p.GetDeviceData(),
                                         rhox.GetDeviceData(),
                                         rhoy.GetDeviceData(),
                                         rhoz.GetDeviceData(),
                                         c2_matrix,
                                         BonA_matrix,
                                         rho0_matrix);
      }
    }
  }

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of SumPressureNonlinearLossless
//--------------------------------------------------------------------------------------------------


/**
 * CUDA kernel that Calculates two temporary sums in the new pressure
 * formula, linear absorbing case.
 *
 * @param [out] sum_rhoxyz  - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [out] sum_rho0_du - rho0* (duxdx + duydy + duzdz);
 * @param [in]  rhox
 * @param [in]  rhoy
 * @param [in]  rhoz
 * @param [in]  dux
 * @param [in]  duy
 * @param [in]  duz
 * @param [in]  rho0_matrix
 */
template<bool is_rho0_scalar>
__global__ void CUDAComputePressurePartsLinear(float*       sum_rhoxyz,
                                               float*       sum_rho0_du,
                                               const float* rhox,
                                               const float* rhoy,
                                               const float* rhoz,
                                               const float* dux,
                                               const float* duy,
                                               const float* duz,
                                               const float* rho0_matrix)
{
  for(auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float rho0 = (is_rho0_scalar) ? cudaDeviceConstants.rho0 : rho0_matrix[i];

    sum_rhoxyz[i]  = rhox[i] + rhoy[i] + rhoz[i];
    sum_rho0_du[i] = rho0 * (dux[i] + duy[i] + duz[i]);
  }
}// end of CUDACalculate_SumRho_SumRhoDu
//------------------------------------------------------------------------------

/**
 * Interface to kernel that Calculates two temporary sums in the new pressure
 * formula, linear absorbing case.
 * @param [out] sum_rhoxyz  - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [out] sum_rho0_du - rho0* (duxdx + duydy + duzdz);
 * @param [in]  rhox
 * @param [in]  rhoy
 * @param [in]  rhoz
 * @param [in]  duxdx
 * @param [in]  duydy
 * @param [in]  duzdz
 * @param [in]  is_rho0_scalar
 * @param [in]  rho0_matrix
 */
void SolverCUDAKernels::ComputePressurePartsLinear(TRealMatrix&       sum_rhoxyz,
                                                   TRealMatrix&       sum_rho0_du,
                                                   const TRealMatrix& rhox,
                                                   const TRealMatrix& rhoy,
                                                   const TRealMatrix& rhoz,
                                                   const TRealMatrix& duxdx,
                                                   const TRealMatrix& duydy,
                                                   const TRealMatrix& duzdz,
                                                   const bool         is_rho0_scalar,
                                                   const float*       rho0_matrix)
{
  if (is_rho0_scalar)
  {
   CUDAComputePressurePartsLinear<true>
                                <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                (sum_rhoxyz.GetDeviceData(),
                                 sum_rho0_du.GetDeviceData(),
                                 rhox.GetDeviceData(),
                                 rhoy.GetDeviceData(),
                                 rhoz.GetDeviceData(),
                                 duxdx.GetDeviceData(),
                                 duydy.GetDeviceData(),
                                 duzdz.GetDeviceData(),
                                 rho0_matrix);
  }
  else
  {
   CUDAComputePressurePartsLinear<false>
                                <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                (sum_rhoxyz.GetDeviceData(),
                                 sum_rho0_du.GetDeviceData(),
                                 rhox.GetDeviceData(),
                                 rhoy.GetDeviceData(),
                                 rhoz.GetDeviceData(),
                                 duxdx.GetDeviceData(),
                                 duydy.GetDeviceData(),
                                 duzdz.GetDeviceData(),
                                 rho0_matrix);
  }
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of Calculate_SumRho_SumRhoDu
//--------------------------------------------------------------------------------------------------

/**
 * CUDA kernel that sums sub-terms for new p, linear lossless case.
 *
 * @param [out] p
 * @param [in]  rhox
 * @param [in]  rhoy
 * @param [in]  rhoz
 * @param [in]  c2_matrix

 */
template <bool is_c2_scalar>
__global__ void CUDASum_new_p_linear_lossless(float*       p,
                                              const float* rhox,
                                              const float* rhoy,
                                              const float* rhoz,
                                              const float* c2_matrix)
{
  for(auto  i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float c2 = (is_c2_scalar) ? cudaDeviceConstants.c2 : c2_matrix[i];
    p[i] = c2 * (rhox[i] + rhoy[i] + rhoz[i]);
  }
}// end of CUDASum_new_p_linear_lossless
//--------------------------------------------------------------------------------------------------

/**
 * Interface to kernel that sums sub-terms for new p, linear lossless case.
 * @param [out] p
 * @param [in]  rhox
 * @param [in]  rhoy
 * @param [in]  rhoz
 * @param [in]  is_c2_scalar
 * @param [in]  c2_matrix
 */
void SolverCUDAKernels::SumPressureLinearLossless(TRealMatrix& p,
                                                  const TRealMatrix& rhox,
                                                  const TRealMatrix& rhoy,
                                                  const TRealMatrix& rhoz,
                                                  const bool         is_c2_scalar,
                                                  const float*       c2_matrix)
{
  if (is_c2_scalar)
  {
    CUDASum_new_p_linear_lossless<true>
                                 <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                 (p.GetDeviceData(),
                                  rhox.GetDeviceData(),
                                  rhoy.GetDeviceData(),
                                  rhoz.GetDeviceData(),
                                  c2_matrix);
  }
  else
  {
    CUDASum_new_p_linear_lossless<false>
                                 <<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                                 (p.GetDeviceData(),
                                  rhox.GetDeviceData(),
                                  rhoy.GetDeviceData(),
                                  rhoz.GetDeviceData(),
                                  c2_matrix);
  }
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of Sum_new_p_linear_lossless
//--------------------------------------------------------------------------------------------------


/**
 * CUDA kernel to transpose a 3D matrix in XY planes of any dimension sizes
 * Every block in a 1D grid transposes a few slabs.
 * Every block is composed of a 2D mesh of threads. The y dim gives the number of tiles processed
 * simultaneously. Each tile is processed by a single thread warp.
 * The shared memory is used to coalesce memory accesses and the padding is to eliminate bank
 * conflicts.  First the full tiles are transposed, then the remainder in the X, then Y and finally
 * the last bit in the bottom right corner.  \n
 * As a part of the transposition, the matrices can be padded to conform with cuFFT
 *
 * @tparam      padding      - Which matrices are padded (template parameter)
 * @tparam      isSquareSlab - Are the slabs of a square shape with sizes
 *                             divisible by the warp size
 * @tparam      warpSize     - Set the warp size. Built in value cannot be used
 *                             due to shared memory allocation
 *
 * @param [out] outputMatrix - Output matrix
 * @param [in]  inputMatrix  - Input  matrix
 * @param [in]  dimSizes     - Dimension sizes of the original matrix
 *
 * @warning A blockDim.x has to of a warp size (typically 32) \n
 *          blockDim.y should be between 1 and 4 (four tiles at once). blockDim.y
 *          has to be equal with the tilesAtOnce parameter.  \n
 *          blockDim.z must stay 1 \n
 *          Grid has to be organized (N, 1 ,1)
 *
 */
template<SolverCUDAKernels::TransposePadding padding,
         bool                                isSquareSlab,
         int                                 warpSize,
         int                                 tilesAtOnce>
__global__ void cudaTrasnposeReal3DMatrixXY(float*       outputMatrix,
                                            const float* inputMatrix,
                                            const dim3   dimSizes)
{
  // this size is fixed shared memory
  // we transpose tilesAtOnce tiles of warp size^2 at the same time, +1 solves bank conflicts
  volatile __shared__ float sharedTile[tilesAtOnce][warpSize][warpSize + 1];

  using TP = SolverCUDAKernels::TransposePadding;
  constexpr int inPad  = ((padding == TP::kInput)  || (padding == TP::kInputOutput)) ? 1 : 0;
  constexpr int outPad = ((padding == TP::kOutput) || (padding == TP::kInputOutput)) ? 1 : 0;

  const dim3 tileCount = {dimSizes.x / warpSize, dimSizes.y / warpSize, 1};

  // run over all slabs, one block per slab
  for (auto slabIdx = blockIdx.x; slabIdx < dimSizes.z; slabIdx += gridDim.x)
  {
    // calculate offset of the slab
    const float* inputSlab  = inputMatrix  + ((dimSizes.x + inPad) * dimSizes.y * slabIdx);
          float* outputSlab = outputMatrix + (dimSizes.x * (dimSizes.y + outPad) * slabIdx);

    dim3 tileIdx = {0, 0, 0};

    // go over all tiles in the row. Transpose tilesAtOnce rows at the same time
    for (tileIdx.y = threadIdx.y; tileIdx.y < tileCount.y; tileIdx.y += blockDim.y)
    {
      //--------------------------- full tiles in X --------------------------//
      // go over all full tiles in the row
      for (tileIdx.x = 0; tileIdx.x < tileCount.x; tileIdx.x++)
      {
        // Go over one tile and load data, unroll does not help
        for (auto row = 0; row < warpSize; row++)
        {
          sharedTile[threadIdx.y][row][threadIdx.x]
                    = inputSlab[(tileIdx.y * warpSize   + row) * (dimSizes.x + inPad) +
                                (tileIdx.x * warpSize)  + threadIdx.x];
        } // load data
        // no need for barrier - warp synchronous programming
        // Go over one tile and store data, unroll does not help
        for (auto row = 0; row < warpSize; row ++)
        {
          outputSlab[(tileIdx.x * warpSize   + row) * (dimSizes.y + outPad) +
                     (tileIdx.y * warpSize)  + threadIdx.x]
                    = sharedTile[threadIdx.y][threadIdx.x][row];

        } // store data
      } // tile X

      // the slab is not a square with edge sizes divisible by warpSize
      if (!isSquareSlab)
      {
        //--------------------------- reminders in X ---------------------------//
        // go over the remainder tile in X (those that can't fill warps)
        if ((tileCount.x * warpSize + threadIdx.x) < dimSizes.x)
        {
          for (auto row = 0; row < warpSize; row++)
          {
            sharedTile[threadIdx.y][row][threadIdx.x]
                      = inputSlab[(tileIdx.y   * warpSize   + row) * (dimSizes.x + inPad) +
                                  (tileCount.x * warpSize)  + threadIdx.x];
          }
        }// load

        // go over the remainder tile in X (those that can't fill a 32-warp)
        for (auto row = 0; (tileCount.x * warpSize + row) < dimSizes.x; row++)
        {
          outputSlab[(tileCount.x * warpSize   + row) * (dimSizes.y + outPad) +
                     (tileIdx.y   * warpSize)  + threadIdx.x]
                    = sharedTile[threadIdx.y][threadIdx.x][row];
        }// store
      } // isSquareSlab
    }// tile Y

    if (!isSquareSlab)
    {
      //--------------------------- reminders in Y ---------------------------//
      // go over the remainder tile in y (those that can't fill 32 warps)
      // go over all full tiles in the row, first in parallel
      for (tileIdx.x = threadIdx.y; tileIdx.x < tileCount.x; tileIdx.x += blockDim.y)
      {
        // go over the remainder tile in Y (only a few rows)
        for (auto row = 0; (tileCount.y * warpSize + row) < dimSizes.y; row++)
        {
          sharedTile[threadIdx.y][row][threadIdx.x]
                    = inputSlab[(tileCount.y * warpSize   + row) * (dimSizes.x + inPad) +
                                (tileIdx.x   * warpSize)  + threadIdx.x];
        } // load

        // go over the remainder tile in Y (and store only columns)
        if ((tileCount.y * warpSize + threadIdx.x) < dimSizes.y)
        {
          for (auto row = 0; row < warpSize ; row++)
          {
            outputSlab[(tileIdx.x   * warpSize   + row) * (dimSizes.y + outPad) +
                       (tileCount.y * warpSize)  + threadIdx.x]
                      = sharedTile[threadIdx.y][threadIdx.x][row];
          }
        }// store
      }// reminder Y

      //------------------------ reminder in X and Y -----------------------------//
      if (threadIdx.y == 0)
      {
      // go over the remainder tile in X and Y (only a few rows and colls)
        if ((tileCount.x * warpSize + threadIdx.x) < dimSizes.x)
        {
          for (auto row = 0; (tileCount.y * warpSize + row) < dimSizes.y; row++)
          {
            sharedTile[threadIdx.y][row][threadIdx.x]
                      = inputSlab[(tileCount.y * warpSize   + row) * (dimSizes.x + inPad) +
                                  (tileCount.x * warpSize)  + threadIdx.x];
          } // load
        }

        // go over the remainder tile in Y (and store only colls)
        if ((tileCount.y * warpSize + threadIdx.x) < dimSizes.y)
        {
          for (auto row = 0; (tileCount.x * warpSize + row) < dimSizes.x ; row++)
          {
            outputSlab[(tileCount.x * warpSize   + row) * (dimSizes.y + outPad) +
                       (tileCount.y * warpSize)  + threadIdx.x]
                      = sharedTile[threadIdx.y][threadIdx.x][row];
          }
        }// store
      }// reminder X  and Y
    } // kRectangle
  } //slab
}// end of cudaTrasnposeReal3DMatrixXY
//--------------------------------------------------------------------------------------------------


/**
 * Transpose a real 3D matrix in the X-Y direction. It is done out-of-place.
 *
 * @tparam      padding     - Which matrices are padded
 *
 * @param [out] outputMatrix - Output matrix data
 * @param [in]  inputMatrix  - Input  matrix data
 * @param [in]  dimSizes     - Dimension sizes of the original matrix
 */
template<SolverCUDAKernels::TransposePadding padding>
void SolverCUDAKernels::TrasposeReal3DMatrixXY(float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes)
{
  // fixed size at the moment, may be tuned based on the domain shape in the future
  // warpSize set to 32, and 4 tiles processed at once
  if ((dimSizes.x % 32 == 0) && (dimSizes.y % 32 == 0))
  {
    cudaTrasnposeReal3DMatrixXY<padding, true, 32, 4>
                               <<<GetSolverTransposeGirdSize(),GetSolverTransposeBlockSize()>>>
                               (outputMatrix, inputMatrix, dimSizes);

  }
  else
  {
    cudaTrasnposeReal3DMatrixXY<padding, false, 32, 4>
                               <<<GetSolverTransposeGirdSize(), GetSolverTransposeBlockSize() >>>
                               (outputMatrix, inputMatrix, dimSizes);
  }

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of TrasposeReal3DMatrixXY
//--------------------------------------------------------------------------------------------------

//------------------------ Explicit instances of TrasposeReal3DMatrixXY --------------------------//
/// Transpose a real 3D matrix in the X-Y direction, input matrix padded, output matrix compact
template
void SolverCUDAKernels::TrasposeReal3DMatrixXY<SolverCUDAKernels::TransposePadding::kInput>
                                              (float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Y direction, input matrix compact, output matrix padded
template
void SolverCUDAKernels::TrasposeReal3DMatrixXY<SolverCUDAKernels::TransposePadding::kOutput>
                                              (float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Y direction, input and output matrix compact
template
void SolverCUDAKernels::TrasposeReal3DMatrixXY<SolverCUDAKernels::TransposePadding::kNone>
                                              (float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes);
/// Transpose a real 3D matrix in the X-Y direction, input and output matrix padded
template
void SolverCUDAKernels::TrasposeReal3DMatrixXY<SolverCUDAKernels::TransposePadding::kInputOutput>
                                              (float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes);


/**
 * CUDA kernel to transpose a 3D matrix in XZ planes of any dimension sizes
 * Every block in a 1D grid transposes a few slabs.
  * Every block is composed of a 2D mesh of threads. The y dim gives the number of tiles processed
 * simultaneously. Each tile is processed by a single thread warp.
 * The shared memory is used to coalesce memory accesses and the padding is to eliminate bank
 * conflicts.  First the full tiles are transposed, then the remainder in the X, then Z and finally
 * the last bit in the bottom right corner.  \n
 * As a part of the transposition, the matrices can be padded to conform with cuFFT
 *
 *
 * @tparam      padding      - Which matrices are padded (template parameter)
 * @tparam      isSquareSlab - Are the slabs of a square shape with sizes
 *                             divisible by the warp size
 * @tparam      warpSize     - Set the warp size. Built in value cannot be used
 *                             due to shared memory allocation
 *
 * @param [out] outputMatrix - Output matrix
 * @param [in]  inputMatrix  - Input  matrix
 * @param [in]  dimSizes     - Dimension sizes of the original matrix
 *
 * @warning A blockDim.x has to of a warp size (typically 32) \n
 *          blockDim.y should be between 1 and 4 (four tiles at once). blockDim.y
 *          has to be equal with the tilesAtOnce parameter.  \n
 *          blockDim.z must stay 1 \n
 *          Grid has to be organized (N, 1 ,1)
 *
 */
template<SolverCUDAKernels::TransposePadding padding,
         bool                                isSquareSlab,
         int                                 warpSize,
         int                                 tilesAtOnce>
__global__ void cudaTrasnposeReal3DMatrixXZ(float*       outputMatrix,
                                            const float* inputMatrix,
                                            const dim3   dimSizes)
{
  // this size is fixed shared memory
  // we transpose tilesAtOnce tiles of warp size^2 at the same time, +1 solves bank conflicts
  volatile __shared__ float shared_tile[tilesAtOnce][warpSize][ warpSize + 1];

  using TP = SolverCUDAKernels::TransposePadding;
  constexpr int inPad  = ((padding == TP::kInput)  || (padding == TP::kInputOutput)) ? 1 : 0;
  constexpr int outPad = ((padding == TP::kOutput) || (padding == TP::kInputOutput)) ? 1 : 0;

  const dim3 tileCount = {dimSizes.x / warpSize, dimSizes.z / warpSize, 1};

  // run over all XZ slabs, one block per slab
  for (auto row = blockIdx.x; row < dimSizes.y; row += gridDim.x )
  {
    dim3 tileIdx = {0, 0, 0};

    // go over all all tiles in the XZ slab. Transpose multiple slabs at the same time (on per Z)
    for (tileIdx.y = threadIdx.y; tileIdx.y < tileCount.y; tileIdx.y += blockDim.y)
    {
      // go over all tiles in the row
      for (tileIdx.x = 0; tileIdx.x < tileCount.x; tileIdx.x++)
      {
        // Go over one tile and load data, unroll does not help
        for (auto slab = 0; slab < warpSize; slab++)
        {
          shared_tile[threadIdx.y][slab][threadIdx.x]
                     = inputMatrix[(tileIdx.y * warpSize + slab) * ((dimSizes.x + inPad) * dimSizes.y) +
                                    row * (dimSizes.x + inPad) +
                                    tileIdx.x * warpSize + threadIdx.x];
        } // load data

        // no need for barrier - warp synchronous programming

        // Go over one tile and store data, unroll does not help
        for (auto slab = 0; slab < warpSize; slab++)
        {
          outputMatrix[(tileIdx.x * warpSize + slab) * (dimSizes.y * (dimSizes.z + outPad)) +
                       (row * (dimSizes.z + outPad)) +
                       tileIdx.y * warpSize + threadIdx.x]
                      = shared_tile[threadIdx.y][threadIdx.x][slab];
        } // store data
      } // tile X

      // the slab is not a square with edge sizes divisible by warpSize
      if (!isSquareSlab)
      {
        //--------------------------- reminders in X ---------------------------//
        // go over the remainder tile in X (those that can't fill warps)
        if ((tileCount.x * warpSize + threadIdx.x) < dimSizes.x)
        {
          for (auto slab = 0; slab < warpSize; slab++)
          {
            shared_tile[threadIdx.y][slab][threadIdx.x]
                       = inputMatrix[(tileIdx.y * warpSize + slab) * ((dimSizes.x + inPad) * dimSizes.y) +
                                     (row * (dimSizes.x + inPad)) +
                                     tileCount.x * warpSize + threadIdx.x];
          }
        }// load

        // go over the remainder tile in X (those that can't fill warps)
        for (auto slab = 0; (tileCount.x * warpSize + slab) < dimSizes.x; slab++)
        {
          outputMatrix[(tileCount.x * warpSize + slab) * (dimSizes.y * (dimSizes.z + outPad)) +
                       (row * (dimSizes.z + outPad)) +
                       tileIdx.y * warpSize + threadIdx.x]
                      = shared_tile[threadIdx.y][threadIdx.x][slab];
        }// store
      } // isSquareSlab
    }// tile Y

    // the slab is not a square with edge sizes divisible by warpSize
    if (!isSquareSlab)
    {
      //--------------------------- reminders in Z -----------------------------//
      // go over the remainder tile in z (those that can't fill a -warp)
      // go over all full tiles in the row, first in parallel
      for (tileIdx.x = threadIdx.y; tileIdx.x < tileCount.x; tileIdx.x += blockDim.y)
      {
        // go over the remainder tile in Y (only a few rows)
        for (auto slab = 0; (tileCount.y * warpSize + slab) < dimSizes.z; slab++)
        {
          shared_tile[threadIdx.y][slab][threadIdx.x]
                     = inputMatrix[(tileCount.y * warpSize + slab) * ((dimSizes.x + inPad) * dimSizes.y) +
                                   (row * (dimSizes.x + inPad)) +
                                   tileIdx.x * warpSize + threadIdx.x];

        } // load

        // go over the remainder tile in Y (and store only columns)
        if ((tileCount.y * warpSize + threadIdx.x) < dimSizes.z)
        {
          for (auto slab = 0; slab < warpSize; slab++)
          {
            outputMatrix[(tileIdx.x * warpSize + slab) * (dimSizes.y * (dimSizes.z + outPad)) +
                         (row * (dimSizes.z + outPad)) +
                         tileCount.y * warpSize + threadIdx.x]
                        = shared_tile[threadIdx.y][threadIdx.x][slab];
          }
        }// store
      }// reminder Y

     //------------------------ reminder in X and Z -----------------------------//
    if (threadIdx.y == 0)
      {
      // go over the remainder tile in X and Y (only a few rows and colls)
        if ((tileCount.x * warpSize + threadIdx.x) < dimSizes.x)
        {
          for (auto slab = 0; (tileCount.y * warpSize + slab) < dimSizes.z; slab++)
          {
            shared_tile[threadIdx.y][slab][threadIdx.x]
                       = inputMatrix[(tileCount.y * warpSize + slab) * ((dimSizes.x + inPad) * dimSizes.y) +
                                     (row * (dimSizes.x + inPad)) +
                                     tileCount.x * warpSize + threadIdx.x];
          } // load
        }

        // go over the remainder tile in Z (and store only colls)
        if ((tileCount.y * warpSize + threadIdx.x) < dimSizes.z)
        {
          for (auto slab = 0; (tileCount.x * warpSize + slab) < dimSizes.x ; slab++)
          {
            outputMatrix[(tileCount.x * warpSize + slab) * (dimSizes.y * (dimSizes.z + outPad)) +
                         (row * (dimSizes.z + outPad)) +
                         tileCount.y * warpSize + threadIdx.x]
                        = shared_tile[threadIdx.y][threadIdx.x][slab];
          }
        }// store
      }// reminder X and Y
    } //isSquareSlab
  } // slab
}// end of cudaTrasnposeReal3DMatrixXZ
//--------------------------------------------------------------------------------------------------



/**
 * Transpose a real 3D matrix in the X-Z direction. It is done out-of-place
 *
 * @tparam      padding      - Which matrices are padded
 *
 * @param [out] outputMatrix - Output matrix
 * @param [in]  inputMatrix  - Input  matrix
 * @param [in]  dimSizes     - Dimension sizes of the original matrix
 */
template<SolverCUDAKernels::TransposePadding padding>
void SolverCUDAKernels::TrasposeReal3DMatrixXZ(float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes)
{
  // fixed size at the moment, may be tuned based on the domain shape in the future
  // warpSize set to 32, and 4 tiles processed at once
  if ((dimSizes.x % 32 == 0) && (dimSizes.z % 32 == 0))
  {
    cudaTrasnposeReal3DMatrixXZ<padding, true, 32, 4>
                                     <<<GetSolverTransposeGirdSize(), GetSolverTransposeBlockSize()>>>
                                     (outputMatrix, inputMatrix, dimSizes);
  }
  else
  {
    cudaTrasnposeReal3DMatrixXZ<padding, false, 32, 4>
                                   <<<GetSolverTransposeGirdSize(), GetSolverTransposeBlockSize()>>>
                                   (outputMatrix, inputMatrix, dimSizes);
  }

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of TrasposeReal3DMatrixXZ
//--------------------------------------------------------------------------------------------------

//------------------------ Explicit instances of TrasposeReal3DMatrixXZ --------------------------//
/// Transpose a real 3D matrix in the X-Z direction, input matrix padded, output matrix compact
template
void SolverCUDAKernels::TrasposeReal3DMatrixXZ<SolverCUDAKernels::TransposePadding::kInput>
                                              (float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Z direction, input matrix compact, output matrix padded
template
void SolverCUDAKernels::TrasposeReal3DMatrixXZ<SolverCUDAKernels::TransposePadding::kOutput>
                                              (float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Z direction, input and output matrix compact
template
void SolverCUDAKernels::TrasposeReal3DMatrixXZ<SolverCUDAKernels::TransposePadding::kNone>
                                              (float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Z direction, input and output matrix padded
template
void SolverCUDAKernels::TrasposeReal3DMatrixXZ<SolverCUDAKernels::TransposePadding::kInputOutput>
                                              (float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes);


/**
 * CUDA kernel to compute velocity shift in the X direction.
 *
 * @param [in, out] cufft_shift_temp - Matrix to calculate 1D FFT to
 * @param [in]      x_shift_neg_r
 */
__global__ void CUDAComputeVelocityShiftInX(cuFloatComplex*       cufft_shift_temp,
                                            const cuFloatComplex* x_shift_neg_r)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.nElementsComplex; i += getStride())
  {
    const auto  x = i % cudaDeviceConstants.nxComplex;

    cufft_shift_temp[i] = cuCmulf(cufft_shift_temp[i], x_shift_neg_r[x]) * cudaDeviceConstants.fftDividerX;
  }
}// end of CUDAComputeVelocityShiftInX
//--------------------------------------------------------------------------------------------------


/**
 * Compute the velocity shift in Fourier space over the X axis.
 * This kernel work with the original space.
 *
 * @param [in,out] cufft_shift_temp - Matrix to calculate 1D FFT to
 * @param [in]     x_shift_neg_r
 */
void SolverCUDAKernels::ComputeVelocityShiftInX(TCUFFTComplexMatrix&  cufft_shift_temp,
                                                const TComplexMatrix& x_shift_neg_r)
{
  CUDAComputeVelocityShiftInX<<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                             (reinterpret_cast<cuFloatComplex*>  (cufft_shift_temp.GetDeviceData()),
                              reinterpret_cast<const cuFloatComplex*> (x_shift_neg_r.GetDeviceData()));
  // check for errors
  cudaCheckErrors(cudaGetLastError());
 }// end of ComputeVelocityShiftInX
//--------------------------------------------------------------------------------------------------



/**
 * CUDA kernel to compute velocity shift in Y. The matrix is XY transposed.
 * @param [in, out] cufft_shift_temp - Matrix to calculate 1D FFT to
 * @param [in]      y_shift_neg_r
 */
__global__ void CUDAComputeVelocityShiftInY(cuFloatComplex*       cufft_shift_temp,
                                            const cuFloatComplex* y_shift_neg_r)
{
  const auto ny_2 = cudaDeviceConstants.ny / 2 + 1;
  const auto nElements = cudaDeviceConstants.nx * ny_2 * cudaDeviceConstants.nz;

  for (auto i = getIndex(); i < nElements; i += getStride())
  {
    // rotated dimensions
    const auto  y = i % ny_2;

    cufft_shift_temp[i] = cuCmulf(cufft_shift_temp[i], y_shift_neg_r[y]) * cudaDeviceConstants.fftDividerY;
  }
}// end of CUDAComputeVelocityShiftInY
//--------------------------------------------------------------------------------------------------

/**
 * Compute the velocity shift in Fourier space over the Y axis.
 * This kernel work with the transposed space.
 *
 * @param [in,out] cufft_shift_temp - Matrix to calculate 1D FFT to
 * @param [in]     y_shift_neg_r
 */
void SolverCUDAKernels::ComputeVelocityShiftInY(TCUFFTComplexMatrix&  cufft_shift_temp,
                                                const TComplexMatrix& y_shift_neg_r)
{
  CUDAComputeVelocityShiftInY<<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                             (reinterpret_cast<cuFloatComplex*>       (cufft_shift_temp.GetDeviceData()),
                              reinterpret_cast<const cuFloatComplex*> (y_shift_neg_r.GetDeviceData()));
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of ComputeVelocityShiftInY
//--------------------------------------------------------------------------------------------------


/**
 * CUDA kernel to compute velocity shift in Z. The matrix is XZ transposed.
 *
 * @param [in, out] cufft_shift_temp - Matrix to calculate 1D FFT to
 * @param [in]      z_shift_neg_r
 */
__global__ void CUDAComputeVelocityShiftInZ(cuFloatComplex*       cufft_shift_temp,
                                            const cuFloatComplex* z_shift_neg_r)
{
  const auto nz_2 = cudaDeviceConstants.nz / 2 + 1;
  const auto nElements = cudaDeviceConstants.nx * cudaDeviceConstants.ny * nz_2;

  for (auto i = getIndex(); i < nElements; i += getStride())
  {
    // rotated dimensions
    const auto  z = i % nz_2;

     cufft_shift_temp[i] = cuCmulf(cufft_shift_temp[i], z_shift_neg_r[z]) * cudaDeviceConstants.fftDividerZ;
  }
}// end of CUDAComputeVelocityShiftInZ
//---------------------------------------------------------------------------------------------------

/**
 * Compute the velocity shift in Fourier space over the Z axis.
 * This kernel work with the transposed space.
 *
 * @param [in,out] cufft_shift_temp - Matrix to calculate 1D FFT to
 * @param [in]     z_shift_neg_r
 */
void SolverCUDAKernels::ComputeVelocityShiftInZ(TCUFFTComplexMatrix&  cufft_shift_temp,
                                                const TComplexMatrix& z_shift_neg_r)
{
  CUDAComputeVelocityShiftInZ<<<GetSolverGridSize1D(), GetSolverBlockSize1D()>>>
                             (reinterpret_cast<cuFloatComplex*>       (cufft_shift_temp.GetDeviceData()),
                              reinterpret_cast<const cuFloatComplex*> (z_shift_neg_r.GetDeviceData()));
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of ComputeVelocityShiftInZ
//--------------------------------------------------------------------------------------------------
