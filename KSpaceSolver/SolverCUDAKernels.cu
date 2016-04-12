/**
 * @file        SolverCUDAKernels.cpp
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the all CUDA kernels
 *              for the GPU implementation
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        11 March    2013, 13:10 (created) \n
 *              12 April    2016, 15:04 (revised)
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

#include <cuComplex.h>

#include <KSpaceSolver/SolverCUDAKernels.cuh>
#include <Parameters/CUDADeviceConstants.cuh>

#include <Utils/ErrorMessages.h>
#include <Utils/CUDAUtils.cuh>

//----------------------------------------------------------------------------//
//-------------------------------- Constants ---------------------------------//
//----------------------------------------------------------------------------//


//----------------------------------------------------------------------------//
//-------------------------------- Variables ---------------------------------//
//----------------------------------------------------------------------------//

/**
 * @variable CUDADeviceConstants
 * @brief    This variable holds basic simulation constants for GPU.
 * @details  This variable holds necessary simulation constants in the CUDA GPU
 *           memory.
 *           The variable is defined in TCUDADeviceConstants.cu
 */
extern __constant__ TCUDADeviceConstants CUDADeviceConstants;


//----------------------------------------------------------------------------//
//---------------------------- Global CUDA routines --------------------------//
//----------------------------------------------------------------------------//

/**
 * Get block size for 1D kernels
 * @return 1D block size
 */
int GetSolverBlockSize1D()
{
  return TParameters::GetInstance()->CUDAParameters.GetSolverBlockSize1D();
};

/**
 * Get grid size for 1D kernels
 * @return 1D grid size
 */
int GetSolverGridSize1D()
{
  return TParameters::GetInstance()->CUDAParameters.GetSolverGridSize1D();
};

/**
 * Get block size for the transposition kernels
 * @return 3D grid size
 */
dim3 GetSolverTransposeBlockSize()
{
  return TParameters::GetInstance()->CUDAParameters.GetSolverTransposeBlockSize();
};

/**
 * Get grid size for complex 3D kernels
 * @return 3D grid size
 */
dim3 GetSolverTransposeGirdSize()
{
  return TParameters::GetInstance()->CUDAParameters.GetSolverTransposeGirdSize();
};


//----------------------------------------------------------------------------//
//----------------------------- Initializations ------------------------------//
//----------------------------------------------------------------------------//


//----------------------------------------------------------------------------//
//---------------------------------- Public ----------------------------------//
//----------------------------------------------------------------------------//

/**
 * Kernel to find out the version of the code
 * The list of GPUs can be found at https://en.wikipedia.org/wiki/CUDA
 * @param [out] cudaCodeVersion
 */
__global__ void CUDAGetCUDACodeVersion(int* cudaCodeVersion)
{
  *cudaCodeVersion = -1;

  #if (__CUDA_ARCH__ == 530)
    *cudaCodeVersion = 53;
  #elif (__CUDA_ARCH__ == 520)
    *cudaCodeVersion = 52;
  #elif (__CUDA_ARCH__ == 500)
    *cudaCodeVersion = 50;
  #elif (__CUDA_ARCH__ == 370)
    *cudaCodeVersion = 37;
  #elif (__CUDA_ARCH__ == 350)
    *cudaCodeVersion = 35;
  #elif (__CUDA_ARCH__ == 320)
    *cudaCodeVersion = 32;
  #elif (__CUDA_ARCH__ == 300)
    *cudaCodeVersion = 30;
  #elif (__CUDA_ARCH__ == 210)
    *cudaCodeVersion = 21;
  #elif (__CUDA_ARCH__ == 200)
    *cudaCodeVersion = 20;
  #endif
}// end of CUDAGetCodeVersion
//------------------------------------------------------------------------------

/**
 * Get the CUDA architecture and GPU code version the code was compiled with
 * @return  the CUDA code version the code was compiled for
 */
int SolverCUDAKernels::GetCUDACodeVersion()
{
  // host and device pointers, data copied over zero copy memory
  int * hCudaCodeVersion;
  int * dCudaCodeVersion;

  // returned value
  int cudaCodeVersion = 0;
  cudaError_t cudaError;

  // allocate for zero copy
  cudaError = cudaHostAlloc<int>(&hCudaCodeVersion, sizeof(int), cudaHostRegisterPortable | cudaHostRegisterMapped);
  // if the device is busy, return 0 - the GPU is not supported
  if (cudaError == cudaSuccess)
  {
    checkCudaErrors(cudaHostGetDevicePointer<int>(&dCudaCodeVersion, hCudaCodeVersion, 0));

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

    checkCudaErrors(cudaFreeHost(hCudaCodeVersion));
  }

  return (cudaCodeVersion);
}// end of GetCodeVersion
//------------------------------------------------------------------------------


/**
 *
 * CUDA kernel to calculate ux_sgx, uy_sgy, uz_sgz.
 * Default (heterogeneous case).
 *
 * @param [in, out] ux_sgx
 * @param [in, out] uy_sgy
 * @param [in, out] uz_sgz
 * @param [in] FFT_X
 * @param [in] FFT_Y
 * @param [in] FFT_Z
 * @param [in] dt_rho0_sgx
 * @param [in] dt_rho0_sgy
 * @param [in] dt_rho0_sgz
 * @param [in] pml_x
 * @param [in] pml_y
 * @param [in]pml_z
 */
__global__ void CUDACompute_uxyz_normalize(float*       ux_sgx,
                                           float*       uy_sgy,
                                           float*       uz_sgz,
                                           const float* FFT_X,
                                           const float* FFT_Y,
                                           const float* FFT_Z,
                                           const float* dt_rho0_sgx,
                                           const float* dt_rho0_sgy,
                                           const float* dt_rho0_sgz,
                                           const float* pml_x,
                                           const float* pml_y,
                                           const float* pml_z)
{
  for (auto i = GetIndex(); i < CUDADeviceConstants.TotalElementCount; i += GetStride())
  {
    const dim3 coords = GetReal3DCoords(i);

    const float FFT_X_el = CUDADeviceConstants.FFTDivider * FFT_X[i] * dt_rho0_sgx[i];
    const float FFT_Y_el = CUDADeviceConstants.FFTDivider * FFT_Y[i] * dt_rho0_sgy[i];
    const float FFT_Z_el = CUDADeviceConstants.FFTDivider * FFT_Z[i] * dt_rho0_sgz[i];

    const float pml_x_data = pml_x[coords.x];
    const float pml_y_data = pml_y[coords.y];
    const float pml_z_data = pml_z[coords.z];

    ux_sgx[i] = (ux_sgx[i] * pml_x_data - FFT_X_el) * pml_x_data;
    uy_sgy[i] = (uy_sgy[i] * pml_y_data - FFT_Y_el) * pml_y_data;
    uz_sgz[i] = (uz_sgz[i] * pml_z_data - FFT_Z_el) * pml_z_data;
  }
}// end of CUDACompute_uxyz_normalize
//------------------------------------------------------------------------------

/**
 *
 * Interface to the CUDA kernel computing new version of ux_sgx.
 * Default (heterogeneous case)
 *
 * @param [in, out] ux_sgx
 * @param [in, out] uy_sgy
 * @param [in, out] uz_sgz
 * @param [in] FFT_X
 * @param [in] FFT_Y
 * @param [in] FFT_Z
 * @param [in] dt_rho0_sgx
 * @param [in] dt_rho0_sgy
 * @param [in] dt_rho0_sgz
 * @param [in] pml_x
 * @param [in] pml_y
 * @param [in] pml_z
 */
void SolverCUDAKernels::Compute_uxyz_normalize(TRealMatrix&       ux_sgx,
                                               TRealMatrix&       uy_sgy,
                                               TRealMatrix&       uz_sgz,
                                               const TRealMatrix& FFT_X,
                                               const TRealMatrix& FFT_Y,
                                               const TRealMatrix& FFT_Z,
                                               const TRealMatrix& dt_rho0_sgx,
                                               const TRealMatrix& dt_rho0_sgy,
                                               const TRealMatrix& dt_rho0_sgz,
                                               const TRealMatrix& pml_x,
                                               const TRealMatrix& pml_y,
                                               const TRealMatrix& pml_z)
  {
    CUDACompute_uxyz_normalize<<<GetSolverGridSize1D(),
                                 GetSolverBlockSize1D() >>>
                              (ux_sgx.GetRawDeviceData(),
                               uy_sgy.GetRawDeviceData(),
                               uz_sgz.GetRawDeviceData(),
                               FFT_X.GetRawDeviceData(),
                               FFT_Y.GetRawDeviceData(),
                               FFT_Z.GetRawDeviceData(),
                               dt_rho0_sgx.GetRawDeviceData(),
                               dt_rho0_sgy.GetRawDeviceData(),
                               dt_rho0_sgz.GetRawDeviceData(),
                               pml_x.GetRawDeviceData(),
                               pml_y.GetRawDeviceData(),
                               pml_z.GetRawDeviceData());

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Compute_uxyz_normalize
//------------------------------------------------------------------------------



/**
 * CUDA kernel to calculate ux_sgx, uy_sgy, uz_sgz.
 * This is the case for rho0 being a scalar and a uniform grid.
 * @param [in, out] ux_sgx - new value of ux
 * @param [in, out] uy_sgy - new value of uy
 * @param [in, out] uz_sgz - new value of ux
 * @param [in] FFT_X - gradient for X
 * @param [in] FFT_Y - gradient for Y
 * @param [in] FFT_Z - gradient for Z
 * @param [in] pml_x
 * @param [in] pml_y
 * @param [in] pml_z
 */
__global__ void CUDACompute_uxyz_normalize_scalar_uniform(float*       ux_sgx,
                                                          float*       uy_sgy,
                                                          float*       uz_sgz,
                                                          const float* FFT_X,
                                                          const float* FFT_Y,
                                                          const float* FFT_Z,
                                                          const float* pml_x,
                                                          const float* pml_y,
                                                          const float* pml_z)
{
  const float Divider_X = CUDADeviceConstants.rho0_sgx_scalar * CUDADeviceConstants.FFTDivider;
  const float Divider_Y = CUDADeviceConstants.rho0_sgy_scalar * CUDADeviceConstants.FFTDivider;
  const float Divider_Z = CUDADeviceConstants.rho0_sgz_scalar * CUDADeviceConstants.FFTDivider;

  for(auto i = GetIndex(); i < CUDADeviceConstants.TotalElementCount; i += GetStride())
  {
    const dim3 coords = GetReal3DCoords(i);

    const float pml_x_el = pml_x[coords.x];
    const float pml_y_el = pml_y[coords.y];
    const float pml_z_el = pml_z[coords.z];

    ux_sgx[i] = (ux_sgx[i] * pml_x_el - Divider_X * FFT_X[i]) * pml_x_el;
    uy_sgy[i] = (uy_sgy[i] * pml_y_el - Divider_Y * FFT_Y[i]) * pml_y_el;
    uz_sgz[i] = (uz_sgz[i] * pml_z_el - Divider_Z * FFT_Z[i]) * pml_z_el;
  }// for
}// end of CUDACompute_ux_sgx_normalize_scalar_uniform
//------------------------------------------------------------------------------

/**
 * Interface to the CUDA kernel computing new version of ux_sgx, uy_sgy, uz_sgz.
 * This is the case for rho0 being a scalar and a uniform grid.
 * @param [in, out] ux_sgx
 * @param [in, out] uy_sgy
 * @param [in, out] uz_sgz
 * @param [in] FFT_X
 * @param [in] FFT_Y
 * @param [in] FFT_Z
 * @param [in] pml_x
 * @param [in] pml_y
 * @param [in] pml_z
 */
void SolverCUDAKernels::Compute_uxyz_normalize_scalar_uniform(TRealMatrix&       ux_sgx,
                                                              TRealMatrix&       uy_sgy,
                                                              TRealMatrix&       uz_sgz,
                                                              const TRealMatrix& FFT_X,
                                                              const TRealMatrix& FFT_Y,
                                                              const TRealMatrix& FFT_Z,
                                                              const TRealMatrix& pml_x,
                                                              const TRealMatrix& pml_y,
                                                              const TRealMatrix& pml_z)
{
  CUDACompute_uxyz_normalize_scalar_uniform<<<GetSolverGridSize1D(),
                                              GetSolverBlockSize1D() >>>
                                             (ux_sgx.GetRawDeviceData(),
                                              uy_sgy.GetRawDeviceData(),
                                              uz_sgz.GetRawDeviceData(),
                                              FFT_X.GetRawDeviceData(),
                                              FFT_Y.GetRawDeviceData(),
                                              FFT_Z.GetRawDeviceData(),
                                              pml_x.GetRawDeviceData(),
                                              pml_y.GetRawDeviceData(),
                                              pml_z.GetRawDeviceData());
  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Compute_ux_sgx_normalize_scalar_uniform
//------------------------------------------------------------------------------


/**
 * CUDA kernel to calculate ux_sgx, uy_sgy and uz_sgz.
 * This is the case for rho0 being a scalar and a non-uniform grid.
 * @param [in,out] ux_sgx     - updated value of ux_sgx
 * @param [in,out] uy_sgy     - updated value of ux_sgx
 * @param [in,out] uz_sgz     - updated value of ux_sgx
 * @param [in]     FFT_X      - gradient of X
 * @param [in]     FFT_Y      - gradient of X
 * @param [in]     FFT_Z      - gradient of X
 * @param [in]     dxudxn_sgx - matrix dx shift
 * @param [in]     dyudyn_sgy - matrix dy shift
 * @param [in]     dzudzn_sgz - matrix dz shift
 * @param [in]     pml_x      - matrix of pml_x
 * @param [in]     pml_y       - matrix of pml_x
 * @param [in]     pml_z       - matrix of pml_x
 */
__global__ void CUDACompute_uxyz_normalize_scalar_nonuniform(float*       ux_sgx,
                                                             float*       uy_sgy,
                                                             float*       uz_sgz,
                                                             const float* FFT_X,
                                                             const float* FFT_Y,
                                                             const float* FFT_Z,
                                                             const float* dxudxn_sgx,
                                                             const float* dyudyn_sgy,
                                                             const float* dzudzn_sgz,
                                                             const float* pml_x,
                                                             const float* pml_y,
                                                             const float* pml_z)
{
  const float Divider_X = CUDADeviceConstants.rho0_sgx_scalar * CUDADeviceConstants.FFTDivider;
  const float Divider_Y = CUDADeviceConstants.rho0_sgy_scalar * CUDADeviceConstants.FFTDivider;;
  const float Divider_Z = CUDADeviceConstants.rho0_sgz_scalar * CUDADeviceConstants.FFTDivider;

  for(auto i = GetIndex(); i < CUDADeviceConstants.TotalElementCount; i += GetStride())
  {
    const dim3 coords = GetReal3DCoords(i);

    const float pml_x_el = pml_x[coords.x];
    const float pml_y_el = pml_y[coords.y];
    const float pml_z_el = pml_z[coords.z];

    const float FFT_X_el = Divider_X * dxudxn_sgx[coords.x] * FFT_X[i];
    const float FFT_Y_el = Divider_Y * dyudyn_sgy[coords.y] * FFT_Y[i];
    const float FFT_Z_el = Divider_Z * dzudzn_sgz[coords.z] * FFT_Z[i];

    ux_sgx[i] = (ux_sgx[i] * pml_x_el - FFT_X_el) * pml_x_el;
    uy_sgy[i] = (uy_sgy[i] * pml_y_el - FFT_Y_el) * pml_y_el;
    uz_sgz[i] = (uz_sgz[i] * pml_z_el - FFT_Z_el) * pml_z_el;
  }// for
}// end of CUDACompute_uxyz_normalize_scalar_nonuniform
//------------------------------------------------------------------------------

/**
 * Interface to  calculate ux_sgx, uy_sgy and uz_sgz.
 * This is the case for rho0 being a scalar and a non-uniform grid.
 * @param [in,out] ux_sgx     - updated value of ux_sgx
 * @param [in,out] uy_sgy     - updated value of ux_sgx
 * @param [in,out] uz_sgz     - updated value of ux_sgx
 * @param [in]     FFT_X      - gradient of X
 * @param [in]     FFT_Y      - gradient of X
 * @param [in]     FFT_Z      - gradient of X
 * @param [in]     dxudxn_sgx - matrix dx shift
 * @param [in]     dyudyn_sgy - matrix dy shift
 * @param [in]     dzudzn_sgz - matrix dz shift
 * @param [in]     pml_x      - matrix of pml_x
 * @param [in]     pml_y       - matrix of pml_x
 * @param [in]     pml_z       - matrix of pml_x
 */
void SolverCUDAKernels::Compute_uxyz_normalize_scalar_nonuniform(TRealMatrix&       ux_sgx,
                                                                 TRealMatrix&       uy_sgy,
                                                                 TRealMatrix&       uz_sgz,
                                                                 const TRealMatrix& FFT_X,
                                                                 const TRealMatrix& FFT_Y,
                                                                 const TRealMatrix& FFT_Z,
                                                                 const TRealMatrix& dxudxn_sgx,
                                                                 const TRealMatrix& dyudyn_sgy,
                                                                 const TRealMatrix& dzudzn_sgz,
                                                                 const TRealMatrix& pml_x,
                                                                 const TRealMatrix& pml_y,
                                                                 const TRealMatrix& pml_z)
{
  CUDACompute_uxyz_normalize_scalar_nonuniform<<<GetSolverGridSize1D(),
                                                 GetSolverBlockSize1D()>>>
                                                (ux_sgx.GetRawDeviceData(),
                                                 uy_sgy.GetRawDeviceData(),
                                                 uz_sgz.GetRawDeviceData(),
                                                 FFT_X.GetRawDeviceData(),
                                                 FFT_Y.GetRawDeviceData(),
                                                 FFT_Z.GetRawDeviceData(),
                                                 dxudxn_sgx.GetRawDeviceData(),
                                                 dyudyn_sgy.GetRawDeviceData(),
                                                 dzudzn_sgz.GetRawDeviceData(),
                                                 pml_x.GetRawDeviceData(),
                                                 pml_y.GetRawDeviceData(),
                                                 pml_z.GetRawDeviceData());

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Compute_ux_sgx_normalize_scalar_nonuniform
//------------------------------------------------------------------------------


/**
 * CUDa kernel adding transducer data to ux_sgx
 * @param [in, out] ux_sgx             - here we add the signal
 * @param [in]      us_index           - where to add the signal (source)
 * @param [in, out] delay_mask         - delay mask to push the signal in the domain (incremented per invocation)
 * @param [in]      transducer_signal  - transducer signal
 */
__global__ void CUDAAddTransducerSource(float*        ux_sgx,
                                        const size_t* u_source_index,
                                        size_t*       delay_mask,
                                        const float*  transducer_signal)
{
  for (auto i = GetIndex(); i < CUDADeviceConstants.u_source_index_size; i += GetStride())
  {
    ux_sgx[u_source_index[i]] += transducer_signal[delay_mask[i]];
    delay_mask[i]++;
  }
}// end of CUDAAddTransducerSource
//------------------------------------------------------------------------------

/**
 * Interface to kernel adding transducer data to ux_sgx.
 * @param [in, out] ux_sgx            - ux value
 * @param [in]      u_source_index    - Index matrix - source matrix
 * @param [in, out] delay_mask        - Index matrix - delay of the signal
 * @param [in]      transducer_signal - Transducer signal
 */
void SolverCUDAKernels::AddTransducerSource(TRealMatrix&        ux_sgx,
                                            const TIndexMatrix& u_source_index,
                                            TIndexMatrix&       delay_mask,
                                            const TRealMatrix&  transducer_signal)
{
  const auto u_source_index_size = u_source_index.GetTotalElementCount();

  // Grid size is calculated based on the source size
  int CUDAGridSize1D  = (u_source_index_size  + GetSolverBlockSize1D() - 1 ) / GetSolverBlockSize1D();

  CUDAAddTransducerSource<<<CUDAGridSize1D, GetSolverBlockSize1D()>>>
                         (ux_sgx.GetRawDeviceData(),
                          u_source_index.GetRawDeviceData(),
                          delay_mask.GetRawDeviceData(),
                          transducer_signal.GetRawDeviceData());
  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of AddTransducerSource
//------------------------------------------------------------------------------


/**
 * CUDA kernel to add in velocity source terms.
 *
 * @param [in, out] uxyz_sgxyz          - velocity matrix to update
 * @param [in]      u_source_input      - Source input to add
 * @param [in]      u_source_index      - Index matrix
 * @param [in]      t_index             - Actual time step
 */
__global__ void CUDAAdd_u_source(float*        uxyz_sgxyz,
                                 const float*  u_source_input,
                                 const size_t* u_source_index,
                                 const size_t  t_index)
{
  // Set 1D or 2D step for source
  auto index2D = (CUDADeviceConstants.u_source_many == 0) ? t_index : t_index * CUDADeviceConstants.u_source_index_size;

  if (CUDADeviceConstants.u_source_mode == 0)
  {
    for (auto i = GetIndex(); i < CUDADeviceConstants.u_source_index_size; i += GetStride())
    {
      uxyz_sgxyz[u_source_index[i]]  = (CUDADeviceConstants.u_source_many == 0) ?  u_source_input[index2D] :
                                                                                   u_source_input[index2D + i];
    }// for
  }// end of Dirichlet

  if (CUDADeviceConstants.u_source_mode == 1)
  {
    for (auto i  = GetIndex(); i < CUDADeviceConstants.u_source_index_size; i += GetStride())
    {
      uxyz_sgxyz[u_source_index[i]] += (CUDADeviceConstants.u_source_many == 0) ?  u_source_input[index2D] :
                                                                                   u_source_input[index2D + i];
    }
  }
}// end of CUDAAdd_u_source
//------------------------------------------------------------------------------


/**
 * Interface to CUDA kernel adding in velocity source terms.
 *
 * @param [in, out] uxyz_sgxyz - velocity matrix to update
 * @param [in] u_source_input  - Source input to add
 * @param [in] u_source_index  - Index matrix
 * @param [in] t_index         - Actual time step
 */
void SolverCUDAKernels::Add_u_source(TRealMatrix&        uxyz_sgxyz,
                                     const TRealMatrix&  u_source_input,
                                     const TIndexMatrix& u_source_index,
                                     const size_t        t_index)
{
  const auto u_source_index_size = u_source_index.GetTotalElementCount();

  // Grid size is calculated based on the source size
  const int CUDAGridSize1D  = (u_source_index_size  + GetSolverBlockSize1D() - 1 ) / GetSolverBlockSize1D();
  //@todo here should be a test not to generate too much blocks, and balance workload

  CUDAAdd_u_source<<< CUDAGridSize1D, GetSolverBlockSize1D()>>>
                  (uxyz_sgxyz.GetRawDeviceData(),
                   u_source_input.GetRawDeviceData(),
                   u_source_index.GetRawDeviceData(),
                   t_index);

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Add_u_source
//------------------------------------------------------------------------------

/**
 * CUDA kernel to add p_source to acoustic density
 * @param [out] rhox - acoustic density
 * @param [out] rhoy - acoustic density
 * @param [out] rhoz - acoustic density
 * @param [in]  p_source_input - source input to add
 * @param [in]  p_source_index - index matrix with source
 * @param [in]  t_index        - actual timestep

 */
__global__ void CUDAAdd_p_source(float*        rhox,
                                 float*        rhoy,
                                 float*        rhoz,
                                 const float*  p_source_input,
                                 const size_t* p_source_index,
                                 const size_t  t_index)
{
  // Set 1D or 2D step for source
  auto index2D = (CUDADeviceConstants.p_source_many == 0) ? t_index : t_index * CUDADeviceConstants.p_source_index_size;

  if (CUDADeviceConstants.p_source_mode == 0)
  {
    if (CUDADeviceConstants.p_source_many == 0)
    { // single signal
      for (auto i = GetIndex(); i < CUDADeviceConstants.p_source_index_size; i += GetStride())
      {
        rhox[p_source_index[i]] = p_source_input[index2D];
        rhoy[p_source_index[i]] = p_source_input[index2D];
        rhoz[p_source_index[i]] = p_source_input[index2D];
      }
    }
    else
    { // multiple signals
      for (auto i = GetIndex(); i < CUDADeviceConstants.p_source_index_size; i += GetStride())
      {
        rhox[p_source_index[i]] = p_source_input[index2D + i];
        rhoy[p_source_index[i]] = p_source_input[index2D + i];
        rhoz[p_source_index[i]] = p_source_input[index2D + i];
      }
    }
  }// end mode == 0 (Cauchy)

  if (CUDADeviceConstants.p_source_mode == 1)
  {
    if (CUDADeviceConstants.p_source_many == 0)
    { // single signal
      for (auto i = GetIndex(); i < CUDADeviceConstants.p_source_index_size; i += GetStride())
      {
        rhox[p_source_index[i]] += p_source_input[index2D];
        rhoy[p_source_index[i]] += p_source_input[index2D];
        rhoz[p_source_index[i]] += p_source_input[index2D];
      }
    }
    else
    { // multiple signals
      for (auto i = GetIndex(); i < CUDADeviceConstants.p_source_index_size; i += GetStride())
      {
        rhox[p_source_index[i]] += p_source_input[index2D + i];
        rhoy[p_source_index[i]] += p_source_input[index2D + i];
        rhoz[p_source_index[i]] += p_source_input[index2D + i];
      }
    }
  }// end mode == 0 (Dirichlet)
}// end of CUDAAdd_p_source
//------------------------------------------------------------------------------

/**
 * Interface to kernel which adds in pressure source (to acoustic density).
 * @param [out] rhox - acoustic density
 * @param [out] rhoy - acoustic density
 * @param [out] rhoz - acoustic density
 * @param [in]  p_source_input - source input to add
 * @param [in]  p_source_index - index matrix with source
 * @param [in]  t_index        - actual timestep
 */
void SolverCUDAKernels::Add_p_source(TRealMatrix&        rhox,
                                     TRealMatrix&        rhoy,
                                     TRealMatrix&        rhoz,
                                     const TRealMatrix&  p_source_input,
                                     const TIndexMatrix& p_source_index,
                                     const size_t        t_index)
{
  const auto p_source_index_size = p_source_index.GetTotalElementCount();

  // Grid size is calculated based on the source size
  int CUDAGridSize1D  = (p_source_index_size  + GetSolverBlockSize1D() - 1 ) / GetSolverBlockSize1D();
  //@todo here should be a test not to generate too much blocks, and balance workload

  CUDAAdd_p_source<<<CUDAGridSize1D,GetSolverBlockSize1D()>>>
                  (rhox.GetRawDeviceData(),
                   rhoy.GetRawDeviceData(),
                   rhoz.GetRawDeviceData(),
                   p_source_input.GetRawDeviceData(),
                   p_source_index.GetRawDeviceData(),
                   t_index);

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Add_p_source
//------------------------------------------------------------------------------

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
__global__  void CUDACompute_dt_rho_sg_mul_u(float*       ux_sgx,
                                             float*       uy_sgy,
                                             float*       uz_sgz,
                                             const float* dt_rho0_sgx = nullptr,
                                             const float* dt_rho0_sgy = nullptr,
                                             const float* dt_rho0_sgz = nullptr)

{
  if (Is_rho0_scalar)
  {
    const float ScaledDivider_X = CUDADeviceConstants.FFTDivider * 0.5f * CUDADeviceConstants.rho0_sgx_scalar;
    const float ScaledDivider_Y = CUDADeviceConstants.FFTDivider * 0.5f * CUDADeviceConstants.rho0_sgy_scalar;
    const float ScaledDivider_Z = CUDADeviceConstants.FFTDivider * 0.5f * CUDADeviceConstants.rho0_sgz_scalar;

    for (auto i = GetIndex(); i < CUDADeviceConstants.TotalElementCount; i += GetStride())
    {
      ux_sgx[i] *= ScaledDivider_X;
      uy_sgy[i] *= ScaledDivider_Y;
      uz_sgz[i] *= ScaledDivider_Z;
    }
  }
  else
  { // heterogeneous
    const float ScaledDivider = CUDADeviceConstants.FFTDivider * 0.5f;

    for (auto i = GetIndex(); i < CUDADeviceConstants.TotalElementCount; i += GetStride())
    {
      ux_sgx[i] *= dt_rho0_sgx[i] * ScaledDivider;
      uy_sgy[i] *= dt_rho0_sgy[i] * ScaledDivider;
      uz_sgz[i] *= dt_rho0_sgz[i] * ScaledDivider;
    }
  }
}// end of CudaCompute_dt_rho_sg_mul_ifft_div_2
//------------------------------------------------------------------------------

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
void SolverCUDAKernels::Compute_dt_rho_sg_mul_ifft_div_2(TRealMatrix&       ux_sgx,
                                                         TRealMatrix&       uy_sgy,
                                                         TRealMatrix&       uz_sgz,
                                                         const TRealMatrix& dt_rho0_sgx,
                                                         const TRealMatrix& dt_rho0_sgy,
                                                         const TRealMatrix& dt_rho0_sgz)
{
  CUDACompute_dt_rho_sg_mul_u<false>
                             <<<GetSolverGridSize1D(),
                                GetSolverBlockSize1D()>>>
                             (ux_sgx.GetRawDeviceData(),
                              uy_sgy.GetRawDeviceData(),
                              uz_sgz.GetRawDeviceData(),
                              dt_rho0_sgx.GetRawDeviceData(),
                              dt_rho0_sgy.GetRawDeviceData(),
                              dt_rho0_sgz.GetRawDeviceData());

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Compute_dt_rho_sg_mul_ifft_div_2
//------------------------------------------------------------------------------

/**
 * Interface to CUDA Compute u = dt ./ rho0_sgx .* ifft(FFT).
 * if rho0_sgx is scalar, uniform case.
 *
 * @param [in, out] ux_sgx   - data stored in u matrix
 * @param [in, out] uy_sgy   - data stored in u matrix
 * @param [in, out] uz_sgz   - data stored in u matrix
 */
void SolverCUDAKernels::Compute_dt_rho_sg_mul_ifft_div_2(TRealMatrix& ux_sgx,
                                                         TRealMatrix& uy_sgy,
                                                         TRealMatrix& uz_sgz)
{
  CUDACompute_dt_rho_sg_mul_u<true>
                             <<<GetSolverGridSize1D(),
                                GetSolverBlockSize1D()>>>
                             (ux_sgx.GetRawDeviceData(),
                              uy_sgy.GetRawDeviceData(),
                              uz_sgz.GetRawDeviceData());

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Compute_dt_rho_sg_mul_ifft_div_2
//------------------------------------------------------------------------------



/**
 * CUDA kernel to Compute u = dt./rho0_sgy .* ifft (FFT).
 * if rho0_sg is scalar, nonuniform  non uniform grid, y component.
 * @param [in, out] ux_sgx
 * @param [in, out] ux_sgx
 * @param [in, out] ux_sgx
 * @param [in] dxudxn_sgx
 * @param [in] dyudyn_sgy
 * @param [in] dzudzn_sgz
 */
__global__ void CUDACompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform(float*       ux_sgx,
                                                                       float*       uy_sgy,
                                                                       float*       uz_sgz,
                                                                       const float* dxudxn_sgx,
                                                                       const float* dyudyn_sgy,
                                                                       const float* dzudzn_sgz)
{
  const float ScaledDivider_X = CUDADeviceConstants.FFTDivider * 0.5f * CUDADeviceConstants.rho0_sgx_scalar;
  const float ScaledDivider_Y = CUDADeviceConstants.FFTDivider * 0.5f * CUDADeviceConstants.rho0_sgy_scalar;
  const float ScaledDivider_Z = CUDADeviceConstants.FFTDivider * 0.5f * CUDADeviceConstants.rho0_sgz_scalar;

  for(auto i = GetIndex(); i < CUDADeviceConstants.TotalElementCount; i += GetStride())
  {
    const dim3 coords = GetReal3DCoords(i);

    ux_sgx[i] *= ScaledDivider_X * dxudxn_sgx[coords.x];
    uy_sgy[i] *= ScaledDivider_Y * dyudyn_sgy[coords.y];
    uz_sgz[i] *= ScaledDivider_Z * dzudzn_sgz[coords.z];
  }
}// end of CUDACompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform
//------------------------------------------------------------------------------


/**
 * Interface to CUDA kernel to Compute u = dt./rho0_sgy .* ifft (FFT).
 * if rho0_sgx is scalar, nonuniform  non uniform Compute_ddx_kappa_fft_pgrid, y component.
 * @param [in, out] ux_sgx
 * @param [in, out] uy_sgy
 * @param [in, out] uz_sgz
 * @param [in] dxudxn_sgx
 * @param [in] dyudyn_sgy
 * @param [in] dzudzn_sgz
 */
  void SolverCUDAKernels::Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform(TRealMatrix&       ux_sgx,
                                                                             TRealMatrix&       uy_sgy,
                                                                             TRealMatrix&       uz_sgz,
                                                                             const TRealMatrix& dxudxn_sgx,
                                                                             const TRealMatrix& dyudyn_sgy,
                                                                             const TRealMatrix& dzudzn_sgz)
{
  CUDACompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform<<<GetSolverGridSize1D(),
                                                           GetSolverBlockSize1D()>>>
                                                          (ux_sgx.GetRawDeviceData(),
                                                           uy_sgy.GetRawDeviceData(),
                                                           uz_sgz.GetRawDeviceData(),
                                                           dxudxn_sgx.GetRawDeviceData(),
                                                           dxudxn_sgx.GetRawDeviceData(),
                                                           dxudxn_sgx.GetRawDeviceData());
// check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform
//------------------------------------------------------------------------------


/**
 *  kernel which compute part of the new velocity term - gradient
 *  of p represented by:
 *  bsxfun(\@times, ddx_k_shift_pos, kappa .* p_k).
 *
 *
 * @param [in, out]    FFT_X - matrix to store input for iFFT (p) /dx
 * @param [out]    FFT_Y - matrix to store input for iFFT (p) /dy
 * @param [out]    FFT_Z - matrix to store input for iFFT (p) /dz
 *
 * @param [in]     kappa - Real matrix of kappa
 *
 * @param [in]     ddx - precomputed value of ddx_k_shift_pos
 * @param [in]     ddy - precomputed value of ddy_k_shift_pos
 * @param [in]     ddz - precomputed value of ddz_k_shift_pos
 */
__global__ void CUDACompute_ddx_kappa_fft_p(cuFloatComplex*       FFT_X,
                                            cuFloatComplex*       FFT_Y,
                                            cuFloatComplex*       FFT_Z,
                                            const float*          kappa,
                                            const cuFloatComplex* ddx,
                                            const cuFloatComplex* ddy,
                                            const cuFloatComplex* ddz)
{
  for(auto i = GetIndex(); i < CUDADeviceConstants.ComplexTotalElementCount; i += GetStride())
  {
    const dim3 coords = GetComplex3DCoords(i);

    const cuFloatComplex p_k_el = FFT_X[i] * kappa[i];

    FFT_X[i] = cuCmulf(p_k_el, ddx[coords.x]);
    FFT_Y[i] = cuCmulf(p_k_el, ddy[coords.y]);
    FFT_Z[i] = cuCmulf(p_k_el, ddz[coords.z]);
  }
}// end of Compute_ddx_kappa_fft_x
//------------------------------------------------------------------------------

/**
 *  Interface to kernet which compute part of the new velocity term - gradient
 *  of p represented by:
 *  bsxfun(\@times, ddx_k_shift_pos, kappa .* p_k).
 *
 * @param [in,out] X_Matrix - 3D pressure matrix
 * @param [out]    FFT_X - matrix to store input for iFFT (p) /dx
 * @param [out]    FFT_Y - matrix to store input for iFFT (p) /dy
 * @param [out]    FFT_Z - matrix to store input for iFFT (p) /dz
 *
 * @param [in]     kappa - Real matrix of kappa
 *
 * @param [in]     ddx - precomputed value of ddx_k_shift_pos
 * @param [in]     ddy - precomputed value of ddy_k_shift_pos
 * @param [in]     ddz - precomputed value of ddz_k_shift_pos
 */
void SolverCUDAKernels::Compute_ddx_kappa_fft_p(TRealMatrix&         X_Matrix,
                                                TCUFFTComplexMatrix& FFT_X,
                                                TCUFFTComplexMatrix& FFT_Y,
                                                TCUFFTComplexMatrix& FFT_Z,
                                                const TRealMatrix&    kappa,
                                                const TComplexMatrix& ddx,
                                                const TComplexMatrix& ddy,
                                                const TComplexMatrix& ddz)
{
  // Compute FFT of X
  FFT_X.Compute_FFT_3D_R2C(X_Matrix);

  CUDACompute_ddx_kappa_fft_p<<<GetSolverGridSize1D(),
                                GetSolverBlockSize1D()>>>
                             (reinterpret_cast<cuFloatComplex*>(FFT_X.GetRawDeviceData()),
                              reinterpret_cast<cuFloatComplex*>(FFT_Y.GetRawDeviceData()),
                              reinterpret_cast<cuFloatComplex*>(FFT_Z.GetRawDeviceData()),
                              kappa.GetRawDeviceData(),
                              reinterpret_cast<const cuFloatComplex*>(ddx.GetRawDeviceData()),
                              reinterpret_cast<const cuFloatComplex*>(ddy.GetRawDeviceData()),
                              reinterpret_cast<const cuFloatComplex*>(ddz.GetRawDeviceData()));

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Compute_ddx_kappa_fft_p
//------------------------------------------------------------------------------


/**
 * Kernel calculating the inner part of du, dy, dz on
 * uniform grid.
 * Complex numbers are passed as float2 structures.
 *
 * @param [in, out] FFT_X - FFT of ux
 * @param [in, out] FFT_Y - FFT of uy
 * @param [in, out] FFT_Z - FFT of uz
 * @param [in]      kappa
 * @param [in]      ddx_neg - ddx_k_shift_neg
 * @param [in]      ddy_neg - ddy_k_shift_neg
 * @param [in]      ddz_neg - ddz_k_shift_neg
 */
__global__  void CUDACompute_duxyz_uniform(cuFloatComplex*       FFT_X,
                                           cuFloatComplex*       FFT_Y,
                                           cuFloatComplex*       FFT_Z,
                                           const float*          kappa,
                                           const cuFloatComplex* ddx_neg,
                                           const cuFloatComplex* ddy_neg,
                                           const cuFloatComplex* ddz_neg)
{
  for(auto i = GetIndex(); i < CUDADeviceConstants.ComplexTotalElementCount; i += GetStride())
  {
    const dim3 coords = GetComplex3DCoords(i);

    const cuFloatComplex ddx_neg_el = ddx_neg[coords.x];
    const cuFloatComplex ddz_neg_el = ddz_neg[coords.z];
    const cuFloatComplex ddy_neg_el = ddy_neg[coords.y];

    const float kappa_el = kappa[i] * CUDADeviceConstants.FFTDivider;

    const cuFloatComplex FFT_X_el = FFT_X[i] * kappa_el;
    const cuFloatComplex FFT_Y_el = FFT_Y[i] * kappa_el;
    const cuFloatComplex FFT_Z_el = FFT_Z[i] * kappa_el;

    FFT_X[i] = cuCmulf(FFT_X_el, ddx_neg_el);
    FFT_Y[i] = cuCmulf(FFT_Y_el, ddy_neg_el);
    FFT_Z[i] = cuCmulf(FFT_Z_el, ddz_neg_el);
  } // for
}// end of CUDACompute_duxyz_uniform
//------------------------------------------------------------------------------

/**
 * Interface to kernel calculating the inner part of du, dy, dz on
 * uniform grid.
 * @param [in, out] FFT_X - FFT of ux
 * @param [in, out] FFT_Y - FFT of uy
 * @param [in, out] FFT_Z - FFT of uz
 * @param [in] kappa
 * @param [in] ddx_k_shift_neg
 * @param [in] ddy_k_shift_neg
 * @param [in] ddz_k_shift_neg
 */
void SolverCUDAKernels::Compute_duxyz_uniform(TCUFFTComplexMatrix&  FFT_X,
                                              TCUFFTComplexMatrix&  FFT_Y,
                                              TCUFFTComplexMatrix&  FFT_Z,
                                              const TRealMatrix&    kappa,
                                              const TComplexMatrix& ddx_k_shift_neg,
                                              const TComplexMatrix& ddy_k_shift_neg,
                                              const TComplexMatrix& ddz_k_shift_neg)
{
  CUDACompute_duxyz_uniform<<<GetSolverGridSize1D(),
                              GetSolverBlockSize1D()>>>
                          (reinterpret_cast<cuFloatComplex *>(FFT_X.GetRawDeviceData()),
                           reinterpret_cast<cuFloatComplex *>(FFT_Y.GetRawDeviceData()),
                           reinterpret_cast<cuFloatComplex *>(FFT_Z.GetRawDeviceData()),
                           kappa.GetRawDeviceData(),
                           reinterpret_cast<const cuFloatComplex *>(ddx_k_shift_neg.GetRawDeviceData()),
                           reinterpret_cast<const cuFloatComplex *>(ddy_k_shift_neg.GetRawDeviceData()),
                           reinterpret_cast<const cuFloatComplex *>(ddz_k_shift_neg.GetRawDeviceData()));

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Compute_duxyz_uniform
//------------------------------------------------------------------------------


/**
 * CUDA kernel to shift du, dy and dz on non-uniform grid.
 * @param [in,out] duxdx
 * @param [in,out] duydy
 * @param [in,out] duzdz
 * @param [in]     dxudxn
 * @param [in]     dyudyn
 * @param [in]     dzudzn
 */
__global__  void CUDACompute_duxyz_non_linear(float*       duxdx,
                                              float*       duydy,
                                              float*       duzdz,
                                              const float* duxdxn,
                                              const float* duydyn,
                                              const float* duzdzn)
{
  for(auto i = GetIndex(); i < CUDADeviceConstants.TotalElementCount; i += GetStride())
  {
    const dim3 coords = GetReal3DCoords(i);

    duxdx[i] *= duxdxn[coords.x];
    duydy[i] *= duydyn[coords.y];
    duzdz[i] *= duzdzn[coords.z];
  }
}// end of CUDACompute_duxyz_non_linear
//------------------------------------------------------------------------------

/**
 * Interface to CUDA kernel which shift new values of dux, duy and duz on non-uniform grid.
 * @param [in,out] duxdx
 * @param [in,out] duydy
 * @param [in,out] duzdz
 * @param [in]     dxudxn
 * @param [in]     dyudyn
 * @param [in]     dzudzn
 */
void SolverCUDAKernels::Compute_duxyz_non_uniform(TRealMatrix&       duxdx,
                                                  TRealMatrix&       duydy,
                                                  TRealMatrix&       duzdz,
                                                  const TRealMatrix& dxudxn,
                                                  const TRealMatrix& dyudyn,
                                                  const TRealMatrix& dzudzn)
{
  CUDACompute_duxyz_non_linear<<<GetSolverGridSize1D(),
                                 GetSolverBlockSize1D()>>>
                              (duxdx.GetRawDeviceData(),
                               duydy.GetRawDeviceData(),
                               duzdz.GetRawDeviceData(),
                               dxudxn.GetRawDeviceData(),
                               dyudyn.GetRawDeviceData(),
                               dzudzn.GetRawDeviceData());

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Compute_duxyz_non_uniform
//------------------------------------------------------------------------------


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
__global__ void CUDACalculate_p0_source_add_initial_pressure(float*       p,
                                                             float*       rhox,
                                                             float*       rhoy,
                                                             float*       rhoz,
                                                             const float* p0,
                                                             const float* c2 = nullptr)
{
  for (auto i = GetIndex(); i < CUDADeviceConstants.TotalElementCount; i += GetStride())
  {
    float tmp = p[i] = p0[i];

    tmp = (Is_c0_scalar) ? tmp / (3.0f * CUDADeviceConstants.c2): tmp / (3.0f * c2[i]);
    rhox[i] = tmp;
    rhoy[i] = tmp;
    rhoz[i] = tmp;
  }
}// end of CUDACalculate_p0_source_add_initial_pressure
//------------------------------------------------------------------------------


/**
 * Interface for kernel to add initial pressure p0 into p, rhox, rhoy, rhoz.
 * @param [out] p       - pressure
 * @param [out] rhox
 * @param [out] rhoy
 * @param [out] rhoz
 * @param [in]  p0       - intial pressure
 * @param [in]  Is_c2_scalar - scalar or vector?
 * @param [in]  c2       - sound speed
 */
void SolverCUDAKernels::Calculate_p0_source_add_initial_pressure(TRealMatrix&       p,
                                                                 TRealMatrix&       rhox,
                                                                 TRealMatrix&       rhoy,
                                                                 TRealMatrix&       rhoz,
                                                                 const TRealMatrix& p0,
                                                                 const bool         Is_c2_scalar,
                                                                 const float*       c2)
{
  if (Is_c2_scalar)
  {
    CUDACalculate_p0_source_add_initial_pressure<true>
                                                <<<GetSolverGridSize1D(),
                                                  GetSolverBlockSize1D()>>>
                                                (p.GetRawDeviceData(),
                                                 rhox.GetRawDeviceData(),
                                                 rhoy.GetRawDeviceData(),
                                                 rhoz.GetRawDeviceData(),
                                                 p0.GetRawDeviceData());
  }
  else
  {
      CUDACalculate_p0_source_add_initial_pressure<false>
                                                <<<GetSolverGridSize1D(),
                                                  GetSolverBlockSize1D()>>>
                                                (p.GetRawDeviceData(),
                                                 rhox.GetRawDeviceData(),
                                                 rhoy.GetRawDeviceData(),
                                                 rhoz.GetRawDeviceData(),
                                                 p0.GetRawDeviceData(),
                                                 c2);
  }

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Calculate_p0_source_add_initial_pressure
//------------------------------------------------------------------------------

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
__global__ void CUDACompute_rhoxyz_nonlinear_homogeneous(float*       rhox,
                                                         float*       rhoy,
                                                         float*       rhoz,
                                                         const float* pml_x,
                                                         const float* pml_y,
                                                         const float* pml_z,
                                                         const float* duxdx,
                                                         const float* duydy,
                                                         const float* duzdz)
{
  for (auto i = GetIndex(); i < CUDADeviceConstants.TotalElementCount; i += GetStride())
  {
    const dim3 coords = GetReal3DCoords(i);

    const float pml_x_el = pml_x[coords.x];
    const float pml_y_el = pml_y[coords.y];
    const float pml_z_el = pml_z[coords.z];

    const float dux = duxdx[i];
    const float duy = duydy[i];
    const float duz = duzdz[i];

    rhox[i] = pml_x_el * ((pml_x_el * rhox[i] - CUDADeviceConstants.dt_rho0_scalar * dux) /
                          (1.0f + CUDADeviceConstants.dt2 * dux));
    rhoy[i] = pml_y_el * ((pml_y_el * rhoy[i] - CUDADeviceConstants.dt_rho0_scalar * duy) /
                          (1.0f + CUDADeviceConstants.dt2 * duy));
    rhoz[i] = pml_z_el * ((pml_z_el * rhoz[i] - CUDADeviceConstants.dt_rho0_scalar * duz) /
                          (1.0f + CUDADeviceConstants.dt2 * duz));
  }
}// end of CUDACompute_rhoxyz_nonlinear_homogeneous
//------------------------------------------------------------------------------

/**
 * Interface to kernel which calculate new values of rho (acoustic density).
 * Non-linear, homogenous case.
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
void SolverCUDAKernels::Compute_rhoxyz_nonlinear_homogeneous(TRealMatrix&       rhox,
                                                             TRealMatrix&       rhoy,
                                                             TRealMatrix&       rhoz,
                                                             const TRealMatrix& pml_x,
                                                             const TRealMatrix& pml_y,
                                                             const TRealMatrix& pml_z,
                                                             const TRealMatrix& duxdx,
                                                             const TRealMatrix& duydy,
                                                             const TRealMatrix& duzdz)
{
  CUDACompute_rhoxyz_nonlinear_homogeneous<<<GetSolverGridSize1D(),
                                             GetSolverBlockSize1D()>>>
                                          (rhox.GetRawDeviceData(),
                                           rhoy.GetRawDeviceData(),
                                           rhoz.GetRawDeviceData(),
                                           pml_x.GetRawDeviceData(),
                                           pml_y.GetRawDeviceData(),
                                           pml_z.GetRawDeviceData(),
                                           duxdx.GetRawDeviceData(),
                                           duydy.GetRawDeviceData(),
                                           duzdz.GetRawDeviceData());
  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Compute_rhoxyz_nonlinear_homogeneous
//------------------------------------------------------------------------------

/*
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
__global__ void CUDACompute_rhoxyz_nonlinear_heterogeneous(float*       rhox,
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
  for (auto i = GetIndex(); i < CUDADeviceConstants.TotalElementCount; i += GetStride())
  {
    const dim3 coords = GetReal3DCoords(i);

    const float pml_x_el = pml_x[coords.x];
    const float pml_y_el = pml_y[coords.y];
    const float pml_z_el = pml_z[coords.z];

    const float dt_rho0 = CUDADeviceConstants.dt * rho0[i];

    const float dux = duxdx[i];
    const float duy = duydy[i];
    const float duz = duzdz[i];

    rhox[i] = pml_x_el * ((pml_x_el * rhox[i] - dt_rho0 * dux) / (1.0f + CUDADeviceConstants.dt2 * dux));
    rhoy[i] = pml_y_el * ((pml_y_el * rhoy[i] - dt_rho0 * duy) / (1.0f + CUDADeviceConstants.dt2 * duy));
    rhoz[i] = pml_z_el * ((pml_z_el * rhoz[i] - dt_rho0 * duz) / (1.0f + CUDADeviceConstants.dt2 * duz));
  }
}//end of CUDACompute_rhoxyz_nonlinear_heterogeneous
//------------------------------------------------------------------------------

/*
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
void SolverCUDAKernels::Compute_rhoxyz_nonlinear_heterogeneous(TRealMatrix&       rhox,
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
  CUDACompute_rhoxyz_nonlinear_heterogeneous<<<GetSolverGridSize1D(),
                                               GetSolverBlockSize1D() >>>
                                              (rhox.GetRawDeviceData(),
                                               rhoy.GetRawDeviceData(),
                                               rhoz.GetRawDeviceData(),
                                               pml_x.GetRawDeviceData(),
                                               pml_y.GetRawDeviceData(),
                                               pml_z.GetRawDeviceData(),
                                               duxdx.GetRawDeviceData(),
                                               duydy.GetRawDeviceData(),
                                               duzdz.GetRawDeviceData(),
                                               rho0.GetRawDeviceData());

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Compute_rhoxyz_nonlinear_heterogeneous
//------------------------------------------------------------------------------

/**
 * Interface to kernel which calculate new values of rho (acoustic density).
 * Linear, homogenous case.
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
__global__ void CUDACompute_rhoxyz_linear_homogeneous(float*       rhox,
                                                      float*       rhoy,
                                                      float*       rhoz,
                                                      const float* pml_x,
                                                      const float* pml_y,
                                                      const float* pml_z,
                                                      const float* duxdx,
                                                      const float* duydy,
                                                      const float* duzdz)
{
  for (auto i = GetIndex(); i < CUDADeviceConstants.TotalElementCount; i += GetStride())
  {
    const dim3 coords = GetReal3DCoords(i);

    const float pml_x_el = pml_x[coords.x];
    const float pml_y_el = pml_y[coords.y];
    const float pml_z_el = pml_z[coords.z];

    rhox[i] = pml_x_el * (pml_x_el * rhox[i] - CUDADeviceConstants.dt_rho0_scalar * duxdx[i]);
    rhoy[i] = pml_y_el * (pml_y_el * rhoy[i] - CUDADeviceConstants.dt_rho0_scalar * duydy[i]);
    rhoz[i] = pml_z_el * (pml_z_el * rhoz[i] - CUDADeviceConstants.dt_rho0_scalar * duzdz[i]);
  }
}// end of CUDACompute_rhoxyz_linear_homogeneous
//------------------------------------------------------------------------------

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
void SolverCUDAKernels::Compute_rhoxyz_linear_homogeneous(TRealMatrix&       rhox,
                                                          TRealMatrix&       rhoy,
                                                          TRealMatrix&       rhoz,
                                                          const TRealMatrix& pml_x,
                                                          const TRealMatrix& pml_y,
                                                          const TRealMatrix& pml_z,
                                                          const TRealMatrix& duxdx,
                                                          const TRealMatrix& duydy,
                                                          const TRealMatrix& duzdz)
{
  CUDACompute_rhoxyz_linear_homogeneous<<<GetSolverGridSize1D(),
                                          GetSolverBlockSize1D() >>>
                                      (rhox.GetRawDeviceData(),
                                       rhoy.GetRawDeviceData(),
                                       rhoz.GetRawDeviceData(),
                                       pml_x.GetRawDeviceData(),
                                       pml_y.GetRawDeviceData(),
                                       pml_z.GetRawDeviceData(),
                                       duxdx.GetRawDeviceData(),
                                       duydy.GetRawDeviceData(),
                                       duzdz.GetRawDeviceData());
  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Compute_rhoxyz_linear_homogeneous
//------------------------------------------------------------------------------

/*
 * CUDA kernel which calculate new values of rho (acoustic density).
 * Linear, heterogenous case.
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
__global__ void CUDACompute_rhoxyz_linear_heterogeneous(float*       rhox,
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
  for (auto i = GetIndex(); i < CUDADeviceConstants.TotalElementCount; i += GetStride())
  {
    const dim3 coords = GetReal3DCoords(i);

    const float pml_x_el = pml_x[coords.x];
    const float pml_y_el = pml_y[coords.y];
    const float pml_z_el = pml_z[coords.z];

    const float dt_rho0  = CUDADeviceConstants.dt * rho0[i];

    rhox[i] = pml_x_el * (pml_x_el * rhox[i] - dt_rho0 * duxdx[i]);
    rhoy[i] = pml_y_el * (pml_y_el * rhoy[i] - dt_rho0 * duydy[i]);
    rhoz[i] = pml_z_el * (pml_z_el * rhoz[i] - dt_rho0 * duzdz[i]);
  }
}// end of CUDACompute_rhoxyz_linear_heterogeneous
//------------------------------------------------------------------------------

/*
 * Interface to kernel which calculate new values of rho (acoustic density).
 * Linear, heterogenous case.
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
void SolverCUDAKernels::Compute_rhoxyz_linear_heterogeneous(TRealMatrix&       rhox,
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
  CUDACompute_rhoxyz_linear_heterogeneous<<<GetSolverGridSize1D(),
                                            GetSolverBlockSize1D()>>>
                                         (rhox.GetRawDeviceData(),
                                          rhoy.GetRawDeviceData(),
                                          rhoz.GetRawDeviceData(),
                                          pml_x.GetRawDeviceData(),
                                          pml_y.GetRawDeviceData(),
                                          pml_z.GetRawDeviceData(),
                                          duxdx.GetRawDeviceData(),
                                          duydy.GetRawDeviceData(),
                                          duzdz.GetRawDeviceData(),
                                          rho0.GetRawDeviceData());

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Compute_rhoxyz_linear_heterogeneous
//------------------------------------------------------------------------------


/**
 *
 * CUDA kernel which calculates three temporary sums in the new pressure formula \n
 * non-linear absorbing case. Homogeneous and heterogenous variants are treated
 * using templates. Homogeneous variables are in constant memory.
 *
 * @param [out] rho_sum      - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [out] BonA_sum     - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [out] du_sum       - rho0* (duxdx + duydy + duzdz)
 * @param [in]  rhox,        - acoustic density X
 * @param [in]  rhoy,        - acoustic density Y
 * @param [in]  rhoz,        - acoustic density Z
 * @param [in]  duxdx        - gradient of velocity in X
 * @param [in]  duydy        - gradient of velocity in X
 * @param [in]  duzdz        - gradient of velocity in X
 * @param [in]  BonA_matrix  - heterogeneous value for BonA
 * @param [in]  rho0_matrix  - heterogeneous value for rho0
 *
 *
 */
template <bool Is_BonA_scalar, bool Is_rho0_scalar>
__global__ void CUDACalculate_SumRho_BonA_SumDu(float*       rho_sum,
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
  for (auto i = GetIndex(); i < CUDADeviceConstants.TotalElementCount; i += GetStride())
  {
    const float BonA = (Is_BonA_scalar) ? CUDADeviceConstants.BonA_scalar : BonA_matrix[i];
    const float rho0 = (Is_rho0_scalar) ? CUDADeviceConstants.rho0_scalar : rho0_matrix[i];

    const float rho_xyz_el = rhox[i] + rhoy[i] + rhoz[i];

    rho_sum[i]  = rho_xyz_el;
    BonA_sum[i] = ((BonA * rho_xyz_el * rho_xyz_el) / (2.0f * rho0)) + rho_xyz_el;
    du_sum[i]   = rho0 * (duxdx[i] + duydy[i] + duzdz[i]);
    }
}// end of CUDACalculate_SumRho_BonA_SumDu
//--------------------------------------------------------------------------

/**
 *
 * Interface to kernel which calculates three temporary sums in the new pressure formula \n
 * non-linear absorbing case. Scalar values are in constant memory
 *
 * @param [out] rho_sum      - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [out] BonA_sum     - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [out] du_sum       - rho0* (duxdx + duydy + duzdz)
 * @param [in]  rhox,        - acoustic density X
 * @param [in]  rhoy,        - acoustic density Y
 * @param [in]  rhoz,        - acoustic density Z
 * @param [in]  duxdx        - gradient of velocity in X
 * @param [in]  duydy        - gradient of velocity in X
 * @param [in]  duzdz        - gradient of velocity in X
 * @param [in]  Is_BonA_scalar - Is BonA a scalar value (homogeneous)
 * @param [in]  BonA_matrix  - heterogeneous value for BonA
 * @param [in]  Is_rho_scalar - Is rho0 a scalar value (homogeneous)
 * @param [in]  rho0_matrix  - heterogeneous value for rho0
 *
 * @todo revise parameter names, and put scalars to constant memory
 */
void SolverCUDAKernels::Calculate_SumRho_BonA_SumDu(TRealMatrix&       rho_sum,
                                                    TRealMatrix&       BonA_sum,
                                                    TRealMatrix&       du_sum,
                                                    const TRealMatrix& rhox,
                                                    const TRealMatrix& rhoy,
                                                    const TRealMatrix& rhoz,
                                                    const TRealMatrix& duxdx,
                                                    const TRealMatrix& duydy,
                                                    const TRealMatrix& duzdz,
                                                    const bool         Is_BonA_scalar,
                                                    const float*       BonA_matrix,
                                                    const bool         Is_rho0_scalar,
                                                    const float*       rho0_matrix)
{
  // all variants are treated by templates, here you can see all 4 variants
  if (Is_BonA_scalar)
  {
    if (Is_rho0_scalar)
    {
      CUDACalculate_SumRho_BonA_SumDu<true, true>
                                     <<<GetSolverGridSize1D(),
                                        GetSolverBlockSize1D()>>>
                                     (rho_sum.GetRawDeviceData(),
                                      BonA_sum.GetRawDeviceData(),
                                      du_sum.GetRawDeviceData(),
                                      rhox.GetRawDeviceData(),
                                      rhoy.GetRawDeviceData(),
                                      rhoz.GetRawDeviceData(),
                                      duxdx.GetRawDeviceData(),
                                      duydy.GetRawDeviceData(),
                                      duzdz.GetRawDeviceData(),
                                      BonA_matrix,
                                      rho0_matrix);
    }
    else
    {
      CUDACalculate_SumRho_BonA_SumDu<true, false>
                                       <<<GetSolverGridSize1D(),
                                          GetSolverBlockSize1D()>>>
                                       (rho_sum.GetRawDeviceData(),
                                        BonA_sum.GetRawDeviceData(),
                                        du_sum.GetRawDeviceData(),
                                        rhox.GetRawDeviceData(),
                                        rhoy.GetRawDeviceData(),
                                        rhoz.GetRawDeviceData(),
                                        duxdx.GetRawDeviceData(),
                                        duydy.GetRawDeviceData(),
                                        duzdz.GetRawDeviceData(),
                                        BonA_matrix,
                                        rho0_matrix);
    }
  }
  else // BonA is false
  {
   if (Is_rho0_scalar)
    {
    CUDACalculate_SumRho_BonA_SumDu<false, true>
                                     <<<GetSolverGridSize1D(),
                                        GetSolverBlockSize1D()>>>
                                     (rho_sum.GetRawDeviceData(),
                                      BonA_sum.GetRawDeviceData(),
                                      du_sum.GetRawDeviceData(),
                                      rhox.GetRawDeviceData(),
                                      rhoy.GetRawDeviceData(),
                                      rhoz.GetRawDeviceData(),
                                      duxdx.GetRawDeviceData(),
                                      duydy.GetRawDeviceData(),
                                      duzdz.GetRawDeviceData(),
                                      BonA_matrix,
                                      rho0_matrix);
    }
    else
    {
    CUDACalculate_SumRho_BonA_SumDu<false, false>
                                     <<<GetSolverGridSize1D(),
                                        GetSolverBlockSize1D()>>>
                                     (rho_sum.GetRawDeviceData(),
                                      BonA_sum.GetRawDeviceData(),
                                      du_sum.GetRawDeviceData(),
                                      rhox.GetRawDeviceData(),
                                      rhoy.GetRawDeviceData(),
                                      rhoz.GetRawDeviceData(),
                                      duxdx.GetRawDeviceData(),
                                      duydy.GetRawDeviceData(),
                                      duzdz.GetRawDeviceData(),
                                      BonA_matrix,
                                      rho0_matrix);
    }
  }
  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Calculate_SumRho_BonA_SumDu
//------------------------------------------------------------------------------


/**
 * Interface to kernel which computes absorbing term with abosrb_nabla1 and
 * absorb_nabla2, SSE2 version. \n
 * Calculate fft_1 = absorb_nabla1 .* fft_1 \n
 * Calculate fft_2 = absorb_nabla2 .* fft_2 \n
 *
 * @param [in,out] FFT_1
 * @param [in,out] FFT_2
 * @param [in]     absorb_nabla1
 * @param [in]     absorb_nabla2
 */
__global__ void CUDACompute_Absorb_nabla1_2(cuFloatComplex* FFT_1,
                                            cuFloatComplex* FFT_2,
                                            const float*    nabla1,
                                            const float*    nabla2)
{
  for(auto i = GetIndex(); i < CUDADeviceConstants.ComplexTotalElementCount; i += GetStride())
  {
    FFT_1[i] *= nabla1[i];
    FFT_2[i] *= nabla2[i];
  }
}// end of CUDACompute_Absorb_nabla1_2
//------------------------------------------------------------------------------


/**
 * Interface to kernel which computes absorbing term with abosrb_nabla1 and
 * absorb_nabla2, SSE2 version. \n
 * Calculate fft_1 = absorb_nabla1 .* fft_1 \n
 * Calculate fft_2 = absorb_nabla2 .* fft_2 \n
 *
 * @param [in,out] FFT_1
 * @param [in,out] FFT_2
 * @param [in]     absorb_nabla1
 * @param [in]     absorb_nabla2
 */
void SolverCUDAKernels::Compute_Absorb_nabla1_2(TCUFFTComplexMatrix& FFT_1,
                                                TCUFFTComplexMatrix& FFT_2,
                                                const TRealMatrix&   absorb_nabla1,
                                                const TRealMatrix&   absorb_nabla2)
{
  CUDACompute_Absorb_nabla1_2<<<GetSolverGridSize1D(),
                                GetSolverBlockSize1D()>>>
                            (reinterpret_cast<cuFloatComplex*> (FFT_1.GetRawDeviceData()),
                             reinterpret_cast<cuFloatComplex*> (FFT_2.GetRawDeviceData()),
                             absorb_nabla1.GetRawDeviceData(),
                             absorb_nabla2.GetRawDeviceData());

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Compute_Absorb_nabla1_2
//------------------------------------------------------------------------------


/**
 * CUDA Sum sub-terms to calculate new pressure, non-linear case.
 * @@todo needs revision
 * @param [out] p           - new value of pressure
 * @param [in] BonA_temp    - rho0 * (duxdx + duydy + duzdz)
 * @param [in] c2_matrix
 * @param [in] Absorb_tau
 * @param [in] tau_matrix
 * @param [in] Absorb_eta   - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [in] eta_matrix
 */
template <bool Is_c2_scalar, bool Is_tau_eta_scalar>
__global__ void CUDASum_Subterms_nonlinear(float*       p,
                                           const float* BonA_temp,
                                           const float* c2_matrix,
                                           const float* Absorb_tau,
                                           const float* tau_matrix,
                                           const float* Absorb_eta,
                                           const float* eta_matrix)
{
  for(auto i = GetIndex(); i < CUDADeviceConstants.TotalElementCount; i += GetStride())
  {
    const float c2  = (Is_c2_scalar)      ? CUDADeviceConstants.c2  : c2_matrix[i];
    const float tau = (Is_tau_eta_scalar) ? CUDADeviceConstants.Absorb_tau_scalar : tau_matrix[i];
    const float eta = (Is_tau_eta_scalar) ? CUDADeviceConstants.Absorb_eta_scalar : eta_matrix[i];

    p[i] = c2 * (BonA_temp[i] + (CUDADeviceConstants.FFTDivider *
                ((Absorb_tau[i] * tau) - (Absorb_eta[i] * eta))));
  }
}// end of CUDASum_Subterms_nonlinear
//------------------------------------------------------------------------------


/**
 * Interface to CUDA Sum sub-terms to calculate new pressure, non-linear case.
 * @param [in,out] p        - new value of pressure
 * @param [in] BonA_temp    - rho0 * (duxdx + duydy + duzdz)
 * @param [in] Is_c2_scalar
 * @param [in] c2_matrix
 * @param [in] Is_tau_eta_scalar
 * @param [in] Absorb_tau
 * @param [in] tau_matrix
 * @param [in] Absorb_eta   - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [in] eta_matrix
 */
void SolverCUDAKernels::Sum_Subterms_nonlinear(TRealMatrix&       p,
                                               const TRealMatrix& BonA_temp,
                                               const bool         Is_c2_scalar,
                                               const float*       c2_matrix,
                                               const bool         Is_tau_eta_scalar,
                                               const float*       Absorb_tau,
                                               const float*       tau_matrix,
                                               const float*       Absorb_eta,
                                               const float*       eta_matrix)
{
  if (Is_c2_scalar)
  {
    if (Is_tau_eta_scalar)
    {
      CUDASum_Subterms_nonlinear<true, true>
                                <<<GetSolverGridSize1D(),
                                   GetSolverBlockSize1D()>>>
                                (p.GetRawDeviceData(),
                                 BonA_temp.GetRawDeviceData(),
                                 c2_matrix,
                                 Absorb_tau,
                                 tau_matrix,
                                 Absorb_eta,
                                 eta_matrix);
    }
    else
    {
      CUDASum_Subterms_nonlinear<true, false>
                                <<<GetSolverGridSize1D(),
                                   GetSolverBlockSize1D()>>>
                                (p.GetRawDeviceData(),
                                 BonA_temp.GetRawDeviceData(),
                                 c2_matrix,
                                 Absorb_tau,
                                 tau_matrix,
                                 Absorb_eta,
                                 eta_matrix);
    }
  }
  else
  { // c2 is matrix
     if (Is_tau_eta_scalar)
    {
      CUDASum_Subterms_nonlinear<false, true>
                                <<<GetSolverGridSize1D(),
                                   GetSolverBlockSize1D()>>>
                                (p.GetRawDeviceData(),
                                 BonA_temp.GetRawDeviceData(),
                                 c2_matrix,
                                 Absorb_tau,
                                 tau_matrix,
                                 Absorb_eta,
                                 eta_matrix);
    }
    else
    {
      CUDASum_Subterms_nonlinear<false, false>
                                <<<GetSolverGridSize1D(),
                                   GetSolverBlockSize1D()>>>
                                (p.GetRawDeviceData(),
                                 BonA_temp.GetRawDeviceData(),
                                 c2_matrix,
                                 Absorb_tau,
                                 tau_matrix,
                                 Absorb_eta,
                                 eta_matrix);
    }
  }
  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Sum_Subterms_nonlinear
//------------------------------------------------------------------------------


/**
 * CUDA kernel that sums sub-terms to calculate new pressure, linear case.
 * @param [out] p              - new value of p
 * @param [in] Absorb_tau_temp - sub-term with absorb_tau
 * @param [in] Absorb_eta_temp - sub-term with absorb_eta
 * @param [in] Sum_rhoxyz      - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [in] c2_matrix
 * @param [in] tau_matrix
 * @param [in] eta_matrix
 */
template <bool Is_c2_scalar, bool Is_tau_eta_scalar>
__global__ void CUDASum_Subterms_linear(float*       p,
                                        const float* Absorb_tau_temp,
                                        const float* Absorb_eta_temp,
                                        const float* Sum_rhoxyz,
                                        const float* c2_matrix,
                                        const float* tau_matrix,
                                        const float* eta_matrix)
{
  for(auto i = GetIndex(); i < CUDADeviceConstants.TotalElementCount; i += GetStride())
  {
    const float c2  = (Is_c2_scalar)      ? CUDADeviceConstants.c2                : c2_matrix[i];
    const float tau = (Is_tau_eta_scalar) ? CUDADeviceConstants.Absorb_tau_scalar : tau_matrix[i];
    const float eta = (Is_tau_eta_scalar) ? CUDADeviceConstants.Absorb_eta_scalar : eta_matrix[i];

    p[i] = c2 * (Sum_rhoxyz[i] + (CUDADeviceConstants.FFTDivider *
                (Absorb_tau_temp[i] * tau - Absorb_eta_temp[i] * eta)));
  }
}// end of CUDASum_Subterms_linear
//------------------------------------------------------------------------------


/**
 * Interface to kernel that sums sub-terms to calculate new pressure, linear case.
 * @param [out] p              - new value of p
 * @param [in] Absorb_tau_temp - sub-term with absorb_tau
 * @param [in] Absorb_eta_temp - sub-term with absorb_eta
 * @param [in] Sum_rhoxyz      - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [in] Is_c2_scalar
 * @param [in] c2_matrix
 * @param [in] Is_tau_eta_scalar
 * @param [in] tau_matrix
 * @param [in] eta_matrix
 */
void SolverCUDAKernels::Sum_Subterms_linear(TRealMatrix&       p,
                                            const TRealMatrix& Absorb_tau_temp,
                                            const TRealMatrix& Absorb_eta_temp,
                                            const TRealMatrix& Sum_rhoxyz,
                                            const bool         Is_c2_scalar,
                                            const float*       c2_matrix,
                                            const bool         Is_tau_eta_scalar,
                                            const float*       tau_matrix,
                                            const float*       eta_matrix)
{
  if (Is_c2_scalar)
  {
    if (Is_tau_eta_scalar)
    {
      CUDASum_Subterms_linear<true,true>
                             <<<GetSolverGridSize1D(),
                                GetSolverBlockSize1D() >>>
                             (p.GetRawDeviceData(),
                              Absorb_tau_temp.GetRawDeviceData(),
                              Absorb_eta_temp.GetRawDeviceData(),
                              Sum_rhoxyz.GetRawDeviceData(),
                              c2_matrix,
                              tau_matrix,
                              eta_matrix);
    }
    else
    {
      CUDASum_Subterms_linear<true,false>
                             <<<GetSolverGridSize1D(),
                                GetSolverBlockSize1D() >>>
                             (p.GetRawDeviceData(),
                              Absorb_tau_temp.GetRawDeviceData(),
                              Absorb_eta_temp.GetRawDeviceData(),
                              Sum_rhoxyz.GetRawDeviceData(),
                              c2_matrix,
                              tau_matrix,
                              eta_matrix);
    }
   }
  else
  {
    if (Is_tau_eta_scalar)
    {
      CUDASum_Subterms_linear<false,true>
                             <<<GetSolverGridSize1D(),
                                GetSolverBlockSize1D() >>>
                             (p.GetRawDeviceData(),
                              Absorb_tau_temp.GetRawDeviceData(),
                              Absorb_eta_temp.GetRawDeviceData(),
                              Sum_rhoxyz.GetRawDeviceData(),
                              c2_matrix,
                              tau_matrix,
                              eta_matrix);
    }
    else
    {
      CUDASum_Subterms_linear<false,false>
                             <<<GetSolverGridSize1D(),
                                GetSolverBlockSize1D() >>>
                             (p.GetRawDeviceData(),
                              Absorb_tau_temp.GetRawDeviceData(),
                              Absorb_eta_temp.GetRawDeviceData(),
                              Sum_rhoxyz.GetRawDeviceData(),
                              c2_matrix,
                              tau_matrix,
                              eta_matrix);
    }
  }
  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Sum_Subterms_linear
//------------------------------------------------------------------------------


/**
 * CUDA kernel that sums sub-terms for new p, non-linear lossless case.
 * @param [out] p           - new value of pressure
 * @param [in]  rhox
 * @param [in]  rhoy
 * @param [in]  rhoz
 * @param [in]  c2_matrix
 * @param [in]  BonA_matrix
 * @param [in]  rho0_matrix
 */
template<bool Is_c2_scalar, bool Is_BonA_scalar, bool Is_rho0_scalar>
__global__ void CUDASum_new_p_nonlinear_lossless(float*       p,
                                                 const float* rhox,
                                                 const float* rhoy,
                                                 const float* rhoz,
                                                 const float* c2_matrix,
                                                 const float* BonA_matrix,
                                                 const float* rho0_matrix)
{
  for(auto i = GetIndex(); i < CUDADeviceConstants.TotalElementCount; i += GetStride())
  {
    const float c2   = (Is_c2_scalar)   ? CUDADeviceConstants.c2          : c2_matrix[i];
    const float BonA = (Is_BonA_scalar) ? CUDADeviceConstants.BonA_scalar : BonA_matrix[i];
    const float rho0 = (Is_rho0_scalar) ? CUDADeviceConstants.rho0_scalar : rho0_matrix[i];

    const float sum_rho = rhox[i] + rhoy[i] + rhoz[i];

    p[i] = c2 * (sum_rho + (BonA * (sum_rho * sum_rho) / (2.0f * rho0)));
  }
}// end of CUDASum_new_p_nonlinear_lossless
//------------------------------------------------------------------------------

/**
 * Interface to kernel that sums sub-terms for new p, non-linear lossless case.
 * @param [out] p           - new value of pressure
 * @param [in]  rhox
 * @param [in]  rhoy
 * @param [in]  rhoz
 * @param [in]  Is_c2_scalar
 * @param [in]  c2_matrix
 * @param [in]  Is_BonA_scalar
 * @param [in]  BonA_matrix
 * @param [in]  Is_rho0_scalar
 * @param [in]  rho0_matrix
 */
void SolverCUDAKernels::Sum_new_p_nonlinear_lossless(TRealMatrix&       p,
                                                     const TRealMatrix& rhox,
                                                     const TRealMatrix& rhoy,
                                                     const TRealMatrix& rhoz,
                                                     const bool         Is_c2_scalar,
                                                     const float*       c2_matrix,
                                                     const bool         Is_BonA_scalar,
                                                     const float*       BonA_matrix,
                                                     const bool         Is_rho0_scalar,
                                                     const float*       rho0_matrix)
{
  if (Is_c2_scalar)
  {
    if (Is_BonA_scalar)
    {
      if (Is_rho0_scalar)
      {
        CUDASum_new_p_nonlinear_lossless<true, true, true>
                                        <<<GetSolverGridSize1D(),
                                           GetSolverBlockSize1D()>>>
                                        (p.GetRawDeviceData(),
                                         rhox.GetRawDeviceData(),
                                         rhoy.GetRawDeviceData(),
                                         rhoz.GetRawDeviceData(),
                                         c2_matrix,
                                         BonA_matrix,
                                         rho0_matrix);
      }
      else
      {
        CUDASum_new_p_nonlinear_lossless<true, true, false>
                                        <<<GetSolverGridSize1D(),
                                           GetSolverBlockSize1D()>>>
                                        (p.GetRawDeviceData(),
                                         rhox.GetRawDeviceData(),
                                         rhoy.GetRawDeviceData(),
                                         rhoz.GetRawDeviceData(),
                                         c2_matrix,
                                         BonA_matrix,
                                         rho0_matrix);
      }
    }// Is_BonA_scalar= true
    else
    {
      if (Is_rho0_scalar)
      {
        CUDASum_new_p_nonlinear_lossless<true, false, true>
                                        <<<GetSolverGridSize1D(),
                                           GetSolverBlockSize1D()>>>
                                        (p.GetRawDeviceData(),
                                         rhox.GetRawDeviceData(),
                                         rhoy.GetRawDeviceData(),
                                         rhoz.GetRawDeviceData(),
                                         c2_matrix,
                                         BonA_matrix,
                                         rho0_matrix);
      }
      else
      {
        CUDASum_new_p_nonlinear_lossless<true, false, false>
                                        <<<GetSolverGridSize1D(),
                                           GetSolverBlockSize1D()>>>
                                        (p.GetRawDeviceData(),
                                         rhox.GetRawDeviceData(),
                                         rhoy.GetRawDeviceData(),
                                         rhoz.GetRawDeviceData(),
                                         c2_matrix,
                                         BonA_matrix,
                                         rho0_matrix);
      }
    }
  }
  else
  { // Is_c2_scalar == false
   if (Is_BonA_scalar)
    {
      if (Is_rho0_scalar)
      {
        CUDASum_new_p_nonlinear_lossless<false, true, true>
                                        <<<GetSolverGridSize1D(),
                                           GetSolverBlockSize1D()>>>
                                        (p.GetRawDeviceData(),
                                         rhox.GetRawDeviceData(),
                                         rhoy.GetRawDeviceData(),
                                         rhoz.GetRawDeviceData(),
                                         c2_matrix,
                                         BonA_matrix,
                                         rho0_matrix);
      }
      else
      {
        CUDASum_new_p_nonlinear_lossless<false, true, false>
                                        <<<GetSolverGridSize1D(),
                                           GetSolverBlockSize1D()>>>
                                        (p.GetRawDeviceData(),
                                         rhox.GetRawDeviceData(),
                                         rhoy.GetRawDeviceData(),
                                         rhoz.GetRawDeviceData(),
                                         c2_matrix,
                                         BonA_matrix,
                                         rho0_matrix);
      }
    }// Is_BonA_scalar= true
    else
    {
      if (Is_rho0_scalar)
      {
        CUDASum_new_p_nonlinear_lossless<false, false, true>
                                        <<<GetSolverGridSize1D(),
                                           GetSolverBlockSize1D()>>>
                                        (p.GetRawDeviceData(),
                                         rhox.GetRawDeviceData(),
                                         rhoy.GetRawDeviceData(),
                                         rhoz.GetRawDeviceData(),
                                         c2_matrix,
                                         BonA_matrix,
                                         rho0_matrix);
      }
      else
      {
        CUDASum_new_p_nonlinear_lossless<false, false, false>
                                        <<<GetSolverGridSize1D(),
                                           GetSolverBlockSize1D()>>>
                                        (p.GetRawDeviceData(),
                                         rhox.GetRawDeviceData(),
                                         rhoy.GetRawDeviceData(),
                                         rhoz.GetRawDeviceData(),
                                         c2_matrix,
                                         BonA_matrix,
                                         rho0_matrix);
      }
    }
  }

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Sum_new_p_nonlinear_lossless
//------------------------------------------------------------------------------


/**
 * CUDA kernel that Calculates two temporary sums in the new pressure
 * formula, linear absorbing case.
 * @param [out] Sum_rhoxyz  - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [out] Sum_rho0_du - rho0* (duxdx + duydy + duzdz);
 * @param [in]  rhox
 * @param [in]  rhoy
 * @param [in]  rhoz
 * @param [in]  duxdx
 * @param [in]  duydy
 * @param [in]  duzdz
 * @param [in]  rho0_matrix
 */
template<bool Is_rho0_scalar>
__global__ void CUDACalculate_SumRho_SumRhoDu(float*       Sum_rhoxyz,
                                              float*       Sum_rho0_du,
                                              const float* rhox,
                                              const float* rhoy,
                                              const float* rhoz,
                                              const float* dux,
                                              const float* duy,
                                              const float* duz,
                                              const float* rho0_matrix)
{
  for(auto i = GetIndex(); i < CUDADeviceConstants.TotalElementCount; i += GetStride())
  {
    const float rho0 = (Is_rho0_scalar) ? CUDADeviceConstants.rho0_scalar : rho0_matrix[i];

    Sum_rhoxyz[i]  = rhox[i] + rhoy[i] + rhoz[i];
    Sum_rho0_du[i] = rho0 * (dux[i] + duy[i] + duz[i]);
  }
}// end of CUDACalculate_SumRho_SumRhoDu
//------------------------------------------------------------------------------

/**
 * Interface to kernel that Calculates two temporary sums in the new pressure
 * formula, linear absorbing case.
 * @param [out] Sum_rhoxyz  - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [out] Sum_rho0_du - rho0* (duxdx + duydy + duzdz);
 * @param [in]  rhox
 * @param [in]  rhoy
 * @param [in]  rhoz
 * @param [in]  duxdx
 * @param [in]  duydy
 * @param [in]  duzdz
 * @param [in]  Is_rho0_scalar
 * @param [in]  rho0_matrix
 */
void SolverCUDAKernels::Calculate_SumRho_SumRhoDu(TRealMatrix&       Sum_rhoxyz,
                                                  TRealMatrix&       Sum_rho0_du,
                                                  const TRealMatrix& rhox,
                                                  const TRealMatrix& rhoy,
                                                  const TRealMatrix& rhoz,
                                                  const TRealMatrix& duxdx,
                                                  const TRealMatrix& duydy,
                                                  const TRealMatrix& duzdz,
                                                  const bool         Is_rho0_scalar,
                                                  const float*       rho0_matrix)
{
  if (Is_rho0_scalar)
  {
   CUDACalculate_SumRho_SumRhoDu<true>
                                <<<GetSolverGridSize1D(),
                                   GetSolverBlockSize1D()>>>
                                (Sum_rhoxyz.GetRawDeviceData(),
                                 Sum_rho0_du.GetRawDeviceData(),
                                 rhox.GetRawDeviceData(),
                                 rhoy.GetRawDeviceData(),
                                 rhoz.GetRawDeviceData(),
                                 duxdx.GetRawDeviceData(),
                                 duydy.GetRawDeviceData(),
                                 duzdz.GetRawDeviceData(),
                                 rho0_matrix);
  }
  else
  {
   CUDACalculate_SumRho_SumRhoDu<false>
                                <<<GetSolverGridSize1D(),
                                   GetSolverBlockSize1D()>>>
                                (Sum_rhoxyz.GetRawDeviceData(),
                                 Sum_rho0_du.GetRawDeviceData(),
                                 rhox.GetRawDeviceData(),
                                 rhoy.GetRawDeviceData(),
                                 rhoz.GetRawDeviceData(),
                                 duxdx.GetRawDeviceData(),
                                 duydy.GetRawDeviceData(),
                                 duzdz.GetRawDeviceData(),
                                 rho0_matrix);
  }
  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Calculate_SumRho_SumRhoDu
//------------------------------------------------------------------------------

/**
 * CUDA kernel that sums sub-terms for new p, linear lossless case.
 * @param [out] p
 * @param [in]  rhox
 * @param [in]  rhoy
 * @param [in]  rhoz
 * @param [in]  c2_matrix

 */
template <bool Is_c2_scalar>
__global__ void CUDASum_new_p_linear_lossless(float*       p,
                                              const float* rhox,
                                              const float* rhoy,
                                              const float* rhoz,
                                              const float* c2_matrix)
{
  for(auto  i = GetIndex(); i < CUDADeviceConstants.TotalElementCount; i += GetStride())
  {
    const float c2 = (Is_c2_scalar) ? CUDADeviceConstants.c2 : c2_matrix[i];
    p[i] = c2 * (rhox[i] + rhoy[i] + rhoz[i]);
  }
}// end of CUDASum_new_p_linear_lossless
//------------------------------------------------------------------------------

/**
 * Interface to kernel that sums sub-terms for new p, linear lossless case.
 * @param [out] p
 * @param [in]  rhox
 * @param [in]  rhoy
 * @param [in]  rhoz
 * @param [in]  Is_c2_scalar
 * @param [in]  c2_matrix
 */
void SolverCUDAKernels::Sum_new_p_linear_lossless(TRealMatrix& p,
                                                  const TRealMatrix& rhox,
                                                  const TRealMatrix& rhoy,
                                                  const TRealMatrix& rhoz,
                                                  const bool         Is_c2_scalar,
                                                  const float*       c2_matrix)
{
  if (Is_c2_scalar)
  {
    CUDASum_new_p_linear_lossless<true>
                                 <<<GetSolverGridSize1D(),
                                    GetSolverBlockSize1D()>>>
                                (p.GetRawDeviceData(),
                                 rhox.GetRawDeviceData(),
                                 rhoy.GetRawDeviceData(),
                                 rhoz.GetRawDeviceData(),
                                 c2_matrix);
  }
  else
  {
    CUDASum_new_p_linear_lossless<false>
                                 <<<GetSolverGridSize1D(),
                                    GetSolverBlockSize1D()>>>
                                (p.GetRawDeviceData(),
                                 rhox.GetRawDeviceData(),
                                 rhoy.GetRawDeviceData(),
                                 rhoz.GetRawDeviceData(),
                                 c2_matrix);
  }
  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of Sum_new_p_linear_lossless
//------------------------------------------------------------------------------



/**
 * CUDA kernel to transpose a 3D matrix in XY planes if the dim sizes are divisible
 * by 32 in X and Y axes.
 * Every block in a 1D grid transposes a few slabs.
 * Every block is composed of a 2D mesh of threads. The y dim is for up to 4 tiles.
 * Each tile is processed by a single 32-thread warp.
 * The shared memory is used to coalesce memory accesses and the padding is to
 * eliminate bank conflicts.
 *
 * @param [out] OutputMatrixData - Output matrix
 * @param [in]  InputMatrixData  - Input  matrix
 * @param [in]  DimSizes - Dimension sizes of the original matrix
 *
 * @warning  The size X and Y dimensions have to be divisible by 32
 *
 * @warning A blockDim.x has to be 32 (one warp) \n
 *          blockDim.y has to between 1 and 4 (for tiles at once) \n
 *          blockDim.z must stay 1 \n
 *          Grid has to be organized (N, 1 ,1)
 *
 */
__global__ void cudaTrasnposeReal3DMatrixXYSquare(float*       OutputData,
                                                  const float* InputData,
                                                  const dim3   Dimensions)
{
  // this size is fixed shared memory
  // we transpose 4 tiles of 32*32 at the same time, +1 solves bank conflicts
  ///@todo - do I need volatile??
  ///@todo - What about Warp shuffle?
  ///@todo http://www.pixel.io/blog/2013/3/25/fast-matrix-transposition-on-kepler-without-using-shared-mem.html
  volatile __shared__ float shared_tile[4][32][32+1];

  // run over all slabs, one block per slab
  for (auto slabIdx = blockIdx.x; slabIdx < Dimensions.z; slabIdx += gridDim.x)
  {
    // calculate offset of the slab
    const float * InputSlab  = InputData  + (Dimensions.x * Dimensions.y * slabIdx);
          float * OutputSlab = OutputData + (Dimensions.x * Dimensions.y * slabIdx);

    dim3 tileIdx    {0,0,0};
    dim3 tileCount  {Dimensions.x >> 5, Dimensions.y >> 5, 1};

    // go over all all tiles in the row. Transpose 4 rows at the same time
    for (tileIdx.y = threadIdx.y; tileIdx.y < tileCount.y; tileIdx.y += blockDim.y)
    {
      // go over all tiles in the row
      for (tileIdx.x = 0; tileIdx.x < tileCount.x; tileIdx.x++)
      {
        // Go over one tile and load data, unroll does not help
        for (auto row = 0; row < 32; row++)
        {
          shared_tile[threadIdx.y][row][threadIdx.x]
                  = InputSlab[(tileIdx.y * 32   + row) * Dimensions.x +
                              (tileIdx.x * 32)  + threadIdx.x];
        } // load data

        // no need for barrier - warp synchronous programming

        // Go over one tile and store data, unroll does not help
        for (auto row = 0; row < 32; row ++)
        {
          OutputSlab[(tileIdx.x * 32   + row) * Dimensions.y +
                     (tileIdx.y * 32)  + threadIdx.x]
                  = shared_tile[threadIdx.y][threadIdx.x][row];

        } // store data
      } // tile X
    }// tile Y
  } //slab
}// end of cudaTrasnposeReal3DMatrixXYSquare
//------------------------------------------------------------------------------


/**
 * CUDA kernel to transpose a 3D matrix in XY planes of any dimension sizes
 * Every block in a 1D grid transposes a few slabs.
 * Every block is composed of a 2D mesh of threads. The y dim is for up to 4 tiles.
 * Each tile is processed by a single 32-thread warp.
 * The shared memory is used to coalesce memory accesses and the padding is to
 * eliminate bank conflicts.
 * First the full tiles are transposed, then the remainder in the X, then Y and
 * finally the last bit in the bottom right corner.
 *
 * @param [out] OutputMatrixData - Output matrix
 * @param [in]  InputMatrixData  - Input  matrix
 * @param [in]  DimSizes - Dimension sizes of the original matrix
 *
 * @warning  The size X and Y dimensions have to be divisible by 32
 *
 * @warning A blockDim.x has to be 32 (one warp) \n
 *          blockDim.y has to between 1 and 4 (for tiles at once) \n
 *          blockDim.z must stay 1 \n
 *          Grid has to be organized (N, 1 ,1)
 *
 */
__global__ void cudaTrasnposeReal3DMatrixXYRect(float*       OutputData,
                                                const float* InputData,
                                                const dim3   Dimensions)
{
  // this size is fixed shared memory
  // we transpose 4 tiles of 32*32 at the same time, +1 solves bank conflicts
  volatile __shared__ float shared_tile[4][32][32+1];

  // run over all slabs, one block per slab
  for (auto slabIdx = blockIdx.x; slabIdx < Dimensions.z; slabIdx += gridDim.x)
  {
    // calculate offset of the slab
    const float * InputSlab  = InputData  + (Dimensions.x * Dimensions.y * slabIdx);
          float * OutputSlab = OutputData + (Dimensions.x * Dimensions.y * slabIdx);

    dim3 tileIdx = {0,0,0};
    dim3 tileCount = {Dimensions.x >> 5, Dimensions.y >> 5, 1};

    // go over all all tiles in the row. Transpose 4 rows at the same time
    for (tileIdx.y = threadIdx.y; tileIdx.y < tileCount.y; tileIdx.y += blockDim.y)
    {
      //--------------------------- full tiles in X --------------------------//
      // go over all full tiles in the row
      for (tileIdx.x = 0; tileIdx.x < tileCount.x; tileIdx.x++)
      {
        // Go over one tile and load data, unroll does not help
        for (auto row = 0; row < 32; row++)
        {
          shared_tile[threadIdx.y][row][threadIdx.x]
                     = InputSlab[(tileIdx.y * 32   + row) * Dimensions.x +
                                 (tileIdx.x * 32)  + threadIdx.x];
        } // load data
        // no need for barrier - warp synchronous programming
        // Go over one tile and store data, unroll does not help
        for (auto row = 0; row < 32; row ++)
        {
          OutputSlab[(tileIdx.x * 32   + row) * Dimensions.y +
                     (tileIdx.y * 32)  + threadIdx.x]
                  = shared_tile[threadIdx.y][threadIdx.x][row];

        } // store data
      } // tile X

      //--------------------------- reminders in X ---------------------------//
      // go over the remainder tile in X (those that can't fill a 32-warps)
      if ((tileCount.x * 32 + threadIdx.x) < Dimensions.x)
      {
        for (auto row = 0; row < 32; row++)
        {
          shared_tile[threadIdx.y][row][threadIdx.x]
                  = InputSlab[(tileIdx.y   * 32   + row) * Dimensions.x +
                              (tileCount.x * 32)  + threadIdx.x];
        }
      }// load

      // go over the remainder tile in X (those that can't fill a 32-warp)
      for (auto row = 0; (tileCount.x * 32 + row) < Dimensions.x; row++)
      {
        OutputSlab[(tileCount.x * 32   + row) * Dimensions.y +
                   (tileIdx.y   * 32)  + threadIdx.x]
                = shared_tile[threadIdx.y][threadIdx.x][row];
      }// store
    }// tile Y

    //--------------------------- reminders in Y ---------------------------//
    // go over the remainder tile in y (those that can't fill 32 warps)
    // go over all full tiles in the row, first in parallel
    for (tileIdx.x = threadIdx.y; tileIdx.x < tileCount.x; tileIdx.x += blockDim.y)
    {
      // go over the remainder tile in Y (only a few rows)
      for (auto row = 0; (tileCount.y * 32 + row) < Dimensions.y; row++)
      {
        shared_tile[threadIdx.y][row][threadIdx.x]
                = InputSlab[(tileCount.y * 32   + row) * Dimensions.x +
                            (tileIdx.x   * 32)  + threadIdx.x];
      } // load

      // go over the remainder tile in Y (and store only columns)
      if ((tileCount.y * 32 + threadIdx.x) < Dimensions.y)
      {
        for (auto row = 0; row < 32 ; row++)
        {
          OutputSlab[(tileIdx.x   * 32   + row) * Dimensions.y +
                     (tileCount.y * 32)  + threadIdx.x]
                  = shared_tile[threadIdx.y][threadIdx.x][row];
        }
      }// store
    }// reminder Y

    //------------------------ reminder in X and Y -----------------------------//
    if (threadIdx.y == 0)
    {
    // go over the remainder tile in X and Y (only a few rows and colls)
      if ((tileCount.x * 32 + threadIdx.x) < Dimensions.x)
      {
        for (auto row = 0; (tileCount.y * 32 + row) < Dimensions.y; row++)
        {
          shared_tile[threadIdx.y][row][threadIdx.x]
                  = InputSlab[(tileCount.y * 32   + row) * Dimensions.x +
                              (tileCount.x * 32)  + threadIdx.x];
        } // load
      }

      // go over the remainder tile in Y (and store only colls)
      if ((tileCount.y * 32 + threadIdx.x) < Dimensions.y)
      {
        for (auto row = 0; (tileCount.x * 32 + row) < Dimensions.x ; row++)
        {
          OutputSlab[(tileCount.x * 32   + row) * Dimensions.y +
                     (tileCount.y * 32)  + threadIdx.x]
                  = shared_tile[threadIdx.y][threadIdx.x][row];
        }
      }// store
    }// reminder X  and Y
  } //slab
}// end of cudaTrasnposeReal3DMatrixXYRect
//------------------------------------------------------------------------------



/**
 * Transpose a real 3D matrix in the X-Y direction. It is done out-of-place
 * @param [out] OutputMatrixData - Output matrix
 * @param [in]  InputMatrixData  - Input  matrix
 * @param [In]  DimSizes - dimension sizes of the original matrix
 */
void SolverCUDAKernels::TrasposeReal3DMatrixXY(float*       OutputMatrixData,
                                               const float* InputMatrixData,
                                               const dim3&  DimSizes)
{
  // fixed size at the moment, may be tuned based on the domain shape in the future
  if ((DimSizes.x % 32 == 0) && (DimSizes.y % 32 == 0))
  {
    // when the dims are multiples of 32, then use a faster implementation
    cudaTrasnposeReal3DMatrixXYSquare<<<GetSolverTransposeGirdSize(),GetSolverTransposeBlockSize()  >>>
                              (OutputMatrixData, InputMatrixData, DimSizes);
  }
  else
  {
    cudaTrasnposeReal3DMatrixXYRect<<<GetSolverTransposeGirdSize(), GetSolverTransposeBlockSize() >>>
                              (OutputMatrixData, InputMatrixData, DimSizes);
  }

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of TrasposeReal3DMatrixXY
//------------------------------------------------------------------------------



/**
 * CUDA kernel to transpose a 3D matrix in XZ planes if the dim sizes are divisible
 * by 32 in X and Z axes.
 * Every block in a 1D grid transposes a few slabs.
 * Every block is composed of a 2D mesh of threads. The y dim is for up to 4 tiles.
 * Each tile is processed by a single 32-thread warp.
 * The shared memory is used to coalesce memory accesses and the padding is to
 * eliminate bank conflicts.
 *
 * @param [out] OutputMatrixData - Output matrix
 * @param [in]  InputMatrixData  - Input  matrix
 * @param [in]  DimSizes - Dimension sizes of the original matrix
 *
 * @warning  The size X and Z dimensions have to be divisible by 32
 *
 * @warning A blockDim.x has to be 32 (one warp) \n
 *          blockDim.y has to between 1 and 4 (for tiles at once) \n
 *          blockDim.z must stay 1 \n
 *          Grid has to be organized (N, 1 ,1)
 *
 */
__global__ void cudaTrasnposeReal3DMatrixXZSquare(float*       OutputData,
                                                  const float* InputData,
                                                  const dim3   Dimensions)
{
  // this size is fixed shared memory
  // we transpose 4 tiles of 32*32 at the same time, +1 solves bank conflicts
  volatile __shared__ float shared_tile[4][32][32+1];

  // run over all XZ slabs, one block per slab
  for (auto row = blockIdx.x; row < Dimensions.y; row += gridDim.x)
  {
    dim3 tileIdx   = {0,0,0};
    dim3 tileCount = {Dimensions.x >> 5, Dimensions.z >> 5, 1};

    // go over all all tiles in the slab. Transpose multiple slabs at the same time
    for (tileIdx.y = threadIdx.y; tileIdx.y < tileCount.y; tileIdx.y += blockDim.y)
    {
      // go over all tiles in the row
      for (tileIdx.x = 0; tileIdx.x < tileCount.x; tileIdx.x ++)
      {
        // Go over one tile and load data, unroll does not help
        for (auto slab = 0; slab < 32; slab++)
        {
          shared_tile[threadIdx.y][slab][threadIdx.x]
                  = InputData[(tileIdx.y * 32   + slab) * (Dimensions.x * Dimensions.y) +
                              (row * Dimensions.x) +
                               tileIdx.x * 32  + threadIdx.x];
        } // load data
        // no need for barrier - warp synchronous programming

        // Go over one tile and store data, unroll does not help
        for (auto slab = 0; slab < 32; slab++)
        {
          OutputData[(tileIdx.x * 32 + slab) * (Dimensions.y * Dimensions.z) +
                      row * Dimensions.z +
                      tileIdx.y * 32  + threadIdx.x]
                  = shared_tile[threadIdx.y][threadIdx.x][slab];
        } // store data
      } // tile X
    }// tile Y
  } //slab
}// end of cudaTrasnposeReal3DMatrixXZSquare
//------------------------------------------------------------------------------



/**
 * CUDA kernel to transpose a 3D matrix in XZ planes of any dimension sizes
 * Every block in a 1D grid transposes a few slabs.
 * Every block is composed of a 2D mesh of threads. The y dim is for up to 4 tiles.
 * Each tile is processed by a single 32-thread warp.
 * The shared memory is used to coalesce memory accesses and the padding is to
 * eliminate bank conflicts.
 * First the full tiles are transposed, then the remainder in the X, then Y and
 * finally the last bit in the bottom right corner.
 *
 * @param [out] OutputMatrixData - Output matrix
 * @param [in]  InputMatrixData  - Input  matrix
 * @param [in]  DimSizes - Dimension sizes of the original matrix
 *
 * @warning  The size X and Z dimensions have to be divisible by 32
 *
 * @warning A blockDim.x has to be 32 (one warp) \n
 *          blockDim.y has to between 1 and 4 (for tiles at once) \n
 *          blockDim.z must stay 1 \n
 *          Grid has to be organized (N, 1 ,1)
 *
 */
__global__ void cudaTrasnposeReal3DMatrixXZRect(float*       OutputData,
                                                const float* InputData,
                                                const dim3   Dimensions)
{
  // this size is fixed shared memory
  // we transpose 4 tiles of 32*32 at the same time, +1 solves bank conflicts
  volatile __shared__ float shared_tile[4][32][32+1];

  // run over all XZ slabs, one block per slab
  for (auto row = blockIdx.x; row < Dimensions.y; row += gridDim.x)
  {
    dim3 tileIdx   = {0,0,0};
    dim3 tileCount = {Dimensions.x >> 5, Dimensions.z >> 5, 1};

    // go over all all tiles in the XZ slab. Transpose multiple slabs at the same time (on per Z)
    for (tileIdx.y = threadIdx.y; tileIdx.y < tileCount.y; tileIdx.y += blockDim.y)
    {
      // go over all tiles in the row
      for (tileIdx.x = 0; tileIdx.x < tileCount.x; tileIdx.x++)
      {
        // Go over one tile and load data, unroll does not help
        for (auto slab = 0; slab < 32; slab++)
        {
          shared_tile[threadIdx.y][slab][threadIdx.x]
                  = InputData[(tileIdx.y * 32   + slab) * (Dimensions.x * Dimensions.y) +
                               row * Dimensions.x +
                               tileIdx.x * 32  + threadIdx.x];
        } // load data

        // no need for barrier - warp synchronous programming

        // Go over one tile and store data, unroll does not help
        for (auto slab = 0; slab < 32; slab++)
        {
          OutputData[(tileIdx.x * 32 + slab) * (Dimensions.y * Dimensions.z) +
                      row * Dimensions.z +
                      tileIdx.y * 32  + threadIdx.x]
                  = shared_tile[threadIdx.y][threadIdx.x][slab];
        } // store data
      } // tile X

      //--------------------------- reminders in X ---------------------------//
      // go over the remainder tile in X (those that can't fill a 32-warp)
      if ((tileCount.x * 32 + threadIdx.x) < Dimensions.x)
      {
        for (auto slab = 0; slab < 32; slab++)
        {
          shared_tile[threadIdx.y][slab][threadIdx.x]
                  = InputData[(tileIdx.y   * 32  + slab) * (Dimensions.x * Dimensions.y) +
                               row * Dimensions.x +
                               tileCount.x * 32  + threadIdx.x];
        }
      }// load

      // go over the remainder tile in X (those that can't fill 32 warps)
      for (auto slab = 0; (tileCount.x * 32 + slab) < Dimensions.x; slab++)
      {
        OutputData[(tileCount.x * 32  + slab) * (Dimensions.y * Dimensions.z) +
                    row * Dimensions.z +
                    tileIdx.y   * 32  + threadIdx.x]
                = shared_tile[threadIdx.y][threadIdx.x][slab];
      }// store
    }// tile Y

    //--------------------------- reminders in Z -----------------------------//
    // go over the remainder tile in z (those that can't fill a 32-warp)
    // go over all full tiles in the row, first in parallel
    for (tileIdx.x = threadIdx.y; tileIdx.x < tileCount.x; tileIdx.x += blockDim.y)
    {
      // go over the remainder tile in Y (only a few rows)
      for (auto slab = 0; (tileCount.y * 32 + slab) < Dimensions.z; slab++)
      {
        shared_tile[threadIdx.y][slab][threadIdx.x]
                = InputData[(tileCount.y  * 32  + slab) * (Dimensions.x * Dimensions.y) +
                             row * Dimensions.x +
                             tileIdx.x    * 32  + threadIdx.x];

      } // load

      // go over the remainder tile in Y (and store only cullomns)
      if ((tileCount.y * 32 + threadIdx.x) < Dimensions.z)
      {
        for (auto slab = 0; slab < 32; slab++)
        {
          OutputData[(tileIdx.x   * 32  + slab) * (Dimensions.y * Dimensions.z) +
                      row * Dimensions.z +
                      tileCount.y * 32  + threadIdx.x]
                  = shared_tile[threadIdx.y][threadIdx.x][slab];
        }
      }// store
    }// reminder Y

   //------------------------ reminder in X and Z -----------------------------//
  if (threadIdx.y == 0)
    {
    // go over the remainder tile in X and Y (only a few rows and colls)
      if ((tileCount.x * 32 + threadIdx.x) < Dimensions.x)
      {
        for (auto slab = 0; (tileCount.y * 32 + slab) < Dimensions.z; slab++)
        {
          shared_tile[threadIdx.y][slab][threadIdx.x]
                  = InputData[(tileCount.y  * 32  + slab) * (Dimensions.x * Dimensions.y) +
                               row * Dimensions.x +
                               tileCount.x  * 32  + threadIdx.x];
        } // load
      }

      // go over the remainder tile in Z (and store only colls)
      if ((tileCount.y * 32 + threadIdx.x) < Dimensions.z)
      {
        for (auto slab = 0; (tileCount.x * 32 + slab) < Dimensions.x ; slab++)
        {
          OutputData[(tileCount.x * 32  + slab) * (Dimensions.y * Dimensions.z) +
                      row * Dimensions.z +
                      tileCount.y * 32  + threadIdx.x]
                  = shared_tile[threadIdx.y][threadIdx.x][slab];
        }
      }// store
    }// reminder X  and Y
  } //slab
}// end of cudaTrasnposeReal3DMatrixXZRect
//------------------------------------------------------------------------------



/**
 * Transpose a real 3D matrix in the X-Y direction. It is done out-of-place
 * @param [out] OutputMatrixData - Output matrix
 * @param [in]  InputMatrixData  - Input  matrix
 * @param [In]  DimSizes - dimension sizes of the original matrix
 */
void SolverCUDAKernels::TrasposeReal3DMatrixXZ(float*       OutputMatrixData,
                                               const float* InputMatrixData,
                                               const dim3&  DimSizes)
{
  if ((DimSizes.x % 32 == 0) && (DimSizes.y % 32 == 0))
  {
    // when the dims are multiples of 32, then use a faster implementation
    cudaTrasnposeReal3DMatrixXZSquare<<<GetSolverTransposeGirdSize(), GetSolverTransposeBlockSize() >>>
                              (OutputMatrixData, InputMatrixData, DimSizes);
  }
  else
  {
    cudaTrasnposeReal3DMatrixXZRect<<<GetSolverTransposeGirdSize(), GetSolverTransposeBlockSize() >>>
                              (OutputMatrixData, InputMatrixData, DimSizes);
  }

  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of TrasposeReal3DMatrixXZ
//------------------------------------------------------------------------------


/**
 * CUDA kernel to compute velocity shift in the X direction.
 * @param [in, out] FFT_shift_temp
 * @param [in]       x_shift_neg_r
 */
__global__ void CUDAComputeVelocityShiftInX(cuFloatComplex*       FFT_shift_temp,
                                            const cuFloatComplex* x_shift_neg_r)
{
  for (auto i = GetIndex(); i < CUDADeviceConstants.ComplexTotalElementCount; i += GetStride())
  {
    const auto  x = i % CUDADeviceConstants.Complex_Nx;

    FFT_shift_temp[i] = cuCmulf(FFT_shift_temp[i], x_shift_neg_r[x]) * CUDADeviceConstants.FFTDividerX;
  }
}// end of CUDAComputeVelocityShiftInX
//------------------------------------------------------------------------------


/**
 * Compute the velocity shift in Fourier space over the X axis.
 * This kernel work with the original space.
 * @param [in,out] FFT_shift_temp
 * @param [in]     x_shift_neg
 */
void SolverCUDAKernels::ComputeVelocityShiftInX(TCUFFTComplexMatrix&   FFT_shift_temp,
                                                const TComplexMatrix&  x_shift_neg_r)
{
  CUDAComputeVelocityShiftInX<<<GetSolverGridSize1D(),
                                GetSolverBlockSize1D()>>>
                             (reinterpret_cast<cuFloatComplex*>  (FFT_shift_temp.GetRawDeviceData()),
                              reinterpret_cast<const cuFloatComplex*> (x_shift_neg_r.GetRawDeviceData()));
  // check for errors
  checkCudaErrors(cudaGetLastError());
 }// end of ComputeVelocityShiftInX
//------------------------------------------------------------------------------



/**
 * CUDA kernel to compute velocity shift in Y. The matrix is XY transposed.
 * @param [in, out] FFT_shift_temp
 * @param [in]      y_shift_neg_r
 */
__global__ void CUDAComputeVelocityShiftInY(cuFloatComplex*       FFT_shift_temp,
                                            const cuFloatComplex* y_shift_neg_r)
{
  const auto Ny_2 = CUDADeviceConstants.Ny / 2 + 1;
  const auto TotalElementCount = CUDADeviceConstants.Nx * Ny_2 * CUDADeviceConstants.Nz;

  for (auto i = GetIndex(); i < TotalElementCount; i += GetStride())
  {
    // rotated dimensions
    const auto  y = i % Ny_2;

    FFT_shift_temp[i] = cuCmulf(FFT_shift_temp[i], y_shift_neg_r[y]) * CUDADeviceConstants.FFTDividerY;
  }
}// end of CUDAComputeVelocityShiftInY
//------------------------------------------------------------------------------

/**
 * Compute the velocity shift in Fourier space over the Y axis.
 * This kernel work with the transposed space.
 * @param [in,out] FFT_shift_temp
 * @param [in]     x_shift_neg
 */
void SolverCUDAKernels::ComputeVelocityShiftInY(TCUFFTComplexMatrix&  FFT_shift_temp,
                                                const TComplexMatrix& y_shift_neg_r)
{
  CUDAComputeVelocityShiftInY<<<GetSolverGridSize1D(),
                                GetSolverBlockSize1D()>>>
                             (reinterpret_cast<cuFloatComplex*>       (FFT_shift_temp.GetRawDeviceData()),
                              reinterpret_cast<const cuFloatComplex*> (y_shift_neg_r.GetRawDeviceData()));
  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of ComputeVelocityShiftInY
//------------------------------------------------------------------------------


/**
 * CUDA kernel to compute velocity shift in Z. The matrix is XZ transposed
 * @param [in, out] FFT_shift_temp
 * @param [in]      z_shift_neg_r
 */
__global__ void CUDAComputeVelocityShiftInZ(cuFloatComplex*       FFT_shift_temp,
                                            const cuFloatComplex* z_shift_neg_r)
{
  const auto Nz_2 = CUDADeviceConstants.Nz / 2 + 1;
  const auto TotalElementCount = CUDADeviceConstants.Nx * CUDADeviceConstants.Ny * Nz_2;

  for (auto i = GetIndex(); i < TotalElementCount; i += GetStride())
  {
    // rotated dimensions
    const auto  z = i % Nz_2;

     FFT_shift_temp[i] = cuCmulf(FFT_shift_temp[i], z_shift_neg_r[z]) * CUDADeviceConstants.FFTDividerZ;
  }
}// end of CUDAComputeVelocityShiftInZ
//------------------------------------------------------------------------------

/**
 * Compute the velocity shift in Fourier space over the Z axis.
 * This kernel work with the transposed space.
 * @param [in,out] FFT_shift_temp
 * @param [in]     z_shift_neg
 */
void SolverCUDAKernels::ComputeVelocityShiftInZ(TCUFFTComplexMatrix&  FFT_shift_temp,
                                                const TComplexMatrix& z_shift_neg_r)
{
  CUDAComputeVelocityShiftInZ<<<GetSolverGridSize1D(),
                                GetSolverBlockSize1D()>>>
                             (reinterpret_cast<cuFloatComplex*>       (FFT_shift_temp.GetRawDeviceData()),
                              reinterpret_cast<const cuFloatComplex*> (z_shift_neg_r.GetRawDeviceData()));
  // check for errors
  checkCudaErrors(cudaGetLastError());
}// end of ComputeVelocityShiftInZ
//------------------------------------------------------------------------------
