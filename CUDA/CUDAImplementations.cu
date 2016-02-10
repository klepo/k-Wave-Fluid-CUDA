/**
 * @file        CUDAImplementations.cpp
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the all CUDA kernels
 *              for the GPU implementation
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        11 March    2013, 13:10 (created) \n
 *              10 February 2016, 13:12 (revised)
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

#include <CUDA/CUDAImplementations.h>


//----------------------------------------------------------------------------//
//--------------------------------- Macros -----------------------------------//
//----------------------------------------------------------------------------//

/**
 * Check errors of the CUDA routines and print error.
 * @param [in] code  - error code of last routine
 * @param [in] file  - The name of the file, where the error was raised
 * @param [in] line  - What is the line
 * @param [in] Abort - Shall the code abort?
 * @todo - check this routine and do it differently!
 */
inline void gpuAssert(cudaError_t code,
                      string file,
                      int line)
{
  if (code != cudaSuccess)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage,"GPUassert: %s %s %d\n",cudaGetErrorString(code),file.c_str(),line);

    // Throw exception
     throw std::runtime_error(ErrorMessage);
  }
}// end of gpuAssert
//------------------------------------------------------------------------------

/// Define to get the usage easier
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


//----------------------------------------------------------------------------//
//-------------------------------- Constants ---------------------------------//
//----------------------------------------------------------------------------//



//----------------------------------------------------------------------------//
//-------------------------------- Variables ---------------------------------//
//----------------------------------------------------------------------------//


/// Device constants
/**
 * @variable DeviceConstants
 * @brief    This variable holds basic simulation constants for GPU.
 * @details  This variable holds necessary simulation constants in the CUDA GPU
 *           memory.
 */
__constant__ TCUDAImplementations::TDeviceConstants DeviceConstants;


//----------------------------------------------------------------------------//
//----------------------------- Global routines ------------------------------//
//----------------------------------------------------------------------------//

/**
 * Get X coordinate for 3D CUDA block
 * @return X coordinate for 3D CUDA block
 */
inline __device__ size_t GetX()
{
  return threadIdx.x + blockIdx.x * blockDim.x;
}// end of GetX
//------------------------------------------------------------------------------

/**
 * Get Y coordinate for 3D CUDA block
 * @return Y coordinate for 3D CUDA block
 */
inline __device__ size_t GetY()
{
  return threadIdx.y + blockIdx.y * blockDim.y;
}// end of GetY
//------------------------------------------------------------------------------

/**
 * Get Z coordinate for 3D CUDA block
 * @return Z coordinate for 3D CUDA block
 */
inline __device__ size_t GetZ()
{
  return threadIdx.z + blockIdx.z * blockDim.z;
}//end of GetZ
//------------------------------------------------------------------------------

/**
 * Get X stride for 3D CUDA block
 * @return X stride for 3D CUDA block
 */
inline __device__ size_t GetX_Stride()
{
  return blockDim.x * gridDim.x;
}// end of GetX_Stride
//------------------------------------------------------------------------------

/**
 * Get Y stride for 3D CUDA block
 * @return Y stride for 3D CUDA block
 */
inline __device__ size_t GetY_Stride()
{
  return blockDim.y * gridDim.y;
}// end of GetY_Stride
//------------------------------------------------------------------------------

/**
 * Get Z stride for 3D CUDA block
 * @return Z stride for 3D CUDA block
 */
inline __device__ size_t GetZ_Stride()
{
  return blockDim.z * gridDim.z;
}//end of GetZ_Stride
//------------------------------------------------------------------------------





//----------------------------------------------------------------------------//
//----------------------------- Initializations ------------------------------//
//----------------------------------------------------------------------------//

// set singleton instance flag.
bool TCUDAImplementations::InstanceFlag = false;
// Set singleton flag.
TCUDAImplementations* TCUDAImplementations::Instance = NULL;


//----------------------------------------------------------------------------//
//---------------------------------- Public ----------------------------------//
//----------------------------------------------------------------------------//

/**
 * Get instance of the singleton class.
 * @return instance of the singleton class.
 */
TCUDAImplementations* TCUDAImplementations::GetInstance()
{
  if(!InstanceFlag)
  {
    Instance = new TCUDAImplementations();
    InstanceFlag = true;
    return Instance;
  }
  else
  {
    return Instance;
  }
}// end of GetInstance
//------------------------------------------------------------------------------

/**
 * Default destructor for singleton class.
 */
TCUDAImplementations::~TCUDAImplementations()
{
    delete Instance;
    Instance = NULL;
    InstanceFlag = false;
}// end of TCUDAImplementations
//------------------------------------------------------------------------------



/**
 * Set up device constants??
 * @param [in]  FullDimensionSizes - full dimension sizes
 * @param [out] ReducedDimensionSizes - reduced dimension sizes
 */
void TCUDAImplementations::SetUpDeviceConstants(const TDimensionSizes & FullDimensionSizes,
                                                const TDimensionSizes & ReducedDimensionSizes)
{
  TDeviceConstants ConstantsToTransfer;

  // Set values for constant memory
  ConstantsToTransfer.X_Size  = FullDimensionSizes.X;
  ConstantsToTransfer.Y_Size  = FullDimensionSizes.Y;
  ConstantsToTransfer.Z_Size  = FullDimensionSizes.Z;
  ConstantsToTransfer.TotalElementCount = FullDimensionSizes.GetElementCount();
  ConstantsToTransfer.SlabSize = FullDimensionSizes.X * FullDimensionSizes.Y;

  ConstantsToTransfer.Complex_X_Size = ReducedDimensionSizes.X;
  ConstantsToTransfer.Complex_Y_Size = ReducedDimensionSizes.Y;
  ConstantsToTransfer.Complex_Z_Size = ReducedDimensionSizes.Z;
  ConstantsToTransfer.ComplexTotalElementCount = ReducedDimensionSizes.GetElementCount();
  ConstantsToTransfer.ComplexSlabSize = ReducedDimensionSizes.X * ReducedDimensionSizes.Y;

  ConstantsToTransfer.Divider  = 1.0f / FullDimensionSizes.GetElementCount();
  ConstantsToTransfer.DividerX = 1.0f / FullDimensionSizes.X;
  ConstantsToTransfer.DividerY = 1.0f / FullDimensionSizes.Y;
  ConstantsToTransfer.DividerZ = 1.0f / FullDimensionSizes.Z;

  // transfer constants to CUDA constant memory
  gpuErrchk(cudaMemcpyToSymbol(DeviceConstants, &ConstantsToTransfer, sizeof(TDeviceConstants)));     
}// end of SetUpDeviceConstants
//------------------------------------------------------------------------------


/**
 * Kernel to find out the version of the code
 * The list of GPUs can be found at https://en.wikipedia.org/wiki/CUDA
 * @param [out] cudaCodeVersion
 */
__global__ void CUDAGetCUDACodeVersion(int * cudaCodeVersion)
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
int TCUDAImplementations::GetCUDACodeVersion()
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
    gpuErrchk(cudaHostGetDevicePointer<int>(&dCudaCodeVersion, hCudaCodeVersion, 0));

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
  
    gpuErrchk(cudaFreeHost(hCudaCodeVersion));
  }
  
  return (cudaCodeVersion);
}// end of GetCodeVersion
//------------------------------------------------------------------------------


/**
 * CUDA kernel to calculate ux_sgx.
 * Default (heterogeneous case).
 * @param [in, out] ux_sgx  - calculated value
 * @param [in]      FFT_p   - gradient of pressure
 * @param [in]      dt_rho0 - dt_rho_sgx
 * @param [in]      pml     - pml_x
 * @todo To be merged with uy_sgy and yz_sgz
 */
__global__ void CUDACompute_ux_sgx_normalize(float      * ux_sgx,
                                             const float* FFT_p,
                                             const float* dt_rho0,
                                             const float* pml)
{
  // this needs to be done better (may work fine with loop 1D grid)
  // may reduce number of Fx operation

  //@todo X should be unrolled -> Block size X = 32 (always and unrolled), Y and Z is questionable
  for (size_t z = GetZ(); z < DeviceConstants.Z_Size; z += GetZ_Stride())
  {
    for (size_t y = GetY(); y < DeviceConstants.Y_Size; y += GetY_Stride())
    {
      for(size_t x = GetX(); x < DeviceConstants.X_Size; x += GetX_Stride())
      {
        const size_t i = z * DeviceConstants.SlabSize + y * DeviceConstants.X_Size + x;

        const float FFT_p_el = DeviceConstants.Divider * FFT_p[i] * dt_rho0[i];
        const float pml_x = pml[x];

        ux_sgx[i]  = ((ux_sgx[i] * pml_x) - FFT_p_el) * pml_x;

      }// X
    }// Y
  }// Z
}// end of CUDACompute_ux_sgx_normalize
//------------------------------------------------------------------------------

/**
 * Interface to the CUDA kernel computing new version of ux_sgx.
 * Default (heterogeneous case).
 * @param [in, out] ux_sgx  - calculated value
 * @param [in]      FFT_p   - gradient of pressure
 * @param [in]      dt_rho0 - dt_rho_sgx
 * @param [in]      pml     - pml_x
 * @todo To be merged with uy_sgy and yz_sgz
 */
void TCUDAImplementations::Compute_ux_sgx_normalize(TRealMatrix& uxyz_sgxyz,
                                                    const TRealMatrix& FFT_p,
                                                    const TRealMatrix& dt_rho0,
                                                    const TRealMatrix& pml)
{ 
  CUDACompute_ux_sgx_normalize<<<GetSolverGridSize3D(),
                                 GetSolverBlockSize3D() >>>
                              (uxyz_sgxyz.GetRawDeviceData(),
                               FFT_p.GetRawDeviceData(),
                               dt_rho0.GetRawDeviceData(),
                               pml.GetRawDeviceData());

  // check for errors
  gpuErrchk(cudaGetLastError());
}//end of Compute_ux_sgx_normalize
//------------------------------------------------------------------------------


/**
 * CUDA kernel to calculate ux_sgx.
 * This is the case for rho0 being a scalar and a uniform grid.
 * @param [in, out] ux_sgx  - new value of ux
 * @param [in]      FFT_p   - gradient of p
 * @param [in]      dt_rho0 - scalar value for homogeneous media
 * @param [in]      pml     - pml_x
 */
__global__ void CUDACompute_ux_sgx_normalize_scalar_uniform(float      * ux_sgx,
                                                            const float* FFT_p,
                                                            const float  dt_rho0,
                                                            const float* pml)
{
  //@todo this could be in constant memory as well
  const float Divider = dt_rho0 * DeviceConstants.Divider;

  for (size_t z = GetZ(); z < DeviceConstants.Z_Size; z += GetZ_Stride())
  {
    for (size_t y = GetY(); y < DeviceConstants.Y_Size; y += GetY_Stride())
    {
      for(size_t x = GetX(); x < DeviceConstants.X_Size; x += GetX_Stride())
      {
        const size_t i = z * DeviceConstants.SlabSize + y * DeviceConstants.X_Size + x;

        const float FFT_p_el = Divider * FFT_p[i];
        const float pml_x = pml[x];

        ux_sgx[i] =  ((ux_sgx[i] * pml_x) - FFT_p_el) * pml_x;
      }//X
    }//Y
  }//Z
}// end of CUDACompute_ux_sgx_normalize_scalar_uniform
//------------------------------------------------------------------------------

/**
 *  Interface to the CUDA kernel computing new version of ux_sgx.
 * This is the case for rho0 being a scalar and a uniform grid.
 * @param [in, out] ux_sgx  - new value of ux
 * @param [in]      FFT_p   - matrix
 * @param [in]      dt_rho0 - scalar
 * @param [in]      pml     - matrix
 * @todo needs to be merged with uy and uz
 */
void TCUDAImplementations::Compute_ux_sgx_normalize_scalar_uniform(TRealMatrix&       ux_sgx,
                                                                   const TRealMatrix& FFT_p,
                                                                   const float        dt_rho0,
                                                                   const TRealMatrix& pml)
{
  CUDACompute_ux_sgx_normalize_scalar_uniform<<<GetSolverGridSize3D(),
                                                GetSolverBlockSize3D() >>>
                                             (ux_sgx.GetRawDeviceData(),
                                              FFT_p.GetRawDeviceData(),
                                              dt_rho0,
                                              pml.GetRawDeviceData());
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Compute_ux_sgx_normalize_scalar_uniform
//------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate ux_sgx.
 * This is the case for rho0 being a scalar and a non-uniform grid.
 * @param [in,out] ux_sgx     - updated value of ux_sgx
 * @param [in]     FFT_p      - gradient of p
 * @param [in]     dt_rho0    - scalar
 * @param [in]     dxudxn_sgx - matrix dx shift
 * @param [in]     pml        - matrix of pml_x
 * @todo to be merged with uy and uz
 */
__global__ void CUDACompute_ux_sgx_normalize_scalar_nonuniform(float      * ux_sgx,
                                                               const float* FFT_p,
                                                               const float dt_rho0,
                                                               const float* dxudxn_sgx,
                                                               const float* pml)
{
  const float Divider = dt_rho0 * DeviceConstants.Divider;

  for (size_t z = GetZ(); z < DeviceConstants.Z_Size; z += GetZ_Stride())
  {
    for (size_t y = GetY(); y < DeviceConstants.Y_Size; y += GetY_Stride())
    {
      for(size_t x = GetX(); x < DeviceConstants.X_Size; x += GetX_Stride())
      {
        const size_t i = z * DeviceConstants.SlabSize + y * DeviceConstants.X_Size + x;

        const float FFT_p_el = (Divider * dxudxn_sgx[x]) * FFT_p[i];
        const float pml_x = pml[x];

        ux_sgx[i] = (( ux_sgx[i] * pml_x) - FFT_p_el) * pml_x;

      }//X
    }//Y
  }// Z
}// end of CUDACompute_ux_sgx_normalize_scalar_nonuniform
//------------------------------------------------------------------------------

/**
 * Compute a new value of ux_sgx.
 * This is the case for rho0 being a scalar and a non-uniform grid.
 * @param [in,out] ux_sgx     - updated value of ux_sgx
 * @param [in]     FFT_p      - gradient of p
 * @param [in]     dt_rho0    - scalar
 * @param [in]     dxudxn_sgx - matrix dx shift
 * @param [in]     pml        - matrix of pml_x
 * @todo to be merged with uy and uz
 */
void TCUDAImplementations::Compute_ux_sgx_normalize_scalar_nonuniform(TRealMatrix      & ux_sgx,
                                                                      const TRealMatrix& FFT_p,
                                                                      const float        dt_rho0,
                                                                      const TRealMatrix& dxudxn_sgx,
                                                                      const TRealMatrix& pml)
{
  CUDACompute_ux_sgx_normalize_scalar_nonuniform<<<GetSolverGridSize3D(),
                                                   GetSolverBlockSize3D()>>>
                                                (ux_sgx.GetRawDeviceData(),
                                                 FFT_p.GetRawDeviceData(),
                                                 dt_rho0,
                                                 dxudxn_sgx.GetRawDeviceData(),
                                                 pml.GetRawDeviceData());

  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Compute_ux_sgx_normalize_scalar_nonuniform
//------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate uy_sgy.
 * Default (heterogeneous case).
 * @param [in, out] uy_sgy  - calculated value
 * @param [in]      FFT_p   - gradient of pressure
 * @param [in]      dt_rho0 - dt_rho_sgy
 * @param [in]      pml     - pml_y
 * @todo To be merged with uy_sgx and yz_sgz
 */
__global__ void CUDACompute_uy_sgy_normalize(float      * uy_sgy,
                                             const float* FFT_p,
                                             const float* dt_rho0,
                                             const float* pml)
{
  for (size_t z = GetZ(); z < DeviceConstants.Z_Size; z += GetZ_Stride())
  {
    for (size_t y = GetY(); y < DeviceConstants.Y_Size; y += GetY_Stride())
    {
      //@todo - can use shared mem
      const float pml_y = pml[y];
      for(size_t x = GetX(); x < DeviceConstants.X_Size; x += GetX_Stride())
      {
        const size_t i = z * DeviceConstants.SlabSize + y * DeviceConstants.X_Size + x;

        const float FFT_p_el = DeviceConstants.Divider * FFT_p[i] * dt_rho0[i];
        uy_sgy[i] = ((uy_sgy[i] * pml_y) - FFT_p_el) * pml_y;

      }// X
    }// Y
  }// Z
}// end of CUDACompute_uy_sgy_normalize
//------------------------------------------------------------------------------

/**
 * Interface to the CUDA kernel computing new version of uy_sgy.
 * Default (heterogeneous case).
 * @param [in, out] uy_sgy  - calculated value
 * @param [in]      FFT_p   - gradient of pressure
 * @param [in]      dt_rho0 - dt_rho_sgy
 * @param [in]      pml     - pml_y
 * @todo To be merged with uy_sgx and yz_sgz
 */
void TCUDAImplementations::Compute_uy_sgy_normalize(TRealMatrix      & uy_sgy,
                                                    const TRealMatrix& FFT_p,
                                                    const TRealMatrix& dt_rho0,
                                                    const TRealMatrix& pml)
{
  CUDACompute_uy_sgy_normalize<<<GetSolverGridSize3D(),
                                 GetSolverBlockSize3D() >>>
                              (uy_sgy.GetRawDeviceData(),
                               FFT_p.GetRawDeviceData(),
                               dt_rho0.GetRawDeviceData(),
                               pml.GetRawDeviceData());
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Compute_uy_sgy_normalize
//------------------------------------------------------------------------------


/**
 * CUDA kernel to calculate uy_sgy.
 * This is the case for rho0 being a scalar and a uniform grid.
 * @param [in, out] uy_sgy  - new value of uy
 * @param [in]      FFT_p   - gradient of p
 * @param [in]      dt_rho0 - scalar value for homogeneous media
 * @param [in]      pml     - pml_y
 */
__global__ void CUDACompute_uy_sgy_normalize_scalar_uniform(float      * uy_sgy,
                                                            const float* FFT_p,
                                                            const float  dt_rho0,
                                                            const float* pml)
{
  const float Divider = dt_rho0 * DeviceConstants.Divider;

  for (size_t z = GetZ(); z < DeviceConstants.Z_Size; z += GetZ_Stride())
  {
    for (size_t y = GetY(); y < DeviceConstants.Y_Size; y += GetY_Stride())
    {
      //@todo - can use shared mem
      const float pml_y = pml[y];
      for(size_t x = GetX(); x < DeviceConstants.X_Size; x += GetX_Stride())
      {
        const size_t i = z * DeviceConstants.SlabSize + y * DeviceConstants.X_Size + x;

        const float FFT_p_el = Divider * FFT_p[i];
        uy_sgy[i] = (( uy_sgy[i] * pml_y) - FFT_p_el) * pml_y;
      } // X
    } // Y
  }// Z
}// end of CUDACompute_uy_sgy_normalize_scalar_uniform
//------------------------------------------------------------------------------

/**
 * Interface to the CUDA kernel computing new version of uy_sgy.
 * This is the case for rho0 being a scalar and a uniform grid.
 * @param [in, out] uy_sgy  - calculated value
 * @param [in]      FFT_p   - gradient of pressure
 * @param [in]      dt_rho0 - dt_rho_sgy
 * @param [in]      pml     - pml_y
 * @todo To be merged with uy_sgy and yz_sgz
 */
void TCUDAImplementations::Compute_uy_sgy_normalize_scalar_uniform(TRealMatrix      & uy_sgy,
                                                                   const TRealMatrix& FFT_p,
                                                                   const float        dt_rho0,
                                                                   const TRealMatrix& pml)
{
  CUDACompute_uy_sgy_normalize_scalar_uniform<<<GetSolverGridSize3D(),
                                                GetSolverBlockSize3D()>>>
                                             (uy_sgy.GetRawDeviceData(),
                                              FFT_p.GetRawDeviceData(),
                                              dt_rho0,
                                              pml.GetRawDeviceData());
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Compute_uy_sgy_normalize_scalar_uniform
//------------------------------------------------------------------------------


/**
 * CUDA kernel to calculate uy_sgy.
 * This is the case for rho0 being a scalar and a non-uniform grid.
 * @param [in,out] uy_sgy     - updated value of uy_sgy
 * @param [in]     FFT_p      - gradient of p
 * @param [in]     dt_rho0    - scalar
 * @param [in]     dyudyn_sgy - matrix dy shift
 * @param [in]     pml        - matrix of pml_y
 * @todo to be merged with ux and uz
 */
__global__ void CUDACompute_uy_sgy_normalize_scalar_nonuniform(float      * uy_sgy,
                                                               const float* FFT_p,
                                                               const float  dt_rho0,
                                                               const float* dyudyn_sgy,
                                                               const float* pml)
{
  const float Divider = dt_rho0 * DeviceConstants.Divider;;
  for (size_t z = GetZ(); z < DeviceConstants.Z_Size; z += GetZ_Stride())
  {
    for (size_t y = GetY(); y < DeviceConstants.Y_Size; y += GetY_Stride())
    {
      //@todo - can use shared mem
      const float pml_y = pml[y];
      const float DyDivider = Divider * dyudyn_sgy[y];

      for(size_t x = GetX(); x < DeviceConstants.X_Size; x += GetX_Stride())
      {
        const size_t i = z * DeviceConstants.SlabSize + y * DeviceConstants.X_Size + x;

        const float FFT_p_el = DyDivider * FFT_p[i];
        uy_sgy[i] = ((uy_sgy[i] * pml_y) - FFT_p_el) * pml_y;

      }//X
    }//Y
  }//Z
}// end of CudaCompute_uy_sgy_normalize_scalar_nonuniform
//------------------------------------------------------------------------------

/**
 * Compute a new value of uy_sgy.
 * This is the case for rho0 being a scalar and a non-uniform grid.
 * @param [in,out] uy_sgy     - updated value of uy_sgy
 * @param [in]     FFT_p      - gradient of p
 * @param [in]     dt_rho0    - scalar
 * @param [in]     dyudyn_sgy - matrix d shift
 * @param [in]     pml        - matrix of pml_y
 * @todo to be merged with ux and uz.
 */
void TCUDAImplementations::Compute_uy_sgy_normalize_scalar_nonuniform(TRealMatrix      & uy_sgy,
                                                                      const TRealMatrix& FFT_p,
                                                                      const float        dt_rho0,
                                                                      const TRealMatrix& dyudyn_sgy,
                                                                      const TRealMatrix& pml)
{
  CUDACompute_uy_sgy_normalize_scalar_nonuniform<<<GetSolverGridSize3D(),
                                                   GetSolverBlockSize3D()>>>
                                                (uy_sgy.GetRawDeviceData(),
                                                 FFT_p.GetRawDeviceData(),
                                                 dt_rho0,
                                                 dyudyn_sgy.GetRawDeviceData(),
                                                 pml.GetRawDeviceData());
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Compute_uy_sgy_normalize_scalar_nonuniform
//------------------------------------------------------------------------------


/**
 * CUDA kernel to calculate uz_sgz.
 * Default (heterogeneous case).
 * @param [in, out] uz_sgz  - calculated value
 * @param [in]      FFT_p   - gradient of pressure
 * @param [in]      dt_rho0 - dt_rho_sgz
 * @param [in]      pml     - pml_z
 * @todo To be merged with uy_sgz and ux_sgx
 */
__global__ void CUDACompute_uz_sgz_normalize(float      * uz_sgz,
                                             const float* FFT_p,
                                             const float* dt_rho0,
                                             const float* pml)
{
  for (size_t z = GetZ(); z < DeviceConstants.Z_Size; z += GetZ_Stride())
  {
    const float pml_z = pml[z];
    for (size_t y = GetY(); y < DeviceConstants.Y_Size; y += GetY_Stride())
    {
      for(size_t x = GetX(); x < DeviceConstants.X_Size; x += GetX_Stride())
      {
        const size_t i = z * DeviceConstants.SlabSize + y * DeviceConstants.X_Size + x;

        const float FFT_p_el = DeviceConstants.Divider * FFT_p[i] * dt_rho0[i];
        uz_sgz[i] = ((uz_sgz[i] * pml_z) - FFT_p_el ) * pml_z;
      }
    }
  }
}// end of CUDACompute_uz_sgz_normalize
//------------------------------------------------------------------------------

/**
 * Interface to the CUDA kernel computing new version of uz_sgz.
 * Default (heterogeneous case).
 * @param [in, out] uz_sgz  - calculated value
 * @param [in]      FFT_p   - gradient of pressure
 * @param [in]      dt_rho0 - dt_rho_sgy
 * @param [in]      pml     - pml_y
 * @todo To be merged with ux_sgx and uy_sgy
 */
void TCUDAImplementations::Compute_uz_sgz_normalize(TRealMatrix      & uz_sgz,
                                                    const TRealMatrix& FFT_p,
                                                    const TRealMatrix& dt_rho0,
                                                    const TRealMatrix& pml)
{
  CUDACompute_uz_sgz_normalize<<<GetSolverGridSize3D(),
                                 GetSolverBlockSize3D()>>>
                              (uz_sgz.GetRawDeviceData(),
                               FFT_p.GetRawDeviceData(),
                               dt_rho0.GetRawDeviceData(),
                               pml.GetRawDeviceData());

  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Compute_uz_sgz_normalize
//------------------------------------------------------------------------------


/**
 * CUDA kernel to calculate uz_sgz.
 * This is the case for rho0 being a scalar and a uniform grid.
 * @param [in, out] uz_sgz  - new value of uz
 * @param [in]      FFT_p   - gradient of p
 * @param [in]      dt_rho0 - scalar value for homogeneous media
 * @param [in]      pml     - pml_z
 */
__global__ void CUDACompute_uz_sgz_normalize_scalar_uniform(float      * uz_sgz,
                                                            const float* FFT_p,
                                                            const float  dt_rho0,
                                                            const float* pml)
{
  const float Divider = dt_rho0 * DeviceConstants.Divider;

  for (size_t z = GetZ(); z < DeviceConstants.Z_Size; z += GetZ_Stride())
  {
    const float pml_z = pml[z];
    for (size_t y = GetY(); y < DeviceConstants.Y_Size; y += GetY_Stride())
    {
      for(size_t x = GetX(); x < DeviceConstants.X_Size; x += GetX_Stride())
      {
        const size_t i = z * DeviceConstants.SlabSize + y * DeviceConstants.X_Size + x;

        const float FFT_p_el = Divider * FFT_p[i];
        uz_sgz[i] = ((uz_sgz[i] * pml_z) - FFT_p_el) * pml_z;
      }// X
    }// Y
  }// Z
}// end of CUDACompute_uz_sgz_normalize_scalar_uniform
//------------------------------------------------------------------------------

/**
 * Interface to the CUDA kernel computing new version of uz_sgz.
 * This is the case for rho0 being a scalar and a uniform grid.
 * @param [in, out] uz_sgz  - calculated value
 * @param [in]      FFT_p   - gradient of pressure
 * @param [in]      dt_rho0 - dt_rho_sgz
 * @param [in]      pml     - pml_z
 * @todo To be merged with uy_sgy and yx_sgx
 */
void TCUDAImplementations::Compute_uz_sgz_normalize_scalar_uniform(TRealMatrix      & uz_sgz,
                                                                   const TRealMatrix& FFT_p,
                                                                   const float        dt_rho0,
                                                                   const TRealMatrix& pml)
{
  CUDACompute_uz_sgz_normalize_scalar_uniform<<<GetSolverGridSize3D(),
                                                GetSolverBlockSize3D()>>>
                                             (uz_sgz.GetRawDeviceData(),
                                              FFT_p.GetRawDeviceData(),
                                              dt_rho0,
                                              pml.GetRawDeviceData());
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Compute_uz_sgz_normalize_scalar_uniform
//------------------------------------------------------------------------------


/**
 * CUDA kernel to calculate uz_sgz.
 * This is the case for rho0 being a scalar and a non-uniform grid.
 * @param [in,out] uz_sgz     - updated value of uz_sgz
 * @param [in]     FFT_p      - gradient of p
 * @param [in]     dt_rho0    - scalar
 * @param [in]     dzudzn_sgz - matrix dz shift
 * @param [in]     pml        - matrix of pml_z
 * @todo to be merged with ux and uy
 */
__global__ void CUDACompute_uz_sgz_normalize_scalar_nonuniform(float      * uz_sgz,
                                                               const float* FFT_p,
                                                               const float  dt_rho0,
                                                               const float* dzudzn_sgz,
                                                               const float* pml)
{
  const float Divider = dt_rho0 * DeviceConstants.Divider;

  for (size_t z = GetZ(); z < DeviceConstants.Z_Size; z += GetZ_Stride())
  {
    const float pml_z = pml[z];
    const float DzDivider = Divider * dzudzn_sgz[z];

    for (size_t y = GetY(); y < DeviceConstants.Y_Size; y += GetY_Stride())
    {
      for(size_t x = GetX(); x < DeviceConstants.X_Size; x += GetX_Stride())
      {
        const size_t i = z * DeviceConstants.SlabSize + y * DeviceConstants.X_Size + x;

        const float FFT_p_el = DzDivider * FFT_p[i];
        uz_sgz[i] = ((uz_sgz[i] * pml_z) - FFT_p_el) * pml_z;
      }//X
    }//Y
  }//Z
}// end of CUDACompute_uz_sgz_normalize_scalar_nonuniform
//-----------------------------------------------------------------------------

/**
 * Interface to the CUDA kernel computing new version of uz_sgz.
 * This is the case for rho0 being a scalar and a non-uniform grid.
 * @param [in,out] uz_sgz     - updated value of uz_sgz
 * @param [in]     FFT_p      - gradient of p
 * @param [in]     dt_rho0    - scalar
 * @param [in]     dyudyn_sgz - matrix d shift
 * @param [in]     pml        - matrix of pml_z
 * @todo to be merged with ux and uy
 */
void TCUDAImplementations::Compute_uz_sgz_normalize_scalar_nonuniform(TRealMatrix      & uz_sgz,
                                                                      const TRealMatrix& FFT_p,
                                                                      const float        dt_rho0,
                                                                      const TRealMatrix& dzudzn_sgz,
                                                                      const TRealMatrix& pml)
{
  CUDACompute_uz_sgz_normalize_scalar_nonuniform<<<GetSolverGridSize3D(),
                                                   GetSolverBlockSize3D()>>>
                                                (uz_sgz.GetRawDeviceData(),
                                                 FFT_p.GetRawDeviceData(),
                                                 dt_rho0,
                                                 dzudzn_sgz.GetRawDeviceData(),
                                                 pml.GetRawDeviceData());
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Compute_uz_sgz_normalize_scalar_nonuniform
//------------------------------------------------------------------------------


/**
 * CUDa kernel adding transducer data to ux_sgx
 * @param [in, out] ux_sgx             - here we add the signal
 * @param [in]      us_index           - where to add the signal (source)
 * @param [in]      us_index_size      - size of source
 * @param [in, out] delay_mask         - delay mask to push the signal in the domain (incremented per invocation)
 * @param [in]      transducer_signal  - transducer signal
 */
__global__ void CUDAAddTransducerSource(float       * ux_sgx,
                                        const size_t* u_source_index,
                                        const size_t  u_source_index_size,
                                              size_t* delay_mask,
                                        const float * transducer_signal)
{
  for (size_t i = GetX(); i < u_source_index_size; i += GetX_Stride())
  {
    ux_sgx[u_source_index[i]] += transducer_signal[delay_mask[i]];
    delay_mask[i] ++;
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
void TCUDAImplementations::AddTransducerSource(TRealMatrix       & ux_sgx,
                                               const TIndexMatrix& u_source_index,
                                               TIndexMatrix      & delay_mask,
                                               const TRealMatrix & transducer_signal)
{
  const size_t u_source_index_size = u_source_index.GetTotalElementCount();

  // Grid size is calculated based on the source size
  int CUDAGridSize1D  = (u_source_index_size  + GetSolverBlockSize1D() - 1 ) / GetSolverBlockSize1D();
  //@todo here should be a test not to generate too much blocks, and balance workload

  //@todo Source signal should go to constant memory
  CUDAAddTransducerSource<<<CUDAGridSize1D, GetSolverBlockSize1D()>>>
                         (ux_sgx.GetRawDeviceData(),
                          u_source_index.GetRawDeviceData(),
                          u_source_index_size,
                          delay_mask.GetRawDeviceData(),
                          transducer_signal.GetRawDeviceData());
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of AddTransducerSource
//------------------------------------------------------------------------------



/**
 * CUDA kernel to add in velocity source terms.
 *
 * @param [in, out] uxyz_sgxyz          - velocity matrix to update
 * @param [in]      u_source_input      - Source input to add
 * @param [in]      u_source_index      - Index matrix
 * @param [in]      u_source_index_size - size of the source
 * @param [in]      t_index             - Actual time step
 * @param [in]      u_source_mode       - Mode 0 = Dirichlet boundary, 1 = add in
 * @param [in]      u_source_many       - 0 = One series, 1 = multiple series
 */
__global__ void CUDAAdd_u_source(float        * uxyz_sgxyz,
                                 const float  * u_source_input,
                                 const size_t * u_source_index,
                                 const size_t   u_source_index_size,
                                 const size_t   t_index,
                                 const size_t   u_source_mode,
                                 const size_t   u_source_many)
{
  // Set 1D or 2D step for source
  size_t index2D = (u_source_many == 0) ? t_index : t_index * u_source_index_size;

  if (u_source_mode == 0)
  {
    for (size_t i = GetX(); i < u_source_index_size; i += GetX_Stride())
    {
      uxyz_sgxyz[u_source_index[i]]  = (u_source_many == 0) ?  u_source_input[index2D] :
                                                               u_source_input[index2D + i];
    }// for
  }// end of Dirichlet

  if (u_source_mode == 1)
  {
    for (size_t i  = GetX(); i < u_source_index_size; i += GetX_Stride())
    {
      uxyz_sgxyz[u_source_index[i]] += (u_source_many == 0) ?  u_source_input[index2D] :
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
 * @param [in] u_source_mode   - Mode 0 = dirichlet boundary, 1 = add in
 * @param [in] u_source_many   - 0 = One series, 1 = multiple series
 */
void TCUDAImplementations::Add_u_source(TRealMatrix       & uxyz_sgxyz,
                                        const TRealMatrix & u_source_input,
                                        const TIndexMatrix& u_source_index,
                                        const size_t        t_index,
                                        const size_t        u_source_mode,
                                        const size_t        u_source_many)
{
  const size_t u_source_index_size = u_source_index.GetTotalElementCount();

  // Grid size is calculated based on the source size
  const int CUDAGridSize1D  = (u_source_index_size  + GetSolverBlockSize1D() - 1 ) / GetSolverBlockSize1D();
  //@todo here should be a test not to generate too much blocks, and balance workload

  CUDAAdd_u_source<<< CUDAGridSize1D, GetSolverBlockSize1D()>>>
                  (uxyz_sgxyz.GetRawDeviceData(),
                   u_source_input.GetRawDeviceData(),
                   u_source_index.GetRawDeviceData(),
                   u_source_index_size,
                   t_index,
                   u_source_mode,
                   u_source_many);

  // check for errors
  gpuErrchk(cudaGetLastError());
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
 * @param [in]  p_source_mode  - Mode 0 = dirichlet boundary, 1 = add in
 * @param [in]  p_source_many  - 0 = One series, 1 = multiple series
 */
__global__ void CUDAAdd_p_source(float       * rhox,
                                 float       * rhoy,
                                 float       * rhoz,
                                 const float * p_source_input,
                                 const size_t* p_source_index,
                                 const size_t  p_source_index_size,
                                 const size_t  t_index,
                                 const size_t  p_source_mode,
                                 const size_t  p_source_many)
{
  // Set 1D or 2D step for source
  size_t index2D = (p_source_many == 0) ? t_index : t_index * p_source_index_size;

  if (p_source_mode == 0)
  {
    if (p_source_many == 0)
    { // single signal
      for (size_t i = GetX(); i < p_source_index_size; i += GetX_Stride())
      {
        rhox[p_source_index[i]] = p_source_input[index2D];
        rhoy[p_source_index[i]] = p_source_input[index2D];
        rhoz[p_source_index[i]] = p_source_input[index2D];
      }
    }
    else
    { // multiple signals
      for (size_t i = GetX(); i < p_source_index_size; i += GetX_Stride())
      {
        rhox[p_source_index[i]] = p_source_input[index2D + i];
        rhoy[p_source_index[i]] = p_source_input[index2D + i];
        rhoz[p_source_index[i]] = p_source_input[index2D + i];
      }
    }
  }// end mode == 0 (Cauchy)

  if (p_source_mode == 1)
  {
    if (p_source_many == 0)
    { // single signal
      for (size_t i = GetX(); i < p_source_index_size; i += GetX_Stride())
      {
        rhox[p_source_index[i]] += p_source_input[index2D];
        rhoy[p_source_index[i]] += p_source_input[index2D];
        rhoz[p_source_index[i]] += p_source_input[index2D];
      }
    }
    else
    { // multiple signals
      for (size_t i = GetX(); i < p_source_index_size; i += GetX_Stride())
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
 * @param [in]  p_source_mode  - Mode 0 = dirichlet boundary, 1 = add in
 * @param [in]  p_source_many  - 0 = One series, 1 = multiple series
 */
void TCUDAImplementations::Add_p_source(TRealMatrix       & rhox,
                                        TRealMatrix       & rhoy,
                                        TRealMatrix       & rhoz,
                                        const TRealMatrix & p_source_input,
                                        const TIndexMatrix& p_source_index,
                                        const size_t        t_index,
                                        const size_t        p_source_mode,
                                        const size_t        p_source_many)
{
  const size_t p_source_index_size = p_source_index.GetTotalElementCount();

  // Grid size is calculated based on the source size  
  int CUDAGridSize1D  = (p_source_index_size  + GetSolverBlockSize1D() - 1 ) / GetSolverBlockSize1D();
  //@todo here should be a test not to generate too much blocks, and balance workload

  CUDAAdd_p_source<<<CUDAGridSize1D,GetSolverBlockSize1D()>>>
                  (rhox.GetRawDeviceData(),
                   rhoy.GetRawDeviceData(),
                   rhoz.GetRawDeviceData(),
                   p_source_input.GetRawDeviceData(),
                   p_source_index.GetRawDeviceData(),
                   p_source_index_size,
                   t_index,
                   p_source_mode,
                   p_source_many);

  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Add_p_source
//------------------------------------------------------------------------------

/**
 * CUDA kernel Compute u = dt ./ rho0_sgx .* u.
 *
 * @param [in, out] uxyz_sgxyz - data stored in u matrix
 * @param [in]      dt_rho0_sg - inner member of the equation
 *
 * @todo to me merged for ux, uy and uz
 */
__global__  void CUDACompute_dt_rho_sg_mul_u(float      * uxyz_sgxyz,
                                             const float* dt_rho0_sg)

{
  const float ScaledDivider = DeviceConstants.Divider * 0.5f;

  for (size_t i = GetX(); i < DeviceConstants.TotalElementCount; i += GetX_Stride())
  {
    uxyz_sgxyz[i] = uxyz_sgxyz[i] * dt_rho0_sg[i] *  ScaledDivider;
  }
}// end of CudaCompute_dt_rho_sg_mul_ifft_div_2
//------------------------------------------------------------------------------

/**
 * Interface to CUDA Compute u = dt ./ rho0_sgx .* ifft(FFT).
 *
 * @param [in, out] uxyz_sgxyz - data stored in u matrix
 * @param [in]      dt_rho0_sg - inner member of the equation
 * @param [in]      FFT        - matix storing the k-space temp result (input).
 *                               This will be overridden
 * @todo to me merged for ux, uy and uz
 */
void TCUDAImplementations::Compute_dt_rho_sg_mul_ifft_div_2(TRealMatrix        & uxyz_sgxyz,
                                                            const TRealMatrix  & dt_rho0_sg,
                                                            TCUFFTComplexMatrix& FFT)
{
  // take the 3D ifft
  FFT.Compute_FFT_3D_C2R(uxyz_sgxyz);

  CUDACompute_dt_rho_sg_mul_u<<<GetSolverGridSize1D(),
                                GetSolverBlockSize1D()>>>
                            (uxyz_sgxyz.GetRawDeviceData(),
                             dt_rho0_sg.GetRawDeviceData());

  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Compute_dt_rho_sg_mul_ifft_div_2
//------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate u = dt ./ rho0_sgx .* u.
 * if rho0_sgx is scalar, uniform case.
 *
 * @param [in, out] uxyz_sgxyz - data stored in u matrix
 * @param [in]      dt_rho0_sg - inner member of the equation
 */
__global__  void CUDACompute_dt_rho_sg_mul_u(float     * uxyz_sgxyz,
                                             const float dt_rho_0_sg)
{
  const float ScaledDivider = DeviceConstants.Divider * 0.5f * dt_rho_0_sg;

  for (size_t i = GetX(); i < DeviceConstants.TotalElementCount; i += GetX_Stride())
  {
    uxyz_sgxyz[i] *= ScaledDivider;
  }
}// end of CUDACompute_dt_rho_sg_mul_u
//------------------------------------------------------------------------------

/**
 * Interface to CUDA Compute u = dt ./ rho0_sgx .* ifft(FFT).
 * if rho0_sgx is scalar, uniform case.
 *
 * @param [in, out] uxyz_sgxyz   - data stored in u matrix
 * @param [in]      dt_rho0_sgx - scalar value
 * @param [in]      FFT          - FFT matrix (data will be overwritten)
 *
 * @todo to me merged for ux, uy and uz
 */
void TCUDAImplementations::Compute_dt_rho_sg_mul_ifft_div_2(TRealMatrix        & uxyz_sgxyz,
                                                            const float          dt_rho0_sg,
                                                            TCUFFTComplexMatrix& FFT)
{
  // take the 3D ifft
  FFT.Compute_FFT_3D_C2R(uxyz_sgxyz);

  CUDACompute_dt_rho_sg_mul_u<<<GetSolverGridSize1D(),
                                GetSolverBlockSize1D()>>>
                             (uxyz_sgxyz.GetRawDeviceData(),
                              dt_rho0_sg);

  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Compute_dt_rho_sg_mul_ifft_div_2
//------------------------------------------------------------------------------



/**
 * CUDA kernel to Compute u = dt./rho0_sgx .* ifft (FFT).
 * if rho0_sgx is scalar, nonuniform  non uniform grid, x component.
 * @param [in, out] ux_sgx       - output value of u
 * @param [in]      dt_rho0_sgx - scalar value
 * @param [in]      dxudxn_sgx   - non-uniform mapping
 */
__global__ void CUDACompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_x(float      * ux_sgx,
                                                                         const float  dt_rho_0_sgx,
                                                                         const float* dxudxn_sgx)
{
  const float ScaledDivider = DeviceConstants.Divider * 0.5f * dt_rho_0_sgx;

  for (size_t z = GetZ(); z < DeviceConstants.Z_Size; z += GetZ_Stride())
    for (size_t y = GetY(); y < DeviceConstants.Y_Size; y += GetY_Stride())
      for (size_t x = GetX(); x < DeviceConstants.X_Size; x += GetX_Stride())
      {
        const size_t i = z * DeviceConstants.SlabSize + y * DeviceConstants.X_Size + x;
        ux_sgx[i] = ux_sgx[i] * ScaledDivider * dxudxn_sgx[x];
      }

}// end of CUDACompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_x
//------------------------------------------------------------------------------


/**
 * Interface to CUDA kernel to Compute ux = dt./rho0_sgx .* ifft (FFT).
 * if rho0_sgx is scalar, nonuniform  non uniform grid, x component.
 * @param [in, out] ux_sgx       - output value of u
 * @param [in]      dt_rho0_sgx - scalar value
 * @param [in]      dxudxn_sgx   - non-uniform mapping
 * @param [in]      FFT          - FFT matrix (data will be overwritten)
 */
void TCUDAImplementations::Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_x(TRealMatrix        & ux_sgx,
                                                                                const float          dt_rho0_sgx,
                                                                                const TRealMatrix  & dxudxn_sgx,
                                                                                TCUFFTComplexMatrix& FFT)
{
  // take the 3D iFFT
  FFT.Compute_FFT_3D_C2R(ux_sgx);

  CUDACompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_x<<<GetSolverGridSize3D(),
                                                             GetSolverBlockSize3D()>>>
                                                          (ux_sgx.GetRawDeviceData(),
                                                           dt_rho0_sgx,
                                                           dxudxn_sgx.GetRawDeviceData());

 // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_x
//------------------------------------------------------------------------------

/**
 * CUDA kernel to Compute uy = dt./rho0_sgy .* ifft (FFT).
 * if rho0_sgy is scalar, nonuniform  non uniform grid, y component.
 * @param [in, out] uy_sgy       - output value of u
 * @param [in]      dt_rho0_sgy - scalar value
 * @param [in]      dxudxn_sgy   - non-uniform mapping
 */
__global__ void CUDACompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_y(float      * uy_sgy,
                                                                         const float  dt_rho_0_sgy,
                                                                         const float* dyudyn_sgy)
{

  const float ScaledDivider = DeviceConstants.Divider * 0.5f * dt_rho_0_sgy;

  for (size_t z = GetZ(); z < DeviceConstants.Z_Size; z += GetZ_Stride())
  {
    for (size_t y = GetY(); y < DeviceConstants.Y_Size; y += GetY_Stride())
    {
      const float dyudyn_sgy_data = dyudyn_sgy[y] * ScaledDivider;
      for(size_t x = GetX(); x < DeviceConstants.X_Size; x += GetX_Stride())
      {
         const size_t i = z * DeviceConstants.SlabSize + y * DeviceConstants.X_Size + x;
         uy_sgy[i] = uy_sgy[i] * dyudyn_sgy_data;
      }
    }
  }
}// end of CUDACompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_y
//------------------------------------------------------------------------------

/**
 * Interface to CUDA kernel to Compute uy = dt./rho0_sgy .* ifft (FFT).
 * if rho0_sgy is scalar, nonuniform  non uniform grid, y component.
 * @param [in, out] uy_sgy       - output value of u
 * @param [in]      dt_rho0_sgy - scalar value
 * @param [in]      dyudyn_sgy   - non-uniform mapping
 * @param [in]      FFT          - FFT matrix (data will be overwritten)
 */
void TCUDAImplementations::Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_y(TRealMatrix        & uy_sgy,
                                                                                const float          dt_rho0_sgy,
                                                                                const TRealMatrix  & dyudyn_sgy,
                                                                                TCUFFTComplexMatrix& FFT)
{
  // take the 3D iFFT
  FFT.Compute_FFT_3D_C2R(uy_sgy);

  CUDACompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_y<<<GetSolverGridSize3D(),
                                                             GetSolverBlockSize3D()>>>
                                                          (uy_sgy.GetRawDeviceData(),
                                                           dt_rho0_sgy,
                                                           dyudyn_sgy.GetRawDeviceData());

  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_y
//------------------------------------------------------------------------------

/**
 * CUDA kernel to Compute uy = dt./rho0_sgz .* ifft (FFT).
 * if rho0_sgz is scalar, nonuniform  non uniform grid, z component.
 * @param [in, out] uy_sgz       - output value of u
 * @param [in]      dt_rho0_sgz - scalar value
 * @param [in]      dxudxn_sgz   - non-uniform mapping
 */
__global__ void CUDACompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_z(float      * uz_sgz,
                                                                         const float  dt_rho_0_sgz,
                                                                         const float* dzudzn_sgz)
{
  const float ScaledDivider = DeviceConstants.Divider * 0.5f * dt_rho_0_sgz;

  for (size_t z = GetZ(); z < DeviceConstants.Z_Size; z += GetZ_Stride())
  {
    const float dzudzn_sgz_data = dzudzn_sgz[z] * ScaledDivider;
    for (size_t y = GetY(); y < DeviceConstants.Y_Size; y += GetY_Stride())
    {
      for(size_t x = GetX(); x < DeviceConstants.X_Size; x += GetX_Stride())
      {
         const size_t i = z * DeviceConstants.SlabSize + y * DeviceConstants.X_Size + x;
         uz_sgz[i] = uz_sgz[i] * dzudzn_sgz_data;
      }//X
    }//Y
  }//Z
}// end of CUDACompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_z
//------------------------------------------------------------------------------

/**
 * Interface to CUDA kernel to Compute uz = dt./rho0_sgz .* ifft (FFT).
 * if rho0_sgz is scalar, nonuniform  non uniform grid, z component.
 * @param [in, out] uz_sgz       - output value of u
 * @param [in]      dt_rho0_szy - scalar value
 * @param [in]      dzudzn_sgz   - non-uniform mapping
 * @param [in]      FFT          - FFT matrix (data will be overwritten)
 */
void TCUDAImplementations::Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_z(TRealMatrix        & uz_sgz,
                                                                                const float          dt_rho0_sgz,
                                                                                const TRealMatrix  & dzudzn_sgz,
                                                                                TCUFFTComplexMatrix& FFT)
{
  // take the D iFFT
  FFT.Compute_FFT_3D_C2R(uz_sgz);

  CUDACompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_z<<<GetSolverGridSize3D(),
                                                             GetSolverBlockSize3D()>>>
                                                          (uz_sgz.GetRawDeviceData(),
                                                           dt_rho0_sgz,
                                                           dzudzn_sgz.GetRawDeviceData());
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_z
//------------------------------------------------------------------------------


/**
 *  kernel which compute part of the new velocity term - gradient
 *  of p represented by:
 *  bsxfun(\@times, ddx_k_shift_pos, kappa .* p_k).
 *
 * Complex numbers are passed as float2 structures.
 *
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
__global__ void CUDACompute_ddx_kappa_fft_p(float2       * FFT_X,
                                            float2       * FFT_Y,
                                            float2       * FFT_Z,
                                            const float  * kappa,
                                            const float2 * ddx,
                                            const float2 * ddy,
                                            const float2 * ddz)
{
  for (size_t z = GetZ(); z < DeviceConstants.Complex_Z_Size; z += GetZ_Stride())
  {
    // float 2 used for imaginary numbers .x  =re, .y = im
    const float2 ddz_el = ddz[z];
    for (size_t y = GetY(); y < DeviceConstants.Complex_Y_Size; y += GetY_Stride())
    {
      const float2 ddy_el = ddy[y];
      for(size_t x = GetX(); x < DeviceConstants.Complex_X_Size; x += GetX_Stride())
      {
        const size_t i = z * (DeviceConstants.Complex_Y_Size * DeviceConstants.Complex_X_Size) +
                         y * DeviceConstants.Complex_X_Size + x;

        const float2 ddx_el = ddx[x];
        // kappa ./ p_k
        const float kappa_el = kappa[i];

        float2 p_k_el    = FFT_X[i];
               p_k_el.x *= kappa_el;
               p_k_el.y *= kappa_el;

        float2 tmp_x;
        float2 tmp_y;
        float2 tmp_z;

        //bxfun(ddx...)
        tmp_x.x = p_k_el.x * ddx_el.x - p_k_el.y * ddx_el.y;
        tmp_x.y = p_k_el.x * ddx_el.y + p_k_el.y * ddx_el.x;

        tmp_y.x = p_k_el.x * ddy_el.x - p_k_el.y * ddy_el.y;
        tmp_y.y = p_k_el.x * ddy_el.y + p_k_el.y * ddy_el.x;

        tmp_z.x = p_k_el.x * ddz_el.x - p_k_el.y * ddz_el.y;
        tmp_z.y = p_k_el.x * ddz_el.y + p_k_el.y * ddz_el.x;

        FFT_X[i] = tmp_x;
        FFT_Y[i] = tmp_y;
        FFT_Z[i] = tmp_z;
      }// X
    }// Y
  }//X
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
void TCUDAImplementations::Compute_ddx_kappa_fft_p(TRealMatrix         & X_Matrix,
                                                   TCUFFTComplexMatrix & FFT_X,
                                                   TCUFFTComplexMatrix & FFT_Y,
                                                   TCUFFTComplexMatrix & FFT_Z,
                                                   const TRealMatrix   & kappa,
                                                   const TComplexMatrix& ddx,
                                                   const TComplexMatrix& ddy,
                                                   const TComplexMatrix& ddz)
{
  // Compute FFT of X
  FFT_X.Compute_FFT_3D_R2C(X_Matrix);

  CUDACompute_ddx_kappa_fft_p<<<GetSolverComplexGridSize3D(),
                                GetSolverBlockSize3D()>>>
                             (reinterpret_cast<float2 *>(FFT_X.GetRawDeviceData()),
                              reinterpret_cast<float2 *>( FFT_Y.GetRawDeviceData()),
                              reinterpret_cast<float2 *>( FFT_Z.GetRawDeviceData()),
                              kappa.GetRawDeviceData(),
                              reinterpret_cast<const float2 *>(ddx.GetRawDeviceData()),
                              reinterpret_cast<const float2 *>(ddy.GetRawDeviceData()),
                              reinterpret_cast<const float2 *>(ddz.GetRawDeviceData()));

  // check for errors
  gpuErrchk(cudaGetLastError());
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
__global__  void CUDACompute_duxyz_uniform(float2       * FFT_X,
                                           float2       * FFT_Y,
                                           float2       * FFT_Z,
                                           const float  * kappa,
                                           const float2 * ddx_neg,
                                           const float2 * ddy_neg,
                                           const float2 * ddz_neg)
{
  for (size_t z = GetZ(); z < DeviceConstants.Complex_Z_Size; z += GetZ_Stride())
  {
    // float 2 used for imaginary numbers .x  =re, .y = im
    const float2 ddz_neg_el = ddz_neg[z];
    for (size_t y = GetY(); y < DeviceConstants.Complex_Y_Size; y += GetY_Stride())
    {
      const float2 ddy_neg_el = ddy_neg[y];
      for(size_t x = GetX(); x < DeviceConstants.Complex_X_Size; x += GetX_Stride())
      {
        const size_t i = z * (DeviceConstants.Complex_Y_Size * DeviceConstants.Complex_X_Size) +
                         y * DeviceConstants.Complex_X_Size + x;

        const float2 ddx_neg_el = ddx_neg[x];
        const float kappa_el = kappa[i];

        float2 FFT_X_el = FFT_X[i];
        float2 FFT_Y_el = FFT_Y[i];
        float2 FFT_Z_el = FFT_Z[i];

        FFT_X_el.x *= kappa_el;
        FFT_X_el.y *= kappa_el;

        FFT_Y_el.x *= kappa_el;
        FFT_Y_el.y *= kappa_el;

        FFT_Z_el.x *= kappa_el;
        FFT_Z_el.y *= kappa_el;

        float2 tmp_x;
        float2 tmp_y;
        float2 tmp_z;

        tmp_x.x = (FFT_X_el.x * ddx_neg_el.x - FFT_X_el.y * ddx_neg_el.y) * DeviceConstants.Divider;
        tmp_x.y = (FFT_X_el.y * ddx_neg_el.x + FFT_X_el.x * ddx_neg_el.y) * DeviceConstants.Divider;;

        tmp_y.x = (FFT_Y_el.x * ddy_neg_el.x - FFT_Y_el.y * ddy_neg_el.y) * DeviceConstants.Divider;
        tmp_y.y = (FFT_Y_el.y * ddy_neg_el.x + FFT_Y_el.x * ddy_neg_el.y) * DeviceConstants.Divider;;

        tmp_z.x = (FFT_Z_el.x * ddz_neg_el.x - FFT_Z_el.y * ddz_neg_el.y) * DeviceConstants.Divider;
        tmp_z.y = (FFT_Z_el.y * ddz_neg_el.x + FFT_Z_el.x * ddz_neg_el.y) * DeviceConstants.Divider;;

        FFT_X[i] = tmp_x;
        FFT_Y[i] = tmp_y;
        FFT_Z[i] = tmp_z;
      } // x
    } // y
  } // z
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
void TCUDAImplementations::Compute_duxyz_uniform(TCUFFTComplexMatrix & FFT_X,
                                                 TCUFFTComplexMatrix & FFT_Y,
                                                 TCUFFTComplexMatrix & FFT_Z,
                                                 const TRealMatrix   & kappa,
                                                 const TComplexMatrix& ddx_k_shift_neg,
                                                 const TComplexMatrix& ddy_k_shift_neg,
                                                 const TComplexMatrix& ddz_k_shift_neg)
{
  CUDACompute_duxyz_uniform<<<GetSolverGridSize3D(),
                              GetSolverBlockSize3D()>>>
                          (reinterpret_cast<float2 *>(FFT_X.GetRawDeviceData()),
                           reinterpret_cast<float2 *>(FFT_Y.GetRawDeviceData()),
                           reinterpret_cast<float2 *>(FFT_Z.GetRawDeviceData()),
                           kappa.GetRawDeviceData(),
                           reinterpret_cast<const float2 *>(ddx_k_shift_neg.GetRawDeviceData()),
                           reinterpret_cast<const float2 *>(ddy_k_shift_neg.GetRawDeviceData()),
                           reinterpret_cast<const float2 *>(ddz_k_shift_neg.GetRawDeviceData()));

  // check for errors
  gpuErrchk(cudaGetLastError());
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
__global__  void CUDACompute_duxyz_non_linear(float      * duxdx,
                                              float      * duydy,
                                              float      * duzdz,
                                              const float* duxdxn,
                                              const float* duydyn,
                                              const float* duzdzn)
{
  for (size_t z = GetZ(); z < DeviceConstants.Z_Size; z += GetZ_Stride())
  {
    const float duzdzn_el = duzdzn[z];
    for (size_t y = GetY(); y < DeviceConstants.Y_Size; y += GetY_Stride())
    {
      const float dyudyn_el = duydyn[y];
      for(size_t x = GetX(); x < DeviceConstants.X_Size; x += GetX_Stride())
      {
        const size_t i = z * DeviceConstants.SlabSize + y * DeviceConstants.X_Size + x;

        duxdx[i] *= duxdxn[x];
        duydy[i] *= dyudyn_el;
        duzdz[i] *= duzdzn_el;
      } // x
    } // y
  } // z
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
void TCUDAImplementations::Compute_duxyz_non_uniform(TRealMatrix      & duxdx,
                                                     TRealMatrix      & duydy,
                                                     TRealMatrix      & duzdz,
                                                     const TRealMatrix& dxudxn,
                                                     const TRealMatrix& dyudyn,
                                                     const TRealMatrix& dzudzn)
{
  CUDACompute_duxyz_non_linear<<<GetSolverGridSize3D(),
                                 GetSolverBlockSize3D()>>>
                              (duxdx.GetRawDeviceData(),
                               duydy.GetRawDeviceData(),
                               duzdz.GetRawDeviceData(),
                               dxudxn.GetRawDeviceData(),
                               dyudyn.GetRawDeviceData(),
                               dzudzn.GetRawDeviceData());

  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Compute_duxyz_non_uniform
//------------------------------------------------------------------------------


/**
 * CUDA kernel to add initial pressure p0 into p, rhox, rhoy, rhoz.
 * c is scalar
 * @param [out] p       - pressure
 * @param [out] rhox
 * @param [out] rhoy
 * @param [out] rhoz
 * @param [in]  p0       - intial pressure
 * @param [in]  c2       - sound speed
 * @param [in]  c2_shift - scalar or vector?
 */
__global__ void CUDACalculate_p0_source_add_initial_pressure(float       * p,
                                                             float       * rhox,
                                                             float       * rhoy,
                                                             float       * rhoz,
                                                             const float * p0,
                                                             const float c2)
{
  const float Divider = 1.0f / (3.0f * c2);

  for (size_t i = GetX(); i < DeviceConstants.TotalElementCount; i += GetX_Stride())
  {
    float tmp = p[i] = p0[i];

    tmp = tmp * Divider;
    rhox[i] = tmp;
    rhoy[i] = tmp;
    rhoz[i] = tmp;
  }
}// end of CUDACalculate_p0_source_add_initial_pressure
//------------------------------------------------------------------------------



/**
 * CUDA kernel to add initial pressure p0 into p, rhox, rhoy, rhoz.
 * c is a matrix.
 * @param [out] p       - pressure
 * @param [out] rhox
 * @param [out] rhoy
 * @param [out] rhoz
 * @param [in]  p0       - intial pressure
 * @param [in]  c2       - sound speed
 * @param [in]  c2_shift - scalar or vector?
 */
__global__ void CUDACalculate_p0_source_add_initial_pressure(float       * p,
                                                             float       * rhox,
                                                             float       * rhoy,
                                                             float       * rhoz,
                                                             const float * p0,
                                                             const float * c2)
{
  for (size_t i = GetX(); i < DeviceConstants.TotalElementCount; i += GetX_Stride())
  {
    float tmp = p[i] = p0[i];

    tmp = tmp / (3.0f * c2[i]);
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
 * @param [in] p0       - intial pressure
 * @param [in] c2       - sound speed
 * @param [in] c2_shift - scalar or vector?
 */
void TCUDAImplementations::Calculate_p0_source_add_initial_pressure(TRealMatrix      & p,
                                                                    TRealMatrix      & rhox,
                                                                    TRealMatrix      & rhoy,
                                                                    TRealMatrix      & rhoz,
                                                                    const TRealMatrix& p0,
                                                                    const float      * c2,
                                                                    const size_t       c2_shift)
{
  if (c2_shift == 0)
  {
    CUDACalculate_p0_source_add_initial_pressure<<<GetSolverGridSize1D(),
                                                   GetSolverBlockSize1D()>>>
                                                (p.GetRawDeviceData(),
                                                 rhox.GetRawDeviceData(),
                                                 rhoy.GetRawDeviceData(),
                                                 rhoz.GetRawDeviceData(),
                                                 p0.GetRawDeviceData(),
                                                 c2[0]);
  }

  if (c2_shift == 1)
  {
    CUDACalculate_p0_source_add_initial_pressure<<<GetSolverGridSize1D(),
                                                   GetSolverBlockSize1D()>>>
                                                (p.GetRawDeviceData(),
                                                 rhox.GetRawDeviceData(),
                                                 rhoy.GetRawDeviceData(),
                                                 rhoz.GetRawDeviceData(),
                                                 p0.GetRawDeviceData(),
                                                 c2);
  }
  // check for errors
  gpuErrchk(cudaGetLastError());
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
 * @param [in]  dt_rho0 - dt * rho0
 * @param [in]  rho0    - dt^2
 */
__global__ void CUDACompute_rhoxyz_nonlinear_homogeneous(float      * rhox,
                                                         float      * rhoy,
                                                         float      * rhoz,
                                                         const float* pml_x,
                                                         const float* pml_y,
                                                         const float* pml_z,
                                                         const float* duxdx,
                                                         const float* duydy,
                                                         const float* duzdz,
                                                         const float  dt_rho0,
                                                         const float  dt2)
{
  for (size_t z = GetZ(); z < DeviceConstants.Z_Size; z += GetZ_Stride())
  {
    const float pml_z_el = pml_z[z];
    for (size_t y = GetY(); y < DeviceConstants.Y_Size; y += GetY_Stride())
    {
      const float pml_y_el = pml_y[y];
      for(size_t x = GetX(); x < DeviceConstants.X_Size; x += GetX_Stride())
      {
        const size_t i = z * DeviceConstants.SlabSize + y * DeviceConstants.X_Size + x;

        const float pml_x_el = pml_x[x];

        const float dux = duxdx[i];
        const float duy = duydy[i];
        const float duz = duzdz[i];

        rhox[i] = pml_x_el * (((pml_x_el * rhox[i]) - (dt_rho0 * dux)) / (1.0f + (dt2 * dux)));
        rhoy[i] = pml_y_el * (((pml_y_el * rhoy[i]) - (dt_rho0 * duy)) / (1.0f + (dt2 * duy)));
        rhoz[i] = pml_z_el * (((pml_z_el * rhoz[i]) - (dt_rho0 * duz)) / (1.0f + (dt2 * duz)));
      }// X
    }// Y
  }// Z
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
 * @param [in]  dt    - time step
 * @param [in]  rho0  - initial density (scalar here)
 */
void TCUDAImplementations::Compute_rhoxyz_nonlinear_homogeneous(TRealMatrix      & rhox,
                                                                TRealMatrix      & rhoy,
                                                                TRealMatrix      & rhoz,
                                                                const TRealMatrix& pml_x,
                                                                const TRealMatrix& pml_y,
                                                                const TRealMatrix& pml_z,
                                                                const TRealMatrix& duxdx,
                                                                const TRealMatrix& duydy,
                                                                const TRealMatrix& duzdz,
                                                                const float        dt,
                                                                const float        rho0)
{
  const float dt2 = 2.0f * dt;
  const float dt_rho0 = rho0 * dt;

  CUDACompute_rhoxyz_nonlinear_homogeneous<<<GetSolverGridSize3D(),
                                             GetSolverBlockSize3D()>>>
                                          (rhox.GetRawDeviceData(),
                                           rhoy.GetRawDeviceData(),
                                           rhoz.GetRawDeviceData(),
                                           pml_x.GetRawDeviceData(),
                                           pml_y.GetRawDeviceData(),
                                           pml_z.GetRawDeviceData(),
                                           duxdx.GetRawDeviceData(),
                                           duydy.GetRawDeviceData(),
                                           duzdz.GetRawDeviceData(),
                                           dt_rho0,
                                           dt2);
  // check for errors
  gpuErrchk(cudaGetLastError());
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
 * @param [in]  dt    - time step
 * @param [in]  rho0  - initial density (matrix here)
 */
__global__ void CUDACompute_rhoxyz_nonlinear_heterogeneous(float      * rhox,
                                                           float      * rhoy,
                                                           float      * rhoz,
                                                           const float* pml_x,
                                                           const float* pml_y,
                                                           const float* pml_z,
                                                           const float* duxdx,
                                                           const float* duydy,
                                                           const float* duzdz,
                                                           const float  dt,
                                                           const float* rho0)
{
  const float dt2 = 2.0f * dt;

  for (size_t z = GetZ(); z < DeviceConstants.Z_Size; z += GetZ_Stride())
  {
    const float pml_z_el = pml_z[z];
    for (size_t y = GetY(); y < DeviceConstants.Y_Size; y += GetY_Stride())
    {
      const float pml_y_el = pml_y[y];
      for(size_t x = GetX(); x < DeviceConstants.X_Size; x += GetX_Stride())
      {
        const size_t i = z * DeviceConstants.SlabSize + y * DeviceConstants.X_Size + x;

        const float pml_x_el = pml_x[x];
        const float dt_rho0 = dt * rho0[i];

        const float dux = duxdx[i];
        const float duy = duydy[i];
        const float duz = duzdz[i];

        rhox[i] = pml_x_el * (((pml_x_el * rhox[i]) - (dt_rho0 * dux)) / (1.0f + (dt2 * dux)));
        rhoy[i] = pml_y_el * (((pml_y_el * rhoy[i]) - (dt_rho0 * duy)) / (1.0f + (dt2 * duy)));
        rhoz[i] = pml_z_el * (((pml_z_el * rhoz[i]) - (dt_rho0 * duz)) / (1.0f + (dt2 * duz)));
      }// X
    }// Y
  }//Z
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
 * @param [in]  dt    - time step
 * @param [in]  rho0  - initial density (matrix here)
 */
void TCUDAImplementations::Compute_rhoxyz_nonlinear_heterogeneous(TRealMatrix&       rhox,
                                                                  TRealMatrix&       rhoy,
                                                                  TRealMatrix&       rhoz,
                                                                  const TRealMatrix& pml_x,
                                                                  const TRealMatrix& pml_y,
                                                                  const TRealMatrix& pml_z,
                                                                  const TRealMatrix& duxdx,
                                                                  const TRealMatrix& duydy,
                                                                  const TRealMatrix& duzdz,
                                                                  const float        dt,
                                                                  const TRealMatrix& rho0)
{
  CUDACompute_rhoxyz_nonlinear_heterogeneous<<<GetSolverGridSize3D(),
                                               GetSolverBlockSize3D() >>>
                                              (rhox.GetRawDeviceData(),
                                               rhoy.GetRawDeviceData(),
                                               rhoz.GetRawDeviceData(),
                                               pml_x.GetRawDeviceData(),
                                               pml_y.GetRawDeviceData(),
                                               pml_z.GetRawDeviceData(),
                                               duxdx.GetRawDeviceData(),
                                               duydy.GetRawDeviceData(),
                                               duzdz.GetRawDeviceData(),
                                               dt,
                                               rho0.GetRawDeviceData());

  // check for errors
  gpuErrchk(cudaGetLastError());
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
 * @param [in]  dt_rho0 - dt * rho0

 */
__global__ void CUDACompute_rhoxyz_linear_homogeneous(float      * rhox,
                                                      float      * rhoy,
                                                      float      * rhoz,
                                                      const float* pml_x,
                                                      const float* pml_y,
                                                      const float* pml_z,
                                                      const float* duxdx,
                                                      const float* duydy,
                                                      const float* duzdz,
                                                      const float  dt_rho0)
{
  for (size_t z = GetZ(); z < DeviceConstants.Z_Size; z += GetZ_Stride())
  {
    const float pml_z_el = pml_z[z];
    for (size_t y = GetY(); y < DeviceConstants.Y_Size; y += GetY_Stride())
    {
      const float pml_y_el = pml_y[y];
      for(size_t x = GetX(); x < DeviceConstants.X_Size; x += GetX_Stride())
      {
        const size_t i = z * DeviceConstants.SlabSize + y * DeviceConstants.X_Size + x;
        const float pml_x_el = pml_x[x];

        rhox[i] = pml_x_el * (((pml_x_el * rhox[i]) - (dt_rho0 * duxdx[i])));
        rhoy[i] = pml_y_el * (((pml_y_el * rhoy[i]) - (dt_rho0 * duydy[i])));
        rhoz[i] = pml_z_el * (((pml_z_el * rhoz[i]) - (dt_rho0 * duzdz[i])));
      }//X
    }//Y
  }//Z
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
 * @param [in]  dt    - time step
 * @param [in]  rho0  - initial density (scalar here)
 */
void TCUDAImplementations::Compute_rhoxyz_linear_homogeneous(TRealMatrix      & rhox,
                                                             TRealMatrix      & rhoy,
                                                             TRealMatrix      & rhoz,
                                                             const TRealMatrix& pml_x,
                                                             const TRealMatrix& pml_y,
                                                             const TRealMatrix& pml_z,
                                                             const TRealMatrix& duxdx,
                                                             const TRealMatrix& duydy,
                                                             const TRealMatrix& duzdz,
                                                             const float dt,
                                                             const float rho0)
{
  const float dt_rho0 = rho0 * dt;

  CUDACompute_rhoxyz_linear_homogeneous<<<GetSolverGridSize3D(),
                                          GetSolverBlockSize3D() >>>
                                      (rhox.GetRawDeviceData(),
                                       rhoy.GetRawDeviceData(),
                                       rhoz.GetRawDeviceData(),
                                       pml_x.GetRawDeviceData(),
                                       pml_y.GetRawDeviceData(),
                                       pml_z.GetRawDeviceData(),
                                       duxdx.GetRawDeviceData(),
                                       duydy.GetRawDeviceData(),
                                       duzdz.GetRawDeviceData(),
                                       dt_rho0);
  // check for errors
  gpuErrchk(cudaGetLastError());
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
 * @param [in]  dt    - time step
 * @param [in]  rho0  - initial density (matrix here)
 */
__global__ void CUDACompute_rhoxyz_linear_heterogeneous(float       * rhox,
                                                        float       * rhoy,
                                                        float       * rhoz,
                                                        const float * pml_x,
                                                        const float * pml_y,
                                                        const float * pml_z,
                                                        const float * duxdx,
                                                        const float * duydy,
                                                        const float * duzdz,
                                                        const float   dt,
                                                        const float * rho0)
{
  for (size_t z = GetZ(); z < DeviceConstants.Z_Size; z += GetZ_Stride())
  {
    const float pml_z_el = pml_z[z];
    for (size_t y = GetY(); y < DeviceConstants.Y_Size; y += GetY_Stride())
    {
      const float pml_y_el = pml_y[y];
      for(size_t x = GetX(); x < DeviceConstants.X_Size; x += GetX_Stride())
      {
        const size_t i = z * DeviceConstants.SlabSize + y * DeviceConstants.X_Size + x;
        const float dt_rho0  = dt * rho0[i];
        const float pml_x_el = pml_x[x];

        rhox[i] = pml_x_el * (((pml_x_el * rhox[i]) - (dt_rho0 * duxdx[i])));
        rhoy[i] = pml_y_el * (((pml_y_el * rhoy[i]) - (dt_rho0 * duydy[i])));
        rhoz[i] = pml_z_el * (((pml_z_el * rhoz[i]) - (dt_rho0 * duzdz[i])));
      } //X
    }// Y
  }// Z
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
 * @param [in]  dt    - time step
 * @param [in]  rho0  - initial density (matrix here)
 */
void TCUDAImplementations::Compute_rhoxyz_linear_heterogeneous(TRealMatrix      & rhox,
                                                               TRealMatrix      & rhoy,
                                                               TRealMatrix      & rhoz,
                                                               const TRealMatrix& pml_x,
                                                               const TRealMatrix& pml_y,
                                                               const TRealMatrix& pml_z,
                                                               const TRealMatrix& duxdx,
                                                               const TRealMatrix& duydy,
                                                               const TRealMatrix& duzdz,
                                                               const float        dt,
                                                               const TRealMatrix& rho0)
{
  CUDACompute_rhoxyz_linear_heterogeneous<<<GetSolverGridSize3D(),
                                            GetSolverBlockSize3D()>>>
                                         (rhox.GetRawDeviceData(),
                                          rhoy.GetRawDeviceData(),
                                          rhoz.GetRawDeviceData(),
                                          pml_x.GetRawDeviceData(),
                                          pml_y.GetRawDeviceData(),
                                          pml_z.GetRawDeviceData(),
                                          duxdx.GetRawDeviceData(),
                                          duydy.GetRawDeviceData(),
                                          duzdz.GetRawDeviceData(),
                                          dt,
                                          rho0.GetRawDeviceData());

  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Compute_rhoxyz_linear_heterogeneous
//------------------------------------------------------------------------------


/**
 *
 * CUDA kernel which calculates three temporary sums in the new pressure formula \n
 * non-linear absorbing case.
 *
 * @param [out] rho_sum      - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [out] BonA_sum     - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [out] du_sum       - rho0* (duxdx + duydy + duzdz)
 * @param [in]  duxdx        - gradient of velocity
 * @param [in]  duydy
 * @param [in]  duzdz
 * @param [in]  BonA_scalar  - scalar value for BonA
 * @param [in]  BonA_matrix  - heterogeneous value for BonA
 * @param [in]  BonA_shift   - scalar or matrix
 * @param [in]  rho0_scalar  - scalar value for rho0
 * @param [in]  rho0_matrix  - heterogeneous value for rho0
 * @param [in]  rho0_shift   - scalar or matrix
 *
 * @todo revise parameter names, and put scalars to constant memory
 */
__global__ void CUDACalculate_SumRho_BonA_SumDu(float       * rho_sum,
                                                float       * BonA_sum,
                                                float       * du_sum,
                                                const float * rhox,
                                                const float * rhoy,
                                                const float * rhoz,
                                                const float * duxdx,
                                                const float * duydy,
                                                const float * duzdz,
                                                const float   BonA_scalar,
                                                const float * BonA_matrix,
                                                const size_t  BonA_shift,
                                                const float   rho0_scalar,
                                                const float * rho0_matrix,
                                                const size_t  rho0_shift)
{
  for (size_t i = GetX(); i < DeviceConstants.TotalElementCount; i += GetX_Stride())
  {
    const float BonA = (BonA_shift == 0) ? BonA_scalar : BonA_matrix[i];
    const float rho0 = (rho0_shift == 0) ? rho0_scalar : rho0_matrix[i];

    const float rho_xyz_el = rhox[i] + rhoy[i] + rhoz[i];

    rho_sum[i]  = rho_xyz_el;
    BonA_sum[i] = ((BonA * (rho_xyz_el * rho_xyz_el)) / (2.0f * rho0)) + rho_xyz_el;
    du_sum[i]   = rho0 * (duxdx[i] + duydy[i] + duzdz[i]);
    }
}// end of CUDACalculate_SumRho_BonA_SumDu
//--------------------------------------------------------------------------

/**
 *
 * Interface to kernel which calculates three temporary sums in the new pressure formula \n
 * non-linear absorbing case.
 *
 * @param [out] rho_sum      - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [out] BonA_sum     - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [out] du_sum       - rho0* (duxdx + duydy + duzdz)
 * @param [in]  duxdx        - gradient of velocity
 * @param [in]  duydy
 * @param [in]  duzdz
 * @param [in]  BonA_scalar  - scalar value for BonA
 * @param [in]  BonA_matrix  - heterogeneous value for BonA
 * @param [in]  BonA_shift   - scalar or matrix
 * @param [in]  rho0_scalar  - scalar value for rho0
 * @param [in]  rho0_matrix  - heterogeneous value for rho0
 * @param [in]  rho0_shift   - scalar or matrix
 *
 * @todo revise parameter names, and put scalars to constant memory
 */
void TCUDAImplementations::Calculate_SumRho_BonA_SumDu(TRealMatrix      & rho_sum,
                                                       TRealMatrix      & BonA_sum,
                                                       TRealMatrix      & du_sum,
                                                       const TRealMatrix& rhox,
                                                       const TRealMatrix& rhoy,
                                                       const TRealMatrix& rhoz,
                                                       const TRealMatrix& duxdx,
                                                       const TRealMatrix& duydy,
                                                       const TRealMatrix& duzdz,
                                                       const float        BonA_scalar,
                                                       const float*       BonA_matrix,
                                                       const size_t       BonA_shift,
                                                       const float        rho0_scalar,
                                                       const float*       rho0_matrix,
                                                       const size_t       rho0_shift)
{
  CUDACalculate_SumRho_BonA_SumDu<<<GetSolverGridSize1D(),
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
                                 BonA_scalar,
                                 BonA_matrix,
                                 BonA_shift,
                                 rho0_scalar,
                                 rho0_matrix,
                                 rho0_shift);

  // check for errors
  gpuErrchk(cudaGetLastError());
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
__global__ void CUDACompute_Absorb_nabla1_2(float2     * FFT_1,
                                            float2     * FFT_2,
                                            const float* nabla1,
                                            const float* nabla2)
{
  for(size_t i = GetX(); i < DeviceConstants.ComplexTotalElementCount; i += GetX_Stride())
  {
    const float nabla_data1 = nabla1[i];
    const float nabla_data2 = nabla2[i];

    float2 FFT_1_el = FFT_1[i];
    float2 FFT_2_el = FFT_2[i];

    FFT_1_el.x *= nabla_data1;
    FFT_1_el.y *= nabla_data1;

    FFT_2_el.x *= nabla_data2;
    FFT_2_el.y *= nabla_data2;

    FFT_1[i] = FFT_1_el;
    FFT_2[i] = FFT_2_el;
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
void TCUDAImplementations::Compute_Absorb_nabla1_2(TCUFFTComplexMatrix& FFT_1,
                                                   TCUFFTComplexMatrix& FFT_2,
                                                   const TRealMatrix  & absorb_nabla1,
                                                   const TRealMatrix  & absorb_nabla2)
{
  CUDACompute_Absorb_nabla1_2<<<GetSolverGridSize1D(),
                                GetSolverBlockSize1D()>>>
                            (reinterpret_cast<float2 *> (FFT_1.GetRawDeviceData()),
                             reinterpret_cast<float2 *> (FFT_2.GetRawDeviceData()),
                             absorb_nabla1.GetRawDeviceData(),
                             absorb_nabla2.GetRawDeviceData());

  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Compute_Absorb_nabla1_2
//------------------------------------------------------------------------------


/**
 * CUDA Sum sub-terms to calculate new pressure, non-linear case.
 * @@todo needs revision
 * @param [out] p           - new value of pressure
 * @param [in] BonA_temp    - rho0 * (duxdx + duydy + duzdz)
 * @param [in] c2_scalar
 * @param [in] c2_matrix
 * @param [in] c2_shift
 * @param [in] Absorb_tau
 * @param [in] tau_scalar
 * @param [in] tau_matrix
 * @param [in] Absorb_eta   - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [in] eta_scalar
 * @param [in] eta_matrix
 * @param [in]tau_eta_shift
 */
__global__ void CUDASum_Subterms_nonlinear(float       * p,
                                           const float * BonA_temp,
                                           const float   c2_scalar,
                                           const float * c2_matrix,
                                           const size_t  c2_shift,
                                           const float * Absorb_tau,
                                           const float   tau_scalar,
                                           const float * tau_matrix,
                                           const float * Absorb_eta,
                                           const float   eta_scalar,
                                           const float * eta_matrix,
                                           const size_t  tau_eta_shift)
{
  for(size_t i = GetX(); i < DeviceConstants.TotalElementCount; i += GetX_Stride())
  {
    const float c2  = (c2_shift == 0)      ? c2_scalar  : c2_matrix[i];
    const float tau = (tau_eta_shift == 0) ? tau_scalar : tau_matrix[i];
    const float eta = (tau_eta_shift == 0) ? eta_scalar : eta_matrix[i];

    p[i] = c2 * (BonA_temp[i] + (DeviceConstants.Divider *
                ((Absorb_tau[i] * tau) - (Absorb_eta[i] * eta))));
  }
}// end of CUDASum_Subterms_nonlinear
//------------------------------------------------------------------------------


/**
 * Interface to CUDA Sum sub-terms to calculate new pressure, non-linear case.
 * @param [in,out] p        - new value of pressure
 * @param [in] BonA_temp    - rho0 * (duxdx + duydy + duzdz)
 * @param [in] c2_scalar
 * @param [in] c2_matrix
 * @param [in] c2_shift
 * @param [in] Absorb_tau
 * @param [in] tau_scalar
 * @param [in] tau_matrix
 * @param [in] Absorb_eta   - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [in] eta_scalar
 * @param [in] eta_matrix
 * @param [in]tau_eta_shift
 */
void TCUDAImplementations::Sum_Subterms_nonlinear(TRealMatrix      & p,
                                                  const TRealMatrix& BonA_temp,
                                                  const float        c2_scalar,
                                                  const float      * c2_matrix,
                                                  const size_t       c2_shift,
                                                  const float      * Absorb_tau,
                                                  const float        tau_scalar,
                                                  const float      * tau_matrix,
                                                  const float      * Absorb_eta,
                                                  const float        eta_scalar,
                                                  const float      * eta_matrix,
                                                  const size_t       tau_eta_shift)
{
  CUDASum_Subterms_nonlinear<<<GetSolverGridSize1D(),
                               GetSolverBlockSize1D()>>>
                            (p.GetRawDeviceData(),
                             BonA_temp.GetRawDeviceData(),
                             c2_scalar,
                             c2_matrix,
                             c2_shift,
                             Absorb_tau,
                             tau_scalar,
                             tau_matrix,
                             Absorb_eta,
                             eta_scalar,
                             eta_matrix,
                             tau_eta_shift);

  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Sum_Subterms_nonlinear
//------------------------------------------------------------------------------


/**
 * CUDA kernel that sums sub-terms to calculate new pressure, linear case.
 * @param [out] p              - new value of p
 * @param [in] Absorb_tau_temp - sub-term with absorb_tau
 * @param [in] Absorb_eta_temp - sub-term with absorb_eta
 * @param [in] Sum_rhoxyz      - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [in] c2_scalar
 * @param [in] c2_matrix
 * @param [in] c2_shift
 * @param [in] tau_scalar
 * @param [in] tau_matrix
 * @param [in] eta_scalar
 * @param [in] eta_matrix
 * @param [in] tau_eta_shift
 */
__global__ void CUDASum_Subterms_linear(float       * p,
                                        const float * Absorb_tau_temp,
                                        const float * Absorb_eta_temp,
                                        const float * Sum_rhoxyz,
                                        const float   c2_scalar,
                                        const float * c2_matrix,
                                        const size_t  c2_shift,
                                        const float   tau_scalar,
                                        const float * tau_matrix,
                                        const float   eta_scalar,
                                        const float * eta_matrix,
                                        const size_t  tau_eta_shift)
{
  for(size_t i = GetX(); i < DeviceConstants.TotalElementCount; i += GetX_Stride())
  {
    const float c2  = (c2_shift == 0)      ? c2_scalar  : c2_matrix[i];
    const float tau = (tau_eta_shift == 0) ? tau_scalar : tau_matrix[i];
    const float eta = (tau_eta_shift == 0) ? eta_scalar : eta_matrix[i];

    p[i] = c2 * (Sum_rhoxyz[i] + (DeviceConstants.Divider *
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
 * @param [in] c2_scalar
 * @param [in] c2_matrix
 * @param [in] c2_shift
 * @param [in] tau_scalar
 * @param [in] tau_matrix
 * @param [in] eta_scalar
 * @param [in] eta_matrix
 * @param [in] tau_eta_shift
 */
void TCUDAImplementations::Sum_Subterms_linear(TRealMatrix      & p,
                                               const TRealMatrix& Absorb_tau_temp,
                                               const TRealMatrix& Absorb_eta_temp,
                                               const TRealMatrix& Sum_rhoxyz,
                                               const float        c2_scalar,
                                               const float      * c2_matrix,
                                               const size_t       c2_shift,
                                               const float        tau_scalar,
                                               const float      * tau_matrix,
                                               const float        eta_scalar,
                                               const float      * eta_matrix,
                                               const size_t       tau_eta_shift)
{
  CUDASum_Subterms_linear<<<GetSolverGridSize1D(),
                            GetSolverBlockSize1D() >>>
                        (p.GetRawDeviceData(),
                         Absorb_tau_temp.GetRawDeviceData(),
                         Absorb_eta_temp.GetRawDeviceData(),
                         Sum_rhoxyz.GetRawDeviceData(),
                         c2_scalar,
                         c2_matrix,
                         c2_shift,
                         tau_scalar,
                         tau_matrix,
                         eta_scalar,
                         eta_matrix,
                         tau_eta_shift);

  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Sum_Subterms_linear
//------------------------------------------------------------------------------


/**
 * CUDA kernel that sums sub-terms for new p, non-linear lossless case.
 * @param [out] p           - new value of pressure
 * @param [in]  rhox
 * @param [in]  rhoy
 * @param [in]  rhoz
 * @param [in]  c2_scalar
 * @param [in]  c2_matrix
 * @param [in]  c2_shift
 * @param [in]  BonA_scalar
 * @param [in]  BonA_matrix
 * @param [in]  BonA_shift
 * @param [in]  rho0_scalar
 * @param [in]  rho0_matrix
 * @param [in]  rho0_shift
 */
__global__ void CUDASum_new_p_nonlinear_lossless(float       * p,
                                                 const float * rhox,
                                                 const float * rhoy,
                                                 const float * rhoz,
                                                 const float   c2_scalar,
                                                 const float * c2_matrix,
                                                 const size_t  c2_shift,
                                                 const float   BonA_scalar,
                                                 const float * BonA_matrix,
                                                 const size_t  BonA_shift,
                                                 const float   rho0_scalar,
                                                 const float * rho0_matrix,
                                                 const size_t  rho0_shift)
{
  for(size_t i = GetX(); i < DeviceConstants.TotalElementCount; i += GetX_Stride())
  {
    const float c2   = (c2_shift   == 0) ? c2_scalar   : c2_matrix[i];
    const float BonA = (BonA_shift == 0) ? BonA_scalar : BonA_matrix[i];
    const float rho0 = (rho0_shift == 0) ? rho0_scalar : rho0_matrix[i];

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
 * @param [in]  c2_scalar
 * @param [in]  c2_matrix
 * @param [in]  c2_shift
 * @param [in]  BonA_scalar
 * @param [in]  BonA_matrix
 * @param [in]  BonA_shift
 * @param [in]  rho0_scalar
 * @param [in]  rho0_matrix
 * @param [in]  rho0_shift
 */
void TCUDAImplementations::Sum_new_p_nonlinear_lossless(TRealMatrix      & p,
                                                        const TRealMatrix& rhox,
                                                        const TRealMatrix& rhoy,
                                                        const TRealMatrix& rhoz,
                                                        const float        c2_scalar,
                                                        const float      * c2_matrix,
                                                        const size_t       c2_shift,
                                                        const float        BonA_scalar,
                                                        const float      * BonA_matrix,
                                                        const size_t       BonA_shift,
                                                        const float        rho0_scalar,
                                                        const float      * rho0_matrix,
                                                        const size_t       rho0_shift)
{
  CUDASum_new_p_nonlinear_lossless<<<GetSolverGridSize1D(),
                                     GetSolverBlockSize1D()>>>
                                  (p.GetRawDeviceData(),
                                   rhox.GetRawDeviceData(),
                                   rhoy.GetRawDeviceData(),
                                   rhoz.GetRawDeviceData(),
                                   c2_scalar,
                                   c2_matrix,
                                   c2_shift,
                                   BonA_scalar,
                                   BonA_matrix,
                                   BonA_shift,
                                   rho0_scalar,
                                   rho0_matrix,
                                   rho0_shift);

  // check for errors
  gpuErrchk(cudaGetLastError());
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
 * @param [in]  rho0_scalar
 * @param [in]  rho0_matrix
 * @param [in]  rho0_shift
 */
__global__ void CUDACalculate_SumRho_SumRhoDu(float      * Sum_rhoxyz,
                                              float      * Sum_rho0_du,
                                              const float* rhox,
                                              const float* rhoy,
                                              const float* rhoz,
                                              const float* dux,
                                              const float* duy,
                                              const float* duz,
                                              const float  rho0_scalar,
                                              const float* rho0_matrix,
                                              const size_t rho0_shift)
{
  for(size_t i = GetX(); i < DeviceConstants.TotalElementCount; i += GetX_Stride())
  {
    const float rho0 = (rho0_shift == 0) ? rho0_scalar : rho0_matrix[i];

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
 * @param [in]  rho0_scalar
 * @param [in]  rho0_matrix
 * @param [in]  rho0_shift
 */
void TCUDAImplementations::Calculate_SumRho_SumRhoDu(TRealMatrix      & Sum_rhoxyz,
                                                     TRealMatrix      & Sum_rho0_du,
                                                     const TRealMatrix& rhox,
                                                     const TRealMatrix& rhoy,
                                                     const TRealMatrix& rhoz,
                                                     const TRealMatrix& duxdx,
                                                     const TRealMatrix& duydy,
                                                     const TRealMatrix& duzdz,
                                                     const float        rho0_scalar,
                                                     const float      * rho0_matrix,
                                                     const size_t       rho0_shift)
{
   CUDACalculate_SumRho_SumRhoDu<<<GetSolverGridSize1D(),
                                   GetSolverBlockSize1D()>>>
                                (Sum_rhoxyz.GetRawDeviceData(),
                                 Sum_rho0_du.GetRawDeviceData(),
                                 rhox.GetRawDeviceData(),
                                 rhoy.GetRawDeviceData(),
                                 rhoz.GetRawDeviceData(),
                                 duxdx.GetRawDeviceData(),
                                 duydy.GetRawDeviceData(),
                                 duzdz.GetRawDeviceData(),
                                 rho0_scalar,
                                 rho0_matrix,
                                 rho0_shift);
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Calculate_SumRho_SumRhoDu
//------------------------------------------------------------------------------

/**
 * CUDA kernel that sums sub-terms for new p, linear lossless case.
 * @param [out] p
 * @param [in]  rhox
 * @param [in]  rhoy
 * @param [in]  rhoz
 * @param [in]  c2_scalar
 * @param [in]  c2_matrix
 * @param [in]  c2_shift
 */
__global__ void CUDASum_new_p_linear_lossless(float       * p,
                                              const float * rhox,
                                              const float * rhoy,
                                              const float * rhoz,
                                              const float   c2_scalar,
                                              const float * c2_matrix,
                                              const size_t  c2_shift)
{
  for(size_t i = GetX(); i < DeviceConstants.TotalElementCount; i += GetX_Stride())
  {
    const float c2 = (c2_shift == 0) ? c2_scalar : c2_matrix[i];
    p[i] = c2 * (rhox[i] + rhoy[i] + rhoz[i]);
  }
}// end of

/**
 * Interface to kernel that sums sub-terms for new p, linear lossless case.
 * @param [out] p
 * @param [in]  rhox
 * @param [in]  rhoy
 * @param [in]  rhoz
 * @param [in]  c2_scalar
 * @param [in]  c2_matrix
 * @param [in]  c2_shift
 */
void TCUDAImplementations::Sum_new_p_linear_lossless(TRealMatrix      & p,
                                                     const TRealMatrix& rhox,
                                                     const TRealMatrix& rhoy,
                                                     const TRealMatrix& rhoz,
                                                     const float        c2_scalar,
                                                     const float      * c2_matrix,
                                                     const size_t       c2_shift)
{
  CUDASum_new_p_linear_lossless<<<GetSolverGridSize1D(),
                                  GetSolverBlockSize1D()>>>
                               (p.GetRawDeviceData(),
                                rhox.GetRawDeviceData(),
                                rhoy.GetRawDeviceData(),
                                rhoz.GetRawDeviceData(),
                                c2_scalar,
                                c2_matrix,
                                c2_shift);
  // check for errors
  gpuErrchk(cudaGetLastError());
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
__global__ void cudaTrasnposeReal3DMatrixXYSquare(float       * OutputData,
                                                  const float * InputData,
                                                  const dim3    Dimensions)
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
__global__ void cudaTrasnposeReal3DMatrixXYRect(float       * OutputData,
                                                const float * InputData,
                                                const dim3    Dimensions)
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
void TCUDAImplementations::TrasposeReal3DMatrixXY(float       * OutputMatrixData,
                                                  const float * InputMatrixData,
                                                  const dim3  & DimSizes)
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
  gpuErrchk(cudaGetLastError());
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
__global__ void cudaTrasnposeReal3DMatrixXZSquare(float       * OutputData,
                                                  const float * InputData,
                                                  const dim3    Dimensions)
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
__global__ void cudaTrasnposeReal3DMatrixXZRect(float       * OutputData,
                                                const float * InputData,
                                                const dim3    Dimensions)
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
void TCUDAImplementations::TrasposeReal3DMatrixXZ(float       * OutputMatrixData,
                                                  const float * InputMatrixData,
                                                  const dim3  & DimSizes)
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
  gpuErrchk(cudaGetLastError());
}// end of TrasposeReal3DMatrixXZ
//------------------------------------------------------------------------------


/**
 * CUDA kernel to compute velocity shift in the X direction.
 * @param [in, out] FFT_shift_temp
 * @param [in]       x_shift_neg_r
 */
__global__ void CUDAComputeVelocityShiftInX(float2       * FFT_shift_temp,
                                            const float2 * x_shift_neg_r)
{
  // the size of the matrix is [Z, Y, X/2 + 1]
  for (size_t z = GetZ(); z < DeviceConstants.Complex_Z_Size; z += GetZ_Stride())
  {
    for (size_t y = GetY(); y < DeviceConstants.Complex_Y_Size; y += GetY_Stride())
    {
      for (size_t x = GetX(); x < DeviceConstants.Complex_X_Size; x += GetX_Stride())
      {
        const size_t i = (z * DeviceConstants.Complex_Y_Size  + y) * DeviceConstants.Complex_X_Size  + x;

        float2 temp;

        temp.x = ((FFT_shift_temp[i].x * x_shift_neg_r[x].x) -
                  (FFT_shift_temp[i].y * x_shift_neg_r[x].y)
                 ) * DeviceConstants.DividerX;


        temp.y = ((FFT_shift_temp[i].y * x_shift_neg_r[x].x) +
                  (FFT_shift_temp[i].x * x_shift_neg_r[x].y)
                 ) * DeviceConstants.DividerX;

        FFT_shift_temp[i] = temp;
      } // x
    } // y
  }// z
}// end of CUDAComputeVelocityShiftInX
//------------------------------------------------------------------------------


/**
 * Compute the velocity shift in Fourier space over the X axis.
 * This kernel work with the original space.
 * @param [in,out] FFT_shift_temp
 * @param [in]     x_shift_neg
 */
void TCUDAImplementations::ComputeVelocityShiftInX(TCUFFTComplexMatrix  & FFT_shift_temp,
                                                   const TComplexMatrix & x_shift_neg_r)
{
  CUDAComputeVelocityShiftInX<<<GetSolverGridSize3D(),
                                GetSolverBlockSize3D()>>>
                             (reinterpret_cast<float2 *>       (FFT_shift_temp.GetRawDeviceData()),
                              reinterpret_cast<const float2 *> (x_shift_neg_r.GetRawDeviceData()));
  // check for errors
  gpuErrchk(cudaGetLastError());
 }// end of ComputeVelocityShiftInX
//------------------------------------------------------------------------------



/**
 * CUDA kernel to compute velocity shift in Y. The matrix is XY transposed.
 * @param [in, out] FFT_shift_temp
 * @param [in]      y_shift_neg_r
 */
__global__ void CUDAComputeVelocityShiftInY(float2       * FFT_shift_temp,
                                            const float2 * y_shift_neg_r)
{
  const auto Y_2 = (DeviceConstants.Y_Size / 2) + 1;

  for (size_t z = GetZ(); z < DeviceConstants.Z_Size; z += GetZ_Stride())
  {
    for (size_t x = GetY(); x < DeviceConstants.X_Size; x += GetY_Stride())
    {
      for (size_t y = GetX(); y < Y_2; y += GetX_Stride())
      {
        const size_t i = (z * DeviceConstants.X_Size + x) * Y_2  + y;

        float2 temp;

        temp.x = ((FFT_shift_temp[i].x * y_shift_neg_r[y].x) -
                  (FFT_shift_temp[i].y * y_shift_neg_r[y].y)
                 ) * DeviceConstants.DividerY;


        temp.y = ((FFT_shift_temp[i].y * y_shift_neg_r[y].x) +
                  (FFT_shift_temp[i].x * y_shift_neg_r[y].y)
                 ) * DeviceConstants.DividerY;

        FFT_shift_temp[i] = temp;
      } // y
    } // x
  }// z
}// end of CUDAComputeVelocityShiftInY
//------------------------------------------------------------------------------

/**
 * Compute the velocity shift in Fourier space over the Y axis.
 * This kernel work with the transposed space.
 * @param [in,out] FFT_shift_temp
 * @param [in]     x_shift_neg
 */
void TCUDAImplementations::ComputeVelocityShiftInY(TCUFFTComplexMatrix  & FFT_shift_temp,
                                                   const TComplexMatrix & y_shift_neg_r)
{
  CUDAComputeVelocityShiftInY<<<GetSolverGridSize3D(),
                                GetSolverBlockSize3D()>>>
                             (reinterpret_cast<float2 *>       (FFT_shift_temp.GetRawDeviceData()),
                              reinterpret_cast<const float2 *> (y_shift_neg_r.GetRawDeviceData()));
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of ComputeVelocityShiftInY
//------------------------------------------------------------------------------


/**
 * CUDA kernel to compute velocity shift in Z. The matrix is XZ transposed
 * @param [in, out] FFT_shift_temp
 * @param [in]      z_shift_neg_r
 */
__global__ void CUDAComputeVelocityShiftInZ(float2       * FFT_shift_temp,
                                            const float2 * z_shift_neg_r)
{
  const auto Z_2 = (DeviceConstants.Z_Size / 2) + 1;

  for (size_t x = GetZ(); x < DeviceConstants.X_Size; x += GetZ_Stride())
  {
    for (size_t y = GetY(); y < DeviceConstants.Y_Size; y += GetY_Stride())
    {
      for (size_t z = GetX(); z < Z_2; z += GetX_Stride())
      {
        const size_t i = (x * DeviceConstants.Y_Size + y) * Z_2 + z;

        float2 temp;

        temp.x = ((FFT_shift_temp[i].x * z_shift_neg_r[z].x) -
                  (FFT_shift_temp[i].y * z_shift_neg_r[z].y)
                 ) * DeviceConstants.DividerZ;


        temp.y = ((FFT_shift_temp[i].y * z_shift_neg_r[z].x) +
                  (FFT_shift_temp[i].x * z_shift_neg_r[z].y)
                 ) * DeviceConstants.DividerZ;

        FFT_shift_temp[i] = temp;
      } // x
    } // y
  }// z
}// end of CUDAComputeVelocityShiftInZ
//------------------------------------------------------------------------------

/**
 * Compute the velocity shift in Fourier space over the Z axis.
 * This kernel work with the transposed space.
 * @param [in,out] FFT_shift_temp
 * @param [in]     z_shift_neg
 */
void TCUDAImplementations::ComputeVelocityShiftInZ(TCUFFTComplexMatrix  & FFT_shift_temp,
                                                   const TComplexMatrix & z_shift_neg_r)
{
  CUDAComputeVelocityShiftInZ<<<GetSolverGridSize3D(),
                                GetSolverBlockSize3D()>>>
                             (reinterpret_cast<float2 *>       (FFT_shift_temp.GetRawDeviceData()),
                              reinterpret_cast<const float2 *> (z_shift_neg_r.GetRawDeviceData()));
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of ComputeVelocityShiftInZ
//------------------------------------------------------------------------------

