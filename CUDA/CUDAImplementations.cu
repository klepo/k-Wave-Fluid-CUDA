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
 * @version     kspaceFirstOrder3D 3.3
 * @date        11 March    2013, 13:10 (created) \n
 *              20 December 2014, 50:15 (revised)
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
 * Set up execution model based on simulation size
 * @param FullDimensionSizes
 * @param ReducedDimensionSizes
 */
void TCUDAImplementations::SetUpExecutionModelWithTuner(const TDimensionSizes & FullDimensionSizes,
                                                        const TDimensionSizes & ReducedDimensionSizes)
{
  //@todo why you create CUDATuner here???
  CUDATuner = TCUDATuner::GetInstance();

  CUDATuner->GenerateExecutionModelForMatrixSize(FullDimensionSizes, ReducedDimensionSizes);
}// end of SetUpExecutionModelWithTuner
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

  // Set values for constatn memory
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

  ConstantsToTransfer.Divider = 1.0f / FullDimensionSizes.GetElementCount();

  // transfer constants to CUDA constant memory
  gpuErrchk(cudaMemcpyToSymbol(DeviceConstants, &ConstantsToTransfer, sizeof(TDeviceConstants)));
}// end of SetUpDeviceConstants
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
  CUDACompute_ux_sgx_normalize<<<CUDATuner->GetNumberOfBlocksFor3D(),
                                 CUDATuner->GetNumberOfThreadsFor3D() >>>
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
  CUDACompute_ux_sgx_normalize_scalar_uniform<<<CUDATuner->GetNumberOfBlocksFor3D(),
                                                CUDATuner->GetNumberOfThreadsFor3D() >>>
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
void TCUDAImplementations::Compute_ux_sgx_normalize_scalar_nonuniform(TRealMatrix&       ux_sgx,
                                                                      const TRealMatrix& FFT_p,
                                                                      const float        dt_rho0,
                                                                      const TRealMatrix& dxudxn_sgx,
                                                                      const TRealMatrix& pml)
{
  CUDACompute_ux_sgx_normalize_scalar_nonuniform<<<CUDATuner->GetNumberOfBlocksFor3D(),
                                                   CUDATuner->GetNumberOfThreadsFor3D()>>>
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
void TCUDAImplementations::Compute_uy_sgy_normalize(TRealMatrix& uy_sgy,
                                                    const TRealMatrix& FFT_p,
                                                    const TRealMatrix& dt_rho0,
                                                    const TRealMatrix& pml)
{
  CUDACompute_uy_sgy_normalize<<<CUDATuner->GetNumberOfBlocksFor3D(),
                                 CUDATuner->GetNumberOfThreadsFor3D() >>>
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
}// end of

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
  CUDACompute_uy_sgy_normalize_scalar_uniform<<<CUDATuner->GetNumberOfBlocksFor3D(),
                                                CUDATuner->GetNumberOfThreadsFor3D()>>>
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
__global__ void CUDACompute_uy_sgy_normalize_scalar_nonuniform(float*       uy_sgy,
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
 * @todo to be merged with ux and uz
 */
void TCUDAImplementations::Compute_uy_sgy_normalize_scalar_nonuniform(TRealMatrix&       uy_sgy,
                                                                      const TRealMatrix& FFT_p,
                                                                      const float        dt_rho0,
                                                                      const TRealMatrix& dyudyn_sgy,
                                                                      const TRealMatrix& pml)
{
  CUDACompute_uy_sgy_normalize_scalar_nonuniform<<<CUDATuner->GetNumberOfBlocksFor3D(),
                                                   CUDATuner->GetNumberOfThreadsFor3D()>>>
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
void TCUDAImplementations::Compute_uz_sgz_normalize(TRealMatrix& uz_sgz,
                                                    const TRealMatrix& FFT_p,
                                                    const TRealMatrix& dt_rho0,
                                                    const TRealMatrix& pml)
{
  CUDACompute_uz_sgz_normalize<<<CUDATuner->GetNumberOfBlocksFor3D(),
                                 CUDATuner->GetNumberOfThreadsFor3D()>>>
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
void TCUDAImplementations::Compute_uz_sgz_normalize_scalar_uniform(TRealMatrix& uz_sgz,
                                                                   const TRealMatrix& FFT_p,
                                                                   const float dt_rho0,
                                                                   const TRealMatrix& pml)
{
  CUDACompute_uz_sgz_normalize_scalar_uniform<<<CUDATuner->GetNumberOfBlocksFor3D(),
                                                CUDATuner->GetNumberOfThreadsFor3D()>>>
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
 * Compute a new value of uz_sgz.
 * This is the case for rho0 being a scalar and a non-uniform grid.
 * @param [in,out] uz_sgz     - updated value of uz_sgz
 * @param [in]     FFT_p      - gradient of p
 * @param [in]     dt_rho0    - scalar
 * @param [in]     dyudyn_sgz - matrix d shift
 * @param [in]     pml        - matrix of pml_z
 * @todo to be merged with ux and uy
 */
void TCUDAImplementations::Compute_uz_sgz_normalize_scalar_nonuniform(TRealMatrix&       uz_sgz,
                                                                      const TRealMatrix& FFT_p,
                                                                      const float        dt_rho0,
                                                                      const TRealMatrix& dzudzn_sgz,
                                                                      const TRealMatrix& pml)
{
  CUDACompute_uz_sgz_normalize_scalar_nonuniform<<<CUDATuner->GetNumberOfBlocksFor3D(),
                                                   CUDATuner->GetNumberOfThreadsFor3D()>>>
                                                (uz_sgz.GetRawDeviceData(),
                                                 FFT_p.GetRawDeviceData(),
                                                 dt_rho0,
                                                 dzudzn_sgz.GetRawDeviceData(),
                                                 pml.GetRawDeviceData());
  // check for errors
  gpuErrchk(cudaGetLastError());
}// end of Compute_uz_sgz_normalize_scalar_nonuniform
//------------------------------------------------------------------------------

__global__ void CudaAddTransducerSource(const size_t us_index_size,
                                              float* ux,
                                        const size_t*  us_index,
                                              size_t*  delay_mask,
                                        const float* transducer_signal)
{

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    while (i < us_index_size) {
        ux[us_index[i]] += transducer_signal[delay_mask[i]];
        delay_mask[i] ++;

        i += stride;
    }
}

void TCUDAImplementations::AddTransducerSource(TRealMatrix& uxyz_sgxyz,
                                              TIndexMatrix& us_index,
                                              TIndexMatrix& delay_mask,
                                              TRealMatrix& transducer_signal)
{

    const size_t us_index_size = us_index.GetTotalElementCount();
    float * ux = uxyz_sgxyz.GetRawDeviceData();
    const size_t* us_index_data = us_index.GetRawDeviceData();
    size_t * delay_mask_data = delay_mask.GetRawDeviceData();
    const float* transducer_signal_data = transducer_signal.GetRawDeviceData();

    CudaAddTransducerSource
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
        (us_index_size,
         ux,
         us_index_data,
         delay_mask_data,
         transducer_signal_data);
}

__global__ void CudaAdd_u_source(float* matrix_data,
                                 const float* u_source_input,
                                 const size_t* us_index,
                                 const int t_index,
                                 const size_t u_source_mode,
                                 const size_t u_source_many,
                                 const size_t us_index_total_element_count)
{

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t index2D = t_index;

    if (u_source_many != 0) { // is 2D
        index2D = t_index * us_index_total_element_count;
    }

    if (u_source_mode == 0){
        while(i < us_index_total_element_count){
            if (u_source_many != 0){
                matrix_data[us_index[i]] = u_source_input[index2D+i];
            }else{
                matrix_data[us_index[i]] = u_source_input[index2D];
            }
            i+= stride;
        }
    }// end of dirichlet

    if (u_source_mode == 1){
        while(i < us_index_total_element_count){
            if (u_source_many != 0){
                matrix_data[us_index[i]] += u_source_input[index2D+i];
            }else{
                matrix_data[us_index[i]] += u_source_input[index2D];
            }
            i+=stride;
        }
    }// end of add
}

void TCUDAImplementations::Add_u_source(TRealMatrix& uxyz_sgxyz,
                                       TRealMatrix& u_source_input,
                                       TIndexMatrix& us_index,
                                       int t_index,
                                       size_t u_source_mode,
                                       size_t u_source_many)
{

    float* matrix_data = uxyz_sgxyz.GetRawDeviceData();
    float* u_source_input_data = u_source_input.GetRawDeviceData();
    size_t*  us_index_data = us_index.GetRawDeviceData();
    size_t us_index_size = us_index.GetTotalElementCount();

    CudaAdd_u_source
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
        (matrix_data,
         u_source_input_data,
         us_index_data,
         t_index,
         u_source_mode,
         u_source_many,
         us_index_size);
}

/*
 * uxyz_sgxyzMatrix functions (fourier)
 */
__global__  void CudaCompute_dt_rho_sg_mul_ifft_div_2(
              float* dt,
        const float* dt_rho0_sg)//,
        //const size_t matrix_size)
{

    const size_t matrix_size = DeviceConstants.TotalElementCount;

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    const float divider = 1.0f/(2.0f * matrix_size);

    while (i < matrix_size) {
        dt[i] = dt_rho0_sg[i] * (dt[i] * divider);
        i += stride;
    }
}

void TCUDAImplementations::Compute_dt_rho_sg_mul_ifft_div_2(
        TRealMatrix& uxyz_sgxyz,
        TRealMatrix& dt_rho0_sg,
        TCUFFTComplexMatrix& FFT)
{

    FFT.Compute_FFT_3D_C2R(uxyz_sgxyz);

    float * matrix_data = uxyz_sgxyz.GetRawDeviceData();
    float * dt_rho0_sg_data = dt_rho0_sg.GetRawDeviceData();
    //size_t matrix_size = uxyz_sgxyz.GetTotalElementCount();

    CudaCompute_dt_rho_sg_mul_ifft_div_2
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
        (matrix_data,
         dt_rho0_sg_data);//,
         //matrix_size);
}

__global__  void CudaCompute_dt_rho_sg_mul_ifft_div_2(
              float* dt,
        const float dt_rho_0_sgx)//,
        //const size_t matrix_size)
{

    const size_t matrix_size = DeviceConstants.TotalElementCount;

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    const float divider = 1.0f/(2.0f * matrix_size) * dt_rho_0_sgx;

    while (i < matrix_size) {
        dt[i] = dt[i] * divider;
        i += stride;
    }
}

void TCUDAImplementations::Compute_dt_rho_sg_mul_ifft_div_2(
        TRealMatrix& uxyz_sgxyz,
        float dt_rho_0_sg,
        TCUFFTComplexMatrix& FFT)
{
    FFT.Compute_FFT_3D_C2R(uxyz_sgxyz);

    float* matrix_data = uxyz_sgxyz.GetRawDeviceData();
    float dt_rho0_sg_data = dt_rho_0_sg;
    //size_t matrix_size = uxyz_sgxyz.GetTotalElementCount();

    CudaCompute_dt_rho_sg_mul_ifft_div_2
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
        (matrix_data,
         dt_rho0_sg_data);//,
         //matrix_size);
}

__global__ void CudaCompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_x(
        float* dt,
        const float  dt_rho_0_sgx,
        const float* dxudxn_sgx)//,
        //const size_t maxX,
        //const size_t maxY,
        //const size_t maxZ)
{

    const size_t max_x = DeviceConstants.X_Size;
    const size_t max_y = DeviceConstants.Y_Size;
    const size_t max_z = DeviceConstants.Z_Size;
    const size_t matrix_size = DeviceConstants.TotalElementCount;

    //size_t matrix_size = maxX*maxY*maxZ;

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    const float divider = 1.0f/(2.0f * matrix_size) * dt_rho_0_sgx;

    for(; z < max_z; z+=zstride){
        for(; y < max_y; y+=ystride){
            for(; x < max_x; x+=xstride){
                register size_t i = z*max_y*max_x + y*max_x + x;
                dt[i] = dt[i] * divider * dxudxn_sgx[x];
            }
        }
    }
}

void TCUDAImplementations::Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_x(
        TRealMatrix& uxyz_sgxyz,
        float dt_rho_0_sgx,
        TRealMatrix& dxudxn_sgx,
        TCUFFTComplexMatrix& FFT)
{
    FFT.Compute_FFT_3D_C2R(uxyz_sgxyz);

    float* matrix_data = uxyz_sgxyz.GetRawDeviceData();
    float  dt_rho0_sgx_data = dt_rho_0_sgx;

    //const size_t max_x = uxyz_sgxyz.GetDimensionSizes().X;
    //const size_t max_y = uxyz_sgxyz.GetDimensionSizes().Y;
    //const size_t max_z = uxyz_sgxyz.GetDimensionSizes().Z;

    float* dxudxn_sgx_data = dxudxn_sgx.GetRawDeviceData();

    CudaCompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_x
        <<< CUDATuner->GetNumberOfBlocksFor3D(),
            CUDATuner->GetNumberOfThreadsFor3D() >>>
        (matrix_data,
         dt_rho0_sgx_data,
         dxudxn_sgx_data);//,
         //max_x,
         //max_y,
         //max_z);
}

__global__ void CudaCompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_y(
              float* dt,
        const float  dt_rho_0_sgy,
        const float* dyudyn_sgy)//,
        //const size_t maxX,
        //const size_t maxY,
        //const size_t maxZ)
{

    const size_t max_x = DeviceConstants.X_Size;
    const size_t max_y = DeviceConstants.Y_Size;
    const size_t max_z = DeviceConstants.Z_Size;
    const size_t matrix_size = DeviceConstants.TotalElementCount;

    //size_t matrix_size = maxX*maxY*maxZ;

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    const float divider = 1.0f/(2.0f * matrix_size) * dt_rho_0_sgy;

    for(; z < max_z; z+=zstride){
        for(; y < max_y; y+=ystride){
            const float dyudyn_sgy_data = dyudyn_sgy[y] * divider;
            for(; x < max_x; x+=xstride){
                register size_t i = z*max_y*max_x + y*max_x + x;

                dt[i] = dt[i] * dyudyn_sgy_data;
            }
        }
    }
}

void TCUDAImplementations::Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_y(
        TRealMatrix& uxyz_sgxyz,
        float dt_rho_0_sgy,
        TRealMatrix& dyudyn_sgy,
        TCUFFTComplexMatrix& FFT)
{
    FFT.Compute_FFT_3D_C2R(uxyz_sgxyz);

    float * matrix_data = uxyz_sgxyz.GetRawDeviceData();
    float dt_rho0_sgy_data = dt_rho_0_sgy;

    //const size_t max_x = uxyz_sgxyz.GetDimensionSizes().X;
    //const size_t max_y = uxyz_sgxyz.GetDimensionSizes().Y;
    //const size_t max_z = uxyz_sgxyz.GetDimensionSizes().Z;

    float* dyudyn_sgy_data = dyudyn_sgy.GetRawDeviceData();

    CudaCompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_y
        <<< CUDATuner->GetNumberOfBlocksFor3D(),
            CUDATuner->GetNumberOfThreadsFor3D() >>>
        (matrix_data,
         dt_rho0_sgy_data,
         dyudyn_sgy_data);//,
         //max_x,
         //max_y,
         //max_z);
}

__global__ void CudaCompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_z(
        float* dt,
        float  dt_rho_0_sgz,
        float* dzudzn_sgz)//,
        //size_t max_x,
        //size_t max_y,
        //size_t max_z)
{

    const size_t max_x = DeviceConstants.X_Size;
    const size_t max_y = DeviceConstants.Y_Size;
    const size_t max_z = DeviceConstants.Z_Size;
    const size_t matrix_size = DeviceConstants.TotalElementCount;

    //size_t matrix_size = maxX*maxY*maxZ;

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    const float divider = 1.0f/(2.0f * matrix_size) * dt_rho_0_sgz;

    for(; z < max_z; z+=zstride){
        const float dzudzn_sgz_data = dzudzn_sgz[z] * divider;
        for(; y < max_y; y+=ystride){
            for(; x < max_x; x+=xstride){
                register size_t i = z*max_y*max_x + y*max_x + x;

                dt[i] = dt[i] * dzudzn_sgz_data;
            }
        }
    }
}

void TCUDAImplementations::Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_z(TRealMatrix& uxyz_sgxyz,
        float dt_rho_0_sgz,
        TRealMatrix& dzudzn_sgz,
        TCUFFTComplexMatrix& FFT)
{
    FFT.Compute_FFT_3D_C2R(uxyz_sgxyz);

    float * matrix_data = uxyz_sgxyz.GetRawDeviceData();
    float dt_rho0_sgz_data = dt_rho_0_sgz;

    //const size_t max_x = uxyz_sgxyz.GetDimensionSizes().X;
    //const size_t max_y = uxyz_sgxyz.GetDimensionSizes().Y;
    //const size_t max_z = uxyz_sgxyz.GetDimensionSizes().Z;

    float* dzudzn_sgz_data = dzudzn_sgz.GetRawDeviceData();

    CudaCompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_z
        <<< CUDATuner->GetNumberOfBlocksFor3D(),
            CUDATuner->GetNumberOfThreadsFor3D() >>>
        (matrix_data,
         dt_rho0_sgz_data,
         dzudzn_sgz_data);//,
         //max_x,
         //max_y,
         //max_z);
}

__global__ void CudaCompute_ddx_kappa_fft_p(float* FFT_X,
                                            float* FFT_Y,
                                            float* FFT_Z,
                                            float* kappa,
                                            float* ddx,
                                            float* ddy,
                                            float* ddz)//,
                                            //size_t max_x,
                                            //size_t max_y,
                                            //size_t max_z)
{

    const size_t max_x = DeviceConstants.Complex_X_Size;
    const size_t max_y = DeviceConstants.Complex_Y_Size;
    const size_t max_z = DeviceConstants.Complex_Z_Size;

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    for(; z < max_z; z+=zstride){
        const float ddz_re = ddz[z<<1];
        const float ddz_im = ddz[(z<<1)+1];
        for(; y < max_y; y+=ystride){
            const float ddy_re = ddy[y<<1];
            const float ddy_im = ddy[(y<<1)+1];
            for(; x < max_x; x+=xstride){
                const float ddx_re = ddx[x<<1];
                const float ddx_im = ddx[(x<<1)+1];

                size_t i = x + y * max_x + z * (max_y * max_x);
                size_t i2 = i << 1;

                // kappa ./ p_k
                const float kappa_el  = kappa[i];
                const float p_k_el_re = FFT_X[i2] * kappa_el;
                const float p_k_el_im = FFT_X[i2+1] * kappa_el;

                //bxfun(ddx...)
                FFT_X[i2] = p_k_el_re * ddx_re - p_k_el_im * ddx_im;
                FFT_X[i2+1] = p_k_el_re * ddx_im + p_k_el_im * ddx_re;

                FFT_Y[i2] = p_k_el_re * ddy_re - p_k_el_im * ddy_im;
                FFT_Y[i2+1] = p_k_el_re * ddy_im + p_k_el_im * ddy_re;

                //bxfun(ddz...)
                FFT_Z[i2] = p_k_el_re * ddz_re - p_k_el_im * ddz_im;
                FFT_Z[i2+1] = p_k_el_re * ddz_im + p_k_el_im * ddz_re;
            }
        }
    }
}// end of Compute_ddx_kappa_fft_x
//----------------------------------------------------------------------------

/*
 * KSpaceFirstOrder3DSolver functions
 */
void TCUDAImplementations::Compute_ddx_kappa_fft_p(TRealMatrix& X_Matrix,
                                                  TCUFFTComplexMatrix& FFT_X,
                                                  TCUFFTComplexMatrix& FFT_Y,
                                                  TCUFFTComplexMatrix& FFT_Z,
                                                  TRealMatrix& kappa,
                                                  TComplexMatrix& ddx,
                                                  TComplexMatrix& ddy,
                                                  TComplexMatrix& ddz)
{
    // Compute FFT of X
    FFT_X.Compute_FFT_3D_R2C(X_Matrix);

    float* p_k_x_data = FFT_X.GetRawDeviceData();
    float* p_k_y_data = FFT_Y.GetRawDeviceData();
    float* p_k_z_data = FFT_Z.GetRawDeviceData();

    float* kappa_data = kappa.GetRawDeviceData();
    float* ddx_data   = ddx.GetRawDeviceData();
    float* ddy_data   = ddy.GetRawDeviceData();
    float* ddz_data   = ddz.GetRawDeviceData();

    //const size_t Z_Size = FFT_X.GetDimensionSizes().Z;
    //const size_t Y_Size = FFT_X.GetDimensionSizes().Y;
    //const size_t X_Size = FFT_X.GetDimensionSizes().X;

    CudaCompute_ddx_kappa_fft_p
        <<< CUDATuner->GetNumberOfBlocksFor3DComplex(),
            CUDATuner->GetNumberOfThreadsFor3D() >>>
            (p_k_x_data,
             p_k_y_data,
             p_k_z_data,
             kappa_data,
             ddx_data,
             ddy_data,
             ddz_data);//,
             //X_Size,
             //Y_Size,
             //Z_Size);
}

__global__  void CudaCompute_duxyz_initial(      float* Temp_FFT_X_Data,
                                                 float* Temp_FFT_Y_Data,
                                                 float* Temp_FFT_Z_Data,
                                           const float* kappa_data,
                                           const float* ddx,
                                           const float* ddy,
                                           const float* ddz)//,
                                           //const size_t max_x,
                                           //const size_t max_y,
                                           //const size_t max_z,
                                           //const float divider)
{

    const size_t max_x = DeviceConstants.Complex_X_Size;
    const size_t max_y = DeviceConstants.Complex_Y_Size;
    const size_t max_z = DeviceConstants.Complex_Z_Size;
    const float  divider = DeviceConstants.Divider;

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    for (; z < max_z; z+=zstride){
        //real
        const float ddz_neg_re = ddz[z*2];
        //imag
        const float ddz_neg_im = ddz[z*2+1];
        for (; y < max_y; y+=ystride){

            const float ddy_neg_re = ddy[y*2];
            const float ddy_neg_im = ddy[y*2+1];

            for (; x < max_x; x+=xstride){
                register size_t i = x + y * max_x + z * (max_y * max_x);
                i = i << 1;
                float FFT_el_x_re = Temp_FFT_X_Data[i];
                float FFT_el_x_im = Temp_FFT_X_Data[i+1];

                float FFT_el_y_re = Temp_FFT_Y_Data[i];
                float FFT_el_y_im = Temp_FFT_Y_Data[i+1];

                float FFT_el_z_re = Temp_FFT_Z_Data[i];
                float FFT_el_z_im = Temp_FFT_Z_Data[i+1];

                const float kappa_el = kappa_data[i >> 1];

                FFT_el_x_re *= kappa_el;
                FFT_el_x_im *= kappa_el;

                FFT_el_y_re *= kappa_el;
                FFT_el_y_im *= kappa_el;

                FFT_el_z_re *= kappa_el;
                FFT_el_z_im *= kappa_el;

                Temp_FFT_X_Data[i]     = ((FFT_el_x_re * ddx[x*2]) -
                        (FFT_el_x_im * ddx[x*2+1])) * divider;
                Temp_FFT_X_Data[i + 1] = ((FFT_el_x_im * ddx[x*2]) +
                        (FFT_el_x_re * ddx[x*2+1])) * divider;

                Temp_FFT_Y_Data[i]     = ((FFT_el_y_re * ddy_neg_re) -
                        (FFT_el_y_im * ddy_neg_im)) * divider;
                Temp_FFT_Y_Data[i + 1] = ((FFT_el_y_im * ddy_neg_re) +
                        (FFT_el_y_re * ddy_neg_im)) * divider;

                Temp_FFT_Z_Data[i]     = ((FFT_el_z_re * ddz_neg_re) -
                        (FFT_el_z_im * ddz_neg_im)) * divider;
                Temp_FFT_Z_Data[i + 1] = ((FFT_el_z_im * ddz_neg_re) +
                        (FFT_el_z_re * ddz_neg_im)) * divider;
            } // x
        } // y
    } // z
}

void TCUDAImplementations::Compute_duxyz_initial(
        TCUFFTComplexMatrix& Temp_FFT_X,
        TCUFFTComplexMatrix& Temp_FFT_Y,
        TCUFFTComplexMatrix& Temp_FFT_Z,
        TRealMatrix& kappa,
        TRealMatrix& ux_sgx,
        TComplexMatrix& ddx_k_shift_neg,
        TComplexMatrix& ddy_k_shift_neg,
        TComplexMatrix& ddz_k_shift_neg)
{

    float * Temp_FFT_X_Data  = Temp_FFT_X.GetRawDeviceData();
    float * Temp_FFT_Y_Data  = Temp_FFT_Y.GetRawDeviceData();
    float * Temp_FFT_Z_Data  = Temp_FFT_Z.GetRawDeviceData();

    const float * kappa_data   = kappa.GetRawDeviceData();

    //const size_t FFT_Z_dim = Temp_FFT_Z.GetDimensionSizes().Z;
    //const size_t FFT_Y_dim = Temp_FFT_Y.GetDimensionSizes().Y;
    //const size_t FFT_X_dim = Temp_FFT_X.GetDimensionSizes().X;

    //const float  Divider = 1.0f / ux_sgx.GetTotalElementCount();

    const float * ddx = (float*) ddx_k_shift_neg.GetRawDeviceData();
    const float * ddy = (float*) ddy_k_shift_neg.GetRawDeviceData();
    const float * ddz = (float*) ddz_k_shift_neg.GetRawDeviceData();

    CudaCompute_duxyz_initial
        <<< CUDATuner->GetNumberOfBlocksFor3D(),
            CUDATuner->GetNumberOfThreadsFor3D() >>>
            (Temp_FFT_X_Data,
             Temp_FFT_Y_Data,
             Temp_FFT_Z_Data,
             kappa_data,
             ddx,
             ddy,
             ddz);//,
             //FFT_X_dim,
             //FFT_Y_dim,
             //FFT_Z_dim,
             //Divider);
}

__global__  void CudaCompute_duxyz_non_linear(float* duxdx_data,
        float* duydy_data,
        float* duzdz_data,
        const float* duxdxn_data,
        const float* duydyn_data,
        const float* duzdzn_data,
        const size_t max_x,
        const size_t max_y,
        const size_t max_z)
{

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    for (; z < max_z; z+=zstride){
        const float duzdzn_el = duzdzn_data[z];
        for (; y < max_y; y+=ystride){
            const float dyudyn_el = duydyn_data[y];
            for (; x < max_x; x+=xstride){
                register size_t i = x + y * max_x + z * (max_y * max_x);
                duxdx_data[i] *= duxdxn_data[x];
                duydy_data[i] *= dyudyn_el;
                duzdz_data[i] *= duzdzn_el;
            } // x
        } // y
    } // z
}

void TCUDAImplementations::Compute_duxyz_non_linear(TRealMatrix& duxdx,
        TRealMatrix& duydy,
        TRealMatrix& duzdz,
        TRealMatrix& dxudxn,
        TRealMatrix& dyudyn,
        TRealMatrix& dzudzn)
{

    float * duxdx_data = duxdx.GetRawDeviceData();
    float * duydy_data = duydy.GetRawDeviceData();
    float * duzdz_data = duzdz.GetRawDeviceData();

    const float * duxdxn_data = dxudxn.GetRawDeviceData();
    const float * duydyn_data = dyudyn.GetRawDeviceData();
    const float * duzdzn_data = dzudzn.GetRawDeviceData();

    const size_t Z_Size = duxdx.GetDimensionSizes().Z;
    const size_t Y_Size = duxdx.GetDimensionSizes().Y;
    const size_t X_Size = duxdx.GetDimensionSizes().X;

    CudaCompute_duxyz_non_linear
        <<< CUDATuner->GetNumberOfBlocksFor3D(),
        CUDATuner->GetNumberOfThreadsFor3D() >>>
            (duxdx_data,
             duydy_data,
             duzdz_data,
             duxdxn_data,
             duydyn_data,
             duzdzn_data,
             X_Size,
             Y_Size,
             Z_Size);
}

__global__ void CudaComputeC2_matrix(float * c2_data, size_t max_i){
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    while (i < max_i){
        c2_data[i] = c2_data[i] * c2_data[i];
        i += istride;
    }
}

void TCUDAImplementations::ComputeC2_matrix(TRealMatrix& c2)
{
    float * c2_data =  c2.GetRawDeviceData();

    CudaComputeC2_matrix
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
            (c2_data,
             c2.GetTotalElementCount());
}

__global__ void CudaCalculate_p0_source_add_initial_pressure(
        float*       rhox_data,
        float*       rhoy_data,
        float*       rhoz_data,
        const float* p0_data,
        const float  c2,
        const size_t max_i)
{

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    float tmp;

    while (i < max_i){
        tmp = p0_data[i] / (3.0f * c2);
        rhox_data[i] = tmp;
        rhoy_data[i] = tmp;
        rhoz_data[i] = tmp;

        i+=istride;
    }
}

__global__ void CudaCalculate_p0_source_add_initial_pressure(
        float*       rhox_data,
        float*       rhoy_data,
        float*       rhoz_data,
        const float* p0_data,
        const size_t c2_shift,
        const float* c2,
        const size_t max_i)
{

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    float tmp;

    while (i < max_i){
        tmp = p0_data[i] / (3.0f * c2[i * c2_shift]);
        rhox_data[i] = tmp;
        rhoy_data[i] = tmp;
        rhoz_data[i] = tmp;

        i+=istride;
    }
}

void TCUDAImplementations::Calculate_p0_source_add_initial_pressure(
        TRealMatrix& rhox,
        TRealMatrix& rhoy,
        TRealMatrix& rhoz,
        TRealMatrix& p0,
        size_t c2_shift,
        float* c2_data)
{
    const float* p0_data = p0.GetRawDeviceData();

    float* rhox_data = rhox.GetRawDeviceData();
    float* rhoy_data = rhoy.GetRawDeviceData();
    float* rhoz_data = rhoz.GetRawDeviceData();

    if (!c2_shift){
        CudaCalculate_p0_source_add_initial_pressure
            <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
                (rhox_data,
                 rhoy_data,
                 rhoz_data,
                 p0_data,
                 c2_data[0],
                 rhox.GetTotalElementCount());
    }
    else {
        CudaCalculate_p0_source_add_initial_pressure
            <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
                (rhox_data,
                 rhoy_data,
                 rhoz_data,
                 p0_data,
                 c2_shift,
                 c2_data,
                 rhox.GetTotalElementCount());
    }
}

__global__ void CudaCompute_rhoxyz_nonlinear_scalar(float* rhox_data,
        float* rhoy_data,
        float* rhoz_data,
        const float* pml_x_data,
        const float* pml_y_data,
        const float* pml_z_data,
        const float* duxdx_data,
        const float* duydy_data,
        const float* duzdz_data,
        const float dt_rho0,
        const float dt2,
        const size_t max_x,
        const size_t max_y,
        const size_t max_z)
{

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    for(; z < max_z; z += zstride){
        const float pml_z = pml_z_data[z];
        for(; y < max_y; y += ystride){
            const float pml_y = pml_y_data[y];
            for(; x < max_x; x += xstride){
                register size_t i = x + y * max_x + z * (max_y * max_x);

                const float pml_x = pml_x_data[x];
                float dux = duxdx_data[i];

                rhox_data[i] = pml_x * (
                        ((pml_x * rhox_data[i]) - (dt_rho0 * dux))/
                        (1.0f + (dt2 * dux))
                        );

                float duy = duydy_data[i];
                rhoy_data[i] = pml_y * (
                        ((pml_y * rhoy_data[i]) - (dt_rho0 * duy))/
                        (1.0f + (dt2 * duy))
                        );

                float duz = duzdz_data[i];
                rhoz_data[i] = pml_z * (
                        ((pml_z * rhoz_data[i]) - (dt_rho0 * duz))/
                        (1.0f + (dt2 * duz))
                        );
            }
        }
    }
}

void TCUDAImplementations::Compute_rhoxyz_nonlinear_scalar(TRealMatrix& rhox,
        TRealMatrix& rhoy,
        TRealMatrix& rhoz,
        TRealMatrix& pml_x,
        TRealMatrix& pml_y,
        TRealMatrix& pml_z,
        TRealMatrix& duxdx,
        TRealMatrix& duydy,
        TRealMatrix& duzdz,
        float dt_el,
        float rho0_scalar)
{
    const size_t max_z = rhox.GetDimensionSizes().Z;
    const size_t max_y = rhox.GetDimensionSizes().Y;
    const size_t max_x = rhox.GetDimensionSizes().X;

    float * rhox_data  = rhox.GetRawDeviceData();
    float * rhoy_data  = rhoy.GetRawDeviceData();
    float * rhoz_data  = rhoz.GetRawDeviceData();

    const float * pml_x_data = pml_x.GetRawDeviceData();
    const float * pml_y_data = pml_y.GetRawDeviceData();
    const float * pml_z_data = pml_z.GetRawDeviceData();
    const float * duxdx_data = duxdx.GetRawDeviceData();
    const float * duydy_data = duydy.GetRawDeviceData();
    const float * duzdz_data = duzdz.GetRawDeviceData();

    const float dt2 = 2.0f * dt_el;
    const float dt_rho0 = rho0_scalar * dt_el;

    CudaCompute_rhoxyz_nonlinear_scalar
        <<< CUDATuner->GetNumberOfBlocksFor3D(),
        CUDATuner->GetNumberOfThreadsFor3D() >>>
            (rhox_data,
             rhoy_data,
             rhoz_data,
             pml_x_data,
             pml_y_data,
             pml_z_data,
             duxdx_data,
             duydy_data,
             duzdz_data,
             dt_rho0,
             dt2,
             max_x,
             max_y,
             max_z);
}

__global__ void CudaCompute_rhoxyz_nonlinear_matrix(float* rhox_data,
        float* rhoy_data,
        float* rhoz_data,
        const float* pml_x_data,
        const float* pml_y_data,
        const float* pml_z_data,
        const float* duxdx_data,
        const float* duydy_data,
        const float* duzdz_data,
        const float dt_el,
        const float* rho0_data,
        const size_t max_x,
        const size_t max_y,
        const size_t max_z)
{

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    const float dt2 = 2.0f * dt_el;

    for(; z < max_z; z += zstride){
        const float pml_z = pml_z_data[z];

        for(; y < max_y; y += ystride){
            const float pml_y = pml_y_data[y];

            for(; x < max_x; x += xstride){
                const float pml_x = pml_x_data[x];

                register size_t i = x + y * max_x + z * (max_y * max_x);
                const float dt_rho0 = dt_el * rho0_data[i];

                float dux = duxdx_data[i];
                rhox_data[i] = pml_x * (
                        ((pml_x * rhox_data[i]) - (dt_rho0 * dux))/
                        (1.0f + (dt2 * dux))
                        );

                float duy = duydy_data[i];
                rhoy_data[i] = pml_y * (
                        ((pml_y * rhoy_data[i]) - (dt_rho0 * duy))/
                        (1.0f + (dt2 * duy))
                        );

                float duz = duzdz_data[i];
                rhoz_data[i] = pml_z * (
                        ((pml_z * rhoz_data[i]) - (dt_rho0 * duz))/
                        (1.0f + (dt2 * duz))
                        );
            }
        }
    }
}

void TCUDAImplementations::Compute_rhoxyz_nonlinear_matrix(TRealMatrix& rhox,
        TRealMatrix& rhoy,
        TRealMatrix& rhoz,
        TRealMatrix& pml_x,
        TRealMatrix& pml_y,
        TRealMatrix& pml_z,
        TRealMatrix& duxdx,
        TRealMatrix& duydy,
        TRealMatrix& duzdz,
        float dt_el,
        TRealMatrix& rho0)
{

    const size_t max_z = rhox.GetDimensionSizes().Z;
    const size_t max_y = rhox.GetDimensionSizes().Y;
    const size_t max_x = rhox.GetDimensionSizes().X;

    float * rhox_data  = rhox.GetRawDeviceData();
    float * rhoy_data  = rhoy.GetRawDeviceData();
    float * rhoz_data  = rhoz.GetRawDeviceData();

    const float * pml_x_data = pml_x.GetRawDeviceData();
    const float * pml_y_data = pml_y.GetRawDeviceData();
    const float * pml_z_data = pml_z.GetRawDeviceData();
    const float * duxdx_data = duxdx.GetRawDeviceData();
    const float * duydy_data = duydy.GetRawDeviceData();
    const float * duzdz_data = duzdz.GetRawDeviceData();

    const float * rho0_data  = rho0.GetRawDeviceData();

    CudaCompute_rhoxyz_nonlinear_matrix
        <<< CUDATuner->GetNumberOfBlocksFor3D(),
        CUDATuner->GetNumberOfThreadsFor3D() >>>
            (rhox_data,
             rhoy_data,
             rhoz_data,
             pml_x_data,
             pml_y_data,
             pml_z_data,
             duxdx_data,
             duydy_data,
             duzdz_data,
             dt_el,
             rho0_data,
             max_x,
             max_y,
             max_z);
}

__global__ void CudaCompute_rhoxyz_linear_scalar(float* rhox_data,
        float* rhoy_data,
        float* rhoz_data,
        const float* pml_x_data,
        const float* pml_y_data,
        const float* pml_z_data,
        const float* duxdx_data,
        const float* duydy_data,
        const float* duzdz_data,
        const float dt_rho0,
        const size_t max_x,
        const size_t max_y,
        const size_t max_z)
{

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    for (; z < max_z; z += zstride){
        const float pml_z = pml_z_data[z];
        for (; y < max_y; y += ystride){
            const float pml_y = pml_y_data[y];
            for (; x < max_x; x += xstride){

                register size_t i = x + y * max_x + z * (max_y * max_x);

                const float pml_x = pml_x_data[x];
                rhox_data[i] = pml_x * (((pml_x * rhox_data[i]) -
                            (dt_rho0 * duxdx_data[i])));

                rhoy_data[i] = pml_y * (((pml_y * rhoy_data[i]) -
                            (dt_rho0 * duydy_data[i])));

                rhoz_data[i] = pml_z * (((pml_z * rhoz_data[i]) -
                            (dt_rho0 * duzdz_data[i])));
            }
        }
    }
}

void TCUDAImplementations::Compute_rhoxyz_linear_scalar(TRealMatrix& rhox,
        TRealMatrix& rhoy,
        TRealMatrix& rhoz,
        TRealMatrix& pml_x,
        TRealMatrix& pml_y,
        TRealMatrix& pml_z,
        TRealMatrix& duxdx,
        TRealMatrix& duydy,
        TRealMatrix& duzdz,
        const float dt_el,
        const float rho0_scalar
        )
{
    const size_t max_z = rhox.GetDimensionSizes().Z;
    const size_t max_y = rhox.GetDimensionSizes().Y;
    const size_t max_x = rhox.GetDimensionSizes().X;

    float * rhox_data  = rhox.GetRawDeviceData();
    float * rhoy_data  = rhoy.GetRawDeviceData();
    float * rhoz_data  = rhoz.GetRawDeviceData();

    const float * pml_x_data = pml_x.GetRawDeviceData();
    const float * pml_y_data = pml_y.GetRawDeviceData();
    const float * pml_z_data = pml_z.GetRawDeviceData();
    const float * duxdx_data = duxdx.GetRawDeviceData();
    const float * duydy_data = duydy.GetRawDeviceData();
    const float * duzdz_data = duzdz.GetRawDeviceData();

    const float dt_rho0 = rho0_scalar * dt_el;

    CudaCompute_rhoxyz_linear_scalar
        <<< CUDATuner->GetNumberOfBlocksFor3D(),
        CUDATuner->GetNumberOfThreadsFor3D() >>>
            (rhox_data,
             rhoy_data,
             rhoz_data,
             pml_x_data,
             pml_y_data,
             pml_z_data,
             duxdx_data,
             duydy_data,
             duzdz_data,
             dt_rho0,
             max_x,
             max_y,
             max_z);
}

__global__ void CudaCompute_rhoxyz_linear_matrix(float * rhox_data,
        float * rhoy_data,
        float * rhoz_data,
        const float * pml_x_data,
        const float * pml_y_data,
        const float * pml_z_data,
        const float * duxdx_data,
        const float * duydy_data,
        const float * duzdz_data,
        const float dt_el,
        const float * rho0_data,
        const size_t max_x,
        const size_t max_y,
        const size_t max_z)
{

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    for (; z < max_z; z += zstride){
        const float pml_z = pml_z_data[z];

        for (; y < max_y; y += ystride){
            const float pml_y = pml_y_data[y];

            for (; x < max_x; x += xstride){
                register size_t i = x + y * max_x + z * (max_y * max_x);

                const float dt_rho0 = dt_el * rho0_data[i];
                const float pml_x   = pml_x_data[x];

                rhox_data[i] = pml_x * (((pml_x * rhox_data[i]) -
                            (dt_rho0 * duxdx_data[i])));

                rhoy_data[i] = pml_y * (((pml_y * rhoy_data[i]) -
                            (dt_rho0 * duydy_data[i])));

                rhoz_data[i] = pml_z * (((pml_z * rhoz_data[i]) -
                            (dt_rho0 * duzdz_data[i])));
            }
        }
    }
}

void TCUDAImplementations::Compute_rhoxyz_linear_matrix(TRealMatrix& rhox,
        TRealMatrix& rhoy,
        TRealMatrix& rhoz,
        TRealMatrix& pml_x,
        TRealMatrix& pml_y,
        TRealMatrix& pml_z,
        TRealMatrix& duxdx,
        TRealMatrix& duydy,
        TRealMatrix& duzdz,
        const float dt_el,
        TRealMatrix& rho0)
{
    const size_t max_z = rhox.GetDimensionSizes().Z;
    const size_t max_y = rhox.GetDimensionSizes().Y;
    const size_t max_x = rhox.GetDimensionSizes().X;
    float * rhox_data  = rhox.GetRawDeviceData();
    float * rhoy_data  = rhoy.GetRawDeviceData();
    float * rhoz_data  = rhoz.GetRawDeviceData();

    const float * pml_x_data = pml_x.GetRawDeviceData();
    const float * pml_y_data = pml_y.GetRawDeviceData();
    const float * pml_z_data = pml_z.GetRawDeviceData();
    const float * duxdx_data = duxdx.GetRawDeviceData();
    const float * duydy_data = duydy.GetRawDeviceData();
    const float * duzdz_data = duzdz.GetRawDeviceData();

    const float * rho0_data  = rho0.GetRawDeviceData();

    CudaCompute_rhoxyz_linear_matrix
        <<< CUDATuner->GetNumberOfBlocksFor3D(),
        CUDATuner->GetNumberOfThreadsFor3D() >>>
            (rhox_data,
             rhoy_data,
             rhoz_data,
             pml_x_data,
             pml_y_data,
             pml_z_data,
             duxdx_data,
             duydy_data,
             duzdz_data,
             dt_el,
             rho0_data,
             max_x,
             max_y,
             max_z);

}

__global__ void CudaAdd_p_sourceReplacement(
        float* rhox_data,
        float* rhoy_data,
        float* rhoz_data,
        const float* p_source_input_data,
        const size_t* p_source_index_data,
        const size_t index2D,
        const size_t p_source_many,
        const size_t max_i)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    if (p_source_many != 0){
        for (; i < max_i; i += istride){

            rhox_data[p_source_index_data[i]] = p_source_input_data[index2D+i];
            rhoy_data[p_source_index_data[i]] = p_source_input_data[index2D+i];
            rhoz_data[p_source_index_data[i]] = p_source_input_data[index2D+i];

        }
    }
    else{
        for (; i < max_i; i += istride){

            rhox_data[p_source_index_data[i]] = p_source_input_data[index2D];
            rhoy_data[p_source_index_data[i]] = p_source_input_data[index2D];
            rhoz_data[p_source_index_data[i]] = p_source_input_data[index2D];

        }
    }
}

__global__ void CudaAdd_p_sourceAddition(
        float* rhox_data,
        float* rhoy_data,
        float* rhoz_data,
        const float* p_source_input_data,
        const size_t* p_source_index_data,
        const size_t index2D,
        const size_t p_source_many,
        const size_t max_i)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    if (p_source_many != 0){
        for (; i < max_i; i += istride){

            rhox_data[p_source_index_data[i]] += p_source_input_data[index2D+i];
            rhoy_data[p_source_index_data[i]] += p_source_input_data[index2D+i];
            rhoz_data[p_source_index_data[i]] += p_source_input_data[index2D+i];

        }
    }
    else{
        for (; i < max_i; i += istride){

            rhox_data[p_source_index_data[i]] += p_source_input_data[index2D];
            rhoy_data[p_source_index_data[i]] += p_source_input_data[index2D];
            rhoz_data[p_source_index_data[i]] += p_source_input_data[index2D];

        }
    }
}

void TCUDAImplementations::Add_p_source(TRealMatrix& rhox,
        TRealMatrix& rhoy,
        TRealMatrix& rhoz,
        TRealMatrix& p_source_input,
        TIndexMatrix& p_source_index,
        size_t p_source_many,
        size_t p_source_mode,
        size_t t_index)
{
    float * rhox_data = rhox.GetRawDeviceData();
    float * rhoy_data = rhoy.GetRawDeviceData();
    float * rhoz_data = rhoz.GetRawDeviceData();

    const float * p_source_input_data = p_source_input.GetRawDeviceData();
    const size_t  * p_source_index_data = p_source_index.GetRawDeviceData();

    size_t index2D = t_index;

    if (p_source_many != 0) { // is 2D
        index2D = t_index * p_source_index.GetTotalElementCount();
    }

    // replacement
    if (p_source_mode == 0){
        CudaAdd_p_sourceReplacement
            <<< CUDATuner->GetNumberOfBlocksForSubmatrixWithSize(
                            p_source_index.GetTotalElementCount()),
                //CUDATuner->GetNumberOfBlocksFor1D(),
                CUDATuner->GetNumberOfThreadsFor1D() >>>
                (rhox_data,
                 rhoy_data,
                 rhoz_data,
                 p_source_input_data,
                 p_source_index_data,
                 index2D,
                 p_source_many,
                 p_source_index.GetTotalElementCount());
        // Addition
    }else{
        CudaAdd_p_sourceAddition
            <<< CUDATuner->GetNumberOfBlocksForSubmatrixWithSize(
                            p_source_index.GetTotalElementCount()),
                //CUDATuner->GetNumberOfBlocksFor1D(),
                CUDATuner->GetNumberOfThreadsFor1D() >>>
                (rhox_data,
                 rhoy_data,
                 rhoz_data,
                 p_source_input_data,
                 p_source_index_data,
                 index2D,
                 p_source_many,
                 p_source_index.GetTotalElementCount());
    }// type of replacement
}

__global__ void CudaCalculate_SumRho_BonA_SumDu(float*       RHO_Temp_Data,
                                                float*       BonA_Temp_Data,
                                                float*       SumDU_Temp_Data,
                                                const float* rhox_data,
                                                const float* rhoy_data,
                                                const float* rhoz_data,
                                                const float* dux_data,
                                                const float* duy_data,
                                                const float* duz_data,
                                                const float  BonA_data_scalar,
                                                const float* BonA_data_matrix,
                                                const size_t BonA_shift,
                                                const float  rho0_data_scalar,
                                                const float* rho0_data_matrix,
                                                const size_t rho0_shift,
                                                const size_t max_i)
{

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < max_i; i += istride){
        float BonA_data, rho0_data;

        if(BonA_shift == 0){
            BonA_data = BonA_data_scalar;
        }
        else{
            BonA_data = BonA_data_matrix[i];
        }

        if(rho0_shift == 0){
            rho0_data = rho0_data_scalar;
        }
        else{
            rho0_data = rho0_data_matrix[i];
        }

        register const float rho_xyz_el = rhox_data[i] +
            rhoy_data[i] +
            rhoz_data[i];
        RHO_Temp_Data[i]   = rho_xyz_el;
        BonA_Temp_Data[i]  = ((BonA_data * (rho_xyz_el * rho_xyz_el))
                / (2.0f * rho0_data)) + rho_xyz_el;
        SumDU_Temp_Data[i] = rho0_data * (dux_data[i] +
                duy_data[i] +
                duz_data[i]);
    }
}

void TCUDAImplementations::Calculate_SumRho_BonA_SumDu(
        TRealMatrix& RHO_Temp,
        TRealMatrix& BonA_Temp,
        TRealMatrix& Sum_du,
        TRealMatrix& rhox,
        TRealMatrix& rhoy,
        TRealMatrix& rhoz,
        TRealMatrix& duxdx,
        TRealMatrix& duydy,
        TRealMatrix& duzdz,
        const float  BonA_data_scalar,
        const float* BonA_data_matrix,
        const size_t BonA_shift,
        const float  rho0_data_scalar,
        const float* rho0_data_matrix,
        const size_t rho0_shift)
{
    const float* rhox_data = rhox.GetRawDeviceData();
    const float* rhoy_data = rhoy.GetRawDeviceData();
    const float* rhoz_data = rhoz.GetRawDeviceData();

    const float* dux_data = duxdx.GetRawDeviceData();
    const float* duy_data = duydy.GetRawDeviceData();
    const float* duz_data = duzdz.GetRawDeviceData();

          float* RHO_Temp_Data  = RHO_Temp.GetRawDeviceData();
          float* BonA_Temp_Data = BonA_Temp.GetRawDeviceData();
          float* SumDU_Temp_Data= Sum_du.GetRawDeviceData();

    CudaCalculate_SumRho_BonA_SumDu
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
            (RHO_Temp_Data,
             BonA_Temp_Data,
             SumDU_Temp_Data,
             rhox_data,
             rhoy_data,
             rhoz_data,
             dux_data,
             duy_data,
             duz_data,
             BonA_data_scalar,
             BonA_data_matrix,
             BonA_shift,
             rho0_data_scalar,
             rho0_data_matrix,
             rho0_shift,
             RHO_Temp.GetTotalElementCount());
}

__global__ void CudaCompute_Absorb_nabla1_2(float* FFT_1_data,
        float* FFT_2_data,
        const float* nabla1,
        const float* nabla2,
        const size_t max_i)
{

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < max_i; i += istride){
        float nabla_data1 = nabla1[i];
        FFT_1_data[(i<<1)]   *= nabla_data1;
        FFT_1_data[(i<<1)+1] *= nabla_data1;

        float nabla_data2 = nabla2[i];
        FFT_2_data[(i<<1)]   *= nabla_data2;
        FFT_2_data[(i<<1)+1] *= nabla_data2;
    }
}

void TCUDAImplementations::Compute_Absorb_nabla1_2(TRealMatrix& absorb_nabla1,
        TRealMatrix& absorb_nabla2,
        TCUFFTComplexMatrix& FFT_1,
        TCUFFTComplexMatrix& FFT_2)
{
    const float * nabla1 = absorb_nabla1.GetRawDeviceData();
    const float * nabla2 = absorb_nabla2.GetRawDeviceData();

    float * FFT_1_data  = FFT_1.GetRawDeviceData();
    float * FFT_2_data  = FFT_2.GetRawDeviceData();

    CudaCompute_Absorb_nabla1_2
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
        CUDATuner->GetNumberOfThreadsFor1D() >>>
            (FFT_1_data,
             FFT_2_data,
             nabla1,
             nabla2,
             FFT_1.GetTotalElementCount());
}

__global__ void CudaSum_Subterms_nonlinear(float*       p_data,
        const float* BonA_data,
        const float  c2_data_scalar,
        const float* c2_data_matrix,
        const size_t c2_shift,
        const float* Absorb_tau_data,
        const float  tau_data_scalar,
        const float* tau_data_matrix,
        const float* Absorb_eta_data,
        const float  eta_data_scalar,
        const float* eta_data_matrix,
        const size_t tau_eta_shift,
        const size_t max_i)
{
    const float divider = 1.0f / (float) max_i;

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < max_i; i += istride){

        float c2_data, eta_data, tau_data;

        if(c2_shift == 0){
            c2_data = c2_data_scalar;
        }
        else{
            c2_data = c2_data_matrix[i];
        }

        if(tau_eta_shift == 0){
            tau_data = tau_data_scalar;
            eta_data = eta_data_scalar;
        }
        else{
            tau_data = tau_data_matrix[i];
            eta_data = eta_data_matrix[i];
        }

        p_data[i] = c2_data *(BonA_data[i] +
                (divider * ((Absorb_tau_data[i] * tau_data) -
                            (Absorb_eta_data[i] * eta_data))));
    }
}

void TCUDAImplementations::Sum_Subterms_nonlinear(TRealMatrix& BonA_temp,
        TRealMatrix& p,
        const float  c2_data_scalar,
        const float* c2_data_matrix,
        const size_t c2_shift,
        const float* Absorb_tau_data,
        const float  tau_data_scalar,
        const float* tau_data_matrix,
        const float* Absorb_eta_data,
        const float  eta_data_scalar,
        const float* eta_data_matrix,
        const size_t tau_eta_shift){

    const size_t TotalElementCount = p.GetTotalElementCount();

    const float * BonA_data = BonA_temp.GetRawDeviceData();
    float * p_data  = p.GetRawDeviceData();

    CudaSum_Subterms_nonlinear
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
            (p_data,
             BonA_data,
             c2_data_scalar,
             c2_data_matrix,
             c2_shift,
             Absorb_tau_data,
             tau_data_scalar,
             tau_data_matrix,
             Absorb_eta_data,
             eta_data_scalar,
             eta_data_matrix,
             tau_eta_shift,
             p.GetTotalElementCount());
}

__global__ void CudaSum_new_p_nonlinear_lossless(
        const size_t TotalElementCount,
        float*       p_data,
        const float* rhox_data,
        const float* rhoy_data,
        const float* rhoz_data,
        const float  c2_data_scalar,
        const float* c2_data_matrix,
        const size_t c2_shift,
        const float  BonA_data_scalar,
        const float* BonA_data_matrix,
        const size_t BonA_shift,
        const float  rho0_data_scalar,
        const float* rho0_data_matrix,
        const size_t rho0_shift
        )
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < TotalElementCount; i += istride){

        float c2_data;
        float BonA_data;
        float rho0_data;

        if (c2_shift == 0){
            c2_data = c2_data_scalar;
        }
        else {
            c2_data = c2_data_matrix[i];
        }
        if (BonA_shift == 0){
            BonA_data = BonA_data_scalar;
        }
        else {
            BonA_data = BonA_data_matrix[i];
        }
        if (rho0_shift == 0){
            rho0_data = rho0_data_scalar;
        }
        else {
            rho0_data = rho0_data_matrix[i];
        }

        const float sum_rho = rhox_data[i] + rhoy_data[i] + rhoz_data[i];
        p_data[i] = c2_data *(
                sum_rho +
                (BonA_data * (sum_rho* sum_rho) /
                 (2.0f* rho0_data))
                );
    }
}

void TCUDAImplementations::Sum_new_p_nonlinear_lossless(
        size_t       TotalElementCount,
        TRealMatrix& p,
        const float* rhox_data,
        const float* rhoy_data,
        const float* rhoz_data,
        const float  c2_data_scalar,
        const float* c2_data_matrix,
        const size_t c2_shift,
        const float  BonA_data_scalar,
        const float* BonA_data_matrix,
        const size_t BonA_shift,
        const float  rho0_data_scalar,
        const float* rho0_data_matrix,
        const size_t rho0_shift)
{
    float * p_data  = p.GetRawDeviceData();

    CudaSum_new_p_nonlinear_lossless
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
            (TotalElementCount,
             p_data,
             rhox_data,
             rhoy_data,
             rhoz_data,
             c2_data_scalar,
             c2_data_matrix,
             c2_shift,
             BonA_data_scalar,
             BonA_data_matrix,
             BonA_shift,
             rho0_data_scalar,
             rho0_data_matrix,
             rho0_shift);
}

__global__ void CudaCalculate_SumRho_SumRhoDu(float* Sum_rhoxyz_data,
        float* Sum_rho0_du_data,
        const float* rhox_data,
        const float* rhoy_data,
        const float* rhoz_data,
        const float* dux_data,
        const float* duy_data,
        const float* duz_data,
        const float* rho0_data,
        const float rho0_data_el,
        const size_t TotalElementCount,
        bool rho0_scalar_flag)
{

    size_t istart = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (size_t i = istart; i < TotalElementCount; i += istride){
        Sum_rhoxyz_data[i] = rhox_data[i] + rhoy_data[i] + rhoz_data[i];
    }

    if (rho0_scalar_flag){ // scalar
        for (size_t i = istart; i < TotalElementCount; i += istride){
            Sum_rho0_du_data[i] = rho0_data_el *
                (dux_data[i] + duy_data[i] + duz_data[i]);
        }
    }
    else
    { // matrix
        for (size_t i = istart; i < TotalElementCount; i += istride){
            Sum_rho0_du_data[i] = rho0_data[i] *
                (dux_data[i] + duy_data[i] + duz_data[i]);
        }
    }
}

void TCUDAImplementations::Calculate_SumRho_SumRhoDu(
        TRealMatrix& Sum_rhoxyz,
        TRealMatrix& Sum_rho0_du,
        TRealMatrix& rhox,
        TRealMatrix& rhoy,
        TRealMatrix& rhoz,
        TRealMatrix& duxdx,
        TRealMatrix& duydy,
        TRealMatrix& duzdz,
        const float* rho0_data,
        const float  rho0_data_el,
        const size_t TotalElementCount,
        const bool   rho0_scalar_flag)
{
    const float * rhox_data = rhox.GetRawDeviceData();
    const float * rhoy_data = rhoy.GetRawDeviceData();
    const float * rhoz_data = rhoz.GetRawDeviceData();

    const float * dux_data = duxdx.GetRawDeviceData();
    const float * duy_data = duydy.GetRawDeviceData();
    const float * duz_data = duzdz.GetRawDeviceData();

    float * Sum_rhoxyz_data  = Sum_rhoxyz.GetRawDeviceData();
    float * Sum_rho0_du_data = Sum_rho0_du.GetRawDeviceData();

    CudaCalculate_SumRho_SumRhoDu
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
            (Sum_rhoxyz_data,
             Sum_rho0_du_data,
             rhox_data,
             rhoy_data,
             rhoz_data,
             dux_data,
             duy_data,
             duz_data,
             rho0_data,
             rho0_data_el,
             TotalElementCount,
             rho0_scalar_flag);
}

__global__ void CudaSum_Subterms_linear(const float* Absorb_tau_data,
        const float* Absorb_eta_data,
        const float* Sum_rhoxyz_data,
              float* p_data,
        const size_t total_element_count,
        const size_t c2_shift,
        const size_t tau_eta_shift,
        const float  tau_data_scalar,
        const float* tau_data_matrix,
        const float  eta_data_scalar,
        const float* eta_data_matrix,
        const float  c2_data_scalar,
        const float* c2_data_matrix)
{

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    const float divider = 1.0f / (float) total_element_count;

    for (; i < total_element_count; i += istride){

        float c2_data;
        float tau_data;
        float eta_data;

        //if c2 is a scalar use that element else use correct matrix element
        if(c2_shift == 0){
            c2_data = c2_data_scalar;
        }
        else{
            c2_data = c2_data_matrix[i];
        }
        //same as above but if tau is scalar then so too is eta
        if(tau_eta_shift == 0){
            tau_data = tau_data_scalar;
            eta_data = eta_data_scalar;
        }
        else{
            tau_data = tau_data_matrix[i];
            eta_data = eta_data_matrix[i];
        }

        p_data[i] = c2_data *(
                Sum_rhoxyz_data[i] +
                (divider * (Absorb_tau_data[i] * tau_data -
                            Absorb_eta_data[i] * eta_data)));
    }
}

void TCUDAImplementations::Sum_Subterms_linear(TRealMatrix& Absorb_tau_temp,
        TRealMatrix& Absorb_eta_temp,
        TRealMatrix& Sum_rhoxyz,
        TRealMatrix& p,
        const size_t total_element_count,
        size_t       c2_shift,
        size_t       tau_eta_shift,
        const float  tau_data_scalar,
        const float* tau_data_matrix,
        const float  eta_data_scalar,
        const float* eta_data_matrix,
        const float  c2_data_scalar,
        const float* c2_data_matrix)
{
    const float *  Absorb_tau_data = Absorb_tau_temp.GetRawDeviceData();
    const float *  Absorb_eta_data = Absorb_eta_temp.GetRawDeviceData();

    const float * Sum_rhoxyz_data = Sum_rhoxyz.GetRawDeviceData();
    float * p_data  = p.GetRawDeviceData();

    CudaSum_Subterms_linear
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
            (Absorb_tau_data,
             Absorb_eta_data,
             Sum_rhoxyz_data,
             p_data,
             total_element_count,
             c2_shift,
             tau_eta_shift,
             tau_data_scalar,
             tau_data_matrix,
             eta_data_scalar,
             eta_data_matrix,
             c2_data_scalar,
             c2_data_matrix);
}

__global__ void CudaSum_new_p_linear_lossless_scalar(float* p_data,
        const float* rhox_data,
        const float* rhoy_data,
        const float* rhoz_data,
        const size_t total_element_count,
        const float  c2_element)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < total_element_count; i += istride){
        p_data[i] = c2_element * ( rhox_data[i] + rhoy_data[i] + rhoz_data[i]);
    }
}

void TCUDAImplementations::Sum_new_p_linear_lossless_scalar(
        TRealMatrix& p,
        TRealMatrix& rhox,
        TRealMatrix& rhoy,
        TRealMatrix& rhoz,
        const size_t total_element_count,
        const float c2_element
        )
{
    float * p_data = p.GetRawDeviceData();
    const float * rhox_data = rhox.GetRawDeviceData();
    const float * rhoy_data = rhoy.GetRawDeviceData();
    const float * rhoz_data = rhoz.GetRawDeviceData();

    CudaSum_new_p_linear_lossless_scalar
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
        (p_data,
         rhox_data,
         rhoy_data,
         rhoz_data,
         total_element_count,
         c2_element);
}

__global__ void CudaSum_new_p_linear_lossless_matrix(float* p_data,
        const float* rhox_data,
        const float* rhoy_data,
        const float* rhoz_data,
        const size_t total_element_count,
        const float* c2_data)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < total_element_count; i += istride){
        p_data[i] = c2_data[i] * ( rhox_data[i] + rhoy_data[i] + rhoz_data[i]);
    }
}

void TCUDAImplementations::Sum_new_p_linear_lossless_matrix(TRealMatrix& p,
        TRealMatrix& rhox,
        TRealMatrix& rhoy,
        TRealMatrix& rhoz,
        const size_t total_element_count,
        TRealMatrix& c2)
{
          float* p_data = p.GetRawDeviceData();
    const float* rhox_data = rhox.GetRawDeviceData();
    const float* rhoy_data = rhoy.GetRawDeviceData();
    const float* rhoz_data = rhoz.GetRawDeviceData();
    const float* c2_data = c2.GetRawDeviceData();

    CudaSum_new_p_linear_lossless_matrix
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
            (p_data,
             rhox_data,
             rhoy_data,
             rhoz_data,
             total_element_count,
             c2_data);
}

__global__ void CudaStoreSensorData_store_p_max(const float* p_data,
        float* p_max,
        const size_t* index,
        const size_t sensor_size)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < sensor_size; i += istride){
        if (p_max[i] < p_data[index[i]]) p_max[i] = p_data[index[i]];
    }
}

void TCUDAImplementations::StoreSensorData_store_p_max(TRealMatrix& p,
        TRealMatrix& p_sensor_max,
        TIndexMatrix& sensor_mask_index
        )
{
    const float* p_data = p.GetRawDeviceData();
    float* p_max = p_sensor_max.GetRawDeviceData();
    const size_t* index = sensor_mask_index.GetRawDeviceData();
    const size_t sensor_size = sensor_mask_index.GetTotalElementCount();

    CudaStoreSensorData_store_p_max
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
        (p_data,
         p_max,
         index,
         sensor_size);
}

__global__ void CudaStoreSensorData_store_p_rms(const float* p_data,
        float* p_rms,
        const size_t* index,
        const size_t sensor_size)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < sensor_size; i += istride){
        p_rms[i] += (p_data[index[i]] * p_data[index[i]]);
    }
}

void TCUDAImplementations::StoreSensorData_store_p_rms(TRealMatrix& p,
        TRealMatrix& p_sensor_rms,
        TIndexMatrix& sensor_mask_index
        )
{
    const float* p_data = p.GetRawDeviceData();
    float* p_rms = p_sensor_rms.GetRawDeviceData();
    const size_t* index = sensor_mask_index.GetRawDeviceData();
    const size_t sensor_size = sensor_mask_index.GetTotalElementCount();

    CudaStoreSensorData_store_p_rms
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
        (p_data,
         p_rms,
         index,
         sensor_size);
}

__global__ void CudaStoreSensorData_store_u_max(const float* ux,
        const float* uy,
        const float* uz,
        float* ux_max,
        float* uy_max,
        float* uz_max,
        const size_t* index,
        const size_t sensor_size)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < sensor_size; i += istride){
        if (ux_max[i] < ux[index[i]]) ux_max[i] = ux[index[i]];
        if (uy_max[i] < uy[index[i]]) uy_max[i] = uy[index[i]];
        if (uz_max[i] < uz[index[i]]) uz_max[i] = uz[index[i]];
    }
}

void TCUDAImplementations::StoreSensorData_store_u_max(TRealMatrix& ux_sgx,
        TRealMatrix& uy_sgy,
        TRealMatrix& uz_sgz,
        TRealMatrix& ux_sensor_max,
        TRealMatrix& uy_sensor_max,
        TRealMatrix& uz_sensor_max,
        TIndexMatrix& sensor_mask_index
        )
{
    const float* ux = ux_sgx.GetRawDeviceData();
    const float* uy = uy_sgy.GetRawDeviceData();
    const float* uz = uz_sgz.GetRawDeviceData();
    float* ux_max = ux_sensor_max.GetRawDeviceData();
    float* uy_max = uy_sensor_max.GetRawDeviceData();
    float* uz_max = uz_sensor_max.GetRawDeviceData();
    const size_t* index = sensor_mask_index.GetRawDeviceData();
    const size_t sensor_size = sensor_mask_index.GetTotalElementCount();

    CudaStoreSensorData_store_u_max
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
        (ux,
         uy,
         uz,
         ux_max,
         uy_max,
         uz_max,
         index,
         sensor_size);
}

__global__ void CudaStoreSensorData_store_u_rms(const float* ux,
        const float* uy,
        const float* uz,
        float* ux_rms,
        float* uy_rms,
        float* uz_rms,
        const size_t* index,
        const size_t sensor_size)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < sensor_size; i += istride){
        ux_rms[i] += (ux[index[i]] * ux[index[i]]);
        uy_rms[i] += (uy[index[i]] * uy[index[i]]);
        uz_rms[i] += (uz[index[i]] * uz[index[i]]);
    }
}

void TCUDAImplementations::StoreSensorData_store_u_rms(TRealMatrix& ux_sgx,
        TRealMatrix& uy_sgy,
        TRealMatrix& uz_sgz,
        TRealMatrix& ux_sensor_rms,
        TRealMatrix& uy_sensor_rms,
        TRealMatrix& uz_sensor_rms,
        TIndexMatrix& sensor_mask_index
        )
{
    const float* ux = ux_sgx.GetRawDeviceData();
    const float* uy = uy_sgy.GetRawDeviceData();
    const float* uz = uz_sgz.GetRawDeviceData();
    float* ux_rms = ux_sensor_rms.GetRawDeviceData();
    float* uy_rms = uy_sensor_rms.GetRawDeviceData();
    float* uz_rms = uz_sensor_rms.GetRawDeviceData();
    const size_t* index = sensor_mask_index.GetRawDeviceData();
    const size_t sensor_size = sensor_mask_index.GetTotalElementCount();

    CudaStoreSensorData_store_u_rms
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
        (ux,
         uy,
         uz,
         ux_rms,
         uy_rms,
         uz_rms,
         index,
         sensor_size);
}

__global__ void CudaStoreIntensityData_first_step(const size_t sensor_size,
        const size_t dims_x,
        const size_t dims_y,
        const size_t * index,
        const float* ux,
        const float* uy,
        const float* uz,
        const float* p,
        float * ux_i_1,
        float * uy_i_1,
        float * uz_i_1,
        float * p_i_1)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < sensor_size; i += istride){
        // calculate positions in the grid
        const size_t sensor_point_ind    = index[i];
        const size_t sensor_point_ind_1x = ((sensor_point_ind % dims_x) == 0) ?
            sensor_point_ind : sensor_point_ind - 1;
        const size_t sensor_point_ind_1y = (((sensor_point_ind /  dims_x) %  dims_y) == 0) ?
            sensor_point_ind : sensor_point_ind - dims_x;
        const size_t sensor_point_ind_1z = (((sensor_point_ind /  (dims_y * dims_x))) == 0) ?
            sensor_point_ind : sensor_point_ind - (dims_x * dims_y);

        //avg of actual values of u in staggered grid
        ux_i_1[i] = (ux[sensor_point_ind] + ux[sensor_point_ind_1x]) * 0.5f;
        uy_i_1[i] = (uy[sensor_point_ind] + uy[sensor_point_ind_1y]) * 0.5f;
        uz_i_1[i] = (uz[sensor_point_ind] + uz[sensor_point_ind_1z]) * 0.5f;
        p_i_1[i]   = p[sensor_point_ind];
    }
}

void TCUDAImplementations::StoreIntensityData_first_step(
        const size_t sensor_size,
        const TDimensionSizes Dims,
        const size_t * index_data,
        const float* ux_data,
        const float* uy_data,
        const float* uz_data,
        const float* p_data,
        float * ux_i_1_data,
        float * uy_i_1_data,
        float * uz_i_1_data,
        float * p_i_1_data
        )
{

    const size_t dims_x = Dims.X;
    const size_t dims_y = Dims.Y;

    CudaStoreIntensityData_first_step
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
            (sensor_size,
             dims_x,
             dims_y,
             index_data,
             ux_data,
             uy_data,
             uz_data,
             p_data,
             ux_i_1_data,
             uy_i_1_data,
             uz_i_1_data,
             p_i_1_data);
}

__global__ void CudaStoreIntensityData_other_step(const size_t sensor_size,
                                                  const size_t dims_x,
                                                  const size_t dims_y,
                                                  const size_t * index,
                                                  const float* ux,
                                                  const float* uy,
                                                  const float* uz,
                                                  const float* p,
                                                  float* ux_i_1,
                                                  float* uy_i_1,
                                                  float* uz_i_1,
                                                  float* p_i_1,
                                                  float* Ix_avg,
                                                  float* Iy_avg,
                                                  float* Iz_avg,
                                                  float* Ix_max,
                                                  float* Iy_max,
                                                  float* Iz_max,
                                                  bool store_I_avg,
                                                  bool store_I_max,
                                                  size_t Nt,
                                                  size_t start_time_index)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < sensor_size; i += istride){
        // calculate positions in the grid
        const size_t sensor_point_ind    = index[i];
        const size_t sensor_point_ind_1x = ((sensor_point_ind % dims_x) == 0) ?
            sensor_point_ind : sensor_point_ind - 1;
        const size_t sensor_point_ind_1y =
            (((sensor_point_ind /  dims_x) % dims_y) == 0) ?
                sensor_point_ind : sensor_point_ind - dims_x;
        const size_t sensor_point_ind_1z =
            (((sensor_point_ind /  (dims_y * dims_x))) == 0) ?
                sensor_point_ind : sensor_point_ind - (dims_x * dims_y);

        //avg of actual values of u in staggered grid
        const float ux_act =
            (ux[sensor_point_ind] + ux[sensor_point_ind_1x]) * 0.5f;
        const float uy_act =
            (uy[sensor_point_ind] + uy[sensor_point_ind_1y]) * 0.5f;
        const float uz_act =
            (uz[sensor_point_ind] + uz[sensor_point_ind_1z]) * 0.5f;
        const float p_data = p_i_1[i];

        // calculate actual intensity based on
        // p(n-1) * 1/4[ u(i-1)(n-1) + u(i)(n-1) + u(i-1)(n) + u(i)(n)
        const float Ix = p_data * ((ux_act + ux_i_1[i]) * 0.5f);
        const float Iy = p_data * ((uy_act + uy_i_1[i]) * 0.5f);
        const float Iz = p_data * ((uz_act + uz_i_1[i]) * 0.5f);

        ux_i_1[i] = ux_act;
        uy_i_1[i] = uy_act;
        uz_i_1[i] = uz_act;
        p_i_1[i]  = p[sensor_point_ind];

        const float Divider = 1.0f / (float) (Nt - start_time_index);

        // easily predictable...
        if (store_I_max) {
            if (Ix_max[i] < Ix) Ix_max[i] = Ix;
            if (Iy_max[i] < Iy) Iy_max[i] = Iy;
            if (Iz_max[i] < Iz) Iz_max[i] = Iz;
        }

        // easily predictable...
        if (store_I_avg) {
            Ix_avg[i] += (Ix * Divider);
            Iy_avg[i] += (Iy * Divider);
            Iz_avg[i] += (Iz * Divider);
        }
    }
}

void TCUDAImplementations::StoreIntensityData_other_step(
        const size_t sensor_size,
        const TDimensionSizes Dims,
        const size_t * index_data,
        const float* ux_data,
        const float* uy_data,
        const float* uz_data,
        const float* p_data,
        float* ux_i_1_data,
        float* uy_i_1_data,
        float* uz_i_1_data,
        float* p_i_1_data,
        float* Ix_avg_data,
        float* Iy_avg_data,
        float* Iz_avg_data,
        float* Ix_max_data,
        float* Iy_max_data,
        float* Iz_max_data,
        bool store_I_avg,
        bool store_I_max,
        size_t Nt,
        size_t start_time_index)
{

    const size_t dims_x = Dims.X;
    const size_t dims_y = Dims.Y;

    CudaStoreIntensityData_other_step
        <<< CUDATuner->GetNumberOfBlocksFor1D(),
            CUDATuner->GetNumberOfThreadsFor1D() >>>
            (sensor_size,
             dims_x,
             dims_y,
             index_data,
             ux_data,
             uy_data,
             uz_data,
             p_data,
             ux_i_1_data,
             uy_i_1_data,
             uz_i_1_data,
             p_i_1_data,
             Ix_avg_data,
             Iy_avg_data,
             Iz_avg_data,
             Ix_max_data,
             Iy_max_data,
             Iz_max_data,
             store_I_avg,
             store_I_max,
             Nt,
             start_time_index);
}

