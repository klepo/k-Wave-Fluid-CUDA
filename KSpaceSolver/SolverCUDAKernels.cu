/**
 * @file        SolverCUDAKernels.cu
 *
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the all cuda kernels for the GPU implementation
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        11 March    2013, 13:10 (created) \n
 *              10 August   2017, 15:22 (revised)
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

//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Constants -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Variables -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


/**
 * @var      cudaDeviceConstants
 * @brief    This variable holds basic simulation constants for GPU.
 * @details  This variable holds necessary simulation constants in the cuda GPU memory.
 *           The variable is defined in CudaDeviceConstants.cu
 */
extern __constant__ CudaDeviceConstants cudaDeviceConstants;


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Global methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief  Get block size for 1D kernels.
 * @return 1D block size.
 */
inline int getSolverBlockSize1D()
{
  return Parameters::getInstance().getCudaParameters().getSolverBlockSize1D();
};// end of getSolverBlockSize1D
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Get grid size for 1D kernels.
 * @return 1D grid size
 */
inline int getSolverGridSize1D()
{
  return Parameters::getInstance().getCudaParameters().getSolverGridSize1D();
};// end of getSolverGridSize1D
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Get block size for the transposition kernels.
 * @return 3D grid size.
 */
inline dim3 getSolverTransposeBlockSize()
{
  return Parameters::getInstance().getCudaParameters().getSolverTransposeBlockSize();
};//end of getSolverTransposeBlockSize()
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Get grid size for complex 3D kernels
 * @return 3D grid size
 */
inline dim3 GetSolverTransposeGirdSize()
{
  return Parameters::getInstance().getCudaParameters().getSolverTransposeGirdSize();
};// end of getSolverTransposeGirdSize()
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public routines --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Kernel to find out the version of the code.
 * The list of GPUs can be found at https://en.wikipedia.org/wiki/CUDA
 *
 * @param [out] cudaCodeVersion
 */
__global__ void cudaGetCudaCodeVersion(int* cudaCodeVersion)
{
  *cudaCodeVersion = -1;

  // Read __CUDA_ARCH__ only in actual kernel compilation pass.
  // NVCC does some more passes, where it isn't defined.
  #ifdef __CUDA_ARCH__
    *cudaCodeVersion = (__CUDA_ARCH__ / 10);
  #endif
}// end of cudaGetCudaCodeVersion
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get the CUDA architecture the code was compiled with.
 */
int SolverCudaKernels::getCudaCodeVersion()
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
    cudaGetCudaCodeVersion<<<1,1>>>(dCudaCodeVersion);
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
//----------------------------------------------------------------------------------------------------------------------


/**
 * Cuda kernel to calculate new particle velocity. Heterogeneous case, uniform gird.
 *
 * <b> Matlab code: </b>
 *
 * \verbatim
    ux_sgx = bsxfun(@times, pml_x_sgx, bsxfun(@times, pml_x_sgx, ux_sgx) - dt .* rho0_sgx_inv .* real(ifftX));
    uy_sgy = bsxfun(@times, pml_y_sgy, bsxfun(@times, pml_y_sgy, uy_sgy) - dt .* rho0_sgy_inv .* real(ifftY));
    uz_sgz = bsxfun(@times, pml_z_sgz, bsxfun(@times, pml_z_sgz, uz_sgz) - dt .* rho0_sgz_inv .* real(ifftZ));
\endverbatim
 *
 * @param [in, out] uxSgx     - Acoustic velocity on staggered grid in x direction.
 * @param [in, out] uySgy     - Acoustic velocity on staggered grid in y direction.
 * @param [in, out] uzSgz     - Acoustic velocity on staggered grid in z direction.
 * @param [in]      ifftX     - ifftn( bsxfun(\@times, ddx_k_shift_pos, kappa .* p_k))
 * @param [in]      ifftY     - ifftn( bsxfun(\@times, ddy_k_shift_pos, kappa .* p_k))
 * @param [in]      ifftZ     - ifftn( bsxfun(\@times, ddz_k_shift_pos, kappa .* p_k))
 * @param [in]      dtRho0Sgx - Acoustic density on staggered grid in x direction.
 * @param [in]      dtRho0Sgy - Acoustic density on staggered grid in y direction.
 * @param [in]      dtRho0Sgz - Acoustic density on staggered grid in z direction.
 * @param [in]      pmlX      - Perfectly matched layer in x direction.
 * @param [in]      pmlY      - Perfectly matched layer in y direction.
 * @param [in]      pmlZ      - Perfectly matched layer in z direction.
 */
__global__ void cudaComputeVelocity(float*       uxSgx,
                                    float*       uySgy,
                                    float*       uzSgz,
                                    const float* ifftX,
                                    const float* ifftY,
                                    const float* ifftZ,
                                    const float* dtRho0Sgx,
                                    const float* dtRho0Sgy,
                                    const float* dtRho0Sgz,
                                    const float* pmlX,
                                    const float* pmlY,
                                    const float* pmlZ)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getReal3DCoords(i);

    const float eIfftX = cudaDeviceConstants.fftDivider * ifftX[i] * dtRho0Sgx[i];
    const float eIfftY = cudaDeviceConstants.fftDivider * ifftY[i] * dtRho0Sgy[i];
    const float eIfftZ = cudaDeviceConstants.fftDivider * ifftZ[i] * dtRho0Sgz[i];

    const float ePmlX = pmlX[coords.x];
    const float ePmlY = pmlY[coords.y];
    const float ePmlZ = pmlZ[coords.z];

    uxSgx[i] = (uxSgx[i] * ePmlX - eIfftX) * ePmlX;
    uySgy[i] = (uySgy[i] * ePmlY - eIfftY) * ePmlY;
    uzSgz[i] = (uzSgz[i] * ePmlZ - eIfftZ) * ePmlZ;
  }
}// end of cudaComputeVelocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to the cuda kernel computing new values for particle velocity.  Default (heterogeneous case).
 */
void SolverCudaKernels::computeVelocity(RealMatrix&       uxSgx,
                                        RealMatrix&       uySgy,
                                        RealMatrix&       uzSgz,
                                        const RealMatrix& ifftX,
                                        const RealMatrix& ifftY,
                                        const RealMatrix& ifftZ,
                                        const RealMatrix& dtRho0Sgx,
                                        const RealMatrix& dtRho0Sgy,
                                        const RealMatrix& dtRho0Sgz,
                                        const RealMatrix& pmlX,
                                        const RealMatrix& pmlY,
                                        const RealMatrix& pmlZ)
  {
    cudaComputeVelocity<<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                       (uxSgx.getDeviceData(),
                        uySgy.getDeviceData(),
                        uzSgz.getDeviceData(),
                        ifftX.getDeviceData(),
                        ifftY.getDeviceData(),
                        ifftZ.getDeviceData(),
                        dtRho0Sgx.getDeviceData(),
                        dtRho0Sgy.getDeviceData(),
                        dtRho0Sgz.getDeviceData(),
                        pmlX.getDeviceData(),
                        pmlY.getDeviceData(),
                        pmlZ.getDeviceData());

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of ComputeVelocity
//----------------------------------------------------------------------------------------------------------------------



/**
 * Cuda kernel to calculate new particle velocity. Homogeneous case, uniform gird.
 *
 * <b> Matlab code: </b>
 *
 * \verbatim
    ux_sgx = bsxfun(@times, pml_x_sgx, bsxfun(@times, pml_x_sgx, ux_sgx) - dt .* rho0_sgx_inv .* real(ifftX));
    uy_sgy = bsxfun(@times, pml_y_sgy, bsxfun(@times, pml_y_sgy, uy_sgy) - dt .* rho0_sgy_inv .* real(ifftY));
    uz_sgz = bsxfun(@times, pml_z_sgz, bsxfun(@times, pml_z_sgz, uz_sgz) - dt .* rho0_sgz_inv .* real(ifftZ));
\endverbatim
 *
 * @param [in, out] uxSgx - Acoustic velocity on staggered grid in x direction.
 * @param [in, out] uySgy - Acoustic velocity on staggered grid in y direction.
 * @param [in, out] uzSgz - Acoustic velocity on staggered grid in z direction.
 * @param [in]      ifftX - ifftn( bsxfun(\@times, ddx_k_shift_pos, kappa .* p_k))
 * @param [in]      ifftY - ifftn( bsxfun(\@times, ddy_k_shift_pos, kappa .* p_k))
 * @param [in]      ifftZ - ifftn( bsxfun(\@times, ddz_k_shift_pos, kappa .* p_k))
 * @param [in]      pmlX  - Perfectly matched layer in x direction.
 * @param [in]      pmlY  - Perfectly matched layer in y direction.
 * @param [in]      pmlZ  - Perfectly matched layer in z direction.
 */
__global__ void cudaComputeVelocityHomogeneousUniform(float*       uxSgx,
                                                      float*       uySgy,
                                                      float*       uzSgz,
                                                      const float* ifftX,
                                                      const float* ifftY,
                                                      const float* ifftZ,
                                                      const float* pmlX,
                                                      const float* pmlY,
                                                      const float* pmlZ)
{
  const float dividerX = cudaDeviceConstants.dtRho0Sgx * cudaDeviceConstants.fftDivider;
  const float dividerY = cudaDeviceConstants.dtRho0Sgy * cudaDeviceConstants.fftDivider;
  const float dividerZ = cudaDeviceConstants.dtRho0Sgz * cudaDeviceConstants.fftDivider;

  for(auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getReal3DCoords(i);

    const float ePmlX = pmlX[coords.x];
    const float ePmlY = pmlY[coords.y];
    const float ePmlZ = pmlZ[coords.z];

    uxSgx[i] = (uxSgx[i] * ePmlX - dividerX * ifftX[i]) * ePmlX;
    uySgy[i] = (uySgy[i] * ePmlY - dividerY * ifftY[i]) * ePmlY;
    uzSgz[i] = (uzSgz[i] * ePmlZ - dividerZ * ifftZ[i]) * ePmlZ;
  }// for
}// end of cudaComputeVelocityHomogeneousUniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute acoustic velocity for homogeneous medium and a uniform grid.
 */
void SolverCudaKernels::computeVelocityHomogeneousUniform(RealMatrix&       uxSgx,
                                                          RealMatrix&       uySgy,
                                                          RealMatrix&       uzSgz,
                                                          const RealMatrix& ifftX,
                                                          const RealMatrix& ifftY,
                                                          const RealMatrix& ifftZ,
                                                          const RealMatrix& pmlX,
                                                          const RealMatrix& pmlY,
                                                          const RealMatrix& pmlZ)
{
  cudaComputeVelocityHomogeneousUniform<<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                       (uxSgx.getDeviceData(),
                                        uySgy.getDeviceData(),
                                        uzSgz.getDeviceData(),
                                        ifftX.getDeviceData(),
                                        ifftY.getDeviceData(),
                                        ifftZ.getDeviceData(),
                                        pmlX.getDeviceData(),
                                        pmlY.getDeviceData(),
                                        pmlZ.getDeviceData());
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeVelocityHomogeneousUniform
//----------------------------------------------------------------------------------------------------------------------


/**
 * Cuda kernel to calculate uxSgx, uySgy and uzSgz. This is the homogeneous medium and a non-uniform grid.
 *
 * * <b> Matlab code: </b>
 *
 * \verbatim
    ux_sgx = bsxfun(@times, pml_x_sgx, ...
               bsxfun(@times, pml_x_sgx, ux_sgx) - dt .* rho0_sgx_inv .* real(ifftX) .* dxudxn_sgx);
    uy_sgy = bsxfun(@times, pml_y_sgy, ...
               bsxfun(@times, pml_y_sgy, uy_sgy) - dt .* rho0_sgy_inv .* real(ifftY) .* dyudyn_sgy);
    uz_sgz = bsxfun(@times, pml_z_sgz, ...
               bsxfun(@times, pml_z_sgz, uz_sgz) - dt .* rho0_sgz_inv .* real(ifftZ) .* dzudzn_sgz);
\endverbatim
 *
 * @param [in,out] uxSgx     - Acoustic velocity on staggered grid in x direction.
 * @param [in,out] uySgy     - Acoustic velocity on staggered grid in y direction.
 * @param [in,out] uzSgz     - Acoustic velocity on staggered grid in z direction.
 * @param [in]      ifftX    - ifftn( bsxfun(\@times, ddx_k_shift_pos, kappa .* p_k))
 * @param [in]      ifftY    - ifftn( bsxfun(\@times, ddy_k_shift_pos, kappa .* p_k))
 * @param [in]      ifftZ    - ifftn( bsxfun(\@times, ddz_k_shift_pos, kappa .* p_k))
 * @param [in]     dxudxnSgx - Non uniform grid shift in x direction.
 * @param [in]     dyudynSgy - Non uniform grid shift in y direction.
 * @param [in]     dzudznSgz - Non uniform grid shift in z direction.
 * @param [in]     pmlX      - Perfectly matched layer in x direction.
 * @param [in]     pmlY      - Perfectly matched layer in y direction.
 * @param [in]     pmlZ      - Perfectly matched layer in z direction.
 */
__global__ void cudaComputeVelocityHomogeneousNonuniform(float*       uxSgx,
                                                         float*       uySgy,
                                                         float*       uzSgz,
                                                         const float* ifftX,
                                                         const float* ifftY,
                                                         const float* ifftZ,
                                                         const float* dxudxnSgx,
                                                         const float* dyudynSgy,
                                                         const float* dzudznSgz,
                                                         const float* pmlX,
                                                         const float* pmlY,
                                                         const float* pmlZ)
{
  const float DividerX = cudaDeviceConstants.dtRho0Sgx * cudaDeviceConstants.fftDivider;
  const float DividerY = cudaDeviceConstants.dtRho0Sgy * cudaDeviceConstants.fftDivider;;
  const float DividerZ = cudaDeviceConstants.dtRho0Sgz * cudaDeviceConstants.fftDivider;

  for(auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getReal3DCoords(i);

    const float ePmlX = pmlX[coords.x];
    const float ePmlY = pmlY[coords.y];
    const float ePmlZ = pmlZ[coords.z];

    const float eIfftX = DividerX * dxudxnSgx[coords.x] * ifftX[i];
    const float eIfftY = DividerY * dyudynSgy[coords.y] * ifftY[i];
    const float eIfftZ = DividerZ * dzudznSgz[coords.z] * ifftZ[i];

    uxSgx[i] = (uxSgx[i] * ePmlX - eIfftX) * ePmlX;
    uySgy[i] = (uySgy[i] * ePmlY - eIfftY) * ePmlY;
    uzSgz[i] = (uzSgz[i] * ePmlZ - eIfftZ) * ePmlZ;
  }// for
}// end of cudaComputeVelocityHomogeneouosNonuniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to  calculate uxSgx, uySgy and uzSgz.
 * This is the homogeneous medium and a non-uniform grid.
 */
void SolverCudaKernels::computeVelocityHomogeneousNonuniform(RealMatrix&       uxSgx,
                                                             RealMatrix&       uySgy,
                                                             RealMatrix&       uzSgz,
                                                             const RealMatrix& ifftX,
                                                             const RealMatrix& ifftY,
                                                             const RealMatrix& ifftZ,
                                                             const RealMatrix& dxudxnSgx,
                                                             const RealMatrix& dyudynSgy,
                                                             const RealMatrix& dzudznSgz,
                                                             const RealMatrix& pmlX,
                                                             const RealMatrix& pmlY,
                                                             const RealMatrix& pmlZ)
{
  cudaComputeVelocityHomogeneousNonuniform<<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                     (uxSgx.getDeviceData(),
                                      uySgy.getDeviceData(),
                                      uzSgz.getDeviceData(),
                                      ifftX.getDeviceData(),
                                      ifftY.getDeviceData(),
                                      ifftZ.getDeviceData(),
                                      dxudxnSgx.getDeviceData(),
                                      dyudynSgy.getDeviceData(),
                                      dzudznSgz.getDeviceData(),
                                      pmlX.getDeviceData(),
                                      pmlY.getDeviceData(),
                                      pmlZ.getDeviceData());

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeVelocityHomogeneousNonuniform
//----------------------------------------------------------------------------------------------------------------------


/**
 * Cuda kernel adding transducer data to uxSgx.
 *
 * @param [in, out] uxSgx                 - Here we add the signal.
 * @param [in]      velocitySourceIndex   - Where to add the signal (source).
 * @param [in]      transducerSourceInput - Transducer signal.
 * @param [in, out] delayMask             - Delay mask to push the signal in the domain (incremented per invocation).

 */
__global__ void cudaAddTransducerSource(float*        uxSgx,
                                        const size_t* velocitySourceIndex,
                                        const float*  transducerSourceInput,
                                        size_t*       delayMask)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.velocitySourceSize; i += getStride())
  {
    uxSgx[velocitySourceIndex[i]] += transducerSourceInput[delayMask[i]];
    delayMask[i]++;
  }
}// end of cudaAddTransducerSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel adding transducer data to uxSgx.
 */
void SolverCudaKernels::addTransducerSource(RealMatrix&        uxSgx,
                                            const IndexMatrix& velocitySourceIndex,
                                            const RealMatrix&  transducerSourceInput,
                                            IndexMatrix&       delayMask)
{
  // cuda only supports 32bits anyway
  const int sourceSize = static_cast<int>(velocitySourceIndex.size());

  // Grid size is calculated based on the source size
  const int gridSize  = (sourceSize < (getSolverGridSize1D() *  getSolverBlockSize1D()))
                        ? (sourceSize  + getSolverBlockSize1D() - 1 ) / getSolverBlockSize1D()
                        : getSolverGridSize1D();

  cudaAddTransducerSource<<<gridSize, getSolverBlockSize1D()>>>
                         (uxSgx.getDeviceData(),
                          velocitySourceIndex.getDeviceData(),
                          transducerSourceInput.getDeviceData(),
                          delayMask.getDeviceData());
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of AddTransducerSource
//----------------------------------------------------------------------------------------------------------------------


/**
 * Cuda kernel to add in velocity source terms.
 *
 * @param [in, out] velocity            - velocity matrix to update.
 * @param [in]      velocitySourceInput - Source input to add.
 * @param [in]      velocitySourceIndex - Index matrix.
 * @param [in]      timeIndex           - Actual time step.
 */
__global__ void cudaAddVelocitySource(float*        velocity,
                                      const float*  velocitySourceInput,
                                      const size_t* velocitySourceIndex,
                                      const size_t  timeIndex)
{
  // Set 1D or 2D step for source
  auto index2D = (cudaDeviceConstants.velocitySourceMany == 0)
                 ? timeIndex : timeIndex * cudaDeviceConstants.velocitySourceSize;

  if (cudaDeviceConstants.velocitySourceMode == 0)
  {
    for (auto i = getIndex(); i < cudaDeviceConstants.velocitySourceSize; i += getStride())
    {
      velocity[velocitySourceIndex[i]] = (cudaDeviceConstants.velocitySourceMany == 0)
                                         ? velocitySourceInput[index2D] : velocitySourceInput[index2D + i];
    }// for
  }// end of Dirichlet

  if (cudaDeviceConstants.velocitySourceMode == 1)
  {
    for (auto i  = getIndex(); i < cudaDeviceConstants.velocitySourceSize; i += getStride())
    {
      velocity[velocitySourceIndex[i]] += (cudaDeviceConstants.velocitySourceMany == 0)
                                          ? velocitySourceInput[index2D] : velocitySourceInput[index2D + i];
    }
  } // end additive
}// end of cudaAddVelocitySource
//----------------------------------------------------------------------------------------------------------------------


/**
 * Interface to Cuda kernel adding in velocity source terms.
 */
void SolverCudaKernels::addVelocitySource(RealMatrix&        velocity,
                                          const RealMatrix&  velocitySourceInput,
                                          const IndexMatrix& velocitySourceIndex,
                                          const size_t       timeIndex)
{
  const int sourceSize = static_cast<int>(velocitySourceIndex.size());

  // Grid size is calculated based on the source size
  // for small sources, a custom number of thread blocks is created,
  // otherwise, a standard number is used

  const int gridSize = (sourceSize < (getSolverGridSize1D() *  getSolverBlockSize1D()))
                       ? (sourceSize  + getSolverBlockSize1D() - 1 ) / getSolverBlockSize1D()
                       :  getSolverGridSize1D();

  cudaAddVelocitySource<<< gridSize, getSolverBlockSize1D()>>>
                       (velocity.getDeviceData(),
                        velocitySourceInput.getDeviceData(),
                        velocitySourceIndex.getDeviceData(),
                        timeIndex);

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of addVelocitySource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to add pressure source to acoustic density.
 *
 * @param [out] rhoX                - Acoustic density.
 * @param [out] rhoY                - Acoustic density.
 * @param [out] rhoZ                - Acoustic density.
 * @param [in]  pressureSourceInput - Source input to add.
 * @param [in]  pressureSourceIndex - Index matrix with source.
 * @param [in]  timeIndex           - Actual time step.

 */
__global__ void cudaAddPressureSource(float*        rhoX,
                                      float*        rhoY,
                                      float*        rhoZ,
                                      const float*  pressureSourceInput,
                                      const size_t* pressureSourceIndex,
                                      const size_t  timeIndex)
{
  // Set 1D or 2D step for source
  auto index2D = (cudaDeviceConstants.presureSourceMany == 0)
                 ? timeIndex : timeIndex * cudaDeviceConstants.presureSourceSize;

  if (cudaDeviceConstants.presureSourceMode == 0)
  {
    if (cudaDeviceConstants.presureSourceMany == 0)
    { // single signal
      for (auto i = getIndex(); i < cudaDeviceConstants.presureSourceSize; i += getStride())
      {
        rhoX[pressureSourceIndex[i]] = pressureSourceInput[index2D];
        rhoY[pressureSourceIndex[i]] = pressureSourceInput[index2D];
        rhoZ[pressureSourceIndex[i]] = pressureSourceInput[index2D];
      }
    }
    else
    { // multiple signals
      for (auto i = getIndex(); i < cudaDeviceConstants.presureSourceSize; i += getStride())
      {
        rhoX[pressureSourceIndex[i]] = pressureSourceInput[index2D + i];
        rhoY[pressureSourceIndex[i]] = pressureSourceInput[index2D + i];
        rhoZ[pressureSourceIndex[i]] = pressureSourceInput[index2D + i];
      }
    }
  }// end mode == 0 (Cauchy)

  if (cudaDeviceConstants.presureSourceMode == 1)
  {
    if (cudaDeviceConstants.presureSourceMany == 0)
    { // single signal
      for (auto i = getIndex(); i < cudaDeviceConstants.presureSourceSize; i += getStride())
      {
        rhoX[pressureSourceIndex[i]] += pressureSourceInput[index2D];
        rhoY[pressureSourceIndex[i]] += pressureSourceInput[index2D];
        rhoZ[pressureSourceIndex[i]] += pressureSourceInput[index2D];
      }
    }
    else
    { // multiple signals
      for (auto i = getIndex(); i < cudaDeviceConstants.presureSourceSize; i += getStride())
      {
        rhoX[pressureSourceIndex[i]] += pressureSourceInput[index2D + i];
        rhoY[pressureSourceIndex[i]] += pressureSourceInput[index2D + i];
        rhoZ[pressureSourceIndex[i]] += pressureSourceInput[index2D + i];
      }
    }
  }// end mode == 0 (Dirichlet)
}// end of cudaAddPressureSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel which adds in pressure source (to acoustic density).
 */
void SolverCudaKernels::addPressureSource(RealMatrix&        rhoX,
                                          RealMatrix&        rhoY,
                                          RealMatrix&        rhoZ,
                                          const RealMatrix&  pressureSourceInput,
                                          const IndexMatrix& pressureSourceIndex,
                                          const size_t       timeIndex)
{
  const int sourceIndex = static_cast<int>(pressureSourceIndex.size());
  // Grid size is calculated based on the source size
  const int gridSize  = (sourceIndex < (getSolverGridSize1D() *  getSolverBlockSize1D()))
                        ? (sourceIndex  + getSolverBlockSize1D() - 1 ) / getSolverBlockSize1D()
                        :  getSolverGridSize1D();

  cudaAddPressureSource<<<gridSize,getSolverBlockSize1D()>>>
                       (rhoX.getDeviceData(),
                        rhoY.getDeviceData(),
                        rhoZ.getDeviceData(),
                        pressureSourceInput.getDeviceData(),
                        pressureSourceIndex.getDeviceData(),
                        timeIndex);

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of addPressureSource
//----------------------------------------------------------------------------------------------------------------------


/**
 * Cuda kernel to add initial pressure initialPerssureSource into p, rhoX, rhoY, rhoZ.
 * c is a matrix. Heterogeneity is treated by a template
 * @tparam      isC2Scalar            - Is sound speed homogenous?
 * @param [out] p                     - New pressure field.
 * @param [out] rhoX                  - Density in x direction.
 * @param [out] rhoY                  - Density in y direction.
 * @param [out] rhoZ                  - Density in z direction.
 * @param [in]  initialPerssureSource - Initial pressure source.
 * @param [in]  c2                    - Sound speed for heterogeneous case.
 */
template<bool isC2Scalar>
__global__ void cudaAddInitialPressureSource(float*       p,
                                             float*       rhoX,
                                             float*       rhoY,
                                             float*       rhoZ,
                                             const float* initialPerssureSource,
                                             const float* c2 = nullptr)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    float tmp = p[i] = initialPerssureSource[i];

    tmp = (isC2Scalar) ? tmp / (3.0f * cudaDeviceConstants.c2): tmp / (3.0f * c2[i]);

    rhoX[i] = tmp;
    rhoY[i] = tmp;
    rhoZ[i] = tmp;
  }
}// end of cudaAddInitialPressureSource
//----------------------------------------------------------------------------------------------------------------------


/**
 * Interface for kernel to add initial pressure initialPerssureSource into p, rhoX, rhoY, rhoZ.
 */
void SolverCudaKernels::addInitialPressureSource(RealMatrix&       p,
                                                 RealMatrix&       rhoX,
                                                 RealMatrix&       rhoY,
                                                 RealMatrix&       rhoZ,
                                                 const RealMatrix& initialPerssureSource,
                                                 const bool        isC2Scalar,
                                                 const float*      c2)
{
  if (isC2Scalar)
  {
    cudaAddInitialPressureSource<true>
                                <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                (p.getDeviceData(),
                                 rhoX.getDeviceData(),
                                 rhoY.getDeviceData(),
                                 rhoZ.getDeviceData(),
                                 initialPerssureSource.getDeviceData());
  }
  else
  {
      cudaAddInitialPressureSource<false>
                                  <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                  (p.getDeviceData(),
                                   rhoX.getDeviceData(),
                                   rhoY.getDeviceData(),
                                   rhoZ.getDeviceData(),
                                   initialPerssureSource.getDeviceData(),
                                   c2);
  }
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of addInitialPressureSource
//----------------------------------------------------------------------------------------------------------------------



/**
 * Cuda kernel Compute  acoustic velocity for initial pressure problem.
 *
 * <b> Matlab code: </b>
 *
 * \verbatim
    ux_sgx = dt ./ rho0_sgx .* ux_sgx.
    uy_sgy = dt ./ rho0_sgy .* uy_sgy.
    uz_sgz = dt ./ rho0_sgz .* uz_sgz.
\endverbatim
 *
 * @tparam          isRho0Scalar - Homogenous or heterogenous medium.
 * @param [in, out] uxSgx     - Velocity matrix in x direction.
 * @param [in, out] uySgy     - Velocity matrix in y direction.
 * @param [in, out] uzSgz     - Velocity matrix in y direction.
 * @param [in]      dtRho0Sgx - Density matrix in x direction.
 * @param [in]      dtRho0Sgy - Density matrix in y direction.
 * @param [in]      dtRho0Sgz - Density matrix in z direction.
 *
 */
template <bool isRho0Scalar>
__global__  void cudaComputeInitialVelocity(float*       uxSgx,
                                            float*       uySgy,
                                            float*       uzSgz,
                                            const float* dtRho0Sgx = nullptr,
                                            const float* dtRho0Sgy = nullptr,
                                            const float* dtRho0Sgz = nullptr)

{
  if (isRho0Scalar)
  {
    const float dividerX = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgx;
    const float dividerY = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgy;
    const float dividerZ = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgz;

    for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
    {
      uxSgx[i] *= dividerX;
      uySgy[i] *= dividerY;
      uzSgz[i] *= dividerZ;
    }
  }
  else
  { // heterogeneous
    const float divider = cudaDeviceConstants.fftDivider * 0.5f;

    for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
    {
      uxSgx[i] *= dtRho0Sgx[i] * divider;
      uySgy[i] *= dtRho0Sgy[i] * divider;
      uzSgz[i] *= dtRho0Sgz[i] * divider;
    }
  }
}// end of cudaComputeInitialVelocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to compute acoustic velocity for initial pressure problem, heterogeneous medium, uniform grid.
 *
 */
void SolverCudaKernels::computeInitialVelocity(RealMatrix&       uxSgx,
                                               RealMatrix&       uySgy,
                                               RealMatrix&       uzSgz,
                                               const RealMatrix& dtRho0Sgx,
                                               const RealMatrix& dtRho0Sgy,
                                               const RealMatrix& dtRho0Sgz)
{
  cudaComputeInitialVelocity<false>
                         <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                         (uxSgx.getDeviceData(),
                          uySgy.getDeviceData(),
                          uzSgz.getDeviceData(),
                          dtRho0Sgx.getDeviceData(),
                          dtRho0Sgy.getDeviceData(),
                          dtRho0Sgz.getDeviceData());

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeInitialVelocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface Compute acoustic velocity for initial pressure problem, homogeneous medium, uniform grid.
 */
void SolverCudaKernels::computeInitialVelocity(RealMatrix& uxSgx,
                                               RealMatrix& uySgy,
                                               RealMatrix& uzSgz)
{
  cudaComputeInitialVelocity<true>
                            <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                            (uxSgx.getDeviceData(),
                             uySgy.getDeviceData(),
                             uzSgz.getDeviceData());

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeInitialVelocity
//----------------------------------------------------------------------------------------------------------------------



/**
 * Compute acoustic velocity for initial pressure problem, homogenous medium, non-uniform grid.
 * <b> Matlab code: </b>
 *
 * \verbatim
    ux_sgx = dt ./ rho0_sgx .* ux_sgx .* dxudxn_sgx.
    uy_sgy = dt ./ rho0_sgy .* uy_sgy .* dyudyn_sgy.
    uz_sgz = dt ./ rho0_sgz .* uz_sgz .* dzudzn_sgz.
\endverbatim
 *
 * @param [in, out] uxSgx - Velocity matrix in x direction.
 * @param [in, out] uySgy - Velocity matrix in y direction.
 * @param [in, out] uzSgz - Velocity matrix in y direction
 * @param [in] dxudxnSgx  - Non uniform grid shift in x direction.
 * @param [in] dyudynSgy  - Non uniform grid shift in y direction.
 * @param [in] dzudznSgz  - Non uniform grid shift in z direction.
 */
__global__ void cudaComputeInitialVelocityHomogeneousNonuniform(float*       uxSgx,
                                                                float*       uySgy,
                                                                float*       uzSgz,
                                                                const float* dxudxnSgx,
                                                                const float* dyudynSgy,
                                                                const float* dzudznSgz)
{
  const float dividerX = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgx;
  const float dividerY = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgy;
  const float dividerZ = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgz;

  for(auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getReal3DCoords(i);

    uxSgx[i] *= dividerX * dxudxnSgx[coords.x];
    uySgy[i] *= dividerY * dyudynSgy[coords.y];
    uzSgz[i] *= dividerZ * dzudznSgz[coords.z];
  }
}// end of cudaComputeInitialVelocityHomogeneousNonuniform
//----------------------------------------------------------------------------------------------------------------------


/**
 * Compute  acoustic velocity for initial pressure problem, homogenous medium, non-uniform grid.
 */
  void SolverCudaKernels::computeInitialVelocityHomogeneousNonuniform(RealMatrix&       uxSgx,
                                                                      RealMatrix&       uySgy,
                                                                      RealMatrix&       uzSgz,
                                                                      const RealMatrix& dxudxnSgx,
                                                                      const RealMatrix& dyudynSgy,
                                                                      const RealMatrix& dzudznSgz)
{
  cudaComputeInitialVelocityHomogeneousNonuniform<<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                         (uxSgx.getDeviceData(),
                                          uySgy.getDeviceData(),
                                          uzSgz.getDeviceData(),
                                          dxudxnSgx.getDeviceData(),
                                          dxudxnSgx.getDeviceData(),
                                          dxudxnSgx.getDeviceData());
// check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeInitialVelocityHomogeneousNonuniform
//----------------------------------------------------------------------------------------------------------------------


/**
 *  Cuda kernel which compute part of the new velocity term between FFTs.
 *
 *<b> Matlab code: </b>
 *
 * \verbatim
    bsxfun(@times, ddx_k_shift_pos, kappa .* p_k).
    bsxfun(@times, ddx_k_shift_pos, kappa .* p_k).
    bsxfun(@times, ddx_k_shift_pos, kappa .* p_k).
\endverbatim
 *
 *
 * @param [in, out] ifftX        - It takes the FFT of pressure (common for all three components) and returns
 *                                 the spectral part in x direction (the input for inverse FFT that follows).
 * @param [out]     ifftY        - spectral part in y dimension (the input for inverse FFT that follows).
 * @param [out]     ifftZ        - spectral part in z dimension (the input for inverse FFT that follows).
 * @param [in]      kappa        - Kappa matrix.
 * @param [in]      ddxKShiftPos - Positive spectral shift in x direction.
 * @param [in]      ddyKShiftPos - Positive spectral shift in y direction.
 * @param [in]      ddzKShiftPos - Positive spectral shift in z direction.
 */
__global__ void cudaComputePressureGradient(cuFloatComplex*       ifftX,
                                            cuFloatComplex*       ifftY,
                                            cuFloatComplex*       ifftZ,
                                            const float*          kappa,
                                            const cuFloatComplex* ddxKShiftPos,
                                            const cuFloatComplex* ddyKShiftPos,
                                            const cuFloatComplex* ddzKShiftPos)
{
  for(auto i = getIndex(); i < cudaDeviceConstants.nElementsComplex; i += getStride())
  {
    const dim3 coords = getComplex3DCoords(i);

    const cuFloatComplex pKappa = ifftX[i] * kappa[i];

    ifftX[i] = cuCmulf(pKappa, ddxKShiftPos[coords.x]);
    ifftY[i] = cuCmulf(pKappa, ddyKShiftPos[coords.y]);
    ifftZ[i] = cuCmulf(pKappa, ddzKShiftPos[coords.z]);
  }
}// end of cudaComputePressureGradient
//----------------------------------------------------------------------------------------------------------------------

/**
 *  Interface to kernel which computes the spectral part of pressure gradient calculation.
 */
void SolverCudaKernels::computePressureGradient(CufftComplexMatrix& ifftX,
                                                CufftComplexMatrix& ifftY,
                                                CufftComplexMatrix& ifftZ,
                                                const RealMatrix&    kappa,
                                                const ComplexMatrix& ddxKShiftPos,
                                                const ComplexMatrix& ddyKShiftPos,
                                                const ComplexMatrix& ddzKShiftPos)
{
  cudaComputePressureGradient<<<getSolverGridSize1D(),getSolverBlockSize1D()>>>
                              (reinterpret_cast<cuFloatComplex*>(ifftX.getDeviceData()),
                               reinterpret_cast<cuFloatComplex*>(ifftY.getDeviceData()),
                               reinterpret_cast<cuFloatComplex*>(ifftZ.getDeviceData()),
                               kappa.getDeviceData(),
                               reinterpret_cast<const cuFloatComplex*>(ddxKShiftPos.getDeviceData()),
                               reinterpret_cast<const cuFloatComplex*>(ddyKShiftPos.getDeviceData()),
                               reinterpret_cast<const cuFloatComplex*>(ddzKShiftPos.getDeviceData()));

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computePressureGradient
//----------------------------------------------------------------------------------------------------------------------


/**
 * Kernel to compute spatial part of the velocity gradient in between FFTs on uniform grid.
 * Complex numbers are passed as float2 structures.
 *
 *<b> Matlab code: </b>
 *
 * \verbatim
    bsxfun(@times, ddx_k_shift_neg, kappa .* fftn(ux_sgx));
    bsxfun(@times, ddy_k_shift_neg, kappa .* fftn(uy_sgy));
    bsxfun(@times, ddz_k_shift_neg, kappa .* fftn(uz_sgz));
 \endverbatim
 *
 * @param [in, out] fftX    - input is the FFT of velocity, output is the spectral part in x.
 * @param [in, out] fftY    - input is the FFT of velocity, output is the spectral part in y.
 * @param [in, out] fftZ    - input is the FFT of velocity, output is the spectral part in z.
 * @param [in] kappa        - Kappa matrix
 * @param [in] ddxKShiftNeg - Negative spectral shift in x direction.
 * @param [in] ddyKShiftNeg - Negative spectral shift in x direction.
 * @param [in] ddzKShiftNeg - Negative spectral shift in x direction.
 */
__global__  void cudaComputeVelocityGradient(cuFloatComplex*       fftX,
                                             cuFloatComplex*       fftY,
                                             cuFloatComplex*       fftZ,
                                             const float*          kappa,
                                             const cuFloatComplex* ddxKShiftNeg,
                                             const cuFloatComplex* ddyKShiftNeg,
                                             const cuFloatComplex* ddzKShiftNeg)
{
  for(auto i = getIndex(); i < cudaDeviceConstants.nElementsComplex; i += getStride())
  {
    const dim3 coords = getComplex3DCoords(i);

    const cuFloatComplex eDdx = ddxKShiftNeg[coords.x];
    const cuFloatComplex eDdy = ddyKShiftNeg[coords.y];
    const cuFloatComplex eDdz = ddzKShiftNeg[coords.z];

    const float eKappa = kappa[i] * cudaDeviceConstants.fftDivider;

    const cuFloatComplex fftKappaX = fftX[i] * eKappa;
    const cuFloatComplex fftKappaY = fftY[i] * eKappa;
    const cuFloatComplex fftKappaZ = fftZ[i] * eKappa;

    fftX[i] = cuCmulf(fftKappaX, eDdx);
    fftY[i] = cuCmulf(fftKappaY, eDdy);
    fftZ[i] = cuCmulf(fftKappaZ, eDdz);
  } // for
}// end of cudaComputeVelocityGradient
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute spatial part of the velocity gradient in between FFTs on uniform grid.
 */
void SolverCudaKernels::computeVelocityGradient(CufftComplexMatrix&  fftX,
                                                CufftComplexMatrix&  fftY,
                                                CufftComplexMatrix&  fftZ,
                                                const RealMatrix&    kappa,
                                                const ComplexMatrix& ddxKShiftNeg,
                                                const ComplexMatrix& ddyKShiftNeg,
                                                const ComplexMatrix& ddzKShiftNeg)
{
  cudaComputeVelocityGradient<<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                             (reinterpret_cast<cuFloatComplex *>(fftX.getDeviceData()),
                              reinterpret_cast<cuFloatComplex *>(fftY.getDeviceData()),
                              reinterpret_cast<cuFloatComplex *>(fftZ.getDeviceData()),
                              kappa.getDeviceData(),
                              reinterpret_cast<const cuFloatComplex *>(ddxKShiftNeg.getDeviceData()),
                              reinterpret_cast<const cuFloatComplex *>(ddyKShiftNeg.getDeviceData()),
                              reinterpret_cast<const cuFloatComplex *>(ddzKShiftNeg.getDeviceData()));

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeVelocityGradient
//----------------------------------------------------------------------------------------------------------------------


/**
 * Cuda kernel to shifts gradient of acoustic velocity on non-uniform grid.
 *
 * @param [in,out] duxdx  - Gradient of particle velocity in x direction.
 * @param [in,out] duydy  - Gradient of particle velocity in y direction.
 * @param [in,out] duzdz  - Gradient of particle velocity in z direction.
 * @param [in]     dxudxn - Non uniform grid shift in x direction.
 * @param [in]     dyudyn - Non uniform grid shift in y direction.
 * @param [in]     dzudzn - Non uniform grid shift in z direction.
 */
__global__  void cudaComputeVelocityGradientShiftNonuniform(float*       duxdx,
                                                            float*       duydy,
                                                            float*       duzdz,
                                                            const float* dxudxn,
                                                            const float* dyudyn,
                                                            const float* dzudzn)
{
  for(auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getReal3DCoords(i);

    duxdx[i] *= dxudxn[coords.x];
    duydy[i] *= dyudyn[coords.y];
    duzdz[i] *= dzudzn[coords.z];
  }
}// end of cudaComputeVelocityGradientShiftNonuniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to cuda kernel which shifts gradient of acoustic velocity on non-uniform grid.
 */
void SolverCudaKernels::computeVelocityGradientShiftNonuniform(RealMatrix&       duxdx,
                                                               RealMatrix&       duydy,
                                                               RealMatrix&       duzdz,
                                                               const RealMatrix& dxudxn,
                                                               const RealMatrix& dyudyn,
                                                               const RealMatrix& dzudzn)
{
  cudaComputeVelocityGradientShiftNonuniform<<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                       (duxdx.getDeviceData(),
                                        duydy.getDeviceData(),
                                        duzdz.getDeviceData(),
                                        dxudxn.getDeviceData(),
                                        dyudyn.getDeviceData(),
                                        dzudzn.getDeviceData());

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeVelocityGradientShiftNonuniform
//----------------------------------------------------------------------------------------------------------------------


/**
 * Cuda kernel which calculate new values of acoustic density, nonlinear case.
 * <b> Matlab code: </b>
 *
 * \verbatim
    rho0_plus_rho = 2 .* (rhox + rhoy + rhoz) + rho0;
    rhox = bsxfun(@times, pml_x, bsxfun(@times, pml_x, rhox) - dt .* rho0_plus_rho .* duxdx);
    rhoy = bsxfun(@times, pml_y, bsxfun(@times, pml_y, rhoy) - dt .* rho0_plus_rho .* duydy);
    rhoz = bsxfun(@times, pml_z, bsxfun(@times, pml_z, rhoz) - dt .* rho0_plus_rho .* duzdz);
 \endverbatim
 *
 * @tparam          isRho0Scalar - Is density homogeneous?
 * @param [in, out] rhoX         - Acoustic density in x direction.
 * @param [in, out] rhoY         - Acoustic density in y direction.
 * @param [in, out] rhoZ         - Acoustic density in z direction.
 * @param [in]      pmlX         - PML layer in x direction.
 * @param [in]      pmlY         - PML layer in x direction.
 * @param [in]      pmlZ         - PML layer in x direction.
 * @param [in]      duxdx        - Gradient of velocity x direction.
 * @param [in]      duydy        - Gradient of velocity y direction.
 * @param [in]      duzdz        - Gradient of velocity z direction.
 * @param [in]      rho0Data     - If density is heterogeneous, here is the matrix with data.
 */
template <bool isRho0Scalar>
__global__ void cudaComputeDensityNonlinear(float*       rhoX,
                                            float*       rhoY,
                                            float*       rhoZ,
                                            const float* pmlX,
                                            const float* pmlY,
                                            const float* pmlZ,
                                            const float* duxdx,
                                            const float* duydy,
                                            const float* duzdz,
                                            const float* rho0Data = nullptr)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getReal3DCoords(i);

    const float ePmlX = pmlX[coords.x];
    const float ePmlY = pmlY[coords.y];
    const float ePmlZ = pmlZ[coords.z];

    const float eRhoX = rhoX[i];
    const float eRhoY = rhoY[i];
    const float eRhoZ = rhoZ[i];

    const float eRho0 = (isRho0Scalar) ? cudaDeviceConstants.rho0 : rho0Data[i];

    const float sumRhosDt = (2.0f * (eRhoX + eRhoY + eRhoZ) + eRho0) * cudaDeviceConstants.dt;

    rhoX[i] = ePmlX * ((ePmlX * eRhoX) - sumRhosDt * duxdx[i]);
    rhoY[i] = ePmlY * ((ePmlY * eRhoY) - sumRhosDt * duydy[i]);
    rhoZ[i] = ePmlZ * ((ePmlZ * eRhoZ) - sumRhosDt * duzdz[i]);
  }
}//end of cudaComputeDensityNonlinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel which calculate new values of acoustic density, nonlinear case.
 */
void SolverCudaKernels::computeDensityNonlinear(RealMatrix&       rhoX,
                                                RealMatrix&       rhoY,
                                                RealMatrix&       rhoZ,
                                                const RealMatrix& pmlX,
                                                const RealMatrix& pmlY,
                                                const RealMatrix& pmlZ,
                                                const RealMatrix& duxdx,
                                                const RealMatrix& duydy,
                                                const RealMatrix& duzdz,
                                                const bool        isRho0Scalar,
                                                const float*      rho0Data)
{
  if (isRho0Scalar)
  {
    cudaComputeDensityNonlinear<true>
                               <<<getSolverGridSize1D(), getSolverBlockSize1D() >>>
                               (rhoX.getDeviceData(),
                                rhoY.getDeviceData(),
                                rhoZ.getDeviceData(),
                                pmlX.getDeviceData(),
                                pmlY.getDeviceData(),
                                pmlZ.getDeviceData(),
                                duxdx.getDeviceData(),
                                duydy.getDeviceData(),
                                duzdz.getDeviceData());
  }
  else
  {
   cudaComputeDensityNonlinear<false>
                               <<<getSolverGridSize1D(), getSolverBlockSize1D() >>>
                               (rhoX.getDeviceData(),
                                rhoY.getDeviceData(),
                                rhoZ.getDeviceData(),
                                pmlX.getDeviceData(),
                                pmlY.getDeviceData(),
                                pmlZ.getDeviceData(),
                                duxdx.getDeviceData(),
                                duydy.getDeviceData(),
                                duzdz.getDeviceData(),
                                rho0Data);
  }

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeDensityNonlinear
//-----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel which calculate new values of acoustic density, linear case.
 *
 * <b> Matlab code: </b>
 *
 * \verbatim
    rhox = bsxfun(@times, pml_x, bsxfun(@times, pml_x, rhox) - dt .* rho0 .* duxdx);
    rhoy = bsxfun(@times, pml_y, bsxfun(@times, pml_y, rhoy) - dt .* rho0 .* duydy);
    rhoz = bsxfun(@times, pml_z, bsxfun(@times, pml_z, rhoz) - dt .* rho0 .* duzdz);
 \endverbatim
 *
 * @tparam      isRhoScalar - is density homogeneous?
 * @param [out] rhoX        - density x
 * @param [out] rhoY        - density y
 * @param [out] rhoZ        - density y
 * @param [in]  pmlX        - pml x
 * @param [in]  pmlY        - pml y
 * @param [in]  pmlZ        - pml z
 * @param [in]  duxdx       - gradient of velocity x
 * @param [in]  duydy       - gradient of velocity x
 * @param [in]  duzdz       - gradient of velocity z
 * @param [in]  rho0        - initial density (matrix here)
 */
template <bool isRho0Scalar>
__global__ void cudaComputeDensityLinear(float*       rhoX,
                                         float*       rhoY,
                                         float*       rhoZ,
                                         const float* pmlX,
                                         const float* pmlY,
                                         const float* pmlZ,
                                         const float* duxdx,
                                         const float* duydy,
                                         const float* duzdz,
                                         const float* rho0 = nullptr)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getReal3DCoords(i);

    const float ePmlX = pmlX[coords.x];
    const float ePmlY = pmlY[coords.y];
    const float ePmlZ = pmlZ[coords.z];

    const float dtRho0  = (isRho0Scalar) ? cudaDeviceConstants.dtRho0 : cudaDeviceConstants.dt * rho0[i];

    rhoX[i] = ePmlX * (ePmlX * rhoX[i] - dtRho0 * duxdx[i]);
    rhoY[i] = ePmlY * (ePmlY * rhoY[i] - dtRho0 * duydy[i]);
    rhoZ[i] = ePmlZ * (ePmlZ * rhoZ[i] - dtRho0 * duzdz[i]);
  }
}// end of cudaComputeDensityLinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate acoustic density for linear case, homogeneous case is default.
 */
void SolverCudaKernels::computeDensityLinear(RealMatrix&       rhoX,
                                             RealMatrix&       rhoY,
                                             RealMatrix&       rhoZ,
                                             const RealMatrix& pmlX,
                                             const RealMatrix& pmlY,
                                             const RealMatrix& pmlZ,
                                             const RealMatrix& duxdx,
                                             const RealMatrix& duydy,
                                             const RealMatrix& duzdz,
                                             const bool        isRho0Scalar,
                                             const float*      rho0Data)
{
  if (isRho0Scalar)
  {
    cudaComputeDensityLinear<true>
                            <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                            (rhoX.getDeviceData(),
                             rhoY.getDeviceData(),
                             rhoZ.getDeviceData(),
                             pmlX.getDeviceData(),
                             pmlY.getDeviceData(),
                             pmlZ.getDeviceData(),
                             duxdx.getDeviceData(),
                             duydy.getDeviceData(),
                             duzdz.getDeviceData());
  }
  else
  {
    cudaComputeDensityLinear<false>
                            <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                            (rhoX.getDeviceData(),
                             rhoY.getDeviceData(),
                             rhoZ.getDeviceData(),
                             pmlX.getDeviceData(),
                             pmlY.getDeviceData(),
                             pmlZ.getDeviceData(),
                             duxdx.getDeviceData(),
                             duydy.getDeviceData(),
                             duzdz.getDeviceData(),
                             rho0Data);
  }
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeDensityLinear
//----------------------------------------------------------------------------------------------------------------------


/**
 * Cuda kernel which calculates three temporary sums in the new pressure formula \n
 * non-linear absorbing case. Homogeneous and heterogenous variants are treated using templates.
 * Homogeneous variables are in constant memory.
 *
 * @tparam [in]  isBonAScalar       - Is B on A homogeneous?
 * @tparam [in]  isRho0Scalar       - Is rho0 a scalar value (homogeneous)?
 *
 * @param [out] densitySum          - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [out] nonlinearTerm       - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [out] velocityGradientSum - rho0* (duxdx + duydy + duzdz)
 * @param [in]  rhoX                - Acoustic density x direction
 * @param [in]  rhoY                - Acoustic density y direction
 * @param [in]  rhoZ                - Acoustic density z direction
 * @param [in]  duxdx               - Gradient of velocity in x direction
 * @param [in]  duydy               - Gradient of velocity in y direction
 * @param [in]  duzdz               - Gradient of velocity in z direction
 * @param [in]  bOnAData            - Heterogeneous value for BonA
 * @param [in]  rho0Data            - Heterogeneous value for rho0
 *
 *
 */
template <bool isBonAScalar, bool isRho0Scalar>
__global__ void cudaComputePressureTermsNonlinear(float*       densitySum,
                                                  float*       nonlinearTerm,
                                                  float*       velocityGradientSum,
                                                  const float* rhoX,
                                                  const float* rhoY,
                                                  const float* rhoZ,
                                                  const float* duxdx,
                                                  const float* duydy,
                                                  const float* duzdz,
                                                  const float* bOnAData,
                                                  const float* rho0Data)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float eBonA = (isBonAScalar) ? cudaDeviceConstants.bOnA : bOnAData[i];
    const float eRho0 = (isRho0Scalar) ? cudaDeviceConstants.rho0 : rho0Data[i];

    const float eRhoSum = rhoX[i] + rhoY[i] + rhoZ[i];

    densitySum[i]          = eRhoSum;
    nonlinearTerm[i]       = ((eBonA * eRhoSum * eRhoSum) / (2.0f * eRho0)) + eRhoSum;
    velocityGradientSum[i] = eRho0 * (duxdx[i] + duydy[i] + duzdz[i]);
    }
}// end of cudaComputePressureTermsNonlinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel which calculates three temporary sums in the new pressure formula \n
 * non-linear absorbing case.
 */
void SolverCudaKernels::computePressureTermsNonlinear(RealMatrix&       densitySum,
                                                      RealMatrix&       nonlinearTerm,
                                                      RealMatrix&       velocityGradientSum,
                                                      const RealMatrix& rhoX,
                                                      const RealMatrix& rhoY,
                                                      const RealMatrix& rhoZ,
                                                      const RealMatrix& duxdx,
                                                      const RealMatrix& duydy,
                                                      const RealMatrix& duzdz,
                                                      const bool         isBonAScalar,
                                                      const float*       bOnAData,
                                                      const bool         isRho0Scalar,
                                                      const float*       rho0Data)
{
  // all variants are treated by templates, here you can see all 4 variants.
  if (isBonAScalar)
  {
    if (isRho0Scalar)
    {
      cudaComputePressureTermsNonlinear<true, true>
                                     <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                     (densitySum.getDeviceData(),
                                      nonlinearTerm.getDeviceData(),
                                      velocityGradientSum.getDeviceData(),
                                      rhoX.getDeviceData(),
                                      rhoY.getDeviceData(),
                                      rhoZ.getDeviceData(),
                                      duxdx.getDeviceData(),
                                      duydy.getDeviceData(),
                                      duzdz.getDeviceData(),
                                      bOnAData,
                                      rho0Data);
    }
    else
    {
      cudaComputePressureTermsNonlinear<true, false>
                                       <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                       (densitySum.getDeviceData(),
                                        nonlinearTerm.getDeviceData(),
                                        velocityGradientSum.getDeviceData(),
                                        rhoX.getDeviceData(),
                                        rhoY.getDeviceData(),
                                        rhoZ.getDeviceData(),
                                        duxdx.getDeviceData(),
                                        duydy.getDeviceData(),
                                        duzdz.getDeviceData(),
                                        bOnAData,
                                        rho0Data);
    }
  }
  else // BonA is false
  {
   if (isRho0Scalar)
    {
    cudaComputePressureTermsNonlinear<false, true>
                                     <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                     (densitySum.getDeviceData(),
                                      nonlinearTerm.getDeviceData(),
                                      velocityGradientSum.getDeviceData(),
                                      rhoX.getDeviceData(),
                                      rhoY.getDeviceData(),
                                      rhoZ.getDeviceData(),
                                      duxdx.getDeviceData(),
                                      duydy.getDeviceData(),
                                      duzdz.getDeviceData(),
                                      bOnAData,
                                      rho0Data);
    }
    else
    {
    cudaComputePressureTermsNonlinear<false, false>
                                     <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                     (densitySum.getDeviceData(),
                                      nonlinearTerm.getDeviceData(),
                                      velocityGradientSum.getDeviceData(),
                                      rhoX.getDeviceData(),
                                      rhoY.getDeviceData(),
                                      rhoZ.getDeviceData(),
                                      duxdx.getDeviceData(),
                                      duydy.getDeviceData(),
                                      duzdz.getDeviceData(),
                                      bOnAData,
                                      rho0Data);
    }
  }
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computePressureTermsNonlinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel that calculates two temporary sums in the new pressure formula, linear absorbing case.
 *
 * @tparam      isRho0Scalar        - Is density  homogeneous?
 *
 * @param [out] densitySum          - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [out] velocityGradientSum - rho0* (duxdx + duydy + duzdz);
 * @param [in]  rhoX                - Acoustic density in x direction.
 * @param [in]  rhoY                - Acoustic density in y direction.
 * @param [in]  rhoZ                - Acoustic density in z direction.
 * @param [in]  duxdx               - Velocity gradient in x direction.
 * @param [in]  duydy               - Velocity gradient in x direction.
 * @param [in]  duzdz               - Velocity gradient in x direction.
 * @param [in]  rho0Data            - Acoustic density data in heterogeneous case.
 */
template<bool isRho0Scalar>
__global__ void cudaComputePressureTermsLinear(float*       densitySum,
                                               float*       velocityGradientSum,
                                               const float* rhoX,
                                               const float* rhoY,
                                               const float* rhoZ,
                                               const float* duxdx,
                                               const float* duydy,
                                               const float* duzdz,
                                               const float* rho0Data)
{
  for(auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float rho0 = (isRho0Scalar) ? cudaDeviceConstants.rho0 : rho0Data[i];

    densitySum[i]          = rhoX[i] + rhoY[i] + rhoZ[i];
    velocityGradientSum[i] = rho0 * (duxdx[i] + duydy[i] + duzdz[i]);
  }
}// end of cudaComputePressureTermsLinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel that Calculates two temporary sums in the new pressure
 * formula, linear absorbing case.

 */
void SolverCudaKernels::computePressureTermsLinear(RealMatrix&       densitySum,
                                                   RealMatrix&       velocityGradientSum,
                                                   const RealMatrix& rhoX,
                                                   const RealMatrix& rhoY,
                                                   const RealMatrix& rhoZ,
                                                   const RealMatrix& duxdx,
                                                   const RealMatrix& duydy,
                                                   const RealMatrix& duzdz,
                                                   const bool        isRho0Scalar,
                                                   const float*      rho0Data)
{
  if (isRho0Scalar)
  {
   cudaComputePressureTermsLinear<true>
                                <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                (densitySum.getDeviceData(),
                                 velocityGradientSum.getDeviceData(),
                                 rhoX.getDeviceData(),
                                 rhoY.getDeviceData(),
                                 rhoZ.getDeviceData(),
                                 duxdx.getDeviceData(),
                                 duydy.getDeviceData(),
                                 duzdz.getDeviceData(),
                                 rho0Data);
  }
  else
  {
   cudaComputePressureTermsLinear<false>
                                <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                (densitySum.getDeviceData(),
                                 velocityGradientSum.getDeviceData(),
                                 rhoX.getDeviceData(),
                                 rhoY.getDeviceData(),
                                 rhoZ.getDeviceData(),
                                 duxdx.getDeviceData(),
                                 duydy.getDeviceData(),
                                 duzdz.getDeviceData(),
                                 rho0Data);
  }
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computePressureTermsLinear
//----------------------------------------------------------------------------------------------------------------------


/**
 * Cuda kernel which computes absorbing term with abosrbNabla1 and  absorbNabla2.
 * \verbatim
   fftPart1 = absorbNabla1 .* fftPart1 \n
   fftPart2 = absorbNabla2 .* fftPart2 \n
 \endverbatim
 *
 * @param [in,out] fftPart1     - Nabla1 part:
 * @param [in,out] fftPart2     - Nabla2 part:
 * @param [in]     absorbNabla1 - Absorption coefficient 1
 * @param [in]     absorbNabla2 - Absorption coefficient 2
 */
__global__ void cudaComputeAbsorbtionTerm(cuFloatComplex* fftPart1,
                                          cuFloatComplex* fftPart2,
                                          const float*    absorbNabla1,
                                          const float*    absorbNabla2)
{
  for(auto i = getIndex(); i < cudaDeviceConstants.nElementsComplex; i += getStride())
  {
    fftPart1[i] *= absorbNabla1[i];
    fftPart2[i] *= absorbNabla2[i];
  }
}// end of computeAbsorbtionTerm
//--------------------------------------------------------------------------------------------------

/**
 * Interface to kernel which computes absorbing term with abosrbNabla1 and  absorbNabla2.
 */
void SolverCudaKernels::computeAbsorbtionTerm(CufftComplexMatrix& fftPart1,
                                              CufftComplexMatrix& fftPart2,
                                              const RealMatrix&   absorbNabla1,
                                              const RealMatrix&   absorbNabla2)
{
  cudaComputeAbsorbtionTerm<<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                           (reinterpret_cast<cuFloatComplex*> (fftPart1.getDeviceData()),
                            reinterpret_cast<cuFloatComplex*> (fftPart2.getDeviceData()),
                            absorbNabla1.getDeviceData(),
                            absorbNabla2.getDeviceData());

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeAbsorbtionTerm
//----------------------------------------------------------------------------------------------------------------------


/**
 * Cuda kernel to sum sub-terms to calculate new pressure, non-linear case.

 * @tparam         isC2Scalar          - is sound speed homogeneous?
 * @tparam         areTauAndEtaScalars - is absorption homogeneous?

 * @param [in,out] p                   - New value of pressure
 * @param [in]     nonlinearTerm       - Nonlinear term
 * @param [in]     absorbTauTerm       - Absorb tau term from the pressure eq.
 * @param [in]     absorbEtaTerm       - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [in]     c2Data              - sound speed data in heterogeneous case.
 * @param [in]     absorbTauData       - Absorb tau data in heterogenous case.
 * @param [in]     absorbEtaData       - Absorb eta data in heterogenous case.
 */
template <bool isC2Scalar, bool areTauAndEtaScalar>
__global__ void cudaSumPressureTermsNonlinear(float*       p,
                                              const float* nonlinearTerm,
                                              const float* absorbTauTerm,
                                              const float* absorbEtaTerm,
                                              const float* c2Data,
                                              const float* absorbTauData,
                                              const float* absorbEtaData)
{
  for(auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float c2        = (isC2Scalar)         ? cudaDeviceConstants.c2  : c2Data[i];
    const float absorbTau = (areTauAndEtaScalar) ? cudaDeviceConstants.absorbTau : absorbTauData[i];
    const float absorbEta = (areTauAndEtaScalar) ? cudaDeviceConstants.absorbEta : absorbEtaData[i];

    p[i] = c2 * (nonlinearTerm[i] + (cudaDeviceConstants.fftDivider *
                                     ((absorbTauTerm[i] * absorbTau) - (absorbEtaTerm[i] * absorbEta))));
  }
}// end of cudaSumPressureTermsNonlinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to cuda sum sub-terms to calculate new pressure, non-linear case.
 */
void SolverCudaKernels::sumPressureTermsNonlinear(RealMatrix&       p,
                                                  const RealMatrix& nonlinearTerm,
                                                  const RealMatrix& absorbTauTerm,
                                                  const RealMatrix& absorbEtaTerm,
                                                  const bool        isC2Scalar,
                                                  const float*      c2Data,
                                                  const bool        areTauAndEtaScalars,
                                                  const float*      absorbTauData,
                                                  const float*      absorbEtaData)
{
  if (isC2Scalar)
  {
    if (areTauAndEtaScalars)
    {
      cudaSumPressureTermsNonlinear<true, true>
                                   <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                   (p.getDeviceData(),
                                    nonlinearTerm.getDeviceData(),
                                    absorbTauTerm.getDeviceData(),
                                    absorbEtaTerm.getDeviceData(),
                                    c2Data,
                                    absorbTauData,
                                    absorbEtaData);
    }
    else
    {
      cudaSumPressureTermsNonlinear<true, false>
                                   <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                   (p.getDeviceData(),
                                    nonlinearTerm.getDeviceData(),
                                    absorbTauTerm.getDeviceData(),
                                    absorbEtaTerm.getDeviceData(),
                                    c2Data,
                                    absorbTauData,
                                    absorbEtaData);
    }
  }
  else
  { // c2 is matrix
     if (areTauAndEtaScalars)
    {
      cudaSumPressureTermsNonlinear<false, true>
                                   <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                   (p.getDeviceData(),
                                    nonlinearTerm.getDeviceData(),
                                    absorbTauTerm.getDeviceData(),
                                    absorbEtaTerm.getDeviceData(),
                                    c2Data,
                                    absorbTauData,
                                    absorbEtaData);
    }
    else
    {
      cudaSumPressureTermsNonlinear<false, false>
                                   <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                   (p.getDeviceData(),
                                    nonlinearTerm.getDeviceData(),
                                    absorbTauTerm.getDeviceData(),
                                    absorbEtaTerm.getDeviceData(),
                                    c2Data,
                                    absorbTauData,
                                    absorbEtaData);
    }
  }
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of sumPressureTermsNonlinear
//----------------------------------------------------------------------------------------------------------------------


/**
 * Cuda kernel that sums sub-terms to calculate new pressure, linear case.
 *
 * @tparam      isC2Scalar          - is sound speed homogeneous?
 * @tparam      areTauAndEtaScalars - is absorption homogeneous?
 *
 * @param [out] p                   - New value of pressure.
 * @param [in]  absorbTauTerm       - Absorb tau term from the pressure eq.
 * @param [in]  absorbEtaTerm       - Absorb tau term from the pressure eq.
 * @param [in]  densitySum          - Sum of acoustic density.
 * @param [in]  c2Data              - sound speed data in heterogeneous case.
 * @param [in]  absorbTauData       - Absorb tau data in heterogenous case.
 * @param [in]  absorbEtaData       - Absorb eta data in heterogenous case.
 */
template <bool isC2Scalar, bool areTauAndEtaScalar>
__global__ void cudaSumPressureTermsLinear(float*       p,
                                           const float* absorbTauTerm,
                                           const float* absorbEtaTerm,
                                           const float* densitySum,
                                           const float* c2Data,
                                           const float* absorbTauData,
                                           const float* absorbEtaData)
{
  for(auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float c2        = (isC2Scalar)         ? cudaDeviceConstants.c2        : c2Data[i];
    const float absorbTau = (areTauAndEtaScalar) ? cudaDeviceConstants.absorbTau : absorbTauData[i];
    const float absorbEta = (areTauAndEtaScalar) ? cudaDeviceConstants.absorbEta : absorbEtaData[i];

    p[i] = c2 * (densitySum[i] + (cudaDeviceConstants.fftDivider *
                (absorbTauTerm[i] * absorbTau - absorbEtaTerm[i] * absorbEta)));
  }
}// end of cudaSumPressureTermsLinear
//----------------------------------------------------------------------------------------------------------------------


/**
 * Interface to kernel that sums sub-terms to calculate new pressure, linear case.
 */
void SolverCudaKernels::sumPressureTermsLinear(RealMatrix&       p,
                                               const RealMatrix& absorbTauTerm,
                                               const RealMatrix& absorbEtaTerm,
                                               const RealMatrix& densitySum,
                                               const bool        isC2Scalar,
                                               const float*      c2Data,
                                               const bool        areTauAndEtaScalars,
                                               const float*      absorbTauData,
                                               const float*      absorbEtaData)
{
  if (isC2Scalar)
  {
    if (areTauAndEtaScalars)
    {
      cudaSumPressureTermsLinear<true,true>
                                <<<getSolverGridSize1D(), getSolverBlockSize1D() >>>
                                (p.getDeviceData(),
                                 absorbTauTerm.getDeviceData(),
                                 absorbEtaTerm.getDeviceData(),
                                 densitySum.getDeviceData(),
                                 c2Data,
                                 absorbTauData,
                                 absorbEtaData);
    }
    else
    {
      cudaSumPressureTermsLinear<true,false>
                                <<<getSolverGridSize1D(), getSolverBlockSize1D() >>>
                                (p.getDeviceData(),
                                 absorbTauTerm.getDeviceData(),
                                 absorbEtaTerm.getDeviceData(),
                                 densitySum.getDeviceData(),
                                 c2Data,
                                 absorbTauData,
                                 absorbEtaData);
    }
   }
  else
  {
    if (areTauAndEtaScalars)
    {
      cudaSumPressureTermsLinear<false,true>
                                <<<getSolverGridSize1D(), getSolverBlockSize1D() >>>
                                (p.getDeviceData(),
                                 absorbTauTerm.getDeviceData(),
                                 absorbEtaTerm.getDeviceData(),
                                 densitySum.getDeviceData(),
                                 c2Data,
                                 absorbTauData,
                                 absorbEtaData);
    }
    else
    {
      cudaSumPressureTermsLinear<false,false>
                                <<<getSolverGridSize1D(), getSolverBlockSize1D() >>>
                                (p.getDeviceData(),
                                 absorbTauTerm.getDeviceData(),
                                 absorbEtaTerm.getDeviceData(),
                                 densitySum.getDeviceData(),
                                 c2Data,
                                 absorbTauData,
                                 absorbEtaData);
    }
  }
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of sumPressureTermsLinea
//----------------------------------------------------------------------------------------------------------------------


/**
 * Cuda kernel that sums sub-terms for new p, non-linear lossless case.
 *
 * @tparam      isC2Scalar   - Is sound speed homogenous?
 * @tparam      isBOnAScalar - Is nonlinearity homogeneous?
 * @tparam      isRho0Scalar - Is density homogeneous?
 *
 * @param [out] p            - New value of pressure
 * @param [in]  rhoX         - Acoustic density in x direction.
 * @param [in]  rhoY         - Acoustic density in y direction.
 * @param [in]  rhoZ         - Acoustic density in z direction.
 * @param [in]  c2Data       - Sound speed data in heterogeneous case.
 * @param [in]  bOnAData     - B on A data in heterogeneous case.
 * @param [in]  rho0Data     - Acoustic density data in heterogeneous case.
 */
template<bool isC2Scalar, bool isBOnAScalar, bool isRho0Scalar>
__global__ void cudaSumPressureNonlinearLossless(float*       p,
                                                 const float* rhoX,
                                                 const float* rhoY,
                                                 const float* rhoZ,
                                                 const float* c2Data,
                                                 const float* bOnAData,
                                                 const float* rho0Data)
{
  for(auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float c2   = (isC2Scalar)   ? cudaDeviceConstants.c2   : c2Data[i];
    const float bOnA = (isBOnAScalar) ? cudaDeviceConstants.bOnA : bOnAData[i];
    const float rho0 = (isRho0Scalar) ? cudaDeviceConstants.rho0 : rho0Data[i];

    const float rhoSum = rhoX[i] + rhoY[i] + rhoZ[i];

    p[i] = c2 * (rhoSum + (bOnA * (rhoSum * rhoSum) / (2.0f * rho0)));
  }
}// end of cudaSumPressureNonlinearLossless
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel that sums sub-terms for new p, non-linear lossless case.
 */
void SolverCudaKernels::sumPressureNonlinearLossless(RealMatrix&       p,
                                                     const RealMatrix& rhoX,
                                                     const RealMatrix& rhoY,
                                                     const RealMatrix& rhoZ,
                                                     const bool        isC2Scalar,
                                                     const float*      c2Data,
                                                     const bool        isBOnAScalar,
                                                     const float*      bOnAData,
                                                     const bool        isRho0Scalar,
                                                     const float*      rho0Data)
{
  if (isC2Scalar)
  {
    if (isBOnAScalar)
    {
      if (isRho0Scalar)
      {
        cudaSumPressureNonlinearLossless<true, true, true>
                                        <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                        (p.getDeviceData(),
                                         rhoX.getDeviceData(),
                                         rhoY.getDeviceData(),
                                         rhoZ.getDeviceData(),
                                         c2Data,
                                         bOnAData,
                                         rho0Data);
      }
      else
      {
        cudaSumPressureNonlinearLossless<true, true, false>
                                        <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                        (p.getDeviceData(),
                                         rhoX.getDeviceData(),
                                         rhoY.getDeviceData(),
                                         rhoZ.getDeviceData(),
                                         c2Data,
                                         bOnAData,
                                         rho0Data);
      }
    }// isBOnAScalar= true
    else
    {
      if (isRho0Scalar)
      {
        cudaSumPressureNonlinearLossless<true, false, true>
                                        <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                        (p.getDeviceData(),
                                         rhoX.getDeviceData(),
                                         rhoY.getDeviceData(),
                                         rhoZ.getDeviceData(),
                                         c2Data,
                                         bOnAData,
                                         rho0Data);
      }
      else
      {
        cudaSumPressureNonlinearLossless<true, false, false>
                                        <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                        (p.getDeviceData(),
                                         rhoX.getDeviceData(),
                                         rhoY.getDeviceData(),
                                         rhoZ.getDeviceData(),
                                         c2Data,
                                         bOnAData,
                                         rho0Data);
      }
    }
  }
  else
  { // isC2Scalar == false
   if (isBOnAScalar)
    {
      if (isRho0Scalar)
      {
        cudaSumPressureNonlinearLossless<false, true, true>
                                        <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                        (p.getDeviceData(),
                                         rhoX.getDeviceData(),
                                         rhoY.getDeviceData(),
                                         rhoZ.getDeviceData(),
                                         c2Data,
                                         bOnAData,
                                         rho0Data);
      }
      else
      {
        cudaSumPressureNonlinearLossless<false, true, false>
                                        <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                        (p.getDeviceData(),
                                         rhoX.getDeviceData(),
                                         rhoY.getDeviceData(),
                                         rhoZ.getDeviceData(),
                                         c2Data,
                                         bOnAData,
                                         rho0Data);
      }
    }// isBOnAScalar= true
    else
    {
      if (isRho0Scalar)
      {
        cudaSumPressureNonlinearLossless<false, false, true>
                                        <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                        (p.getDeviceData(),
                                         rhoX.getDeviceData(),
                                         rhoY.getDeviceData(),
                                         rhoZ.getDeviceData(),
                                         c2Data,
                                         bOnAData,
                                         rho0Data);
      }
      else
      {
        cudaSumPressureNonlinearLossless<false, false, false>
                                        <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                        (p.getDeviceData(),
                                         rhoX.getDeviceData(),
                                         rhoY.getDeviceData(),
                                         rhoZ.getDeviceData(),
                                         c2Data,
                                         bOnAData,
                                         rho0Data);
      }
    }
  }

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of sumPressureNonlinearLossless
//----------------------------------------------------------------------------------------------------------------------


/**
 * @brief Cuda kernel that sums sub-terms for new p, linear lossless case.
 *
 * @tparam      isC2Scalar - Is sound speed homogenous?
 *
 * @param [out] p          - New value of pressure
 * @param [in]  rhoX       - Acoustic density in x direction.
 * @param [in]  rhoY       - Acoustic density in x direction.
 * @param [in]  rhoZ       - Acoustic density in x direction.
 * @param [in]  c2Data     - Sound speed data in heterogeneous case.
 */
template <bool isC2Scalar>
__global__ void cudaSumPressureLinearLossless(float*       p,
                                              const float* rhoX,
                                              const float* rhoY,
                                              const float* rhoZ,
                                              const float* c2Data)
{
  for(auto  i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float c2 = (isC2Scalar) ? cudaDeviceConstants.c2 : c2Data[i];
    p[i] = c2 * (rhoX[i] + rhoY[i] + rhoZ[i]);
  }
}// end of cudaSumPressureLinearLossless
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel that sums sub-terms for new p, linear lossless case.
 */
void SolverCudaKernels::sumPressureLinearLossless(RealMatrix&       p,
                                                  const RealMatrix& rhoX,
                                                  const RealMatrix& rhoY,
                                                  const RealMatrix& rhoZ,
                                                  const bool        isC2Scalar,
                                                  const float*      c2Data)
{
  if (isC2Scalar)
  {
    cudaSumPressureLinearLossless<true>
                                 <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                 (p.getDeviceData(),
                                  rhoX.getDeviceData(),
                                  rhoY.getDeviceData(),
                                  rhoZ.getDeviceData(),
                                  c2Data);
  }
  else
  {
    cudaSumPressureLinearLossless<false>
                                 <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                 (p.getDeviceData(),
                                  rhoX.getDeviceData(),
                                  rhoY.getDeviceData(),
                                  rhoZ.getDeviceData(),
                                  c2Data);
  }
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of sumPressureLinearLossless
//----------------------------------------------------------------------------------------------------------------------


/**
 * Cuda kernel to transpose a 3D matrix in XY planes of any dimension sizes.
 * Every block in a 1D grid transposes a few slabs.
 * Every block is composed of a 2D mesh of threads. The y dim gives the number of tiles processed
 * simultaneously. Each tile is processed by a single thread warp.
 * The shared memory is used to coalesce memory accesses and the padding is to eliminate bank
 * conflicts.  First the full tiles are transposed, then the remainder in the X, then Y and finally
 * the last bit in the bottom right corner.  \n
 * As a part of the transposition, the matrices can be padded to conform with cuFFT.
 *
 * @tparam      padding      - Which matrices are padded (template parameter).
 * @tparam      isSquareSlab - Are the slabs of a square shape with sizes divisible by the warp size.
 * @tparam      warpSize     - Set the warp size. Built in value cannot be used due to shared memory allocation.
 *
 * @param [out] outputMatrix - Output matrix.
 * @param [in]  inputMatrix  - Input  matrix.
 * @param [in]  dimSizes     - Dimension sizes of the original matrix.
 *
 * @warning A blockDim.x has to be of a warp size (typically 32) \n
 *          blockDim.y should be between 1 and 4 (four tiles at once).
 *          blockDim.y has to be equal with the tilesAtOnce parameter.  \n
 *          blockDim.z must stay 1 \n
 *          Grid has to be organized (N, 1 ,1)
 *
 */
template<SolverCudaKernels::TransposePadding padding,
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

  using TP = SolverCudaKernels::TransposePadding;
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
      //-------------------------------- full tiles in X -------------------------------//
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
        //-------------------------------- reminders in X --------------------------------//
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
      //-------------------------------- reminders in Y --------------------------------//
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

      //----------------------------- reminder in X and Y ----------------------------------//
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
//----------------------------------------------------------------------------------------------------------------------


/**
 * Transpose a real 3D matrix in the X-Y direction. It is done out-of-place.
 */
template<SolverCudaKernels::TransposePadding padding>
void SolverCudaKernels::trasposeReal3DMatrixXY(float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes)
{
  // fixed size at the moment, may be tuned based on the domain shape in the future
  // warpSize set to 32, and 4 tiles processed at once
  if ((dimSizes.x % 32 == 0) && (dimSizes.y % 32 == 0))
  {
    cudaTrasnposeReal3DMatrixXY<padding, true, 32, 4>
                               <<<GetSolverTransposeGirdSize(),getSolverTransposeBlockSize()>>>
                               (outputMatrix, inputMatrix, dimSizes);

  }
  else
  {
    cudaTrasnposeReal3DMatrixXY<padding, false, 32, 4>
                               <<<GetSolverTransposeGirdSize(), getSolverTransposeBlockSize() >>>
                               (outputMatrix, inputMatrix, dimSizes);
  }

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of trasposeReal3DMatrixXY
//----------------------------------------------------------------------------------------------------------------------

//---------------------------------- Explicit instances of TrasposeReal3DMatrixXY ------------------------------------//
/// Transpose a real 3D matrix in the X-Y direction, input matrix padded, output matrix compact
template
void SolverCudaKernels::trasposeReal3DMatrixXY<SolverCudaKernels::TransposePadding::kInput>
                                              (float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Y direction, input matrix compact, output matrix padded
template
void SolverCudaKernels::trasposeReal3DMatrixXY<SolverCudaKernels::TransposePadding::kOutput>
                                              (float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Y direction, input and output matrix compact
template
void SolverCudaKernels::trasposeReal3DMatrixXY<SolverCudaKernels::TransposePadding::kNone>
                                              (float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes);
/// Transpose a real 3D matrix in the X-Y direction, input and output matrix padded
template
void SolverCudaKernels::trasposeReal3DMatrixXY<SolverCudaKernels::TransposePadding::kInputOutput>
                                              (float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes);
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to transpose a 3D matrix in XZ planes of any dimension sizes.
 * Every block in a 1D grid transposes a few slabs.
 * Every block is composed of a 2D mesh of threads. The y dim gives the number of tiles processed
 * simultaneously. Each tile is processed by a single thread warp.
 * The shared memory is used to coalesce memory accesses and the padding is to eliminate bank
 * conflicts.  First the full tiles are transposed, then the remainder in the X, then Z and finally
 * the last bit in the bottom right corner.  \n
 * As a part of the transposition, the matrices can be padded to conform with cuFFT.
 *
 *
 * @tparam      padding      - Which matrices are padded (template parameter).
 * @tparam      isSquareSlab - Are the slabs of a square shape with sizes divisible by the warp size.
 * @tparam      warpSize     - Set the warp size. Built in value cannot be used due to shared memory allocation.
 *
 * @param [out] outputMatrix - Output matrix.
 * @param [in]  inputMatrix  - Input  matrix.
 * @param [in]  dimSizes     - Dimension sizes of the original matrix.
 *
 * @warning A blockDim.x has to of a warp size (typically 32) \n
 *          blockDim.y should be between 1 and 4 (four tiles at once).
 *          blockDim.y has to be equal with the tilesAtOnce parameter.  \n
 *          blockDim.z must stay 1. \n
 *          Grid has to be organized (N, 1 ,1).
 *
 */
template<SolverCudaKernels::TransposePadding padding,
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

  using TP = SolverCudaKernels::TransposePadding;
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
        //-------------------------------- reminders in X --------------------------------//
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
      //-------------------------------- reminders in Z ----------------------------------//
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

     //----------------------------- reminder in X and Z ----------------------------------//
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
//----------------------------------------------------------------------------------------------------------------------



/**
 * Transpose a real 3D matrix in the X-Z direction. It is done out-of-place.
 */
template<SolverCudaKernels::TransposePadding padding>
void SolverCudaKernels::trasposeReal3DMatrixXZ(float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes)
{
  // fixed size at the moment, may be tuned based on the domain shape in the future
  // warpSize set to 32, and 4 tiles processed at once
  if ((dimSizes.x % 32 == 0) && (dimSizes.z % 32 == 0))
  {
    cudaTrasnposeReal3DMatrixXZ<padding, true, 32, 4>
                               <<<GetSolverTransposeGirdSize(), getSolverTransposeBlockSize()>>>
                               (outputMatrix, inputMatrix, dimSizes);
  }
  else
  {
    cudaTrasnposeReal3DMatrixXZ<padding, false, 32, 4>
                               <<<GetSolverTransposeGirdSize(), getSolverTransposeBlockSize()>>>
                               (outputMatrix, inputMatrix, dimSizes);
  }

  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of trasposeReal3DMatrixXZ
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------- Explicit instances of TrasposeReal3DMatrixXZ -------------------------------------//
/// Transpose a real 3D matrix in the X-Z direction, input matrix padded, output matrix compact
template
void SolverCudaKernels::trasposeReal3DMatrixXZ<SolverCudaKernels::TransposePadding::kInput>
                                              (float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Z direction, input matrix compact, output matrix padded
template
void SolverCudaKernels::trasposeReal3DMatrixXZ<SolverCudaKernels::TransposePadding::kOutput>
                                              (float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Z direction, input and output matrix compact
template
void SolverCudaKernels::trasposeReal3DMatrixXZ<SolverCudaKernels::TransposePadding::kNone>
                                              (float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Z direction, input and output matrix padded
template
void SolverCudaKernels::trasposeReal3DMatrixXZ<SolverCudaKernels::TransposePadding::kInputOutput>
                                              (float*       outputMatrix,
                                               const float* inputMatrix,
                                               const dim3&  dimSizes);
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute velocity shift in the X direction.
 *
 * @param [in,out] cufftShiftTemp - Matrix to shift, shifted matrix.
 * @param [in]     xShiftNegR     - negative Fourier shift.
 */
__global__ void cudaComputeVelocityShiftInX(cuFloatComplex*       cufftShiftTemp,
                                            const cuFloatComplex* xShiftNegR)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.nElementsComplex; i += getStride())
  {
    const auto  x = i % cudaDeviceConstants.nxComplex;

    cufftShiftTemp[i] = cuCmulf(cufftShiftTemp[i], xShiftNegR[x]) * cudaDeviceConstants.fftDividerX;
  }
}// end of cudaComputeVelocityShiftInX
//----------------------------------------------------------------------------------------------------------------------


/**
 * Compute the velocity shift in Fourier space over the X axis.
 */
void SolverCudaKernels::computeVelocityShiftInX(CufftComplexMatrix&  cufftShiftTemp,
                                                const ComplexMatrix& xShiftNegR)
{
  cudaComputeVelocityShiftInX<<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                             (reinterpret_cast<cuFloatComplex*>  (cufftShiftTemp.getDeviceData()),
                              reinterpret_cast<const cuFloatComplex*> (xShiftNegR.getDeviceData()));
  // check for errors
  cudaCheckErrors(cudaGetLastError());
 }// end of computeVelocityShiftInX
//----------------------------------------------------------------------------------------------------------------------



/**
 * Cuda kernel to compute velocity shift in Y. The matrix is XY transposed.
 *
 * @param [in,out] cufftShiftTemp - Matrix to shift, shifted matrix.
 * @param [in]     yShiftNegR     - negative Fourier shift.
 */
__global__ void cudaComputeVelocityShiftInY(cuFloatComplex*       cufftShiftTemp,
                                            const cuFloatComplex* yShiftNegR)
{
  const auto nyR       = cudaDeviceConstants.ny / 2 + 1;
  const auto nElements = cudaDeviceConstants.nx * nyR * cudaDeviceConstants.nz;

  for (auto i = getIndex(); i < nElements; i += getStride())
  {
    // rotated dimensions
    const auto  y = i % nyR;

    cufftShiftTemp[i] = cuCmulf(cufftShiftTemp[i], yShiftNegR[y]) * cudaDeviceConstants.fftDividerY;
  }
}// end of cudaComputeVelocityShiftInY
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute the velocity shift in Fourier space over the Y axis.
 */
void SolverCudaKernels::computeVelocityShiftInY(CufftComplexMatrix&  cufftShiftTemp,
                                                const ComplexMatrix& yShiftNegR)
{
  cudaComputeVelocityShiftInY<<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                             (reinterpret_cast<cuFloatComplex*>       (cufftShiftTemp.getDeviceData()),
                              reinterpret_cast<const cuFloatComplex*> (yShiftNegR.getDeviceData()));
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of ComputeVelocityShiftInY
//----------------------------------------------------------------------------------------------------------------------


/**
 * Cuda kernel to compute velocity shift in Z. The matrix is XZ transposed.
 *
 * @param [in,out] cufftShiftTemp - Matrix to shift, shifted matrix.
 * @param [in]     zShiftNegR     - negative Fourier shift.
 */

__global__ void cudaComputeVelocityShiftInZ(cuFloatComplex*       cufftShiftTemp,
                                            const cuFloatComplex* zShiftNegR)
{
  const auto nzR       = cudaDeviceConstants.nz / 2 + 1;
  const auto nElements = cudaDeviceConstants.nx * cudaDeviceConstants.ny * nzR;

  for (auto i = getIndex(); i < nElements; i += getStride())
  {
    // rotated dimensions
    const auto  z = i % nzR;

    cufftShiftTemp[i] = cuCmulf(cufftShiftTemp[i], zShiftNegR[z]) * cudaDeviceConstants.fftDividerZ;
  }
}// end of cudaComputeVelocityShiftInZ
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute the velocity shift in Fourier space over the Z axis.
 */
void SolverCudaKernels::computeVelocityShiftInZ(CufftComplexMatrix&  cufftShiftTemp,
                                                const ComplexMatrix& zShiftNegR)
{
  cudaComputeVelocityShiftInZ<<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                             (reinterpret_cast<cuFloatComplex*>       (cufftShiftTemp.getDeviceData()),
                              reinterpret_cast<const cuFloatComplex*> (zShiftNegR.getDeviceData()));
  // check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeVelocityShiftInZ
//----------------------------------------------------------------------------------------------------------------------
