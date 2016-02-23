/**
 * @file        CUDADeviceConstants.cuh
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file for the class for storing constants residing in CUDA constant memory.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        17 February 2016, 10:53 (created) \n
 *              23 February 2016, 13:40 (revised)
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


#ifndef CUDA_DEVICE_CONSTANTS_CUH
#define CUDA_DEVICE_CONSTANTS_CUH

 /**
   * @struct TCUDAConstants
   * @brief  Structure for CUDA parameters to be placed in constant memory
 */
struct TCUDADeviceConstants
{
  /// size of X dimension.
  size_t Nx;
  /// size of Y dimension.
  size_t Ny;
  /// size of Z dimension.
  size_t Nz;
  /// total number of elements.
  size_t TotalElementCount;
  /// 2D Slab size
  size_t SlabSize;
  /// size of complex X dimension.
  size_t Complex_Nx;
  /// size of complex Y dimension.
  size_t Complex_Ny;
  /// size of complex Z dimension.
  size_t Complex_Nz;
  /// complex number of elements.
  size_t ComplexTotalElementCount;
  /// complex slab size.
  size_t ComplexSlabSize;
   /// normalization constant for 3D FFT.
  float  FFTDivider;
  /// normalization constant for 1D FFT over X.
  float  FFTDividerX;
  /// normalization constant for 1D FFT over Y.
  float  FFTDividerY;
  /// normalization constant for 1D FFT over Z.
  float  FFTDividerZ;

  /// dt
  float dt;
  /// 2.0 * dt
  float dt2;


  /// dt * rho0 in homogeneous case
  float dt_rho0_scalar;
  /// dt / rho0_sgx in homogeneous case
  float rho0_sgx_scalar;
  /// dt / rho0_sgy in homogeneous case
  float rho0_sgy_scalar;
  /// dt / rho0_sgz in homogeneous case
  float rho0_sgz_scalar;


  ///  size of the u source index
  size_t u_source_index_size;
  /// u source mode
  size_t u_source_mode;
  /// u source many
  size_t u_source_many;

  /// size of the p_source mask
  size_t p_source_index_size;
  /// p source mode
  size_t p_source_mode;
  /// p source many
  size_t p_source_many;

  /// Set constant memory
  __host__ void SetUpCUDADeviceConstatns();
}; // end of TCUDAConstants




#endif /* CUDA_DEVICE_CONSTANTS_CUH */

