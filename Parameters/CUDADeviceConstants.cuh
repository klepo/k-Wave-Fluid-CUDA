/**
 * @file        CUDADeviceConstants.cuh
 * 
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file for the class for storing constants residing in CUDA constant memory.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        17 February 2016, 10:53 (created) \n
 *              25 July     2016, 12:55 (revised)
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


#ifndef CUDA_DEVICE_CONSTANTS_CUH
#define CUDA_DEVICE_CONSTANTS_CUH

/**
  * @struct TCUDADeviceConstants
  * @brief  Structure for CUDA parameters to be placed in constant memory. Only 32b values are used,
  *         since CUDA does not allow to allocate more than 2^32 elements and dim3 datatype
  *         is based on unsigned int.
  */
struct TCUDADeviceConstants
{
  /// Set constant memory
  __host__ void SetUpCUDADeviceConstatns();

  /// size of X dimension.
  unsigned int nx;
  /// size of Y dimension.
  unsigned int ny;
  /// size of Z dimension.
  unsigned int nz;
  /// total number of elements.
  unsigned int nElements;
  /// 2D Slab size
  unsigned int slabSize;
  /// size of complex X dimension.
  unsigned int nxComplex;
  /// size of complex Y dimension.
  unsigned int nyComplex;
  /// size of complex Z dimension.
  unsigned int nzComplex;
  /// complex number of elements.
  unsigned int nElementsComplex;
  /// complex slab size.
  unsigned int slabSizeComplex;
  /// normalization constant for 3D FFT.
  float  fftDivider;
  /// normalization constant for 1D FFT over X.
  float  fftDividerX;
  /// normalization constant for 1D FFT over Y.
  float  fftDividerY;
  /// normalization constant for 1D FFT over Z.
  float  fftDividerZ;

  /// dt
  float dt;
  /// 2.0 * dt
  float dt2;
  /// c^2
  float c2;

  /// rho0 in homogeneous case
  float rho0_scalar;
  /// dt * rho0 in homogeneous case
  float dt_rho0_scalar;
  /// dt / rho0_sgx in homogeneous case
  float rho0_sgx_scalar;
  /// dt / rho0_sgy in homogeneous case
  float rho0_sgy_scalar;
  /// dt / rho0_sgz in homogeneous case
  float rho0_sgz_scalar;

  /// BonA value for homogeneous case
  float BonA_scalar;

  /// Absorb_tau value for homogeneous case
  float absorb_tau_scalar;
  /// Absorb_eta value for homogeneous case
  float absorb_eta_scalar;

  ///  size of the u source index
  unsigned int u_source_index_size;
  /// u source mode
  unsigned int u_source_mode;
  /// u source many
  unsigned int u_source_many;

  /// size of the p_source mask
  unsigned int p_source_index_size;
  /// p source mode
  unsigned int p_source_mode;
  /// p source many
  unsigned int p_source_many;

}; // end of TCUDAConstants
//--------------------------------------------------------------------------------------------------

#endif /* CUDA_DEVICE_CONSTANTS_CUH */

