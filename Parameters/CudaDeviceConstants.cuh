/**
 * @file        CudaDeviceConstants.cuh
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
 *              12 July     2017, 10:13 (revised)
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


#ifndef CudaDeviceConstantsCuh
#define CudaDeviceConstantsCuh

/**
  * @struct CudaDeviceConstants
  * @brief  Structure for CUDA parameters to be placed in constant memory. Only 32b values are used,
  *         since CUDA does not allow to allocate more than 2^32 elements and dim3 datatype
  *         is based on unsigned int.
  */
struct CudaDeviceConstants
{
  /// Upload device constants into GPU memory
  __host__ void uploadDeviceConstants();

  /// size of X dimension.
  unsigned int nx;
  /// size of Y dimension.
  unsigned int ny;
  /// size of Z dimension.
  unsigned int nz;
  /// total number of elements.
  unsigned int nElements;
  /// size of complex X dimension.
  unsigned int nxComplex;
  /// size of complex Y dimension.
  unsigned int nyComplex;
  /// size of complex Z dimension.
  unsigned int nzComplex;
  /// complex number of elements.
  unsigned int nElementsComplex;
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
  float dtBy2;
  /// c^2
  float cSquare;

  /// rho0 in homogeneous case
  float rho0;
  /// dt * rho0 in homogeneous case
  float dtRho0;
  /// dt / rho0_sgx in homogeneous case
  float dtRho0Sgx;
  /// dt / rho0_sgy in homogeneous case
  float dtRho0Sgy;
  /// dt / rho0_sgz in homogeneous case
  float dtRho0Sgz;

  /// BonA value for homogeneous case
  float bOnA;

  /// Absorb_tau value for homogeneous case
  float absorbTau;
  /// Absorb_eta value for homogeneous case
  float absorbEta;

  ///  size of the u source index
  unsigned int velocitySourceSize;
  /// u source mode
  unsigned int velocitySourceMode;
  /// u source many
  unsigned int velocitySourceMany;

  /// size of the p_source mask
  unsigned int presureSourceSize;
  /// p source mode
  unsigned int presureSourceMode;
  /// p source many
  unsigned int presureSourceMany;

}; // end of CudaDeviceConstants
//----------------------------------------------------------------------------------------------------------------------

#endif /* CudaDeviceConstantsCuh */

