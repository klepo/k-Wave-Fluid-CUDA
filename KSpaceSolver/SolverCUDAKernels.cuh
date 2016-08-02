/**
 * @file        SolverCUDAKernels.cuh
 *
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       Name space for all CUDA kernels used in the 3D solver
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        11 March    2013, 13:10 (created) \n
 *              27 July     2016, 15:09 (revised)
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

#ifndef SOLVER_CUDA_KERNELS_CUH
#define	SOLVER_CUDA_KERNELS_CUH

//#include <iostream>

#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>
#include <MatrixClasses/IndexMatrix.h>
#include <MatrixClasses/CUFFTComplexMatrix.h>

#include <Utils/DimensionSizes.h>

#include <Parameters/Parameters.h>
#include <Parameters/CUDAParameters.h>

/**
 * @namespace   SolverCUDAKernels
 * @brief       List of cuda kernels used k-space first order 3D solver
 * @details     List of cuda kernels used k-space first order 3D solver
 *
 */
namespace SolverCUDAKernels
{
  /// Get the CUDA architecture and GPU code version the code was compiled with
  int GetCUDACodeVersion();

  //----------------------------------- velocity operations --------------------------------------//
  /// Compute acoustic velocity for default case (heterogeneous).
  void ComputeVelocity(TRealMatrix&       ux_sgx,
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
                       const TRealMatrix& pml_z);


  /// Compute acoustic velocity, scalar and uniform case.
  void ComputeVelocityScalarUniform(TRealMatrix&       ux_sgx,
                                    TRealMatrix&       uy_sgy,
                                    TRealMatrix&       uz_sgz,
                                    const TRealMatrix& ifft_x,
                                    const TRealMatrix& ifft_y,
                                    const TRealMatrix& ifft_z,
                                    const TRealMatrix& pml_x,
                                    const TRealMatrix& pml_y,
                                    const TRealMatrix& pml_z);

  /// Compute acoustic velocity, scalar, non-uniform case.
  void ComputeVelocityScalarNonuniform(TRealMatrix&       ux_sgx,
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
                                       const TRealMatrix& pml_z);

  //----------------------------------------- Sources --------------------------------------------//
  /// Add transducer data  source to X component.
  void AddTransducerSource(TRealMatrix&        ux_sgx,
                           const TIndexMatrix& u_source_index,
                           TIndexMatrix&       delay_mask,
                           const TRealMatrix & transducer_signal);

  /// Add in velocity source terms.
  void AddVelocitySource(TRealMatrix&        uxyz_sgxyz,
                         const TRealMatrix&  u_source_input,
                         const TIndexMatrix& u_source_index,
                         const size_t        t_index);

  /// Add in pressure source term
  void AddPressureSource(TRealMatrix&        rhox,
                         TRealMatrix&        rhoy,
                         TRealMatrix&        rhoz,
                         const TRealMatrix&  p_source_input,
                         const TIndexMatrix& p_source_index,
                         const size_t        t_index);

  /// Compute velocity for the initial pressure problem.
  void Compute_p0_Velocity(TRealMatrix&       ux_sgx,
                           TRealMatrix&       uy_sgy,
                           TRealMatrix&       uz_sgz,
                           const TRealMatrix& dt_rho0_sgx,
                           const TRealMatrix& dt_rho0_sgy,
                           const TRealMatrix& dt_rho0_sgz);

  /// Compute  acoustic velocity for initial pressure problem, if rho0_sgx is scalar, uniform grid.
  void Compute_p0_Velocity(TRealMatrix& ux_sgx,
                           TRealMatrix& uy_sgy,
                           TRealMatrix& uz_sgz);


  /// Compute  acoustic velocity for initial pressure problem, if rho0_sgx is scalar, non uniform grid, x component.
  void Compute_p0_VelocityScalarNonUniform(TRealMatrix&      ux_sgx,
                                           TRealMatrix&      uy_sgy,
                                           TRealMatrix&      uz_sgz,
                                           const TRealMatrix& dxudxn_sgx,
                                           const TRealMatrix& dyudyn_sgy,
                                           const TRealMatrix& dzudzn_sgz);


  //------------------------------------- pressure kernels ---------------------------------------//
   /// Compute part of the new velocity - gradient of p.
   void ComputePressurelGradient(TCUFFTComplexMatrix&  fft_x,
                                 TCUFFTComplexMatrix&  fft_y,
                                 TCUFFTComplexMatrix&  fft_z,
                                 const TRealMatrix&    kappa,
                                 const TComplexMatrix& ddx,
                                 const TComplexMatrix& ddy,
                                 const TComplexMatrix& ddz);

  /// Compute gradient of acoustic velocity on uniform grid.
  void ComputeVelocityGradient(TCUFFTComplexMatrix&  fft_x,
                               TCUFFTComplexMatrix&  fft_y,
                               TCUFFTComplexMatrix&  fft_z,
                               const TRealMatrix&    kappa,
                               const TComplexMatrix& ddx_k_shift_neg,
                               const TComplexMatrix& ddy_k_shift_neg,
                               const TComplexMatrix& ddz_k_shift_neg);

  ///  Shift gradient of acoustic velocity on non-uniform grid.
  void ComputeVelocityGradientNonuniform(TRealMatrix&        duxdx,
                                         TRealMatrix&        duydy,
                                         TRealMatrix&        duzdz,
                                         const TRealMatrix& dxudxn,
                                         const TRealMatrix& dyudyn,
                                         const TRealMatrix& dzudzn);

    /// Add initial pressure to p0 (as p0 source).
  void Compute_p0_AddInitialPressure(TRealMatrix&       p,
                                     TRealMatrix&       rhox,
                                     TRealMatrix&       rhoy,
                                     TRealMatrix&       rhoz,
                                     const TRealMatrix& p0,
                                     const bool         Is_c2_scalar,
                                     const float*       c2);

  //------------------------------------- density kernels ----------------------------------------//
  /// Calculate acoustic density for non-linear case, homogenous case.
  void ComputeDensityNonlinearHomogeneous(TRealMatrix&       rhox,
                                          TRealMatrix&       rhoy,
                                          TRealMatrix&       rhoz,
                                          const TRealMatrix& pml_x,
                                          const TRealMatrix& pml_y,
                                          const TRealMatrix& pml_z,
                                          const TRealMatrix& duxdx,
                                          const TRealMatrix& duydy,
                                          const TRealMatrix& duzdz);

  /// Calculate acoustic density for non-linear case, heterogenous case.
  void ComputeDensityNonlinearHeterogeneous(TRealMatrix&       rhox,
                                            TRealMatrix&       rhoy,
                                            TRealMatrix&       rhoz,
                                            const TRealMatrix& pml_x,
                                            const TRealMatrix& pml_y,
                                            const TRealMatrix& pml_z,
                                            const TRealMatrix& duxdx,
                                            const TRealMatrix& duydy,
                                            const TRealMatrix& duzdz,
                                            const TRealMatrix& rho0);

  /// Calculate acoustic density for linear case, homogenous case.
  void ComputeDensityLinearHomogeneous(TRealMatrix&       rhox,
                                       TRealMatrix&       rhoy,
                                       TRealMatrix&       rhoz,
                                       const TRealMatrix& pml_x,
                                       const TRealMatrix& pml_y,
                                       const TRealMatrix& pml_z,
                                       const TRealMatrix& duxdx,
                                       const TRealMatrix& duydy,
                                       const TRealMatrix& duzdz);

  /// Calculate acoustic density for linear case, heterogeneous case.
  void ComputeDensityLinearHeterogeneous(TRealMatrix&       rhox,
                                          TRealMatrix&       rhoy,
                                          TRealMatrix&       rhoz,
                                          const TRealMatrix& pml_x,
                                          const TRealMatrix& pml_y,
                                          const TRealMatrix& pml_z,
                                          const TRealMatrix& duxdx,
                                          const TRealMatrix& duydy,
                                          const TRealMatrix& duzdz,
                                          const TRealMatrix& rho0);

  //---------------------------------- new value of pressure -------------------------------------//
  /// Calculate three temporary sums in the new pressure formula, non-linear absorbing case.
  void ComputePressurePartsNonLinear(TRealMatrix&       rho_sum,
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
                                     const float*       rho0_matrix);

  /// Compute absorbing term with abosrb_nabla1 and absorb_nabla2.
  void ComputeAbsorbtionTerm(TCUFFTComplexMatrix& fft1,
                             TCUFFTComplexMatrix& fft2,
                             const TRealMatrix&   absorb_nabla1,
                             const TRealMatrix&   absorb_nabla2);

  /// Sum sub-terms to calculate new pressure, non-linear case.
  void SumPressureTermsNonlinear(TRealMatrix&       p,
                                 const TRealMatrix& BonA_temp,
                                 const bool         is_c2_scalar,
                                 const float*       c2_matrix,
                                 const bool         is_tau_eta_scalar,
                                 const float*       absorb_tau,
                                 const float*       tau_matrix,
                                 const float*       absorb_eta,
                                 const float*       eta_matrix);

  /// Sum sub-terms to calculate new pressure, linear case.
  void SumPressureTermsLinear(TRealMatrix&       p,
                              const TRealMatrix& absorb_tau_temp,
                              const TRealMatrix& absorb_eta_temp,
                              const TRealMatrix& sum_rhoxyz,
                              const bool         is_c2_scalar,
                              const float*       c2_matrix,
                              const bool         is_tau_eta_scalar,
                              const float*       tau_matrix,
                              const float*       eta_matrix);

  /// Sum sub-terms for new p, linear lossless case.
  void SumPressureNonlinearLossless(TRealMatrix&       p,
                                    const TRealMatrix& rhox,
                                    const TRealMatrix& rhoy,
                                    const TRealMatrix& rhoz,
                                    const bool         is_c2_scalar,
                                    const float*       c2_matrix,
                                    const bool         is_BonA_scalar,
                                    const float*       BonA_matrix,
                                    const bool         is_rho0_scalar,
                                    const float*       rho0_matrix);

  /// Calculate two temporary sums in the new pressure formula, linear absorbing case.
  void ComputePressurePartsLinear(TRealMatrix&       sum_rhoxyz,
                                  TRealMatrix&       sum_rho0_du,
                                  const TRealMatrix& rhox,
                                  const TRealMatrix& rhoy,
                                  const TRealMatrix& rhoz,
                                  const TRealMatrix& duxdx,
                                  const TRealMatrix& duydy,
                                  const TRealMatrix& duzdz,
                                  const bool         is_rho0_scalar,
                                  const float*       rho0_matrix);

  /// Sum sub-terms for new p, linear lossless case.
  void SumPressureLinearLossless(TRealMatrix&       p,
                                 const TRealMatrix& rhox,
                                 const TRealMatrix& rhoy,
                                 const TRealMatrix& rhoz,
                                 const bool         is_c2_scalar,
                                 const float*       c2_matrix);

  //----------------------------------- unstaggered velocity -------------------------------------//

  /// Transpose a real 3D matrix in the X-Y direction
  void TrasposeReal3DMatrixXY(float*       outputMatrix,
                              const float* inputMatrix,
                              const dim3&  dimSizes);

  /// Transpose a real 3D matrix in the X-Y direction
  void TrasposeReal3DMatrixXZ(float*       outputMatrix,
                              const float* inputMatrix,
                              const dim3&  dimSizes);

  /// Compute the velocity shift in Fourier space over the X axis
  void ComputeVelocityShiftInX(TCUFFTComplexMatrix&  cufft_shift_temp,
                               const TComplexMatrix& x_shift_neg_r);
  /// Compute the velocity shift in Fourier space over the Y axis
  void ComputeVelocityShiftInY(TCUFFTComplexMatrix&  cufft_shift_temp,
                               const TComplexMatrix& y_shift_neg_r);
  /// Compute the velocity shift in Fourier space over the Z axis
  void ComputeVelocityShiftInZ(TCUFFTComplexMatrix&  cufft_shift_temp,
                               const TComplexMatrix& z_shift_neg_r);
}

#endif /*SOLVER_CUDA_KERNELS_CUH*/
