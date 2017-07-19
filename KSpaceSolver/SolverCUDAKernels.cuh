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
 *              19 July     2017, 12:08 (revised)
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

#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>
#include <MatrixClasses/IndexMatrix.h>
#include <MatrixClasses/CufftComplexMatrix.h>

#include <Utils/DimensionSizes.h>

#include <Parameters/Parameters.h>
#include <Parameters/CudaParameters.h>

/**
 * @namespace   SolverCUDAKernels
 * @brief       List of cuda kernels used k-space first order 3D solver
 * @details     List of cuda kernels used k-space first order 3D solver
 *
 */
namespace SolverCUDAKernels
{
  /**
   * @enum TransposePadding
   * @brief How is the data during matrix transposition padded.
   */
  enum class TransposePadding
  {
    kNone,       ///< none
    kInput,      ///< input matrix is padded
    kOutput,     ///< output matrix is padded
    kInputOutput ///< both matrices are padded
  };

  /// Get the CUDA architecture and GPU code version the code was compiled with
  int GetCUDACodeVersion();

  //----------------------------------- velocity operations --------------------------------------//
  /// Compute acoustic velocity for default case (heterogeneous).
  void ComputeVelocity(RealMatrix&       ux_sgx,
                       RealMatrix&       uy_sgy,
                       RealMatrix&       uz_sgz,
                       const RealMatrix& ifft_x,
                       const RealMatrix& ifft_y,
                       const RealMatrix& ifft_z,
                       const RealMatrix& dt_rho0_sgx,
                       const RealMatrix& dt_rho0_sgy,
                       const RealMatrix& dt_rho0_sgz,
                       const RealMatrix& pml_x,
                       const RealMatrix& pml_y,
                       const RealMatrix& pml_z);


  /// Compute acoustic velocity, scalar and uniform case.
  void ComputeVelocityScalarUniform(RealMatrix&       ux_sgx,
                                    RealMatrix&       uy_sgy,
                                    RealMatrix&       uz_sgz,
                                    const RealMatrix& ifft_x,
                                    const RealMatrix& ifft_y,
                                    const RealMatrix& ifft_z,
                                    const RealMatrix& pml_x,
                                    const RealMatrix& pml_y,
                                    const RealMatrix& pml_z);

  /// Compute acoustic velocity, scalar, non-uniform case.
  void ComputeVelocityScalarNonuniform(RealMatrix&       ux_sgx,
                                       RealMatrix&       uy_sgy,
                                       RealMatrix&       uz_sgz,
                                       const RealMatrix& ifft_x,
                                       const RealMatrix& ifft_y,
                                       const RealMatrix& ifft_z,
                                       const RealMatrix& dxudxn_sgx,
                                       const RealMatrix& dyudyn_sgy,
                                       const RealMatrix& dzudzn_sgz,
                                       const RealMatrix& pml_x,
                                       const RealMatrix& pml_y,
                                       const RealMatrix& pml_z);

  //----------------------------------------- Sources --------------------------------------------//
  /// Add transducer data  source to X component.
  void AddTransducerSource(RealMatrix&        ux_sgx,
                           const IndexMatrix& u_source_index,
                           IndexMatrix&       delay_mask,
                           const RealMatrix & transducer_signal);

  /// Add in velocity source terms.
  void AddVelocitySource(RealMatrix&        uxyz_sgxyz,
                         const RealMatrix&  u_source_input,
                         const IndexMatrix& u_source_index,
                         const size_t        t_index);

  /// Add in pressure source term
  void AddPressureSource(RealMatrix&        rhox,
                         RealMatrix&        rhoy,
                         RealMatrix&        rhoz,
                         const RealMatrix&  p_source_input,
                         const IndexMatrix& p_source_index,
                         const size_t        t_index);

  /// Compute velocity for the initial pressure problem.
  void Compute_p0_Velocity(RealMatrix&       ux_sgx,
                           RealMatrix&       uy_sgy,
                           RealMatrix&       uz_sgz,
                           const RealMatrix& dt_rho0_sgx,
                           const RealMatrix& dt_rho0_sgy,
                           const RealMatrix& dt_rho0_sgz);

  /// Compute  acoustic velocity for initial pressure problem, if rho0_sgx is scalar, uniform grid.
  void Compute_p0_Velocity(RealMatrix& ux_sgx,
                           RealMatrix& uy_sgy,
                           RealMatrix& uz_sgz);


  /// Compute  acoustic velocity for initial pressure problem, if rho0_sgx is scalar, non uniform grid, x component.
  void Compute_p0_VelocityScalarNonUniform(RealMatrix&      ux_sgx,
                                           RealMatrix&      uy_sgy,
                                           RealMatrix&      uz_sgz,
                                           const RealMatrix& dxudxn_sgx,
                                           const RealMatrix& dyudyn_sgy,
                                           const RealMatrix& dzudzn_sgz);


  //------------------------------------- pressure kernels ---------------------------------------//
   /// Compute part of the new velocity - gradient of p.
   void ComputePressurelGradient(CufftComplexMatrix&  fft_x,
                                 CufftComplexMatrix&  fft_y,
                                 CufftComplexMatrix&  fft_z,
                                 const RealMatrix&    kappa,
                                 const ComplexMatrix& ddx,
                                 const ComplexMatrix& ddy,
                                 const ComplexMatrix& ddz);

  /// Compute gradient of acoustic velocity on uniform grid.
  void ComputeVelocityGradient(CufftComplexMatrix&  fft_x,
                               CufftComplexMatrix&  fft_y,
                               CufftComplexMatrix&  fft_z,
                               const RealMatrix&    kappa,
                               const ComplexMatrix& ddx_k_shift_neg,
                               const ComplexMatrix& ddy_k_shift_neg,
                               const ComplexMatrix& ddz_k_shift_neg);

  ///  Shift gradient of acoustic velocity on non-uniform grid.
  void ComputeVelocityGradientNonuniform(RealMatrix&        duxdx,
                                         RealMatrix&        duydy,
                                         RealMatrix&        duzdz,
                                         const RealMatrix& dxudxn,
                                         const RealMatrix& dyudyn,
                                         const RealMatrix& dzudzn);

    /// Add initial pressure to p0 (as p0 source).
  void Compute_p0_AddInitialPressure(RealMatrix&       p,
                                     RealMatrix&       rhox,
                                     RealMatrix&       rhoy,
                                     RealMatrix&       rhoz,
                                     const RealMatrix& p0,
                                     const bool         Is_c2_scalar,
                                     const float*       c2);

  //------------------------------------- density kernels ----------------------------------------//
  /// Calculate acoustic density for non-linear case, homogenous case.
  void ComputeDensityNonlinearHomogeneous(RealMatrix&       rhox,
                                          RealMatrix&       rhoy,
                                          RealMatrix&       rhoz,
                                          const RealMatrix& pml_x,
                                          const RealMatrix& pml_y,
                                          const RealMatrix& pml_z,
                                          const RealMatrix& duxdx,
                                          const RealMatrix& duydy,
                                          const RealMatrix& duzdz);

  /// Calculate acoustic density for non-linear case, heterogenous case.
  void ComputeDensityNonlinearHeterogeneous(RealMatrix&       rhox,
                                            RealMatrix&       rhoy,
                                            RealMatrix&       rhoz,
                                            const RealMatrix& pml_x,
                                            const RealMatrix& pml_y,
                                            const RealMatrix& pml_z,
                                            const RealMatrix& duxdx,
                                            const RealMatrix& duydy,
                                            const RealMatrix& duzdz,
                                            const RealMatrix& rho0);

  /// Calculate acoustic density for linear case, homogenous case.
  void ComputeDensityLinearHomogeneous(RealMatrix&       rhox,
                                       RealMatrix&       rhoy,
                                       RealMatrix&       rhoz,
                                       const RealMatrix& pml_x,
                                       const RealMatrix& pml_y,
                                       const RealMatrix& pml_z,
                                       const RealMatrix& duxdx,
                                       const RealMatrix& duydy,
                                       const RealMatrix& duzdz);

  /// Calculate acoustic density for linear case, heterogeneous case.
  void ComputeDensityLinearHeterogeneous(RealMatrix&       rhox,
                                          RealMatrix&       rhoy,
                                          RealMatrix&       rhoz,
                                          const RealMatrix& pml_x,
                                          const RealMatrix& pml_y,
                                          const RealMatrix& pml_z,
                                          const RealMatrix& duxdx,
                                          const RealMatrix& duydy,
                                          const RealMatrix& duzdz,
                                          const RealMatrix& rho0);

  //---------------------------------- new value of pressure -------------------------------------//
  /// Calculate three temporary sums in the new pressure formula, non-linear absorbing case.
  void ComputePressurePartsNonLinear(RealMatrix&       rho_sum,
                                     RealMatrix&       BonA_sum,
                                     RealMatrix&       du_sum,
                                     const RealMatrix& rhox,
                                     const RealMatrix& rhoy,
                                     const RealMatrix& rhoz,
                                     const RealMatrix& duxdx,
                                     const RealMatrix& duydy,
                                     const RealMatrix& duzdz,
                                     const bool         is_BonA_scalar,
                                     const float*       BonA_matrix,
                                     const bool         is_rho0_scalar,
                                     const float*       rho0_matrix);

  /// Compute absorbing term with abosrb_nabla1 and absorb_nabla2.
  void ComputeAbsorbtionTerm(CufftComplexMatrix& fft1,
                             CufftComplexMatrix& fft2,
                             const RealMatrix&   absorb_nabla1,
                             const RealMatrix&   absorb_nabla2);

  /// Sum sub-terms to calculate new pressure, non-linear case.
  void SumPressureTermsNonlinear(RealMatrix&       p,
                                 const RealMatrix& BonA_temp,
                                 const bool         is_c2_scalar,
                                 const float*       c2_matrix,
                                 const bool         is_tau_eta_scalar,
                                 const float*       absorb_tau,
                                 const float*       tau_matrix,
                                 const float*       absorb_eta,
                                 const float*       eta_matrix);

  /// Sum sub-terms to calculate new pressure, linear case.
  void SumPressureTermsLinear(RealMatrix&       p,
                              const RealMatrix& absorb_tau_temp,
                              const RealMatrix& absorb_eta_temp,
                              const RealMatrix& sum_rhoxyz,
                              const bool         is_c2_scalar,
                              const float*       c2_matrix,
                              const bool         is_tau_eta_scalar,
                              const float*       tau_matrix,
                              const float*       eta_matrix);

  /// Sum sub-terms for new p, linear lossless case.
  void SumPressureNonlinearLossless(RealMatrix&       p,
                                    const RealMatrix& rhox,
                                    const RealMatrix& rhoy,
                                    const RealMatrix& rhoz,
                                    const bool         is_c2_scalar,
                                    const float*       c2_matrix,
                                    const bool         is_BonA_scalar,
                                    const float*       BonA_matrix,
                                    const bool         is_rho0_scalar,
                                    const float*       rho0_matrix);

  /// Calculate two temporary sums in the new pressure formula, linear absorbing case.
  void ComputePressurePartsLinear(RealMatrix&       sum_rhoxyz,
                                  RealMatrix&       sum_rho0_du,
                                  const RealMatrix& rhox,
                                  const RealMatrix& rhoy,
                                  const RealMatrix& rhoz,
                                  const RealMatrix& duxdx,
                                  const RealMatrix& duydy,
                                  const RealMatrix& duzdz,
                                  const bool         is_rho0_scalar,
                                  const float*       rho0_matrix);

  /// Sum sub-terms for new p, linear lossless case.
  void SumPressureLinearLossless(RealMatrix&       p,
                                 const RealMatrix& rhox,
                                 const RealMatrix& rhoy,
                                 const RealMatrix& rhoz,
                                 const bool         is_c2_scalar,
                                 const float*       c2_matrix);

  //----------------------------------- unstaggered velocity -------------------------------------//

  /// Transpose a real 3D matrix in the X-Y direction
  template<TransposePadding padding>
  void TrasposeReal3DMatrixXY(float*       outputMatrix,
                              const float* inputMatrix,
                              const dim3&  dimSizes);

  /// Transpose a real 3D matrix in the X-Y direction
  template<TransposePadding padding>
  void TrasposeReal3DMatrixXZ(float*       outputMatrix,
                              const float* inputMatrix,
                              const dim3&  dimSizes);

  /// Compute the velocity shift in Fourier space over the X axis
  void ComputeVelocityShiftInX(CufftComplexMatrix&  cufft_shift_temp,
                               const ComplexMatrix& x_shift_neg_r);
  /// Compute the velocity shift in Fourier space over the Y axis
  void ComputeVelocityShiftInY(CufftComplexMatrix&  cufft_shift_temp,
                               const ComplexMatrix& y_shift_neg_r);
  /// Compute the velocity shift in Fourier space over the Z axis
  void ComputeVelocityShiftInZ(CufftComplexMatrix&  cufft_shift_temp,
                               const ComplexMatrix& z_shift_neg_r);
}

#endif /*SOLVER_CUDA_KERNELS_CUH*/
