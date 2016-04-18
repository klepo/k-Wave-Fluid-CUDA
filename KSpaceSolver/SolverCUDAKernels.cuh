/**
 * @file        SolverCUDAKernels.cuh
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       Name space for all CUDA kernels used in the 3D solver
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        11 March    2013, 13:10 (created) \n
 *              18 April    2016, 14:36 (revised)
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

#ifndef SOLVER_CUDA_KERNELS_CUH
#define	SOLVER_CUDA_KERNELS_CUH

#include <iostream>

#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>
#include <MatrixClasses/IndexMatrix.h>
#include <MatrixClasses/CUFFTComplexMatrix.h>

#include <Utils/DimensionSizes.h>

#include <Parameters/Parameters.h>
#include <Parameters/CUDAParameters.h>

/**
 * @namespace   SolverCUDAKernels
 * @brief       List of cuda kernels used k-space FirstOrder 3D solver
 * @details     List of cuda kernels used kspace FirstOrder 3D solver
 *
 */
namespace SolverCUDAKernels
{
  /// Get the CUDA architecture and GPU code version the code was compiled with
  int GetCUDACodeVersion();

  /// Compute a new value of ux_sgx,uy_sgy, uy_sgz default case (heterogeneous).
  void Compute_uxyz_normalize(TRealMatrix&       ux_sgx,
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
                              const TRealMatrix& pml_z);


  /// Compute a new value of ux_sgx,uy_sgy, uy_sgz, scalar and uniform case.
  void Compute_uxyz_normalize_scalar_uniform(TRealMatrix&       ux_sgx,
                                             TRealMatrix&       uy_sgy,
                                             TRealMatrix&       uz_sgz,
                                             const TRealMatrix& FFT_X,
                                             const TRealMatrix& FFT_Y,
                                             const TRealMatrix& FFT_Z,
                                             const TRealMatrix& pml_x,
                                             const TRealMatrix& pml_y,
                                             const TRealMatrix& pml_z);

  /// Compute a new value of ux_sgx, scalar, non-uniform case.
  void Compute_uxyz_normalize_scalar_nonuniform(TRealMatrix&       ux_sgx,
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
                                                const TRealMatrix& pml_z);

  //------------------------ transducers -------------------------------------//
  /// Add transducer data  source to X component.
  void AddTransducerSource(TRealMatrix&        ux_sgx,
                           const TIndexMatrix& u_source_index,
                           TIndexMatrix&       delay_mask,
                           const TRealMatrix & transducer_signal);

  /// Add in velocity source terms.
  void Add_u_source(TRealMatrix&        uxyz_sgxyz,
                    const TRealMatrix&  u_source_input,
                    const TIndexMatrix& u_source_index,
                    const size_t        t_index);

  /// Add in pressure source term
  void Add_p_source(TRealMatrix&        rhox,
                    TRealMatrix&        rhoy,
                    TRealMatrix&        rhoz,
                    const TRealMatrix&  p_source_input,
                    const TIndexMatrix& p_source_index,
                    const size_t        t_index);

  //------------------velocity spectral operations ---------------------------//
  /// Compute u = dt ./ rho0_sg .* ifft (FFT).
  void Compute_dt_rho_sg_mul_ifft_div_2(TRealMatrix&       ux_sgx,
                                        TRealMatrix&       uy_sgy,
                                        TRealMatrix&       uz_sgz,
                                        const TRealMatrix& dt_rho0_sgx,
                                        const TRealMatrix& dt_rho0_sgy,
                                        const TRealMatrix& dt_rho0_sgz);





  /// Compute u = dt ./ rho0_sgx .* ifft (FFT), if rho0_sgx is scalar, uniform grid.
  void Compute_dt_rho_sg_mul_ifft_div_2(TRealMatrix& ux_sgx,
                                        TRealMatrix& uy_sgy,
                                        TRealMatrix& uz_sgz);


  /// Compute dt ./ rho0_sgx .* ifft (FFT), if rho0_sgx is scalar, non uniform grid, x component.
  void Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform(TRealMatrix&      ux_sgx,
                                                          TRealMatrix&      uy_sgy,
                                                          TRealMatrix&      uz_sgz,
                                                          const TRealMatrix& dxudxn_sgx,
                                                          const TRealMatrix& dyudyn_sgy,
                                                          const TRealMatrix& dzudzn_sgz);


   //------------------------- pressure kernels-------------------------------//
   /// Compute part of the new velocity - gradient of p.
   void Compute_ddx_kappa_fft_p(TCUFFTComplexMatrix&  FFT_X,
                                TCUFFTComplexMatrix&  FFT_Y,
                                TCUFFTComplexMatrix&  FFT_Z,
                                const TRealMatrix&    kappa,
                                const TComplexMatrix& ddx,
                                const TComplexMatrix& ddy,
                                const TComplexMatrix& ddz);

  /// Compute new values for duxdx, duydy, duzdz on uniform grid.
  void Compute_duxyz_uniform(TCUFFTComplexMatrix&  FFT_X,
                             TCUFFTComplexMatrix&  FFT_Y,
                             TCUFFTComplexMatrix&  FFT_Z,
                             const TRealMatrix&    kappa,
                             const TComplexMatrix& ddx_k_shift_neg,
                             const TComplexMatrix& ddy_k_shift_neg,
                             const TComplexMatrix& ddz_k_shift_neg);

  ///  Shift new values for duxdx, duydy, duzdz on non-uniform grid.
  void Compute_duxyz_non_uniform(TRealMatrix&        duxdx,
                                 TRealMatrix&        duydy,
                                 TRealMatrix&        duzdz,
                                 const TRealMatrix& dxudxn,
                                 const TRealMatrix& dyudyn,
                                 const TRealMatrix& dzudzn);



    /// Add initial pressure to p0 (as p0 source).
  void Calculate_p0_source_add_initial_pressure(TRealMatrix&       p,
                                                TRealMatrix&       rhox,
                                                TRealMatrix&       rhoy,
                                                TRealMatrix&       rhoz,
                                                const TRealMatrix& p0,
                                                const bool         Is_c2_scalar,
                                                const float*       c2);

  //---------------------- density kernels ---------------------------------//
  /// Calculate new values of rhox, rhoy and rhoz for non-linear case, homogenous case.
  void Compute_rhoxyz_nonlinear_homogeneous(TRealMatrix&       rhox,
                                            TRealMatrix&       rhoy,
                                            TRealMatrix&       rhoz,
                                            const TRealMatrix& pml_x,
                                            const TRealMatrix& pml_y,
                                            const TRealMatrix& pml_z,
                                            const TRealMatrix& duxdx,
                                            const TRealMatrix& duydy,
                                            const TRealMatrix& duzdz);

  /// Calculate new values of rhox, rhoy and rhoz for non-linear case, heterogenous case.
  void Compute_rhoxyz_nonlinear_heterogeneous(TRealMatrix&       rhox,
                                              TRealMatrix&       rhoy,
                                              TRealMatrix&       rhoz,
                                              const TRealMatrix& pml_x,
                                              const TRealMatrix& pml_y,
                                              const TRealMatrix& pml_z,
                                              const TRealMatrix& duxdx,
                                              const TRealMatrix& duydy,
                                              const TRealMatrix& duzdz,
                                              const TRealMatrix& rho0);

  /// Calculate new values of rhox, rhoy and rhoz for linear case, homogenous case.
  void Compute_rhoxyz_linear_homogeneous(TRealMatrix&       rhox,
                                         TRealMatrix&       rhoy,
                                         TRealMatrix&       rhoz,
                                         const TRealMatrix& pml_x,
                                         const TRealMatrix& pml_y,
                                         const TRealMatrix& pml_z,
                                         const TRealMatrix& duxdx,
                                         const TRealMatrix& duydy,
                                         const TRealMatrix& duzdz);

  /// Calculate new values of rhox, rhoy and rhoz for linear case, heterogeneous case.
  void Compute_rhoxyz_linear_heterogeneous(TRealMatrix&       rhox,
                                           TRealMatrix&       rhoy,
                                           TRealMatrix&       rhoz,
                                           const TRealMatrix& pml_x,
                                           const TRealMatrix& pml_y,
                                           const TRealMatrix& pml_z,
                                           const TRealMatrix& duxdx,
                                           const TRealMatrix& duydy,
                                           const TRealMatrix& duzdz,
                                           const TRealMatrix& rho0);

    //----------------------- new value of pressure --------------------------//
  /// Calculate three temporary sums in the new pressure formula, non-linear absorbing case.
  void Calculate_SumRho_BonA_SumDu(TRealMatrix&       rho_sum,
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
                                   const float*       rho0_matrix);

  /// Compute absorbing term with abosrb_nabla1 and absorb_nabla2.
  void Compute_Absorb_nabla1_2(TCUFFTComplexMatrix& FFT_1,
                               TCUFFTComplexMatrix& FFT_2,
                               const TRealMatrix&   absorb_nabla1,
                               const TRealMatrix&   absorb_nabla2);

  /// Sum sub-terms to calculate new pressure, non-linear case.
  void Sum_Subterms_nonlinear(TRealMatrix&       p,
                              const TRealMatrix& BonA_temp,
                              const bool         Is_c2_scalar,
                              const float*       c2_matrix,
                              const bool         Is_tau_eta_scalar,
                              const float*       Absorb_tau,
                              const float*       tau_matrix,
                              const float*       Absorb_eta,
                              const float*       eta_matrix);

  /// Sum sub-terms to calculate new pressure, linear case.
  void Sum_Subterms_linear(TRealMatrix&       p,
                           const TRealMatrix& Absorb_tau_temp,
                           const TRealMatrix& Absorb_eta_temp,
                           const TRealMatrix& Sum_rhoxyz,
                           const bool         Is_c2_scalar,
                           const float*       c2_matrix,
                           const bool         Is_tau_eta_scalar,
                           const float*       tau_matrix,
                           const float*       eta_matrix);

  /// Sum sub-terms for new p, linear lossless case.
  void Sum_new_p_nonlinear_lossless(TRealMatrix&       p,
                                    const TRealMatrix& rhox,
                                    const TRealMatrix& rhoy,
                                    const TRealMatrix& rhoz,
                                    const bool         Is_c2_scalar,
                                    const float*       c2_matrix,
                                    const bool         Is_BonA_scalar,
                                    const float*       BonA_matrix,
                                    const bool         Is_rho0_scalar,
                                    const float*       rho0_matrix);

  /// Calculate two temporary sums in the new pressure formula, linear absorbing case.
  void Calculate_SumRho_SumRhoDu(TRealMatrix&       Sum_rhoxyz,
                                 TRealMatrix&       Sum_rho0_du,
                                 const TRealMatrix& rhox,
                                 const TRealMatrix& rhoy,
                                 const TRealMatrix& rhoz,
                                 const TRealMatrix& duxdx,
                                 const TRealMatrix& duydy,
                                 const TRealMatrix& duzdz,
                                 const bool         Is_rho0_scalar,
                                 const float*       rho0_matrix);

  /// Sum sub-terms for new p, linear lossless case.
  void Sum_new_p_linear_lossless(TRealMatrix&       p,
                                 const TRealMatrix& rhox,
                                 const TRealMatrix& rhoy,
                                 const TRealMatrix& rhoz,
                                 const bool         Is_c2_scalar,
                                 const float*       c2_matrix);

  //------------------- unstaggered velocity ------------------------------//

  /// Transpose a real 3D matrix in the X-Y direction
  void TrasposeReal3DMatrixXY(float*       OutputMatrixData,
                              const float* InputMatrixData,
                              const dim3&  DimSizes);

  /// Transpose a real 3D matrix in the X-Y direction
  void TrasposeReal3DMatrixXZ(float*       OutputMatrixData,
                              const float* InputMatrixData,
                              const dim3&  DimSizes);

  /// Compute the velocity shift in Fourier space over the X axis
  void ComputeVelocityShiftInX(TCUFFTComplexMatrix&  FFT_shift_temp,
                               const TComplexMatrix& x_shift_neg_r);
  /// Compute the velocity shift in Fourier space over the Y axis
  void ComputeVelocityShiftInY(TCUFFTComplexMatrix&  FFT_shift_temp,
                               const TComplexMatrix& y_shift_neg_r);
  /// Compute the velocity shift in Fourier space over the Z axis
  void ComputeVelocityShiftInZ(TCUFFTComplexMatrix&  FFT_shift_temp,
                               const TComplexMatrix& z_shift_neg_r);
};

#endif /*SOLVER_CUDA_KERNELS_CUH*/
