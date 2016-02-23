/**
 * @file        CUDAImplementations.h
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
 *              13 February 2016, 13:47 (revised)
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

#ifndef CUDA_IMPLEMENTATIONS_H
#define	CUDA_IMPLEMENTATIONS_H

#include <iostream>

#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>
#include <MatrixClasses/IndexMatrix.h>
#include <MatrixClasses/CUFFTComplexMatrix.h>

#include <Utils/DimensionSizes.h>

#include <Parameters/Parameters.h>
#include <Parameters/CUDAParameters.h>

/**
 * @class TCUDAImplementations
 * @brief This singleton class implements interface for CUDA kernels.
 * @details This singleton class implements interface for CUDA kernels.
 */
class TCUDAImplementations
{
  public:

  /// Get instance of singleton class.
  static TCUDAImplementations* GetInstance();
  /// Destructor - may be virtual (once we merge OMP and CUDA).
  virtual ~TCUDAImplementations();

  /// Get the CUDA architecture and GPU code version the code was compiled with
  int GetCUDACodeVersion();

  //----------------------- ux calculation------------------------------------//

  /// compute a new value of ux_sgx, default case.
  void Compute_ux_sgx_normalize(TRealMatrix      & ux_sgx,
                                const TRealMatrix& FFT_p,
                                const TRealMatrix& dt_rho0,
                                const TRealMatrix& pml);

  /// Compute a new value of ux_sgx, scalar, uniform case.
  void Compute_ux_sgx_normalize_scalar_uniform(TRealMatrix      & ux_sgx,
                                               const TRealMatrix& FFT_p,
                                               const TRealMatrix& pml);

  /// Compute a new value of ux_sgx, scalar, non-uniform case.
  void Compute_ux_sgx_normalize_scalar_nonuniform(TRealMatrix      & ux_sgx,
                                                  const TRealMatrix& FFT_p,
                                                  const TRealMatrix& dxudxn_sgx,
                                                  const TRealMatrix& pml);

  //----------------------- uy calculation------------------------------------//
  /// compute a new value of uy_sgy, default case.
  void Compute_uy_sgy_normalize(TRealMatrix      & uy_sgy,
                                const TRealMatrix& FFT_p,
                                const TRealMatrix& dt_rho0,
                                const TRealMatrix& pml);

  /// Compute a new value of uy_sgy, scalar, uniform case.
  void Compute_uy_sgy_normalize_scalar_uniform(TRealMatrix      & uy_sgy,
                                               const TRealMatrix& FFT_p,
                                               const TRealMatrix& pml);

  /// Compute a new value of uy_sgy, scalar, non-uniform case.
  void Compute_uy_sgy_normalize_scalar_nonuniform(TRealMatrix      & uy_sgy,
                                                  const TRealMatrix& FFT_p,
                                                  const TRealMatrix& dyudyn_sgy,
                                                  const TRealMatrix& pml);

  //----------------------- uz calculation -----------------------------------//
  /// compute a new value of uz_sgz, default case.
  void Compute_uz_sgz_normalize(TRealMatrix      & uz_sgz,
                                const TRealMatrix& FFT_p,
                                const TRealMatrix& dt_rho0,
                                const TRealMatrix& pml);

  /// Compute a new value of uz_sgz, scalar, uniform case.
  void Compute_uz_sgz_normalize_scalar_uniform(TRealMatrix      & uz_sgz,
                                               const TRealMatrix& FFT_p,
                                               const TRealMatrix& pml);

  /// Compute a new value of uz_sgz, scalar, non-uniform case.
  void Compute_uz_sgz_normalize_scalar_nonuniform(TRealMatrix&       uz_sgz,
                                                  const TRealMatrix& FFT_p,
                                                  const TRealMatrix& dzudzn_sgz,
                                                  const TRealMatrix& pml);

  //------------------------ transducers -------------------------------------//

  /// Add transducer data  source to X component.
  void AddTransducerSource(TRealMatrix       & ux_sgx,
                           const TIndexMatrix& u_source_index,
                           TIndexMatrix      & delay_mask,
                           const TRealMatrix & transducer_signal);

  /// Add in velocity source terms.
  void Add_u_source(TRealMatrix       & uxyz_sgxyz,
                    const TRealMatrix & u_source_input,
                    const TIndexMatrix& u_source_index,
                    const size_t        t_index);

  /// Add in pressure source term
  void Add_p_source(TRealMatrix       & rhox,
                    TRealMatrix       & rhoy,
                    TRealMatrix       & rhoz,
                    const TRealMatrix & p_source_input,
                    const TIndexMatrix& p_source_index,
                    const size_t        t_index);

  //------------------velocity spectral operations ---------------------------//
  /// Compute u = dt ./ rho0_sg .* ifft (FFT).
  void Compute_dt_rho_sg_mul_ifft_div_2(TRealMatrix        & uxyz_sgxyz,
                                        const TRealMatrix  & dt_rho0_sg,
                                        TCUFFTComplexMatrix& FFT);

  /// Compute u = dt ./ rho0_sgx .* ifft (FFT), if rho0_sgx is scalar, uniform grid.
  void Compute_dt_rho_sg_mul_ifft_div_2(TRealMatrix        & uxyz_sgxyz,
                                        const float          dt_rho0_sg,
                                        TCUFFTComplexMatrix& FFT);


  /// Compute dt ./ rho0_sgx .* ifft (FFT), if rho0_sgx is scalar, non uniform grid, x component.
  void Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_x(TRealMatrix        & ux_sgx,
                                                            const float          dt_rho0_sgx,
                                                            const TRealMatrix  & dxudxn_sgx,
                                                            TCUFFTComplexMatrix& FFT);

  /// Compute dt ./ rho0_sgy .* ifft (FFT), if rho0_sgy is scalar, non uniform grid, y component.
  void Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_y(TRealMatrix        & uy_sgy,
                                                            const float          dt_rho0_sgy,
                                                            const TRealMatrix  & dyudyn_sgy,
                                                            TCUFFTComplexMatrix& FFT);

  /// Compute dt ./ rho0_sgz .* ifft (FFT), if rho0_sgz is scalar, non uniform grid, z component.
  void Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_z(TRealMatrix        & uz_sgz,
                                                            const float          dt_rho_0_sgz,
                                                            const TRealMatrix  & dzudzn_sgz,
                                                            TCUFFTComplexMatrix& FFT);


   //------------------------- pressure kernels-------------------------------//
   /// Compute part of the new velocity - gradient of p.
   void Compute_ddx_kappa_fft_p(TRealMatrix         & X_Matrix,
                                TCUFFTComplexMatrix & FFT_X,
                                TCUFFTComplexMatrix & FFT_Y,
                                TCUFFTComplexMatrix & FFT_Z,
                                const TRealMatrix   & kappa,
                                const TComplexMatrix& ddx,
                                const TComplexMatrix& ddy,
                                const TComplexMatrix& ddz);

  ///  Compute new values for duxdx, duydy, duzdz on uniform grid.
  void Compute_duxyz_uniform(TCUFFTComplexMatrix & FFT_X,
                             TCUFFTComplexMatrix & FFT_Y,
                             TCUFFTComplexMatrix & FFT_Z,
                             const TRealMatrix   & kappa,
                             const TComplexMatrix& ddx_k_shift_neg,
                             const TComplexMatrix& ddy_k_shift_neg,
                             const TComplexMatrix& ddz_k_shift_neg);

    ///  Shift new values for duxdx, duydy, duzdz on non-uniform grid.
    void Compute_duxyz_non_uniform(TRealMatrix      & duxdx,
                                   TRealMatrix      & duydy,
                                   TRealMatrix      & duzdz,
                                   const TRealMatrix& dxudxn,
                                   const TRealMatrix& dyudyn,
                                   const TRealMatrix& dzudzn);



    /// Add initial pressure to p0 (as p0 source).
    void Calculate_p0_source_add_initial_pressure(TRealMatrix      & p,
                                                  TRealMatrix      & rhox,
                                                  TRealMatrix      & rhoy,
                                                  TRealMatrix      & rhoz,
                                                  const TRealMatrix& p0,
                                                  const float      * c2,
                                                  const size_t       c2_shift);

    //---------------------- density kernels ---------------------------------//
    /// Calculate new values of rhox, rhoy and rhoz for non-linear case, homogenous case.
    void Compute_rhoxyz_nonlinear_homogeneous(TRealMatrix      & rhox,
                                              TRealMatrix      & rhoy,
                                              TRealMatrix      & rhoz,
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
    void Compute_rhoxyz_linear_homogeneous(TRealMatrix      & rhox,
                                           TRealMatrix      & rhoy,
                                           TRealMatrix      & rhoz,
                                           const TRealMatrix& pml_x,
                                           const TRealMatrix& pml_y,
                                           const TRealMatrix& pml_z,
                                           const TRealMatrix& duxdx,
                                           const TRealMatrix& duydy,
                                           const TRealMatrix& duzdz);

    /// Calculate new values of rhox, rhoy and rhoz for linear case, heterogeneous case.
    void Compute_rhoxyz_linear_heterogeneous(TRealMatrix      & rhox,
                                             TRealMatrix      & rhoy,
                                             TRealMatrix      & rhoz,
                                             const TRealMatrix& pml_x,
                                             const TRealMatrix& pml_y,
                                             const TRealMatrix& pml_z,
                                             const TRealMatrix& duxdx,
                                             const TRealMatrix& duydy,
                                             const TRealMatrix& duzdz,
                                             const TRealMatrix& rho0);

    //----------------------- new value of pressure --------------------------//
  /// Calculate three temporary sums in the new pressure formula, non-linear absorbing case.
  void Calculate_SumRho_BonA_SumDu(TRealMatrix      & rho_sum,
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
                                   const size_t       rho0_shift);

  /// Compute absorbing term with abosrb_nabla1 and absorb_nabla2.
  void Compute_Absorb_nabla1_2(TCUFFTComplexMatrix& FFT_1,
                               TCUFFTComplexMatrix& FFT_2,
                               const TRealMatrix  & absorb_nabla1,
                               const TRealMatrix  & absorb_nabla2);

  /// Sum sub-terms to calculate new pressure, non-linear case.
  void Sum_Subterms_nonlinear(TRealMatrix      & p,
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
                              const size_t       tau_eta_shift);

    /// Sum sub-terms to calculate new pressure, linear case.
    void Sum_Subterms_linear(TRealMatrix      & p,
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
                             const size_t       tau_eta_shift);

    /// Sum sub-terms for new p, linear lossless case.
    void Sum_new_p_nonlinear_lossless(TRealMatrix      & p,
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
                                      const size_t       rho0_shift);

    /// Calculate two temporary sums in the new pressure formula, linear absorbing case.
    void Calculate_SumRho_SumRhoDu(TRealMatrix      & Sum_rhoxyz,
                                   TRealMatrix      & Sum_rho0_du,
                                   const TRealMatrix& rhox,
                                   const TRealMatrix& rhoy,
                                   const TRealMatrix& rhoz,
                                   const TRealMatrix& duxdx,
                                   const TRealMatrix& duydy,
                                   const TRealMatrix& duzdz,
                                   const float        rho0_scalar,
                                   const float      * rho0_matrix,
                                   const size_t       rho0_shift);

    /// Sum sub-terms for new p, linear lossless case.
    void Sum_new_p_linear_lossless(TRealMatrix      & p,
                                   const TRealMatrix& rhox,
                                   const TRealMatrix& rhoy,
                                   const TRealMatrix& rhoz,
                                   const float        c2_scalar,
                                   const float      * c2_matrix,
                                   const size_t       c2_shift);

    //------------------- unstaggered velocity ------------------------------//

    /// Transpose a real 3D matrix in the X-Y direction
    void TrasposeReal3DMatrixXY(float       * OutputMatrixData,
                                const float * InputMatrixData,
                                const dim3  & DimSizes);

    /// Transpose a real 3D matrix in the X-Y direction
    void TrasposeReal3DMatrixXZ(float       * OutputMatrixData,
                                const float * InputMatrixData,
                                const dim3  & DimSizes);

    /// Compute the velocity shift in Fourier space over the X axis
    void ComputeVelocityShiftInX(TCUFFTComplexMatrix & FFT_shift_temp,
                                 const TComplexMatrix& x_shift_neg_r);
    /// Compute the velocity shift in Fourier space over the Y axis
    void ComputeVelocityShiftInY(TCUFFTComplexMatrix & FFT_shift_temp,
                                 const TComplexMatrix& y_shift_neg_r);
    /// Compute the velocity shift in Fourier space over the Z axis
    void ComputeVelocityShiftInZ(TCUFFTComplexMatrix & FFT_shift_temp,
                                 const TComplexMatrix& z_shift_neg_r);


  private:
    /// Default constructor for a singleton class
    TCUDAImplementations() {};

    /**
     * Get block size for 1D kernels
     * @return 1D block size
     */
    int GetSolverBlockSize1D() const
    {
      return TParameters::GetInstance()->CUDAParameters.GetSolverBlockSize1D();
    };

    /**
     * Get grid size for 1D kernels
     * @return 1D grid size
     */
    int GetSolverGridSize1D() const
    {
      return TParameters::GetInstance()->CUDAParameters.GetSolverGridSize1D();
    };

    /**
     * Get block size for 3D kernels
     * @return 3D size
     */
    dim3 GetSolverBlockSize3D() const
    {
      return TParameters::GetInstance()->CUDAParameters.GetSolverBlockSize3D();
    };

    /**
     * Get grid size for 3D kernels
     * @return 3D grid size
     */
    dim3 GetSolverGridSize3D() const
    {
      return TParameters::GetInstance()->CUDAParameters.GetSolverGridSize3D();
    };

    /**
     * Get grid size for complex 3D kernels
     * @return 3D grid size
     */
    dim3 GetSolverComplexGridSize3D() const
    {
      return TParameters::GetInstance()->CUDAParameters.GetSolverComplexGridSize3D();
    };


    /**
     * Get block size for the transposition kernels
     * @return 3D grid size
     */
    dim3 GetSolverTransposeBlockSize() const
    {
      return TParameters::GetInstance()->CUDAParameters.GetSolverTransposeBlockSize();
    };

    /**
     * Get grid size for complex 3D kernels
     * @return 3D grid size
     */
    dim3 GetSolverTransposeGirdSize() const
    {
      return TParameters::GetInstance()->CUDAParameters.GetSolverTransposeGirdSize();
    };

    /// Singleton instance flag
    static bool InstanceFlag;
    /// Singleton instance
    static TCUDAImplementations *Instance;

};

#endif /* CUDA_IMPLEMENTATIONS_H */
