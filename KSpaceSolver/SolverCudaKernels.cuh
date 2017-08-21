/**
 * @file      SolverCudaKernels.cuh
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file for all cuda kernels of the GPU implementation used in the 3D solver.
 *
 * @version   kspaceFirstOrder3D 3.5
 *
 * @date      11 March     2013, 13:10 (created) \n
 *            16 August    2017, 13:49 (revised)
 *
 * @copyright Copyright (C) 2017 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the C++ extension of the [k-Wave Toolbox](http://www.k-wave.org).
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify it under the terms
 * of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 */

#ifndef SOLVER_CUDA_KERNELS_H
#define	SOLVER_CUDA_KERNELS_H

#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>
#include <MatrixClasses/IndexMatrix.h>
#include <MatrixClasses/CufftComplexMatrix.h>

#include <Utils/DimensionSizes.h>

#include <Parameters/Parameters.h>
#include <Parameters/CudaParameters.h>

/**
 * @namespace   SolverCudaKernels
 * @brief       List of cuda kernels used k-space first order 3D solver.
 * @details     List of cuda kernels used k-space first order 3D solver.
 *
 */
namespace SolverCudaKernels
{
  /**
   * @enum TransposePadding
   * @brief How is the data during matrix transposition padded.
   */
  enum class TransposePadding
  {
    /// none
    kNone,
    /// input matrix is padded
    kInput,
    /// output matrix is padded
    kOutput,
    /// both matrices are padded
    kInputOutput
  };

  /**
   * @brief Get the CUDA architecture the code was compiled with.
   *
   * It is done by calling a kernel that  reads a variable set by nvcc compiler
   *
   * @return The CUDA code version the code was compiled for.
   */
  int getCudaCodeVersion();

  //--------------------------------------------- velocity operations ------------------------------------------------//
  /**
   * @brief Compute acoustic velocity for heterogeneous medium and a uniform grid.
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
  void computeVelocity(RealMatrix&       uxSgx,
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
                       const RealMatrix& pmlZ);


  /**
   * @brief Compute acoustic velocity for homogeneous medium and a uniform grid.
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
  void computeVelocityHomogeneousUniform(RealMatrix&       uxSgx,
                                         RealMatrix&       uySgy,
                                         RealMatrix&       uzSgz,
                                         const RealMatrix& ifftX,
                                         const RealMatrix& ifftY,
                                         const RealMatrix& ifftZ,
                                         const RealMatrix& pmlX,
                                         const RealMatrix& pmlY,
                                         const RealMatrix& pmlZ);

  /**
   * @brief Compute acoustic velocity for homogenous medium and non-uniform grid.
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
  void computeVelocityHomogeneousNonuniform(RealMatrix&       uxSgx,
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
                                            const RealMatrix& pmlZ);

  //--------------------------------------------------- Sources ------------------------------------------------------//

  /**
   * @brief Add transducer data  source to velocity x component.
   *
   * @param [in, out] uxSgx                 - Here we add the signal.
   * @param [in]      velocitySourceIndex   - Where to add the signal (source geometry).
   * @param [in]      transducerSourceInput - Transducer signal.
   * @param [in]      delayMask             - Delay mask to push the signal in the domain (incremented per invocation).
   * @param [in]      timeIndex             - Actual time step.
   */
  void addTransducerSource(RealMatrix&        uxSgx,
                           const IndexMatrix& velocitySourceIndex,
                           const RealMatrix&  transducerSourceInput,
                           const IndexMatrix& delayMask,
                           const size_t       timeIndex);

  /**
   * @brief Add in velocity source terms.
   *
   * @param [in, out] velocity       - Velocity matrix to update.
   * @param [in] velocitySourceInput - Source input to add.
   * @param [in] velocitySourceIndex - Source geometry index matrix.
   * @param [in] timeIndex           - Actual time step.
   */
  void addVelocitySource(RealMatrix&        velocity,
                         const RealMatrix&  velocitySourceInput,
                         const IndexMatrix& velocitySourceIndex,
                         const size_t       timeIndex);

  /**
   * @brief Add in pressure source term.
   *
   * @param [out] rhoX                - Acoustic density.
   * @param [out] rhoY                - Acoustic density.
   * @param [out] rhoZ                - Acoustic density.
   * @param [in]  pressureSourceInput - Source input to add.
   * @param [in]  pressureSourceIndex - Index matrix with source.
   * @param [in]  timeIndex           - Actual times step.
   */
  void addPressureSource(RealMatrix&        rhoX,
                         RealMatrix&        rhoY,
                         RealMatrix&        rhoZ,
                         const RealMatrix&  pressureSourceInput,
                         const IndexMatrix& pressureSourceIndex,
                         const size_t       timeIndex);

   /**
    * @brief Add initial pressure source to the pressure matrix and update density matrices.
    *
    * @param [out] p                     - New pressure field.
    * @param [out] rhoX                  - Density in x direction.
    * @param [out] rhoY                  - Density in y direction.
    * @param [out] rhoZ                  - Density in z direction.
    * @param [in]  initialPerssureSource - Initial pressure source.
    * @param [in]  isC2Scalar            - is sound speed homogenous?
    * @param [in]  c2                    - Sound speed for heterogeneous case.
    */
  void addInitialPressureSource(RealMatrix&       p,
                                RealMatrix&       rhoX,
                                RealMatrix&       rhoY,
                                RealMatrix&       rhoZ,
                                const RealMatrix& initialPerssureSource,
                                const bool        isC2Scalar,
                                const float*      c2);

  /**
   * @brief Compute velocity for the initial pressure problem, heterogeneous medium, uniform grid.
   *
   * @param [in, out] uxSgx     - Velocity matrix in x direction.
   * @param [in, out] uySgy     - Velocity matrix in y direction.
   * @param [in, out] uzSgz     - Velocity matrix in y direction.
   * @param [in]      dtRho0Sgx - Density matrix in x direction.
   * @param [in]      dtRho0Sgy - Density matrix in y direction.
   * @param [in]      dtRho0Sgz - Density matrix in z direction.
   */
  void computeInitialVelocity(RealMatrix&       uxSgx,
                              RealMatrix&       uySgy,
                              RealMatrix&       uzSgz,
                              const RealMatrix& dtRho0Sgx,
                              const RealMatrix& dtRho0Sgy,
                              const RealMatrix& dtRho0Sgz);

  /**
   * @brief Compute acoustic velocity for initial pressure problem, homogeneous medium, uniform grid.
   *
   * @param [in, out] uxSgx - Velocity matrix in x direction.
   * @param [in, out] uySgy - Velocity matrix in y direction.
   * @param [in, out] uzSgz - Velocity matrix in y direction.
   */
  void computeInitialVelocity(RealMatrix& uxSgx,
                              RealMatrix& uySgy,
                              RealMatrix& uzSgz);

  /**
   * @brief Compute acoustic velocity for initial pressure problem, homogenous medium, non-uniform grid.
   *
   * @param [in, out] uxSgx - Velocity matrix in x direction.
   * @param [in, out] uySgy - Velocity matrix in y direction.
   * @param [in, out] uzSgz - Velocity matrix in y direction
   * @param [in] dxudxnSgx  - Non uniform grid shift in x direction.
   * @param [in] dyudynSgy  - Non uniform grid shift in y direction.
   * @param [in] dzudznSgz  - Non uniform grid shift in z direction.
   */
  void computeInitialVelocityHomogeneousNonuniform(RealMatrix&       uxSgx,
                                                   RealMatrix&       uySgy,
                                                   RealMatrix&       uzSgz,
                                                   const RealMatrix& dxudxnSgx,
                                                   const RealMatrix& dyudynSgy,
                                                   const RealMatrix& dzudznSgz);


  //----------------------------------------------- pressure kernels -------------------------------------------------//
  /**
   * @brief Compute spectral part of pressure gradient in between FFTs.
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
   void computePressureGradient(CufftComplexMatrix& ifftX,
                                CufftComplexMatrix& ifftY,
                                CufftComplexMatrix& ifftZ,
                                const RealMatrix&    kappa,
                                const ComplexMatrix& ddxKShiftPos,
                                const ComplexMatrix& ddyKShiftPos,
                                const ComplexMatrix& ddzKShiftPos);

   /**
    * @brief Compute spatial part of the velocity gradient in between FFTs on uniform grid.
    *
    * @param [in, out] fftX    - input is the FFT of velocity, output is the spectral part in x.
    * @param [in, out] fftY    - input is the FFT of velocity, output is the spectral part in y.
    * @param [in, out] fftZ    - input is the FFT of velocity, output is the spectral part in z.
    * @param [in] kappa        - Kappa matrix
    * @param [in] ddxKShiftNeg - Negative spectral shift in x direction.
    * @param [in] ddyKShiftNeg - Negative spectral shift in x direction.
    * @param [in] ddzKShiftNeg - Negative spectral shift in x direction.
    */
  void computeVelocityGradient(CufftComplexMatrix&  fftX,
                               CufftComplexMatrix&  fftY,
                               CufftComplexMatrix&  fftZ,
                               const RealMatrix&    kappa,
                               const ComplexMatrix& ddxKShiftNeg,
                               const ComplexMatrix& ddyKShiftNeg,
                               const ComplexMatrix& ddzKShiftNeg);

  /**
   * @brief Shift gradient of acoustic velocity on non-uniform grid.
   *
   * @param [in,out] duxdx  - Gradient of particle velocity in x direction.
   * @param [in,out] duydy  - Gradient of particle velocity in y direction.
   * @param [in,out] duzdz  - Gradient of particle velocity in z direction.
   * @param [in]     dxudxn - Non uniform grid shift in x direction.
   * @param [in]     dyudyn - Non uniform grid shift in y direction.
   * @param [in]     dzudzn - Non uniform grid shift in z direction.
   */
  void computeVelocityGradientShiftNonuniform(RealMatrix&       duxdx,
                                              RealMatrix&       duydy,
                                              RealMatrix&       duzdz,
                                              const RealMatrix& dxudxn,
                                              const RealMatrix& dyudyn,
                                              const RealMatrix& dzudzn);

  //---------------------------------------------- density kernels ---------------------------------------------------//

  /**
   * @brief Calculate acoustic density for non-linear case, homogeneous case is default.
   *
   * @param [in, out] rhoX         - Acoustic density in x direction.
   * @param [in, out] rhoY         - Acoustic density in y direction.
   * @param [in, out] rhoZ         - Acoustic density in z direction.
   * @param [in]      pmlX         - PML layer in x direction.
   * @param [in]      pmlY         - PML layer in x direction.
   * @param [in]      pmlZ         - PML layer in x direction.
   * @param [in]      duxdx        - Gradient of velocity x direction.
   * @param [in]      duydy        - Gradient of velocity y direction.
   * @param [in]      duzdz        - Gradient of velocity z direction.
   * @param [in]      isRho0Scalar - Is the density homogeneous?
   * @param [in]      rho0Data     - If density is heterogeneous, here is the matrix with data.
   */
  void computeDensityNonlinear(RealMatrix&       rhoX,
                               RealMatrix&       rhoY,
                               RealMatrix&       rhoZ,
                               const RealMatrix& pmlX,
                               const RealMatrix& pmlY,
                               const RealMatrix& pmlZ,
                               const RealMatrix& duxdx,
                               const RealMatrix& duydy,
                               const RealMatrix& duzdz,
                               const bool        isRho0Scalar = true,
                               const float*      rho0Data     = nullptr);

  /**
   * @brief Calculate acoustic density for linear case, homogeneous case is default.
   *
   * @param [in, out] rhoX         - Acoustic density in x direction.
   * @param [in, out] rhoY         - Acoustic density in y direction.
   * @param [in, out] rhoZ         - Acoustic density in z direction.
   * @param [in]      pmlX         - PML layer in x direction.
   * @param [in]      pmlY         - PML layer in x direction.
   * @param [in]      pmlZ         - PML layer in x direction.
   * @param [in]      duxdx        - Gradient of velocity x direction.
   * @param [in]      duydy        - Gradient of velocity y direction.
   * @param [in]      duzdz        - Gradient of velocity z direction.
   * @param [in]      isRho0Scalar - Is the density homogeneous?
   * @param [in]      rho0Data     - If density is heterogeneous, here is the matrix with data.
   */
  void computeDensityLinear(RealMatrix&       rhoX,
                            RealMatrix&       rhoY,
                            RealMatrix&       rhoZ,
                            const RealMatrix& pmlX,
                            const RealMatrix& pmlY,
                            const RealMatrix& pmlZ,
                            const RealMatrix& duxdx,
                            const RealMatrix& duydy,
                            const RealMatrix& duzdz,
                            const bool        isRho0Scalar = true,
                            const float*      rho0Data     = nullptr);


  //-------------------------------------------- new value of pressure -----------------------------------------------//
  ///
  /**
   * @brief Calculate three temporary sums in the new pressure formula, non-linear absorbing case.
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
   * @param [in]  isBonAScalar        - Is nonlinear coefficient B/A homogeneous?
   * @param [in]  bOnAData            - Heterogeneous value for BonA
   * @param [in]  isRho0Scalar        - Is density homogeneous?
   * @param [in]  rho0Data            - Heterogeneous value for rho0
   */
  void computePressureTermsNonlinear(RealMatrix&       densitySum,
                                     RealMatrix&       nonlinearTerm,
                                     RealMatrix&       velocityGradientSum,
                                     const RealMatrix& rhoX,
                                     const RealMatrix& rhoY,
                                     const RealMatrix& rhoZ,
                                     const RealMatrix& duxdx,
                                     const RealMatrix& duydy,
                                     const RealMatrix& duzdz,
                                     const bool        is_BonA_scalar,
                                     const float*      bOnAData,
                                     const bool        is_rho0_scalar,
                                     const float*      rho0Data);

  /**
   * @brief Calculate two temporary sums in the new pressure formula, linear absorbing case.
   *
   * @param [out] densitySum          - rhox_sgx + rhoy_sgy + rhoz_sgz
   * @param [out] velocityGradientSum - rho0* (duxdx + duydy + duzdz);
   * @param [in]  rhoX                - Acoustic density in x direction.
   * @param [in]  rhoY                - Acoustic density in y direction.
   * @param [in]  rhoZ                - Acoustic density in z direction.
   * @param [in]  duxdx               - Velocity gradient in x direction.
   * @param [in]  duydy               - Velocity gradient in x direction.
   * @param [in]  duzdz               - Velocity gradient in x direction.
   * @param [in]  isRho0Scalar        - Is density  homogeneous?
   * @param [in]  rho0Data            - Acoustic density data in heterogeneous case.
   */
  void computePressureTermsLinear(RealMatrix&       densitySum,
                                  RealMatrix&       velocityGradientSum,
                                  const RealMatrix& rhoX,
                                  const RealMatrix& rhoY,
                                  const RealMatrix& rhoZ,
                                  const RealMatrix& duxdx,
                                  const RealMatrix& duydy,
                                  const RealMatrix& duzdz,
                                  const bool         isRho0Scalar,
                                  const float*       rho0Data);



  /**
   * @brief Compute absorbing term with abosrbNabla1 and absorbNabla2.
   *
   * @param [in,out] fftPart1     - fftPart1 = absorbNabla1 .* fftPart1
   * @param [in,out] fftPart2     - fftPart1 = absorbNabla1 .* fftPart2
   * @param [in]     absorbNabla1 - Absorption coefficient 1.
   * @param [in]     absorbNabla2 - Absorption coefficient 2.
   */
  void computeAbsorbtionTerm(CufftComplexMatrix& fftPart1,
                             CufftComplexMatrix& fftPart2,
                             const RealMatrix&   absorbNabla1,
                             const RealMatrix&   absorbNabla2);

  /**
   * @brief Sum sub-terms to calculate new pressure in non-linear case.
   *
   * @param [in,out] p                   - New value of pressure
   * @param [in]     nonlinearTerm       - Nonlinear term
   * @param [in]     absorbTauTerm       - Absorb tau term from the pressure eq.
   * @param [in]     absorbEtaTerm       - Absorb eta term from the pressure eq.
   * @param [in]     isC2Scalar          - is sound speed homogeneous?
   * @param [in]     c2Data              - sound speed data in heterogeneous case.
   * @param [in]     areTauAndEtaScalars - is absorption homogeneous?
   * @param [in]     absorbTauData       - Absorb tau data in heterogenous case.
   * @param [in]     absorbEtaData       - Absorb eta data in heterogenous case.
   */
  void sumPressureTermsNonlinear(RealMatrix&       p,
                                 const RealMatrix& nonlinearTerm,
                                 const RealMatrix& absorbTauTerm,
                                 const RealMatrix& absorbEtaTerm,
                                 const bool        isC2Scalar,
                                 const float*      c2Data,
                                 const bool        areTauAndEtaScalars,
                                 const float*      absorbTauData,
                                 const float*      absorbEtaData);

  /**
   * @brief Sum sub-terms to calculate new pressure, linear case.
   * @param [out] p                   - New value of pressure.
   * @param [in]  absorbTauTerm       - Absorb tau term from the pressure eq.
   * @param [in]  absorbEtaTerm       - Absorb tau term from the pressure eq.
   * @param [in]  densitySum          - Sum of acoustic density.
   * @param [in]  isC2Scalar          - is sound speed homogeneous?
   * @param [in]  c2Data              - sound speed data in heterogeneous case.
   * @param [in]  areTauAndEtaScalars - is absorption homogeneous?
   * @param [in]  absorbTauData       - Absorb tau data in heterogenous case.
   * @param [in]  absorbEtaData       - Absorb eta data in heterogenous case.
   */
  void sumPressureTermsLinear(RealMatrix&       p,
                              const RealMatrix& absorbTauTerm,
                              const RealMatrix& absorbEtaTerm,
                              const RealMatrix& densitySum,
                              const bool        isC2Scalar,
                              const float*      c2Data,
                              const bool        areTauAndEtaScalars,
                              const float*      absorbTauData,
                              const float*      absorbEtaData);

  /**
   * @brief Sum sub-terms for new p, linear lossless case.
   * @param [out] p            - New value of pressure
   * @param [in]  rhoX         - Acoustic density in x direction.
   * @param [in]  rhoY         - Acoustic density in y direction.
   * @param [in]  rhoZ         - Acoustic density in z direction.
   * @param [in]  isC2Scalar   - Is sound speed homogenous?
   * @param [in]  c2Data       - Sound speed data in heterogeneous case.
   * @param [in]  isBOnAScalar - Is nonlinearity homogeneous?
   * @param [in]  bOnAData     - B on A data in heterogeneous case.
   * @param [in]  isRho0Scalar - Is density homogeneous?
   * @param [in]  rho0Data     - Acoustic density data in heterogeneous case.
   */
  void sumPressureNonlinearLossless(RealMatrix&       p,
                                    const RealMatrix& rhoX,
                                    const RealMatrix& rhoY,
                                    const RealMatrix& rhoZ,
                                    const bool        isC2Scalar,
                                    const float*      c2Data,
                                    const bool        isBOnAScalar,
                                    const float*      bOnAData,
                                    const bool        isRho0Scalar,
                                    const float*      rho0Data);

  /**
   * @brief Sum sub-terms for new p, linear lossless case.
   *
   * @param [out] p          - New value of pressure
   * @param [in]  rhoX       - Acoustic density in x direction.
   * @param [in]  rhoY       - Acoustic density in x direction.
   * @param [in]  rhoZ       - Acoustic density in x direction.
   * @param [in]  isC2Scalar - Is sound speed homogenous?
   * @param [in]  c2Data     - Sound speed data in heterogeneous case.
   */
  void sumPressureLinearLossless(RealMatrix&       p,
                                 const RealMatrix& rhoX,
                                 const RealMatrix& rhoY,
                                 const RealMatrix& rhoZ,
                                 const bool        isC2Scalar,
                                 const float*      c2Data);

  //--------------------------------------------- unstaggered velocity -----------------------------------------------//

  /**
   * @brief Transpose a real 3D matrix in the X-Y direction. It is done out-of-place.
   *
   * @tparam      padding      - Which matrices are padded.
   *
   * @param [out] outputMatrix - Output matrix data.
   * @param [in]  inputMatrix  - Input  matrix data.
   * @param [in]  dimSizes     - Dimension sizes of the original matrix.
   */
  template<TransposePadding padding>
  void trasposeReal3DMatrixXY(float*       outputMatrix,
                              const float* inputMatrix,
                              const dim3&  dimSizes);
/**
 * @brief  Transpose a real 3D matrix in the X-Z direction. It is done out-of-place.
 *
 * @tparam      padding      - Which matrices are padded.
 *
 * @param [out] outputMatrix - Output matrix.
 * @param [in]  inputMatrix  - Input  matrix.
 * @param [in]  dimSizes     - Dimension sizes of the original matrix.
 */
  template<TransposePadding padding>
  void trasposeReal3DMatrixXZ(float*       outputMatrix,
                              const float* inputMatrix,
                              const dim3&  dimSizes);


  /**
   * @brief Compute the velocity shift in Fourier space over x direction.
   *
   * @param [in,out] cufftShiftTemp - Matrix to shift, shifted matrix.
   * @param [in]     xShiftNegR     - negative Fourier shift.
   */
  void computeVelocityShiftInX(CufftComplexMatrix&  cufftShiftTemp,
                               const ComplexMatrix& xShiftNegR);
  /**
   * @brief Compute the velocity shift in Fourier space over z direction.
   *
   * @param [in,out] cufftShiftTemp - Matrix to shift, shifted matrix.
   * @param [in]     yShiftNegR     - negative Fourier shift.
   */
  void computeVelocityShiftInY(CufftComplexMatrix&  cufftShiftTemp,
                               const ComplexMatrix& yShiftNegR);
  /**
   * @brief Compute the velocity shift in Fourier space over z direction.
   *
   * @param [in,out] cufftShiftTemp - Matrix to shift, shifted matrix.
   * @param [in]     zShiftNegR     - negative Fourier shift.
   */
  void computeVelocityShiftInZ(CufftComplexMatrix&  cufftShiftTemp,
                               const ComplexMatrix& zShiftNegR);
}// SolverCudaKernels
//----------------------------------------------------------------------------------------------------------------------
#endif /* SOLVER_CUDA_KERNELS_H */
