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
 * @version   kspaceFirstOrder 3.6
 *
 * @date      11 March     2013, 13:10 (created) \n
 *            06 March     2019, 13:11 (revised)
 *
 * @copyright Copyright (C) 2019 Jiri Jaros and Bradley Treeby.
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

#include <Containers/MatrixContainer.h>
#include <Utils/DimensionSizes.h>

#include <Parameters/Parameters.h>
#include <Parameters/CudaParameters.h>

/**
 * @namespace   SolverCudaKernels
 * @brief       List of cuda kernels used k-space first order 2D/3D solver.
 * @details     List of cuda kernels used k-space first order 2D/3D solver.
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
   * @tparam simulationDimension - Dimensionality of the simulation.
   * @param [in] container       - Container with all matrices.
   *
   * <b> Matlab code: </b> \code
   *  ux_sgx = bsxfun(@times, pml_x_sgx, bsxfun(@times, pml_x_sgx, ux_sgx) - dt .* rho0_sgx_inv .* real(ifftX)
   *  uy_sgy = bsxfun(@times, pml_y_sgy, bsxfun(@times, pml_y_sgy, uy_sgy) - dt .* rho0_sgy_inv .* real(ifftY)
   *  uz_sgz = bsxfun(@times, pml_z_sgz, bsxfun(@times, pml_z_sgz, uz_sgz) - dt .* rho0_sgz_inv .* real(ifftZ)
   * \endcode
   */
  template<Parameters::SimulationDimension simulationDimension>
  void computeVelocityHeterogeneous(const MatrixContainer& container);


  /**
   * @brief Compute acoustic velocity for homogeneous medium and a uniform grid.
   * @tparam simulationDimension - Dimensionality of the simulation.
   * @param [in] container       - Container with all matrices.
   *
   * <b> Matlab code: </b> \code
   *  ux_sgx = bsxfun(@times, pml_x_sgx, bsxfun(@times, pml_x_sgx, ux_sgx) - dt .* rho0_sgx_inv .* real(ifftX)
   *  uy_sgy = bsxfun(@times, pml_y_sgy, bsxfun(@times, pml_y_sgy, uy_sgy) - dt .* rho0_sgy_inv .* real(ifftY)
   *  uz_sgz = bsxfun(@times, pml_z_sgz, bsxfun(@times, pml_z_sgz, uz_sgz) - dt .* rho0_sgz_inv .* real(ifftZ)
   *\endcode
   */
  template<Parameters::SimulationDimension simulationDimension>
  void computeVelocityHomogeneousUniform(const MatrixContainer& container);

  /**
   * @brief  Compute acoustic velocity for homogenous medium and nonuniform grid.
   * @tparam simulationDimension - Dimensionality of the simulation.
   * @param [in] container       - Container with all matrices.
   *
   * <b> Matlab code: </b> \code
   *  ux_sgx = bsxfun(@times, pml_x_sgx, bsxfun(@times, pml_x_sgx, ux_sgx)  ...
   *                  - dt .* rho0_sgx_inv .* dxudxnSgx.* real(ifftX))
   *  uy_sgy = bsxfun(@times, pml_y_sgy, bsxfun(@times, pml_y_sgy, uy_sgy) ...
   *                  - dt .* rho0_sgy_inv .* dyudynSgy.* real(ifftY)
   *  uz_sgz = bsxfun(@times, pml_z_sgz, bsxfun(@times, pml_z_sgz, uz_sgz)
   *                  - dt .* rho0_sgz_inv .* dzudznSgz.* real(ifftZ)
   *\endcode
   */
  template<Parameters::SimulationDimension simulationDimension>
  void computeVelocityHomogeneousNonuniform(const MatrixContainer& container);

  //--------------------------------------------------- Sources ------------------------------------------------------//

  /**
   * @brief Add transducer data source to velocity x component.
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
   * @tparam simulationDimension - Dimensionality of the simulation.
   * @param [in] container       - Container with all matrices.
   */
  template<Parameters::SimulationDimension simulationDimension>
  void addPressureSource(const MatrixContainer& container);


  /**
   * @brief Insert source signal into scaling matrix.
   *
   * @param [out] scaledSource - Temporary matrix to insert the source into before scaling.
   * @param [in]  sourceInput  - Source input signal.
   * @param [in]  sourceIndex  - Source geometry.
   * @param [in]  manyFlag     - Number of time series in the source input.
   * @param [in]  timeIndex    - Actual time step.
   */
  void insertSourceIntoScalingMatrix(RealMatrix&        scaledSource,
                                     const RealMatrix&  sourceInput,
                                     const IndexMatrix& sourceIndex,
                                     const size_t       manyFlag,
                                     const size_t       timeIndex);

  /**
   * @brief Calculate source gradient.
   * @param [in, out] sourceSpectrum - Source spectrum.
   * @param [in]      sourceKappa    - Source kappa.
   */
  void computeSourceGradient(CufftComplexMatrix& sourceSpectrum,
                             const RealMatrix&   sourceKappa);

  /**
   * @brief Add scaled velocity source to acoustic density.
   * @param [in, out] velocity     - Velocity matrix to update.
   * @param [in]      scaledSource - Scaled source.
   */
  void addVelocityScaledSource(RealMatrix&        velocity,
                               const RealMatrix&  scalingSource);

  /**
   * @brief Add scaled pressure source to acoustic density, 3D case.
   * @param [in, out] rhoX         - Acoustic density.
   * @param [in, out] rhoY         - Acoustic density.
   * @param [in, out] rhoZ         - Acoustic density.
   * @param [in]      scaledSource - Scaled source.
   */
  void addPressureScaledSource(RealMatrix&        rhoX,
                               RealMatrix&        rhoY,
                               RealMatrix&        rhoZ,
                               const RealMatrix&  scalingSource);
  /**
   * @brief Add scaled pressure source to acoustic density, 2D case.
   * @param [in, out] rhoX         - Acoustic density.
   * @param [in, out] rhoY         - Acoustic density.
   * @param [in]      scaledSource - Scaled source.
   */
  void addPressureScaledSource(RealMatrix&        rhoX,
                               RealMatrix&        rhoY,
                               const RealMatrix&  scalingSource);

  /**
   * @brief Add initial pressure source to the pressure matrix and update density matrices.
   * @tparam simulationDimension - Dimensionality of the simulation.
   *
   * <b>Matlab code:</b> \code
   *  % add the initial pressure to rho as a mass source (3D code)
   *  p = source.p0;
   *  rhox = source.p0 ./ (3 .* c.^2);
   *  rhoy = source.p0 ./ (3 .* c.^2);
   *  rhoz = source.p0 ./ (3 .* c.^2);
   */
  template<Parameters::SimulationDimension simulationDimension>
  void addInitialPressureSource(const MatrixContainer& container);

  /**
   * @brief Compute velocity for the initial pressure problem, heterogeneous medium, uniform grid.
   *
   * @tparam simulationDimension - Dimensionality of the simulation.
   *
   * <b> Matlab code: </b> \code
   *  ux_sgx = dt ./ rho0_sgx .* ifft(ux_sgx).
   *  uy_sgy = dt ./ rho0_sgy .* ifft(uy_sgy).
   *  uz_sgz = dt ./ rho0_sgz .* ifft(uz_sgz).
     * \endcode
   */
  template<Parameters::SimulationDimension simulationDimension>
  void computeInitialVelocityHeterogeneous(const MatrixContainer& container);

  /**
   * @brief Compute acoustic velocity for initial pressure problem, homogeneous medium, uniform grid.
   * @tparam simulationDimension - Dimensionality of the simulation.
   *
   * <b> Matlab code: </b> \code
   *  ux_sgx = dt ./ rho0_sgx .* ifft(ux_sgx).
   *  uy_sgy = dt ./ rho0_sgy .* ifft(uy_sgy).
   *  uz_sgz = dt ./ rho0_sgz .* ifft(uz_sgz).
   * \endcode
   */
  template<Parameters::SimulationDimension simulationDimension>
  void computeInitialVelocityHomogeneousUniform(const MatrixContainer& container);

  /**
   * @brief Compute acoustic velocity for initial pressure problem, homogenous medium, non-uniform grid.
   * @tparam simulationDimension - Dimensionality of the simulation.
   *
   * <b> Matlab code: </b> \code
   *  ux_sgx = dt ./ rho0_sgx .* dxudxn_sgx .* ifft(ux_sgx)
   *  uy_sgy = dt ./ rho0_sgy .* dyudxn_sgy .* ifft(uy_sgy)
   *  uz_sgz = dt ./ rho0_sgz .* dzudzn_sgz .* ifft(uz_sgz)
   * \endcode
   */
  template<Parameters::SimulationDimension simulationDimension>
  void computeInitialVelocityHomogeneousNonuniform(const MatrixContainer& container);


  //----------------------------------------------- pressure kernels -------------------------------------------------//
  /**
   * @brief Compute spectral part of pressure gradient in between FFTs.
   * @tparam simulationDimension - Dimensionality of the simulation.
   * @param [in] container       - Container with all matrices.
   *
   * <b> Matlab code: </b> \code
   *  bsxfun(@times, ddx_k_shift_pos, kappa .* p_k).
   *  bsxfun(@times, ddx_k_shift_pos, kappa .* p_k).
   *  bsxfun(@times, ddx_k_shift_pos, kappa .* p_k).
   * \endcode
   *
   */
  template<Parameters::SimulationDimension simulationDimension>
  void computePressureGradient(const MatrixContainer& container);

  /**
   * @brief Compute spatial part of the velocity gradient in between FFTs on uniform grid.
   * @tparam simulationDimension - Dimensionality of the simulation.
   * @param [in] container       - Container with all matrices.
   *
   * <b> Matlab code: </b> \code
   *  bsxfun(@times, ddx_k_shift_neg, kappa .* fftn(ux_sgx));
   *  bsxfun(@times, ddy_k_shift_neg, kappa .* fftn(uy_sgy));
   *  bsxfun(@times, ddz_k_shift_neg, kappa .* fftn(uz_sgz));
   * \endcode
   */
  template<Parameters::SimulationDimension simulationDimension>
  void computeVelocityGradient(const MatrixContainer& container);

  /**
   * @brief Shift gradient of acoustic velocity on non-uniform grid.
   * @tparam simulationDimension - Dimensionality of the simulation.
   * @param [in] container       - Container with all matrices.
   */
  template<Parameters::SimulationDimension simulationDimension>
  void computeVelocityGradientShiftNonuniform(const MatrixContainer& container);

  //---------------------------------------------- density kernels ---------------------------------------------------//

  /**
   * @brief Calculate acoustic density for non-linear case, homogeneous case is default.
   * @tparam simulationDimension - Dimensionality of the simulation.
   * @param [in] container       - Container with all matrices.
   *
   * <b>Matlab code:</b> \code
   *  rho0_plus_rho = 2 .* (rhox + rhoy + rhoz) + rho0;
   *  rhox = bsxfun(@times, pml_x, bsxfun(@times, pml_x, rhox) - dt .* rho0_plus_rho .* duxdx);
   *  rhoy = bsxfun(@times, pml_y, bsxfun(@times, pml_y, rhoy) - dt .* rho0_plus_rho .* duydy);
   *  rhoz = bsxfun(@times, pml_z, bsxfun(@times, pml_z, rhoz) - dt .* rho0_plus_rho .* duzdz);
   * \endcode
   */
  template<Parameters::SimulationDimension simulationDimension>
  void computeDensityNonlinear(const MatrixContainer& container);

  /**
   * @brief Calculate acoustic density for linear case, homogeneous case is default.
   * @tparam simulationDimension - Dimensionality of the simulation.
   * @param [in] container       - Container with all matrices.
   *
   * <b>Matlab code:</b> \code
   *  rhox = bsxfun(@times, pml_x, bsxfun(@times, pml_x, rhox) - dt .* rho0 .* duxdx);
   *  rhoy = bsxfun(@times, pml_y, bsxfun(@times, pml_y, rhoy) - dt .* rho0 .* duydy);
   *  rhoz = bsxfun(@times, pml_z, bsxfun(@times, pml_z, rhoz) - dt .* rho0 .* duzdz);
   * \endcode
   */
  template<Parameters::SimulationDimension simulationDimension>
  void computeDensityLinear(const MatrixContainer& container);


  //-------------------------------------------- new value of pressure -----------------------------------------------//
  ///
  /**
   * @brief Calculate three temporary sums in the new pressure formula, non-linear absorbing case.
   *
   * @tparam simulationDimension      - Dimensionality of the simulation.
   * @param [out] densitySum          - rhox_sgx + rhoy_sgy + rhoz_sgz
   * @param [out] nonlinearTerm       - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz)
   * @param [out] velocityGradientSum - rho0* (duxdx + duydy + duzdz)
   * @param [in]  container           - Container with all matrices.
   */
  template<Parameters::SimulationDimension simulationDimension>
  void computePressureTermsNonlinear(RealMatrix&            densitySum,
                                     RealMatrix&            nonlinearTerm,
                                     RealMatrix&            velocityGradientSum,
                                     const MatrixContainer& container);

  /**
   * @brief Calculate two temporary sums in the new pressure formula, linear absorbing case.
   * @tparam simulationDimension      - Dimensionality of the simulation.
   * @param [out] densitySum          - rhox_sgx + rhoy_sgy + rhoz_sgz
   * @param [out] velocityGradientSum - rho0* (duxdx + duydy + duzdz);
   * @param [in]  container           - Container with all matrices.
   */
  template<Parameters::SimulationDimension simulationDimension>
  void computePressureTermsLinear(RealMatrix&            densitySum,
                                  RealMatrix&            velocityGradientSum,
                                  const MatrixContainer& container);

  /**
   * @brief Compute absorbing term with abosrbNabla1 and absorbNabla2.
   *
   * @param [in,out] fftPart1     - fftPart1 = absorbNabla1 .* fftPart1
   * @param [in,out] fftPart2     - fftPart1 = absorbNabla1 .* fftPart2
   * @param [in]     absorbNabla1 - Absorption coefficient 1.
   * @param [in]     absorbNabla2 - Absorption coefficient 2.
   *
   * <b>Matlab code:</b> \code
   *  fftPart1 = absorbNabla1 .* fftPart1 \n
   *  fftPart2 = absorbNabla2 .* fftPart2 \n
   * \endcode
   */
  void computeAbsorbtionTerm(CufftComplexMatrix& fftPart1,
                             CufftComplexMatrix& fftPart2,
                             const RealMatrix&   absorbNabla1,
                             const RealMatrix&   absorbNabla2);

  /**
   * @brief Sum sub-terms to calculate new pressure in non-linear case.
   *        The output is stored into the pressure matrix.
   *
   * @param [in] nonlinearTerm - Nonlinear term
   * @param [in] absorbTauTerm - Absorb tau term from the pressure eq.
   * @param [in] absorbEtaTerm - Absorb eta term from the pressure eq.
   * @param [in] container         - Container with all matrices.
   */
  void sumPressureTermsNonlinear(const RealMatrix&      nonlinearTerm,
                                 const RealMatrix&      absorbTauTerm,
                                 const RealMatrix&      absorbEtaTerm,
                                 const MatrixContainer& container);

  /**
   * @brief Sum sub-terms to calculate new pressure, linear case.
   *        The output is stored into the pressure matrix.
   *
   * @param [in]  absorbTauTerm - Absorb tau term from the pressure eq.
   * @param [in]  absorbEtaTerm - Absorb tau term from the pressure eq.
   * @param [in]  densitySum    - Sum of acoustic density.
   * @param [in] container      - Container with all matrices.
   */
  void sumPressureTermsLinear(const RealMatrix&      absorbTauTerm,
                              const RealMatrix&      absorbEtaTerm,
                              const RealMatrix&      densitySum,
                              const MatrixContainer& container);

  /**
   * @brief Sum sub-terms for new pressure, linear lossless case.
   *
   * @tparam simulationDimension - Dimensionality of the simulation.
   * @param [in] container       - Container with all matrices.
   *
   */
  template<Parameters::SimulationDimension simulationDimension>
  void sumPressureNonlinearLossless(const MatrixContainer& container);

  /**
   * @brief Sum sub-terms for new pressure, linear lossless case.
   *
   * @tparam simulationDimension - Dimensionality of the simulation.
   * @param [in] container       - Container with all matrices.
   */
  template<Parameters::SimulationDimension simulationDimension>
  void sumPressureLinearLossless(const MatrixContainer& container);

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
