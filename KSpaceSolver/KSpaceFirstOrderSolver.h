/**
 * @file      KSpaceFirstOrderSolver.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file containing k-space first order solver in 3D fluid medium. This is the main class
 *            controlling the simulation.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      12 July      2012, 10:27 (created)\n
 *            07 March     2019, 09:05 (revised)
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

#ifndef KSPACE_FIRST_ORDER_SOLVER_H
#define	KSPACE_FIRST_ORDER_SOLVER_H


#include <Parameters/Parameters.h>
#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>
#include <MatrixClasses/IndexMatrix.h>

#include <Containers/MatrixContainer.h>
#include <Containers//OutputStreamContainer.h>

#include <Utils/TimeMeasure.h>

#include <KSpaceSolver/SolverCudaKernels.cuh>


/**
 * @class   KSpaceFirstOrderSolver
 * @brief   Class responsible for running the k-space first order method in 2D and 3D media.
 * @details Class responsible for running the k-space first order method in 2D and 3D media. This class maintain
 *          the whole k-wave (implements the time loop).
 *
 */
class KSpaceFirstOrderSolver
{
  public:
    /// Constructor.
    KSpaceFirstOrderSolver();
    /// Copy constructor not allowed.
    KSpaceFirstOrderSolver(const KSpaceFirstOrderSolver&) = delete;
    /// Destructor.
    virtual ~KSpaceFirstOrderSolver();

    /// operator= not allowed.
    KSpaceFirstOrderSolver& operator=(const KSpaceFirstOrderSolver&) = delete;

    /// Memory allocation.
    virtual void allocateMemory();
    /// Memory deallocation.
    virtual void freeMemory();

    /**
     * @brief Load simulation data.
     *
     * If checkpointing is enabled, this may include reading data from checkpoint and output file.
     */
    virtual void loadInputData();

    /**
     * @brief This method computes k-space First Order 2D/3D simulation.
     *
     * This method computes k-space First Order 2D/3D simulation. It launches calculation on a given
     * dataset going through FFT initialization, pre-processing, main loop and post-processing phases.
     */
    virtual void compute();

    /**
     * @brief  Get memory usage in MB on the host side.
     * @return Memory consumed on the host side in MB.
     */
    size_t getHostMemoryUsage() const;
    /**
     * @brief  Get memory usage in MB on the device side.
     * @return Memory consumed on the device side in MB.
     */
    size_t getDeviceMemoryUsage() const;

    /**
     * @brief  Get code name - release code version.
     * @return Release code version.
     */
    const std::string getCodeName() const;

    /// Print the code name and license.
    void printFullCodeNameAndLicense() const;

    /**
     * @brief  Get total simulation time.
     * @return Total simulation time in seconds.
     */
    double getTotalTime()          const;
    /**
     * @brief  Get pre-processing time.
     * @return Pre-processing time in seconds.
     */
    double getPreProcessingTime()  const;
    /**
     * @brief  Get data load time.
     * @return Time to load data in seconds.
     */
    double getDataLoadTime()       const;
    /**
     * @brief  Get simulation time (time loop).
     * @return Time to execute the simulation in seconds.
     */
    double getSimulationTime()     const;
    /**
     * @brief  Get post-processing time.
     * @return Time to postprocess simulation data in seconds.
     */
    double getPostProcessingTime() const;


    /**
     * @brief  Get total simulation time accumulated over all legs.
     * @return Total execution time in seconds accumulated over all legs.
     */
    double getCumulatedTotalTime()          const;
    /**
     * @brief  Get pre-processing time accumulated over all legs.
     * @return Time to load data in seconds accumulated over all legs.
     */
    double getCumulatedPreProcessingTime()  const;
    /**
     * @brief  Get data load time cumulated over all legs.
     * @return Time to load data in seconds accumulated over all legs.
     */
    double getCumulatedDataLoadTime()       const;
    /**
     * @brief  Get simulation time (time loop) accumulated over all legs.
     * @return Time to execute the simulation in seconds accumulated over all legs.
     */
    double getCumulatedSimulationTime()     const;
    /**
     * @brief  Get post-processing time accumulated over all legs.
     * @return Time to post-processing simulation data in seconds accumulated over all legs.
     */
    double getCumulatedPostProcessingTime() const;

  protected:
    /// Initialize cuda FFT plans.
    void initializeCufftPlans();
    /**
     * @brief Compute pre-processing phase.
     * @tparam simulationDimension - Dimensionality of the simulation.
     * Initialize all indices, pre-compute constants such as c^2, rho0Sgx * dt  and create kappa,
     * absorbEta, absorbTau, absorbNabla1, absorbNabla2  matrices.
     *
     * @note Calculation is done on the host side.
     */
    template<Parameters::SimulationDimension simulationDimension>
    void preProcessing();
    /**
     * @brief  Compute the main time loop of the kspaceFirstOrder solver.
     * @tparam simulationDimension - Dimensionality of the simulation.
     */
    template<Parameters::SimulationDimension simulationDimension>
    void computeMainLoop();
    /// Post processing, and closing the output streams.
    void postProcessing();

    /**
     * @brief Store sensor data.
     *
     * This routine exploits asynchronous behavior. It first performs IO from the i-1th step while
     * waiting for ith step to come to the point of sampling.
     */
    void storeSensorData();
    /// Write statistics and header into the output file.
    void writeOutputDataInfo();
    /// Save checkpoint data and flush aggregated outputs into the output file.
    void saveCheckpointData();

    /**
     * @brief  Compute new values of acoustic velocity in all used dimensions (UxSgx, UySgy, UzSgz).
     * @tparam simulationDimension - Dimensionality of the simulation.
     *
     * <b>Matlab code:</b> \code
     *  p_k = fftn(p);
     *  ux_sgx = bsxfun(@times, pml_x_sgx, ...
     *       bsxfun(@times, pml_x_sgx, ux_sgx) ...
     *       - dt .* rho0_sgx_inv .* real(ifftn( bsxfun(@times, ddx_k_shift_pos, kappa .* fftn(p)) )) ...
     *       );
     *  uy_sgy = bsxfun(@times, pml_y_sgy, ...
     *       bsxfun(@times, pml_y_sgy, uy_sgy) ...
     *       - dt .* rho0_sgy_inv .* real(ifftn( bsxfun(@times, ddy_k_shift_pos, kappa .* fftn(p)) )) ...
     *       );
     *  uz_sgz = bsxfun(@times, pml_z_sgz, ...
     *       bsxfun(@times, pml_z_sgz, uz_sgz) ...
     *       - dt .* rho0_sgz_inv .* real(ifftn( bsxfun(@times, ddz_k_shift_pos, kappa .* fftn(p)) )) ...
     *       );
     \endcode
     */
    template<Parameters::SimulationDimension simulationDimension>
    void computeVelocity();

    /**
     * @brief  Compute new values of acoustic velocity gradients.
     * @tparam simulationDimension - Dimensionality of the simulation.
     *
     * <b>Matlab code:</b> \code
     *  duxdx = real(ifftn( bsxfun(@times, ddx_k_shift_neg, kappa .* fftn(ux_sgx)) ));
     *  duydy = real(ifftn( bsxfun(@times, ddy_k_shift_neg, kappa .* fftn(uy_sgy)) ));
     *  duzdz = real(ifftn( bsxfun(@times, ddz_k_shift_neg, kappa .* fftn(uz_sgz)) ));
     \endcode
     */
    template<Parameters::SimulationDimension simulationDimension>
    void computeVelocityGradient();

    /**
     * @brief  Compute new values of acoustic density for nonlinear case.
     * @tparam simulationDimension - Dimensionality of the simulation.
     *
     * <b>Matlab code:</b> \code
     *  rho0_plus_rho = 2 .* (rhox + rhoy + rhoz) + rho0;
     *  rhox = bsxfun(@times, pml_x, bsxfun(@times, pml_x, rhox) - dt .* rho0_plus_rho .* duxdx);
     *  rhoy = bsxfun(@times, pml_y, bsxfun(@times, pml_y, rhoy) - dt .* rho0_plus_rho .* duydy);
     *  rhoz = bsxfun(@times, pml_z, bsxfun(@times, pml_z, rhoz) - dt .* rho0_plus_rho .* duzdz);
     * \endcode
     */
    template<Parameters::SimulationDimension simulationDimension>
    void computeDensityNonliner();

    /**
     * @brief  Compute new values of acoustic density for linear case.
     * @tparam simulationDimension - Dimensionality of the simulation.
     *
     * <b>Matlab code:</b> \code
     *  rhox = bsxfun(@times, pml_x, bsxfun(@times, pml_x, rhox) - dt .* rho0 .* duxdx);
     *  rhoy = bsxfun(@times, pml_y, bsxfun(@times, pml_y, rhoy) - dt .* rho0 .* duydy);
     *  rhoz = bsxfun(@times, pml_z, bsxfun(@times, pml_z, rhoz) - dt .* rho0 .* duzdz);
     * \endcode
     */
    template<Parameters::SimulationDimension simulationDimension>
    void computeDensityLinear();

    /**
     * Compute acoustic pressure for nonlinear case.
     * @tparam simulationDimension - Dimensionality of the simulation.
     *
     * <b>Matlab code:</b> \code
     *  case 'lossless'
     *    % calculate p using a nonlinear adiabatic equation of state
     *    p = c.^2 .* (rhox + rhoy + rhoz + medium.BonA .* (rhox + rhoy + rhoz).^2 ./ (2 .* rho0));
     *
     *  case 'absorbing'
     *    % calculate p using a nonlinear absorbing equation of state
     *    p = c.^2 .* (...
     *        (rhox + rhoy + rhoz) ...
     *        + absorb_tau .* real(ifftn( absorb_nabla1 .* fftn(rho0 .* (duxdx + duydy + duzdz)) ))...
     *        - absorb_eta .* real(ifftn( absorb_nabla2 .* fftn(rhox + rhoy + rhoz) ))...
     *        + medium.BonA .*(rhox + rhoy + rhoz).^2 ./ (2 .* rho0) ...
     *       );
     * \endcode
     */
    template<Parameters::SimulationDimension simulationDimension>
    void computePressureNonlinear();

    /**
     * @brief Compute acoustic pressure for linear case.
     * <b>Matlab code:</b> \code
     *  case 'lossless'
     *
     *    % calculate p using a linear adiabatic equation of state
     *    p = c.^2 .* (rhox + rhoy + rhoz);
     *
     *  case 'absorbing'
     *
     *    % calculate p using a linear absorbing equation of state
     *    p = c.^2 .* ( ...
     *        (rhox + rhoy + rhoz) ...
     *        + absorb_tau .* real(ifftn( absorb_nabla1 .* fftn(rho0 .* (duxdx + duydy + duzdz)) )) ...
     *        - absorb_eta .* real(ifftn( absorb_nabla2 .* fftn(rhox + rhoy + rhoz) )) ...
     *       );
     * \endcode
     */
    template<Parameters::SimulationDimension simulationDimension>
    void computePressureLinear();

    /// Add in velocity source.
    void addVelocitySource();
    /**
     * @brief  Add in pressure source.
     * @tparam simulationDimension - Dimensionality of the simulation.
     */
    template<Parameters::SimulationDimension simulationDimension>
    void addPressureSource();
    /**
     * @brief Scale velocity or pressure source.
     *
     * @param [in] scaledSource - Generated scaled source
     * @param [in] sourceInput  - Source input signal
     * @param [in] sourceIndex  - Source geometry
     * @param [in] manyFlag     - How many time series
     */
    void scaleSource(RealMatrix&        scaledSource,
                     const RealMatrix&  sourceInput,
                     const IndexMatrix& sourceIndex,
                     const size_t       manyFlag);
    /**
     * @brief Calculate initial pressure source.
     * @tparam simulationDimension - Dimensionality of the simulation.
     *
     * <b>Matlab code:</b> \code
     *  % add the initial pressure to rho as a mass source
     *  p = source.p0;
     *  rhox = source.p0 ./ (3 .* c.^2);
     *  rhoy = source.p0 ./ (3 .* c.^2);
     *  rhoz = source.p0 ./ (3 .* c.^2);
     *
     *  % compute u(t = t1 + dt/2) based on the assumption u(dt/2) = -u(-dt/2)
     *  % which forces u(t = t1) = 0
     *  ux_sgx = dt .* rho0_sgx_inv .* real(ifftn( bsxfun(@times, ddx_k_shift_pos, kappa .* fftn(p)) )) / 2;
     *  uy_sgy = dt .* rho0_sgy_inv .* real(ifftn( bsxfun(@times, ddy_k_shift_pos, kappa .* fftn(p)) )) / 2;
     *  uz_sgz = dt .* rho0_sgz_inv .* real(ifftn( bsxfun(@times, ddz_k_shift_pos, kappa .* fftn(p)) )) / 2;
     * \endcode
     */
    template<Parameters::SimulationDimension simulationDimension>
    void addInitialPressureSource();

    /// Generate kappa matrix for  lossless medium.
    void generateKappa();
    /// Generate sourceKappa matrix for additive sources.
    void generateSourceKappa();
    /// Generate kappa matrix, absorbNabla1, absorbNabla2 for absorbing medium.
    void generateKappaAndNablas();
    /// Generate absorbTau, absorbEta for heterogenous medium.
    void generateTauAndEta();
    /**
     * @brief Calculate dt ./ rho0 for nonuniform grids.
     * @tparam simulationDimension - Dimensionality of the simulation.
     */
    template<Parameters::SimulationDimension simulationDimension>
    void generateInitialDenisty();
    /// Calculate square of velocity
    void computeC2();

    /// Compute shifted velocity for --u_non_staggered flag.
    void computeShiftedVelocity();

    /// Print progress statistics.
    void printStatistics();

    /**
     * @brief  Is time to checkpoint (save actual state on disk).
     * @return true if it is time to interrupt the simulation and checkpoint.
     */
    bool isTimeToCheckpoint();

    /**
     * @brief  Was the loop interrupted to checkpoint?
     * @return true if the simulation has been interrupted.
     */
    bool isCheckpointInterruption() const;

    /**
     * @brief Check the output file has the correct format and version.
     * @throw ios::failure - If an error happens.
     */
    void checkOutputFile();

    /**
     * @brief Check the file type and the version of the checkpoint file.
     * @throw ios::failure - If an error happens
     */
    void checkCheckpointFile();

    /// Reads the header of the output file and sets the cumulative elapsed time from the first log.
    void loadElapsedTimeFromOutputFile();

    /**
     * @brief Compute 1D index using 3 spatial coordinates and the size of the matrix.
     * @param [in] z              - z coordinate
     * @param [in] y              - y coordinate
     * @param [in] x              - x coordinate
     * @param [in] dimensionSizes - Size of the matrix.
     * @return
     */
    size_t get1DIndex(const size_t          z,
                      const size_t          y,
                      const size_t          x,
                      const DimensionSizes& dimensionSizes);

    //---------------------------- Shortcut for the most used matrices from the container ----------------------------//
    /**
     * @brief  Get temporary matrix for 1D fft in x
     * @return Temporary complex 2D/3D matrix.
     */
    CufftComplexMatrix& getTempCufftX() const { return mMatrixContainer.getTempCufftX(); };
    /**
     * @brief  Get temporary matrix for 1D fft in y.
     * @return Temporary complex 2D/3D matrix.
     */
    CufftComplexMatrix& getTempCufftY() const { return mMatrixContainer.getTempCufftY(); };

    /**
     * @brief  Get temporary matrix for 1D fft in z.
     * @return Temporary complex 3D matrix.
     */
    CufftComplexMatrix& getTempCufftZ() const { return mMatrixContainer.getTempCufftZ(); };

private:

    /// Matrix container with all the matrix classes.
    MatrixContainer       mMatrixContainer;
    /// Output stream container.
    OutputStreamContainer mOutputStreamContainer;

    /// Global parameters of the simulation.
    Parameters& mParameters;

    /// Percentage of the simulation done.
    size_t mActPercent;

    /// This variable is true when calculating first time step after restore from checkpoint (to allow asynchronous IO).
    bool   mIsTimestepRightAfterRestore;

    /// Total time of the simulation.
    TimeMeasure mTotalTime;
    /// Pre-processing time of the simulation.
    TimeMeasure mPreProcessingTime;
    /// Data load time of the simulation.
    TimeMeasure mDataLoadTime;
    /// Simulation time of the simulation.
    TimeMeasure mSimulationTime;
    /// Post-processing time of the simulation.
    TimeMeasure mPostProcessingTime;
    /// Iteration time of the simulation.
    TimeMeasure mIterationTime;

};// end of KSpaceFirstOrderSolver
//----------------------------------------------------------------------------------------------------------------------

#endif /* KSPACE_FIRST_ORDER_SOLVER_H */
