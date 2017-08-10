/**
 * @file        KSpaceFirstOrder3DSolver.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the main class of the project responsible for
 *              the entire 3D fluid simulation.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        12 July      2012, 10:27 (created)\n
 *              10 August    2017, 15:30 (revised)
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

#ifndef KSpaceFirstOrder3DSolverH
#define	KSpaceFirstOrder3DSolverH


#include <Parameters/Parameters.h>
#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>
#include <MatrixClasses/IndexMatrix.h>

#include <Containers/MatrixContainer.h>
#include <Containers//OutputStreamContainer.h>

#include <Utils/TimeMeasure.h>

#include <KSpaceSolver/SolverCUDAKernels.cuh>


/**
 * @class   KSpaceFirstOrder3DSolver
 * @brief   Class responsible for running the k-space first order 3D method.
 * @details Class responsible for running the k-space first order 3D method. This class maintain
 *          the whole k-wave (implements the time loop).
 *
 */
class KSpaceFirstOrder3DSolver
{
  public:
    /// Constructor.
    KSpaceFirstOrder3DSolver();
    /// Copy constructor not allowed.
    KSpaceFirstOrder3DSolver(const KSpaceFirstOrder3DSolver&) = delete;
    /// Destructor.
    virtual ~KSpaceFirstOrder3DSolver();

    /// operator= not allowed.
    KSpaceFirstOrder3DSolver& operator=(const KSpaceFirstOrder3DSolver&) = delete;

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
     * @brief This method computes k-space First Order 3D simulation.
     *
     * This method computes k-space First Order 3D simulation. It launches calculation on a given
     * dataset going through FFT initialization, pre-processing, main loop and post-processing phases.
     */
    virtual void compute();

    /**
     * @brief  Get memory usage in MB on the host side.
     * @return Memory consumed on the host side in MB.
     */
    size_t getHostMemoryUsage();
    /**
     * @brief  Get memory usage in MB on the device side.
     * @return Memory consumed on the device side in MB.
     */
    size_t getDeviceMemoryUsage();

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
     *
     * Initialize all indices, pre-compute constants such as c^2, rho0Sgx * dt  and create kappa,
     * absorbEta, absorbTau, absorbNabla1, absorbNabla2  matrices.  \n
     *
     * @note Calculation is done on the host side.
     */
    void preProcessing();
    /// Compute the main time loop of the kspaceFirstOrder3D.
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

    /// Compute new values of acoustic velocity.
    void computeVelocity();
    /// Compute new values of acoustic velocity gradients.
    void computeVelocityGradient();

    /// Compute new values of acoustic density for non-linear case.
    void computeDensityNonliner();
    /// Compute new values of acoustic density for linear case.
    void computeDensityLinear();

    /// Compute acoustic pressure for nonlinear case.
    void computePressureNonlinear();
    /// Compute acoustic pressure for linear case.
    void computePressureLinear();

    /// Add in velocity source
    void addVelocitySource();
    /// Add in pressure source.
    void addPressureSource();
    /// Calculate initial pressure source.
    void addInitialPressureSource();

    /// Generate kappa matrix for non-absorbing media.
    void generateKappa();
    /// Generate kappa matrix, absorbNabla1, absorbNabla2 for absorbing media.
    void generateKappaAndNablas();
    /// Generate absorbTau, absorbEta for heterogenous media.
    void generateTauAndEta();
    /// Calculate dt ./ rho0 for non-uniform grids.
    void generateInitialDenisty();
    /// Calculate square of velocity
    void computeC2();

    /**
     * @brief Calculate three temporary sums in the new pressure formula before taking the FFT,
     *        non-linear absorbing case.
     * @param [out] densitySum          - rhoX + rhoY + rhoZ
     * @param [out] nonlinearTerm       - BOnA + densitySum ^2 / 2 * rho0
     * @param [out] velocityGradientSum - rho0* (duxdx + duydy + duzdz)
     */
    void computePressureTermsNonlinear(RealMatrix& densitySum,
                                       RealMatrix& nonlinearTerm,
                                       RealMatrix& velocityGradientSum);
    /**
     * @brief Calculate two temporary sums in the new pressure formula before taking the FFT,
     *        linear absorbing case.
     * @param [out] densitySum          - rhox_sgx + rhoy_sgy + rhoz_sgz
     * @param [out] velocityGradientSum - rho0* (duxdx + duydy + duzdz);
     */
    void computePressureTermsLinear(RealMatrix& densitySum,
                                    RealMatrix& velocityGradientSum);


    /**
     * @brief Sum sub-terms to calculate new pressure, after FFTs, non-linear case.
     * @param [in] absorbTauTerm - tau component
     * @param [in] absorbEtaTerm - eta component  of the pressure term
     * @param [in] nonlinearTerm - rho0 * (duxdx + duydy + duzdz)
     */
    void sumPressureTermsNonlinear(const RealMatrix& absorbTauTerm,
                                   const RealMatrix& absorbEtaTerm,
                                   const RealMatrix& nonlinearTerm);


    /**
     * @brief Sum sub-terms to calculate new pressure, after FFTs, linear case.
     * @param [in] absorbTauTerm - tau component
     * @param [in] absorbEtaTerm - eta component  of the pressure term
     * @param [in] densitySum    - Sum of three components of density (rhoXSgx + rhoYSgy + rhoZSgx)
     */
    void sumPressureTermsLinear(const RealMatrix& absorbTauTerm,
                                const RealMatrix& absorbEtaTerm,
                                const RealMatrix& densitySum);

    /// Sum sub-terms for new p, linear lossless case.
    void sumPressureTermsNonlinearLossless();

    /// Sum sub-terms for new p, linear lossless case.
    void sumPressureTermsLinearLossless();

    /// compute shifted velocity for --u_non_staggered flag.
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

    //----------------------------------------------- Get matrices ---------------------------------------------------//

    /**
     * @brief  Get the kappa matrix from the container.
     * @return kappa matrix
     */
    RealMatrix& getKappa()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kKappa);
    };

    /**
     * @brief  Get the c^2 matrix from the container.
     * @return c^2 matrix.
     */
    RealMatrix& getC2()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kC2);
    };

    /**
     * @brief  Get pressure matrix
     * @return Pressure matrix
     */
    RealMatrix& getP()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kP);
    };

    /**
     * @brief  Get velocity matrix on staggered grid in x direction.
     * @return Velocity matrix on staggered grid.
     */
    RealMatrix& getUxSgx()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kUxSgx);
    };
        /**
     * @brief  Get velocity matrix on staggered grid in y direction.
     * @return Velocity matrix on staggered grid.
     */
    RealMatrix& getUySgy()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kUySgy);
    };
    /**
     * @brief  Get velocity matrix on staggered grid in z direction.
     * @return Velocity matrix on staggered grid.
     */
    RealMatrix& getUzSgz()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kUzSgz);
    };

    /**
     * @brief  Get velocity shifted on normal grid in x direction
     * @return Unstaggeted velocity matrix.
     */
    RealMatrix& getUxShifted()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kUxShifted);
    };
    /**
     * @brief  Get velocity shifted on normal grid in y direction
     * @return Unstaggered velocity matrix.
     */
    RealMatrix& getUyShifted()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kUyShifted);
    };
    /**
     * @brief  Get velocity shifted on normal grid in z direction
     * @return Unstaggered velocity matrix.
     */
    RealMatrix& getUzShifted()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kUzShifted);
    };

    /**
     * @brief  Get velocity gradient on in x direction.
     * @return Velocity gradient matrix.
     */
    RealMatrix& getDuxdx()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDuxdx);
    };
    /**
     * @brief  Get velocity gradient on in y direction.
     * @return Velocity gradient matrix.
     */
    RealMatrix& getDuydy()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDuydy);
    };
    /**
     * @brief  Get velocity gradient on in z direction.
     * @return Velocity gradient matrix.
     */
    RealMatrix& getDuzdz()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDuzdz);
    };

    /**
     * @brief  Get dt * rho0Sgx matrix (time step size * ambient velocity on staggered grid in x direction).
     * @return Density matrix
     */
    RealMatrix& getDtRho0Sgx()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDtRho0Sgx);
    };
    /**
     * @brief  Get dt * rho0Sgy matrix (time step size * ambient velocity on staggered grid in y direction).
     * @return Density matrix
     */
    RealMatrix& getDtRho0Sgy()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDtRho0Sgy);
    };
    /**
     * @brief  Get dt * rho0Sgz matrix (time step size * ambient velocity on staggered grid in z direction).
     * @return Density matrix
     */
    RealMatrix& getDtRho0Sgz()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDtRho0Sgz);
    };

    /**
     * @brief  Get density matrix in x direction.
     * @return Density matrix.
     */
    RealMatrix& getRhoX()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kRhoX);
    };
    /**
     * @brief  Get density matrix in y direction.
     * @return Density matrix.
     */
    RealMatrix& getRhoY()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kRhoY);
    };
    /**
     * @brief  Get density matrix in z direction.
     * @return Density matrix.
     */
    RealMatrix& getRhoZ()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kRhoZ);
    };
    /**
     * @brief  Get ambient density matrix.
     * @return Density matrix.
     */
    RealMatrix& getRho0()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kRho0);
    };

    /**
     * @brief  Get positive Fourier shift in x.
     * @return Shift matrix.
     */
    ComplexMatrix& getDdxKShiftPos()
    {
      return mMatrixContainer.getMatrix<ComplexMatrix>(MatrixContainer::MatrixIdx::kDdxKShiftPosR);
    };
    /**
     * @brief  Get positive Fourier shift in y.
     * @return Shift matrix.
     */
    ComplexMatrix& getDdyKShiftPos()
    {
      return mMatrixContainer.getMatrix<ComplexMatrix>(MatrixContainer::MatrixIdx::kDdyKShiftPos);
    };
    /**
     * @brief  Get positive Fourier shift in z.
     * @return Shift matrix.
     */
    ComplexMatrix& getDdzKShiftPos()
    {
      return mMatrixContainer.getMatrix<ComplexMatrix>(MatrixContainer::MatrixIdx::kDdzKShiftPos);
    };
    /**
     * @brief  Get negative Fourier shift in x.
     * @return Shift matrix.
     */
    ComplexMatrix& getDdxKShiftNeg()
    {
      return mMatrixContainer.getMatrix<ComplexMatrix>(MatrixContainer::MatrixIdx::kDdxKShiftNegR);
    };
    /**
     * @brief  Get negative Fourier shift in y.
     * @return Shift matrix.
     */
    ComplexMatrix& getDdyKShiftNeg()
    {
      return mMatrixContainer.getMatrix<ComplexMatrix>(MatrixContainer::MatrixIdx::kDdyKShiftNeg);
    };
    /**
     * @brief  Get negative Fourier shift in z.
     * @return shift matrix.
     */
    ComplexMatrix& getDdzKShiftNeg()
    {
      return mMatrixContainer.getMatrix<ComplexMatrix>(MatrixContainer::MatrixIdx::kDdzKShiftNeg);
    };

    /**
     * @brief  Get negative shift for non-staggered velocity in x.
     * @return Shift matrix.
     */
    ComplexMatrix& getXShiftNegR()
    {
      return mMatrixContainer.getMatrix<ComplexMatrix>(MatrixContainer::MatrixIdx::kXShiftNegR);
    };
    /**
     * @brief  Get negative shift for non-staggered velocity in y.
     * @return Shift matrix.
     */
    ComplexMatrix& getYShiftNegR()
    {
      return mMatrixContainer.getMatrix<ComplexMatrix>(MatrixContainer::MatrixIdx::kYShiftNegR);
    };
    /**
     * @brief  Get negative shift for non-staggered velocity in z.
     * @return Shift matrix.
     */
    ComplexMatrix& getZShiftNegR()
    {
      return mMatrixContainer.getMatrix<ComplexMatrix>(MatrixContainer::MatrixIdx::kZShiftNegR);
    };


    /**
     * @brief  Get PML on staggered grid x.
     * @return PML matrix.
     */
    RealMatrix& getPmlXSgx()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kPmlXSgx);
    };
    /**
     * @brief  Get PML on staggered grid y.
     * @return PML matrix.
     */
    RealMatrix& getPmlYSgy()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kPmlYSgy);
    };
    /**
     * @brief  Get PML on staggered grid z.
     * @return PML matrix.
     */
    RealMatrix& getPmlZSgz()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kPmlZSgz);
    };
    /**
     * @brief  Get PML in x.
     * @return PML matrix.
     */
    RealMatrix& getPmlX()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kPmlX);
    };
    /**
     * @brief  Get PML in y.
     * @return PML matrix.
     */
    RealMatrix& getPmlY()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kPmlY);
    };
    /**
     * @brief  Get PML in z.
     * @return PML matrix.
     */
    RealMatrix& getPmlZ()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kPmlZ);
    };

    /**
     * @brief  Non uniform grid acoustic velocity in x.
     * @return Velocity matrix.
     */
    RealMatrix& getDxudxn()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDxudxn);
    };
    /**
     * @brief  Non uniform grid acoustic velocity in y.
     * @return Velocity matrix.
     */
    RealMatrix& getDyudyn()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDyudyn);
    };
    /**
     * @brief  Non uniform grid acoustic velocity in z.
     * @return Velocity matrix.
     */
    RealMatrix& getDzudzn()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDzudzn);
    };
    /**
     * @brief  Non uniform grid acoustic velocity on staggered grid x.
     * @return Velocity matrix.
     */
    RealMatrix& getDxudxnSgx()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDxudxnSgx);
    };
    /**
     * @brief  Non uniform grid acoustic velocity on staggered grid x.
     * @return Velocity matrix.
     */
    RealMatrix& getDyudynSgy()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDyudynSgy);
    };
    /**
     * @brief  Non uniform grid acoustic velocity on staggered grid x.
     * @return Velocity matrix.
     */
    RealMatrix& getDzudznSgz()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDzudznSgz);
    };


    /**
     * @brief  Get B on A (nonlinear coefficient).
     * @return Nonlinear coefficient.
     */
    RealMatrix& geBOnA()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kBOnA);
    };
    /**
     * @brief  Get absorbing coefficient Tau.
     * @return Absorbing coefficient.
     */
    RealMatrix& getAbsorbTau()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kAbsorbTau);
    };
    /**
     * @brief  Get absorbing coefficient Eta.
     * @return Absorbing coefficient.
     */
    RealMatrix& getAbsorbEta()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kAbsorbEta);
    };

    /**
     * @brief  Get absorbing coefficient Nabla1.
     * @return Absorbing coefficient.
     */
    RealMatrix& getAbsorbNabla1()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kAbsorbNabla1);
    };
    /**
     * @brief  Get absorbing coefficient Nabla2.
     * @return Absorbing coefficient.
     */
    RealMatrix& getAbsorbNabla2()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kAbsorbNabla2);
    };


    //----------------------------------------------- Index matrices -------------------------------------------------//
    /**
     * @brief  Get linear sensor mask (spatial geometry of the sensor).
     * @return Sensor mask data.
     */
    IndexMatrix& getSensorMaskIndex()
    {
      return mMatrixContainer.getMatrix<IndexMatrix>(MatrixContainer::MatrixIdx::kSensorMaskIndex);
    };
    /**
     * @brief  Get cuboid corners sensor mask. (Spatial geometry of mulitple sensors).
     * @return Sensor mask data.
     */
    IndexMatrix& getSensorMaskCorners()
    {
      return mMatrixContainer.getMatrix<IndexMatrix>(MatrixContainer::MatrixIdx::kSensorMaskCorners);
    };
    /**
     * @brief  Get velocity source geometry data.
     * @return Source geometry indices
     */
    IndexMatrix& getVelocitySourceIndex()
    {
      return mMatrixContainer.getMatrix<IndexMatrix>(MatrixContainer::MatrixIdx::kVelocitySourceIndex);
    };
    /**
     * @brief  Get pressure source geometry data.
     * @return Source geometry indices
     */
    IndexMatrix& getPressureSourceIndex()
    {
      return mMatrixContainer.getMatrix<IndexMatrix>(MatrixContainer::MatrixIdx::kPressureSourceIndex);
    };
    /**
     * @brief  Get delay mask for many types sources
     * @return delay mask.
     */
    IndexMatrix& getDelayMask()
    {
      return mMatrixContainer.getMatrix<IndexMatrix>(MatrixContainer::MatrixIdx::kDelayMask);
    }


    //-------------------------------------------------- Sources  ----------------------------------------------------//

    /**
     * @brief  Get transducer source input data (signal).
     * @return Transducer source input data.
     */
    RealMatrix& getTransducerSourceInput()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kTransducerSourceInput);
    };
    /**
     * @brief  Get pressure source input data (signal).
     * @return Pressure source input data.
     */
    RealMatrix& getPressureSourceInput()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kPressureSourceInput);
    };
    /**
     * @brief  Get initial pressure source input data (whole matrix).
     * @return Initial pressure source input data.
     */
    RealMatrix& getInitialPressureSourceInput()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kInitialPressureSourceInput);
    };


    /**
     * @brief  Get Velocity source input data in x direction.
     * @return Velocity source input data.
     */
    RealMatrix& GetVelocityXSourceInput()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kVelocityXSourceInput);
    };
    /**
     * @brief  Get Velocity source input data in y direction.
     * @return Velocity source input data.
     */
    RealMatrix& GetVelocityYSourceInput()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kVelocityYSourceInput);
    };
    /**
     * @brief  Get Velocity source input data in z direction.
     * @return Velocity source input data.
     */
    RealMatrix& getVelocityZSourceInput()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kVelocityZSourceInput);
    };


    //--------------------------------------------- Temporary matrices -----------------------------------------------//

    /**
     * @brief  Get first real 3D temporary matrix.
     * @return Temporary real 3D matrix.
     */
    RealMatrix& getTemp1Real3D()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kTemp1Real3D);
    };
    /**
     * @brief  Get second real 3D temporary matrix.
     * @return Temporary real 3D matrix.
     */
    RealMatrix& getTemp2Real3D()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kTemp2Real3D);
    };
    /**
     * @brief  Get third real 3D temporary matrix.
     * @return Temporary real 3D matrix.
     */
    RealMatrix& getTemp3Real3D()
    {
      return mMatrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kTemp3Real3D);
    };


    /**
     * @brief  Get Temporary matrix for 1D fft in x.
     * @return Temporary complex 3D matrix.
     */
    CufftComplexMatrix& getTempCufftX()
    {
      return mMatrixContainer.getMatrix<CufftComplexMatrix>(MatrixContainer::MatrixIdx::kTempCufftX);
    };
    /**
     * @brief  Get Temporary matrix for 1D fft in y.
     * @return Temporary complex 3D matrix.
     */
    CufftComplexMatrix& getTempCufftY()
    {
      return mMatrixContainer.getMatrix<CufftComplexMatrix>(MatrixContainer::MatrixIdx::kTempCufftY);
    };
    /**
     * @brief  Get Temporary matrix for 1D fft in z.
     * @return Temporary complex 3D matrix.
     */
    CufftComplexMatrix& getTempCufftZ()
    {
      return mMatrixContainer.getMatrix<CufftComplexMatrix>(MatrixContainer::MatrixIdx::kTempCufftZ);
    };

    /**
     * @brief  Get Temporary matrix for cufft shift.
     * @return Temporary complex 3D matrix.
     */
    CufftComplexMatrix& getTempCufftShift()
    {
      return mMatrixContainer.getMatrix<CufftComplexMatrix>(MatrixContainer::MatrixIdx::kTempCufftShift);
    };

private:

    /// Matrix container with all the matrix classes.
    MatrixContainer mMatrixContainer;
    /// Output stream container.
    OutputStreamContainer mOutputStreamContainer;

    /// Global parameters of the simulation.
    Parameters& mParameters;

    /// Percentage of the simulation done.
    size_t mActPercent;

    /// This variable is true when calculating first time step after restore from checkpoint (to allow asynchronous IO).
    bool mIsTimestepRightAfterRestore;

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

};// end of KSpaceFirstOrder3DSolver
//----------------------------------------------------------------------------------------------------------------------

#endif /* TKSpaceFirstOrder3DSolverH */
