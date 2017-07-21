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
 *              21 July      2017, 16:50 (revised)
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

#ifndef TKSPACE_FIRST_ORDER_3D_SOLVER_H
#define	TKSPACE_FIRST_ORDER_3D_SOLVER_H


#include <Parameters/Parameters.h>
#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>
#include <MatrixClasses/IndexMatrix.h>

#include <Containers/MatrixContainer.h>
#include <Containers//OutputStreamContainer.h>

#include <Utils/TimeMeasure.h>

#include <KSpaceSolver/SolverCUDAKernels.cuh>


/**
 * @class TKSpaceFirstOrder3DSolver
 * @brief  Class responsible for running the k-space first order 3D method.
 * @details Class responsible for running the k-space first order 3D method. This class maintain
 *          the whole k-wave (implements the time loop).
 *
 */
class TKSpaceFirstOrder3DSolver
{
  public:
    /// Constructor.
    TKSpaceFirstOrder3DSolver();
    /// Copy constructor not allowed.
    TKSpaceFirstOrder3DSolver(const TKSpaceFirstOrder3DSolver&) = delete;
    /// Destructor.
    virtual ~TKSpaceFirstOrder3DSolver();

    /// operator= not allowed.
    TKSpaceFirstOrder3DSolver& operator=(const TKSpaceFirstOrder3DSolver&) = delete;

    /// Memory allocation.
    virtual void AllocateMemory();
    /// Memory deallocation.
    virtual void FreeMemory();

    /// Load simulation data from the input file.
    virtual void LoadInputData();

    /// Compute the k-space simulation.
    virtual void Compute();

    /// Get memory usage in MB on the CPU side.
    size_t GetHostMemoryUsageInMB();
    /// Get memory usage in MB on the GPU side.
    size_t GetDeviceMemoryUsageInMB();

    /// Get code name - release code version.
    const std::string GetCodeName() const;

    /// Print the code name and license.
    void PrintFullNameCodeAndLicense() const;

    /// Get total simulation time.
    double GetTotalTime()          const;
    /// Get pre-processing time.
    double GetPreProcessingTime()  const;
    /// Get data load time.
    double GetDataLoadTime()       const;
    /// Get simulation time (time loop).
    double GetSimulationTime()     const;
    /// Get post-processing time.
    double GetPostProcessingTime() const;

    /// Get total simulation time cumulated over all legs.
    double GetCumulatedTotalTime()          const;
    /// Get pre-processing time cumulated over all legs.
    double GetCumulatedPreProcessingTime()  const;
    /// Get data load time cumulated over all legs.
    double GetCumulatedDataLoadTime()       const;
    /// Get simulation time (time loop) cumulated over all legs.
    double GetCumulatedSimulationTime()     const;
    /// Get post-processing time cumulated over all legs.
    double GetCumulatedPostProcessingTime() const;

protected:
    /// Initialize FFT plans.
    void InitializeFFTPlans();
    /// Compute pre-processing phase.
    void PreProcessingPhase();
    /// Compute the main time loop of the kspaceFirstOrder3D.
    void ComputeMainLoop();
    /// Post processing, and closing the output streams.
    void PostProcessing();

    /// Store sensor data.
    void StoreSensorData();
    /// Write statistics and header into the output file.
    void WriteOutputDataInfo();
    /// Save checkpoint data.
    void SaveCheckpointData();

    /// Compute new values of acoustic velocity.
    void ComputeVelocity();
    /// Compute new values of acoustic velocity gradients.
    void ComputeGradientVelocity();

    /// Compute new values of acoustic density for non-linear case.
    void ComputeDensityNonliner();
    /// Compute new values of acoustic density for linear case.
    void ComputeDensityLinear();

    /// Compute acoustic pressure for nonlinear case
    void ComputePressureNonlinear();
    /// Compute acoustic pressure for linear case
    void ComputePressureLinear();

    /// Add in velocity source
    void AddVelocitySource();
    /// Add in pressure source.
    void AddPressureSource();
    /// Calculate p0_ ource.
    void Calculate_p0_source();

    /// Generate kappa matrix for non-absorbing media.
    void GenerateKappa();
    /// Generate kappa matrix, absorb_nabla1, absorb_nabla2 for absorbing media
    void GenerateKappaAndNablas();
    /// Generate absorb_tau, absorb_eta for heterogenous media.
    void GenerateTauAndEta();
    /// Calculate dt ./ rho0 for non-uniform grids.
    void GenerateInitialDenisty();
    /// Calculate square of velocity
    void Compute_c2();

    /// Calculate three temporary sums in the new pressure formula, non-linear absorbing case.
    void ComputePressurePartsNonLinear(RealMatrix& rho_part,
                                       RealMatrix& BonA_part,
                                       RealMatrix& du_part);

    /// Calculate two temporary sums in the new pressure formula, linear absorbing case.
    void ComputePressurePartsLinear(RealMatrix& rhoxyz_sum,
                                    RealMatrix& rho0_du_sum);

    /// Sum sub-terms to calculate new pressure, non-linear case.
    void SumPressureTermsNonlinear(RealMatrix& absorb_tau_temp,
                                   RealMatrix& absorb_eta_temp,
                                   RealMatrix& BonA_temp);

    /// Sum sub-terms to calculate new pressure, linear case.
    void SumPressureTermsLinear(RealMatrix& absorb_tau_temp,
                                RealMatrix& absorb_eta_temp,
                                RealMatrix& rhoxyz_sum);

    /// Sum sub-terms for new p, linear lossless case.
    void SumPressureNonlinearLossless();

    /// Sum sub-terms for new p, linear lossless case.
    void SumPressureLinearLossless();

    /// Calculate ux_shifted, uy_shifted and uz_shifted.
    void CalculateShiftedVelocity();

    /// Print progress statistics.
    void PrintStatistics();

    /// Is time to checkpoint (save actual state on disk).
    bool IsTimeToCheckpoint();

    /// Was the loop interrupted to checkpoint?
    bool IsCheckpointInterruption() const;

    /// Check the output file has the correct format and version.
    void CheckOutputFile();

    /// Check the checkpoint file has the correct format and version.
    void CheckCheckpointFile();

    /// Reads the header of the output file and sets the cumulative elapsed time from the first log.
    void LoadElapsedTimeFromOutputFile(Hdf5File& outputFile);

//-------------------------------------- Get matrices ----------------------------------------//
    /// Get the kappa matrix from the container.
    RealMatrix& Get_kappa()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kKappa);
    };
    /// Get the c^2 matrix from the container.
    RealMatrix& Get_c2()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixContainer::MatrixIdx::kC2);
    };

    /// Get the p matrix from the container.
    RealMatrix& Get_p()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kP);
    };

    /// Get the ux_sgx matrix from the container.
    RealMatrix& Get_ux_sgx()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kUxSgx);
    };
    /// Get the uy_sgy matrix from the container.
    RealMatrix& Get_uy_sgy()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kUySgy);
    };
    /// Get the uz_sgz matrix from the container.
    RealMatrix& Get_uz_sgz()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kUzSgz);
    };

    /// Get the ux_shifted matrix from the container.
    RealMatrix& Get_ux_shifted()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kUxShifted);
    };
    /// Get the uy_shifted matrix from the container.
    RealMatrix& Get_uy_shifted()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kUyShifted);
    };
    /// Get the uz_shifted matrix from the container.
    RealMatrix& Get_uz_shifted()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kUzShifted);
    };

    /// Get the duxdx matrix from the container.
    RealMatrix& Get_duxdx()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDuxdx);
    };
    /// Get the duydy matrix from the container.
    RealMatrix& Get_duydy()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDuydy);
    };
    /// Get the duzdz matrix from the container.
    RealMatrix& Get_duzdz()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDuzdz);
    };

    /// Get the dt.*rho0_sgx matrix from the container.
    RealMatrix& Get_dt_rho0_sgx()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDtRho0Sgx);
    };
    /// Get the dt.*rho0_sgy matrix from the container.
    RealMatrix& Get_dt_rho0_sgy()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDtRho0Sgy);
    };
    /// Get the dt.*rho0_sgz matrix from the container.
    RealMatrix& Get_dt_rho0_sgz()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDtRho0Sgz);
    };

    /// Get the rhox matrix from the container.
    RealMatrix& Get_rhox()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kRhoX);
    };
    /// Get the rhoy matrix from the container.
    RealMatrix& Get_rhoy()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kRhoY);
    };
    /// Get the rhoz matrix from the container.
    RealMatrix& Get_rhoz()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kRhoZ);
    };
    /// Get the rho0 matrix from the container.
    RealMatrix& Get_rho0()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kRho0);
    };

    /// Get the ddx_k_shift_pos matrix from the container.
    ComplexMatrix& Get_ddx_k_shift_pos()
    {
      return matrixContainer.getMatrix<ComplexMatrix>(MatrixContainer::MatrixIdx::kDdxKShiftPosR);
    };
    /// Get the ddy_k_shift_pos matrix from the container.
    ComplexMatrix& Get_ddy_k_shift_pos()
    {
      return matrixContainer.getMatrix<ComplexMatrix>(MatrixContainer::MatrixIdx::kDdyKShiftPos);
    };
    /// Get the ddz_k_shift_pos matrix from the container.
    ComplexMatrix& Get_ddz_k_shift_pos()
    {
      return matrixContainer.getMatrix<ComplexMatrix>(MatrixContainer::MatrixIdx::kDdzKShiftPos);
    };
    /// Get the ddx_k_shift_neg matrix from the container.
    ComplexMatrix& Get_ddx_k_shift_neg()
    {
      return matrixContainer.getMatrix<ComplexMatrix>(MatrixContainer::MatrixIdx::kDdxKShiftNegR);
    };
    /// Get the ddy_k_shift_neg matrix from the container.
    ComplexMatrix& Get_ddy_k_shift_neg()
    {
      return matrixContainer.getMatrix<ComplexMatrix>(MatrixContainer::MatrixIdx::kDdyKShiftNeg);
    };
    /// Get the ddz_k_shift_neg matrix from the container.
    ComplexMatrix& Get_ddz_k_shift_neg()
    {
      return matrixContainer.getMatrix<ComplexMatrix>(MatrixContainer::MatrixIdx::kDdzKShiftNeg);
    };

    /// Get the x_shift_neg_r matrix from the container.
    ComplexMatrix& Get_x_shift_neg_r()
    {
      return matrixContainer.getMatrix<ComplexMatrix>(MatrixContainer::MatrixIdx::kXShiftNegR);
    };
    /// Get the y_shift_neg_r from the container.
    ComplexMatrix& Get_y_shift_neg_r()
    {
      return matrixContainer.getMatrix<ComplexMatrix>(MatrixContainer::MatrixIdx::kYShiftNegR);
    };
    /// Get the y_shift_neg_r from the container.
    ComplexMatrix& Get_z_shift_neg_r()
    {
      return matrixContainer.getMatrix<ComplexMatrix>(MatrixContainer::MatrixIdx::kZShiftNegR);
    };

    /// Get the pml_x_sgx matrix from the container.
    RealMatrix& Get_pml_x_sgx()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kPmlXSgx);
    };
    /// Get the pml_y_sgy matrix from the container.
    RealMatrix& Get_pml_y_sgy()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kPmlYSgy);
    };
    /// Get the pml_z_sgz matrix from the container.
    RealMatrix& Get_pml_z_sgz()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kPmlZSgz);
    };

    /// Get the pml_x matrix from the container.
    RealMatrix& Get_pml_x()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kPmlX);
    };
    /// Get the pml_y matrix from the container.
    RealMatrix& Get_pml_y()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kPmlY);
    };
    /// Get the pml_z matrix from the container.
    RealMatrix& Get_pml_z()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kPmlZ);
    };


    /// Get the dxudxn matrix from the container.
    RealMatrix& Get_dxudxn()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDxudxn);
    };
    /// Get the dyudyn matrix from the container.
    RealMatrix& Get_dyudyn()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDyudyn);
    };
    /// Get the dzudzn matrix from the container.
    RealMatrix& Get_dzudzn()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDzudzn);
    };

    /// Get the dxudxn_sgx matrix from the container.
    RealMatrix& Get_dxudxn_sgx()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDxudxnSgx);
    };
    /// Get the dyudyn_sgy matrix from the container.
    RealMatrix& Get_dyudyn_sgy()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDyudynSgy);
    };
    /// Get the dzudzn_sgz matrix from the container.
    RealMatrix& Get_dzudzn_sgz()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kDzudznSgz);
    };


    /// Get the BonA matrix from the container.
    RealMatrix& Get_BonA()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kBOnA);
    };
    /// Get the absorb_tau matrix from the container.
    RealMatrix& Get_absorb_tau()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kAbsorbTau);
    };
    /// Get the absorb_eta matrix from the container.
    RealMatrix& Get_absorb_eta()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kAbsorbEta);
    };

    /// Get the absorb_nabla1 matrix from the container.
    RealMatrix& Get_absorb_nabla1()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kAbsorbNabla1);
    };
    /// Get the absorb_nabla2 matrix from the container.
    RealMatrix& Get_absorb_nabla2()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kAbsorbNabla2);
    };


    //-- Index matrices --//

    /// Get the sensor_mask_index matrix from the container.
    IndexMatrix& Get_sensor_mask_index()
    {
      return matrixContainer.getMatrix<IndexMatrix>(MatrixContainer::MatrixIdx::kSensorMaskIndex);
    };


    /// Get the sensor_mask_corners matrix from the container.
    IndexMatrix& Get_sensor_mask_corners()
    {
      return matrixContainer.getMatrix<IndexMatrix>(MatrixContainer::MatrixIdx::kSensorMaskCorners);
    };

    /// Get the u_source_index matrix from the container.
    IndexMatrix& Get_u_source_index()
    {
      return matrixContainer.getMatrix<IndexMatrix>(MatrixContainer::MatrixIdx::kVelocitySourceIndex);
    };
    /// Get the p_source_index matrix from the container.
    IndexMatrix& Get_p_source_index()
    {
      return matrixContainer.getMatrix<IndexMatrix>(MatrixContainer::MatrixIdx::kPressureSourceIndex);
    };
    /// Get the delay_mask matrix from the container.
    IndexMatrix& Get_delay_mask()
    {
      return matrixContainer.getMatrix<IndexMatrix>(MatrixContainer::MatrixIdx::kDelayMask);
    }


    //-- sources  --//

    /// Get the transducer_source_input matrix from the container.
    RealMatrix& Get_transducer_source_input()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kTransducerSourceInput);
    };

    /// Get the p_source_input matrix from the container.
    RealMatrix& Get_p_source_input()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kPressureSourceInput);
    };

    /// Get the p0_source_input from the container.
    RealMatrix& Get_p0_source_input()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kInitialPressureSourceInput);
    };


    /// Get the ux_source_input matrix from the container.
    RealMatrix& Get_ux_source_input()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kVelocityXSourceInput);
    };
    /// Get the uy_source_input matrix from the container.
    RealMatrix& Get_uy_source_input()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kVelocityYSourceInput);
    };
    /// Get the uz_source_input matrix from the container.
    RealMatrix& Get_uz_source_input()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kVelocityZSourceInput);
    };


    //--Temporary matrices --//

    /// Get the Temp_1_RS3D matrix from the container.
    RealMatrix& Get_temp_1_real_3D()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kTemp1Real3D);
    };
    /// Get the Temp_2_RS3D matrix from the container.
    RealMatrix& Get_temp_2_real_3D()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kTemp2Real3D);
    };
    /// Get the Temp_3_RS3D matrix from the container.
    RealMatrix& Get_temp_3_real_3D()
    {
      return matrixContainer.getMatrix<RealMatrix>(MatrixContainer::MatrixIdx::kTemp3Real3D);
    };


    /// Get the CUFFT_X_temp from the container.
    CufftComplexMatrix& Get_cufft_x_temp()
    {
      return matrixContainer.getMatrix<CufftComplexMatrix>(MatrixContainer::MatrixIdx::kTempCufftX);
    };
    /// Get the FFT_Y_temp from the container.
    CufftComplexMatrix& Get_cufft_y_temp()
    {
      return matrixContainer.getMatrix<CufftComplexMatrix>(MatrixContainer::MatrixIdx::kTempCufftY);
    };
    /// Get the FFT_Z_temp from the container.
    CufftComplexMatrix& Get_cufft_z_temp()
    {
      return matrixContainer.getMatrix<CufftComplexMatrix>(MatrixContainer::MatrixIdx::kTempCufftZ);
    };

    /// Get the FFT_shift_temp the container.
    CufftComplexMatrix& Get_cufft_shift_temp()
    {
      return matrixContainer.getMatrix<CufftComplexMatrix>(MatrixContainer::MatrixIdx::kTempCufftShift);
    };

private:

    /// Matrix container with all the matrix classes
    MatrixContainer matrixContainer;
    /// Output stream container
    OutputStreamContainer outputStreamContainer;

    /// Global parameters of the simulation
    Parameters& parameters;

    /// Percentage of the simulation done
    size_t actPercent;

    /// This variable is true when calculating first time step after restore from checkpoint (to allow asynchronous IO)
    bool isTimestepRightAfterRestore;

    /// Total time of the simulation
    TimeMeasure totalTime;
    /// Pre-processing time of the simulation
    TimeMeasure preProcessingTime;
    /// Data load time of the simulation
    TimeMeasure dataLoadTime;
    /// Simulation time of the simulation
    TimeMeasure simulationTime;
    /// Post-processing time of the simulation
    TimeMeasure postProcessingTime;
    /// Iteration time of the simulation
    TimeMeasure iterationTime;

};// end of TKSpaceFirstOrder3DSolver
//--------------------------------------------------------------------------------------------------

#endif /* TKSPACE_FIRST_ORDER_3D_SOLVER_H */
