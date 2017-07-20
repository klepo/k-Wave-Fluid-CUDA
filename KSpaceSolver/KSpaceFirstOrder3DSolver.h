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
 * @date        12 July     2012, 10:27 (created)\n
 *              20 July     2017, 14:14 (revised)
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
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::kappa);
    };
    /// Get the c^2 matrix from the container.
    RealMatrix& Get_c2()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixContainer::TMatrixIdx::c2);
    };

    /// Get the p matrix from the container.
    RealMatrix& Get_p()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::p);
    };

    /// Get the ux_sgx matrix from the container.
    RealMatrix& Get_ux_sgx()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::ux_sgx);
    };
    /// Get the uy_sgy matrix from the container.
    RealMatrix& Get_uy_sgy()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::uy_sgy);
    };
    /// Get the uz_sgz matrix from the container.
    RealMatrix& Get_uz_sgz()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::uz_sgz);
    };

    /// Get the ux_shifted matrix from the container.
    RealMatrix& Get_ux_shifted()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::ux_shifted);
    };
    /// Get the uy_shifted matrix from the container.
    RealMatrix& Get_uy_shifted()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::uy_shifted);
    };
    /// Get the uz_shifted matrix from the container.
    RealMatrix& Get_uz_shifted()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::uz_shifted);
    };

    /// Get the duxdx matrix from the container.
    RealMatrix& Get_duxdx()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::duxdx);
    };
    /// Get the duydy matrix from the container.
    RealMatrix& Get_duydy()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::duydy);
    };
    /// Get the duzdz matrix from the container.
    RealMatrix& Get_duzdz()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::duzdz);
    };

    /// Get the dt.*rho0_sgx matrix from the container.
    RealMatrix& Get_dt_rho0_sgx()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::dt_rho0_sgx);
    };
    /// Get the dt.*rho0_sgy matrix from the container.
    RealMatrix& Get_dt_rho0_sgy()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::dt_rho0_sgy);
    };
    /// Get the dt.*rho0_sgz matrix from the container.
    RealMatrix& Get_dt_rho0_sgz()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::dt_rho0_sgz);
    };

    /// Get the rhox matrix from the container.
    RealMatrix& Get_rhox()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::rhox);
    };
    /// Get the rhoy matrix from the container.
    RealMatrix& Get_rhoy()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::rhoy);
    };
    /// Get the rhoz matrix from the container.
    RealMatrix& Get_rhoz()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::rhoz);
    };
    /// Get the rho0 matrix from the container.
    RealMatrix& Get_rho0()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::rho0);
    };

    /// Get the ddx_k_shift_pos matrix from the container.
    ComplexMatrix& Get_ddx_k_shift_pos()
    {
      return matrixContainer.GetMatrix<ComplexMatrix>(TMatrixContainer::TMatrixIdx::ddx_k_shift_pos);
    };
    /// Get the ddy_k_shift_pos matrix from the container.
    ComplexMatrix& Get_ddy_k_shift_pos()
    {
      return matrixContainer.GetMatrix<ComplexMatrix>(TMatrixContainer::TMatrixIdx::ddy_k_shift_pos);
    };
    /// Get the ddz_k_shift_pos matrix from the container.
    ComplexMatrix& Get_ddz_k_shift_pos()
    {
      return matrixContainer.GetMatrix<ComplexMatrix>(TMatrixContainer::TMatrixIdx::ddz_k_shift_pos);
    };
    /// Get the ddx_k_shift_neg matrix from the container.
    ComplexMatrix& Get_ddx_k_shift_neg()
    {
      return matrixContainer.GetMatrix<ComplexMatrix>(TMatrixContainer::TMatrixIdx::ddx_k_shift_neg);
    };
    /// Get the ddy_k_shift_neg matrix from the container.
    ComplexMatrix& Get_ddy_k_shift_neg()
    {
      return matrixContainer.GetMatrix<ComplexMatrix>(TMatrixContainer::TMatrixIdx::ddy_k_shift_neg);
    };
    /// Get the ddz_k_shift_neg matrix from the container.
    ComplexMatrix& Get_ddz_k_shift_neg()
    {
      return matrixContainer.GetMatrix<ComplexMatrix>(TMatrixContainer::TMatrixIdx::ddz_k_shift_neg);
    };

    /// Get the x_shift_neg_r matrix from the container.
    ComplexMatrix& Get_x_shift_neg_r()
    {
      return matrixContainer.GetMatrix<ComplexMatrix>(TMatrixContainer::TMatrixIdx::x_shift_neg_r);
    };
    /// Get the y_shift_neg_r from the container.
    ComplexMatrix& Get_y_shift_neg_r()
    {
      return matrixContainer.GetMatrix<ComplexMatrix>(TMatrixContainer::TMatrixIdx::y_shift_neg_r);
    };
    /// Get the y_shift_neg_r from the container.
    ComplexMatrix& Get_z_shift_neg_r()
    {
      return matrixContainer.GetMatrix<ComplexMatrix>(TMatrixContainer::TMatrixIdx::z_shift_neg_r);
    };

    /// Get the pml_x_sgx matrix from the container.
    RealMatrix& Get_pml_x_sgx()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::pml_x_sgx);
    };
    /// Get the pml_y_sgy matrix from the container.
    RealMatrix& Get_pml_y_sgy()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::pml_y_sgy);
    };
    /// Get the pml_z_sgz matrix from the container.
    RealMatrix& Get_pml_z_sgz()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::pml_z_sgz);
    };

    /// Get the pml_x matrix from the container.
    RealMatrix& Get_pml_x()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::pml_x);
    };
    /// Get the pml_y matrix from the container.
    RealMatrix& Get_pml_y()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::pml_y);
    };
    /// Get the pml_z matrix from the container.
    RealMatrix& Get_pml_z()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::pml_z);
    };


    /// Get the dxudxn matrix from the container.
    RealMatrix& Get_dxudxn()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::dxudxn);
    };
    /// Get the dyudyn matrix from the container.
    RealMatrix& Get_dyudyn()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::dyudyn);
    };
    /// Get the dzudzn matrix from the container.
    RealMatrix& Get_dzudzn()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::dzudzn);
    };

    /// Get the dxudxn_sgx matrix from the container.
    RealMatrix& Get_dxudxn_sgx()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::dxudxn_sgx);
    };
    /// Get the dyudyn_sgy matrix from the container.
    RealMatrix& Get_dyudyn_sgy()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::dyudyn_sgy);
    };
    /// Get the dzudzn_sgz matrix from the container.
    RealMatrix& Get_dzudzn_sgz()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::dzudzn_sgz);
    };


    /// Get the BonA matrix from the container.
    RealMatrix& Get_BonA()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::BonA);
    };
    /// Get the absorb_tau matrix from the container.
    RealMatrix& Get_absorb_tau()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::absorb_tau);
    };
    /// Get the absorb_eta matrix from the container.
    RealMatrix& Get_absorb_eta()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::absorb_eta);
    };

    /// Get the absorb_nabla1 matrix from the container.
    RealMatrix& Get_absorb_nabla1()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::absorb_nabla1);
    };
    /// Get the absorb_nabla2 matrix from the container.
    RealMatrix& Get_absorb_nabla2()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::absorb_nabla2);
    };


    //-- Index matrices --//

    /// Get the sensor_mask_index matrix from the container.
    IndexMatrix& Get_sensor_mask_index()
    {
      return matrixContainer.GetMatrix<IndexMatrix>(TMatrixContainer::TMatrixIdx::sensor_mask_index);
    };


    /// Get the sensor_mask_corners matrix from the container.
    IndexMatrix& Get_sensor_mask_corners()
    {
      return matrixContainer.GetMatrix<IndexMatrix>(TMatrixContainer::TMatrixIdx::sensor_mask_corners);
    };

    /// Get the u_source_index matrix from the container.
    IndexMatrix& Get_u_source_index()
    {
      return matrixContainer.GetMatrix<IndexMatrix>(TMatrixContainer::TMatrixIdx::u_source_index);
    };
    /// Get the p_source_index matrix from the container.
    IndexMatrix& Get_p_source_index()
    {
      return matrixContainer.GetMatrix<IndexMatrix>(TMatrixContainer::TMatrixIdx::p_source_index);
    };
    /// Get the delay_mask matrix from the container.
    IndexMatrix& Get_delay_mask()
    {
      return matrixContainer.GetMatrix<IndexMatrix>(TMatrixContainer::TMatrixIdx::delay_mask);
    }


    //-- sources  --//

    /// Get the transducer_source_input matrix from the container.
    RealMatrix& Get_transducer_source_input()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::transducer_source_input);
    };

    /// Get the p_source_input matrix from the container.
    RealMatrix& Get_p_source_input()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::p_source_input);
    };

    /// Get the p0_source_input from the container.
    RealMatrix& Get_p0_source_input()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::p0_source_input);
    };


    /// Get the ux_source_input matrix from the container.
    RealMatrix& Get_ux_source_input()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::ux_source_input);
    };
    /// Get the uy_source_input matrix from the container.
    RealMatrix& Get_uy_source_input()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::uy_source_input);
    };
    /// Get the uz_source_input matrix from the container.
    RealMatrix& Get_uz_source_input()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::uz_source_input);
    };


    //--Temporary matrices --//

    /// Get the Temp_1_RS3D matrix from the container.
    RealMatrix& Get_temp_1_real_3D()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::temp_1_real_3D);
    };
    /// Get the Temp_2_RS3D matrix from the container.
    RealMatrix& Get_temp_2_real_3D()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::temp_2_real_3D);
    };
    /// Get the Temp_3_RS3D matrix from the container.
    RealMatrix& Get_temp_3_real_3D()
    {
      return matrixContainer.GetMatrix<RealMatrix>(TMatrixContainer::TMatrixIdx::temp_3_real_3D);
    };


    /// Get the CUFFT_X_temp from the container.
    CufftComplexMatrix& Get_cufft_x_temp()
    {
      return matrixContainer.GetMatrix<CufftComplexMatrix>(TMatrixContainer::TMatrixIdx::cufft_x_temp);
    };
    /// Get the FFT_Y_temp from the container.
    CufftComplexMatrix& Get_cufft_y_temp()
    {
      return matrixContainer.GetMatrix<CufftComplexMatrix>(TMatrixContainer::TMatrixIdx::cufft_y_temp);
    };
    /// Get the FFT_Z_temp from the container.
    CufftComplexMatrix& Get_cufft_z_temp()
    {
      return matrixContainer.GetMatrix<CufftComplexMatrix>(TMatrixContainer::TMatrixIdx::cufft_z_temp);
    };

    /// Get the FFT_shift_temp the container.
    CufftComplexMatrix& Get_cufft_shift_temp()
    {
      return matrixContainer.GetMatrix<CufftComplexMatrix>(TMatrixContainer::TMatrixIdx::cufft_shift_temp);
    };

private:

    /// Matrix container with all the matrix classes
    TMatrixContainer matrixContainer;
    /// Output stream container
    TOutputStreamContainer outputStreamContainer;

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
