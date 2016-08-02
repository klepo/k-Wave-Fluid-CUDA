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
 *              26 July     2016, 15:42 (revised)
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


#include <iostream>

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
    /// Destructor.
    virtual ~TKSpaceFirstOrder3DSolver();

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

    /// Copy constructor not allowed for public.
    TKSpaceFirstOrder3DSolver(const TKSpaceFirstOrder3DSolver& src);
    /// operator = not allowed for public.
    TKSpaceFirstOrder3DSolver& operator= (const TKSpaceFirstOrder3DSolver& src);

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
    void ComputePressurePartsNonLinear(TRealMatrix& rho_part,
                                       TRealMatrix& BonA_part,
                                       TRealMatrix& du_part);

    /// Calculate two temporary sums in the new pressure formula, linear absorbing case.
    void ComputePressurePartsLinear(TRealMatrix& rhoxyz_sum,
                                    TRealMatrix& rho0_du_sum);

    /// Sum sub-terms to calculate new pressure, non-linear case.
    void SumPressureTermsNonlinear(TRealMatrix& absorb_tau_temp,
                                   TRealMatrix& absorb_eta_temp,
                                   TRealMatrix& BonA_temp);

    /// Sum sub-terms to calculate new pressure, linear case.
    void SumPressureTermsLinear(TRealMatrix& absorb_tau_temp,
                                TRealMatrix& absorb_eta_temp,
                                TRealMatrix& rhoxyz_sum);

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
    void LoadElapsedTimeFromOutputFile(THDF5_File& o1utputFile);

    //------------------------- Get matrices --------------------------------//
    /// Get the kappa matrix from the container.
    TRealMatrix& Get_kappa()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(kappa);
    };
    /// Get the c^2 matrix from the container.
    TRealMatrix& Get_c2()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(c2);
    };

    /// Get the p matrix from the container.
    TRealMatrix& Get_p()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(p);
    };

    /// Get the ux_sgx matrix from the container.
    TRealMatrix& Get_ux_sgx()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(ux_sgx);
    };
    /// Get the uy_sgy matrix from the container.
    TRealMatrix& Get_uy_sgy()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(uy_sgy);
    };
    /// Get the uz_sgz matrix from the container.
    TRealMatrix& Get_uz_sgz()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(uz_sgz);
    };

    /// Get the ux_shifted matrix from the container.
    TRealMatrix& Get_ux_shifted()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(ux_shifted);
    };
    /// Get the uy_shifted matrix from the container.
    TRealMatrix& Get_uy_shifted()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(uy_shifted);
    };
    /// Get the uz_shifted matrix from the container.
    TRealMatrix& Get_uz_shifted()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(uz_shifted);
    };

    /// Get the duxdx matrix from the container.
    TRealMatrix& Get_duxdx()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(duxdx);
    };
    /// Get the duydy matrix from the container.
    TRealMatrix& Get_duydy()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(duydy);
    };
    /// Get the duzdz matrix from the container.
    TRealMatrix& Get_duzdz()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(duzdz);
    };

    /// Get the dt.*rho0_sgx matrix from the container.
    TRealMatrix& Get_dt_rho0_sgx()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(dt_rho0_sgx);
    };
    /// Get the dt.*rho0_sgy matrix from the container.
    TRealMatrix& Get_dt_rho0_sgy()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(dt_rho0_sgy);
    };
    /// Get the dt.*rho0_sgz matrix from the container.
    TRealMatrix& Get_dt_rho0_sgz()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(dt_rho0_sgz);
    };

    /// Get the rhox matrix from the container.
    TRealMatrix& Get_rhox()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(rhox);
    };
    /// Get the rhoy matrix from the container.
    TRealMatrix& Get_rhoy()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(rhoy);
    };
    /// Get the rhoz matrix from the container.
    TRealMatrix& Get_rhoz()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(rhoz);
    };
    /// Get the rho0 matrix from the container.
    TRealMatrix& Get_rho0()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(rho0);
    };

    /// Get the ddx_k_shift_pos matrix from the container.
    TComplexMatrix& Get_ddx_k_shift_pos()
    {
      return matrixContainer.GetMatrix<TComplexMatrix>(ddx_k_shift_pos);
    };
    /// Get the ddy_k_shift_pos matrix from the container.
    TComplexMatrix& Get_ddy_k_shift_pos()
    {
      return matrixContainer.GetMatrix<TComplexMatrix>(ddy_k_shift_pos);
    };
    /// Get the ddz_k_shift_pos matrix from the container.
    TComplexMatrix& Get_ddz_k_shift_pos()
    {
      return matrixContainer.GetMatrix<TComplexMatrix>(ddz_k_shift_pos);
    };
    /// Get the ddx_k_shift_neg matrix from the container.
    TComplexMatrix& Get_ddx_k_shift_neg()
    {
      return matrixContainer.GetMatrix<TComplexMatrix>(ddx_k_shift_neg);
    };
    /// Get the ddy_k_shift_neg matrix from the container.
    TComplexMatrix& Get_ddy_k_shift_neg()
    {
      return matrixContainer.GetMatrix<TComplexMatrix>(ddy_k_shift_neg);
    };
    /// Get the ddz_k_shift_neg matrix from the container.
    TComplexMatrix& Get_ddz_k_shift_neg()
    {
      return matrixContainer.GetMatrix<TComplexMatrix>(ddz_k_shift_neg);
    };

    /// Get the x_shift_neg_r matrix from the container.
    TComplexMatrix& Get_x_shift_neg_r()
    {
      return matrixContainer.GetMatrix<TComplexMatrix>(x_shift_neg_r);
    };
    /// Get the y_shift_neg_r from the container.
    TComplexMatrix& Get_y_shift_neg_r()
    {
      return matrixContainer.GetMatrix<TComplexMatrix>(y_shift_neg_r);
    };
    /// Get the y_shift_neg_r from the container.
    TComplexMatrix& Get_z_shift_neg_r()
    {
      return matrixContainer.GetMatrix<TComplexMatrix>(z_shift_neg_r);
    };

    /// Get the pml_x_sgx matrix from the container.
    TRealMatrix& Get_pml_x_sgx()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(pml_x_sgx);
    };
    /// Get the pml_y_sgy matrix from the container.
    TRealMatrix& Get_pml_y_sgy()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(pml_y_sgy);
    };
    /// Get the pml_z_sgz matrix from the container.
    TRealMatrix& Get_pml_z_sgz()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(pml_z_sgz);
    };

    /// Get the pml_x matrix from the container.
    TRealMatrix& Get_pml_x()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(pml_x);
    };
    /// Get the pml_y matrix from the container.
    TRealMatrix& Get_pml_y()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(pml_y);
    };
    /// Get the pml_z matrix from the container.
    TRealMatrix& Get_pml_z()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(pml_z);
    };


    /// Get the dxudxn matrix from the container.
    TRealMatrix& Get_dxudxn()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(dxudxn);
    };
    /// Get the dyudyn matrix from the container.
    TRealMatrix& Get_dyudyn()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(dyudyn);
    };
    /// Get the dzudzn matrix from the container.
    TRealMatrix& Get_dzudzn()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(dzudzn);
    };

    /// Get the dxudxn_sgx matrix from the container.
    TRealMatrix& Get_dxudxn_sgx()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(dxudxn_sgx);
    };
    /// Get the dyudyn_sgy matrix from the container.
    TRealMatrix& Get_dyudyn_sgy()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(dyudyn_sgy);
    };
    /// Get the dzudzn_sgz matrix from the container.
    TRealMatrix& Get_dzudzn_sgz()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(dzudzn_sgz);
    };


    /// Get the BonA matrix from the container.
    TRealMatrix& Get_BonA()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(BonA);
    };
    /// Get the absorb_tau matrix from the container.
    TRealMatrix& Get_absorb_tau()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(absorb_tau);
    };
    /// Get the absorb_eta matrix from the container.
    TRealMatrix& Get_absorb_eta()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(absorb_eta);
    };

    /// Get the absorb_nabla1 matrix from the container.
    TRealMatrix& Get_absorb_nabla1()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(absorb_nabla1);
    };
    /// Get the absorb_nabla2 matrix from the container.
    TRealMatrix& Get_absorb_nabla2()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(absorb_nabla2);
    };


    //-- Index matrices --//

    /// Get the sensor_mask_index matrix from the container.
    TIndexMatrix& Get_sensor_mask_index()
    {
      return matrixContainer.GetMatrix<TIndexMatrix>(sensor_mask_index);
    };


    /// Get the sensor_mask_corners matrix from the container.
    TIndexMatrix& Get_sensor_mask_corners()
    {
      return matrixContainer.GetMatrix<TIndexMatrix>(sensor_mask_corners);
    };

    /// Get the u_source_index matrix from the container.
    TIndexMatrix& Get_u_source_index()
    {
      return matrixContainer.GetMatrix<TIndexMatrix>(u_source_index);
    };
    /// Get the p_source_index matrix from the container.
    TIndexMatrix& Get_p_source_index()
    {
      return matrixContainer.GetMatrix<TIndexMatrix>(p_source_index);
    };
    /// Get the delay_mask matrix from the container.
    TIndexMatrix& Get_delay_mask()
    {
      return matrixContainer.GetMatrix<TIndexMatrix>(delay_mask);
    }


    //-- sources  --//

    /// Get the transducer_source_input matrix from the container.
    TRealMatrix& Get_transducer_source_input()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(transducer_source_input);
    };

    /// Get the p_source_input matrix from the container.
    TRealMatrix& Get_p_source_input()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(p_source_input);
    };

    /// Get the p0_source_input from the container.
    TRealMatrix& Get_p0_source_input()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(p0_source_input);
    };


    /// Get the ux_source_input matrix from the container.
    TRealMatrix& Get_ux_source_input()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(ux_source_input);
    };
    /// Get the uy_source_input matrix from the container.
    TRealMatrix& Get_uy_source_input()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(uy_source_input);
    };
    /// Get the uz_source_input matrix from the container.
    TRealMatrix& Get_uz_source_input()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(uz_source_input);
    };


    //--Temporary matrices --//

    /// Get the Temp_1_RS3D matrix from the container.
    TRealMatrix& Get_temp_1_real_3D()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(temp_1_real_3D);
    };
    /// Get the Temp_2_RS3D matrix from the container.
    TRealMatrix& Get_temp_2_real_3D()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(temp_2_real_3D);
    };
    /// Get the Temp_3_RS3D matrix from the container.
    TRealMatrix& Get_temp_3_real_3D()
    {
      return matrixContainer.GetMatrix<TRealMatrix>(temp_3_real_3D);
    };


    /// Get the CUFFT_X_temp from the container.
    TCUFFTComplexMatrix& Get_cufft_x_temp()
    {
      return matrixContainer.GetMatrix<TCUFFTComplexMatrix>(cufft_x_temp);
    };
    /// Get the FFT_Y_temp from the container.
    TCUFFTComplexMatrix& Get_cufft_y_temp()
    {
      return matrixContainer.GetMatrix<TCUFFTComplexMatrix>(cufft_y_temp);
    };
    /// Get the FFT_Z_temp from the container.
    TCUFFTComplexMatrix& Get_cufft_z_temp()
    {
      return matrixContainer.GetMatrix<TCUFFTComplexMatrix>(cufft_z_temp);
    };

    /// Get the FFT_shift_temp the container.
    TCUFFTComplexMatrix& Get_cufft_shift_temp()
    {
      return matrixContainer.GetMatrix<TCUFFTComplexMatrix>(cufft_shift_temp);
    };

private:

    /// Matrix container with all the matrix classes
    TMatrixContainer matrixContainer;
    /// Output stream container
    TOutputStreamContainer outputStreamContainer;

    /// Global parameters of the simulation
    TParameters& parameters;

    /// Percentage of the simulation done
    size_t actPercent;

    /// This variable is true when calculating first time step after restore from checkpoint (to allow asynchronous IO)
    bool isTimestepRightAfterRestore;

    /// Total time of the simulation
    TTimeMeasure totalTime;
    /// Pre-processing time of the simulation
    TTimeMeasure preProcessingTime;
    /// Data load time of the simulation
    TTimeMeasure dataLoadTime;
    /// Simulation time of the simulation
    TTimeMeasure simulationTime;
    /// Post-processing time of the simulation
    TTimeMeasure postProcessingTime;
    /// Iteration time of the simulation
    TTimeMeasure iterationTime;

};// end of TKSpaceFirstOrder3DSolver
//--------------------------------------------------------------------------------------------------

#endif /* TKSPACE_FIRST_ORDER_3D_SOLVER_H */
