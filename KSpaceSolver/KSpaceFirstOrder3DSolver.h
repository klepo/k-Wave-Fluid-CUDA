/**
 * @file        KSpaceFirstOrder3DSolver.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the main class of the
 *              project responsible for the entire simulation.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        12 July     2012, 10:27 (created)\n
 *              05 December 2014, 13:40 (revised)
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

#ifndef TKSPACEFIRSTORDER3DSOLVER_H
#define	TKSPACEFIRSTORDER3DSOLVER_H



#include <iostream>

#include <Parameters/Parameters.h>
#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>
#include <MatrixClasses/IndexMatrix.h>

#include <Containers/MatrixContainer.h>
#include <Containers//OutputStreamContainer.h>

#include <Utils/TimeMeasure.h>

#include <CUDA/CUDAImplementations.h>


using namespace std;

/**
 * @class TKSpaceFirstOrder3DSolver
 * @brief Class responsible for running the k-space first order 3D method.
 * @details Class responsible for running the k-space first order 3D method. This
 *          class maintain the whole k-wave (implements the time loop).
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

    /// Print parameters of the simulation.
    void PrintParametersOfSimulation(FILE * file);

    /// Get memory usage in MB on the CPU side.
    size_t GetHostMemoryUsageInMB();
    /// Get memory usage in MB on the GPU side.
    size_t GetDeviceMemoryUsageInMB();


    // Get code name.
    const string GetCodeName() const { return "kspaceFirstOrder3D-CUDA v1.1"; };

    /// Print the code name and license.
    void PrintFullNameCodeAndLicense(FILE * file);

    /// Set processor affinity.
    void SetProcessorAffinity();

    /// Get total simulation time.
    double GetTotalTime()          const { return TotalTime.GetElapsedTime(); };
    /// Get pre-processing time.
    double GetPreProcessingTime()  const { return PreProcessingTime.GetElapsedTime(); };
    /// Get data load time.
    double GetDataLoadTime()       const { return DataLoadTime.GetElapsedTime(); };
    /// Get simulation time (time loop).
    double GetSimulationTime()     const { return SimulationTime.GetElapsedTime(); };
    /// Get post-processing time.
    double GetPostProcessingTime() const  { return PostProcessingTime.GetElapsedTime(); };

    /// Get total simulation time cumulated over all legs.
    double GetCumulatedTotalTime()          const { return TotalTime.GetCumulatedElapsedTimeOverAllLegs();          };
    /// Get pre-processing time cumulated over all legs.
    double GetCumulatedPreProcessingTime()  const { return PreProcessingTime.GetCumulatedElapsedTimeOverAllLegs();  };
    /// Get data load time cumulated over all legs.
    double GetCumulatedDataLoadTime()       const { return DataLoadTime.GetCumulatedElapsedTimeOverAllLegs();       };
    /// Get simulation time (time loop) cumulated over all legs.
    double GetCumulatedSimulationTime()     const { return SimulationTime.GetCumulatedElapsedTimeOverAllLegs();     };
    /// Get post-processing time cumulated over all legs.
    double GetCumulatedPostProcessingTime() const { return PostProcessingTime.GetCumulatedElapsedTimeOverAllLegs(); };

protected:

    /// Copy constructor not allowed for public.
    TKSpaceFirstOrder3DSolver(const TKSpaceFirstOrder3DSolver& src);

    /// operator = not allowed for public.
    TKSpaceFirstOrder3DSolver& operator = (const TKSpaceFirstOrder3DSolver& src);

    /// Determine device memory.
    bool DoesDeviceHaveEnoughMemory();

    /// Initialize FFT plans.
    void InitializeFFTPlans();

    /// Compute pre-processing phase.
    void PreProcessingPhase();

    /// Compute the main time loop of the kspaceFirstOrder3D.
    void ComputeMainLoop();

    /// Post processing, and closing the output streams.
    void PostProcessing();

    // Store sensor data.
    void StoreSensorData();

    // Write statistics and header into the output file.
    void WriteOutputDataInfo();

    /// Save checkpoint data.
    void SaveCheckpointData();

    /// compute new values of for ux_sgx, uy_sgy, uz_sgz.
    void Compute_uxyz();

    /// Compute new values of for duxdx, duydy, dzdz.
    void Compute_duxyz();

    /// Compute new values of rhox, rhoy, rhoz for non-linear case.
    void Compute_rhoxyz_nonlinear();

    /// Compute new values of rhox, rhoy, rhoz for linear case.
    void Compute_rhoxyz_linear();

    /// Add u source to the particle velocity.
    void Add_u_source();

    /// Add in pressure source.
    void Add_p_source();

    /// Generate kappa matrix for non-absorbing media.
    void Generate_kappa();

    /// Generate kappa matrix, absorb_nabla1, absorb_nabla2 for absorbing media
    void Generate_kappa_absorb_nabla1_absorb_nabla2();

    /// Generate absorb_tau, absorb_eta for heterogenous media.
    void Generate_absorb_tau_absorb_eta_matrix();

    /// Calculate dt ./ rho0 for non-uniform grids.
    void Calculate_dt_rho0_non_uniform();

    /// Calculate p0_source.
    void Calculate_p0_source();

    void Compute_c2();

    /// Sum sub-terms non-linear case,
    void Compute_new_p_nonlinear();

    /// Sum sub-terms linear-case, absorbing.
    void Compute_new_p_linear();

    /// Calculate three temporary sums in the new pressure formula, non-linear absorbing case.
    void Calculate_SumRho_BonA_SumDu(TRealMatrix& RHO_Temp,
                                     TRealMatrix& BonA_Temp,
                                     TRealMatrix& Sum_du);

    /// Calculate two temporary sums in the new pressure formula, linear absorbing case.
    void Calculate_SumRho_SumRhoDu(TRealMatrix& Sum_rhoxyz,
                                   TRealMatrix& Sum_rho0_du);

    /// Sum sub-terms to calculate new pressure, non-linear case.
    void Sum_Subterms_nonlinear(TRealMatrix& Absorb_tau_temp,
                                TRealMatrix& Absorb_eta_temp,
                                TRealMatrix& BonA_temp);

    /// Sum sub-terms to calculate new pressure, linear case.
    void Sum_Subterms_linear(TRealMatrix& Absorb_tau_temp,
                             TRealMatrix& Absorb_eta_temp,
                             TRealMatrix& Sum_rhoxyz);

    /// Sum sub-terms for new p, linear lossless case.
    void Sum_new_p_nonlinear_lossless();

    /// Sum sub-terms for new p, linear lossless case.
    void Sum_new_p_linear_lossless();

    /// Print progress statistics.
    void PrintStatistics();

    /// Print the header of the progress statistics.
    void PrintOutputHeader();

    /// Is time to checkpoint (save actual state on disk).
    bool IsTimeToCheckpoint();

    /// Was the loop interrupted to checkpoint?
    bool IsCheckpointInterruption() const
    {
      return (Parameters->Get_t_index() != Parameters->Get_Nt());
    };

    /// Check the output file has the correct format and version.
    void CheckOutputFile();

    /// Check the checkpoint file has the correct format and version.
    void CheckCheckpointFile();

    /// Reads the header of the output file and sets the cumulative elapsed time from the first log.
    void RestoreCumulatedElapsedFromOutputFile(THDF5_File& HDF5_OutputFile);

    //------------------------- Get matrices --------------------------------//
    /// Get the kappa matrix from the container.
    TRealMatrix& Get_kappa()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(kappa);
    };
    /// Get the c^2 matrix from the container.
    TRealMatrix& Get_c2()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(c2);
    };

    /// Get the p matrix from the container.
    TRealMatrix& Get_p()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(p);
    };

    /// Get the ux_sgx matrix from the container.
    TRealMatrix& Get_ux_sgx()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(ux_sgx);
    };
    /// Get the uy_sgy matrix from the container.
    TRealMatrix& Get_uy_sgy()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(uy_sgy);
    };
    /// Get the uz_sgz matrix from the container.
    TRealMatrix& Get_uz_sgz()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(uz_sgz);
    };

    /// Get the ux_shifted matrix from the container.
    TRealMatrix& Get_ux_shifted()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(ux_shifted);
    };
    /// Get the uy_shifted matrix from the container.
    TRealMatrix& Get_uy_shifted()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(uy_shifted);
    };
    /// Get the uz_shifted matrix from the container.
    TRealMatrix& Get_uz_shifted()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(uz_shifted);
    };

    /// Get the duxdx matrix from the container.
    TRealMatrix& Get_duxdx()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(duxdx);
    };
    /// Get the duydy matrix from the container.
    TRealMatrix& Get_duydy()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(duydy);
    };
    /// Get the duzdz matrix from the container.
    TRealMatrix& Get_duzdz()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(duzdz);
    };

    /// Get the dt.*rho0_sgx matrix from the container.
    TRealMatrix& Get_dt_rho0_sgx()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(dt_rho0_sgx);
    };
    /// Get the dt.*rho0_sgy matrix from the container.
    TRealMatrix& Get_dt_rho0_sgy()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(dt_rho0_sgy);
    };
    /// Get the dt.*rho0_sgz matrix from the container.
    TRealMatrix& Get_dt_rho0_sgz()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(dt_rho0_sgz);
    };

    /// Get the rhox matrix from the container.
    TRealMatrix& Get_rhox()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(rhox);
    };
    /// Get the rhoy matrix from the container.
    TRealMatrix& Get_rhoy()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(rhoy);
    };
    /// Get the rhoz matrix from the container.
    TRealMatrix& Get_rhoz()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(rhoz);
    };
    /// Get the rho0 matrix from the container.
    TRealMatrix& Get_rho0()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(rho0);
    };

    /// Get the ddx_k_shift_pos matrix from the container.
    TComplexMatrix& Get_ddx_k_shift_pos()
    {
      return MatrixContainer.GetMatrix<TComplexMatrix>(ddx_k_shift_pos);
    };
    /// Get the ddy_k_shift_pos matrix from the container.
    TComplexMatrix& Get_ddy_k_shift_pos()
    {
      return MatrixContainer.GetMatrix<TComplexMatrix>(ddy_k_shift_pos);
    };
    /// Get the ddz_k_shift_pos matrix from the container.
    TComplexMatrix& Get_ddz_k_shift_pos()
    {
      return MatrixContainer.GetMatrix<TComplexMatrix>(ddz_k_shift_pos);
    };
    /// Get the ddx_k_shift_neg matrix from the container.
    TComplexMatrix& Get_ddx_k_shift_neg()
    {
      return MatrixContainer.GetMatrix<TComplexMatrix>(ddx_k_shift_neg);
    };
    /// Get the ddy_k_shift_neg matrix from the container.
    TComplexMatrix& Get_ddy_k_shift_neg()
    {
      return MatrixContainer.GetMatrix<TComplexMatrix>(ddy_k_shift_neg);
    };
    /// Get the ddz_k_shift_neg matrix from the container.
    TComplexMatrix& Get_ddz_k_shift_neg()
    {
      return MatrixContainer.GetMatrix<TComplexMatrix>(ddz_k_shift_neg);
    };

    /// Get the x_shift_neg_r matrix from the container.
    TComplexMatrix& Get_x_shift_neg_r()
    {
      return MatrixContainer.GetMatrix<TComplexMatrix>(x_shift_neg_r);
    };
    /// Get the y_shift_neg_r from the container.
    TComplexMatrix& Get_y_shift_neg_r()
    {
      return MatrixContainer.GetMatrix<TComplexMatrix>(y_shift_neg_r);
    };
    /// Get the y_shift_neg_r from the container.
    TComplexMatrix& Get_z_shift_neg_r()
    {
      return MatrixContainer.GetMatrix<TComplexMatrix>(z_shift_neg_r);
    };

    /// Get the pml_x_sgx matrix from the container.
    TRealMatrix& Get_pml_x_sgx()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(pml_x_sgx);
    };
    /// Get the pml_y_sgy matrix from the container.
    TRealMatrix& Get_pml_y_sgy()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(pml_y_sgy);
    };
    /// Get the pml_z_sgz matrix from the container.
    TRealMatrix& Get_pml_z_sgz()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(pml_z_sgz);
    };

    /// Get the pml_x matrix from the container.
    TRealMatrix& Get_pml_x()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(pml_x);
    };
    /// Get the pml_y matrix from the container.
    TRealMatrix& Get_pml_y()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(pml_y);
    };
    /// Get the pml_z matrix from the container.
    TRealMatrix& Get_pml_z()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(pml_z);
    };


    /// Get the dxudxn matrix from the container.
    TRealMatrix& Get_dxudxn()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(dxudxn);
    };
    /// Get the dyudyn matrix from the container.
    TRealMatrix& Get_dyudyn()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(dyudyn);
    };
    /// Get the dzudzn matrix from the container.
    TRealMatrix& Get_dzudzn()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(dzudzn);
    };

    /// Get the dxudxn_sgx matrix from the container.
    TRealMatrix& Get_dxudxn_sgx()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(dxudxn_sgx);
    };
    /// Get the dyudyn_sgy matrix from the container.
    TRealMatrix& Get_dyudyn_sgy()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(dyudyn_sgy);
    };
    /// Get the dzudzn_sgz matrix from the container.
    TRealMatrix& Get_dzudzn_sgz()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(dzudzn_sgz);
    };


    /// Get the BonA matrix from the container.
    TRealMatrix& Get_BonA()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(BonA);
    };
    /// Get the absorb_tau matrix from the container.
    TRealMatrix& Get_absorb_tau()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(absorb_tau);
    };
    /// Get the absorb_eta matrix from the container.
    TRealMatrix& Get_absorb_eta()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(absorb_eta);
    };

    /// Get the absorb_nabla1 matrix from the container.
    TRealMatrix& Get_absorb_nabla1()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(absorb_nabla1);
    };
    /// Get the absorb_nabla2 matrix from the container.
    TRealMatrix& Get_absorb_nabla2()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(absorb_nabla2);
    };


    //-- Index matrices --//

    /// Get the sensor_mask_index matrix from the container.
    TIndexMatrix& Get_sensor_mask_index()
    {
      return MatrixContainer.GetMatrix<TIndexMatrix>(sensor_mask_index);
    };


    /// Get the sensor_mask_corners matrix from the container.
    TIndexMatrix& Get_sensor_mask_corners()
    {
      return MatrixContainer.GetMatrix<TIndexMatrix>(sensor_mask_corners);
    };

    /// Get the u_source_index matrix from the container.
    TIndexMatrix& Get_u_source_index()
    {
      return MatrixContainer.GetMatrix<TIndexMatrix>(u_source_index);
    };
    /// Get the p_source_index matrix from the container.
    TIndexMatrix& Get_p_source_index()
    {
      return MatrixContainer.GetMatrix<TIndexMatrix>(p_source_index);
    };
    /// Get the delay_mask matrix from the container.
    TIndexMatrix& Get_delay_mask()
    {
      return MatrixContainer.GetMatrix<TIndexMatrix>(delay_mask);
    }


    //-- sources  --//

    /// Get the transducer_source_input matrix from the container.
    TRealMatrix& Get_transducer_source_input()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(transducer_source_input);
    };

    /// Get the p_source_input matrix from the container.
    TRealMatrix& Get_p_source_input()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(p_source_input);
    };

    /// Get the p0_source_input from the container.
    TRealMatrix& Get_p0_source_input()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(p0_source_input);
    };


    /// Get the ux_source_input matrix from the container.
    TRealMatrix& Get_ux_source_input()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(ux_source_input);
    };
    /// Get the uy_source_input matrix from the container.
    TRealMatrix& Get_uy_source_input()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(uy_source_input);
    };
    /// Get the uz_source_input matrix from the container.
    TRealMatrix& Get_uz_source_input()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(uz_source_input);
    };


    //--Temporary matrices --//

    /// Get the Temp_1_RS3D matrix from the container.
    TRealMatrix& Get_Temp_1_RS3D()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(Temp_1_RS3D);
    };
    /// Get the Temp_2_RS3D matrix from the container.
    TRealMatrix& Get_Temp_2_RS3D()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(Temp_2_RS3D);
    };
    /// Get the Temp_3_RS3D matrix from the container.
    TRealMatrix& Get_Temp_3_RS3D()
    {
      return MatrixContainer.GetMatrix<TRealMatrix>(Temp_3_RS3D);
    };


    /// Get the CUFFT_X_temp from the container.
    TCUFFTComplexMatrix& Get_CUFFT_X_temp()
    {
      return MatrixContainer.GetMatrix<TCUFFTComplexMatrix>(CUFFT_X_temp);
    };
    /// Get the FFT_Y_temp from the container.
    TCUFFTComplexMatrix  & Get_CUFFT_Y_temp()
    {
      return MatrixContainer.GetMatrix<TCUFFTComplexMatrix>(CUFFT_Y_temp);
    };
    /// Get the FFT_Z_temp from the container.
    TCUFFTComplexMatrix  & Get_CUFFT_Z_temp()
    {
      return MatrixContainer.GetMatrix<TCUFFTComplexMatrix>(CUFFT_Z_temp);
    };


private:


    /// Use CUDA GPU functions for big computational blocks - what is this?
    TCUDAImplementations* cuda_implementations;


    /// Matrix container with all the matrix classes
    TMatrixContainer MatrixContainer;

    /// Output stream container
    TOutputStreamContainer OutputStreamContainer;

    /// Percentage of the simulation done
    size_t ActPercent;

    /// Global parameters of the simulation
    TParameters* Parameters;

    /// Total time of the simulation
    TTimeMeasure TotalTime;

    /// Pre-processing time of the simulation
    TTimeMeasure PreProcessingTime;

    /// Data load time of the simulation
    TTimeMeasure DataLoadTime;

    /// Simulation time of the simulation
    TTimeMeasure SimulationTime;

    /// Post-processing time of the simulation
    TTimeMeasure PostProcessingTime;

    /// Iteration time of the simulation
    TTimeMeasure IterationTime;

};// end of  TKSpaceFirstOrder3DSolver
//------------------------------------------------------------------------------

#endif
