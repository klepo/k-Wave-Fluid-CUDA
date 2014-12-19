/**
 * @file        KSpaceFirstOrder3DSolver.cpp
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the main class of the
 *              project responsible for the entire simulation.
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        12 July     2012, 10:27 (created)\n
 *              05 December 2014, 15:07 (revised)
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

// Linux build
#ifdef __linux__
  #include <sys/resource.h>
  #include <cmath>
#endif

// Windows build
#ifdef _WIN64
  #define _USE_MATH_DEFINES
  #include <cmath>
  #include <Windows.h>
  #include <Psapi.h>
  #pragma comment(lib, "Psapi.lib")
#endif

#ifdef _OPENMP
  #include <omp.h>
#endif

#include <iostream>
#include <omp.h>
#include <time.h>
#include <sstream>
#include <limits>
#include <cstdio>

#include <KSpaceSolver/KSpaceFirstOrder3DSolver.h>

#include <Utils/ErrorMessages.h>
#include <CUDA/CUDAImplementations.h>
#include <Containers/MatrixContainer.h>


using namespace std;

//----------------------------------------------------------------------------//
//                               Constants                                    //
//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//
//                             Public methods                                 //
//----------------------------------------------------------------------------//

/*
 * Constructor of the class.
 */
TKSpaceFirstOrder3DSolver::TKSpaceFirstOrder3DSolver() :
        cuda_implementations(NULL), MatrixContainer(), OutputStreamContainer(),
        ActPercent(0), Parameters(NULL),
        TotalTime(), PreProcessingTime(), DataLoadTime(), SimulationTime(),
        PostProcessingTime(), IterationTime()
{
  TotalTime.Start();

  Parameters  = TParameters::GetInstance();

  //Switch off default HDF5 error messages
  H5Eset_auto(H5E_DEFAULT, NULL, NULL);
}// end of TKSpace3DSolver
//------------------------------------------------------------------------------

/*
 * Destructor of the class.
 */
TKSpaceFirstOrder3DSolver::~TKSpaceFirstOrder3DSolver()
{
  // Delete CUDA FFT plans and related data
  TCUFFTComplexMatrix::DestroyAllPlansAndStaticData();

  // Free memory
  FreeMemory();
}// end of ~TKSpace3DSolver
//------------------------------------------------------------------------------


/**
 * Try to estimate how much memory is needed for the simulation and if it seems
 * it won't be possible to run the simulation, print warning
 * @return false if the simulation is likely to fail because of out of memory
 *
 * @todo take a look and remove
 */
bool TKSpaceFirstOrder3DSolver::DoesDeviceHaveEnoughMemory()
{
  size_t free, total, available_device_memory, estimate_of_required_memory;

  cudaMemGetInfo(&free,&total);
  available_device_memory = (free >> 20);

  estimate_of_required_memory = MatrixContainer.GetSpeculatedMemoryFootprintInMegabytes();

  return !(estimate_of_required_memory > available_device_memory);
}// end of DoesDeviceHaveEnoughMemory
//------------------------------------------------------------------------------

/*
 * The method allocates the matrix container and create all matrices and
 * creates all output streams.
 *
 * @todo remove the warning after going over
 */
void TKSpaceFirstOrder3DSolver::AllocateMemory()
{
  // create container, then all matrices
  MatrixContainer.AddMatricesIntoContainer();

  //if the size of the simulation will use more memory than the target
  //device contains, notify the user that the simulation will probably crash.
  if(!DoesDeviceHaveEnoughMemory())
  {
    fprintf(stdout,"\n");
    fprintf(stdout,"Warning!\n");
    fprintf(stdout,"The simulation may be too big for the target device!");
    fprintf(stdout,"\n");
    fprintf(stdout,"If there is a crash (GPUassert or K-Wave panic) ");
    fprintf(stdout,"this is the probable cause.\n");
    fflush(stdout);
  }

  MatrixContainer.CreateAllObjects();

  // add output streams into container
  //@todo Think about moving under LoadInputData routine...
  OutputStreamContainer.AddStreamsIntoContainer(MatrixContainer);

}// end of AllocateMemory
//------------------------------------------------------------------------------

/*
 * The method frees all memory allocated by the class.
 */
void TKSpaceFirstOrder3DSolver::FreeMemory()
{
  MatrixContainer.FreeAllMatrices();
  OutputStreamContainer.FreeAllStreams();
}// end of FreeMemory
//------------------------------------------------------------------------------

/*
 * Load data from the input file provided by the Parameter class and creates
 * the output time series streams.
 */
void TKSpaceFirstOrder3DSolver::LoadInputData()
{

  DataLoadTime.Start();

  // open and load input file
  THDF5_File& HDF5_InputFile      = Parameters->HDF5_InputFile; // file is opened (in Parameters)
  THDF5_File& HDF5_OutputFile     = Parameters->HDF5_OutputFile;
  THDF5_File& HDF5_CheckpointFile = Parameters->HDF5_CheckpointFile;

  // Load data from disk
  MatrixContainer.LoadDataFromInputHDF5File(HDF5_InputFile);

  // close the input file
  HDF5_InputFile.Close();

  // The simulation does not use checkpointing or this is the first turn
  bool RecoverFromPrevState = (Parameters->IsCheckpointEnabled() &&
                               THDF5_File::IsHDF5(Parameters->GetCheckpointFileName().c_str()));

  //-------------------- Read data from the checkpoint file ------------------//
  if (RecoverFromPrevState)
  {
    // Open checkpoint file
    HDF5_CheckpointFile.Open(Parameters->GetCheckpointFileName().c_str());

    // Check the checkpoint file
    CheckCheckpointFile();

    // read the actual value of t_index
    size_t new_t_index;
    HDF5_CheckpointFile.ReadScalarValue(HDF5_CheckpointFile.GetRootGroup(),
                                        t_index_Name,
                                        new_t_index);
    Parameters->Set_t_index(new_t_index);

    // Read necessary matrices from the checkpoint file
    MatrixContainer.LoadDataFromCheckpointHDF5File(HDF5_CheckpointFile);

    HDF5_CheckpointFile.Close();

    //------------- Read data from the output file ---------------------------//
    // Reopen output file for RW access
    HDF5_OutputFile.Open(Parameters->GetOutputFileName().c_str(), H5F_ACC_RDWR);

    //Read file header of the output file
    Parameters->HDF5_FileHeader.ReadHeaderFromOutputFile(HDF5_OutputFile);

    // Restore elapsed time
    RestoreCumulatedElapsedFromOutputFile(HDF5_OutputFile);

    // Reopen streams
    OutputStreamContainer.ReopenStreams();
  }
  else
  {
    //-------------------- First round of multi-leg simulation ---------------//
    // Create the output file
    HDF5_OutputFile.Create(Parameters->GetOutputFileName().c_str());

    // Create the steams, link them with the sampled matrices
    // however DO NOT allocate memory!
    OutputStreamContainer.CreateStreams();
  }

  DataLoadTime.Stop();
}// end of LoadInputData
//------------------------------------------------------------------------------

/*
 * This method computes k-space First Order 3D simulation.
 * It launches calculation on a given dataset going through
 * FFT initialization, pre-processing, main loop and post-processing phases.
 *
 */
void TKSpaceFirstOrder3DSolver::Compute()
{
  PreProcessingTime.Start();

  fprintf(stdout,"FFT plans creation............."); fflush(stdout);

  // initilaise all cuda FFT plans
  InitializeFFTPlans();

  fprintf(stdout,"Done \n");
  fprintf(stdout,"Pre-processing phase..........."); fflush(stdout);

  //initialise the cpu and gpu computation handler
  //@todo  take look a this
  cuda_implementations = CUDAImplementations::GetInstance();

  // Set up kernel sizes (default grid and block sizes)
  cuda_implementations->SetUpExecutionModelWithTuner(Parameters->GetFullDimensionSizes(),
                                                     Parameters->GetReducedDimensionSizes());

  // Set up constant memory - copy over to GPU
  cuda_implementations->SetUpDeviceConstants(
            Parameters->GetFullDimensionSizes().X,
            Parameters->GetFullDimensionSizes().Y,
            Parameters->GetFullDimensionSizes().Z,
            Parameters->GetReducedDimensionSizes().X,
            Parameters->GetReducedDimensionSizes().Y,
            Parameters->GetReducedDimensionSizes().Z);

  PreProcessingPhase();
  PreProcessingTime.Stop();
  fprintf(stdout,"Done \n");


  fprintf(stdout,"Current Host memory in use:   %3ldMB\n", GetHostMemoryUsageInMB());
  fprintf(stdout,"Current Device memory in use: %3ldMB\n", GetDeviceMemoryUsageInMB());
  fprintf(stdout,"Elapsed time:             %8.2fs\n",      PreProcessingTime.GetElapsedTime());


  SimulationTime.Start();
    ComputeMainLoop();
  SimulationTime.Stop();

  //Post processing region
  PostProcessingTime.Start();
  if (IsCheckpointInterruption())
  { // Checkpoint
    fprintf(stdout,"-------------------------------------------------------------\n");
    fprintf(stdout,".............. Interrupted to checkpoint! ...................\n");
    fprintf(stdout,"Number of time steps completed:                    %10ld\n",
            Parameters->Get_t_index());
    fprintf(stdout,"Elapsed time:                                       %8.2fs\n",
            SimulationTime.GetElapsedTime());
    fprintf(stdout,"-------------------------------------------------------------\n");
    fprintf(stdout,"Checkpoint in progress......"); fflush(stdout);

    SaveCheckpointData();
  }
  else
  { // Finish
    fprintf(stdout,"-------------------------------------------------------------\n");
    fprintf(stdout,"Elapsed time:                                       %8.2fs\n",
            SimulationTime.GetElapsedTime());
    fprintf(stdout,"-------------------------------------------------------------\n");
    fprintf(stdout,"Post-processing phase......."); fflush(stdout);

    PostProcessing();

    // if checkpointing is enabled and the checkpoint file was created in the past, delete it
    if (Parameters->IsCheckpointEnabled())
    {
      std::remove(Parameters->GetCheckpointFileName().c_str());
    }
  }

  PostProcessingTime.Stop();

  fprintf(stdout,"Done \n");
  fprintf(stdout,"Elapsed time:          %8.2fs\n", PostProcessingTime.GetElapsedTime());

  WriteOutputDataInfo();
  Parameters->HDF5_OutputFile.Close();
}// end of Compute()
//------------------------------------------------------------------------------

/**
 * Print parameters of the simulation.
 * @param [in,out] file - where to print the parameters
 */
void TKSpaceFirstOrder3DSolver::PrintParametersOfSimulation(FILE * file)
{
  fprintf(file, "Domain dims:     [%4ld, %4ld, %4ld]\n",
                Parameters->GetFullDimensionSizes().X,
                Parameters->GetFullDimensionSizes().Y,
                Parameters->GetFullDimensionSizes().Z);

  fprintf(file,"Simulation time steps:  %11ld\n", Parameters->Get_Nt());
}// end of PrintParametersOfTask
//------------------------------------------------------------------------------

/**
 * Get peak GPU memory usage.
 * @return Peak memory usage in MBs.
 *
 */
size_t TKSpaceFirstOrder3DSolver::GetDeviceMemoryUsageInMB()
{
  size_t free, total;
  cudaMemGetInfo(&free,&total);

  return ((total-free) >> 20);
}// end of GetDeviceMemoryUsageInMB
//------------------------------------------------------------------------------

/**
 * Get peak CPU memory usage.
 * @return Peak memory usage in MBs.
 *
 */
size_t TKSpaceFirstOrder3DSolver::GetHostMemoryUsageInMB()
{
  // Linux build
  #ifdef __linux__
    struct rusage mem_usage;
    getrusage(RUSAGE_SELF, &mem_usage);

    return mem_usage.ru_maxrss >> 10;
  #endif

  // Windows build
  #ifdef _WIN64
    HANDLE hProcess;
    PROCESS_MEMORY_COUNTERS pmc;

    hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
                           FALSE,
                           GetCurrentProcessId());

    GetProcessMemoryInfo(hProcess, &pmc, sizeof(pmc));
    CloseHandle(hProcess);

    return pmc.PeakWorkingSetSize >> 20;
  #endif
}// end of ShowMemoryUsageInMB
//------------------------------------------------------------------------------

/*
 * Print Full code name and the license
 *
 * @param [in] file - file to print the data (stdout)
 */
void TKSpaceFirstOrder3DSolver::PrintFullNameCodeAndLicense(FILE * file)
{
  fprintf(file,"\n");
  fprintf(file,"+----------------------------------------------------+\n");
  fprintf(file,"| Build Number:     kspaceFirstOrder3D v3.3          |\n");
  fprintf(file,"| Build date:       %*.*s                      |\n", 10,11,__DATE__);
  fprintf(file,"| Build time:       %*.*s                         |\n", 8,8,__TIME__);
  #if (defined (__KWAVE_GIT_HASH__))
    fprintf(file,"| Git hash: %s |\n",__KWAVE_GIT_HASH__);
  #endif
  fprintf(file,"|                                                    |\n");

  // OS detection
  #ifdef __linux__
    fprintf(file,"| Operating System: Linux x64                        |\n");
  #elif __APPLE__
    fprintf(file,"| Operating System: Mac OS X x64                   |\n");
  #elif _WIN32
    fprintf(file,"| Operating System: Windows x64                    |\n");
  #endif

  // Compiler detections
  #if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
    fprintf(file,"| Compiler name:    GNU C++ %.19s                    |\n", __VERSION__);
  #endif
  #ifdef __INTEL_COMPILER
    fprintf(file,"| Compiler name:    Intel C++ %d                   |\n", __INTEL_COMPILER);
  #endif
  #ifdef __NVCC__
    fprintf(file,"| Compiler name:    Nvidia CUDA %d                    |\n", __NVCC__);
  #endif

      // instruction set
  #if (defined (__AVX2__))
    fprintf(file,"| Instruction set:  Intel AVX 2                      |\n");
  #elif (defined (__AVX__))
    fprintf(file,"| Instruction set:  Intel AVX                        |\n");
  #elif (defined (__SSE4_2__))
    fprintf(file,"| Instruction set:  Intel SSE 4.2                    |\n");
  #elif (defined (__SSE4_1__))
    fprintf(file,"| Instruction set:  Intel SSE 4.1                    |\n");
  #elif (defined (__SSE3__))
    fprintf(file,"| Instruction set:  Intel SSE 3                      |\n");
  #elif (defined (__SSE2__))
    fprintf(file,"| Instruction set:  Intel SSE 2                      |\n");
  #endif

  fprintf(file,"|                                                    |\n");
  fprintf(file,"| Copyright (C) 2014 Jiri Jaros, Bradley Treeby and  |\n");
  fprintf(file,"|                    Beau Johnston                   |\n");
  fprintf(file,"| http://www.k-wave.org                              |\n");
  fprintf(file,"+----------------------------------------------------+\n");
  fprintf(file,"\n");
}// end of GetFullCodeAndLincence
//------------------------------------------------------------------------------


/**
 * Set processor affinity.
 */
void TKSpaceFirstOrder3DSolver::SetProcessorAffinity()
{
  // Linux Build
  #ifdef __linux__
    //GNU compiler
    #if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
      setenv("OMP_PROC_BIND","TRUE",1);
    #endif

    #ifdef __INTEL_COMPILER
      setenv("KMP_AFFINITY","none",1);
    #endif
  #endif

  // Windows build is always compiled by the Intel Compiler
  #ifdef _WIN64
    _putenv_s("KMP_AFFINITY","none");
  #endif
}//end of SetProcessorAffinity
//------------------------------------------------------------------------------


//---------------------------------------------------------------------------//
//                            Protected methods                              //
//---------------------------------------------------------------------------//

/*
 * Initialize FFT plans.
 */
void TKSpaceFirstOrder3DSolver::InitializeFFTPlans()
{
  // create real to complex plans
  TCUFFTComplexMatrix::Create_FFT_Plan_3D_R2C(Parameters->GetFullDimensionSizes());

 // create complex to real plans
  TCUFFTComplexMatrix::Create_FFT_Plan_3D_C2R(Parameters->GetFullDimensionSizes());

}// end of InitializeFFTPlans
//------------------------------------------------------------------------------

/*
 * Compute pre-processing phase. \n
 * Initialize all indices, pre-compute constants such as c^2, rho0_sg* x dt
 * and create kappa, absorb_eta, absorb_tau, absorb_nabla1, absorb_nabla2
 * matrices. Calculate this on the CPU side.
 */
void TKSpaceFirstOrder3DSolver::PreProcessingPhase()
{
  // get the correct sensor mask and recompute indices
  if (Parameters->Get_sensor_mask_type() == TParameters::smt_index)
  {
    Get_sensor_mask_index().RecomputeIndicesToCPP();
  }

  /* not yet implemented
  if (Parameters->Get_sensor_mask_type() == TParameters::smt_corners)
  {
    Get_sensor_mask_corners().RecomputeIndicesToCPP();
  }*/

  if ((Parameters->Get_transducer_source_flag() != 0) ||
      (Parameters->Get_ux_source_flag() != 0)         ||
      (Parameters->Get_uy_source_flag() != 0)         ||
      (Parameters->Get_uz_source_flag() != 0)
     )
  {
    Get_u_source_index().RecomputeIndicesToCPP();
  }

  if (Parameters->Get_transducer_source_flag() != 0)
  {
    Get_delay_mask().RecomputeIndicesToCPP();
  }

  if (Parameters->Get_p_source_flag() != 0)
  {
    Get_p_source_index().RecomputeIndicesToCPP();
  }

  // compute dt / rho0_sg...
  if (Parameters->Get_rho0_scalar_flag())
  { // rho is scalar
    Parameters->Get_rho0_sgx_scalar() = Parameters->Get_dt() / Parameters->Get_rho0_sgx_scalar();
    Parameters->Get_rho0_sgy_scalar() = Parameters->Get_dt() / Parameters->Get_rho0_sgy_scalar();
    Parameters->Get_rho0_sgz_scalar() = Parameters->Get_dt() / Parameters->Get_rho0_sgz_scalar();
  }
  else
  { // non-uniform grid cannot be pre-calculated :-(
    // rho is matrix
    if (Parameters->Get_nonuniform_grid_flag())
    {
      Calculate_dt_rho0_non_uniform();
    }
    else
    {
      Get_dt_rho0_sgx().ScalarDividedBy(Parameters->Get_dt());
      Get_dt_rho0_sgy().ScalarDividedBy(Parameters->Get_dt());
      Get_dt_rho0_sgz().ScalarDividedBy(Parameters->Get_dt());
    }
  }

  // generate different matrices
  if (Parameters->Get_absorbing_flag() != 0)
  {
    Generate_kappa_absorb_nabla1_absorb_nabla2();
    Generate_absorb_tau_absorb_eta_matrix();
  }
  else
  {
    Generate_kappa();
  }

  // calculate c^2. It has to be after kappa gen... because of c modification
  Compute_c2();

}// end of PreProcessingPhase
//------------------------------------------------------------------------------


/**
 * Generate kappa matrix for non-absorbing mode.
 *
 */
void TKSpaceFirstOrder3DSolver::Generate_kappa()
{
  #pragma omp parallel
  {
    const float dx_sq_rec = 1.0f / (Parameters->Get_dx() * Parameters->Get_dx());
    const float dy_sq_rec = 1.0f / (Parameters->Get_dy() * Parameters->Get_dy());
    const float dz_sq_rec = 1.0f / (Parameters->Get_dz() * Parameters->Get_dz());

    const float c_ref_dt_pi = Parameters->Get_c_ref() * Parameters->Get_dt() * float(M_PI);

    const float Nx_rec = 1.0f / static_cast<float>(Parameters->GetFullDimensionSizes().X);
    const float Ny_rec = 1.0f / static_cast<float>(Parameters->GetFullDimensionSizes().Y);
    const float Nz_rec = 1.0f / static_cast<float>(Parameters->GetFullDimensionSizes().Z);

    const size_t X_Size  = Parameters->GetReducedDimensionSizes().X;
    const size_t Y_Size  = Parameters->GetReducedDimensionSizes().Y;
    const size_t Z_Size  = Parameters->GetReducedDimensionSizes().Z;

    float * kappa = Get_kappa().GetRawData();

    #pragma omp for schedule (static)
    for (size_t z = 0; z < Z_Size; z++)
    {
      const float z_f    = (float) z;
            float z_part = 0.5f - fabs(0.5f - z_f * Nz_rec );
                  z_part = (z_part * z_part) * dz_sq_rec;

      for (size_t y = 0; y < Y_Size; y++)
      {
        const float y_f    = (float) y;
              float y_part = 0.5f - fabs(0.5f - y_f * Ny_rec);
                    y_part = (y_part * y_part) * dy_sq_rec;

        const float yz_part = z_part + y_part;
        for (size_t x = 0; x < X_Size; x++)
        {
          const float x_f = (float) x;
                float x_part = 0.5f - fabs(0.5f - x_f * Nx_rec);
                      x_part = (x_part * x_part) * dx_sq_rec;

                float k = c_ref_dt_pi * sqrt(x_part + yz_part);

          // kappa element
          kappa[(z*Y_Size + y) * X_Size + x ] = (k == 0.0f) ? 1.0f : sin(k)/k;
        }//x
      }//y
    }// z
  }// parallel
}// end of GenerateKappa
//------------------------------------------------------------------------------

/*
 * Generate kappa, absorb_nabla1, absorb_nabla2 for absorbing media.
 */
void TKSpaceFirstOrder3DSolver::Generate_kappa_absorb_nabla1_absorb_nabla2()
{
  #pragma omp parallel
  {
    const float dx_sq_rec = 1.0f / (Parameters->Get_dx() * Parameters->Get_dx());
    const float dy_sq_rec = 1.0f / (Parameters->Get_dy() * Parameters->Get_dy());
    const float dz_sq_rec = 1.0f / (Parameters->Get_dz() * Parameters->Get_dz());

    const float c_ref_dt_2 = Parameters->Get_c_ref() * Parameters->Get_dt() * 0.5f;
    const float pi_2       = float(M_PI) * 2.0f;

    const size_t Nx = Parameters->GetFullDimensionSizes().X;
    const size_t Ny = Parameters->GetFullDimensionSizes().Y;
    const size_t Nz = Parameters->GetFullDimensionSizes().Z;

    const float Nx_rec   = 1.0f / (float) Nx;
    const float Ny_rec   = 1.0f / (float) Ny;
    const float Nz_rec   = 1.0f / (float) Nz;

    const size_t X_Size  = Parameters->GetReducedDimensionSizes().X;
    const size_t Y_Size  = Parameters->GetReducedDimensionSizes().Y;
    const size_t Z_Size  = Parameters->GetReducedDimensionSizes().Z;

    float * kappa           = Get_kappa().GetRawData();
    float * absorb_nabla1   = Get_absorb_nabla1().GetRawData();
    float * absorb_nabla2   = Get_absorb_nabla2().GetRawData();
    const float alpha_power = Parameters->Get_alpha_power();

    #pragma omp for schedule (static)
    for (size_t z = 0; z < Z_Size; z++)
    {
      const float z_f    = (float) z;
            float z_part = 0.5f - fabs(0.5f - z_f * Nz_rec );
                  z_part = (z_part * z_part) * dz_sq_rec;

      for (size_t y = 0; y < Y_Size; y++)
      {
        const float y_f    = (float) y;
              float y_part = 0.5f - fabs(0.5f - y_f * Ny_rec);
                    y_part = (y_part * y_part) * dy_sq_rec;

        const float yz_part = z_part + y_part;

        size_t i = (z * Y_Size + y) * X_Size;

        for (size_t x = 0; x < X_Size; x++)
        {
          const float x_f    = (float) x;
                float x_part = 0.5f - fabs(0.5f - x_f * Nx_rec);
                      x_part = (x_part * x_part) * dx_sq_rec;

                float  k         = pi_2 * sqrt(x_part + yz_part);
                float  c_ref_k   = c_ref_dt_2 * k;

          absorb_nabla1[i] = pow(k, alpha_power - 2);
          absorb_nabla2[i] = pow(k, alpha_power - 1);

          kappa[i]         =  (c_ref_k == 0.0f) ? 1.0f : sin(c_ref_k)/c_ref_k;

          if (absorb_nabla1[i] == std::numeric_limits<float>::infinity()) absorb_nabla1[i] = 0.0f;
          if (absorb_nabla2[i] == std::numeric_limits<float>::infinity()) absorb_nabla2[i] = 0.0f;

          i++;
        }//x
      }//y
    }// z
  }// parallel
}// end of Generate_kappa_absorb_nabla1_absorb_nabla2
//------------------------------------------------------------------------------

/**
 * Generate absorb_tau and absorb_eta in for heterogenous media.
 */
void TKSpaceFirstOrder3DSolver::Generate_absorb_tau_absorb_eta_matrix()
{
  // test for scalars
  if ((Parameters->Get_alpha_coeff_scalar_flag()) && (Parameters->Get_c0_scalar_flag()))
  {
    const float alpha_power = Parameters->Get_alpha_power();
    const float tan_pi_y_2  = tan(static_cast<float> (M_PI_2) * alpha_power);
    const float alpha_db_neper_coeff = (100.0f * pow(1.0e-6f /(2.0f * static_cast<float>(M_PI)), alpha_power)) /
                                       (20.0f * static_cast<float>(M_LOG10E));

    const float alpha_coeff_2 = 2.0f * Parameters->Get_alpha_coeff_scalar() * alpha_db_neper_coeff;

    Parameters->Get_absorb_tau_scalar() = (-alpha_coeff_2) * pow(Parameters->Get_c0_scalar(), alpha_power - 1);
    Parameters->Get_absorb_eta_scalar() =   alpha_coeff_2  * pow(Parameters->Get_c0_scalar(), alpha_power) * tan_pi_y_2;
  }
  else
  { // matrix
    #pragma omp parallel
    {
      const size_t Z_Size  = Parameters->GetFullDimensionSizes().Z;
      const size_t Y_Size  = Parameters->GetFullDimensionSizes().Y;
      const size_t X_Size  = Parameters->GetFullDimensionSizes().X;

      float * absorb_tau = Get_absorb_tau().GetRawData();
      float * absorb_eta = Get_absorb_eta().GetRawData();

      float * alpha_coeff;
      size_t  alpha_shift;

      if (Parameters->Get_alpha_coeff_scalar_flag())
      {
        alpha_coeff = &(Parameters->Get_alpha_coeff_scalar());
        alpha_shift = 0;
      }
      else
      {
        alpha_coeff = Get_Temp_1_RS3D().GetRawData();
        alpha_shift = 1;
      }

      float * c0;
      size_t c0_shift;
      if (Parameters->Get_c0_scalar_flag())
      {
        c0 = &(Parameters->Get_c0_scalar());
        c0_shift = 0;
      }
      else
      {
        c0 = Get_c2().GetRawData();
        c0_shift = 1;
      }

      const float alpha_power = Parameters->Get_alpha_power();
      const float tan_pi_y_2  = tan(static_cast<float>(M_PI_2) * alpha_power);

      //alpha = 100*alpha.*(1e-6/(2*pi)).^y./
      //                  (20*log10(exp(1)));
      const float alpha_db_neper_coeff = (100.0f * pow(1.0e-6f / (2.0f * static_cast<float>(M_PI)), alpha_power)) /
                                         (20.0f * static_cast<float>(M_LOG10E));


      #pragma omp for schedule (static)
      for (size_t z = 0; z < Z_Size; z++)
      {
        for (size_t y = 0; y < Y_Size; y++)
        {
          size_t i = (z * Y_Size + y) * X_Size;
          for (size_t x = 0; x < X_Size; x++)
          {
            const float alpha_coeff_2 = 2.0f * alpha_coeff[i*alpha_shift] * alpha_db_neper_coeff;

            absorb_tau[i] = (-alpha_coeff_2) * pow(c0[i * c0_shift],alpha_power - 1);
            absorb_eta[i] =   alpha_coeff_2  * pow(c0[i * c0_shift],alpha_power) * tan_pi_y_2;

            i++;
          }//x
        }//y
      }// z
    }// parallel
  } // absorb_tau and aborb_eta = matrics
}// end of Generate_absorb_tau_absorb_eta_matrix
//---------------------------------------------------------------------------//

/**
 * Prepare dt./ rho0  for non-uniform grid.
 *
 */
void TKSpaceFirstOrder3DSolver::Calculate_dt_rho0_non_uniform()
{
  #pragma omp parallel
  {
    float * dt_rho0_sgx   = Get_dt_rho0_sgx().GetRawData();
    float * dt_rho0_sgy   = Get_dt_rho0_sgy().GetRawData();
    float * dt_rho0_sgz   = Get_dt_rho0_sgz().GetRawData();

    const float dt = Parameters->Get_dt();

    const float * duxdxn_sgx = Get_dxudxn_sgx().GetRawData();
    const float * duydyn_sgy = Get_dyudyn_sgy().GetRawData();
    const float * duzdzn_sgz = Get_dzudzn_sgz().GetRawData();

    const size_t Z_Size = Get_dt_rho0_sgx().GetDimensionSizes().Z;
    const size_t Y_Size = Get_dt_rho0_sgx().GetDimensionSizes().Y;
    const size_t X_Size = Get_dt_rho0_sgx().GetDimensionSizes().X;

    const size_t SliceSize = (X_Size * Y_Size );

    #pragma omp for schedule (static)
    for (size_t z = 0; z < Z_Size; z++)
    {
      register size_t i = z* SliceSize;
      for (size_t y = 0; y < Y_Size; y++)
      {
        for (size_t x = 0; x < X_Size; x++)
        {
          dt_rho0_sgx[i] = (dt * duxdxn_sgx[x]) / dt_rho0_sgx[i];
          i++;
        } // x
      } // y
    } // z

    #pragma omp for schedule (static)
    for (size_t z = 0; z < Z_Size; z++)
    {
      register size_t i = z* SliceSize;
      for (size_t y = 0; y < Y_Size; y++)
      {
        const float duydyn_el = duydyn_sgy[y];
        for (size_t x = 0; x < X_Size; x++)
        {
          dt_rho0_sgy[i] = (dt * duydyn_el) / dt_rho0_sgy[i];
          i++;
        } // x
      } // y
    } // z


    #pragma omp for schedule (static)
    for (size_t z = 0; z < Z_Size; z++)
    {
      register size_t i = z* SliceSize;
      const float duzdzn_el = duzdzn_sgz[z];
      for (size_t y = 0; y < Y_Size; y++)
      {
        for (size_t x = 0; x < X_Size; x++)
        {
          dt_rho0_sgz[i] = (dt * duzdzn_el) / dt_rho0_sgz[i];
          i++;
        } // x
      } // y
    } // z
  } // parallel
}// end of Calculate_dt_rho0_non_uniform
//---------------------------------------------------------------------------//

/**
 * Calculate p0 source when necessary.
 * @todo revise this method
 */
void TKSpaceFirstOrder3DSolver::Calculate_p0_source()
{
  // Why on the CPU side?
  Get_p().CopyData(Get_p0_source_input());

  //if it's for the gpu build make first copy
  // @todo - Why to do this, p0 should be on device???
  Get_p().CopyToDevice();


  const float * p0 = Get_p0_source_input().GetRawDeviceData();

  // get over the scalar problem
  float * c2;
  size_t c2_shift;

  if (Parameters->Get_c0_scalar_flag())
  {
    c2 = &Parameters->Get_c0_scalar();
    c2_shift = 0;
  }
  else
  {
    c2 = Get_c2().GetRawDeviceData();
    c2_shift = 1;
  }

  //-- add the initial pressure to rho as a mass source --//
  cuda_implementations->Calculate_p0_source_add_initial_pressure(Get_rhox(),
                                                                 Get_rhoy(),
                                                                 Get_rhoz(),
                                                                 Get_p0_source_input(),
                                                                 c2_shift,
                                                                 c2);

  //-----------------------------------------------------------------------//
  //--compute u(t = t1 + dt/2) based on the assumption u(dt/2) = -u(-dt/2)-//
  //--    which forces u(t = t1) = 0                                      -//
  //-----------------------------------------------------------------------//
  cuda_implementations->Compute_ddx_kappa_fft_p(Get_p(),
                                                Get_CUFFT_X_temp(),
                                                Get_CUFFT_Y_temp(),
                                                Get_CUFFT_Z_temp(),
                                                Get_kappa(),
                                                Get_ddx_k_shift_pos(),
                                                Get_ddy_k_shift_pos(),
                                                Get_ddz_k_shift_pos());

  if (Parameters->Get_rho0_scalar_flag())
  {
    if (Parameters->Get_nonuniform_grid_flag())
    { // non uniform grid
      cuda_implementations->Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_x(
                        Get_ux_sgx(),
                        Parameters->Get_rho0_sgx_scalar(),
                        Get_dxudxn_sgx(),
                        Get_CUFFT_X_temp());
      cuda_implementations->Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_y(
                        Get_uy_sgy(),
                        Parameters->Get_rho0_sgy_scalar(),
                        Get_dyudyn_sgy(),
                        Get_CUFFT_Y_temp());
      cuda_implementations->Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_z(
                        Get_uz_sgz(),
                        Parameters->Get_rho0_sgz_scalar(),
                        Get_dzudzn_sgz(),
                        Get_CUFFT_Z_temp());
    }
    else
    { //uniform grid, heterogeneous
      cuda_implementations->Compute_dt_rho_sg_mul_ifft_div_2(Get_ux_sgx(),
                                                             Parameters->Get_rho0_sgx_scalar(),
                                                             Get_CUFFT_X_temp());
      cuda_implementations->Compute_dt_rho_sg_mul_ifft_div_2(Get_uy_sgy(),
                                                             Parameters->Get_rho0_sgy_scalar(),
                                                             Get_CUFFT_Y_temp());
      cuda_implementations->Compute_dt_rho_sg_mul_ifft_div_2(Get_uz_sgz(),
                                                             Parameters->Get_rho0_sgz_scalar(),
                                                             Get_CUFFT_Z_temp());
    }
  }
  else
  {
    // homogeneous, non-unifrom grid
    // divide the matrix by 2 and multiply with st./rho0_sg
  cuda_implementations->Compute_dt_rho_sg_mul_ifft_div_2(Get_ux_sgx(),
                                                         Get_dt_rho0_sgx(),
                                                         Get_CUFFT_X_temp());
  cuda_implementations->Compute_dt_rho_sg_mul_ifft_div_2(Get_uy_sgy(),
                                                         Get_dt_rho0_sgy(),
                                                         Get_CUFFT_Y_temp());
  cuda_implementations->Compute_dt_rho_sg_mul_ifft_div_2(Get_uz_sgz(),
                                                         Get_dt_rho0_sgz(),
                                                         Get_CUFFT_Z_temp());
  }
}// end of Calculate_p0_source
//------------------------------------------------------------------------------


/**
 * Compute c^2 on the CPU side.
 *
 */
void TKSpaceFirstOrder3DSolver::Compute_c2()
{
  if (Parameters->Get_c0_scalar_flag())
  { // scalar
      float c = Parameters->Get_c0_scalar();
      Parameters->Get_c0_scalar() = c * c;
  }
  else
  { // matrix
    float * c2 =  Get_c2().GetRawData();

    #pragma omp parallel for schedule (static)
    for (size_t i=0; i < Get_c2().GetTotalElementCount(); i++)
    {
      c2[i] = c2[i] * c2[i];
    }
  }// matrix
}// ComputeC2
//------------------------------------------------------------------------------

/**
 * Compute new values for duxdx, duydy, duzdz.
 *
 */
void TKSpaceFirstOrder3DSolver::Compute_duxyz()
{
  Get_CUFFT_X_temp().Compute_FFT_3D_R2C(Get_ux_sgx());
  Get_CUFFT_Y_temp().Compute_FFT_3D_R2C(Get_uy_sgy());
  Get_CUFFT_Z_temp().Compute_FFT_3D_R2C(Get_uz_sgz());

  //@todo what is this???
  cuda_implementations->Compute_duxyz_initial(Get_CUFFT_X_temp(),
                                              Get_CUFFT_Y_temp(),
                                              Get_CUFFT_Z_temp(),
                                              Get_kappa(),
                                              Get_ux_sgx(),
                                              Get_ddx_k_shift_neg(),
                                              Get_ddy_k_shift_neg(),
                                              Get_ddz_k_shift_neg());

  Get_CUFFT_X_temp().Compute_FFT_3D_C2R(Get_duxdx());
  Get_CUFFT_Y_temp().Compute_FFT_3D_C2R(Get_duydy());
  Get_CUFFT_Z_temp().Compute_FFT_3D_C2R(Get_duzdz());

  //-----------------------------------------------------------------------//
  //--------------------- Non linear grid ---------------------------------//
  //-----------------------------------------------------------------------//
  if (Parameters->Get_nonuniform_grid_flag() != 0)
  {
    cuda_implementations->Compute_duxyz_non_linear(Get_duxdx(),
                                                   Get_duydy(),
                                                   Get_duzdz(),
                                                   Get_dxudxn(),
                                                   Get_dyudyn(),
                                                   Get_dzudzn());
  }// nonlinear
}// end of Compute_duxyz
//------------------------------------------------------------------------------

/*
 * Calculate new values of rhox, rhoy and rhoz for non-linear case.
 */
void TKSpaceFirstOrder3DSolver::Compute_rhoxyz_nonlinear()
{
  // Scalar
  if (Parameters->Get_rho0_scalar())
  {
    cuda_implementations->Compute_rhoxyz_nonlinear_scalar(Get_rhox(),
                                                         Get_rhoy(),
                                                         Get_rhoz(),
                                                         Get_pml_x(),
                                                         Get_pml_y(),
                                                         Get_pml_z(),
                                                         Get_duxdx(),
                                                         Get_duydy(),
                                                         Get_duzdz(),
                                                         Parameters->Get_dt(),
                                                         Parameters->Get_rho0_scalar());
  }
  else
  {
    // rho0 is a matrix
     cuda_implementations->Compute_rhoxyz_nonlinear_matrix(Get_rhox(),
                                                          Get_rhoy(),
                                                          Get_rhoz(),
                                                          Get_pml_x(),
                                                          Get_pml_y(),
                                                          Get_pml_z(),
                                                          Get_duxdx(),
                                                          Get_duydy(),
                                                          Get_duzdz(),
                                                          Parameters->Get_dt(),
                                                          Get_rho0());
  } // end matrix
}// end of Compute_rhoxyz
//------------------------------------------------------------------------------

/**
 * Calculate new values of rhox, rhoy and rhoz for linear case.
 *
 */
void TKSpaceFirstOrder3DSolver::Compute_rhoxyz_linear()
{
  // Scalar
  if (Parameters->Get_rho0_scalar())
  {
    cuda_implementations->Compute_rhoxyz_linear_scalar(Get_rhox(),
                                                       Get_rhoy(),
                                                       Get_rhoz(),
                                                       Get_pml_x(),
                                                       Get_pml_y(),
                                                       Get_pml_z(),
                                                       Get_duxdx(),
                                                       Get_duydy(),
                                                       Get_duzdz(),
                                                       Parameters->Get_dt(),
                                                       Parameters->Get_rho0_scalar());
  }
  else
  {
    // rho0 is a matrix
    cuda_implementations->Compute_rhoxyz_linear_matrix(Get_rhox(),
                                                       Get_rhoy(),
                                                       Get_rhoz(),
                                                       Get_pml_x(),
                                                       Get_pml_y(),
                                                       Get_pml_z(),
                                                       Get_duxdx(),
                                                       Get_duydy(),
                                                       Get_duzdz(),
                                                       Parameters->Get_dt(),
                                                       Get_rho0());
  } // end matrix
}// end of Compute_rhoxyz
//------------------------------------------------------------------------------

/*
 * Calculate two temporary sums in the new pressure formula,
 * linear absorbing case.
 * @param [out] Sum_rhoxyz    -rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [out] Sum_rho0_du   - rho0* (duxdx + duydy + duzdz);
 */
void TKSpaceFirstOrder3DSolver::Calculate_SumRho_SumRhoDu(TRealMatrix& Sum_rhoxyz,
                                                          TRealMatrix& Sum_rho0_du)
{
  const size_t TotalElementCount = Parameters->GetFullDimensionSizes().GetElementCount();

  const float * rho0_data = NULL;

  const float rho0_data_el = Parameters->Get_rho0_scalar();
  if (!Parameters->Get_rho0_scalar_flag())
  {
    rho0_data = Get_rho0().GetRawDeviceData();
  }

  cuda_implementations->Calculate_SumRho_SumRhoDu(Sum_rhoxyz,
                                                  Sum_rho0_du,
                                                  Get_rhox(),
                                                  Get_rhoy(),
                                                  Get_rhoz(),
                                                  Get_duxdx(),
                                                  Get_duydy(),
                                                  Get_duzdz(),
                                                  rho0_data,
                                                  rho0_data_el,
                                                  TotalElementCount,
                                                  Parameters->Get_rho0_scalar_flag());
}// end of Calculate_SumRho_SumRhoDu
//------------------------------------------------------------------------------

/*
 * Sum sub-terms to calculate new pressure, non-linear case.
 * @param [in] Absorb_tau_temp  -
 * @param [in] Absorb_eta_temp  -   BonA + rho ^2 / 2 rho0  +
 *                                      (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [in] BonA_temp        -   rho0* (duxdx + duydy + duzdz)
 */
void TKSpaceFirstOrder3DSolver::Sum_Subterms_nonlinear(TRealMatrix& Absorb_tau_temp,
                                                       TRealMatrix& Absorb_eta_temp,
                                                       TRealMatrix& BonA_temp)
{
  float  tau_data_scalar;
  float  eta_data_scalar;
  float  c2_data_scalar;

  float* tau_data_matrix;
  float* eta_data_matrix;
  float* c2_data_matrix;

  size_t  c2_shift;
  size_t  tau_eta_shift;

  const float*  Absorb_tau_data = Absorb_tau_temp.GetRawDeviceData();
  const float*  Absorb_eta_data = Absorb_eta_temp.GetRawDeviceData();

  // c2 scalar
  if (Parameters->Get_c0_scalar_flag())
  {
    c2_data_scalar = Parameters->Get_c0_scalar();
    c2_shift = 0;
  }
  else
  {
    c2_data_matrix = Get_c2().GetRawDeviceData();
    c2_shift = 1;
  }

  // eta and tau scalars
  if (Parameters->Get_c0_scalar_flag() &&
      Parameters->Get_alpha_coeff_scalar_flag())
  {
    tau_data_scalar = Parameters->Get_absorb_tau_scalar();
    eta_data_scalar = Parameters->Get_absorb_eta_scalar();
    tau_eta_shift = 0;
  }
  else
  {
    tau_data_matrix = Get_absorb_tau().GetRawDeviceData();
    eta_data_matrix = Get_absorb_eta().GetRawDeviceData();
    tau_eta_shift = 1;
  }

  cuda_implementations->Sum_Subterms_nonlinear(BonA_temp,
                                               Get_p(),
                                               c2_data_scalar,
                                               c2_data_matrix,
                                               c2_shift,
                                               Absorb_tau_data,
                                               tau_data_scalar,
                                               tau_data_matrix,
                                               Absorb_eta_data,
                                               eta_data_scalar,
                                               eta_data_matrix,
                                               tau_eta_shift);

}// end of Sum_Subterms_nonlinear
//------------------------------------------------------------------------------

/*
 * Sum sub-terms to calculate new pressure, linear case.
 * @param [in] Absorb_tau_temp - sub-term with absorb_tau
 * @param [in] Absorb_eta_temp - sub-term with absorb_eta
 * @param [in] Sum_rhoxyz      - rhox_sgx + rhoy_sgy + rhoz_sgz
 */
void TKSpaceFirstOrder3DSolver::Sum_Subterms_linear(TRealMatrix& Absorb_tau_temp,
                                                    TRealMatrix& Absorb_eta_temp,
                                                    TRealMatrix& Sum_rhoxyz)
{

  float tau_data_scalar;
  float eta_data_scalar;
  float c2_data_scalar;
  float* tau_data_matrix;
  float* eta_data_matrix;
  float* c2_data_matrix;

  size_t c2_shift = 0;
  size_t tau_eta_shift = 0;

  const size_t total_element_count = Parameters->GetFullDimensionSizes().GetElementCount();

  // c2 scalar
  if (Parameters->Get_c0_scalar_flag())
  {
    c2_data_scalar = Parameters->Get_c0_scalar();
    c2_shift = 0;
  }
  else
  {
    c2_data_matrix = Get_c2().GetRawDeviceData();
    c2_shift = 1;
  }

    // eta and tau scalars
  if (Parameters->Get_c0_scalar_flag() && Parameters->Get_alpha_coeff_scalar_flag())
  {
    tau_data_scalar = Parameters->Get_absorb_tau_scalar();
    eta_data_scalar = Parameters->Get_absorb_eta_scalar();
    tau_eta_shift = 0;
  }
  else
  {
    tau_data_matrix = Get_absorb_tau().GetRawDeviceData();
    eta_data_matrix = Get_absorb_eta().GetRawDeviceData();
    tau_eta_shift = 1;
  }

  cuda_implementations->Sum_Subterms_linear(Absorb_tau_temp,
                                            Absorb_eta_temp,
                                            Sum_rhoxyz,
                                            Get_p(),
                                            total_element_count,
                                            c2_shift,
                                            tau_eta_shift,
                                            tau_data_scalar,
                                            tau_data_matrix,
                                            eta_data_scalar,
                                            eta_data_matrix,
                                            c2_data_scalar,
                                            c2_data_matrix);

}// end of Sum_Subterms_linear
//------------------------------------------------------------------------------

/*
 * Sum sub-terms for new p, non-linear lossless case.
 */
void TKSpaceFirstOrder3DSolver::Sum_new_p_nonlinear_lossless()
{
  const size_t TotalElementCount = Parameters->GetFullDimensionSizes().GetElementCount();

  const float * rhox_data = Get_rhox().GetRawDeviceData();
  const float * rhoy_data = Get_rhoy().GetRawDeviceData();
  const float * rhoz_data = Get_rhoz().GetRawDeviceData();

  float* c2_data_matrix;
  float c2_data_scalar;
  size_t c2_shift;

  if (Parameters->Get_c0_scalar_flag())
  {
    c2_data_scalar = Parameters->Get_c0_scalar();
    c2_shift = 0;
  }
  else
  {
    c2_data_matrix = Get_c2().GetRawDeviceData();
    c2_shift = 1;
  }

  float* BonA_data_matrix;
  float BonA_data_scalar;
  size_t BonA_shift;

  if (Parameters->Get_BonA_scalar_flag())
  {
    BonA_data_scalar = Parameters->Get_BonA_scalar();
    BonA_shift = 0;
  }
  else
  {
    BonA_data_matrix = Get_BonA().GetRawDeviceData();
    BonA_shift = 1;
  }

  float* rho0_data_matrix;
  float rho0_data_scalar;
  size_t rho0_shift;

  if (Parameters->Get_rho0_scalar_flag())
  {
    rho0_data_scalar = Parameters->Get_rho0_scalar();
    rho0_shift = 0;
  }
  else
  {
    rho0_data_matrix = Get_rho0().GetRawDeviceData();
    rho0_shift = 1;
  }

  cuda_implementations->Sum_new_p_nonlinear_lossless(TotalElementCount,
                                                     Get_p(),
                                                     rhox_data,
                                                     rhoy_data,
                                                     rhoz_data,
                                                     c2_data_scalar,
                                                     c2_data_matrix,
                                                     c2_shift,
                                                     BonA_data_scalar,
                                                     BonA_data_matrix,
                                                     BonA_shift,
                                                     rho0_data_scalar,
                                                     rho0_data_matrix,
                                                     rho0_shift);

}// end of Sum_new_p_nonlinear_lossless
//------------------------------------------------------------------------------


/*
 * Sum sub-terms for new p, linear lossless case.
 */
void TKSpaceFirstOrder3DSolver::Sum_new_p_linear_lossless()
{
  const size_t total_element_count = Parameters->GetFullDimensionSizes().GetElementCount();

  if (Parameters->Get_c0_scalar_flag())
  {
    const float c2_element = Parameters->Get_c0_scalar();
    cuda_implementations->Sum_new_p_linear_lossless_scalar(Get_p(),
                                                           Get_rhox(),
                                                           Get_rhoy(),
                                                           Get_rhoz(),
                                                           total_element_count,
                                                           c2_element);
  }
  else
  {
    cuda_implementations->Sum_new_p_linear_lossless_matrix(Get_p(),
                                                           Get_rhox(),
                                                           Get_rhoy(),
                                                           Get_rhoz(),
                                                           total_element_count,
                                                           Get_c2());
  }

}// end of Sum_new_p_linear_lossless
//------------------------------------------------------------------------------


/*
 * Calculate three temporary sums in the new pressure formula
 * non-linear absorbing case.
 * @param [out] RHO_Temp  - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [out] BonA_Temp - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [out] Sum_du    - rho0* (duxdx + duydy + duzdz)
 */
void TKSpaceFirstOrder3DSolver::Calculate_SumRho_BonA_SumDu(TRealMatrix& RHO_Temp,
                                                            TRealMatrix& BonA_Temp,
                                                            TRealMatrix& Sum_du)
{
  float  BonA_data_scalar;
  float* BonA_data_matrix;
  size_t BonA_shift;
  const bool BonA_flag = Parameters->Get_BonA_scalar_flag();

  if (BonA_flag)
  {
    BonA_data_scalar = Parameters->Get_BonA_scalar();
    BonA_shift = 0;
  }
  else
  {
    BonA_data_matrix = Get_BonA().GetRawDeviceData();
    BonA_shift = 1;
  }

  float  rho0_data_scalar;
  float* rho0_data_matrix;
  size_t rho0_shift;
  const bool rho0_flag = Parameters->Get_rho0_scalar_flag();

  if (rho0_flag)
  {
    rho0_data_scalar = Parameters->Get_rho0_scalar();
    rho0_shift = 0;
  }
  else
  {
    rho0_data_matrix = Get_rho0().GetRawDeviceData();
    rho0_shift = 1;
  }

  cuda_implementations->Calculate_SumRho_BonA_SumDu(RHO_Temp,
                                                    BonA_Temp,
                                                    Sum_du,
                                                    Get_rhox(),
                                                    Get_rhoy(),
                                                    Get_rhoz(),
                                                    Get_duxdx(),
                                                    Get_duydy(),
                                                    Get_duzdz(),
                                                    BonA_data_scalar,
                                                    BonA_data_matrix,
                                                    BonA_shift,
                                                    rho0_data_scalar,
                                                    rho0_data_matrix,
                                                    rho0_shift);

}// end of Calculate_SumRho_BonA_SumDu
//------------------------------------------------------------------------------

/**
 * Compute new p for non-linear case.
 */
void TKSpaceFirstOrder3DSolver::Compute_new_p_nonlinear()
{
  if (Parameters->Get_absorbing_flag())
  { // absorbing case
    TRealMatrix& Sum_rhoxyz      = Get_Temp_1_RS3D();
    TRealMatrix& BonA_rho_rhoxyz = Get_Temp_2_RS3D();
    TRealMatrix& Sum_du          = Get_Temp_3_RS3D();

    TRealMatrix& Absorb_tau_temp = Sum_du;
    TRealMatrix& Absorb_eta_temp = Sum_rhoxyz;

    Calculate_SumRho_BonA_SumDu(Sum_rhoxyz,BonA_rho_rhoxyz, Sum_du);

    Get_CUFFT_X_temp().Compute_FFT_3D_R2C(Sum_du);
    Get_CUFFT_Y_temp().Compute_FFT_3D_R2C(Sum_rhoxyz);


    cuda_implementations->Compute_Absorb_nabla1_2(Get_absorb_nabla1(),
                                                  Get_absorb_nabla2(),
                                                  Get_CUFFT_X_temp(),
                                                  Get_CUFFT_Y_temp());


    Get_CUFFT_X_temp().Compute_FFT_3D_C2R(Absorb_tau_temp);
    Get_CUFFT_Y_temp().Compute_FFT_3D_C2R(Absorb_eta_temp);



    Sum_Subterms_nonlinear(Absorb_tau_temp,
                           Absorb_eta_temp,
                           BonA_rho_rhoxyz);
  }
  else
  {
        Sum_new_p_nonlinear_lossless();
  }
}// end of Compute_new_p_nonlinear()
//---------------------------------------------------------------------------

/*
 * Compute new p for linear case.
 */
void TKSpaceFirstOrder3DSolver::Compute_new_p_linear()
{
  if (Parameters->Get_absorbing_flag())
  { // absorbing case
    TRealMatrix& Sum_rhoxyz  = Get_Temp_1_RS3D();
    TRealMatrix& Sum_rho0_du = Get_Temp_2_RS3D();

    TRealMatrix& Absorb_tau_temp = Get_Temp_2_RS3D();
    TRealMatrix& Absorb_eta_temp = Get_Temp_3_RS3D();

    Calculate_SumRho_SumRhoDu(Sum_rhoxyz,Sum_rho0_du);


    // ifftn ( absorb_nabla1 * fftn (rho0 * (duxdx+duydy+duzdz))
    Get_CUFFT_X_temp().Compute_FFT_3D_R2C(Sum_rho0_du);
    Get_CUFFT_Y_temp().Compute_FFT_3D_R2C(Sum_rhoxyz);

    cuda_implementations->Compute_Absorb_nabla1_2(Get_absorb_nabla1(),
                                                  Get_absorb_nabla2(),
                                                  Get_CUFFT_X_temp(),
                                                  Get_CUFFT_Y_temp());

    Get_CUFFT_X_temp().Compute_FFT_3D_C2R(Absorb_tau_temp);
    Get_CUFFT_Y_temp().Compute_FFT_3D_C2R(Absorb_eta_temp);

    Sum_Subterms_linear(Absorb_tau_temp, Absorb_eta_temp, Sum_rhoxyz);
  }
  else
  {
    // lossless case
    Sum_new_p_linear_lossless();
  }
}// end of Compute_new_p_linear
//------------------------------------------------------------------------------

/*
 * Compute new values of ux_sgx, uy_sgy, uz_sgz.
 */
void TKSpaceFirstOrder3DSolver::Compute_uxyz()
{
  cuda_implementations->Compute_ddx_kappa_fft_p(Get_p(),
                                                Get_CUFFT_X_temp(),
                                                Get_CUFFT_Y_temp(),
                                                Get_CUFFT_Z_temp(),
                                                Get_kappa(),
                                                Get_ddx_k_shift_pos(),
                                                Get_ddy_k_shift_pos(),
                                                Get_ddz_k_shift_pos());

  Get_CUFFT_X_temp().Compute_FFT_3D_C2R(Get_Temp_1_RS3D());
  Get_CUFFT_Y_temp().Compute_FFT_3D_C2R(Get_Temp_2_RS3D());
  Get_CUFFT_Z_temp().Compute_FFT_3D_C2R(Get_Temp_3_RS3D());

  if (Parameters->Get_rho0_scalar_flag())
  { // scalars
    if (Parameters->Get_nonuniform_grid_flag())
    {
      cuda_implementations->Compute_ux_sgx_normalize_scalar_nonuniform(Get_ux_sgx(),
                                                                       Get_Temp_1_RS3D(),
                                                                       Parameters->Get_rho0_sgx_scalar(),
                                                                       Get_dxudxn_sgx(),
                                                                       Get_pml_x_sgx());
      cuda_implementations->Compute_uy_sgy_normalize_scalar_nonuniform(Get_uy_sgy(),
                                                                       Get_Temp_2_RS3D(),
                                                                       Parameters->Get_rho0_sgy_scalar(),
                                                                       Get_dyudyn_sgy(),
                                                                       Get_pml_y_sgy());
      cuda_implementations->Compute_uz_sgz_normalize_scalar_nonuniform(Get_uz_sgz(),
                                                                       Get_Temp_3_RS3D(),
                                                                       Parameters->Get_rho0_sgz_scalar(),
                                                                       Get_dzudzn_sgz(),
                                                                       Get_pml_z_sgz());
    }
    else
    {
      cuda_implementations->Compute_ux_sgx_normalize_scalar_uniform(Get_ux_sgx(),
                                                                    Get_Temp_1_RS3D(),
                                                                    Parameters->Get_rho0_sgx_scalar(),
                                                                    Get_pml_x_sgx());
      cuda_implementations->Compute_uy_sgy_normalize_scalar_uniform(Get_uy_sgy(),
                                                                    Get_Temp_2_RS3D(),
                                                                    Parameters->Get_rho0_sgy_scalar(),
                                                                    Get_pml_y_sgy());
      cuda_implementations->Compute_uz_sgz_normalize_scalar_uniform(Get_uz_sgz(),
                                                                    Get_Temp_3_RS3D(),
                                                                    Parameters->Get_rho0_sgz_scalar(),
                                                                    Get_pml_z_sgz());
    }
  }
  else
  {// matrices
    cuda_implementations->Compute_ux_sgx_normalize(Get_ux_sgx(),
                                                   Get_Temp_1_RS3D(),
                                                   Get_dt_rho0_sgx(),
                                                   Get_pml_x_sgx());
    cuda_implementations->Compute_uy_sgy_normalize(Get_uy_sgy(),
                                                   Get_Temp_2_RS3D(),
                                                   Get_dt_rho0_sgy(),
                                                   Get_pml_y_sgy());
    cuda_implementations->Compute_uz_sgz_normalize(Get_uz_sgz(),
                                                   Get_Temp_3_RS3D(),
                                                   Get_dt_rho0_sgz(),
                                                   Get_pml_z_sgz());
  }
}// end of Compute_uxyz()
//------------------------------------------------------------------------------

/*
 * Add u source to the particle velocity.
 */
void TKSpaceFirstOrder3DSolver::Add_u_source()
{
  size_t t_index = Parameters->Get_t_index();

  if (Parameters->Get_ux_source_flag() > t_index)
  {
    cuda_implementations->Add_u_source(Get_ux_sgx(),
                                       Get_ux_source_input(),
                                       Get_u_source_index(),
                                       t_index,
                                       Parameters->Get_u_source_mode(),
                                       Parameters->Get_u_source_many());
  }
  if (Parameters->Get_uy_source_flag() > t_index)
  {
    cuda_implementations->Add_u_source(Get_uy_sgy(),
                                       Get_uy_source_input(),
                                       Get_u_source_index(),
                                       t_index,
                                       Parameters->Get_u_source_mode(),
                                       Parameters->Get_u_source_many());
  }
  if (Parameters->Get_uz_source_flag() > t_index)
  {
      cuda_implementations->Add_u_source(Get_uz_sgz(),
                                         Get_uz_source_input(),
                                         Get_u_source_index(),
                                         t_index,
                                         Parameters->Get_u_source_mode(),
                                         Parameters->Get_u_source_many());
  }
}// end of Add_u_source
//------------------------------------------------------------------------------

/*
 * Add in pressure source.
 */
void TKSpaceFirstOrder3DSolver::Add_p_source()
{
  size_t t_index = Parameters->Get_t_index();

  if (Parameters->Get_p_source_flag() > t_index)
  {
    cuda_implementations->Add_p_source(Get_rhox(),
                                       Get_rhoy(),
                                       Get_rhoz(),
                                       Get_p_source_input(),
                                       Get_p_source_index(),
                                       Parameters->Get_p_source_many(),
                                       Parameters->Get_p_source_mode(),
                                       t_index);
  }// if do at all
}// end of Add_p_source
//------------------------------------------------------------------------------

/*
 * Compute the main time loop of KSpaceFirstOrder3D.
 */
void TKSpaceFirstOrder3DSolver::ComputeMainLoop()
{
  ActPercent = 0;

  // if resuming from a checkpoint,
  // set ActPercent to correspond the t_index after recovery
  if (Parameters->Get_t_index() > 0)
  {
    ActPercent = (Parameters->Get_t_index() / (Parameters->Get_Nt() / 100));
  }

  PrintOutputHeader();

  // Initial copy of data to the GPU
  MatrixContainer.CopyAllMatricesToGPU();

  IterationTime.Start();

  // execute main loop
  while (Parameters->Get_t_index() < Parameters->Get_Nt() && (!IsTimeToCheckpoint()))
  {
    const size_t t_index = Parameters->Get_t_index();

    Compute_uxyz();

    // add in the velocity u source term
    Add_u_source();

    // add in the transducer source term (t = t1) to ux
    if (Parameters->Get_transducer_source_flag() > t_index)
    {
      cuda_implementations->AddTransducerSource(Get_ux_sgx(),
                                                Get_u_source_index(),
                                                Get_delay_mask(),
                                                Get_transducer_source_input());
    }

    Compute_duxyz();

    if (Parameters->Get_nonlinear_flag())
    {
      Compute_rhoxyz_nonlinear();
    }
    else
    {
      Compute_rhoxyz_linear();
    }


    // add in the source pressure term
    Add_p_source();

    if (Parameters->Get_nonlinear_flag())
    {
      Compute_new_p_nonlinear();
    }
    else
    {
     Compute_new_p_linear();
    }


    //-- calculate initial pressure
    if ((t_index == 0) && (Parameters->Get_p0_source_flag() == 1))  Calculate_p0_source();



     StoreSensorData();
     PrintStatistics();
     Parameters->Increment_t_index();
  }

  // WHAT THE HELL IS THIS!!!! - checkpoint??
  MatrixContainer.CopyAllMatricesFromGPU();
}// end of ComputeMainLoop()
//------------------------------------------------------------------------------

/*
 * Print progress statistics.
 */
void TKSpaceFirstOrder3DSolver::PrintStatistics()
{
  const float  Nt = (float) Parameters->Get_Nt();
  const size_t t_index = Parameters->Get_t_index();


  if (t_index > (ActPercent * Nt * 0.01f) )
  {
    ActPercent += Parameters->GetVerboseInterval();

    IterationTime.Stop();

    const double ElTime = IterationTime.GetElapsedTime();
    const double ElTimeWithLegs = IterationTime.GetElapsedTime() + SimulationTime.GetCumulatedElapsedTimeOverPreviousLegs();
    const double ToGo   = ((ElTimeWithLegs / (float) (t_index + 1)) *  Nt) - ElTimeWithLegs;

    struct tm *current;
    time_t now;
    time(&now);
    now += ToGo;
    current = localtime(&now);

    fprintf(stdout, "%5li%c      %9.3fs      %9.3fs      %02i/%02i/%02i %02i:%02i:%02i\n",
            (size_t) ((t_index) / (Nt * 0.01f)),'%',
            ElTime, ToGo,
            current->tm_mday, current->tm_mon+1, current->tm_year-100,
            current->tm_hour, current->tm_min, current->tm_sec
            );

    fflush(stdout);
  }
}// end of KSpaceFirstOrder3DSolver
//------------------------------------------------------------------------------

/**
 * Print the header of the progress statistics.
 */
void TKSpaceFirstOrder3DSolver::PrintOutputHeader()
{
  fprintf(stdout, "-------------------------------------------------------------\n");
  fprintf(stdout, "....................... Simulation ..........................\n");
  fprintf(stdout, "Progress...ElapsedTime........TimeToGo......TimeOfTermination\n");
}// end of PrintOtputHeader
//------------------------------------------------------------------------------

/**
 * Is time to checkpoint?
 * @return true if it is necessary to stop to checkpoint
 */
bool TKSpaceFirstOrder3DSolver::IsTimeToCheckpoint()
{
  if (!Parameters->IsCheckpointEnabled()) return false;

  TotalTime.Stop();

  return (TotalTime.GetElapsedTime() > static_cast<float>(Parameters->GetCheckpointInterval()));
}// end of IsTimeToCheckpoint
//------------------------------------------------------------------------------

/*
 * Post processing, and closing the output streams.
 */
void TKSpaceFirstOrder3DSolver::PostProcessing()
{
  if (Parameters->IsStore_p_final())
  {
    Get_p().WriteDataToHDF5File(Parameters->HDF5_OutputFile,
                                p_final_Name,
                                Parameters->GetCompressionLevel());
  }// p_final

  if (Parameters->IsStore_u_final())
  {
    Get_ux_sgx().WriteDataToHDF5File(Parameters->HDF5_OutputFile,
                                     ux_final_Name,
                                     Parameters->GetCompressionLevel());
    Get_uy_sgy().WriteDataToHDF5File(Parameters->HDF5_OutputFile,
                                     uy_final_Name,
                                     Parameters->GetCompressionLevel());
    Get_uz_sgz().WriteDataToHDF5File(Parameters->HDF5_OutputFile,
                                     uz_final_Name,
                                     Parameters->GetCompressionLevel());
  }// u_final

  // Apply post-processing and close
  OutputStreamContainer.PostProcessStreams();
  OutputStreamContainer.CloseStreams();

  // store sensor mask if wanted
  if (Parameters->IsCopySensorMask())
  {
    if (Parameters->Get_sensor_mask_type() == TParameters::smt_index)
    {
      Get_sensor_mask_index().RecomputeIndicesToMatlab();
      Get_sensor_mask_index().WriteDataToHDF5File(Parameters->HDF5_OutputFile,sensor_mask_index_Name,
                                                  Parameters->GetCompressionLevel());
    }
    //@todo implement this
    /*if (Parameters->Get_sensor_mask_type() == TParameters::smt_corners)
    {
      Get_sensor_mask_corners().RecomputeIndicesToMatlab();
      Get_sensor_mask_corners().WriteDataToHDF5File(Parameters->HDF5_OutputFile,sensor_mask_corners_Name,
                                                    Parameters->GetCompressionLevel());
    }*/
  }
}// end of PostProcessing
//------------------------------------------------------------------------------

/*
 * Store sensor data.
 * @todo this must be reworked to be done on the GPU side
 */
void TKSpaceFirstOrder3DSolver::StoreSensorData()
{
  // Unless the time for sampling has come, exit
  if (Parameters->Get_t_index() >= Parameters->GetStartTimeIndex())
  {
    OutputStreamContainer.SampleStreams();
  }
}// end of StoreData
//------------------------------------------------------------------------------

/**
 * Save checkpoint data into the checkpoint file, flush aggregated outputs into
 * the output file.
 */
void TKSpaceFirstOrder3DSolver::SaveCheckpointData()
{

  // @todo Here it should download data from GPU...

  // Create Checkpoint file
  THDF5_File & HDF5_CheckpointFile = Parameters->HDF5_CheckpointFile;
  // if it happens and the file is opened (from the recovery, close it)
  if (HDF5_CheckpointFile.IsOpened()) HDF5_CheckpointFile.Close();

  // Create the new file (overwrite the old one)
  HDF5_CheckpointFile.Create(Parameters->GetCheckpointFileName().c_str());

  //--------------------- Store Matrices ------------------------------//

  // Store all necessary matrices in Checkpoint file
  MatrixContainer.StoreDataIntoCheckpointHDF5File(HDF5_CheckpointFile);
  // Write t_index
  HDF5_CheckpointFile.WriteScalarValue(HDF5_CheckpointFile.GetRootGroup(),
                                       t_index_Name,
                                       Parameters->Get_t_index());

  // store basic dimension sizes (Nx, Ny, Nz) - Nt is not necessary
  HDF5_CheckpointFile.WriteScalarValue(HDF5_CheckpointFile.GetRootGroup(),
                                       Nx_Name,
                                       Parameters->GetFullDimensionSizes().X);
  HDF5_CheckpointFile.WriteScalarValue(HDF5_CheckpointFile.GetRootGroup(),
                                       Ny_Name,
                                       Parameters->GetFullDimensionSizes().Y);
  HDF5_CheckpointFile.WriteScalarValue(HDF5_CheckpointFile.GetRootGroup(),
                                       Nz_Name,
                                       Parameters->GetFullDimensionSizes().Z);

  // Write checkpoint file header
  THDF5_FileHeader CheckpointFileHeader = Parameters->HDF5_FileHeader;

  CheckpointFileHeader.SetFileType(THDF5_FileHeader::hdf5_ft_checkpoint);
  CheckpointFileHeader.SetCodeName(GetCodeName());
  CheckpointFileHeader.SetActualCreationTime();

  CheckpointFileHeader.WriteHeaderToCheckpointFile(HDF5_CheckpointFile);

  HDF5_CheckpointFile.Close();

  // checkpoint only if necessary (t_index >= start_index)
  if (Parameters->Get_t_index() >= Parameters->GetStartTimeIndex())
  {
    OutputStreamContainer.CheckpointStreams();
  }

  OutputStreamContainer.CloseStreams();
}// end of SaveCheckpointData()
//------------------------------------------------------------------------------

/**
 * Write statistics and the header into the output file.
 */
void  TKSpaceFirstOrder3DSolver::WriteOutputDataInfo()
{
  // write t_index into the output file
  Parameters->HDF5_OutputFile.WriteScalarValue(Parameters->HDF5_OutputFile.GetRootGroup(),
                                               t_index_Name,
                                               Parameters->Get_t_index());

  // Write scalars
  Parameters->SaveScalarsToHDF5File(Parameters->HDF5_OutputFile);
  THDF5_FileHeader & HDF5_FileHeader = Parameters->HDF5_FileHeader;

  // Write File header
  HDF5_FileHeader.SetCodeName(GetCodeName());
  HDF5_FileHeader.SetMajorFileVersion();
  HDF5_FileHeader.SetMinorFileVersion();
  HDF5_FileHeader.SetActualCreationTime();
  HDF5_FileHeader.SetFileType(THDF5_FileHeader::hdf5_ft_output);
  HDF5_FileHeader.SetHostName();

  // @todo - what to store here???
  HDF5_FileHeader.SetMemoryConsumption(GetHostMemoryUsageInMB());

  // Stop total timer here
  TotalTime.Stop();
  HDF5_FileHeader.SetExecutionTimes(GetTotalTime(),
                                    GetDataLoadTime(),
                                    GetPreProcessingTime(),
                                    GetSimulationTime(),
                                    GetPostProcessingTime());

  HDF5_FileHeader.SetNumberOfCores();

  HDF5_FileHeader.WriteHeaderToOutputFile(Parameters->HDF5_OutputFile);
}// end of WriteOutputDataInfo
//------------------------------------------------------------------------------

/**
 *
 * Restore cumulated elapsed time form Output file
 * Open the header, read this and store into TMPI_Time classes
 * @param [in] OutputFile
 */
void TKSpaceFirstOrder3DSolver::RestoreCumulatedElapsedFromOutputFile(THDF5_File& HDF5_OutputFile)
{
  double ElapsedTotalTime, ElapsedDataLoadTime, ElapsedPreProcessingTime,
         ElapsedSimulationTime, ElapsedPostProcessingTime;

  // Get execution times stored in the output file header
  Parameters->HDF5_FileHeader.GetExecutionTimes(ElapsedTotalTime,
                                                ElapsedDataLoadTime,
                                                ElapsedPreProcessingTime,
                                                ElapsedSimulationTime,
                                                ElapsedPostProcessingTime);

  TotalTime.SetCumulatedElapsedTimeOverPreviousLegs(ElapsedTotalTime);
  DataLoadTime.SetCumulatedElapsedTimeOverPreviousLegs(ElapsedDataLoadTime);
  PreProcessingTime.SetCumulatedElapsedTimeOverPreviousLegs(ElapsedPreProcessingTime);
  SimulationTime.SetCumulatedElapsedTimeOverPreviousLegs(ElapsedSimulationTime);
  PostProcessingTime.SetCumulatedElapsedTimeOverPreviousLegs(ElapsedPostProcessingTime);

}// end of RestoreCumulatedElapsedFromOutputFile
//------------------------------------------------------------------------------

/**
 * Check the output file has the correct format and version.
 * @throw ios::failure if an error happens
 */
void TKSpaceFirstOrder3DSolver::CheckOutputFile()
{
  // The header has already been read
  THDF5_FileHeader & OutputFileHeader = Parameters->HDF5_FileHeader;
  THDF5_File       & OutputFile       = Parameters->HDF5_OutputFile;

  // test file type
  if (OutputFileHeader.GetFileType() != THDF5_FileHeader::hdf5_ft_output)
  {
    char ErrorMessage[256] = "";
    sprintf(ErrorMessage,
            KSpaceFirstOrder3DSolver_ERR_FMT_IncorrectOutputFileFormat,
            Parameters->GetOutputFileName().c_str());
    throw ios::failure(ErrorMessage);
  }

  // test file major version
  if (!OutputFileHeader.CheckMajorFileVersion())
  {
    char ErrorMessage[256] = "";
    sprintf(ErrorMessage,
            Parameters_ERR_FMT_IncorrectMajorHDF5FileVersion,
            Parameters->GetCheckpointFileName().c_str(),
            OutputFileHeader.GetCurrentHDF5_MajorVersion().c_str());
    throw ios::failure(ErrorMessage);
  }

  // test file minor version
  if (!OutputFileHeader.CheckMinorFileVersion())
  {
    char ErrorMessage[256] = "";
    sprintf(ErrorMessage,
            Parameters_ERR_FMT_IncorrectMinorHDF5FileVersion,
            Parameters->GetCheckpointFileName().c_str(),
            OutputFileHeader.GetCurrentHDF5_MinorVersion().c_str());
    throw ios::failure(ErrorMessage);
  }


  // Check dimension sizes
  TDimensionSizes OutputDimSizes;
  OutputFile.ReadScalarValue(OutputFile.GetRootGroup(),
                             Nx_Name,
                             OutputDimSizes.X);

  OutputFile.ReadScalarValue(OutputFile.GetRootGroup(),
                             Ny_Name,
                             OutputDimSizes.Y);

  OutputFile.ReadScalarValue(OutputFile.GetRootGroup(),
                             Nz_Name,
                             OutputDimSizes.Z);

 if (Parameters->GetFullDimensionSizes() != OutputDimSizes)
 {
    char ErrorMessage[256] = "";
    sprintf(ErrorMessage,
            KSpaceFirstOrder3DSolver_ERR_FMT_OutputDimensionsDoNotMatch,
            OutputDimSizes.X,
            OutputDimSizes.Y,
            OutputDimSizes.Z,
            Parameters->GetFullDimensionSizes().X,
            Parameters->GetFullDimensionSizes().Y,
            Parameters->GetFullDimensionSizes().Z);

   throw ios::failure(ErrorMessage);
 }
}// end of CheckOutputFile
//------------------------------------------------------------------------------


/**
 * Check the file type and the version of the checkpoint file.
 * @throw ios::failure if an error happens
 *
 */
void TKSpaceFirstOrder3DSolver::CheckCheckpointFile()
{
  // read the header and check the file version
  THDF5_FileHeader CheckpointFileHeader;
  THDF5_File &     HDF5_CheckpointFile = Parameters->HDF5_CheckpointFile;

  CheckpointFileHeader.ReadHeaderFromCheckpointFile(HDF5_CheckpointFile);

  // test file type
  if (CheckpointFileHeader.GetFileType() != THDF5_FileHeader::hdf5_ft_checkpoint)
  {
    char ErrorMessage[256] = "";
    sprintf(ErrorMessage,
            KSpaceFirstOrder3DSolver_ERR_FMT_IncorrectCheckpointFileFormat,
            Parameters->GetCheckpointFileName().c_str());
    throw ios::failure(ErrorMessage);
  }

  // test file major version
  if (!CheckpointFileHeader.CheckMajorFileVersion())
  {
    char ErrorMessage[256] = "";
    sprintf(ErrorMessage,
            Parameters_ERR_FMT_IncorrectMajorHDF5FileVersion,
            Parameters->GetCheckpointFileName().c_str(),
            CheckpointFileHeader.GetCurrentHDF5_MajorVersion().c_str());
    throw ios::failure(ErrorMessage);
  }

  // test file minor version
  if (!CheckpointFileHeader.CheckMinorFileVersion())
  {
    char ErrorMessage[256] = "";
    sprintf(ErrorMessage,
            Parameters_ERR_FMT_IncorrectMinorHDF5FileVersion,
            Parameters->GetCheckpointFileName().c_str(),
            CheckpointFileHeader.GetCurrentHDF5_MinorVersion().c_str());
    throw ios::failure(ErrorMessage);
  }


  // Check dimension sizes
  TDimensionSizes CheckpointDimSizes;
  HDF5_CheckpointFile.ReadScalarValue(HDF5_CheckpointFile.GetRootGroup(),
                                      Nx_Name,
                                      CheckpointDimSizes.X);

  HDF5_CheckpointFile.ReadScalarValue(HDF5_CheckpointFile.GetRootGroup(),
                                      Ny_Name,
                                      CheckpointDimSizes.Y);

  HDF5_CheckpointFile.ReadScalarValue(HDF5_CheckpointFile.GetRootGroup(),
                                      Nz_Name,
                                      CheckpointDimSizes.Z);

 if (Parameters->GetFullDimensionSizes() != CheckpointDimSizes)
 {
    char ErrorMessage[256] = "";
    sprintf(ErrorMessage,
            KSpaceFirstOrder3DSolver_ERR_FMT_CheckpointDimensionsDoNotMatch,
            CheckpointDimSizes.X,
            CheckpointDimSizes.Y,
            CheckpointDimSizes.Z,
            Parameters->GetFullDimensionSizes().X,
            Parameters->GetFullDimensionSizes().Y,
            Parameters->GetFullDimensionSizes().Z);

   throw ios::failure(ErrorMessage);
 }
}// end of CheckCheckpointFile
//------------------------------------------------------------------------------