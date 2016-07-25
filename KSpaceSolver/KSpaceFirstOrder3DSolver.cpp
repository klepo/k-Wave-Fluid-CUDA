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
 * @version     kspaceFirstOrder3D 3.4
 * @date        12 July     2012, 10:27 (created)\n
 *              14 July     2016, 15:19 (revised)
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

#include <omp.h>
#include <time.h>
#include <limits>
#include <cstdio>

#include <KSpaceSolver/KSpaceFirstOrder3DSolver.h>

#include <Logger/ErrorMessages.h>
#include <Logger/Logger.h>

#include <KSpaceSolver/SolverCUDAKernels.cuh>
#include <Containers/MatrixContainer.h>


using namespace std;

//----------------------------------------------------------------------------//
//                               Constants                                    //
//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//
//                             Public methods                                 //
//----------------------------------------------------------------------------//

/**
 * Constructor of the class.
 */
TKSpaceFirstOrder3DSolver::TKSpaceFirstOrder3DSolver() :
        MatrixContainer(), OutputStreamContainer(),
        ActPercent(0), IsTimestepRightAfterRestore(false), Parameters(TParameters::GetInstance()),
        TotalTime(), PreProcessingTime(), DataLoadTime(), SimulationTime(),
        PostProcessingTime(), IterationTime()
{
  TotalTime.Start();

  //Switch off default HDF5 error messages
  H5Eset_auto(H5E_DEFAULT, NULL, NULL);
}// end of TKSpace3DSolver
//------------------------------------------------------------------------------

/**
 * Destructor of the class.
 */
TKSpaceFirstOrder3DSolver::~TKSpaceFirstOrder3DSolver()
{
  // Delete CUDA FFT plans and related data
  TCUFFTComplexMatrix::DestroyAllPlansAndStaticData();

  // Free memory
  FreeMemory();

  //Reset device after the run - recommended by CUDA SDK
  cudaDeviceReset();
}// end of ~TKSpace3DSolver
//------------------------------------------------------------------------------


/**
 * The method allocates the matrix container and create all matrices and
 * creates all output streams.
 *
 */
void TKSpaceFirstOrder3DSolver::AllocateMemory()
{
  TLogger::Log(TLogger::BASIC, OUT_FMT_MEMORY_ALLOCATION);
  TLogger::Flush(TLogger::BASIC);

  // create container, then all matrices
  MatrixContainer.AddMatrices();

  MatrixContainer.CreateMatrices();

  // add output streams into container
  //@todo Think about moving under LoadInputData routine...
  OutputStreamContainer.AddStreams(MatrixContainer);

  TLogger::Log(TLogger::BASIC, OUT_FMT_DONE);
}// end of AllocateMemory
//------------------------------------------------------------------------------

/*
 * The method frees all memory allocated by the class.
 */
void TKSpaceFirstOrder3DSolver::FreeMemory()
{
  MatrixContainer.FreeMatrices();
  OutputStreamContainer.FreeStreams();
}// end of FreeMemory
//------------------------------------------------------------------------------

/**
 * Load data from the input file provided by the Parameter class and creates
 * the output time series streams.
 */
void TKSpaceFirstOrder3DSolver::LoadInputData()
{
  // Load data from disk
  TLogger::Log(TLogger::BASIC, OUT_FMT_DATA_LOADING);
  TLogger::Flush(TLogger::BASIC);

  DataLoadTime.Start();

  // open and load input file
  THDF5_File& HDF5_InputFile      = Parameters.GetInputFile(); // file is opened (in Parameters)
  THDF5_File& HDF5_OutputFile     = Parameters.GetOutputFile();
  THDF5_File& HDF5_CheckpointFile = Parameters.GetCheckpointFile();

  // Load data from disk
  TLogger::Log(TLogger::FULL, OUT_FMT_FINSIH_NO_DONE);
  TLogger::Log(TLogger::FULL, OUT_FMT_READING_INPUT_FILE);
  TLogger::Flush(TLogger::FULL);

  MatrixContainer.LoadDataFromInputFile(HDF5_InputFile);

  // close the input file
  HDF5_InputFile.Close();

  TLogger::Log(TLogger::FULL, OUT_FMT_DONE);

  // The simulation does not use check pointing or this is the first turn
  bool RecoverFromPrevState = (Parameters.IsCheckpointEnabled() &&
                               THDF5_File::IsHDF5(Parameters.GetCheckpointFileName().c_str()));

  //-------------------- Read data from the checkpoint file ------------------//
  if (RecoverFromPrevState)
  {
    TLogger::Log(TLogger::FULL, OUT_FMT_READING_CHECKPOINT_FILE);
    TLogger::Flush(TLogger::FULL);

    // Open checkpoint file
    HDF5_CheckpointFile.Open(Parameters.GetCheckpointFileName().c_str());

    // Check the checkpoint file
    CheckCheckpointFile();

    // read the actual value of t_index
    size_t new_t_index;
    HDF5_CheckpointFile.ReadScalarValue(HDF5_CheckpointFile.GetRootGroup(),
                                        t_index_NAME,
                                        new_t_index);
    Parameters.Set_t_index(new_t_index);

    // Read necessary matrices from the checkpoint file
    MatrixContainer.LoadDataFromCheckpointFile(HDF5_CheckpointFile);

    HDF5_CheckpointFile.Close();
    TLogger::Log(TLogger::FULL, OUT_FMT_DONE);

    //------------- Read data from the output file ---------------------------//
    // Reopen output file for RW access
    TLogger::Log(TLogger::FULL, OUT_FMT_READING_OUTPUT_FILE);
    TLogger::Flush(TLogger::FULL);

    HDF5_OutputFile.Open(Parameters.GetOutputFileName().c_str(), H5F_ACC_RDWR);

    //Read file header of the output file
    Parameters.GetFileHeader().ReadHeaderFromOutputFile(HDF5_OutputFile);

    // Restore elapsed time
    RestoreCumulatedElapsedFromOutputFile(HDF5_OutputFile);

    // Reopen streams
    OutputStreamContainer.ReopenStreams();
    TLogger::Log(TLogger::FULL, OUT_FMT_DONE);
  }
  else
  {
    //-------------------- First round of multi-leg simulation ---------------//
    // Create the output file
    TLogger::Log(TLogger::FULL, OUT_FMT_CREATING_OUTPUT_FILE);
    TLogger::Flush(TLogger::FULL);

    HDF5_OutputFile.Create(Parameters.GetOutputFileName().c_str());
    TLogger::Log(TLogger::FULL, OUT_FMT_DONE);

    // Create the steams, link them with the sampled matrices
    // however DO NOT allocate memory!
    OutputStreamContainer.CreateStreams();
  }

  DataLoadTime.Stop();

  if (TLogger::GetLevel() != TLogger::FULL)
  {
    TLogger::Log(TLogger::BASIC, OUT_FMT_DONE);
  }
}// end of LoadInputData
//------------------------------------------------------------------------------

/**
 * This method computes k-space First Order 3D simulation.
 * It launches calculation on a given dataset going through
 * FFT initialization, pre-processing, main loop and post-processing phases.
 *
 */
void TKSpaceFirstOrder3DSolver::Compute()
{
  PreProcessingTime.Start();

  TLogger::Log(TLogger::BASIC, OUT_FMT_FFT_PLANS);
  TLogger::Flush(TLogger::BASIC);

  TCUDAParameters& cudaParameters = Parameters.GetCudaParameters();

  // fft initialisation and preprocessing
  try
  {
    // initilaise all cuda FFT plans
    InitializeFFTPlans();
    TLogger::Log(TLogger::BASIC, OUT_FMT_DONE);

    TLogger::Log(TLogger::BASIC,OUT_FMT_PRE_PROCESSING);
    TLogger::Flush(TLogger::BASIC);

    // preprocessing is done on CPU and must pretend the CUDA configuration
    PreProcessingPhase();

    PreProcessingTime.Stop();
    // Set kernel configurations
    cudaParameters.SetKernelConfiguration();

    // Set up constant memory - copy over to GPU
    // Constant memory uses some variables calculated during preprocessing
    cudaParameters.SetUpDeviceConstants();

    TLogger::Log(TLogger::BASIC, OUT_FMT_DONE);
  }
  catch (const exception& e)
  {
    TLogger::Log(TLogger::BASIC, OUT_FMT_FAILED);
    TLogger::Log(TLogger::BASIC, OUT_FMT_LAST_SEPARATOR);

    TLogger::ErrorAndTerminate(TLogger::WordWrapString(e.what(),ERR_FMT_PATH_DELIMITERS,9).c_str());
  }

  // Logger header for simulation
  TLogger::Log(TLogger::BASIC,
               OUT_FMT_ELAPSED_TIME,
               PreProcessingTime.GetElapsedTime());

  TLogger::Log(TLogger::BASIC,OUT_FMT_COMP_RESOURCES_HEADER);

  TLogger::Log(TLogger::BASIC,
               OUT_FMT_CURRENT_HOST_MEMORY,
               GetHostMemoryUsageInMB());

  TLogger::Log(TLogger::BASIC,
               OUT_FMT_CURRENT_DEVICE_MEMORY,
               GetDeviceMemoryUsageInMB());

  char GridDimensions[19];
  snprintf(GridDimensions, 19, OUT_FMT_CUDA_GRID_SHAPE_FORMAT,
           cudaParameters.GetSolverGridSize1D(),
           cudaParameters.GetSolverBlockSize1D());

  TLogger::Log(TLogger::FULL,
               OUT_FMT_CUDA_SOLVER_GRID_SHAPE,
               GridDimensions);

  snprintf(GridDimensions, 18, OUT_FMT_CUDA_GRID_SHAPE_FORMAT,
           cudaParameters.GetSamplerGridSize1D(),
           cudaParameters.GetSamplerBlockSize1D());

  TLogger::Log(TLogger::FULL,
               OUT_FMT_CUDA_SAMPLER_GRID_SHAPE,
               GridDimensions);

  // Main loop
  try
  {
    SimulationTime.Start();

    ComputeMainLoop();
    SimulationTime.Stop();
    TLogger::Log(TLogger::BASIC,OUT_FMT_SIMULATOIN_END_SEPARATOR);
  }
  catch (const exception& e)
  {
    TLogger::Log(TLogger::BASIC,
                 OUT_FMT_SIMULATION_FINAL_SEPARATOR,
                 GridDimensions);
    TLogger::ErrorAndTerminate(TLogger::WordWrapString(e.what(),ERR_FMT_PATH_DELIMITERS,9).c_str());
  }

  //Post processing region
  PostProcessingTime.Start();

  try
  {
    if (IsCheckpointInterruption())
    { // Checkpoint
      TLogger::Log(TLogger::BASIC,
                   OUT_FMT_ELAPSED_TIME,
                   SimulationTime.GetElapsedTime());

      TLogger::Log(TLogger::BASIC,
                   OUT_FMT_CHECKPOINT_TIME_STEPS,
                   Parameters.Get_t_index());

      TLogger::Log(TLogger::BASIC, OUT_FMT_CHECKPOINT_HEADER);

      TLogger::Log(TLogger::BASIC, OUT_FMT_CREATING_CHECKPOINT);
      TLogger::Flush(TLogger::BASIC);

      if (TLogger::GetLevel() == TLogger::FULL)
      {
        TLogger::Log(TLogger::BASIC, OUT_FMT_FINSIH_NO_DONE);
      }

      SaveCheckpointData();

      if (TLogger::GetLevel() != TLogger::FULL)
      {
        TLogger::Log(TLogger::BASIC, OUT_FMT_DONE);
      }
    }
    else
    { // Finish

      TLogger::Log(TLogger::BASIC,
                   OUT_FMT_ELAPSED_TIME,
                   SimulationTime.GetElapsedTime());

      TLogger::Log(TLogger::BASIC,OUT_FMT_SEPARATOR);

      TLogger::Log(TLogger::BASIC, OUT_FMT_POST_PROCESSING);

      TLogger::Flush(TLogger::BASIC);
      PostProcessing();

      // if checkpointing is enabled and the checkpoint file was created in the past, delete it
      if (Parameters.IsCheckpointEnabled())
      {
        std::remove(Parameters.GetCheckpointFileName().c_str());
      }
      TLogger::Log(TLogger::BASIC, OUT_FMT_DONE);
    }
  }
  catch (const exception &e)
  {
    TLogger::Log(TLogger::BASIC, OUT_FMT_FAILED);
    TLogger::Log(TLogger::BASIC, OUT_FMT_LAST_SEPARATOR);

    TLogger::ErrorAndTerminate(TLogger::WordWrapString(e.what(),ERR_FMT_PATH_DELIMITERS,9).c_str());
  }
  PostProcessingTime.Stop();

  // Final data written
  try
  {
    WriteOutputDataInfo();
    Parameters.GetOutputFile().Close();

    TLogger::Log(TLogger::BASIC,
                 OUT_FMT_ELAPSED_TIME,
                 PostProcessingTime.GetElapsedTime());
    }
  catch (const exception &e)
  {
    TLogger::Log(TLogger::BASIC, OUT_FMT_LAST_SEPARATOR);

    TLogger::ErrorAndTerminate(TLogger::WordWrapString(e.what(),ERR_FMT_PATH_DELIMITERS,9).c_str());
  }

}// end of Compute()
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
 * Get release code version
 * @return core name
 */
const string TKSpaceFirstOrder3DSolver::GetCodeName() const
{
  return string(OUT_FMT_KWAVE_VERSION);
}// end of GetCodeName
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

/**
 * Print Full code name and the license
 *
 * @param [in] file - file to print the data (stdout)
 */
void TKSpaceFirstOrder3DSolver::PrintFullNameCodeAndLicense()
{
  TLogger::Log(TLogger::BASIC,
               OUT_FMT_BUILD_NO_DATE_TIME,
               10,11,__DATE__,
               8,8,__TIME__);


  if (Parameters.GetGitHash() != "")
  {
    TLogger::Log(TLogger::BASIC,
                 OUT_FMT_VERSION_GIT_HASH,
                 Parameters.GetGitHash().c_str());
  }

  TLogger::Log(TLogger::BASIC, OUT_FMT_SEPARATOR);

  // OS detection
  #ifdef __linux__
    TLogger::Log(TLogger::BASIC, OUT_FMT_LINUX_BUILD);
  #elif __APPLE__
    TLogger::Log(TLogger::BASIC, OUT_FMT_MAC_OS_BUILD);
  #elif _WIN32
    TLogger::Log(TLogger::BASIC, OUT_FMT_WINDOWS_BUILD);
  #endif

  // Compiler detections
  #if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
    TLogger::Log(TLogger::BASIC,
                 OUT_FMT_GNU_COMPILER,
                 __VERSION__);
  #endif
  #ifdef __INTEL_COMPILER
    TLogger::Log(TLogger::BASIC,
                 OUT_FMT_INTEL_COMPILER,
                 __INTEL_COMPILER);
  #endif
  #ifdef _MSC_VER
	TLogger::Log(TLogger::BASIC,
               OUT_FMT_VISUAL_STUDIO_COMPILER,
               _MSC_VER);
  #endif

      // instruction set
  #if (defined (__AVX2__))
    TLogger::Log(TLogger::BASIC, OUT_FMT_AVX2);
  #elif (defined (__AVX__))
    TLogger::Log(TLogger::BASIC, OUT_FMT_AVX);
  #elif (defined (__SSE4_2__))
    TLogger::Log(TLogger::BASIC, OUT_FMT_SSE42);
  #elif (defined (__SSE4_1__))
    TLogger::Log(TLogger::BASIC, OUT_FMT_SSE41);
  #elif (defined (__SSE3__))
    TLogger::Log(TLogger::BASIC, OUT_FMT_SSE3);
  #elif (defined (__SSE2__))
    TLogger::Log(TLogger::BASIC, OUT_FMT_SSE2);
  #endif

  TLogger::Log(TLogger::BASIC, OUT_FMT_SEPARATOR);

 // CUDA detection
  int cudaRuntimeVersion;
  if (cudaRuntimeGetVersion(&cudaRuntimeVersion) != cudaSuccess)
  {
    TLogger::Log(TLogger::BASIC, OUT_FMT_CUDA_RUNTIME_NA);
  }
  else
  {
    TLogger::Log(TLogger::BASIC,
                 OUT_FMT_CUDA_RUNTIME,
                 cudaRuntimeVersion/1000, (cudaRuntimeVersion%100)/10);
  }

  int cudaDriverVersion;
  cudaDriverGetVersion(&cudaDriverVersion);
  TLogger::Log(TLogger::BASIC,
               OUT_FMT_CUDA_DRIVER,
               cudaDriverVersion/1000, (cudaDriverVersion%100)/10);


  const TCUDAParameters& cudaParameters = Parameters.GetCudaParameters();
  // no GPU was found
  if (cudaParameters.GetDeviceIdx() == TCUDAParameters::DEFAULT_DEVICE_IDX)
  {
    TLogger::Log(TLogger::BASIC, OUT_FMT_CUDA_DEVICE_INFO_NA);
  }
  else
  {
    TLogger::Log(TLogger::BASIC,
                  OUT_FMT_CUDA_CODE_ARCH,
                  SolverCUDAKernels::GetCUDACodeVersion()/10.f);
    TLogger::Log(TLogger::BASIC, OUT_FMT_SEPARATOR);

    TLogger::Log(TLogger::BASIC,
                 OUT_FMT_CUDA_DEVICE,
                 cudaParameters.GetDeviceIdx());

    int paddingLength = 65 - ( 22 +   strlen(cudaParameters.GetDeviceName().c_str()));

    TLogger::Log(TLogger::BASIC,
                 OUT_FMT_CUDA_DEVICE_NAME,
                 cudaParameters.GetDeviceName().c_str(),
                 paddingLength,
                 OUT_FMT_CUDA_DEVICE_NAME_PADDING);

    TLogger::Log(TLogger::BASIC,
                 OUT_FMT_CUDA_CAPABILITY,
                 cudaParameters.GetDeviceProperties().major,
                 cudaParameters.GetDeviceProperties().minor);
  }

  TLogger::Log(TLogger::BASIC, OUT_FMT_LICENCE);
}// end of GetFullCodeAndLincence
//------------------------------------------------------------------------------


//---------------------------------------------------------------------------//
//                            Protected methods                              //
//---------------------------------------------------------------------------//

/**
 * Initialize FFT plans.
 */
void TKSpaceFirstOrder3DSolver::InitializeFFTPlans()
{
  // create real to complex plans
  TCUFFTComplexMatrix::Create_FFT_Plan_3D_R2C(Parameters.GetFullDimensionSizes());

 // create complex to real plans
  TCUFFTComplexMatrix::Create_FFT_Plan_3D_C2R(Parameters.GetFullDimensionSizes());

  // if necessary, create 1D shift plans.
  // in this case, the matrix has a bit bigger dimensions to be able to store
  // shifted matrices.
  if (TParameters::GetInstance().IsStore_u_non_staggered_raw())
  {
    // X shifts
    TCUFFTComplexMatrix::Create_FFT_Plan_1DX_R2C(Parameters.GetFullDimensionSizes());
    TCUFFTComplexMatrix::Create_FFT_Plan_1DX_C2R(Parameters.GetFullDimensionSizes());

    // Y shifts
    TCUFFTComplexMatrix::Create_FFT_Plan_1DY_R2C(Parameters.GetFullDimensionSizes());
    TCUFFTComplexMatrix::Create_FFT_Plan_1DY_C2R(Parameters.GetFullDimensionSizes());

    // Z shifts
    TCUFFTComplexMatrix::Create_FFT_Plan_1DZ_R2C(Parameters.GetFullDimensionSizes());
    TCUFFTComplexMatrix::Create_FFT_Plan_1DZ_C2R(Parameters.GetFullDimensionSizes());
  }// end u_non_staggered
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
  if (Parameters.Get_sensor_mask_type() == TParameters::INDEX)
  {
    Get_sensor_mask_index().RecomputeIndicesToCPP();
  }

  if (Parameters.Get_sensor_mask_type() == TParameters::CORNERS)
  {
    Get_sensor_mask_corners().RecomputeIndicesToCPP();
  }

  if ((Parameters.Get_transducer_source_flag() != 0) ||
      (Parameters.Get_ux_source_flag() != 0)         ||
      (Parameters.Get_uy_source_flag() != 0)         ||
      (Parameters.Get_uz_source_flag() != 0)
     )
  {
    Get_u_source_index().RecomputeIndicesToCPP();
  }

  if (Parameters.Get_transducer_source_flag() != 0)
  {
    Get_delay_mask().RecomputeIndicesToCPP();
  }

  if (Parameters.Get_p_source_flag() != 0)
  {
    Get_p_source_index().RecomputeIndicesToCPP();
  }

  // compute dt / rho0_sg...
  if (Parameters.Get_rho0_scalar_flag())
  { // rho is scalar
    Parameters.Get_rho0_sgx_scalar() = Parameters.Get_dt() / Parameters.Get_rho0_sgx_scalar();
    Parameters.Get_rho0_sgy_scalar() = Parameters.Get_dt() / Parameters.Get_rho0_sgy_scalar();
    Parameters.Get_rho0_sgz_scalar() = Parameters.Get_dt() / Parameters.Get_rho0_sgz_scalar();
  }
  else
  { // non-uniform grid cannot be pre-calculated :-(
    // rho is matrix
    if (Parameters.Get_nonuniform_grid_flag())
    {
      Calculate_dt_rho0_non_uniform();
    }
    else
    {
      Get_dt_rho0_sgx().ScalarDividedBy(Parameters.Get_dt());
      Get_dt_rho0_sgy().ScalarDividedBy(Parameters.Get_dt());
      Get_dt_rho0_sgz().ScalarDividedBy(Parameters.Get_dt());
    }
  }

  // generate different matrices
  if (Parameters.Get_absorbing_flag() != 0)
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
    const float dx_sq_rec = 1.0f / (Parameters.Get_dx() * Parameters.Get_dx());
    const float dy_sq_rec = 1.0f / (Parameters.Get_dy() * Parameters.Get_dy());
    const float dz_sq_rec = 1.0f / (Parameters.Get_dz() * Parameters.Get_dz());

    const float c_ref_dt_pi = Parameters.Get_c_ref() * Parameters.Get_dt() * float(M_PI);

    const float Nx_rec = 1.0f / static_cast<float>(Parameters.GetFullDimensionSizes().nx);
    const float Ny_rec = 1.0f / static_cast<float>(Parameters.GetFullDimensionSizes().ny);
    const float Nz_rec = 1.0f / static_cast<float>(Parameters.GetFullDimensionSizes().nz);

    const size_t X_Size  = Parameters.GetReducedDimensionSizes().nx;
    const size_t Y_Size  = Parameters.GetReducedDimensionSizes().ny;
    const size_t Z_Size  = Parameters.GetReducedDimensionSizes().nz;

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
    const float dx_sq_rec = 1.0f / (Parameters.Get_dx() * Parameters.Get_dx());
    const float dy_sq_rec = 1.0f / (Parameters.Get_dy() * Parameters.Get_dy());
    const float dz_sq_rec = 1.0f / (Parameters.Get_dz() * Parameters.Get_dz());

    const float c_ref_dt_2 = Parameters.Get_c_ref() * Parameters.Get_dt() * 0.5f;
    const float pi_2       = float(M_PI) * 2.0f;

    const size_t Nx = Parameters.GetFullDimensionSizes().nx;
    const size_t Ny = Parameters.GetFullDimensionSizes().ny;
    const size_t Nz = Parameters.GetFullDimensionSizes().nz;

    const float Nx_rec   = 1.0f / (float) Nx;
    const float Ny_rec   = 1.0f / (float) Ny;
    const float Nz_rec   = 1.0f / (float) Nz;

    const size_t X_Size  = Parameters.GetReducedDimensionSizes().nx;
    const size_t Y_Size  = Parameters.GetReducedDimensionSizes().ny;
    const size_t Z_Size  = Parameters.GetReducedDimensionSizes().nz;

    float * kappa           = Get_kappa().GetRawData();
    float * absorb_nabla1   = Get_absorb_nabla1().GetRawData();
    float * absorb_nabla2   = Get_absorb_nabla2().GetRawData();
    const float alpha_power = Parameters.Get_alpha_power();

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
  if ((Parameters.Get_alpha_coeff_scalar_flag()) && (Parameters.Get_c0_scalar_flag()))
  {
    const float alpha_power = Parameters.Get_alpha_power();
    const float tan_pi_y_2  = tan(static_cast<float> (M_PI_2) * alpha_power);
    const float alpha_db_neper_coeff = (100.0f * pow(1.0e-6f /(2.0f * static_cast<float>(M_PI)), alpha_power)) /
                                       (20.0f * static_cast<float>(M_LOG10E));

    const float alpha_coeff_2 = 2.0f * Parameters.Get_alpha_coeff_scalar() * alpha_db_neper_coeff;

    Parameters.Get_absorb_tau_scalar() = (-alpha_coeff_2) * pow(Parameters.Get_c0_scalar(), alpha_power - 1);
    Parameters.Get_absorb_eta_scalar() =   alpha_coeff_2  * pow(Parameters.Get_c0_scalar(), alpha_power) * tan_pi_y_2;
  }
  else
  { // matrix
    #pragma omp parallel
    {
      const size_t Z_Size  = Parameters.GetFullDimensionSizes().nz;
      const size_t Y_Size  = Parameters.GetFullDimensionSizes().ny;
      const size_t X_Size  = Parameters.GetFullDimensionSizes().nx;

      float * absorb_tau = Get_absorb_tau().GetRawData();
      float * absorb_eta = Get_absorb_eta().GetRawData();

      float * alpha_coeff;
      size_t  alpha_shift;

      if (Parameters.Get_alpha_coeff_scalar_flag())
      {
        alpha_coeff = &(Parameters.Get_alpha_coeff_scalar());
        alpha_shift = 0;
      }
      else
      {
        alpha_coeff = Get_Temp_1_RS3D().GetRawData();
        alpha_shift = 1;
      }

      float * c0;
      size_t c0_shift;
      if (Parameters.Get_c0_scalar_flag())
      {
        c0 = &(Parameters.Get_c0_scalar());
        c0_shift = 0;
      }
      else
      {
        c0 = Get_c2().GetRawData();
        c0_shift = 1;
      }

      const float alpha_power = Parameters.Get_alpha_power();
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

    const float dt = Parameters.Get_dt();

    const float * duxdxn_sgx = Get_dxudxn_sgx().GetRawData();
    const float * duydyn_sgy = Get_dyudyn_sgy().GetRawData();
    const float * duzdzn_sgz = Get_dzudzn_sgz().GetRawData();

    const size_t Z_Size = Get_dt_rho0_sgx().GetDimensionSizes().nz;
    const size_t Y_Size = Get_dt_rho0_sgx().GetDimensionSizes().ny;
    const size_t X_Size = Get_dt_rho0_sgx().GetDimensionSizes().nx;

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
  // get over the scalar problem
  bool Is_c2_scalar = Parameters.Get_c0_scalar_flag();
  const float* c2 = (Is_c2_scalar) ? nullptr : Get_c2().GetRawDeviceData();

  //-- add the initial pressure to rho as a mass source --//
  SolverCUDAKernels::Calculate_p0_source_add_initial_pressure(Get_p(),
                                                              Get_rhox(),
                                                              Get_rhoy(),
                                                              Get_rhoz(),
                                                              Get_p0_source_input(),
                                                              Is_c2_scalar,
                                                              c2);

  //-----------------------------------------------------------------------//
  //--compute u(t = t1 + dt/2) based on the assumption u(dt/2) = -u(-dt/2)-//
  //--    which forces u(t = t1) = 0                                      -//
  //-----------------------------------------------------------------------//
  Get_CUFFT_X_temp().Compute_FFT_3D_R2C(Get_p());

  SolverCUDAKernels::Compute_ddx_kappa_fft_p(Get_CUFFT_X_temp(),
                                             Get_CUFFT_Y_temp(),
                                             Get_CUFFT_Z_temp(),
                                             Get_kappa(),
                                             Get_ddx_k_shift_pos(),
                                             Get_ddy_k_shift_pos(),
                                             Get_ddz_k_shift_pos());

  Get_CUFFT_X_temp().Compute_FFT_3D_C2R(Get_ux_sgx());
  Get_CUFFT_Y_temp().Compute_FFT_3D_C2R(Get_uy_sgy());
  Get_CUFFT_Z_temp().Compute_FFT_3D_C2R(Get_uz_sgz());

  if (Parameters.Get_rho0_scalar_flag())
  {
    if (Parameters.Get_nonuniform_grid_flag())
    { // non uniform grid, homogeneous

      SolverCUDAKernels::Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform(Get_ux_sgx(),
                                                                            Get_uy_sgy(),
                                                                            Get_uz_sgz(),
                                                                            Get_dxudxn_sgx(),
                                                                            Get_dyudyn_sgy(),
                                                                            Get_dzudzn_sgz());

    }
    else
    { //uniform grid, homogeneous

      SolverCUDAKernels::Compute_dt_rho_sg_mul_ifft_div_2(Get_ux_sgx(),
                                                          Get_uy_sgy(),
                                                          Get_uz_sgz());
    }
  }
  else
  {
    // heterogeneous, uniform grid
    // divide the matrix by 2 and multiply with st./rho0_sg
    SolverCUDAKernels::Compute_dt_rho_sg_mul_ifft_div_2(Get_ux_sgx(),
                                                        Get_uy_sgy(),
                                                        Get_uz_sgz(),
                                                        Get_dt_rho0_sgx(),
                                                        Get_dt_rho0_sgy(),
                                                        Get_dt_rho0_sgz());
  }
}// end of Calculate_p0_source
//------------------------------------------------------------------------------


/**
 * Compute c^2 on the CPU side.
 *
 */
void TKSpaceFirstOrder3DSolver::Compute_c2()
{
  if (Parameters.Get_c0_scalar_flag())
  { // scalar
      float c = Parameters.Get_c0_scalar();
      Parameters.Get_c0_scalar() = c * c;
  }
  else
  { // matrix
    float * c2 =  Get_c2().GetRawData();

    #pragma omp parallel for schedule (static)
    for (size_t i=0; i < Get_c2().GetElementCount(); i++)
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

  /// calculate duxyz on uniform grid
  SolverCUDAKernels::Compute_duxyz_uniform(Get_CUFFT_X_temp(),
                                           Get_CUFFT_Y_temp(),
                                           Get_CUFFT_Z_temp(),
                                           Get_kappa(),
                                           Get_ddx_k_shift_neg(),
                                           Get_ddy_k_shift_neg(),
                                           Get_ddz_k_shift_neg());

  Get_CUFFT_X_temp().Compute_FFT_3D_C2R(Get_duxdx());
  Get_CUFFT_Y_temp().Compute_FFT_3D_C2R(Get_duydy());
  Get_CUFFT_Z_temp().Compute_FFT_3D_C2R(Get_duzdz());

  //-----------------------------------------------------------------------//
  //--------------------- Non-uniform grid ---------------------------------//
  //-----------------------------------------------------------------------//
  if (Parameters.Get_nonuniform_grid_flag() != 0)
  {
    SolverCUDAKernels::Compute_duxyz_non_uniform(Get_duxdx(),
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
  if (Parameters.Get_rho0_scalar())
  {
    SolverCUDAKernels::Compute_rhoxyz_nonlinear_homogeneous(Get_rhox(),
                                                            Get_rhoy(),
                                                            Get_rhoz(),
                                                            Get_pml_x(),
                                                            Get_pml_y(),
                                                            Get_pml_z(),
                                                            Get_duxdx(),
                                                            Get_duydy(),
                                                            Get_duzdz());
}
else
{
  // rho0 is a matrix
  SolverCUDAKernels::Compute_rhoxyz_nonlinear_heterogeneous(Get_rhox(),
                                                            Get_rhoy(),
                                                            Get_rhoz(),
                                                            Get_pml_x(),
                                                            Get_pml_y(),
                                                            Get_pml_z(),
                                                            Get_duxdx(),
                                                            Get_duydy(),
                                                            Get_duzdz(),
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
  if (Parameters.Get_rho0_scalar())
  {
    SolverCUDAKernels::Compute_rhoxyz_linear_homogeneous(Get_rhox(),
                                                         Get_rhoy(),
                                                         Get_rhoz(),
                                                         Get_pml_x(),
                                                         Get_pml_y(),
                                                         Get_pml_z(),
                                                         Get_duxdx(),
                                                         Get_duydy(),
                                                         Get_duzdz());
  }
  else
  {
    // rho0 is a matrix
    SolverCUDAKernels::Compute_rhoxyz_linear_heterogeneous(Get_rhox(),
                                                           Get_rhoy(),
                                                           Get_rhoz(),
                                                           Get_pml_x(),
                                                           Get_pml_y(),
                                                           Get_pml_z(),
                                                           Get_duxdx(),
                                                           Get_duydy(),
                                                           Get_duzdz(),
                                                           Get_rho0());
  } // end matrix
}// end of Compute_rhoxyz
//------------------------------------------------------------------------------

/*
 * Calculate two temporary sums in the new pressure formula,
 * linear absorbing case.
 * @param [out] Sum_rhoxyz    -rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [out] Sum_rho0_du   - rho0* (duxdx + duydy + duzdz);
 * @todo needs a bit of polishing
 */
void TKSpaceFirstOrder3DSolver::Calculate_SumRho_SumRhoDu(TRealMatrix& Sum_rhoxyz,
                                                          TRealMatrix& Sum_rho0_du)
{

  const bool   Is_rho0_scalar = Parameters.Get_rho0_scalar();
  const float * rho0_matrix = (Is_rho0_scalar) ? nullptr : Get_rho0().GetRawDeviceData();
  SolverCUDAKernels::Calculate_SumRho_SumRhoDu(Sum_rhoxyz,
                                               Sum_rho0_du,
                                               Get_rhox(),
                                               Get_rhoy(),
                                               Get_rhoz(),
                                               Get_duxdx(),
                                               Get_duydy(),
                                               Get_duzdz(),
                                               Is_rho0_scalar,
                                               rho0_matrix);
}// end of Calculate_SumRho_SumRhoDu
//------------------------------------------------------------------------------

/*
 * Sum sub-terms to calculate new pressure, non-linear case.
 * @param [in] Absorb_tau_temp  -
 * @param [in] Absorb_eta_temp  -   BonA + rho ^2 / 2 rho0  +
 *                                      (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [in] BonA_temp        -   rho0* (duxdx + duydy + duzdz)
 * @tode revise this routine
 */
void TKSpaceFirstOrder3DSolver::Sum_Subterms_nonlinear(TRealMatrix& Absorb_tau_temp,
                                                       TRealMatrix& Absorb_eta_temp,
                                                       TRealMatrix& BonA_temp)
{
  const bool  Is_c2_scalar      = Parameters.Get_c0_scalar_flag();
  const bool  Is_tau_eta_scalar = Parameters.Get_c0_scalar_flag() && Parameters.Get_alpha_coeff_scalar_flag();

  const float* c2_data_matrix  = (Is_c2_scalar)      ? nullptr : Get_c2().GetRawDeviceData();
  const float* tau_data_matrix = (Is_tau_eta_scalar) ? nullptr : Get_absorb_tau().GetRawDeviceData();
  const float* eta_data_matrix = (Is_tau_eta_scalar) ? nullptr : Get_absorb_eta().GetRawDeviceData();

  const float* Absorb_tau_data = Absorb_tau_temp.GetRawDeviceData();
  const float* Absorb_eta_data = Absorb_eta_temp.GetRawDeviceData();

  SolverCUDAKernels::Sum_Subterms_nonlinear(Get_p(),
                                            BonA_temp,
                                            Is_c2_scalar,
                                            c2_data_matrix,
                                            Is_tau_eta_scalar,
                                            Absorb_tau_data,
                                            tau_data_matrix,
                                            Absorb_eta_data,
                                            eta_data_matrix);
}// end of Sum_Subterms_nonlinear
//------------------------------------------------------------------------------

/*
 * Sum sub-terms to calculate new pressure, linear case.
 * @param [in] Absorb_tau_temp - sub-term with absorb_tau
 * @param [in] Absorb_eta_temp - sub-term with absorb_eta
 * @param [in] Sum_rhoxyz      - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @todo need to be reviewed - constant memory
 */
void TKSpaceFirstOrder3DSolver::Sum_Subterms_linear(TRealMatrix& Absorb_tau_temp,
                                                    TRealMatrix& Absorb_eta_temp,
                                                    TRealMatrix& Sum_rhoxyz)
{
  const bool  Is_c2_scalar      = Parameters.Get_c0_scalar_flag();
  const bool  Is_tau_eta_scalar = Parameters.Get_c0_scalar_flag() && Parameters.Get_alpha_coeff_scalar_flag();

  const float* c2_data_matrix  = (Is_c2_scalar)      ? nullptr : Get_c2().GetRawDeviceData();
  const float* tau_data_matrix = (Is_tau_eta_scalar) ? nullptr : Get_absorb_tau().GetRawDeviceData();
  const float* eta_data_matrix = (Is_tau_eta_scalar) ? nullptr : Get_absorb_eta().GetRawDeviceData();

  SolverCUDAKernels::Sum_Subterms_linear(Get_p(),
                                         Absorb_tau_temp,
                                         Absorb_eta_temp,
                                         Sum_rhoxyz,
                                         Is_c2_scalar,
                                         c2_data_matrix,
                                         Is_tau_eta_scalar,
                                         tau_data_matrix,
                                         eta_data_matrix);
}// end of Sum_Subterms_linear
//------------------------------------------------------------------------------

/*
 * Sum sub-terms for new p, non-linear lossless case.
 *
 * @todo this needs revision
 */
void TKSpaceFirstOrder3DSolver::Sum_new_p_nonlinear_lossless()
{
  const bool   Is_c2_scalar   = Parameters.Get_c0_scalar_flag();
  const bool   Is_BonA_scalar = Parameters.Get_BonA_scalar_flag();
  const bool   Is_rho0_scalar = Parameters.Get_rho0_scalar_flag();

  const float* c2_data_matrix   = (Is_c2_scalar)   ? nullptr : Get_c2().GetRawDeviceData();
  const float* BonA_data_matrix = (Is_BonA_scalar) ? nullptr : Get_BonA().GetRawDeviceData();
  const float* rho0_data_matrix = (Is_rho0_scalar) ? nullptr : Get_rho0().GetRawDeviceData();

  SolverCUDAKernels::Sum_new_p_nonlinear_lossless(Get_p(),
                                                  Get_rhox(),
                                                  Get_rhoy(),
                                                  Get_rhoz(),
                                                  Is_c2_scalar,
                                                  c2_data_matrix,
                                                  Is_BonA_scalar,
                                                  BonA_data_matrix,
                                                  Is_rho0_scalar,
                                                  rho0_data_matrix);

}// end of Sum_new_p_nonlinear_lossless
//------------------------------------------------------------------------------


/*
 * Sum sub-terms for new p, linear lossless case.
 */
void TKSpaceFirstOrder3DSolver::Sum_new_p_linear_lossless()
{
  const float   Is_c2_scalar =  Parameters.Get_c0_scalar();
  const float * c2_matrix = (Is_c2_scalar) ? nullptr : Get_c2().GetRawDeviceData();

  SolverCUDAKernels::Sum_new_p_linear_lossless(Get_p(),
                                               Get_rhox(),
                                               Get_rhoy(),
                                               Get_rhoz(),
                                               Is_c2_scalar,
                                               c2_matrix);

}// end of Sum_new_p_linear_lossless
//------------------------------------------------------------------------------


/*
 * Calculate three temporary sums in the new pressure formula
 * non-linear absorbing case.
 * @param [out] RHO_Temp  - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [out] BonA_Temp - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [out] Sum_du    - rho0* (duxdx + duydy + duzdz)
 *
 * @todo revise this method.
 */
void TKSpaceFirstOrder3DSolver::Calculate_SumRho_BonA_SumDu(TRealMatrix& RHO_Temp,
                                                            TRealMatrix& BonA_Temp,
                                                            TRealMatrix& Sum_du)
{
  const bool Is_BonA_scalar = Parameters.Get_BonA_scalar_flag();
  const bool Is_rho0_scalar = Parameters.Get_rho0_scalar_flag();

  const float* BonA_data = (Is_BonA_scalar) ? nullptr : Get_BonA().GetRawDeviceData();
  const float* rho0_data = (Is_rho0_scalar) ? nullptr : Get_rho0().GetRawDeviceData();

  SolverCUDAKernels::Calculate_SumRho_BonA_SumDu(RHO_Temp,
                                                 BonA_Temp,
                                                 Sum_du,
                                                 Get_rhox(),
                                                 Get_rhoy(),
                                                 Get_rhoz(),
                                                 Get_duxdx(),
                                                 Get_duydy(),
                                                 Get_duzdz(),
                                                 Is_BonA_scalar,
                                                 BonA_data,
                                                 Is_rho0_scalar,
                                                 rho0_data);

}// end of Calculate_SumRho_BonA_SumDu
//------------------------------------------------------------------------------

/**
 * Compute new p for non-linear case.
 */
void TKSpaceFirstOrder3DSolver::Compute_new_p_nonlinear()
{
  if (Parameters.Get_absorbing_flag())
  { // absorbing case
    TRealMatrix& Sum_rhoxyz      = Get_Temp_1_RS3D();
    TRealMatrix& BonA_rho_rhoxyz = Get_Temp_2_RS3D();
    TRealMatrix& Sum_du          = Get_Temp_3_RS3D();

    TRealMatrix& Absorb_tau_temp = Sum_du;
    TRealMatrix& Absorb_eta_temp = Sum_rhoxyz;

    Calculate_SumRho_BonA_SumDu(Sum_rhoxyz,BonA_rho_rhoxyz, Sum_du);

    Get_CUFFT_X_temp().Compute_FFT_3D_R2C(Sum_du);
    Get_CUFFT_Y_temp().Compute_FFT_3D_R2C(Sum_rhoxyz);


    SolverCUDAKernels::Compute_Absorb_nabla1_2(Get_CUFFT_X_temp(),
                                               Get_CUFFT_Y_temp(),
                                               Get_absorb_nabla1(),
                                               Get_absorb_nabla2());


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
  if (Parameters.Get_absorbing_flag())
  { // absorbing case
    TRealMatrix& Sum_rhoxyz  = Get_Temp_1_RS3D();
    TRealMatrix& Sum_rho0_du = Get_Temp_2_RS3D();

    TRealMatrix& Absorb_tau_temp = Get_Temp_2_RS3D();
    TRealMatrix& Absorb_eta_temp = Get_Temp_3_RS3D();

    Calculate_SumRho_SumRhoDu(Sum_rhoxyz,Sum_rho0_du);


    // ifftn ( absorb_nabla1 * fftn (rho0 * (duxdx+duydy+duzdz))
    Get_CUFFT_X_temp().Compute_FFT_3D_R2C(Sum_rho0_du);
    Get_CUFFT_Y_temp().Compute_FFT_3D_R2C(Sum_rhoxyz);

    SolverCUDAKernels::Compute_Absorb_nabla1_2(Get_CUFFT_X_temp(),
                                               Get_CUFFT_Y_temp(),
                                               Get_absorb_nabla1(),
                                               Get_absorb_nabla2());

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

  Get_CUFFT_X_temp().Compute_FFT_3D_R2C(Get_p());

  SolverCUDAKernels::Compute_ddx_kappa_fft_p(Get_CUFFT_X_temp(),
                                             Get_CUFFT_Y_temp(),
                                             Get_CUFFT_Z_temp(),
                                             Get_kappa(),
                                             Get_ddx_k_shift_pos(),
                                             Get_ddy_k_shift_pos(),
                                             Get_ddz_k_shift_pos());

  Get_CUFFT_X_temp().Compute_FFT_3D_C2R(Get_Temp_1_RS3D());
  Get_CUFFT_Y_temp().Compute_FFT_3D_C2R(Get_Temp_2_RS3D());
  Get_CUFFT_Z_temp().Compute_FFT_3D_C2R(Get_Temp_3_RS3D());

  if (Parameters.Get_rho0_scalar_flag())
  { // scalars
    if (Parameters.Get_nonuniform_grid_flag())
    {
      SolverCUDAKernels::Compute_uxyz_normalize_scalar_nonuniform(Get_ux_sgx(),
                                                                  Get_uy_sgy(),
                                                                  Get_uz_sgz(),
                                                                  Get_Temp_1_RS3D(),
                                                                  Get_Temp_2_RS3D(),
                                                                  Get_Temp_3_RS3D(),
                                                                  Get_dxudxn_sgx(),
                                                                  Get_dyudyn_sgy(),
                                                                  Get_dzudzn_sgz(),
                                                                  Get_pml_x_sgx(),
                                                                  Get_pml_y_sgy(),
                                                                  Get_pml_z_sgz());
    }
    else
    {
      SolverCUDAKernels::Compute_uxyz_normalize_scalar_uniform(Get_ux_sgx(),
                                                               Get_uy_sgy(),
                                                               Get_uz_sgz(),
                                                               Get_Temp_1_RS3D(),
                                                               Get_Temp_2_RS3D(),
                                                               Get_Temp_3_RS3D(),
                                                               Get_pml_x_sgx(),
                                                               Get_pml_y_sgy(),
                                                               Get_pml_z_sgz());
    }
  }
  else
  {// matrices
    SolverCUDAKernels::Compute_uxyz_normalize(Get_ux_sgx(),
                                              Get_uy_sgy(),
                                              Get_uz_sgz(),
                                              Get_Temp_1_RS3D(),
                                              Get_Temp_2_RS3D(),
                                              Get_Temp_3_RS3D(),
                                              Get_dt_rho0_sgx(),
                                              Get_dt_rho0_sgy(),
                                              Get_dt_rho0_sgz(),
                                              Get_pml_x_sgx(),
                                              Get_pml_y_sgy(),
                                              Get_pml_z_sgz());
  }
}// end of Compute_uxyz()
//------------------------------------------------------------------------------

/**
 * Add u source to the particle velocity.
 */
void TKSpaceFirstOrder3DSolver::Add_u_source()
{
  size_t t_index = Parameters.Get_t_index();

  if (Parameters.Get_ux_source_flag() > t_index)
  {
    SolverCUDAKernels::Add_u_source(Get_ux_sgx(),
                                    Get_ux_source_input(),
                                    Get_u_source_index(),
                                    t_index);
  }
  if (Parameters.Get_uy_source_flag() > t_index)
  {
    SolverCUDAKernels::Add_u_source(Get_uy_sgy(),
                                    Get_uy_source_input(),
                                    Get_u_source_index(),
                                    t_index);
  }
  if (Parameters.Get_uz_source_flag() > t_index)
  {
    SolverCUDAKernels::Add_u_source(Get_uz_sgz(),
                                    Get_uz_source_input(),
                                    Get_u_source_index(),
                                    t_index);
  }
}// end of Add_u_source
//------------------------------------------------------------------------------

/*
 * Add in pressure source.
 */
void TKSpaceFirstOrder3DSolver::Add_p_source()
{
  size_t t_index = Parameters.Get_t_index();

  if (Parameters.Get_p_source_flag() > t_index)
  {
    SolverCUDAKernels::Add_p_source(Get_rhox(),
                                    Get_rhoy(),
                                    Get_rhoz(),
                                    Get_p_source_input(),
                                    Get_p_source_index(),
                                    t_index);
  }// if do at all
}// end of Add_p_source
//------------------------------------------------------------------------------


/**
 * Calculated shifted velocities.
 * \n
 * ux_shifted = real(ifft(bsxfun(\@times, x_shift_neg, fft(ux_sgx, [], 1)), [], 1)); \n
 * uy_shifted = real(ifft(bsxfun(\@times, y_shift_neg, fft(uy_sgy, [], 2)), [], 2)); \n
 * uz_shifted = real(ifft(bsxfun(\@times, z_shift_neg, fft(uz_sgz, [], 3)), [], 3)); \n
 */

void TKSpaceFirstOrder3DSolver::Calculate_shifted_velocity()
{
  // ux_shifted
  Get_CUFFT_shift_temp().Compute_FFT_1DX_R2C(Get_ux_sgx());
  SolverCUDAKernels::ComputeVelocityShiftInX(Get_CUFFT_shift_temp(), Get_x_shift_neg_r());
  Get_CUFFT_shift_temp().Compute_FFT_1DX_C2R(Get_ux_shifted());

  // uy_shifted
  Get_CUFFT_shift_temp().Compute_FFT_1DY_R2C(Get_uy_sgy());
  SolverCUDAKernels::ComputeVelocityShiftInY(Get_CUFFT_shift_temp(), Get_y_shift_neg_r());
  Get_CUFFT_shift_temp().Compute_FFT_1DY_C2R(Get_uy_shifted());

  // uz_shifted
  Get_CUFFT_shift_temp().Compute_FFT_1DZ_R2C(Get_uz_sgz());
  SolverCUDAKernels::ComputeVelocityShiftInZ(Get_CUFFT_shift_temp(), Get_z_shift_neg_r());
  Get_CUFFT_shift_temp().Compute_FFT_1DZ_C2R(Get_uz_shifted());

}// end of Calculate_shifted_velocity
//------------------------------------------------------------------------------


/**
 * Compute the main time loop of KSpaceFirstOrder3D.
 */
void TKSpaceFirstOrder3DSolver::ComputeMainLoop()
{
  ActPercent = 0;

  // if resuming from a checkpoint,
  // set ActPercent to correspond the t_index after recovery
  if (Parameters.Get_t_index() > 0)
  {
    // We're restarting after checkpoint
    IsTimestepRightAfterRestore = true;
    ActPercent = (Parameters.Get_t_index() / (Parameters.Get_nt() / 100));
  }

  // Progress header
  TLogger::Log(TLogger::BASIC,OUT_FMT_SIMULATION_HEADER);

  // Initial copy of data to the GPU
  MatrixContainer.CopyMatricesToDevice();

  IterationTime.Start();

  // execute main loop
  while (Parameters.Get_t_index() < Parameters.Get_nt() && (!IsTimeToCheckpoint()))
  {
    const size_t t_index = Parameters.Get_t_index();

    Compute_uxyz();

    // add in the velocity u source term
    Add_u_source();

    // add in the transducer source term (t = t1) to ux
    if (Parameters.Get_transducer_source_flag() > t_index)
    {
      SolverCUDAKernels::AddTransducerSource(Get_ux_sgx(),
                                             Get_u_source_index(),
                                             Get_delay_mask(),
                                             Get_transducer_source_input());
    }

    Compute_duxyz();

    if (Parameters.Get_nonlinear_flag())
    {
      Compute_rhoxyz_nonlinear();
    }
    else
    {
      Compute_rhoxyz_linear();
    }


    // add in the source pressure term
    Add_p_source();

    if (Parameters.Get_nonlinear_flag())
    {
      Compute_new_p_nonlinear();
    }
    else
    {
     Compute_new_p_linear();
    }


    //-- calculate initial pressure
    if ((t_index == 0) && (Parameters.Get_p0_source_flag() == 1))  Calculate_p0_source();



    StoreSensorData();
    PrintStatistics();

    Parameters.Increment_t_index();
    IsTimestepRightAfterRestore = false;
  }

    // Since disk operations are one step delayed, we have to do the last one here.
    // However we need to check if the loop wasn't skipped due to very short checkpoint interval
    if (Parameters.Get_t_index() > Parameters.GetStartTimeIndex() && (!IsTimestepRightAfterRestore))
    {
      OutputStreamContainer.FlushRawStreams();
    }
}// end of ComputeMainLoop()
//------------------------------------------------------------------------------

/**
 * Print progress statistics.
 */
void TKSpaceFirstOrder3DSolver::PrintStatistics()
{
  const float  Nt = (float) Parameters.Get_nt();
  const size_t t_index = Parameters.Get_t_index();


  if (t_index > (ActPercent * Nt * 0.01f) )
  {
    ActPercent += Parameters.GetProgressPrintInterval();

    IterationTime.Stop();

    const double ElTime = IterationTime.GetElapsedTime();
    const double ElTimeWithLegs = IterationTime.GetElapsedTime() + SimulationTime.GetCumulatedElapsedTimeOverPreviousLegs();
    const double ToGo   = ((ElTimeWithLegs / (float) (t_index + 1)) *  Nt) - ElTimeWithLegs;

    struct tm *current;
    time_t now;
    time(&now);
    now += ToGo;
    current = localtime(&now);

    TLogger::Log(TLogger::BASIC,
                 OUT_FMT_SIMULATION_PROGRESS,
                 (size_t) ((t_index) / (Nt * 0.01f)),'%',
                 ElTime, ToGo,
                 current->tm_mday, current->tm_mon+1, current->tm_year-100,
                 current->tm_hour, current->tm_min, current->tm_sec);
    TLogger::Flush(TLogger::BASIC);
  }
}// end of KSpaceFirstOrder3DSolver
//------------------------------------------------------------------------------


/**
 * Is time to checkpoint?
 * @return true if it is necessary to stop to checkpoint
 */
bool TKSpaceFirstOrder3DSolver::IsTimeToCheckpoint()
{
  if (!Parameters.IsCheckpointEnabled()) return false;

  TotalTime.Stop();

  return (TotalTime.GetElapsedTime() > static_cast<float>(Parameters.GetCheckpointInterval()));
}// end of IsTimeToCheckpoint
//------------------------------------------------------------------------------

/*
 * Post processing, and closing the output streams.
 */
void TKSpaceFirstOrder3DSolver::PostProcessing()
{
  if (Parameters.IsStore_p_final())
  {
    Get_p().CopyFromDevice();
    Get_p().WriteDataToHDF5File(Parameters.GetOutputFile(),
                                p_final_NAME,
                                Parameters.GetCompressionLevel());
  }// p_final

  if (Parameters.IsStore_u_final())
  {
    Get_ux_sgx().CopyFromDevice();
    Get_uy_sgy().CopyFromDevice();
    Get_uz_sgz().CopyFromDevice();

    Get_ux_sgx().WriteDataToHDF5File(Parameters.GetOutputFile(),
                                     ux_final_NAME,
                                     Parameters.GetCompressionLevel());
    Get_uy_sgy().WriteDataToHDF5File(Parameters.GetOutputFile(),
                                     uy_final_NAME,
                                     Parameters.GetCompressionLevel());
    Get_uz_sgz().WriteDataToHDF5File(Parameters.GetOutputFile(),
                                     uz_final_NAME,
                                     Parameters.GetCompressionLevel());
  }// u_final

  // Apply post-processing, flush data on disk/
  OutputStreamContainer.PostProcessStreams();
  OutputStreamContainer.CloseStreams();

  // store sensor mask if wanted
  if (Parameters.IsCopySensorMask())
  {
    if (Parameters.Get_sensor_mask_type() == TParameters::INDEX)
    {
      Get_sensor_mask_index().RecomputeIndicesToMatlab();
      Get_sensor_mask_index().WriteDataToHDF5File(Parameters.GetOutputFile(),sensor_mask_index_NAME,
                                                  Parameters.GetCompressionLevel());
    }

    if (Parameters.Get_sensor_mask_type() == TParameters::CORNERS)
    {
      Get_sensor_mask_corners().RecomputeIndicesToMatlab();
      Get_sensor_mask_corners().WriteDataToHDF5File(Parameters.GetOutputFile(),sensor_mask_corners_NAME,
                                                    Parameters.GetCompressionLevel());
    }
  }
}// end of PostProcessing
//------------------------------------------------------------------------------

/**
 * Store sensor data.
 * This routine exploits asynchronous behavior. It first performs IO from the i-1th
 * step while waiting for ith step to come to the point of sampling.
 *
 */
void TKSpaceFirstOrder3DSolver::StoreSensorData()
{
  // Unless the time for sampling has come, exit.
  if (Parameters.Get_t_index() >= Parameters.GetStartTimeIndex())
  {

    // Read event for t_index-1. If sampling did not occur by then, ignored it.
    // if it did store data on disk (flush) - the GPU is running asynchronously.
    // But be careful, flush has to be one step delayed to work correctly.
    // when restoring from checkpoint we have to skip the first flush
    if (Parameters.Get_t_index() > Parameters.GetStartTimeIndex() && !IsTimestepRightAfterRestore)
    {
      OutputStreamContainer.FlushRawStreams();
    }

    // if --u_non_staggered is switched on, calculate unstaggered velocity.
    if (Parameters.IsStore_u_non_staggered_raw())
    {
      Calculate_shifted_velocity();
    }

    // Sample data for step t  (store event for sampling in next turn)
    OutputStreamContainer.SampleStreams();

    // the last step (or data after) checkpoint are flushed in the main loop
  }
}// end of StoreSensorData
//------------------------------------------------------------------------------

/**
 * Save checkpoint data into the checkpoint file, flush aggregated outputs into
 * the output file.
 */
void TKSpaceFirstOrder3DSolver::SaveCheckpointData()
{
  // Create Checkpoint file
  THDF5_File & HDF5_CheckpointFile = Parameters.GetCheckpointFile();
  // if it happens and the file is opened (from the recovery, close it)
  if (HDF5_CheckpointFile.IsOpen()) HDF5_CheckpointFile.Close();

  TLogger::Log(TLogger::FULL,OUT_FMT_STORING_CHECKPOINT_DATA);
  TLogger::Flush(TLogger::FULL);

  // Create the new file (overwrite the old one)
  HDF5_CheckpointFile.Create(Parameters.GetCheckpointFileName().c_str());

  //--------------------- Store Matrices ------------------------------//

  // Store all necessary matrices in Checkpoint file
  MatrixContainer.StoreDataIntoCheckpointFile(HDF5_CheckpointFile);
  // Write t_index
  HDF5_CheckpointFile.WriteScalarValue(HDF5_CheckpointFile.GetRootGroup(),
                                       t_index_NAME,
                                       Parameters.Get_t_index());

  // store basic dimension sizes (Nx, Ny, Nz) - Nt is not necessary
  HDF5_CheckpointFile.WriteScalarValue(HDF5_CheckpointFile.GetRootGroup(),
                                       Nx_NAME,
                                       Parameters.GetFullDimensionSizes().nx);
  HDF5_CheckpointFile.WriteScalarValue(HDF5_CheckpointFile.GetRootGroup(),
                                       Ny_NAME,
                                       Parameters.GetFullDimensionSizes().ny);
  HDF5_CheckpointFile.WriteScalarValue(HDF5_CheckpointFile.GetRootGroup(),
                                       Nz_NAME,
                                       Parameters.GetFullDimensionSizes().nz);

  // Write checkpoint file header
  THDF5_FileHeader CheckpointFileHeader = Parameters.GetFileHeader();

  CheckpointFileHeader.SetFileType(THDF5_FileHeader::CHECKPOINT);
  CheckpointFileHeader.SetCodeName(GetCodeName());
  CheckpointFileHeader.SetActualCreationTime();

  CheckpointFileHeader.WriteHeaderToCheckpointFile(HDF5_CheckpointFile);

  HDF5_CheckpointFile.Close();
  TLogger::Log(TLogger::FULL, OUT_FMT_DONE);

  // checkpoint only if necessary (t_index > start_index), we're here one step ahead!
  if (Parameters.Get_t_index() > Parameters.GetStartTimeIndex())
  {
    TLogger::Log(TLogger::FULL,OUT_FMT_STORING_SENSOR_DATA);
    TLogger::Flush(TLogger::FULL);

    OutputStreamContainer.CheckpointStreams();

    TLogger::Log(TLogger::FULL, OUT_FMT_DONE);
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
  Parameters.GetOutputFile().WriteScalarValue(Parameters.GetOutputFile().GetRootGroup(),
                                               t_index_NAME,
                                               Parameters.Get_t_index());

  // Write scalars
  Parameters.SaveScalarsToFile(Parameters.GetOutputFile());
  THDF5_FileHeader & HDF5_FileHeader = Parameters.GetFileHeader();

  // Write File header
  HDF5_FileHeader.SetCodeName(GetCodeName());
  HDF5_FileHeader.SetMajorFileVersion();
  HDF5_FileHeader.SetMinorFileVersion();
  HDF5_FileHeader.SetActualCreationTime();
  HDF5_FileHeader.SetFileType(THDF5_FileHeader::OUTPUT);
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

  HDF5_FileHeader.WriteHeaderToOutputFile(Parameters.GetOutputFile());
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
  Parameters.GetFileHeader().GetExecutionTimes(ElapsedTotalTime,
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
  THDF5_FileHeader & OutputFileHeader = Parameters.GetFileHeader();
  THDF5_File       & OutputFile       = Parameters.GetOutputFile();

  // test file type
  if (OutputFileHeader.GetFileType() != THDF5_FileHeader::OUTPUT)
  {
    char ErrorMessage[256] = "";
    snprintf(ErrorMessage,
             256,
             ERR_FMT_BAD_OUTPUT_FILE_FORMAT,
             Parameters.GetOutputFileName().c_str());
    throw ios::failure(ErrorMessage);
  }

  // test file major version
  if (!OutputFileHeader.CheckMajorFileVersion())
  {
    char ErrorMessage[256] = "";
    snprintf(ErrorMessage,
             256,
             ERR_FMT_BAD_MAJOR_File_Version,
             Parameters.GetCheckpointFileName().c_str(),
             OutputFileHeader.GetCurrentHDF5_MajorVersion().c_str());
    throw ios::failure(ErrorMessage);
  }

  // test file minor version
  if (!OutputFileHeader.CheckMinorFileVersion())
  {
    char ErrorMessage[256] = "";
    snprintf(ErrorMessage,
             256,
             ERR_FMT_BAD_MINOR_FILE_VERSION,
             Parameters.GetCheckpointFileName().c_str(),
             OutputFileHeader.GetCurrentHDF5_MinorVersion().c_str());
    throw ios::failure(ErrorMessage);
  }


  // Check dimension sizes
  TDimensionSizes OutputDimSizes;
  OutputFile.ReadScalarValue(OutputFile.GetRootGroup(),
                             Nx_NAME,
                             OutputDimSizes.nx);

  OutputFile.ReadScalarValue(OutputFile.GetRootGroup(),
                             Ny_NAME,
                             OutputDimSizes.ny);

  OutputFile.ReadScalarValue(OutputFile.GetRootGroup(),
                             Nz_NAME,
                             OutputDimSizes.nz);

 if (Parameters.GetFullDimensionSizes() != OutputDimSizes)
 {
    char ErrorMessage[256] = "";
    snprintf(ErrorMessage,
             256,
             ERR_FMT_OUTPUT_DIMENSIONS_NOT_MATCH,
             OutputDimSizes.nx,
             OutputDimSizes.ny,
             OutputDimSizes.nz,
             Parameters.GetFullDimensionSizes().nx,
             Parameters.GetFullDimensionSizes().ny,
             Parameters.GetFullDimensionSizes().nz);

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
  THDF5_File &     HDF5_CheckpointFile = Parameters.GetCheckpointFile();

  CheckpointFileHeader.ReadHeaderFromCheckpointFile(HDF5_CheckpointFile);

  // test file type
  if (CheckpointFileHeader.GetFileType() != THDF5_FileHeader::CHECKPOINT)
  {
    char ErrorMessage[256] = "";
    snprintf(ErrorMessage,
             256,
             ERR_FMT_BAD_CHECKPOINT_FILE_FORMAT,
             Parameters.GetCheckpointFileName().c_str());
    throw ios::failure(ErrorMessage);
  }

  // test file major version
  if (!CheckpointFileHeader.CheckMajorFileVersion())
  {
    char ErrorMessage[256] = "";
    snprintf(ErrorMessage,
             256,
             ERR_FMT_BAD_MAJOR_File_Version,
             Parameters.GetCheckpointFileName().c_str(),
             CheckpointFileHeader.GetCurrentHDF5_MajorVersion().c_str());
    throw ios::failure(ErrorMessage);
  }

  // test file minor version
  if (!CheckpointFileHeader.CheckMinorFileVersion())
  {
    char ErrorMessage[256] = "";
    snprintf(ErrorMessage,
             256,
             ERR_FMT_BAD_MINOR_FILE_VERSION,
             Parameters.GetCheckpointFileName().c_str(),
             CheckpointFileHeader.GetCurrentHDF5_MinorVersion().c_str());
    throw ios::failure(ErrorMessage);
  }


  // Check dimension sizes
  TDimensionSizes CheckpointDimSizes;
  HDF5_CheckpointFile.ReadScalarValue(HDF5_CheckpointFile.GetRootGroup(),
                                      Nx_NAME,
                                      CheckpointDimSizes.nx);

  HDF5_CheckpointFile.ReadScalarValue(HDF5_CheckpointFile.GetRootGroup(),
                                      Ny_NAME,
                                      CheckpointDimSizes.ny);

  HDF5_CheckpointFile.ReadScalarValue(HDF5_CheckpointFile.GetRootGroup(),
                                      Nz_NAME,
                                      CheckpointDimSizes.nz);

 if (Parameters.GetFullDimensionSizes() != CheckpointDimSizes)
 {
    char ErrorMessage[256] = "";
    snprintf(ErrorMessage,
             256,
             ERR_FMT_CHECKPOINT_DIMENSIONS_NOT_MATCH,
             CheckpointDimSizes.nx,
             CheckpointDimSizes.ny,
             CheckpointDimSizes.nz,
             Parameters.GetFullDimensionSizes().nx,
             Parameters.GetFullDimensionSizes().ny,
             Parameters.GetFullDimensionSizes().nz);

   throw ios::failure(ErrorMessage);
 }
}// end of CheckCheckpointFile
//------------------------------------------------------------------------------