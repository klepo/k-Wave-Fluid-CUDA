/**
 * @file        KSpaceFirstOrder3DSolver.cpp
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the main class of the project responsible for the
 *              entire the entire 3D fluid simulation..
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        12 July     2012, 10:27 (created)\n
 *              27 July     2016, 10:37 (revised)
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

#include <limits>

#include <KSpaceSolver/KSpaceFirstOrder3DSolver.h>

#include <Logger/ErrorMessages.h>
#include <Logger/Logger.h>

#include <KSpaceSolver/SolverCUDAKernels.cuh>
#include <Containers/MatrixContainer.h>

using std::string;


//------------------------------------------------------------------------------------------------//
//------------------------------------------ CONSTANTS -------------------------------------------//
//------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Constructor of the class.
 */
TKSpaceFirstOrder3DSolver::TKSpaceFirstOrder3DSolver() :
        matrixContainer(), outputStreamContainer(), parameters(TParameters::GetInstance()),
        actPercent(0), isTimestepRightAfterRestore(false),
        totalTime(), preProcessingTime(), dataLoadTime(), simulationTime(),
        postProcessingTime(), iterationTime()
{
  totalTime.Start();

  //Switch off default HDF5 error messages
  H5Eset_auto(H5E_DEFAULT, NULL, NULL);
}// end of TKSpaceFirstOrder3DSolver
//--------------------------------------------------------------------------------------------------

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
}// end of ~TKSpaceFirstOrder3DSolver
//--------------------------------------------------------------------------------------------------


/**
 * The method allocates the matrix container and create all matrices and creates all output streams.
 */
void TKSpaceFirstOrder3DSolver::AllocateMemory()
{
  TLogger::Log(TLogger::BASIC, OUT_FMT_MEMORY_ALLOCATION);
  TLogger::Flush(TLogger::BASIC);

  // create container, then all matrices
  matrixContainer.AddMatrices();
  matrixContainer.CreateMatrices();

  // add output streams into container
  outputStreamContainer.AddStreams(matrixContainer);

  TLogger::Log(TLogger::BASIC, OUT_FMT_DONE);
}// end of AllocateMemory
//--------------------------------------------------------------------------------------------------

/*
 * The method frees all memory allocated by the class.
 */
void TKSpaceFirstOrder3DSolver::FreeMemory()
{
  matrixContainer.FreeMatrices();
  outputStreamContainer.FreeStreams();
}// end of FreeMemory
//--------------------------------------------------------------------------------------------------

/**
 * Load data from the input file provided by the parameter class and creates the output time
 * series streams.
 */
void TKSpaceFirstOrder3DSolver::LoadInputData()
{
  // Load data from disk
  TLogger::Log(TLogger::BASIC, OUT_FMT_DATA_LOADING);
  TLogger::Flush(TLogger::BASIC);

  dataLoadTime.Start();

  // open and load input file
  THDF5_File& inputFile      = parameters.GetInputFile(); // file is opened (in Parameters)
  THDF5_File& outputFile     = parameters.GetOutputFile();
  THDF5_File& checkpointFile = parameters.GetCheckpointFile();

  // Load data from disk
  TLogger::Log(TLogger::FULL, OUT_FMT_FINSIH_NO_DONE);
  TLogger::Log(TLogger::FULL, OUT_FMT_READING_INPUT_FILE);
  TLogger::Flush(TLogger::FULL);

  // load data from the input file
  matrixContainer.LoadDataFromInputFile(inputFile);

  // close the input file
  inputFile.Close();

  TLogger::Log(TLogger::FULL, OUT_FMT_DONE);

  // The simulation does not use check pointing or this is the first turn
  bool recoverFromCheckpoint = (parameters.IsCheckpointEnabled() &&
                                THDF5_File::IsHDF5(parameters.GetCheckpointFileName()));


  if (recoverFromCheckpoint)
  {
    //--------------------------- Read data from the checkpoint file -----------------------------//
    TLogger::Log(TLogger::FULL, OUT_FMT_READING_CHECKPOINT_FILE);
    TLogger::Flush(TLogger::FULL);

    // Open checkpoint file
    checkpointFile.Open(parameters.GetCheckpointFileName());

    // Check the checkpoint file
    CheckCheckpointFile();

    // read the actual value of t_index
    size_t new_t_index;
    checkpointFile.ReadScalarValue(checkpointFile.GetRootGroup(), t_index_NAME, new_t_index);
    parameters.Set_t_index(new_t_index);

    // Read necessary matrices from the checkpoint file
    matrixContainer.LoadDataFromCheckpointFile(checkpointFile);

    checkpointFile.Close();
    TLogger::Log(TLogger::FULL, OUT_FMT_DONE);

    //----------------------------- Read data from the output file -------------------------------//
    // Reopen output file for RW access
    TLogger::Log(TLogger::FULL, OUT_FMT_READING_OUTPUT_FILE);
    TLogger::Flush(TLogger::FULL);

    outputFile.Open(parameters.GetOutputFileName(), H5F_ACC_RDWR);

    //Read file header of the output file
    parameters.GetFileHeader().ReadHeaderFromOutputFile(outputFile);

    // Restore elapsed time
    LoadElapsedTimeFromOutputFile(outputFile);

    // Reopen streams
    outputStreamContainer.ReopenStreams();
    TLogger::Log(TLogger::FULL, OUT_FMT_DONE);
  }
  else
  {
    //-------------------------- First round of multi-leg simulation -----------------------------//
    // Create the output file
    TLogger::Log(TLogger::FULL, OUT_FMT_CREATING_OUTPUT_FILE);
    TLogger::Flush(TLogger::FULL);

    outputFile.Create(parameters.GetOutputFileName());
    TLogger::Log(TLogger::FULL, OUT_FMT_DONE);

    // Create the steams, link them with the sampled matrices
    // however DO NOT allocate memory!
    outputStreamContainer.CreateStreams();
  }

  dataLoadTime.Stop();

  if (TLogger::GetLevel() != TLogger::FULL)
  {
    TLogger::Log(TLogger::BASIC, OUT_FMT_DONE);
  }
}// end of LoadInputData
//--------------------------------------------------------------------------------------------------

/**
 * This method computes k-space First Order 3D simulation. It launches calculation on a given
 * dataset going through  FFT initialization, pre-processing, main loop and post-processing phases.
 *
 */
void TKSpaceFirstOrder3DSolver::Compute()
{
  preProcessingTime.Start();

  TLogger::Log(TLogger::BASIC, OUT_FMT_FFT_PLANS);
  TLogger::Flush(TLogger::BASIC);

  TCUDAParameters& cudaParameters = parameters.GetCudaParameters();

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

    preProcessingTime.Stop();
    // Set kernel configurations
    cudaParameters.SetKernelConfiguration();

    // Set up constant memory - copy over to GPU
    // Constant memory uses some variables calculated during preprocessing
    cudaParameters.SetUpDeviceConstants();

    TLogger::Log(TLogger::BASIC, OUT_FMT_DONE);
  }
  catch (const std::exception& e)
  {
    TLogger::Log(TLogger::BASIC, OUT_FMT_FAILED);
    TLogger::Log(TLogger::BASIC, OUT_FMT_LAST_SEPARATOR);

    TLogger::ErrorAndTerminate(TLogger::WordWrapString(e.what(),ERR_FMT_PATH_DELIMITERS, 9).c_str());
  }

  // Logger header for simulation
  TLogger::Log(TLogger::BASIC, OUT_FMT_ELAPSED_TIME, preProcessingTime.GetElapsedTime());
  TLogger::Log(TLogger::BASIC, OUT_FMT_COMP_RESOURCES_HEADER);
  TLogger::Log(TLogger::BASIC, OUT_FMT_CURRENT_HOST_MEMORY,   GetHostMemoryUsageInMB());
  TLogger::Log(TLogger::BASIC, OUT_FMT_CURRENT_DEVICE_MEMORY, GetDeviceMemoryUsageInMB());

  char GridDimensions[19];
  snprintf(GridDimensions, 19, OUT_FMT_CUDA_GRID_SHAPE_FORMAT,
           cudaParameters.GetSolverGridSize1D(),
           cudaParameters.GetSolverBlockSize1D());

  TLogger::Log(TLogger::FULL, OUT_FMT_CUDA_SOLVER_GRID_SHAPE, GridDimensions);

  snprintf(GridDimensions, 18, OUT_FMT_CUDA_GRID_SHAPE_FORMAT,
           cudaParameters.GetSamplerGridSize1D(),
           cudaParameters.GetSamplerBlockSize1D());

  TLogger::Log(TLogger::FULL, OUT_FMT_CUDA_SAMPLER_GRID_SHAPE, GridDimensions);

  // Main loop
  try
  {
    simulationTime.Start();

    ComputeMainLoop();

    simulationTime.Stop();

    TLogger::Log(TLogger::BASIC,OUT_FMT_SIMULATOIN_END_SEPARATOR);
  }
  catch (const std::exception& e)
  {
    TLogger::Log(TLogger::BASIC,
                 OUT_FMT_SIMULATION_FINAL_SEPARATOR,
                 GridDimensions);
    TLogger::ErrorAndTerminate(TLogger::WordWrapString(e.what(),ERR_FMT_PATH_DELIMITERS, 9).c_str());
  }

  // Post processing region
  postProcessingTime.Start();

  try
  {
    if (IsCheckpointInterruption())
    { // Checkpoint
      TLogger::Log(TLogger::BASIC, OUT_FMT_ELAPSED_TIME, simulationTime.GetElapsedTime());
      TLogger::Log(TLogger::BASIC, OUT_FMT_CHECKPOINT_TIME_STEPS, parameters.Get_t_index());
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

      TLogger::Log(TLogger::BASIC,  OUT_FMT_ELAPSED_TIME, simulationTime.GetElapsedTime());
      TLogger::Log(TLogger::BASIC,OUT_FMT_SEPARATOR);
      TLogger::Log(TLogger::BASIC, OUT_FMT_POST_PROCESSING);
      TLogger::Flush(TLogger::BASIC);

      PostProcessing();

      // if checkpointing is enabled and the checkpoint file was created in the past, delete it
      if (parameters.IsCheckpointEnabled())
      {
        std::remove(parameters.GetCheckpointFileName().c_str());
      }
      TLogger::Log(TLogger::BASIC, OUT_FMT_DONE);
    }
  }
  catch (const std::exception &e)
  {
    TLogger::Log(TLogger::BASIC, OUT_FMT_FAILED);
    TLogger::Log(TLogger::BASIC, OUT_FMT_LAST_SEPARATOR);

    TLogger::ErrorAndTerminate(TLogger::WordWrapString(e.what(),ERR_FMT_PATH_DELIMITERS,9).c_str());
  }
  postProcessingTime.Stop();

  // Final data written
  try
  {
    WriteOutputDataInfo();
    parameters.GetOutputFile().Close();

    TLogger::Log(TLogger::BASIC, OUT_FMT_ELAPSED_TIME, postProcessingTime.GetElapsedTime());
    }
  catch (const std::exception &e)
  {
    TLogger::Log(TLogger::BASIC, OUT_FMT_LAST_SEPARATOR);
    TLogger::ErrorAndTerminate(TLogger::WordWrapString(e.what(),ERR_FMT_PATH_DELIMITERS, 9).c_str());
  }
}// end of Compute()
//--------------------------------------------------------------------------------------------------

/**
 * Get peak CPU memory usage.
 *
 * @return peak memory usage in MBs.
 */
size_t TKSpaceFirstOrder3DSolver::GetHostMemoryUsageInMB()
{
  // Linux build
  #ifdef __linux__
    struct rusage memUsage;
    getrusage(RUSAGE_SELF, &memUsage);

    return memUsage.ru_maxrss >> 10;
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
}// end of GetHostMemoryUsageInMB
//--------------------------------------------------------------------------------------------------

/**
 * Get peak GPU memory usage.
 *
 * @return Peak memory usage in MBs.
 */
size_t TKSpaceFirstOrder3DSolver::GetDeviceMemoryUsageInMB()
{
  size_t free, total;
  cudaMemGetInfo(&free,&total);

  return ((total-free) >> 20);
}// end of GetDeviceMemoryUsageInMB
//--------------------------------------------------------------------------------------------------

/**
 * Get release code version.
 *
 * @return core name
 */
const string TKSpaceFirstOrder3DSolver::GetCodeName() const
{
  return string(OUT_FMT_KWAVE_VERSION);
}// end of GetCodeName
//--------------------------------------------------------------------------------------------------

/**
 * Print full code name and the license
 */
void TKSpaceFirstOrder3DSolver::PrintFullNameCodeAndLicense() const
{
  TLogger::Log(TLogger::BASIC,
               OUT_FMT_BUILD_NO_DATE_TIME,
               10,11,__DATE__,
               8,8,__TIME__);

  if (parameters.GetGitHash() != "")
  {
    TLogger::Log(TLogger::BASIC, OUT_FMT_VERSION_GIT_HASH, parameters.GetGitHash().c_str());
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
    TLogger::Log(TLogger::BASIC, OUT_FMT_GNU_COMPILER, __VERSION__);
  #endif
  #ifdef __INTEL_COMPILER
    TLogger::Log(TLogger::BASIC, OUT_FMT_INTEL_COMPILER, __INTEL_COMPILER);
  #endif
  #ifdef _MSC_VER
	TLogger::Log(TLogger::BASIC, OUT_FMT_VISUAL_STUDIO_COMPILER, _MSC_VER);
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

  const TCUDAParameters& cudaParameters = parameters.GetCudaParameters();
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
    TLogger::Log(TLogger::BASIC, OUT_FMT_CUDA_DEVICE, cudaParameters.GetDeviceIdx());

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
//--------------------------------------------------------------------------------------------------

 /**
  * Get total simulation time.
  *
  * @return  total simulation time.
  */
double TKSpaceFirstOrder3DSolver::GetTotalTime() const
{
  return totalTime.GetElapsedTime();
}// end of GetTotalTime()
//--------------------------------------------------------------------------------------------------

/**
 * Get pre-processing time.
 *
 * @return pre-processing time
 */
double TKSpaceFirstOrder3DSolver::GetPreProcessingTime() const
{
  return preProcessingTime.GetElapsedTime();
}// end of GetPreProcessingTime
//--------------------------------------------------------------------------------------------------

/**
 * Get data load time.
 *
 * @return  time to load data
 */
double TKSpaceFirstOrder3DSolver::GetDataLoadTime() const
{
  return dataLoadTime.GetElapsedTime();
}// end of GetDataLoadTime
//--------------------------------------------------------------------------------------------------


/**
 * Get simulation time (time loop).
 *
 * @return  simulation time
 */
double TKSpaceFirstOrder3DSolver::GetSimulationTime() const
{
  return simulationTime.GetElapsedTime();
}// end of GetSimulationTime
//--------------------------------------------------------------------------------------------------

/**
 * Get post-processing time.
 *
 * @return post-processing time
 */
double TKSpaceFirstOrder3DSolver::GetPostProcessingTime() const
{
  return postProcessingTime.GetElapsedTime();
}// end of GetPostProcessingTime
//--------------------------------------------------------------------------------------------------

/**
 * Get total simulation time cumulated over all legs.
 * @return  simulation time cumulated over multiple legs
 */
double TKSpaceFirstOrder3DSolver::GetCumulatedTotalTime() const
{
  return totalTime.GetCumulatedElapsedTimeOverAllLegs();
}// end of GetCumulatedTotalTime
//--------------------------------------------------------------------------------------------------

/**
 * Get pre-processing time cumulated over all legs.
 *
 * @return pre-processing time cumulated over multiple legs
 */
double TKSpaceFirstOrder3DSolver::GetCumulatedPreProcessingTime() const
{
  return preProcessingTime.GetCumulatedElapsedTimeOverAllLegs();
} // end of GetCumulatedPreProcessingTime
//--------------------------------------------------------------------------------------------------

/**
 * Get data load time cumulated over all legs.
 *
 * @return time to load data cumulated over multiple legs
 */
double TKSpaceFirstOrder3DSolver::GetCumulatedDataLoadTime() const
{
  return dataLoadTime.GetCumulatedElapsedTimeOverAllLegs();
}// end of GetCumulatedDataLoadTime
//--------------------------------------------------------------------------------------------------

/**
 * Get simulation time (time loop) cumulated over all legs.
 *
 * @return simulation time cumulated over multiple legs
 */
double TKSpaceFirstOrder3DSolver::GetCumulatedSimulationTime() const
{
  return simulationTime.GetCumulatedElapsedTimeOverAllLegs();
}// end of GetCumulatedSimulationTime
//--------------------------------------------------------------------------------------------------

/**
 * Get post-processing time cumulated over all legs.
 *
 * @return cumulated time to do post-processing over multiple legs
 */
double TKSpaceFirstOrder3DSolver::GetCumulatedPostProcessingTime() const
{
  return postProcessingTime.GetCumulatedElapsedTimeOverAllLegs();
}// end of GetCumulatedPostProcessingTime
//--------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//


/**
 * Initialize FFT plans.
 */
void TKSpaceFirstOrder3DSolver::InitializeFFTPlans()
{
  // create real to complex plans
  TCUFFTComplexMatrix::Create_FFT_Plan_3D_R2C(parameters.GetFullDimensionSizes());

 // create complex to real plans
  TCUFFTComplexMatrix::Create_FFT_Plan_3D_C2R(parameters.GetFullDimensionSizes());

  // if necessary, create 1D shift plans.
  // in this case, the matrix has a bit bigger dimensions to be able to store shifted matrices.
  if (TParameters::GetInstance().IsStore_u_non_staggered_raw())
  {
    // X shifts
    TCUFFTComplexMatrix::Create_FFT_Plan_1DX_R2C(parameters.GetFullDimensionSizes());
    TCUFFTComplexMatrix::Create_FFT_Plan_1DX_C2R(parameters.GetFullDimensionSizes());

    // Y shifts
    TCUFFTComplexMatrix::Create_FFT_Plan_1DY_R2C(parameters.GetFullDimensionSizes());
    TCUFFTComplexMatrix::Create_FFT_Plan_1DY_C2R(parameters.GetFullDimensionSizes());

    // Z shifts
    TCUFFTComplexMatrix::Create_FFT_Plan_1DZ_R2C(parameters.GetFullDimensionSizes());
    TCUFFTComplexMatrix::Create_FFT_Plan_1DZ_C2R(parameters.GetFullDimensionSizes());
  }// end u_non_staggered
}// end of InitializeFFTPlans
//--------------------------------------------------------------------------------------------------

/*
 * Compute pre-processing phase. \n
 * Initialize all indices, pre-compute constants such as c^2, rho0_sg* x dt  and create kappa,
 * absorb_eta, absorb_tau, absorb_nabla1, absorb_nabla2  matrices. Calculate this on the CPU side.
 */
void TKSpaceFirstOrder3DSolver::PreProcessingPhase()
{
  // get the correct sensor mask and recompute indices
  if (parameters.Get_sensor_mask_type() == TParameters::INDEX)
  {
    Get_sensor_mask_index().RecomputeIndicesToCPP();
  }

  if (parameters.Get_sensor_mask_type() == TParameters::CORNERS)
  {
    Get_sensor_mask_corners().RecomputeIndicesToCPP();
  }

  if ((parameters.Get_transducer_source_flag() != 0) ||
      (parameters.Get_ux_source_flag() != 0)         ||
      (parameters.Get_uy_source_flag() != 0)         ||
      (parameters.Get_uz_source_flag() != 0)
     )
  {
    Get_u_source_index().RecomputeIndicesToCPP();
  }

  if (parameters.Get_transducer_source_flag() != 0)
  {
    Get_delay_mask().RecomputeIndicesToCPP();
  }

  if (parameters.Get_p_source_flag() != 0)
  {
    Get_p_source_index().RecomputeIndicesToCPP();
  }

  // compute dt / rho0_sg...
  if (parameters.Get_rho0_scalar_flag())
  { // rho is scalar
    parameters.Get_rho0_sgx_scalar() = parameters.Get_dt() / parameters.Get_rho0_sgx_scalar();
    parameters.Get_rho0_sgy_scalar() = parameters.Get_dt() / parameters.Get_rho0_sgy_scalar();
    parameters.Get_rho0_sgz_scalar() = parameters.Get_dt() / parameters.Get_rho0_sgz_scalar();
  }
  else
  { // non-uniform grid cannot be pre-calculated :-(
    // rho is matrix
    if (parameters.Get_nonuniform_grid_flag())
    {
      GenerateInitialDenisty();
    }
    else
    {
      Get_dt_rho0_sgx().ScalarDividedBy(parameters.Get_dt());
      Get_dt_rho0_sgy().ScalarDividedBy(parameters.Get_dt());
      Get_dt_rho0_sgz().ScalarDividedBy(parameters.Get_dt());
    }
  }

  // generate different matrices
  if (parameters.Get_absorbing_flag() != 0)
  {
    GenerateKappaAndNablas();
    GenerateTauAndEta();
  }
  else
  {
    GenerateKappa();
  }

  // calculate c^2. It has to be after kappa gen... because of c modification
  Compute_c2();

}// end of PreProcessingPhase
//--------------------------------------------------------------------------------------------------

/**
 * Compute the main time loop of KSpaceFirstOrder3D.
 */
void TKSpaceFirstOrder3DSolver::ComputeMainLoop()
{
  actPercent = 0;

  // if resuming from a checkpoint,
  // set ActPercent to correspond the t_index after recovery
  if (parameters.Get_t_index() > 0)
  {
    // We're restarting after checkpoint
    isTimestepRightAfterRestore = true;
    actPercent = (parameters.Get_t_index() / (parameters.Get_nt() / 100));
  }

  // Progress header
  TLogger::Log(TLogger::BASIC,OUT_FMT_SIMULATION_HEADER);

  // Initial copy of data to the GPU
  matrixContainer.CopyMatricesToDevice();

  iterationTime.Start();

  // execute main loop
  while (parameters.Get_t_index() < parameters.Get_nt() && (!IsTimeToCheckpoint()))
  {
    const size_t t_index = parameters.Get_t_index();

    // compute velocity
    ComputeVelocity();

    // add in the velocity u source term
    AddVelocitySource();

    // add in the transducer source term (t = t1) to ux
    if (parameters.Get_transducer_source_flag() > t_index)
    {
      SolverCUDAKernels::AddTransducerSource(Get_ux_sgx(),
                                             Get_u_source_index(),
                                             Get_delay_mask(),
                                             Get_transducer_source_input());
    }

    // compute gradient of velocity
    ComputeGradientVelocity();

    // compute density
    if (parameters.Get_nonlinear_flag())
    {
      ComputeDensityNonliner();
    }
    else
    {
      ComputeDensityLinear();
    }

    // add in the source pressure term
    AddPressureSource();

    if (parameters.Get_nonlinear_flag())
    {
      ComputePressureNonlinear();
    }
    else
    {
      ComputePressureLinear();
    }

    //-- calculate initial pressure
    if ((t_index == 0) && (parameters.Get_p0_source_flag() == 1))  Calculate_p0_source();

    StoreSensorData();
    PrintStatistics();

    parameters.Increment_t_index();
    isTimestepRightAfterRestore = false;
  }

    // Since disk operations are one step delayed, we have to do the last one here.
    // However we need to check if the loop wasn't skipped due to very short checkpoint interval
    if (parameters.Get_t_index() > parameters.GetStartTimeIndex() && (!isTimestepRightAfterRestore))
    {
      outputStreamContainer.FlushRawStreams();
    }
}// end of ComputeMainLoop()
//--------------------------------------------------------------------------------------------------

/*
 * Post processing, and closing the output streams.
 */
void TKSpaceFirstOrder3DSolver::PostProcessing()
{
  if (parameters.IsStore_p_final())
  {
    Get_p().CopyFromDevice();
    Get_p().WriteDataToHDF5File(parameters.GetOutputFile(),
                                p_final_NAME,
                                parameters.GetCompressionLevel());
  }// p_final

  if (parameters.IsStore_u_final())
  {
    Get_ux_sgx().CopyFromDevice();
    Get_uy_sgy().CopyFromDevice();
    Get_uz_sgz().CopyFromDevice();

    Get_ux_sgx().WriteDataToHDF5File(parameters.GetOutputFile(),
                                     ux_final_NAME,
                                     parameters.GetCompressionLevel());
    Get_uy_sgy().WriteDataToHDF5File(parameters.GetOutputFile(),
                                     uy_final_NAME,
                                     parameters.GetCompressionLevel());
    Get_uz_sgz().WriteDataToHDF5File(parameters.GetOutputFile(),
                                     uz_final_NAME,
                                     parameters.GetCompressionLevel());
  }// u_final

  // Apply post-processing, flush data on disk/
  outputStreamContainer.PostProcessStreams();
  outputStreamContainer.CloseStreams();

  // store sensor mask if wanted
  if (parameters.IsCopySensorMask())
  {
    if (parameters.Get_sensor_mask_type() == TParameters::INDEX)
    {
      Get_sensor_mask_index().RecomputeIndicesToMatlab();
      Get_sensor_mask_index().WriteDataToHDF5File(parameters.GetOutputFile(),sensor_mask_index_NAME,
                                                  parameters.GetCompressionLevel());
    }

    if (parameters.Get_sensor_mask_type() == TParameters::CORNERS)
    {
      Get_sensor_mask_corners().RecomputeIndicesToMatlab();
      Get_sensor_mask_corners().WriteDataToHDF5File(parameters.GetOutputFile(),sensor_mask_corners_NAME,
                                                    parameters.GetCompressionLevel());
    }
  }
}// end of PostProcessing
//--------------------------------------------------------------------------------------------------

/**
 * Store sensor data.
 * This routine exploits asynchronous behavior. It first performs IO from the i-1th step while
 * waiting for ith step to come to the point of sampling.
 */
void TKSpaceFirstOrder3DSolver::StoreSensorData()
{
  // Unless the time for sampling has come, exit.
  if (parameters.Get_t_index() >= parameters.GetStartTimeIndex())
  {

    // Read event for t_index-1. If sampling did not occur by then, ignored it.
    // if it did store data on disk (flush) - the GPU is running asynchronously.
    // But be careful, flush has to be one step delayed to work correctly.
    // when restoring from checkpoint we have to skip the first flush
    if (parameters.Get_t_index() > parameters.GetStartTimeIndex() && !isTimestepRightAfterRestore)
    {
      outputStreamContainer.FlushRawStreams();
    }

    // if --u_non_staggered is switched on, calculate unstaggered velocity.
    if (parameters.IsStore_u_non_staggered_raw())
    {
      CalculateShiftedVelocity();
    }

    // Sample data for step t  (store event for sampling in next turn)
    outputStreamContainer.SampleStreams();

    // the last step (or data after) checkpoint are flushed in the main loop
  }
}// end of StoreSensorData
//--------------------------------------------------------------------------------------------------

/**
 * Write statistics and the header into the output file.
 */
void  TKSpaceFirstOrder3DSolver::WriteOutputDataInfo()
{
  // write t_index into the output file
  parameters.GetOutputFile().WriteScalarValue(parameters.GetOutputFile().GetRootGroup(),
                                              t_index_NAME,
                                              parameters.Get_t_index());

  // Write scalars
  parameters.SaveScalarsToFile(parameters.GetOutputFile());
  THDF5_FileHeader& fileHeader = parameters.GetFileHeader();

  // Write File header
  fileHeader.SetCodeName(GetCodeName());
  fileHeader.SetMajorFileVersion();
  fileHeader.SetMinorFileVersion();
  fileHeader.SetActualCreationTime();
  fileHeader.SetFileType(THDF5_FileHeader::OUTPUT);
  fileHeader.SetHostName();

  fileHeader.SetMemoryConsumption(GetHostMemoryUsageInMB());

  // Stop total timer here
  totalTime.Stop();
  fileHeader.SetExecutionTimes(GetTotalTime(),
                                    GetDataLoadTime(),
                                    GetPreProcessingTime(),
                                    GetSimulationTime(),
                                    GetPostProcessingTime());

  fileHeader.SetNumberOfCores();

  fileHeader.WriteHeaderToOutputFile(parameters.GetOutputFile());
}// end of WriteOutputDataInfo
//--------------------------------------------------------------------------------------------------

/**
 * Save checkpoint data into the checkpoint file, flush aggregated outputs into the output file.
 */
void TKSpaceFirstOrder3DSolver::SaveCheckpointData()
{
  // Create Checkpoint file
  THDF5_File& checkpointFile = parameters.GetCheckpointFile();
  // if it happens and the file is opened (from the recovery, close it)
  if (checkpointFile.IsOpen()) checkpointFile.Close();

  TLogger::Log(TLogger::FULL,OUT_FMT_STORING_CHECKPOINT_DATA);
  TLogger::Flush(TLogger::FULL);

  // Create the new file (overwrite the old one)
  checkpointFile.Create(parameters.GetCheckpointFileName());

  //-------------------------------------- Store Matrices ----------------------------------------//

  // Store all necessary matrices in Checkpoint file
  matrixContainer.StoreDataIntoCheckpointFile(checkpointFile);
  // Write t_index
  checkpointFile.WriteScalarValue(checkpointFile.GetRootGroup(),
                                  t_index_NAME,
                                  parameters.Get_t_index());

  // store basic dimension sizes (Nx, Ny, Nz) - Nt is not necessary
  checkpointFile.WriteScalarValue(checkpointFile.GetRootGroup(),
                                  Nx_NAME,
                                  parameters.GetFullDimensionSizes().nx);
  checkpointFile.WriteScalarValue(checkpointFile.GetRootGroup(),
                                  Ny_NAME,
                                  parameters.GetFullDimensionSizes().ny);
  checkpointFile.WriteScalarValue(checkpointFile.GetRootGroup(),
                                  Nz_NAME,
                                  parameters.GetFullDimensionSizes().nz);

  // Write checkpoint file header
  THDF5_FileHeader checkpointFileHeader = parameters.GetFileHeader();

  checkpointFileHeader.SetFileType(THDF5_FileHeader::CHECKPOINT);
  checkpointFileHeader.SetCodeName(GetCodeName());
  checkpointFileHeader.SetActualCreationTime();

  checkpointFileHeader.WriteHeaderToCheckpointFile(checkpointFile);

  checkpointFile.Close();
  TLogger::Log(TLogger::FULL, OUT_FMT_DONE);

  // checkpoint only if necessary (t_index > start_index), we're here one step ahead!
  if (parameters.Get_t_index() > parameters.GetStartTimeIndex())
  {
    TLogger::Log(TLogger::FULL,OUT_FMT_STORING_SENSOR_DATA);
    TLogger::Flush(TLogger::FULL);

    outputStreamContainer.CheckpointStreams();

    TLogger::Log(TLogger::FULL, OUT_FMT_DONE);
  }

  outputStreamContainer.CloseStreams();
}// end of SaveCheckpointData()
//--------------------------------------------------------------------------------------------------


/**
 * Compute new values of ux_sgx, uy_sgy, uz_sgz (acoustic velocity).
 */
void TKSpaceFirstOrder3DSolver::ComputeVelocity()
{
  Get_cufft_x_temp().Compute_FFT_3D_R2C(Get_p());

  SolverCUDAKernels::ComputePressurelGradient(Get_cufft_x_temp(),
                                              Get_cufft_y_temp(),
                                              Get_cufft_z_temp(),
                                              Get_kappa(),
                                              Get_ddx_k_shift_pos(),
                                              Get_ddy_k_shift_pos(),
                                              Get_ddz_k_shift_pos());

  Get_cufft_x_temp().Compute_FFT_3D_C2R(Get_temp_1_real_3D());
  Get_cufft_y_temp().Compute_FFT_3D_C2R(Get_temp_2_real_3D());
  Get_cufft_z_temp().Compute_FFT_3D_C2R(Get_temp_3_real_3D());

  if (parameters.Get_rho0_scalar_flag())
  { // scalars
    if (parameters.Get_nonuniform_grid_flag())
    {
      SolverCUDAKernels::ComputeVelocityScalarNonuniform(Get_ux_sgx(),
                                                         Get_uy_sgy(),
                                                         Get_uz_sgz(),
                                                         Get_temp_1_real_3D(),
                                                         Get_temp_2_real_3D(),
                                                         Get_temp_3_real_3D(),
                                                         Get_dxudxn_sgx(),
                                                         Get_dyudyn_sgy(),
                                                         Get_dzudzn_sgz(),
                                                         Get_pml_x_sgx(),
                                                         Get_pml_y_sgy(),
                                                         Get_pml_z_sgz());
    }
    else
    {
      SolverCUDAKernels::ComputeVelocityScalarUniform(Get_ux_sgx(),
                                                      Get_uy_sgy(),
                                                      Get_uz_sgz(),
                                                      Get_temp_1_real_3D(),
                                                      Get_temp_2_real_3D(),
                                                      Get_temp_3_real_3D(),
                                                      Get_pml_x_sgx(),
                                                      Get_pml_y_sgy(),
                                                      Get_pml_z_sgz());
    }
  }
  else
  {// matrices
    SolverCUDAKernels::ComputeVelocity(Get_ux_sgx(),
                                       Get_uy_sgy(),
                                       Get_uz_sgz(),
                                       Get_temp_1_real_3D(),
                                       Get_temp_2_real_3D(),
                                       Get_temp_3_real_3D(),
                                       Get_dt_rho0_sgx(),
                                       Get_dt_rho0_sgy(),
                                       Get_dt_rho0_sgz(),
                                       Get_pml_x_sgx(),
                                       Get_pml_y_sgy(),
                                       Get_pml_z_sgz());
  }
}// end of ComputeVelocity
//--------------------------------------------------------------------------------------------------


/**
 * Compute new values for duxdx, duydy, duzdz, gradient of velocity.
 */
void TKSpaceFirstOrder3DSolver::ComputeGradientVelocity()
{
  Get_cufft_x_temp().Compute_FFT_3D_R2C(Get_ux_sgx());
  Get_cufft_y_temp().Compute_FFT_3D_R2C(Get_uy_sgy());
  Get_cufft_z_temp().Compute_FFT_3D_R2C(Get_uz_sgz());

  // calculate duxyz on uniform grid
  SolverCUDAKernels::ComputeVelocityGradient(Get_cufft_x_temp(),
                                             Get_cufft_y_temp(),
                                             Get_cufft_z_temp(),
                                             Get_kappa(),
                                             Get_ddx_k_shift_neg(),
                                             Get_ddy_k_shift_neg(),
                                             Get_ddz_k_shift_neg());

  Get_cufft_x_temp().Compute_FFT_3D_C2R(Get_duxdx());
  Get_cufft_y_temp().Compute_FFT_3D_C2R(Get_duydy());
  Get_cufft_z_temp().Compute_FFT_3D_C2R(Get_duzdz());

  // Non-uniform grid
  if (parameters.Get_nonuniform_grid_flag() != 0)
  {
    SolverCUDAKernels::ComputeVelocityGradientNonuniform(Get_duxdx(),
                                                         Get_duydy(),
                                                         Get_duzdz(),
                                                         Get_dxudxn(),
                                                         Get_dyudyn(),
                                                         Get_dzudzn());
  }// nonlinear
}// end of ComputeGradientVelocity()
//--------------------------------------------------------------------------------------------------

/**
 * Calculate new values of rhox, rhoy and rhoz for non-linear case.
 */
void TKSpaceFirstOrder3DSolver::ComputeDensityNonliner()
{
  // Scalar
  if (parameters.Get_rho0_scalar())
  {
    SolverCUDAKernels::ComputeDensityNonlinearHomogeneous(Get_rhox(),
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
  SolverCUDAKernels::ComputeDensityNonlinearHeterogeneous(Get_rhox(),
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
}// end of ComputeDensityNonliner
//--------------------------------------------------------------------------------------------------

/**
 * Calculate new values of rhox, rhoy and rhoz for linear case.
 *
 */
void TKSpaceFirstOrder3DSolver::ComputeDensityLinear()
{
  // Scalar
  if (parameters.Get_rho0_scalar())
  {
    SolverCUDAKernels::ComputeDensityLinearHomogeneous(Get_rhox(),
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
    SolverCUDAKernels::ComputeDensityLinearHeterogeneous(Get_rhox(),
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
}// end of ComputeDensityLinear
//--------------------------------------------------------------------------------------------------


/**
 * Compute acoustic pressure for non-linear case.
 */
void TKSpaceFirstOrder3DSolver::ComputePressureNonlinear()
{
  if (parameters.Get_absorbing_flag())
  { // absorbing case
    TRealMatrix& rhoxyz_part = Get_temp_1_real_3D();
    TRealMatrix& BonA_part   = Get_temp_2_real_3D();
    TRealMatrix& du_part     = Get_temp_3_real_3D();

    TRealMatrix& absorb_tau_temp = du_part;
    TRealMatrix& absorb_eta_temp = rhoxyz_part;

    ComputePressurePartsNonLinear(rhoxyz_part, BonA_part, du_part);

    Get_cufft_x_temp().Compute_FFT_3D_R2C(du_part);
    Get_cufft_y_temp().Compute_FFT_3D_R2C(rhoxyz_part);

    SolverCUDAKernels::ComputeAbsorbtionTerm(Get_cufft_x_temp(),
                                             Get_cufft_y_temp(),
                                             Get_absorb_nabla1(),
                                             Get_absorb_nabla2());

    Get_cufft_x_temp().Compute_FFT_3D_C2R(absorb_tau_temp);
    Get_cufft_y_temp().Compute_FFT_3D_C2R(absorb_eta_temp);

    SumPressureTermsNonlinear(absorb_tau_temp, absorb_eta_temp, BonA_part);
  }
  else
  {
    SumPressureNonlinearLossless();
  }
}// end of ComputePressureNonlinear
//--------------------------------------------------------------------------------------------------

/*
 * Compute new p for linear case.
 */
void TKSpaceFirstOrder3DSolver::ComputePressureLinear()
{
  if (parameters.Get_absorbing_flag())
  { // absorbing case
    TRealMatrix& sum_rhoxyz  = Get_temp_1_real_3D();
    TRealMatrix& sum_rho0_du = Get_temp_2_real_3D();

    TRealMatrix& absorb_tau_temp = Get_temp_2_real_3D();
    TRealMatrix& absorb_eta_temp = Get_temp_3_real_3D();

    ComputePressurePartsLinear(sum_rhoxyz, sum_rho0_du);

    // ifftn ( absorb_nabla1 * fftn (rho0 * (duxdx+duydy+duzdz))
    Get_cufft_x_temp().Compute_FFT_3D_R2C(sum_rho0_du);
    Get_cufft_y_temp().Compute_FFT_3D_R2C(sum_rhoxyz);

    SolverCUDAKernels::ComputeAbsorbtionTerm(Get_cufft_x_temp(),
                                             Get_cufft_y_temp(),
                                             Get_absorb_nabla1(),
                                             Get_absorb_nabla2());

    Get_cufft_x_temp().Compute_FFT_3D_C2R(absorb_tau_temp);
    Get_cufft_y_temp().Compute_FFT_3D_C2R(absorb_eta_temp);

    SumPressureTermsLinear(absorb_tau_temp, absorb_eta_temp, sum_rhoxyz);
  }
  else
  {
    // lossless case
    SumPressureLinearLossless();
  }
}// end of ComputePressureLinear()
//--------------------------------------------------------------------------------------------------

/**
 * Add velocity source to the particle velocity.
 */
void TKSpaceFirstOrder3DSolver::AddVelocitySource()
{
  size_t t_index = parameters.Get_t_index();

  if (parameters.Get_ux_source_flag() > t_index)
  {
    SolverCUDAKernels::AddVelocitySource(Get_ux_sgx(),
                                         Get_ux_source_input(),
                                         Get_u_source_index(),
                                         t_index);
  }
  if (parameters.Get_uy_source_flag() > t_index)
  {
    SolverCUDAKernels::AddVelocitySource(Get_uy_sgy(),
                                         Get_uy_source_input(),
                                         Get_u_source_index(),
                                         t_index);
  }
  if (parameters.Get_uz_source_flag() > t_index)
  {
    SolverCUDAKernels::AddVelocitySource(Get_uz_sgz(),
                                         Get_uz_source_input(),
                                         Get_u_source_index(),
                                         t_index);
  }
}// end of AddVelocitySource
//--------------------------------------------------------------------------------------------------

/*
 * Add in pressure source.
 */
void TKSpaceFirstOrder3DSolver::AddPressureSource()
{
  size_t t_index = parameters.Get_t_index();

  if (parameters.Get_p_source_flag() > t_index)
  {
    SolverCUDAKernels::AddPressureSource(Get_rhox(),
                                         Get_rhoy(),
                                         Get_rhoz(),
                                         Get_p_source_input(),
                                         Get_p_source_index(),
                                         t_index);
  }//
}// end of AddPressureSource
//--------------------------------------------------------------------------------------------------

/**
 * Calculate p0 source when necessary.
 */
void TKSpaceFirstOrder3DSolver::Calculate_p0_source()
{
  // get over the scalar problem
  bool is_c2_scalar = parameters.Get_c0_scalar_flag();
  const float* c2 = (is_c2_scalar) ? nullptr : Get_c2().GetDeviceData();

  //-- add the initial pressure to rho as a mass source --//
  SolverCUDAKernels::Compute_p0_AddInitialPressure(Get_p(),
                                                   Get_rhox(),
                                                   Get_rhoy(),
                                                   Get_rhoz(),
                                                   Get_p0_source_input(),
                                                   is_c2_scalar,
                                                   c2);

  //-----------------------------------------------------------------------//
  //--compute u(t = t1 + dt/2) based on the assumption u(dt/2) = -u(-dt/2)-//
  //--    which forces u(t = t1) = 0                                      -//
  //-----------------------------------------------------------------------//
  Get_cufft_x_temp().Compute_FFT_3D_R2C(Get_p());

  SolverCUDAKernels::ComputePressurelGradient(Get_cufft_x_temp(),
                                              Get_cufft_y_temp(),
                                              Get_cufft_z_temp(),
                                              Get_kappa(),
                                              Get_ddx_k_shift_pos(),
                                              Get_ddy_k_shift_pos(),
                                              Get_ddz_k_shift_pos());

  Get_cufft_x_temp().Compute_FFT_3D_C2R(Get_ux_sgx());
  Get_cufft_y_temp().Compute_FFT_3D_C2R(Get_uy_sgy());
  Get_cufft_z_temp().Compute_FFT_3D_C2R(Get_uz_sgz());

  if (parameters.Get_rho0_scalar_flag())
  {
    if (parameters.Get_nonuniform_grid_flag())
    { // non uniform grid, homogeneous
      SolverCUDAKernels::Compute_p0_VelocityScalarNonUniform(Get_ux_sgx(),
                                                             Get_uy_sgy(),
                                                             Get_uz_sgz(),
                                                             Get_dxudxn_sgx(),
                                                             Get_dyudyn_sgy(),
                                                             Get_dzudzn_sgz());
    }
    else
    { //uniform grid, homogeneous
      SolverCUDAKernels::Compute_p0_Velocity(Get_ux_sgx(), Get_uy_sgy(), Get_uz_sgz());
    }
  }
  else
  {
    // heterogeneous, uniform grid
    // divide the matrix by 2 and multiply with st./rho0_sg
    SolverCUDAKernels::Compute_p0_Velocity(Get_ux_sgx(),
                                           Get_uy_sgy(),
                                           Get_uz_sgz(),
                                           Get_dt_rho0_sgx(),
                                           Get_dt_rho0_sgy(),
                                           Get_dt_rho0_sgz());
  }
}// end of Calculate_p0_source
//--------------------------------------------------------------------------------------------------


/**
 * Generate kappa matrix for non-absorbing mode.
 */
void TKSpaceFirstOrder3DSolver::GenerateKappa()
{
  #pragma omp parallel
  {
    const float dx_sq_rec = 1.0f / (parameters.Get_dx() * parameters.Get_dx());
    const float dy_sq_rec = 1.0f / (parameters.Get_dy() * parameters.Get_dy());
    const float dz_sq_rec = 1.0f / (parameters.Get_dz() * parameters.Get_dz());

    const float c_ref_dt_pi = parameters.Get_c_ref() * parameters.Get_dt() * float(M_PI);

    const float nx_rec = 1.0f / static_cast<float>(parameters.GetFullDimensionSizes().nx);
    const float ny_rec = 1.0f / static_cast<float>(parameters.GetFullDimensionSizes().ny);
    const float nz_rec = 1.0f / static_cast<float>(parameters.GetFullDimensionSizes().nz);

    const size_t nx = parameters.GetReducedDimensionSizes().nx;
    const size_t ny = parameters.GetReducedDimensionSizes().ny;
    const size_t nz = parameters.GetReducedDimensionSizes().nz;

    float* kappa = Get_kappa().GetData();

    #pragma omp for schedule (static)
    for (size_t z = 0; z < nz; z++)
    {
      const float z_f    = (float) z;
            float z_part = 0.5f - fabs(0.5f - z_f * nz_rec );
                  z_part = (z_part * z_part) * dz_sq_rec;

      for (size_t y = 0; y < ny; y++)
      {
        const float y_f    = (float) y;
              float y_part = 0.5f - fabs(0.5f - y_f * ny_rec);
                    y_part = (y_part * y_part) * dy_sq_rec;

        const float yz_part = z_part + y_part;
        for (size_t x = 0; x < nx; x++)
        {
          const float x_f = (float) x;
                float x_part = 0.5f - fabs(0.5f - x_f * nx_rec);
                      x_part = (x_part * x_part) * dx_sq_rec;

                float k = c_ref_dt_pi * sqrt(x_part + yz_part);

          // kappa element
          kappa[(z*ny + y) * nx + x ] = (k == 0.0f) ? 1.0f : sin(k)/k;
        }//x
      }//y
    }// z
  }// parallel
}// end of Generate_kappa
//-------------------------------------------------------------------------------------------------

/*
 * Generate kappa, absorb_nabla1, absorb_nabla2 for absorbing media.
 */
void TKSpaceFirstOrder3DSolver::GenerateKappaAndNablas()
{
  #pragma omp parallel
  {
    const float dx_sq_rec = 1.0f / (parameters.Get_dx() * parameters.Get_dx());
    const float dy_sq_rec = 1.0f / (parameters.Get_dy() * parameters.Get_dy());
    const float dz_sq_rec = 1.0f / (parameters.Get_dz() * parameters.Get_dz());

    const float c_ref_dt_2 = parameters.Get_c_ref() * parameters.Get_dt() * 0.5f;
    const float pi_2       = float(M_PI) * 2.0f;

    const size_t nx = parameters.GetFullDimensionSizes().nx;
    const size_t ny = parameters.GetFullDimensionSizes().ny;
    const size_t nz = parameters.GetFullDimensionSizes().nz;

    const float nx_rec   = 1.0f / (float) nx;
    const float ny_rec   = 1.0f / (float) ny;
    const float nz_rec   = 1.0f / (float) nz;

    const size_t nxComplex  = parameters.GetReducedDimensionSizes().nx;
    const size_t nyComplex  = parameters.GetReducedDimensionSizes().ny;
    const size_t nzComplex  = parameters.GetReducedDimensionSizes().nz;

    float* kappa            = Get_kappa().GetData();
    float* absorb_nabla1    = Get_absorb_nabla1().GetData();
    float* absorb_nabla2    = Get_absorb_nabla2().GetData();
    const float alpha_power = parameters.Get_alpha_power();

    #pragma omp for schedule (static)
    for (size_t z = 0; z < nzComplex; z++)
    {
      const float z_f    = (float) z;
            float z_part = 0.5f - fabs(0.5f - z_f * nz_rec );
                  z_part = (z_part * z_part) * dz_sq_rec;

      for (size_t y = 0; y < nyComplex; y++)
      {
        const float y_f    = (float) y;
              float y_part = 0.5f - fabs(0.5f - y_f * ny_rec);
                    y_part = (y_part * y_part) * dy_sq_rec;

        const float yz_part = z_part + y_part;

        size_t i = (z * nyComplex + y) * nxComplex;

        for (size_t x = 0; x < nxComplex; x++)
        {
          const float x_f    = (float) x;
                float x_part = 0.5f - fabs(0.5f - x_f * nx_rec);
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
}// end of GenerateKappaAndNablas
//--------------------------------------------------------------------------------------------------

/**
 * Generate absorb_tau and absorb_eta in for heterogenous media.
 */
void TKSpaceFirstOrder3DSolver::GenerateTauAndEta()
{
  // test for scalars
  if ((parameters.Get_alpha_coeff_scalar_flag()) && (parameters.Get_c0_scalar_flag()))
  {
    const float alpha_power = parameters.Get_alpha_power();
    const float tan_pi_y_2  = tan(static_cast<float> (M_PI_2) * alpha_power);
    const float alpha_db_neper_coeff = (100.0f * pow(1.0e-6f /(2.0f * static_cast<float>(M_PI)), alpha_power)) /
                                       (20.0f * static_cast<float>(M_LOG10E));

    const float alpha_coeff_2 = 2.0f * parameters.Get_alpha_coeff_scalar() * alpha_db_neper_coeff;

    parameters.Get_absorb_tau_scalar() = (-alpha_coeff_2) * pow(parameters.Get_c0_scalar(), alpha_power - 1);
    parameters.Get_absorb_eta_scalar() =   alpha_coeff_2  * pow(parameters.Get_c0_scalar(), alpha_power) * tan_pi_y_2;
  }
  else
  { // matrix
    #pragma omp parallel
    {
      const size_t nx  = parameters.GetFullDimensionSizes().nx;
      const size_t ny  = parameters.GetFullDimensionSizes().ny;
      const size_t nz  = parameters.GetFullDimensionSizes().nz;

      float* absorb_tau = Get_absorb_tau().GetData();
      float* absorb_eta = Get_absorb_eta().GetData();

      float* alpha_coeff;
      size_t alpha_shift;

      if (parameters.Get_alpha_coeff_scalar_flag())
      {
        alpha_coeff = &(parameters.Get_alpha_coeff_scalar());
        alpha_shift = 0;
      }
      else
      {
        alpha_coeff = Get_temp_1_real_3D().GetData();
        alpha_shift = 1;
      }

      float* c0;
      size_t c0_shift;
      if (parameters.Get_c0_scalar_flag())
      {
        c0 = &(parameters.Get_c0_scalar());
        c0_shift = 0;
      }
      else
      {
        c0 = Get_c2().GetData();
        c0_shift = 1;
      }

      const float alpha_power = parameters.Get_alpha_power();
      const float tan_pi_y_2  = tan(static_cast<float>(M_PI_2) * alpha_power);

      //alpha = 100*alpha.*(1e-6/(2*pi)).^y./
      //                  (20*log10(exp(1)));
      const float alpha_db_neper_coeff = (100.0f * pow(1.0e-6f / (2.0f * static_cast<float>(M_PI)), alpha_power)) /
                                         (20.0f * static_cast<float>(M_LOG10E));


      #pragma omp for schedule (static)
      for (size_t z = 0; z < nz; z++)
      {
        for (size_t y = 0; y < ny; y++)
        {
          size_t i = (z * ny + y) * nx;
          for (size_t x = 0; x < nx; x++)
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
}// end of GenerateTauAndEta
//--------------------------------------------------------------------------------------------------

/**
 * Prepare dt./ rho0  for non-uniform grid.
 *
 */
void TKSpaceFirstOrder3DSolver::GenerateInitialDenisty()
{
  #pragma omp parallel
  {
    float* dt_rho0_sgx   = Get_dt_rho0_sgx().GetData();
    float* dt_rho0_sgy   = Get_dt_rho0_sgy().GetData();
    float* dt_rho0_sgz   = Get_dt_rho0_sgz().GetData();

    const float dt = parameters.Get_dt();

    const float* duxdxn_sgx = Get_dxudxn_sgx().GetData();
    const float* duydyn_sgy = Get_dyudyn_sgy().GetData();
    const float* duzdzn_sgz = Get_dzudzn_sgz().GetData();

    const size_t nz = Get_dt_rho0_sgx().GetDimensionSizes().nz;
    const size_t ny = Get_dt_rho0_sgx().GetDimensionSizes().ny;
    const size_t nx = Get_dt_rho0_sgx().GetDimensionSizes().nx;

    const size_t sliceSize = (nx * ny );

    #pragma omp for schedule (static)
    for (size_t z = 0; z < nz; z++)
    {
      register size_t i = z* sliceSize;
      for (size_t y = 0; y < ny; y++)
      {
        for (size_t x = 0; x < nx; x++)
        {
          dt_rho0_sgx[i] = (dt * duxdxn_sgx[x]) / dt_rho0_sgx[i];
          i++;
        } // x
      } // y
    } // z

    #pragma omp for schedule (static)
    for (size_t z = 0; z < nz; z++)
    {
      register size_t i = z* sliceSize;
      for (size_t y = 0; y < ny; y++)
      {
        const float duydyn_el = duydyn_sgy[y];
        for (size_t x = 0; x < nx; x++)
        {
          dt_rho0_sgy[i] = (dt * duydyn_el) / dt_rho0_sgy[i];
          i++;
        } // x
      } // y
    } // z


    #pragma omp for schedule (static)
    for (size_t z = 0; z < nz; z++)
    {
      register size_t i = z* sliceSize;
      const float duzdzn_el = duzdzn_sgz[z];
      for (size_t y = 0; y < ny; y++)
      {
        for (size_t x = 0; x < nx; x++)
        {
          dt_rho0_sgz[i] = (dt * duzdzn_el) / dt_rho0_sgz[i];
          i++;
        } // x
      } // y
    } // z
  } // parallel
}// end of GenerateInitialDenisty
//--------------------------------------------------------------------------------------------------


/**
 * Compute c^2 on the CPU side.
 */
void TKSpaceFirstOrder3DSolver::Compute_c2()
{
  if (parameters.Get_c0_scalar_flag())
  { // scalar
      float c = parameters.Get_c0_scalar();
      parameters.Get_c0_scalar() = c * c;
  }
  else
  { // matrix
    float* c2 =  Get_c2().GetData();

    #pragma omp parallel for schedule (static)
    for (size_t i=0; i < Get_c2().GetElementCount(); i++)
    {
      c2[i] = c2[i] * c2[i];
    }
  }// matrix
}// ComputeC2
//--------------------------------------------------------------------------------------------------

/**
 * Compute three temporary sums in the new pressure formula, non-linear absorbing case.
 *
 * @param [out] rho_part  - rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [out] BonA_part - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [out] du_part   - rho0* (duxdx + duydy + duzdz)
 *
 */
void TKSpaceFirstOrder3DSolver::ComputePressurePartsNonLinear(TRealMatrix& rho_part,
                                                              TRealMatrix& BonA_part,
                                                              TRealMatrix& du_part)
{
  const bool is_BonA_scalar = parameters.Get_BonA_scalar_flag();
  const bool is_rho0_scalar = parameters.Get_rho0_scalar_flag();

  const float* BonA_data = (is_BonA_scalar) ? nullptr : Get_BonA().GetDeviceData();
  const float* rho0_data = (is_rho0_scalar) ? nullptr : Get_rho0().GetDeviceData();

  SolverCUDAKernels::ComputePressurePartsNonLinear(rho_part,
                                                   BonA_part,
                                                   du_part,
                                                   Get_rhox(),
                                                   Get_rhoy(),
                                                   Get_rhoz(),
                                                   Get_duxdx(),
                                                   Get_duydy(),
                                                   Get_duzdz(),
                                                   is_BonA_scalar,
                                                   BonA_data,
                                                   is_rho0_scalar,
                                                   rho0_data);

}// end of ComputePressurePartsNonLinear
//--------------------------------------------------------------------------------------------------


/**
 * Calculate two temporary sums in the new pressure formula, linear absorbing case.
 *
 * @param [out] rhoxyz_sum   -rhox_sgx + rhoy_sgy + rhoz_sgz
 * @param [out] rho0_du_sum - rho0* (duxdx + duydy + duzdz);
 */
void TKSpaceFirstOrder3DSolver::ComputePressurePartsLinear(TRealMatrix& rhoxyz_sum,
                                                           TRealMatrix& rho0_du_sum)
{
  const bool   is_rho0_scalar = parameters.Get_rho0_scalar();
  const float* rho0_matrix    = (is_rho0_scalar) ? nullptr : Get_rho0().GetDeviceData();

  SolverCUDAKernels::ComputePressurePartsLinear(rhoxyz_sum,
                                                rho0_du_sum,
                                                Get_rhox(),
                                                Get_rhoy(),
                                                Get_rhoz(),
                                                Get_duxdx(),
                                                Get_duydy(),
                                                Get_duzdz(),
                                                is_rho0_scalar,
                                                rho0_matrix);
}// end of ComputePressurePartsLinear
//--------------------------------------------------------------------------------------------------

/**
 * Sum sub-terms to calculate new pressure, non-linear case.

 * @param [in] absorb_tau_temp  -  tau component
 * @param [in] absorb_eta_temp  -   BonA + rho ^2 / 2 rho0  +
 *                                      (rhox_sgx + rhoy_sgy + rhoz_sgz)
 * @param [in] BonA_temp        -   rho0* (duxdx + duydy + duzdz)
 */
void TKSpaceFirstOrder3DSolver::SumPressureTermsNonlinear(TRealMatrix& absorb_tau_temp,
                                                          TRealMatrix& absorb_eta_temp,
                                                          TRealMatrix& BonA_temp)
{
  const bool is_c2_scalar      = parameters.Get_c0_scalar_flag();
  const bool is_tau_eta_scalar = parameters.Get_c0_scalar_flag() &&
                                 parameters.Get_alpha_coeff_scalar_flag();

  const float* c2_data_matrix  = (is_c2_scalar)      ? nullptr : Get_c2().GetDeviceData();
  const float* tau_data_matrix = (is_tau_eta_scalar) ? nullptr : Get_absorb_tau().GetDeviceData();
  const float* eta_data_matrix = (is_tau_eta_scalar) ? nullptr : Get_absorb_eta().GetDeviceData();

  const float* Absorb_tau_data = absorb_tau_temp.GetDeviceData();
  const float* Absorb_eta_data = absorb_eta_temp.GetDeviceData();

  SolverCUDAKernels::SumPressureTermsNonlinear(Get_p(),
                                               BonA_temp,
                                               is_c2_scalar,
                                               c2_data_matrix,
                                               is_tau_eta_scalar,
                                               Absorb_tau_data,
                                               tau_data_matrix,
                                               Absorb_eta_data,
                                               eta_data_matrix);
}// end of SumPressureTermsNonlinear
//--------------------------------------------------------------------------------------------------

/**
 * Sum sub-terms to calculate new pressure, linear case.
 *
 * @param [in] absorb_tau_temp - sub-term with absorb_tau
 * @param [in] absorb_eta_temp - sub-term with absorb_eta
 * @param [in] rhoxyz_sum      - rhox_sgx + rhoy_sgy + rhoz_sgz
 */
void TKSpaceFirstOrder3DSolver::SumPressureTermsLinear(TRealMatrix& absorb_tau_temp,
                                                       TRealMatrix& absorb_eta_temp,
                                                       TRealMatrix& rhoxyz_sum)
{
  const bool is_c2_scalar      = parameters.Get_c0_scalar_flag();
  const bool is_tau_eta_scalar = parameters.Get_c0_scalar_flag() &&
                                 parameters.Get_alpha_coeff_scalar_flag();

  const float* c2_data_matrix  = (is_c2_scalar)      ? nullptr : Get_c2().GetDeviceData();
  const float* tau_data_matrix = (is_tau_eta_scalar) ? nullptr : Get_absorb_tau().GetDeviceData();
  const float* eta_data_matrix = (is_tau_eta_scalar) ? nullptr : Get_absorb_eta().GetDeviceData();

  SolverCUDAKernels::SumPressureTermsLinear(Get_p(),
                                            absorb_tau_temp,
                                            absorb_eta_temp,
                                            rhoxyz_sum,
                                            is_c2_scalar,
                                            c2_data_matrix,
                                            is_tau_eta_scalar,
                                            tau_data_matrix,
                                            eta_data_matrix);
}// end of SumPressureTermsLinear
//--------------------------------------------------------------------------------------------------

/**
 * Sum sub-terms for new p, non-linear lossless case.
 */
void TKSpaceFirstOrder3DSolver::SumPressureNonlinearLossless()
{
  const bool   is_c2_scalar   = parameters.Get_c0_scalar_flag();
  const bool   is_BonA_scalar = parameters.Get_BonA_scalar_flag();
  const bool   is_rho0_scalar = parameters.Get_rho0_scalar_flag();

  const float* c2_data_matrix   = (is_c2_scalar)   ? nullptr : Get_c2().GetDeviceData();
  const float* BonA_data_matrix = (is_BonA_scalar) ? nullptr : Get_BonA().GetDeviceData();
  const float* rho0_data_matrix = (is_rho0_scalar) ? nullptr : Get_rho0().GetDeviceData();

  SolverCUDAKernels::SumPressureNonlinearLossless(Get_p(),
                                                  Get_rhox(),
                                                  Get_rhoy(),
                                                  Get_rhoz(),
                                                  is_c2_scalar,
                                                  c2_data_matrix,
                                                  is_BonA_scalar,
                                                  BonA_data_matrix,
                                                  is_rho0_scalar,
                                                  rho0_data_matrix);

}// end of SumPressureNonlinearLossless
//--------------------------------------------------------------------------------------------------

/**
 * Sum sub-terms for new p, linear lossless case.
 */
void TKSpaceFirstOrder3DSolver::SumPressureLinearLossless()
{
  const float  is_c2_scalar =  parameters.Get_c0_scalar();
  const float* c2_matrix    = (is_c2_scalar) ? nullptr : Get_c2().GetDeviceData();

  SolverCUDAKernels::SumPressureLinearLossless(Get_p(),
                                               Get_rhox(),
                                               Get_rhoy(),
                                               Get_rhoz(),
                                               is_c2_scalar,
                                               c2_matrix);

}// end of SumPressureLinearLossless
//--------------------------------------------------------------------------------------------------


/**
 * Calculated shifted velocities.
 * \n
 * ux_shifted = real(ifft(bsxfun(\@times, x_shift_neg, fft(ux_sgx, [], 1)), [], 1)); \n
 * uy_shifted = real(ifft(bsxfun(\@times, y_shift_neg, fft(uy_sgy, [], 2)), [], 2)); \n
 * uz_shifted = real(ifft(bsxfun(\@times, z_shift_neg, fft(uz_sgz, [], 3)), [], 3)); \n
 */

void TKSpaceFirstOrder3DSolver::CalculateShiftedVelocity()
{

  // ux_shifted
  Get_cufft_shift_temp().Compute_FFT_1DX_R2C(Get_ux_sgx());
  SolverCUDAKernels::ComputeVelocityShiftInX(Get_cufft_shift_temp(), Get_x_shift_neg_r());
  Get_cufft_shift_temp().Compute_FFT_1DX_C2R(Get_ux_shifted());

  // uy_shifted
  Get_cufft_shift_temp().Compute_FFT_1DY_R2C(Get_uy_sgy());
  SolverCUDAKernels::ComputeVelocityShiftInY(Get_cufft_shift_temp(), Get_y_shift_neg_r());
  Get_cufft_shift_temp().Compute_FFT_1DY_C2R(Get_uy_shifted());

  // uz_shifted
  Get_cufft_shift_temp().Compute_FFT_1DZ_R2C(Get_uz_sgz());
  SolverCUDAKernels::ComputeVelocityShiftInZ(Get_cufft_shift_temp(), Get_z_shift_neg_r());
  Get_cufft_shift_temp().Compute_FFT_1DZ_C2R(Get_uz_shifted());

}// end of CalculateShiftedVelocity
//--------------------------------------------------------------------------------------------------

/**
 * Print progress statistics.
 */
void TKSpaceFirstOrder3DSolver::PrintStatistics()
{
  const float  nt = (float) parameters.Get_nt();
  const size_t t_index = parameters.Get_t_index();

  if (t_index > (actPercent * nt * 0.01f) )
  {
    actPercent += parameters.GetProgressPrintInterval();

    iterationTime.Stop();

    const double elTime = iterationTime.GetElapsedTime();
    const double elTimeWithLegs = iterationTime.GetElapsedTime() +
                                  simulationTime.GetCumulatedElapsedTimeOverPreviousLegs();
    const double toGo   = ((elTimeWithLegs / (float) (t_index + 1)) *  nt) - elTimeWithLegs;

    struct tm* current;
    time_t now;
    time(&now);
    now += toGo;
    current = localtime(&now);

    TLogger::Log(TLogger::BASIC,
                 OUT_FMT_SIMULATION_PROGRESS,
                 (size_t) ((t_index) / (nt * 0.01f)),'%',
                 elTime, toGo,
                 current->tm_mday, current->tm_mon+1, current->tm_year-100,
                 current->tm_hour, current->tm_min, current->tm_sec);
    TLogger::Flush(TLogger::BASIC);
  }
}// end of PrintStatistics
//--------------------------------------------------------------------------------------------------

/**
 * Is time to checkpoint?
 *
 * @return true if it is necessary to stop to checkpoint
 */
bool TKSpaceFirstOrder3DSolver::IsTimeToCheckpoint()
{
  if (!parameters.IsCheckpointEnabled()) return false;

  totalTime.Stop();

  return (totalTime.GetElapsedTime() > static_cast<float>(parameters.GetCheckpointInterval()));
}// end of IsTimeToCheckpoint
//--------------------------------------------------------------------------------------------------


/**
 * Was the loop interrupted to checkpoint?
 *
 * @return true if it is time to checkpoint
 */
bool TKSpaceFirstOrder3DSolver::IsCheckpointInterruption() const
{
  return (parameters.Get_t_index() != parameters.Get_nt());
}// end of IsCheckpointInterruption
//--------------------------------------------------------------------------------------------------

/**
 * Check the output file has the correct format and version.
 *
 * @throw ios::failure if an error happens
 */
void TKSpaceFirstOrder3DSolver::CheckOutputFile()
{
  // The header has already been read
  THDF5_FileHeader& fileHeader = parameters.GetFileHeader();
  THDF5_File&       outputFile = parameters.GetOutputFile();

  // test file type
  if (fileHeader.GetFileType() != THDF5_FileHeader::OUTPUT)
  {
    char errMsg[256] = "";
    snprintf(errMsg,
             256,
             ERR_FMT_BAD_OUTPUT_FILE_FORMAT,
             parameters.GetOutputFileName().c_str());
    throw std::ios::failure(errMsg);
  }

  // test file major version
  if (!fileHeader.CheckMajorFileVersion())
  {
    char errMsg[256] = "";
    snprintf(errMsg,
             256,
             ERR_FMT_BAD_MAJOR_File_Version,
             parameters.GetCheckpointFileName().c_str(),
             fileHeader.GetCurrentHDF5_MajorVersion().c_str());
    throw std::ios::failure(errMsg);
  }

  // test file minor version
  if (!fileHeader.CheckMinorFileVersion())
  {
    char errMsg[256] = "";
    snprintf(errMsg,
             256,
             ERR_FMT_BAD_MINOR_FILE_VERSION,
             parameters.GetCheckpointFileName().c_str(),
             fileHeader.GetCurrentHDF5_MinorVersion().c_str());
    throw std::ios::failure(errMsg);
  }

  // Check dimension sizes
  TDimensionSizes outputDimSizes;
  outputFile.ReadScalarValue(outputFile.GetRootGroup(), Nx_NAME, outputDimSizes.nx);
  outputFile.ReadScalarValue(outputFile.GetRootGroup(), Ny_NAME, outputDimSizes.ny);
  outputFile.ReadScalarValue(outputFile.GetRootGroup(), Nz_NAME, outputDimSizes.nz);

 if (parameters.GetFullDimensionSizes() != outputDimSizes)
 {
    char errMsg[256] = "";
    snprintf(errMsg,
             256,
             ERR_FMT_OUTPUT_DIMENSIONS_NOT_MATCH,
             outputDimSizes.nx,
             outputDimSizes.ny,
             outputDimSizes.nz,
             parameters.GetFullDimensionSizes().nx,
             parameters.GetFullDimensionSizes().ny,
             parameters.GetFullDimensionSizes().nz);

   throw std::ios::failure(errMsg);
 }
}// end of CheckOutputFile
//--------------------------------------------------------------------------------------------------


/**
 * Check the file type and the version of the checkpoint file.
 *
 * @throw ios::failure if an error happens
 *
 */
void TKSpaceFirstOrder3DSolver::CheckCheckpointFile()
{
  // read the header and check the file version
  THDF5_FileHeader fileHeader;
  THDF5_File&     checkpointFile = parameters.GetCheckpointFile();

  fileHeader.ReadHeaderFromCheckpointFile(checkpointFile);

  // test file type
  if (fileHeader.GetFileType() != THDF5_FileHeader::CHECKPOINT)
  {
    char errMsg[256] = "";
    snprintf(errMsg,
             256,
             ERR_FMT_BAD_CHECKPOINT_FILE_FORMAT,
             parameters.GetCheckpointFileName().c_str());
    throw std::ios::failure(errMsg);
  }

  // test file major version
  if (!fileHeader.CheckMajorFileVersion())
  {
    char errMsg[256] = "";
    snprintf(errMsg,
             256,
             ERR_FMT_BAD_MAJOR_File_Version,
             parameters.GetCheckpointFileName().c_str(),
             fileHeader.GetCurrentHDF5_MajorVersion().c_str());
    throw std::ios::failure(errMsg);
  }

  // test file minor version
  if (!fileHeader.CheckMinorFileVersion())
  {
    char errMsg[256] = "";
    snprintf(errMsg,
             256,
             ERR_FMT_BAD_MINOR_FILE_VERSION,
             parameters.GetCheckpointFileName().c_str(),
             fileHeader.GetCurrentHDF5_MinorVersion().c_str());
    throw std::ios::failure(errMsg);
  }

  // Check dimension sizes
  TDimensionSizes checkpointDimSizes;
  checkpointFile.ReadScalarValue(checkpointFile.GetRootGroup(), Nx_NAME, checkpointDimSizes.nx);
  checkpointFile.ReadScalarValue(checkpointFile.GetRootGroup(), Ny_NAME, checkpointDimSizes.ny);
  checkpointFile.ReadScalarValue(checkpointFile.GetRootGroup(), Nz_NAME, checkpointDimSizes.nz);

 if (parameters.GetFullDimensionSizes() != checkpointDimSizes)
 {
    char errMsg[256] = "";
    snprintf(errMsg,
             256,
             ERR_FMT_CHECKPOINT_DIMENSIONS_NOT_MATCH,
             checkpointDimSizes.nx,
             checkpointDimSizes.ny,
             checkpointDimSizes.nz,
             parameters.GetFullDimensionSizes().nx,
             parameters.GetFullDimensionSizes().ny,
             parameters.GetFullDimensionSizes().nz);

   throw std::ios::failure(errMsg);
 }
}// end of CheckCheckpointFile
//--------------------------------------------------------------------------------------------------

/**
 * Restore cumulated elapsed time from the output file.  Open the header, read this and store
 * into TMPI_Time classes
 *
 * @param [in] outputFile - Output file
 */
void TKSpaceFirstOrder3DSolver::LoadElapsedTimeFromOutputFile(THDF5_File& outputFile)
{
  double totalTime, dataLoadTime, preProcessingTime, simulationTime, postProcessingTime;

  // Get execution times stored in the output file header
  parameters.GetFileHeader().GetExecutionTimes(totalTime,
                                               dataLoadTime,
                                               preProcessingTime,
                                               simulationTime,
                                               postProcessingTime);

  this->totalTime.SetCumulatedElapsedTimeOverPreviousLegs(totalTime);
  this->dataLoadTime.SetCumulatedElapsedTimeOverPreviousLegs(dataLoadTime);
  this->preProcessingTime.SetCumulatedElapsedTimeOverPreviousLegs(preProcessingTime);
  this->simulationTime.SetCumulatedElapsedTimeOverPreviousLegs(simulationTime);
  this->postProcessingTime.SetCumulatedElapsedTimeOverPreviousLegs(postProcessingTime);

}// end of LoadElapsedTimeFromOutputFile
//--------------------------------------------------------------------------------------------------
