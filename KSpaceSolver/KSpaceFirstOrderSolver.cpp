/**
 * @file      KSpaceFirstOrderSolver.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file containing k-space first order solver in 3D fluid medium. This is the main class
 *            controlling the simulation.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      12 July      2012, 10:27 (created) \n
 *            12 March     2019, 10:56 (revised)
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

// Linux build
#ifdef __linux__
#include <sys/resource.h>
#include <cmath>
#endif

// Windows build
#ifdef _WIN64
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <cmath>
#include <Windows.h>
#include <Psapi.h>
#pragma comment(lib, "Psapi.lib")
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <ctime>
#include <limits>
#include <numeric>
#include <algorithm>
#include <iostream>

#include <KSpaceSolver/KSpaceFirstOrderSolver.h>

#include <Hdf5/Hdf5FileHeader.h>
#include <Hdf5/Hdf5File.h>

#include <Logger/ErrorMessages.h>
#include <Logger/Logger.h>

#include <KSpaceSolver/SolverCUDAKernels.cuh>
#include <Containers/MatrixContainer.h>
#include <Containers/OutputStreamContainer.h>

using std::ios;
using std::string;

/// shortcut for Simulation dimensions
using SD = Parameters::SimulationDimension;

//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Constants -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor of the class.
 */
KSpaceFirstOrderSolver::KSpaceFirstOrderSolver()
    : mMatrixContainer(), mOutputStreamContainer(),
      mParameters(Parameters::getInstance()),
      mActPercent(0.0f),
      mIsTimestepRightAfterRestore(false),
      mTotalTime(), mPreProcessingTime(), mDataLoadTime(), mSimulationTime(),
      mPostProcessingTime(), mIterationTime() {
  mTotalTime.start();

  //Switch off default HDF5 error messages
  H5Eset_auto(H5E_DEFAULT, NULL, NULL);
} // end of KSpaceFirstOrderSolver
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor of the class.
 */
KSpaceFirstOrderSolver::~KSpaceFirstOrderSolver() {
  // Delete CUDA FFT plans and related data
  CufftComplexMatrix::destroyAllPlansAndStaticData();

  // Free memory
  freeMemory();

  //Reset device after the run - recommended by CUDA SDK
  cudaDeviceReset();
} // end of ~KSpaceFirstOrderSolver
//----------------------------------------------------------------------------------------------------------------------

/**
 * The method allocates the matrix container and create all matrices and creates all output streams.
 */
void KSpaceFirstOrderSolver::allocateMemory() {
  Logger::log(Logger::LogLevel::kBasic, kOutFmtMemoryAllocation);
  Logger::flush(Logger::LogLevel::kBasic);

  // create container, then all matrices
  mMatrixContainer.init();
  mMatrixContainer.createMatrices();

  /*Logger::log(Logger::LogLevel::kBasic, kOutFmtNoDone);
  Logger::log(Logger::LogLevel::kBasic, kOutFmtPeakHostMemory, getHostMemoryUsage());
  Logger::log(Logger::LogLevel::kBasic, kOutFmtCurrentHostMemory, getCurrentMemoryUsage());
  Logger::log(Logger::LogLevel::kBasic, kOutFmtEmpty);*/
  // add output streams into container
  mOutputStreamContainer.init(mMatrixContainer);

  Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
} // end of allocateMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * The method frees all memory allocated by the class.
 */
void KSpaceFirstOrderSolver::freeMemory() {
  mMatrixContainer.freeMatrices();
  mOutputStreamContainer.freeStreams();
} // end of freeMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Load data from the input file provided by the Parameter class and creates the output time series streams.
 */
void KSpaceFirstOrderSolver::loadInputData() {
  // Load data from disk
  Logger::log(Logger::LogLevel::kBasic, kOutFmtDataLoading);
  Logger::flush(Logger::LogLevel::kBasic);

  mDataLoadTime.start();

  // open and load input file
  Hdf5File& inputFile = mParameters.getInputFile(); // file is opened (in Parameters)
  Hdf5File& outputFile = mParameters.getOutputFile();
  Hdf5File& checkpointFile = mParameters.getCheckpointFile();

  // Load data from disk
  Logger::log(Logger::LogLevel::kFull, kOutFmtNoDone);
  Logger::log(Logger::LogLevel::kFull, kOutFmtReadingInputFile);
  Logger::flush(Logger::LogLevel::kFull);

  // load data from the input file
  mMatrixContainer.loadDataFromInputFile();

  // close the input file since we don't need it anymore.
  inputFile.close();

  Logger::log(Logger::LogLevel::kFull, kOutFmtDone);

  // The simulation does not use checkpointing or this is the first turn
  bool recoverFromCheckpoint = (mParameters.isCheckpointEnabled() && Hdf5File::canAccess(mParameters.getCheckpointFileName()));

  if (recoverFromCheckpoint) {
    //------------------------------------- Read data from the checkpoint file ---------------------------------------//
    Logger::log(Logger::LogLevel::kFull, kOutFmtReadingCheckpointFile);
    Logger::flush(Logger::LogLevel::kFull);

    // Open checkpoint file
    checkpointFile.open(mParameters.getCheckpointFileName());

    // Check the checkpoint file
    checkCheckpointFile();

    // read the actual value of t_index
    size_t checkpointedTimeIndex;
    checkpointFile.readScalarValue(checkpointFile.getRootGroup(), kTimeIndexName, checkpointedTimeIndex);
    mParameters.setTimeIndex(checkpointedTimeIndex);

    // Read necessary matrices from the checkpoint file
    mMatrixContainer.loadDataFromCheckpointFile();

    Logger::log(Logger::LogLevel::kFull, kOutFmtDone);

    //--------------------------------------- Read data from the output file -----------------------------------------//
    Logger::log(Logger::LogLevel::kFull, kOutFmtReadingOutputFile);
    Logger::flush(Logger::LogLevel::kFull);

    // Reopen output file for RW access
    outputFile.open(mParameters.getOutputFileName(), H5F_ACC_RDWR);
    // Read file header of the output file
    mParameters.getFileHeader().readHeaderFromOutputFile(outputFile);
    // Check the checkpoint file
    checkOutputFile();
    // Restore elapsed time
    loadElapsedTimeFromOutputFile();

    // Reopen streams
    mOutputStreamContainer.reopenStreams();
    Logger::log(Logger::LogLevel::kFull, kOutFmtDone);
    checkpointFile.close();
  } else {
    if (mParameters.getOnlyPostProcessingFlag() &&
        (mParameters.getStoreIntensityAvgFlag() ||
         mParameters.getStoreIntensityAvgCFlag() ||
         mParameters.getStoreQTermFlag() ||
         mParameters.getStoreQTermCFlag()) &&
        Hdf5File::canAccess(mParameters.getOutputFileName())) {
      // Open output file
      // TODO check existing datasets and their sizes (streams)
      outputFile.open(mParameters.getOutputFileName(), H5F_ACC_RDWR);
    } else {
      //------------------------------------ First round of multi-leg simulation ---------------------------------------//
      // Create the output file
      Logger::log(Logger::LogLevel::kFull, kOutFmtCreatingOutputFile);
      Logger::flush(Logger::LogLevel::kFull);

      outputFile.create(mParameters.getOutputFileName());
      Logger::log(Logger::LogLevel::kFull, kOutFmtDone);
    }
    // Create the steams, link them with the sampled matrices, however DO NOT allocate memory!
    mOutputStreamContainer.createStreams();
  }

  // Stop timer
  mDataLoadTime.stop();
  if (Logger::getLevel() != Logger::LogLevel::kFull) {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
  }
} // end of loadInputData
//----------------------------------------------------------------------------------------------------------------------

/**
 * This method computes k-space First Order 2D/3D simulation.
 */
void KSpaceFirstOrderSolver::compute() {
  CudaParameters& cudaParameters = mParameters.getCudaParameters();

  // fft initialization and preprocessing
  try {
    mPreProcessingTime.start();

    Logger::log(Logger::LogLevel::kBasic, kOutFmtFftPlans);
    Logger::flush(Logger::LogLevel::kBasic);
    // initialize all CUDA FFT plans
    initializeCufftPlans();
    Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);

    Logger::log(Logger::LogLevel::kBasic, kOutFmtPreProcessing);
    Logger::flush(Logger::LogLevel::kBasic);

    // preprocessing is done on CPU and must pretend the CUDA configuration
    if (mParameters.isSimulation3D())
      preProcessing<SD::k3D>();
    else
      preProcessing<SD::k2D>();

    mPreProcessingTime.stop();
    // Set kernel configurations
    cudaParameters.setKernelConfiguration();

    // Set up constant memory - copy over to GPU
    // Constant memory uses some variables calculated during preprocessing
    cudaParameters.setUpDeviceConstants();

    Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
  } catch (const std::exception& e) {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtFailed);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtLastSeparator);

    Logger::errorAndTerminate(Logger::wordWrapString(e.what(), kErrFmtPathDelimiters, 9));
  }

  // Logger header for simulation
  Logger::log(Logger::LogLevel::kBasic, kOutFmtElapsedTime, mPreProcessingTime.getElapsedTime());
  Logger::log(Logger::LogLevel::kBasic, kOutFmtCompResourcesHeader);
  Logger::log(Logger::LogLevel::kBasic, kOutFmtPeakHostMemory, getHostMemoryUsage());
  Logger::log(Logger::LogLevel::kBasic, kOutFmtCurrentHostMemory, getCurrentHostMemoryUsage());
  Logger::log(Logger::LogLevel::kBasic, kOutFmtCurrentDeviceMemory, getDeviceMemoryUsage());

  const string blockDims = Logger::formatMessage(kOutFmtCudaGridShapeFormat,
                                                 cudaParameters.getSolverGridSize1D(),
                                                 cudaParameters.getSolverBlockSize1D());

  Logger::log(Logger::LogLevel::kFull, kOutFmtCudaSolverGridShape, blockDims.c_str());

  const string gridDims = Logger::formatMessage(kOutFmtCudaGridShapeFormat,
                                                cudaParameters.getSamplerGridSize1D(),
                                                cudaParameters.getSamplerBlockSize1D());

  Logger::log(Logger::LogLevel::kFull, kOutFmtCudaSamplerGridShape, gridDims.c_str());

  // Main loop
  if (!mParameters.getOnlyPostProcessingFlag()) {
    try {
      mSimulationTime.start();

      if (mParameters.isSimulation3D())
        computeMainLoop<SD::k3D>();
      else
        computeMainLoop<SD::k2D>();

      mSamplingTime = std::accumulate(mSamplingIterationTimes.begin(), mSamplingIterationTimes.end(), double(0));
      mAverageSamplingIterationTime = mSamplingTime / mSamplingIterationTimes.size();
      mNotSamplingTime = std::accumulate(mNotSamplingIterationTimes.begin(), mNotSamplingIterationTimes.end(), double(0));
      mAverageNotSamplingIterationTime = mNotSamplingTime / mNotSamplingIterationTimes.size();

      mSimulationTime.stop();

      Logger::log(Logger::LogLevel::kBasic, kOutFmtSimulationEndSeparator);
    } catch (const std::exception& e) {
      Logger::log(Logger::LogLevel::kBasic, kOutFmtSimulatoinFinalSeparator);
      Logger::errorAndTerminate(Logger::wordWrapString(e.what(), kErrFmtPathDelimiters, 9));
    }

    mSimulationPeakHostMemoryConsumption = getHostMemoryUsage();
    mSimulationPeakDeviceMemoryConsumption = getDeviceMemoryUsage();
    Logger::log(Logger::LogLevel::kBasic, kOutFmtPeakHostMemory, getHostMemoryUsage());
    Logger::log(Logger::LogLevel::kBasic, kOutFmtCurrentHostMemory, getCurrentHostMemoryUsage());
    Logger::log(Logger::LogLevel::kBasic, kOutFmtCurrentDeviceMemory, getDeviceMemoryUsage());
    //Logger::log(Logger::LogLevel::kBasic, kOutFmtElapsedTime, mSimulationTime.getElapsedTime(), mIterationTime.getElapsedTime() - mIterationTime.getElapsedTimeOverPreviousLegs());
    Logger::log(Logger::LogLevel::kBasic, kOutFmtmSamplingTime, mSamplingTime);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtmAverageSamplingIterationTime, mAverageSamplingIterationTime);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtmNotSamplingTime, mNotSamplingTime);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtmAverageNotSamplingIterationTime, mAverageNotSamplingIterationTime);
  }

  // Post processing region
  mPostProcessingTime.start();

  try {
    if (!mParameters.getOnlyPostProcessingFlag() && isCheckpointInterruption()) { // Checkpoint
      Logger::log(Logger::LogLevel::kBasic, kOutFmtElapsedTime, mSimulationTime.getElapsedTime());
      Logger::log(Logger::LogLevel::kBasic, kOutFmtCheckpointCompletedTimeSteps, mParameters.getTimeIndex());
      Logger::log(Logger::LogLevel::kBasic, kOutFmtCheckpointHeader);
      Logger::log(Logger::LogLevel::kBasic, kOutFmtCreatingCheckpoint);
      Logger::flush(Logger::LogLevel::kBasic);

      if (Logger::getLevel() == Logger::LogLevel::kFull) {
        Logger::log(Logger::LogLevel::kBasic, kOutFmtNoDone);
      }

      saveCheckpointData();

      if (Logger::getLevel() != Logger::LogLevel::kFull) {
        Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
      }
    } else { // Finish
      Logger::log(Logger::LogLevel::kBasic, kOutFmtElapsedTime, mSimulationTime.getElapsedTime());
      Logger::log(Logger::LogLevel::kBasic, kOutFmtSeparator);
      Logger::log(Logger::LogLevel::kBasic, kOutFmtPostProcessing);
      Logger::flush(Logger::LogLevel::kBasic);

      if (mParameters.isSimulation3D()) {
        postProcessing<SD::k3D>();
      } else {
        postProcessing<SD::k2D>();
      }

      // if checkpointing is enabled and the checkpoint file was created in the past, delete it
      if (mParameters.isCheckpointEnabled()) {
        std::remove(mParameters.getCheckpointFileName().c_str());
      }
      Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
    }
  } catch (const std::exception& e) {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtFailed);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtLastSeparator);

    Logger::errorAndTerminate(Logger::wordWrapString(e.what(), kErrFmtPathDelimiters, 9));
  }
  mPostProcessingTime.stop();

  // Final data written
  try {
    writeOutputDataInfo();
    mParameters.getOutputFile().close();

    Logger::log(Logger::LogLevel::kBasic, kOutFmtElapsedTime, mPostProcessingTime.getElapsedTime());
  } catch (const std::exception& e) {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtLastSeparator);
    Logger::errorAndTerminate(Logger::wordWrapString(e.what(), kErrFmtPathDelimiters, 9));
  }
} // end of compute()
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get peak CPU memory usage in MB.
 */
size_t KSpaceFirstOrderSolver::getHostMemoryUsage() const {
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
} // end of getHostMemoryUsage
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get available CPU memory in MB.
 */
size_t KSpaceFirstOrderSolver::getAvailableHostMemory() const {
#ifdef __linux__
  string token;
  std::ifstream file("/proc/meminfo");
  while (file >> token) {
    if (token == "MemAvailable:") {
      unsigned long mem;
      if (file >> mem) {
        return mem >> 10;
      } else {
        return 0;
      }
    }
    // ignore rest of the line
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return 0; // nothing found
#endif
// This is wrong, probably without cache and other.
/*#ifdef __linux__
      long pages = sysconf(_SC_AVPHYS_PAGES);
      long page_size = sysconf(_SC_PAGE_SIZE);
      return pages * page_size >> 20;
  #endif*/
#ifdef _WIN64
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return size_t(status.ullAvailPhys) >> 20;
#endif
} // end of getAvailableMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get peak CPU memory usage in MB.
 */
size_t KSpaceFirstOrderSolver::getPeakHostMemoryUsage() const {
#ifdef __linux__
  // linux file contains this-process info
  FILE* file = fopen("/proc/self/status", "r");
  char buffer[1024] = "";
  int peakRealMem = 0;
  //int peakVirtMem;
  // read the entire file
  while (fscanf(file, " %1023s", buffer) == 1) {
    if (strcmp(buffer, "VmHWM:") == 0) { // kilobytes
      fscanf(file, " %d", &peakRealMem);
    }
    /*if (strcmp(buffer, "VmPeak:") == 0)
      {
        fscanf(file, " %d", &peakVirtMem);
      }*/
  }
  fclose(file);
  return size_t(peakRealMem) >> 10;
  //return size_t(peakVirtMem) >> 10;
#endif
#ifdef _WIN64
  PROCESS_MEMORY_COUNTERS pmc;
  GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
  return size_t(pmc.PeakWorkingSetSize) >> 20;
#endif
} // end of getPeakMemoryUsage
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get current CPU memory usage in MB.
 */
size_t KSpaceFirstOrderSolver::getCurrentHostMemoryUsage() const {
#ifdef __linux__
  // linux file contains this-process info
  FILE* file = fopen("/proc/self/status", "r");
  char buffer[1024] = "";
  int currRealMem = 0;
  //int currVirtMem;
  // read the entire file
  while (fscanf(file, " %1023s", buffer) == 1) {
    if (strcmp(buffer, "VmRSS:") == 0) { // kilobytes
      fscanf(file, " %d", &currRealMem);
    }
    /*if (strcmp(buffer, "VmSize:") == 0)
      {
        fscanf(file, " %d", &currVirtMem);
      }*/
  }
  fclose(file);
  return size_t(currRealMem) >> 10;
  //return size_t(currVirtMem) >> 10;
#endif
#ifdef _WIN64
  PROCESS_MEMORY_COUNTERS pmc;
  GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
  return size_t(pmc.WorkingSetSize) >> 20;
#endif
} // end of getCurrentMemoryUsage
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get total CPU memory in MB.
 */
size_t KSpaceFirstOrderSolver::getTotalHostMemory() const {
#ifdef __linux__
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size >> 20;
#endif
#ifdef _WIN64
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return size_t(status.ullTotalPhys) >> 20;
#endif
} // end of getTotalMemory

/**
 * Get peak GPU memory usage in MB.
 */
size_t KSpaceFirstOrderSolver::getDeviceMemoryUsage() const {
  size_t free, total;
  cudaMemGetInfo(&free, &total);

  return ((total - free) >> 20);
} // end of getDeviceMemoryUsage
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get available GPU memory in MB.
 */
size_t KSpaceFirstOrderSolver::getAvailableDeviceMemory() const {
  size_t free, total;
  cudaMemGetInfo(&free, &total);

  return ((free) >> 20);
} // end of getAvailableDeviceMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get release code version.
 */
const string KSpaceFirstOrderSolver::getCodeName() const {
  return string(kOutFmtKWaveVersion);
} // end of getCodeName
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print full code name and the license.
 */
void KSpaceFirstOrderSolver::printFullCodeNameAndLicense() const {
  Logger::log(Logger::LogLevel::kBasic,
              kOutFmtBuildNoDataTime,
              10, 11, __DATE__,
              8, 8, __TIME__);

  if (mParameters.getGitHash() != "") {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtVersionGitHash, mParameters.getGitHash().c_str());
  }
  Logger::log(Logger::LogLevel::kBasic, kOutFmtSeparator);

// OS detection
#ifdef __linux__
  Logger::log(Logger::LogLevel::kBasic, kOutFmtLinuxBuild);
#elif __APPLE__
  Logger::log(Logger::LogLevel::kBasic, kOutFmtMacOsBuild);
#elif _WIN32
  Logger::log(Logger::LogLevel::kBasic, kOutFmtWindowsBuild);
#endif

// Compiler detections
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
  Logger::log(Logger::LogLevel::kBasic, kOutFmtGnuCompiler, __VERSION__);
#endif
#ifdef __INTEL_COMPILER
  Logger::log(Logger::LogLevel::kBasic, kOutFmtIntelCompiler, __INTEL_COMPILER);
#endif
#ifdef _MSC_VER
  Logger::log(Logger::LogLevel::kBasic, kOutFmtVisualStudioCompiler, _MSC_VER);
#endif

  // instruction set
#if (defined(__AVX2__))
  Logger::log(Logger::LogLevel::kBasic, kOutFmtAVX2);
#elif (defined(__AVX__))
  Logger::log(Logger::LogLevel::kBasic, kOutFmtAVX);
#elif (defined(__SSE4_2__))
  Logger::log(Logger::LogLevel::kBasic, kOutFmtSSE42);
#elif (defined(__SSE4_1__))
  Logger::log(Logger::LogLevel::kBasic, kOutFmtSSE41);
#elif (defined(__SSE3__))
  Logger::log(Logger::LogLevel::kBasic, kOutFmtSSE3);
#elif (defined(__SSE2__))
  Logger::log(Logger::LogLevel::kBasic, kOutFmtSSE2);
#endif

  Logger::log(Logger::LogLevel::kBasic, kOutFmtSeparator);

  // CUDA detection
  int cudaRuntimeVersion;
  if (cudaRuntimeGetVersion(&cudaRuntimeVersion) != cudaSuccess) {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtCudaRuntimeNA);
  } else {
    Logger::log(Logger::LogLevel::kBasic,
                ((cudaRuntimeVersion / 1000) < 10) ? kOutFmtCudaRuntime : kOutFmtCudaRuntime10,
                cudaRuntimeVersion / 1000, (cudaRuntimeVersion % 100) / 10);
  }

  int cudaDriverVersion;
  cudaDriverGetVersion(&cudaDriverVersion);
  Logger::log(Logger::LogLevel::kBasic,
              ((cudaDriverVersion / 1000) < 10) ? kOutFmtCudaDriver : kOutFmtCudaDriver10,
              cudaDriverVersion / 1000, (cudaDriverVersion % 100) / 10);

  const CudaParameters& cudaParameters = mParameters.getCudaParameters();
  // no GPU was found
  if (cudaParameters.getDeviceIdx() == CudaParameters::kDefaultDeviceIdx) {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtCudaDeviceInfoNA);
  } else {
    Logger::log(Logger::LogLevel::kBasic,
                kOutFmtCudaCodeArch,
                SolverCudaKernels::getCudaCodeVersion() / 10.f);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtSeparator);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtCudaDevice, cudaParameters.getDeviceIdx());

    const int paddingLength = static_cast<int>(65 - (22 + cudaParameters.getDeviceName().length()));

    Logger::log(Logger::LogLevel::kBasic,
                kOutFmtCudaDeviceName,
                cudaParameters.getDeviceName().c_str(),
                paddingLength,
                kOutFmtCudaDeviceNamePadding.c_str());

    Logger::log(Logger::LogLevel::kBasic,
                kOutFmtCudaCapability,
                cudaParameters.getDeviceProperties().major,
                cudaParameters.getDeviceProperties().minor);
  }

  Logger::log(Logger::LogLevel::kBasic, kOutFmtLicense);
} // end of printFullCodeNameAndLicense
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Initialize CUDA FFT plans.
 */
void KSpaceFirstOrderSolver::initializeCufftPlans() {
  // create real to complex plans
  CufftComplexMatrix::createR2CFftPlanND(mParameters.getFullDimensionSizes());

  // create complex to real plans
  CufftComplexMatrix::createC2RFftPlanND(mParameters.getFullDimensionSizes());

  // if necessary, create 1D shift plans.
  // in this case, the matrix has a bit bigger dimensions to be able to store shifted matrices.
  if (Parameters::getInstance().getStoreVelocityNonStaggeredRawFlag() || Parameters::getInstance().getStoreVelocityNonStaggeredCFlag() || Parameters::getInstance().getStoreIntensityAvgFlag() || Parameters::getInstance().getStoreQTermFlag() || Parameters::getInstance().getStoreIntensityAvgCFlag() || Parameters::getInstance().getStoreQTermCFlag()) {
    // X shifts
    CufftComplexMatrix::createR2CFftPlan1DX(mParameters.getFullDimensionSizes());
    CufftComplexMatrix::createC2RFftPlan1DX(mParameters.getFullDimensionSizes());

    // Y shifts
    CufftComplexMatrix::createR2CFftPlan1DY(mParameters.getFullDimensionSizes());
    CufftComplexMatrix::createC2RFftPlan1DY(mParameters.getFullDimensionSizes());

    // Z shifts
    if (mParameters.isSimulation3D()) {
      CufftComplexMatrix::createR2CFftPlan1DZ(mParameters.getFullDimensionSizes());
      CufftComplexMatrix::createC2RFftPlan1DZ(mParameters.getFullDimensionSizes());
    }
  } // end u_non_staggered
} // end of initializeCufftPlans
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute pre-processing phase.
 */
template <Parameters::SimulationDimension simulationDimension>
void KSpaceFirstOrderSolver::preProcessing() {
  // get the correct sensor mask and recompute indices
  if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kIndex) {
    getSensorMaskIndex().recomputeIndicesToCPP();
  }

  if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kCorners) {
    getSensorMaskCorners().recomputeIndicesToCPP();
  }

  if (!mParameters.getOnlyPostProcessingFlag()) {
    if ((mParameters.getTransducerSourceFlag() != 0) || (mParameters.getVelocityXSourceFlag() != 0) || (mParameters.getVelocityYSourceFlag() != 0) || (mParameters.getVelocityZSourceFlag() != 0)) {
      getVelocitySourceIndex().recomputeIndicesToCPP();
    }

    if (mParameters.getTransducerSourceFlag() != 0) {
      getDelayMask().recomputeIndicesToCPP();
    }

    if (mParameters.getPressureSourceFlag() != 0) {
      getPressureSourceIndex().recomputeIndicesToCPP();
    }

    // compute dt / rho0_sg...
    if (!mParameters.getRho0ScalarFlag()) { // non-uniform grid cannot be pre-calculated :-(
      // rho is matrix
      if (mParameters.getNonUniformGridFlag()) {
        generateInitialDenisty<simulationDimension>();
      } else {
        getDtRho0Sgx().scalarDividedBy(mParameters.getDt());
        getDtRho0Sgy().scalarDividedBy(mParameters.getDt());
        if (simulationDimension == SD::k3D) {
          getDtRho0Sgz().scalarDividedBy(mParameters.getDt());
        }
      }
    }

    // generate different matrices
    if (mParameters.getAbsorbingFlag() != 0) {
      generateKappaAndNablas();
      generateTauAndEta();
    } else {
      generateKappa();
    }

    // Generate sourceKappa
    if (((mParameters.getVelocitySourceMode() == Parameters::SourceMode::kAdditive) || (mParameters.getPressureSourceMode() == Parameters::SourceMode::kAdditive)) && (mParameters.getPressureSourceFlag() || mParameters.getVelocityXSourceFlag() || mParameters.getVelocityYSourceFlag() || mParameters.getVelocityZSourceFlag())) {
      generateSourceKappa();
    }

    // calculate c^2. It has to be after kappa gen... because of c modification
    computeC2();
  }
} // end of preProcessing
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute the main time loop of KSpaceFirstOrder solver.
 */
template <Parameters::SimulationDimension simulationDimension>
void KSpaceFirstOrderSolver::computeMainLoop() {
  mActPercent = 0.0f;
  // if resuming from a checkpoint,
  // set ActPercent to correspond the t_index after recovery
  if (mParameters.getTimeIndex() > 0) {
    // We're restarting after checkpoint
    mIsTimestepRightAfterRestore = true;
    mActPercent = float(100.0f * mParameters.getTimeIndex()) / mParameters.getNt();
  }

  // Progress header
  Logger::log(Logger::LogLevel::kBasic, kOutFmtSimulationHeader);

  // Initial copy of data to the GPU
  mMatrixContainer.copyMatricesToDevice();

  mIterationTime.start();

  // execute main loop
  while ((mParameters.getTimeIndex() < mParameters.getNt()) && (!mParameters.isTimeToCheckpoint(mTotalTime))) {
    const size_t timeIndex = mParameters.getTimeIndex();
    // compute velocity
    computeVelocity<simulationDimension>();
    // add in the velocity source term
    addVelocitySource();

    // add in the transducer source term (t = t1) to ux
    if (mParameters.getTransducerSourceFlag() > timeIndex) {
      SolverCudaKernels::addTransducerSource(mMatrixContainer);
    }

    // compute gradient of velocity
    computeVelocityGradient<simulationDimension>();

    // compute density
    if (mParameters.getNonLinearFlag()) {
      computeDensityNonliner<simulationDimension>();
    } else {
      computeDensityLinear<simulationDimension>();
    }

    // add in the source pressure term
    addPressureSource<simulationDimension>();

    if (mParameters.getNonLinearFlag()) {
      computePressureNonlinear<simulationDimension>();
    } else {
      computePressureLinear<simulationDimension>();
    }

    // calculate initial pressure
    if ((timeIndex == 0) && (mParameters.getInitialPressureSourceFlag() == 1)) {
      addInitialPressureSource<simulationDimension>();
    }

    storeSensorData();
    printStatistics();

    mParameters.incrementTimeIndex();
    mIsTimestepRightAfterRestore = false;
  }

  // Since disk operations are one step delayed, we have to do the last one here.
  // However we need to check if the loop wasn't skipped due to very short checkpoint interval
  if (mParameters.getTimeIndex() > mParameters.getSamplingStartTimeIndex() && (!mIsTimestepRightAfterRestore)) {
    mOutputStreamContainer.flushRawStreams();
  }
} // end of computeMainLoop()
//----------------------------------------------------------------------------------------------------------------------

/**
 * Post processing, and closing the output streams.
 */
template <Parameters::SimulationDimension simulationDimension>
void KSpaceFirstOrderSolver::postProcessing() {
  if (mParameters.getStorePressureFinalAllFlag()) {
    getP().copyFromDevice();
    getP().writeData(mParameters.getOutputFile(),
                     kPressureFinalName,
                     mParameters.getCompressionLevel());
  } // p_final

  if (mParameters.getStoreVelocityFinalAllFlag()) {
    getUxSgx().copyFromDevice();
    getUySgy().copyFromDevice();
    if (mParameters.isSimulation3D()) {
      getUzSgz().copyFromDevice();
    }

    getUxSgx().writeData(mParameters.getOutputFile(),
                         kUxFinalName,
                         mParameters.getCompressionLevel());
    getUySgy().writeData(mParameters.getOutputFile(),
                         kUyFinalName,
                         mParameters.getCompressionLevel());
    if (mParameters.isSimulation3D()) {
      getUzSgz().writeData(mParameters.getOutputFile(),
                           kUzFinalName,
                           mParameters.getCompressionLevel());
    }
  } // u_final

  if (mParameters.getStoreIntensityAvgFlag() ||
      mParameters.getStoreIntensityAvgCFlag() ||
      mParameters.getStoreQTermFlag() ||
      mParameters.getStoreQTermCFlag()) {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtNoDone);
  }

  // Compute average intensity from stored p and u without compression
  if (mParameters.getStoreQTermFlag() ||
      mParameters.getStoreIntensityAvgFlag()) {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtComputingAverageIntensity);
    computeAverageIntensities<simulationDimension>();
    Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
  }

  // Compute average intensity from stored p and u compression coefficients
  if (mParameters.getOnlyPostProcessingFlag() &&
      (mParameters.getStoreQTermCFlag() ||
       mParameters.getStoreIntensityAvgCFlag())) {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtComputingAverageIntensityC);
    computeAverageIntensitiesC<simulationDimension>();
    Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
  }

  // Apply post-processing, flush data on disk/
  mOutputStreamContainer.postProcessStreams();

  // Compute and store Q term (volume rate of heat deposition) from average intensity
  if (mParameters.getStoreQTermFlag()) {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtComputingQTerm);
    computeQTerm<simulationDimension>(OutputStreamContainer::OutputStreamIdx::kIntensityXAvg,
                                      OutputStreamContainer::OutputStreamIdx::kIntensityYAvg,
                                      OutputStreamContainer::OutputStreamIdx::kIntensityZAvg,
                                      OutputStreamContainer::OutputStreamIdx::kQTerm);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
  }

  // Compute and store Q term (volume rate of heat deposition) from average intensity computed using compression
  if (mParameters.getStoreQTermCFlag()) {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtComputingQTermC);
    computeQTerm<simulationDimension>(OutputStreamContainer::OutputStreamIdx::kIntensityXAvgC,
                                      OutputStreamContainer::OutputStreamIdx::kIntensityYAvgC,
                                      OutputStreamContainer::OutputStreamIdx::kIntensityZAvgC,
                                      OutputStreamContainer::OutputStreamIdx::kQTermC);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
  }

  if (mParameters.getStoreIntensityAvgFlag() ||
      mParameters.getStoreIntensityAvgCFlag() ||
      mParameters.getStoreQTermFlag() ||
      mParameters.getStoreQTermCFlag()) {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtEmpty);
  }

  // Apply post-processing 2
  mOutputStreamContainer.postProcessStreams2();

  // Close
  mOutputStreamContainer.closeStreams();

  // store sensor mask if wanted
  if (mParameters.getCopySensorMaskFlag()) {
    if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kIndex && !mParameters.getOutputFile().datasetExists(mParameters.getOutputFile().getRootGroup(), kSensorMaskIndexName)) {
      getSensorMaskIndex().recomputeIndicesToMatlab();
      getSensorMaskIndex().writeData(mParameters.getOutputFile(), kSensorMaskIndexName,
                                     mParameters.getCompressionLevel());
    }
    if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kCorners && !mParameters.getOutputFile().datasetExists(mParameters.getOutputFile().getRootGroup(), kSensorMaskCornersName)) {
      getSensorMaskCorners().recomputeIndicesToMatlab();
      getSensorMaskCorners().writeData(mParameters.getOutputFile(), kSensorMaskCornersName,
                                       mParameters.getCompressionLevel());
    }
  }
} // end of postProcessing
//----------------------------------------------------------------------------------------------------------------------

/**
 * Store sensor data.
 */
void KSpaceFirstOrderSolver::storeSensorData() {
  // Unless the time for sampling has come, exit.
  if (mParameters.getTimeIndex() >= mParameters.getSamplingStartTimeIndex()) {
    // Read event for t_index-1. If sampling did not occur by then, ignored it.
    // if it did store data on disk (flush) - the GPU is running asynchronously.
    // But be careful, flush has to be one step delayed to work correctly.
    // when restoring from checkpoint we have to skip the first flush
    if (mParameters.getTimeIndex() > mParameters.getSamplingStartTimeIndex() && !mIsTimestepRightAfterRestore) {
      mOutputStreamContainer.flushRawStreams();
    }

    // if --u_non_staggered is switched on, calculate unstaggered velocity.
    if (mParameters.getStoreVelocityNonStaggeredRawFlag() || mParameters.getStoreVelocityNonStaggeredCFlag() || mParameters.getStoreIntensityAvgFlag() || mParameters.getStoreQTermFlag() || mParameters.getStoreIntensityAvgCFlag() || mParameters.getStoreQTermCFlag()) {
      if (mParameters.isSimulation3D()) {
        computeShiftedVelocity<SD::k3D>();
      } else {
        computeShiftedVelocity<SD::k2D>();
      }
    }

    // Sample data for step t  (store event for sampling in next turn)
    mOutputStreamContainer.sampleStreams();
    // the last step (or data after) checkpoint are flushed in the main loop
  }
} // end of storeSensorData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write statistics and the header into the output file.
 */
void KSpaceFirstOrderSolver::writeOutputDataInfo() {
  // write timeIndex into the output file
  mParameters.getOutputFile().writeScalarValue(mParameters.getOutputFile().getRootGroup(),
                                               kTimeIndexName,
                                               mParameters.getTimeIndex());

  // Write scalars
  mParameters.saveScalarsToOutputFile();
  Hdf5FileHeader& fileHeader = mParameters.getFileHeader();

  // Write File header
  fileHeader.setCodeName(getCodeName());
  fileHeader.setMajorFileVersion();
  fileHeader.setMinorFileVersion();
  fileHeader.setActualCreationTime();
  fileHeader.setFileType(Hdf5FileHeader::FileType::kOutput);
  fileHeader.setHostName();

  fileHeader.setMemoryConsumption(getHostMemoryUsage());

  // Stop total timer here
  mTotalTime.stop();
  fileHeader.setExecutionTimes(getCumulatedTotalTime(),
                               getCumulatedDataLoadTime(),
                               getCumulatedPreProcessingTime(),
                               getCumulatedSimulationTime(),
                               getCumulatedPostProcessingTime());

  fileHeader.setNumberOfCores();
  fileHeader.writeHeaderToOutputFile(mParameters.getOutputFile());

  // Write some other info
  hid_t rootGroup = mParameters.getOutputFile().getRootGroup();
  mParameters.getOutputFile().writeStringAttribute(rootGroup, "/", "mStoreQTermFlag", std::to_string(mParameters.getStoreQTermFlag()));
  mParameters.getOutputFile().writeStringAttribute(rootGroup, "/", "mStoreQTermCFlag", std::to_string(mParameters.getStoreQTermCFlag()));
  mParameters.getOutputFile().writeStringAttribute(rootGroup, "/", "mStoreIntensityAvgFlag", std::to_string(mParameters.getStoreIntensityAvgFlag()));
  mParameters.getOutputFile().writeStringAttribute(rootGroup, "/", "mStoreIntensityAvgCFlag", std::to_string(mParameters.getStoreIntensityAvgCFlag()));
  mParameters.getOutputFile().writeStringAttribute(rootGroup, "/", "mNoCompressionOverlapFlag", std::to_string(mParameters.getNoCompressionOverlapFlag()));
  mParameters.getOutputFile().writeStringAttribute(rootGroup, "/", "mPeriod", std::to_string((&CompressHelper::getInstance())->getPeriod()));
  mParameters.getOutputFile().writeStringAttribute(rootGroup, "/", "mMOS", std::to_string((&CompressHelper::getInstance())->getMos()));
  mParameters.getOutputFile().writeStringAttribute(rootGroup, "/", "mHarmonics", std::to_string((&CompressHelper::getInstance())->getHarmonics()));
  mParameters.getOutputFile().writeStringAttribute(rootGroup, "/", "mBlockSizeDefault", std::to_string(mBlockSizeDefault));
  mParameters.getOutputFile().writeStringAttribute(rootGroup, "/", "mBlockSizeDefaultC", std::to_string(mBlockSizeDefaultC));
  mParameters.getOutputFile().writeStringAttribute(rootGroup, "/", "mSamplingStartTimeStep", std::to_string(mParameters.getSamplingStartTimeIndex()));
  mParameters.getOutputFile().writeStringAttribute(rootGroup, "/", "output_file_size", std::to_string(mParameters.getOutputFile().getFileSize()));
  mParameters.getOutputFile().writeStringAttribute(rootGroup, "/", "simulation_peak_host_memory_in_use", Logger::formatMessage("%ld MB", mSimulationPeakHostMemoryConsumption));
  mParameters.getOutputFile().writeStringAttribute(rootGroup, "/", "simulation_peak_device_memory_in_use", Logger::formatMessage("%ld MB", mSimulationPeakDeviceMemoryConsumption));
  mParameters.getOutputFile().writeStringAttribute(rootGroup, "/", "average_sampling_iteration_time", Logger::formatMessage("%8.2fs", mAverageSamplingIterationTime));
  mParameters.getOutputFile().writeStringAttribute(rootGroup, "/", "sampling_time", Logger::formatMessage("%8.2fs", mSamplingTime));
  mParameters.getOutputFile().writeStringAttribute(rootGroup, "/", "average_not_sampling_iteration_time", Logger::formatMessage("%8.2fs", mAverageNotSamplingIterationTime));
  mParameters.getOutputFile().writeStringAttribute(rootGroup, "/", "not_sampling_time", Logger::formatMessage("%8.2fs", mNotSamplingTime));
} // end of writeOutputDataInfo
//----------------------------------------------------------------------------------------------------------------------

/**
 * Save checkpoint data into the checkpoint file, flush aggregated outputs into the output file.
 */
void KSpaceFirstOrderSolver::saveCheckpointData() {
  // Create Checkpoint file
  Hdf5File& checkpointFile = mParameters.getCheckpointFile();
  // if it happens and the file is opened (from the recovery, close it)
  if (checkpointFile.isOpen())
    checkpointFile.close();

  Logger::log(Logger::LogLevel::kFull, kOutFmtStoringCheckpointData);
  Logger::flush(Logger::LogLevel::kFull);

  // Create the new file (overwrite the old one)
  checkpointFile.create(mParameters.getCheckpointFileName());

  //------------------------------------------------ Store Matrices --------------------------------------------------//
  // Store all necessary matrices in Checkpoint file
  mMatrixContainer.storeDataIntoCheckpointFile();

  // Write t_index
  checkpointFile.writeScalarValue(checkpointFile.getRootGroup(), kTimeIndexName, mParameters.getTimeIndex());

  // store basic dimension sizes (Nx, Ny, Nz) - Nt is not necessary
  checkpointFile.writeScalarValue(checkpointFile.getRootGroup(), kNxName, mParameters.getFullDimensionSizes().nx);
  checkpointFile.writeScalarValue(checkpointFile.getRootGroup(), kNyName, mParameters.getFullDimensionSizes().ny);
  checkpointFile.writeScalarValue(checkpointFile.getRootGroup(), kNzName, mParameters.getFullDimensionSizes().nz);

  // Write checkpoint file header
  Hdf5FileHeader fileHeader = mParameters.getFileHeader();

  fileHeader.setFileType(Hdf5FileHeader::FileType::kCheckpoint);
  fileHeader.setCodeName(getCodeName());
  fileHeader.setActualCreationTime();

  fileHeader.writeHeaderToCheckpointFile(checkpointFile);

  Logger::log(Logger::LogLevel::kFull, kOutFmtDone);

  // checkpoint output streams only if necessary (t_index > start_index), we're here one step ahead!
  if (mParameters.getTimeIndex() > mParameters.getSamplingStartTimeIndex()) {
    Logger::log(Logger::LogLevel::kFull, kOutFmtStoringSensorData);
    Logger::flush(Logger::LogLevel::kFull);
    mOutputStreamContainer.checkpointStreams();
    Logger::log(Logger::LogLevel::kFull, kOutFmtDone);
  }
  mOutputStreamContainer.closeStreams();
  // Close the checkpoint file
  checkpointFile.close();
} // end of saveCheckpointData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute average intensities from stored p and u without compression.
 */
template <Parameters::SimulationDimension simulationDimension>
void KSpaceFirstOrderSolver::computeAverageIntensities() {
  mOutputStreamContainer[OutputStreamContainer::OutputStreamIdx::kIntensityXAvg].zeroCurrentStoreBuffer();
  float* intensityXAvgData = mOutputStreamContainer[OutputStreamContainer::OutputStreamIdx::kIntensityXAvg].getCurrentStoreBuffer();
  mOutputStreamContainer[OutputStreamContainer::OutputStreamIdx::kIntensityYAvg].zeroCurrentStoreBuffer();
  float* intensityYAvgData = mOutputStreamContainer[OutputStreamContainer::OutputStreamIdx::kIntensityYAvg].getCurrentStoreBuffer();
  if (simulationDimension == SD::k3D)
    mOutputStreamContainer[OutputStreamContainer::OutputStreamIdx::kIntensityZAvg].zeroCurrentStoreBuffer();
  float* intensityZAvgData = (simulationDimension == SD::k3D) ? mOutputStreamContainer[OutputStreamContainer::OutputStreamIdx::kIntensityZAvg].getCurrentStoreBuffer() : nullptr;

  size_t steps = 0;
  if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kIndex) {
    steps = mParameters.getOutputFile().getDatasetDimensionSizes(mParameters.getOutputFile().getRootGroup(), kPName).ny;
  } else if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kCorners) {
    steps = mParameters.getOutputFile().getDatasetDimensionSizes(mParameters.getOutputFile().getRootGroup(), kPName + "/1").nt;
  }

  // Compute shifts
  const float pi2 = static_cast<float>(M_PI) * 2.0f;
  const size_t stepsComplex = steps / 2 + 1;
  FloatComplex* kx = reinterpret_cast<FloatComplex*>(_mm_malloc(stepsComplex * sizeof(FloatComplex), kDataAlignment));
  //#pragma omp simd - Intel exp bug
  for (size_t i = 0; i < stepsComplex; i++) {
    const ssize_t shift = ssize_t((i + (steps / 2)) % steps - (steps / 2));
    kx[i] = exp(FloatComplex(0.0f, 1.0f) * (pi2 * 0.5f) * (float(shift) / float(steps)));
  }

  size_t numberOfDatasets = 1;
  if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kCorners) {
    numberOfDatasets = getSensorMaskCorners().getDimensionSizes().ny;
  }

  hid_t datasetP = 0;
  hid_t datasetUx = 0;
  hid_t datasetUy = 0;
  hid_t datasetUz = 0;

  size_t maxBlockSize = mParameters.getBlockSize();
  DimensionSizes fullDims = mParameters.getFullDimensionSizes();

  // Compute max block size for dataset reading
  if (maxBlockSize == 0) {
    // To bytes
    size_t deviceMemory = getAvailableDeviceMemory() << 20;
    size_t hostMemory = getAvailableHostMemory() << 20;
    // std::cout << std::endl;
    // std::cout << "getHostMemoryUsage     " << getHostMemoryUsage() << std::endl;
    // std::cout << "getTotalMemory:        " << getTotalMemory() << std::endl;
    // std::cout << "getCurrentMemoryUsage: " << getCurrentMemoryUsage() << std::endl;
    // std::cout << "getPeakMemoryUsage:    " << getPeakMemoryUsage() << std::endl;
    // std::cout << "getAvailableMemory:    " << getAvailableMemory() << std::endl;
    //  dataP, dataUx, dataUy, dataUz, fftwTimeShiftMatrix -> 5 matrices x 4 (size of float)
    maxBlockSize = std::min(size_t(float(hostMemory) / 20 * 0.98f), size_t(float(deviceMemory) / 20 * 0.90f));
  }

  size_t sliceSize = fullDims.nx * fullDims.ny;
  size_t fullSize = fullDims.nx * fullDims.ny * fullDims.nz;
  mBlockSizeDefault = maxBlockSize / steps;
  // Minimal size is sliceSize
  mBlockSizeDefault = mBlockSizeDefault < sliceSize ? sliceSize : mBlockSizeDefault;
  // Maximal size is fullSize
  mBlockSizeDefault = mBlockSizeDefault > fullSize ? fullSize : mBlockSizeDefault;
  DimensionSizes shiftDims(mBlockSizeDefault, steps / 2 + 1, 1);
  mBlockSizeDefault *= steps;
  // Temporary matrix for time shift
  CufftComplexMatrix* fftwTimeShiftMatrix = new CufftComplexMatrix(shiftDims);

  //std::cout << "maxBlockSize:          " << maxBlockSize <<  std::endl;
  //std::cout << "mBlockSizeDefault:     " << mBlockSizeDefault << std::endl;
  //std::cout << std::endl;

  // For every dataset
  // TODO test with more cuboids
  for (size_t indexOfDataset = 1; indexOfDataset <= numberOfDatasets; indexOfDataset++) {
    size_t datasetSize = 0;
    DimensionSizes datasetDimensionSizes;
    size_t cSliceSize = 0;
    // Block size for given dataset
    size_t blockSize = fftwTimeShiftMatrix->getDimensionSizes().nx;
    DimensionSizes datasetBlockSizes;

    // Open datasets
    if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kIndex) {
      datasetP = mParameters.getOutputFile().openDataset(mParameters.getOutputFile().getRootGroup(), kPName);
      datasetUx = mParameters.getOutputFile().openDataset(mParameters.getOutputFile().getRootGroup(), kUxNonStaggeredName);
      datasetUy = mParameters.getOutputFile().openDataset(mParameters.getOutputFile().getRootGroup(), kUyNonStaggeredName);
      if (simulationDimension == SD::k3D) {
        datasetUz = mParameters.getOutputFile().openDataset(mParameters.getOutputFile().getRootGroup(), kUzNonStaggeredName);
      }
      datasetSize = mParameters.getOutputFile().getDatasetSize(mParameters.getOutputFile().getRootGroup(), kPName) / steps;
      // Maximum size is datasetSize
      if (blockSize > datasetSize) {
        blockSize = datasetSize;
      }
      datasetBlockSizes = DimensionSizes(blockSize, steps, 1);
    } else if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kCorners) {
      datasetP = mParameters.getOutputFile().openDataset(mParameters.getOutputFile().getRootGroup(), kPName + "/" + std::to_string(indexOfDataset));
      datasetUx = mParameters.getOutputFile().openDataset(mParameters.getOutputFile().getRootGroup(), kUxNonStaggeredName + "/" + std::to_string(indexOfDataset));
      datasetUy = mParameters.getOutputFile().openDataset(mParameters.getOutputFile().getRootGroup(), kUyNonStaggeredName + "/" + std::to_string(indexOfDataset));
      if (simulationDimension == SD::k3D) {
        datasetUz = mParameters.getOutputFile().openDataset(mParameters.getOutputFile().getRootGroup(), kUzNonStaggeredName + "/" + std::to_string(indexOfDataset));
      }
      datasetSize = mParameters.getOutputFile().getDatasetSize(mParameters.getOutputFile().getRootGroup(), kPName + "/" + std::to_string(indexOfDataset)) / steps;
      datasetDimensionSizes = mParameters.getOutputFile().getDatasetDimensionSizes(mParameters.getOutputFile().getRootGroup(), kPName + "/" + std::to_string(indexOfDataset));
      cSliceSize = datasetDimensionSizes.nx * datasetDimensionSizes.ny;
      // Maximum size is datasetSize
      if (blockSize > datasetSize) {
        blockSize = datasetSize;
      }
      size_t zCount = blockSize / cSliceSize;
      blockSize = cSliceSize * zCount;
      datasetBlockSizes = DimensionSizes(datasetDimensionSizes.nx, datasetDimensionSizes.ny, zCount, steps);
    }

    Logger::log(Logger::LogLevel::kBasic, kOutFmtNoDone);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtBlockSizePostProcessing, (Logger::formatMessage(kOutFmt2DDomainSizeFormat, blockSize, steps)).c_str(), (blockSize * steps * 4) >> 20);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtEmpty);

    RealMatrix dataP(DimensionSizes(blockSize, steps, 1));
    RealMatrix dataUx(DimensionSizes(blockSize, steps, 1));
    RealMatrix dataUy(DimensionSizes(blockSize, steps, 1));
    RealMatrix dataUz(DimensionSizes(blockSize, steps, 1));
    fftwTimeShiftMatrix->createR2CFftPlan1DY(dataP.getDimensionSizes());
    fftwTimeShiftMatrix->createC2RFftPlan1DY(dataP.getDimensionSizes());

    for (size_t i = 0; i < datasetSize; i += blockSize) {
      // Read block by sensor mask type
      if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kIndex) {
        if (i + blockSize > datasetSize) {
          blockSize = datasetSize - i;
          datasetBlockSizes = DimensionSizes(blockSize, steps, 1);
          dataP.resize(DimensionSizes(blockSize, steps, 1));
          dataUx.resize(DimensionSizes(blockSize, steps, 1));
          dataUy.resize(DimensionSizes(blockSize, steps, 1));
          dataUz.resize(DimensionSizes(blockSize, steps, 1));
          fftwTimeShiftMatrix->createR2CFftPlan1DY(dataP.getDimensionSizes());
          fftwTimeShiftMatrix->createC2RFftPlan1DY(dataP.getDimensionSizes());
        }

        mParameters.getOutputFile().readHyperSlab(datasetP, DimensionSizes(i, 0, 0), datasetBlockSizes, dataP.getHostData());
        mParameters.getOutputFile().readHyperSlab(datasetUx, DimensionSizes(i, 0, 0), datasetBlockSizes, dataUx.getHostData());
        mParameters.getOutputFile().readHyperSlab(datasetUy, DimensionSizes(i, 0, 0), datasetBlockSizes, dataUy.getHostData());
        if (simulationDimension == SD::k3D) {
          mParameters.getOutputFile().readHyperSlab(datasetUz, DimensionSizes(i, 0, 0), datasetBlockSizes, dataUz.getHostData());
        }
      } else if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kCorners) {
        if (i + blockSize > datasetSize) {
          blockSize = datasetSize - i;
          datasetBlockSizes = DimensionSizes(datasetDimensionSizes.nx,
                                             datasetDimensionSizes.ny,
                                             blockSize / cSliceSize,
                                             steps);
          dataP.resize(DimensionSizes(blockSize, steps, 1));
          dataUx.resize(DimensionSizes(blockSize, steps, 1));
          dataUy.resize(DimensionSizes(blockSize, steps, 1));
          dataUz.resize(DimensionSizes(blockSize, steps, 1));
          fftwTimeShiftMatrix->createR2CFftPlan1DY(dataP.getDimensionSizes());
          fftwTimeShiftMatrix->createC2RFftPlan1DY(dataP.getDimensionSizes());
        }
        size_t zOffset = i / cSliceSize;

        mParameters.getOutputFile().readHyperSlab(datasetP, DimensionSizes(0, 0, zOffset, 0), datasetBlockSizes, dataP.getHostData());
        mParameters.getOutputFile().readHyperSlab(datasetUx, DimensionSizes(0, 0, zOffset, 0), datasetBlockSizes, dataUx.getHostData());
        mParameters.getOutputFile().readHyperSlab(datasetUy, DimensionSizes(0, 0, zOffset, 0), datasetBlockSizes, dataUy.getHostData());
        if (simulationDimension == SD::k3D) {
          mParameters.getOutputFile().readHyperSlab(datasetUz, DimensionSizes(0, 0, zOffset, 0), datasetBlockSizes, dataUz.getHostData());
        }
      }

      // Phase shifts using FFT
      const float divider = 1.0f / static_cast<float>(steps);
      FloatComplex* tempFftTimeShift = fftwTimeShiftMatrix->getComplexHostData();
      dataUx.copyToDevice(); // TODO use CUDA kernel
      fftwTimeShiftMatrix->computeR2CFft1DY(dataUx);
      fftwTimeShiftMatrix->copyFromDevice();
#pragma omp parallel for schedule(static)
      for (size_t step = 0; step < stepsComplex; step++) {
        for (size_t x = 0; x < blockSize; x++) {
          // TODO X/Y is transposed
          tempFftTimeShift[x * stepsComplex + step] *= divider * kx[step];
        }
      }
      fftwTimeShiftMatrix->copyToDevice();
      fftwTimeShiftMatrix->computeC2RFft1DY(dataUx);
      dataUx.copyFromDevice();

      dataUy.copyToDevice();
      fftwTimeShiftMatrix->computeR2CFft1DY(dataUy);
      fftwTimeShiftMatrix->copyFromDevice();
#pragma omp parallel for schedule(static)
      for (size_t step = 0; step < stepsComplex; step++) {
        for (size_t x = 0; x < blockSize; x++) {
          tempFftTimeShift[x * stepsComplex + step] *= divider * kx[step];
        }
      }
      fftwTimeShiftMatrix->copyToDevice();
      fftwTimeShiftMatrix->computeC2RFft1DY(dataUy);
      dataUy.copyFromDevice();

      if (simulationDimension == SD::k3D) {
        dataUz.copyToDevice();
        fftwTimeShiftMatrix->computeR2CFft1DY(dataUz);
        fftwTimeShiftMatrix->copyFromDevice();
#pragma omp parallel for schedule(static)
        for (size_t step = 0; step < stepsComplex; step++) {
          for (size_t x = 0; x < blockSize; x++) {
            tempFftTimeShift[x * stepsComplex + step] *= divider * kx[step];
          }
        }
        fftwTimeShiftMatrix->copyToDevice();
        fftwTimeShiftMatrix->computeC2RFft1DY(dataUz);
        dataUz.copyFromDevice();
      }

      // Compute average of intensity
      for (size_t step = 0; step < steps; step++) {
        for (size_t x = 0; x < blockSize; x++) {
          intensityXAvgData[i + x] += dataUx[step * blockSize + x] * dataP[step * blockSize + x];
          intensityYAvgData[i + x] += dataUy[step * blockSize + x] * dataP[step * blockSize + x];
          if (simulationDimension == SD::k3D) {
            intensityZAvgData[i + x] += dataUz[step * blockSize + x] * dataP[step * blockSize + x];
          }
          if (step == steps - 1) {
            intensityXAvgData[i + x] /= steps;
            intensityYAvgData[i + x] /= steps;
            if (simulationDimension == SD::k3D) {
              intensityZAvgData[i + x] /= steps;
            }
          }
        }
      }
    }

    mParameters.getOutputFile().closeDataset(datasetP);
    mParameters.getOutputFile().closeDataset(datasetUx);
    mParameters.getOutputFile().closeDataset(datasetUy);
    if (simulationDimension == SD::k3D) {
      mParameters.getOutputFile().closeDataset(datasetUz);
    }
  }
  _mm_free(kx);
  delete fftwTimeShiftMatrix;

} // end of computeAverageIntensities
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute average intensities from stored p and u compression coefficients.
 *
 * NOTE does not work with 40-bit complex floats
 */
template <Parameters::SimulationDimension simulationDimension>
void KSpaceFirstOrderSolver::computeAverageIntensitiesC() {
  float* intensityXAvgData = mOutputStreamContainer[OutputStreamContainer::OutputStreamIdx::kIntensityXAvgC].getCurrentStoreBuffer();
  float* intensityYAvgData = mOutputStreamContainer[OutputStreamContainer::OutputStreamIdx::kIntensityYAvgC].getCurrentStoreBuffer();
  float* intensityZAvgData = (simulationDimension == SD::k3D) ? mOutputStreamContainer[OutputStreamContainer::OutputStreamIdx::kIntensityZAvgC].getCurrentStoreBuffer() : nullptr;
  size_t steps = 1;
  if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kIndex) {
    steps = mParameters.getOutputFile().getDatasetDimensionSizes(mParameters.getOutputFile().getRootGroup(), kPName + kCompressSuffix).ny;
  } else if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kCorners) {
    steps = mParameters.getOutputFile().getDatasetDimensionSizes(mParameters.getOutputFile().getRootGroup(), kPName + kCompressSuffix + "/1").nt;
  }

  size_t numberOfDatasets = 1;
  if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kCorners) {
    numberOfDatasets = getSensorMaskCorners().getDimensionSizes().ny;
  }

  hid_t datasetP = 0;
  hid_t datasetUx = 0;
  hid_t datasetUy = 0;
  hid_t datasetUz = 0;

  size_t maxBlockSize = mParameters.getBlockSize();
  DimensionSizes fullDims = mParameters.getFullDimensionSizes();
  fullDims.nx *= CompressHelper::getInstance().getHarmonics() * 2; // TODO size of complex (40-bit complex floats)

  // Compute max block size for dataset reading
  if (maxBlockSize == 0) {
    size_t deviceMemory = getAvailableDeviceMemory() << 20;
    size_t hostMemory = getAvailableHostMemory() << 20;
    //  dataP, dataUx, dataUy, dataUz, fftwTimeShiftMatrix -> 5 matrices x 4 (size of float)
    maxBlockSize = std::min(size_t(float(hostMemory) / 20 * 0.98f), size_t(float(deviceMemory) / 20 * 0.90f));
  }
  size_t sliceSize = fullDims.nx * fullDims.ny;
  size_t fullSize = fullDims.nx * fullDims.ny * fullDims.nz;
  mBlockSizeDefaultC = maxBlockSize / steps;
  // Minimal size is sliceSize
  mBlockSizeDefaultC = mBlockSizeDefaultC < sliceSize ? sliceSize : mBlockSizeDefaultC;
  // Maximal size is fullSize
  mBlockSizeDefaultC = mBlockSizeDefaultC > fullSize ? fullSize : mBlockSizeDefaultC;
  mBlockSizeDefault *= steps;

  // For every dataset
  // TODO test with more cuboids
  for (size_t indexOfDataset = 1; indexOfDataset <= numberOfDatasets; indexOfDataset++) {
    size_t datasetSize = 0;
    DimensionSizes datasetDimensionSizes;
    size_t cSliceSize = 0;
    // Block size for given dataset
    size_t blockSize = mBlockSizeDefaultC / steps;
    DimensionSizes datasetBlockSizes;

    // Open datasets
    if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kIndex) {
      datasetP = mParameters.getOutputFile().openDataset(mParameters.getOutputFile().getRootGroup(), kPName + kCompressSuffix);
      datasetUx = mParameters.getOutputFile().openDataset(mParameters.getOutputFile().getRootGroup(), kUxNonStaggeredName + kCompressSuffix);
      datasetUy = mParameters.getOutputFile().openDataset(mParameters.getOutputFile().getRootGroup(), kUyNonStaggeredName + kCompressSuffix);
      if (simulationDimension == SD::k3D) {
        datasetUz = mParameters.getOutputFile().openDataset(mParameters.getOutputFile().getRootGroup(), kUzNonStaggeredName + kCompressSuffix);
      }
      datasetSize = mParameters.getOutputFile().getDatasetSize(mParameters.getOutputFile().getRootGroup(), kPName + kCompressSuffix) / steps;
      // Maximum size is datasetSize
      if (blockSize > datasetSize) {
        blockSize = datasetSize;
      }
      datasetBlockSizes = DimensionSizes(blockSize, steps, 1);
    } else if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kCorners) {
      datasetP = mParameters.getOutputFile().openDataset(mParameters.getOutputFile().getRootGroup(), kPName + kCompressSuffix + "/" + std::to_string(indexOfDataset));
      datasetUx = mParameters.getOutputFile().openDataset(mParameters.getOutputFile().getRootGroup(), kUxNonStaggeredName + kCompressSuffix + "/" + std::to_string(indexOfDataset));
      datasetUy = mParameters.getOutputFile().openDataset(mParameters.getOutputFile().getRootGroup(), kUyNonStaggeredName + kCompressSuffix + "/" + std::to_string(indexOfDataset));
      if (simulationDimension == SD::k3D) {
        datasetUz = mParameters.getOutputFile().openDataset(mParameters.getOutputFile().getRootGroup(), kUzNonStaggeredName + kCompressSuffix + "/" + std::to_string(indexOfDataset));
      }
      datasetSize = mParameters.getOutputFile().getDatasetSize(mParameters.getOutputFile().getRootGroup(), kPName + kCompressSuffix + "/" + std::to_string(indexOfDataset)) / steps;
      datasetDimensionSizes = mParameters.getOutputFile().getDatasetDimensionSizes(mParameters.getOutputFile().getRootGroup(), kPName + kCompressSuffix + "/" + std::to_string(indexOfDataset));
      cSliceSize = datasetDimensionSizes.nx * datasetDimensionSizes.ny;
      // Maximum size is datasetSize
      if (blockSize > datasetSize) {
        blockSize = datasetSize;
      }
      size_t zCount = blockSize / cSliceSize;
      blockSize = cSliceSize * zCount;
      datasetBlockSizes = DimensionSizes(datasetDimensionSizes.nx, datasetDimensionSizes.ny, zCount, steps);
    }

    Logger::log(Logger::LogLevel::kBasic, kOutFmtNoDone);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtBlockSizePostProcessing, (Logger::formatMessage(kOutFmt2DDomainSizeFormat, blockSize, steps)).c_str(), (blockSize * steps * 4) >> 20);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtEmpty);

    RealMatrix dataP(DimensionSizes(blockSize, steps, 1));
    RealMatrix dataUx(DimensionSizes(blockSize, steps, 1));
    RealMatrix dataUy(DimensionSizes(blockSize, steps, 1));
    RealMatrix dataUz(DimensionSizes(blockSize, steps, 1));

    for (size_t i = 0; i < datasetSize; i += blockSize) {
      const size_t outputIndex = i / (CompressHelper::getInstance().getHarmonics() * 2); // TODO size of complex (40-bit complex floats)

      // Read block by sensor mask type
      if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kIndex) {
        if (i + blockSize > datasetSize) {
          blockSize = datasetSize - i;
          datasetBlockSizes = DimensionSizes(blockSize, steps, 1);
          dataP.resize(DimensionSizes(blockSize, steps, 1));
          dataUx.resize(DimensionSizes(blockSize, steps, 1));
          dataUy.resize(DimensionSizes(blockSize, steps, 1));
          dataUz.resize(DimensionSizes(blockSize, steps, 1));
        }

        mParameters.getOutputFile().readHyperSlab(datasetP, DimensionSizes(i, 0, 0), datasetBlockSizes, dataP.getHostData());
        mParameters.getOutputFile().readHyperSlab(datasetUx, DimensionSizes(i, 0, 0), datasetBlockSizes, dataUx.getHostData());
        mParameters.getOutputFile().readHyperSlab(datasetUy, DimensionSizes(i, 0, 0), datasetBlockSizes, dataUy.getHostData());
        if (simulationDimension == SD::k3D) {
          mParameters.getOutputFile().readHyperSlab(datasetUz, DimensionSizes(i, 0, 0), datasetBlockSizes, dataUz.getHostData());
        }
      } else if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kCorners) {
        if (i + blockSize > datasetSize) {
          blockSize = datasetSize - i;
          datasetBlockSizes = DimensionSizes(datasetDimensionSizes.nx,
                                             datasetDimensionSizes.ny,
                                             blockSize / cSliceSize,
                                             steps);
          dataP.resize(DimensionSizes(blockSize, steps, 1));
          dataUx.resize(DimensionSizes(blockSize, steps, 1));
          dataUy.resize(DimensionSizes(blockSize, steps, 1));
          dataUz.resize(DimensionSizes(blockSize, steps, 1));
        }
        size_t zOffset = i / cSliceSize;

        mParameters.getOutputFile().readHyperSlab(datasetP, DimensionSizes(0, 0, zOffset, 0), datasetBlockSizes, dataP.getHostData());
        mParameters.getOutputFile().readHyperSlab(datasetUx, DimensionSizes(0, 0, zOffset, 0), datasetBlockSizes, dataUx.getHostData());
        mParameters.getOutputFile().readHyperSlab(datasetUy, DimensionSizes(0, 0, zOffset, 0), datasetBlockSizes, dataUy.getHostData());
        if (simulationDimension == SD::k3D) {
          mParameters.getOutputFile().readHyperSlab(datasetUz, DimensionSizes(0, 0, zOffset, 0), datasetBlockSizes, dataUz.getHostData());
        }
      }

      FloatComplex* bufferP = reinterpret_cast<FloatComplex*>(dataP.getHostData());
      FloatComplex* bufferUx = reinterpret_cast<FloatComplex*>(dataUx.getHostData());
      FloatComplex* bufferUy = reinterpret_cast<FloatComplex*>(dataUy.getHostData());
      FloatComplex* bufferUz = reinterpret_cast<FloatComplex*>(dataUz.getHostData());

      size_t outputBlockSize = blockSize / (CompressHelper::getInstance().getHarmonics() * 2); // TODO size of complex (40-bit complex floats)
      // Compute average of intensity
      for (size_t step = 0; step < steps; step++) {
        for (size_t x = 0; x < outputBlockSize; x++) {
          size_t offset = step * (blockSize / 2) + CompressHelper::getInstance().getHarmonics() * x;
          // For every harmonics
          for (size_t ih = 0; ih < CompressHelper::getInstance().getHarmonics(); ih++) {
            size_t pH = offset + ih;
            intensityXAvgData[outputIndex + x] += real(bufferP[pH] * conj(bufferUx[pH])) / 2.0f;
            intensityYAvgData[outputIndex + x] += real(bufferP[pH] * conj(bufferUy[pH])) / 2.0f;
            if (simulationDimension == SD::k3D) {
              intensityZAvgData[outputIndex + x] += real(bufferP[pH] * conj(bufferUz[pH])) / 2.0f;
            }
          }
          if (step == steps - 1) {
            intensityXAvgData[outputIndex + x] /= steps;
            intensityYAvgData[outputIndex + x] /= steps;
            if (simulationDimension == SD::k3D) {
              intensityZAvgData[outputIndex + x] /= steps;
            }
          }
        }
      }
    }

    mParameters.getOutputFile().closeDataset(datasetP);
    mParameters.getOutputFile().closeDataset(datasetUx);
    mParameters.getOutputFile().closeDataset(datasetUy);
    if (simulationDimension == SD::k3D) {
      mParameters.getOutputFile().closeDataset(datasetUz);
    }
  }
} // end of computeAverageIntensitiesC
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute Q term (volume rate of heat deposition) from average intensities.
 */
template <Parameters::SimulationDimension simulationDimension>
void KSpaceFirstOrderSolver::computeQTerm(OutputStreamContainer::OutputStreamIdx intensityXAvgStreamIndex,
                                          OutputStreamContainer::OutputStreamIdx intensityYAvgStreamIndex,
                                          OutputStreamContainer::OutputStreamIdx intensityZAvgStreamIndex,
                                          OutputStreamContainer::OutputStreamIdx qTermStreamIdx) {
  float* intensityXAvgData = mOutputStreamContainer[intensityXAvgStreamIndex].getCurrentStoreBuffer();
  float* intensityYAvgData = mOutputStreamContainer[intensityYAvgStreamIndex].getCurrentStoreBuffer();
  float* intensityZAvgData = (simulationDimension == SD::k3D) ? mOutputStreamContainer[intensityZAvgStreamIndex].getCurrentStoreBuffer() : nullptr;

  // Full sized matrices for intensities
  RealMatrix intensityXAvg(mParameters.getFullDimensionSizes());
  RealMatrix intensityYAvg(mParameters.getFullDimensionSizes());
  RealMatrix intensityZAvg(mParameters.getFullDimensionSizes());

  const DimensionSizes& fullDimensionSizes = mParameters.getFullDimensionSizes();

  // Copy values from store buffer to positions defined by sensor mask indices
  if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kIndex) {
    const size_t sensorMaskSize = getSensorMaskIndex().capacity();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < sensorMaskSize; i++) {
      intensityXAvg[getSensorMaskIndex()[i]] = intensityXAvgData[i];
      intensityYAvg[getSensorMaskIndex()[i]] = intensityYAvgData[i];
      if (simulationDimension == SD::k3D) {
        intensityZAvg[getSensorMaskIndex()[i]] = intensityZAvgData[i];
      }
    }
  }

  // Copy values from store buffer to positions defined by sensor mask corners
  if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kCorners) {
    const size_t slabSize = fullDimensionSizes.ny * fullDimensionSizes.nx;
    const size_t rowSize = fullDimensionSizes.nx;
    const size_t nCuboids = getSensorMaskCorners().getDimensionSizes().ny;
    size_t cuboidInBufferStart = 0;

#pragma omp parallel
    for (size_t cuboidIdx = 0; cuboidIdx < nCuboids; cuboidIdx++) {
      const DimensionSizes topLeftCorner = getSensorMaskCorners().getTopLeftCorner(cuboidIdx);
      const DimensionSizes bottomRightCorner = getSensorMaskCorners().getBottomRightCorner(cuboidIdx);

      size_t cuboidSlabSize = (bottomRightCorner.ny - topLeftCorner.ny + 1) * (bottomRightCorner.nx - topLeftCorner.nx + 1);
      size_t cuboidRowSize = (bottomRightCorner.nx - topLeftCorner.nx + 1);

      DimensionSizes cuboidSize(0, 0, 0, 0); // Size of the cuboid
      cuboidSize = bottomRightCorner - topLeftCorner;
      cuboidSize.nt = 1;

#pragma omp for schedule(static)
      for (size_t z = topLeftCorner.nz; z <= bottomRightCorner.nz; z++) {
        for (size_t y = topLeftCorner.ny; y <= bottomRightCorner.ny; y++) {
          for (size_t x = topLeftCorner.nx; x <= bottomRightCorner.nx; x++) {
            const size_t storeBufferIndex = cuboidInBufferStart + (z - topLeftCorner.nz) * cuboidSlabSize + (y - topLeftCorner.ny) * cuboidRowSize + (x - topLeftCorner.nx);
            const size_t sourceIndex = z * slabSize + y * rowSize + x;
            intensityXAvg[sourceIndex] = intensityXAvgData[storeBufferIndex];
            intensityYAvg[sourceIndex] = intensityYAvgData[storeBufferIndex];
            if (simulationDimension == SD::k3D) {
              intensityZAvg[sourceIndex] = intensityZAvgData[storeBufferIndex];
            }
          }
        }
      }
// must be done only once
#pragma omp single
      {
        cuboidInBufferStart += cuboidSize.nElements();
      }
    }
  }

  // Helper complex memory
  FloatComplex* tempFftShift = getTempCufftShift().getComplexHostData();

  const float pi2 = static_cast<float>(M_PI) * 2.0f;

  // Normalization constants for FFT
  const float dividerX = 1.0f / static_cast<float>(fullDimensionSizes.nx);
  const float dividerY = 1.0f / static_cast<float>(fullDimensionSizes.ny);
  const float dividerZ = 1.0f / static_cast<float>(fullDimensionSizes.nz);
  // Helper values
  const size_t nx = fullDimensionSizes.nx;
  const size_t ny = fullDimensionSizes.ny;
  const size_t nz = fullDimensionSizes.nz;
  const float dx = mParameters.getDx();
  const float dy = mParameters.getDy();
  const float dz = mParameters.getDz();

  DimensionSizes xShiftDims = mParameters.getFullDimensionSizes();
  xShiftDims.nx = xShiftDims.nx / 2 + 1;
  DimensionSizes yShiftDims = mParameters.getFullDimensionSizes();
  yShiftDims.ny = yShiftDims.ny / 2 + 1;
  DimensionSizes yShiftDimsTXY = mParameters.getFullDimensionSizes();
  yShiftDimsTXY.nx = yShiftDims.ny;
  yShiftDimsTXY.ny = yShiftDims.nx;
  DimensionSizes zShiftDims = mParameters.getFullDimensionSizes();
  zShiftDims.nz = zShiftDims.nz / 2 + 1;
  DimensionSizes zShiftDimsTXZ = mParameters.getFullDimensionSizes();
  zShiftDimsTXZ.nx = zShiftDims.nz;
  zShiftDimsTXZ.nz = zShiftDims.nx;

  // Helper memory for shifts
  FloatComplex* kx = reinterpret_cast<FloatComplex*>(_mm_malloc(xShiftDims.nx * sizeof(FloatComplex), kDataAlignment));
  FloatComplex* ky = reinterpret_cast<FloatComplex*>(_mm_malloc(yShiftDims.ny * sizeof(FloatComplex), kDataAlignment));
  FloatComplex* kz = (simulationDimension == SD::k3D) ? reinterpret_cast<FloatComplex*>(_mm_malloc(zShiftDims.nz * sizeof(FloatComplex), kDataAlignment)) : nullptr;

  // Compute shifts for x gradient
  for (size_t i = 0; i < xShiftDims.nx; i++) {
    const ssize_t shift = ssize_t((i + (nx / 2)) % nx - (nx / 2));
    kx[i] = FloatComplex(0.0f, 1.0f) * (pi2 / dx) * (float(shift) / float(nx));
  }
// Compute shifts for y gradient
  for (size_t i = 0; i < yShiftDims.ny; i++) {
    const ssize_t shift = ssize_t((i + (ny / 2)) % ny - (ny / 2));
    ky[i] = FloatComplex(0.0f, 1.0f) * (pi2 / dy) * (float(shift) / float(ny));
  }
  if (simulationDimension == SD::k3D) {
// Compute shifts for z gradient
    for (size_t i = 0; i < zShiftDims.nz; i++) {
      const ssize_t shift = ssize_t((i + (nz / 2)) % nz - (nz / 2));
      kz[i] = FloatComplex(0.0f, 1.0f) * (pi2 / dz) * (float(shift) / float(nz));
    }
  }

  // X shifts
  CufftComplexMatrix::createR2CFftPlan1DX(mParameters.getFullDimensionSizes());
  CufftComplexMatrix::createC2RFftPlan1DX(mParameters.getFullDimensionSizes());

  // Y shifts
  CufftComplexMatrix::createR2CFftPlan1DY(mParameters.getFullDimensionSizes());
  CufftComplexMatrix::createC2RFftPlan1DY(mParameters.getFullDimensionSizes());

  // Z shifts
  if (mParameters.isSimulation3D()) {
    CufftComplexMatrix::createR2CFftPlan1DZ(mParameters.getFullDimensionSizes());
    CufftComplexMatrix::createC2RFftPlan1DZ(mParameters.getFullDimensionSizes());
  }

  intensityXAvg.copyToDevice();
  getTempCufftShift().computeR2CFft1DX(intensityXAvg);
  getTempCufftShift().copyFromDevice();
#pragma omp parallel for schedule(static) if (simulationDimension == SD::k3D)
  for (size_t z = 0; z < xShiftDims.nz; z++) {
#pragma omp parallel for schedule(static) if (simulationDimension == SD::k2D)
    for (size_t y = 0; y < xShiftDims.ny; y++) {
      for (size_t x = 0; x < xShiftDims.nx; x++) {
        const size_t i = get1DIndex(z, y, x, xShiftDims);
        tempFftShift[i] *= dividerX * kx[x];
      } // x
    }   // y
  }     // z
  getTempCufftShift().copyToDevice();
  getTempCufftShift().computeC2RFft1DX(intensityXAvg);
  intensityXAvg.copyFromDevice();

  intensityYAvg.copyToDevice();
  getTempCufftShift().computeR2CFft1DY(intensityYAvg);
  getTempCufftShift().copyFromDevice();
#pragma omp parallel for schedule(static) if (simulationDimension == SD::k3D)
  for (size_t z = 0; z < yShiftDims.nz; z++) {
#pragma omp parallel for schedule(static) if (simulationDimension == SD::k2D)
    for (size_t y = 0; y < yShiftDims.ny; y++) {
      for (size_t x = 0; x < yShiftDims.nx; x++) {
        // TODO transposed X/Y
        const size_t i = get1DIndex(z, x, y, yShiftDimsTXY);
        tempFftShift[i] *= dividerY * ky[y];
      } // x
    }   // y
  }     // z
  getTempCufftShift().copyToDevice();
  getTempCufftShift().computeC2RFft1DY(intensityYAvg);
  intensityYAvg.copyFromDevice();

  if (simulationDimension == SD::k3D) {
    intensityZAvg.copyToDevice();
    getTempCufftShift().computeR2CFft1DZ(intensityZAvg);
    getTempCufftShift().copyFromDevice();
#pragma omp parallel for schedule(static)
    for (size_t z = 0; z < zShiftDims.nz; z++) {
#pragma omp parallel for schedule(static)
      for (size_t y = 0; y < zShiftDims.ny; y++) {
        for (size_t x = 0; x < zShiftDims.nx; x++) {
          // TODO transposed X/Z
          const size_t i = get1DIndex(x, y, z, zShiftDimsTXZ);
          tempFftShift[i] *= dividerZ * kz[z];
        } // x
      }   // y
    }     // z
    getTempCufftShift().copyToDevice();
    getTempCufftShift().computeC2RFft1DZ(intensityZAvg);
    intensityZAvg.copyFromDevice();
  }

  _mm_free(kx);
  _mm_free(ky);
  _mm_free(kz);

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < fullDimensionSizes.nElements(); i++) {
    if (simulationDimension == SD::k3D) {
      intensityXAvg[i] = -(intensityXAvg[i] + intensityYAvg[i] + intensityZAvg[i]);
    } else {
      intensityXAvg[i] = -(intensityXAvg[i] + intensityYAvg[i]);
    }
  }

  mOutputStreamContainer[qTermStreamIdx].zeroCurrentStoreBuffer();
  float* qTermData = mOutputStreamContainer[qTermStreamIdx].getCurrentStoreBuffer();

  // Copy values from positions defined by sensor mask indices to store buffer
  if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kIndex) {
    const size_t sensorMaskSize = getSensorMaskIndex().capacity();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < sensorMaskSize; i++) {
      qTermData[i] = intensityXAvg[getSensorMaskIndex()[i]];
    }
  }

  // Copy values from positions defined by sensor mask corners to store buffer
  if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kCorners) {
    const size_t slabSize = fullDimensionSizes.ny * fullDimensionSizes.nx;
    const size_t rowSize = fullDimensionSizes.nx;
    const size_t nCuboids = getSensorMaskCorners().getDimensionSizes().ny;
    size_t cuboidInBufferStart = 0;

#pragma omp parallel
    for (size_t cuboidIdx = 0; cuboidIdx < nCuboids; cuboidIdx++) {
      const DimensionSizes topLeftCorner = getSensorMaskCorners().getTopLeftCorner(cuboidIdx);
      const DimensionSizes bottomRightCorner = getSensorMaskCorners().getBottomRightCorner(cuboidIdx);

      size_t cuboidSlabSize = (bottomRightCorner.ny - topLeftCorner.ny + 1) * (bottomRightCorner.nx - topLeftCorner.nx + 1);
      size_t cuboidRowSize = (bottomRightCorner.nx - topLeftCorner.nx + 1);

      DimensionSizes cuboidSize(0, 0, 0, 0); // Size of the cuboid
      cuboidSize = bottomRightCorner - topLeftCorner;
      cuboidSize.nt = 1;

#pragma omp for schedule(static)
      for (size_t z = topLeftCorner.nz; z <= bottomRightCorner.nz; z++) {
        for (size_t y = topLeftCorner.ny; y <= bottomRightCorner.ny; y++) {
          for (size_t x = topLeftCorner.nx; x <= bottomRightCorner.nx; x++) {
            const size_t storeBufferIndex = cuboidInBufferStart + (z - topLeftCorner.nz) * cuboidSlabSize + (y - topLeftCorner.ny) * cuboidRowSize + (x - topLeftCorner.nx);
            const size_t sourceIndex = z * slabSize + y * rowSize + x;
            qTermData[storeBufferIndex] = intensityXAvg[sourceIndex];
          }
        }
      }
// must be done only once
#pragma omp single
      {
        cuboidInBufferStart += cuboidSize.nElements();
      }
    }
  }
} // end of computeQTerm
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute new values of acoustic velocity in all used dimensions (UxSgx, UySgy, UzSgz).
 */
template <Parameters::SimulationDimension simulationDimension>
void KSpaceFirstOrderSolver::computeVelocity() {
  // fftn(p);
  getTempCufftX().computeR2CFftND(getP());

  // bsxfun(@times, ddx_k_shift_pos, kappa .* pre_result) , for all 3 dims
  SolverCudaKernels::computePressureGradient<simulationDimension>(mMatrixContainer);

  // ifftn(pre_result)
  getTempCufftX().computeC2RFftND(getTemp1RealND());
  getTempCufftY().computeC2RFftND(getTemp2RealND());
  if (simulationDimension == SD::k3D) {
    getTempCufftZ().computeC2RFftND(getTemp3RealND());
  }

  // bsxfun(@times, pml_x_sgx, bsxfun(@times, pml_x_sgx, ux_sgx) - dt .* rho0_sgx_inv .* (pre_result))
  if (mParameters.getRho0ScalarFlag()) { // scalars
    if (mParameters.getNonUniformGridFlag()) {
      SolverCudaKernels::computeVelocityHomogeneousNonuniform<simulationDimension>(mMatrixContainer);
    } else {
      SolverCudaKernels::computeVelocityHomogeneousUniform<simulationDimension>(mMatrixContainer);
    }
  } else { // matrices
    SolverCudaKernels::computeVelocityHeterogeneous<simulationDimension>(mMatrixContainer);
  }
} // end of computeVelocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute new gradient of velocity (duxdx, duydy, duzdz).
 */
template <Parameters::SimulationDimension simulationDimension>
void KSpaceFirstOrderSolver::computeVelocityGradient() {
  getTempCufftX().computeR2CFftND(getUxSgx());
  getTempCufftY().computeR2CFftND(getUySgy());
  if (simulationDimension == SD::k3D) {
    getTempCufftZ().computeR2CFftND(getUzSgz());
  }

  // calculate Duxyz on uniform grid
  SolverCudaKernels::computeVelocityGradient<simulationDimension>(mMatrixContainer);

  getTempCufftX().computeC2RFftND(getDuxdx());
  getTempCufftY().computeC2RFftND(getDuydy());
  if (simulationDimension == SD::k3D) {
    getTempCufftZ().computeC2RFftND(getDuzdz());
  }

  // Non-uniform grid
  if (mParameters.getNonUniformGridFlag() != 0) {
    SolverCudaKernels::computeVelocityGradientShiftNonuniform<simulationDimension>(mMatrixContainer);
  } // non-uniform grid
} // end of computeVelocityGradient
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate new values of acoustic density for non-linear case (rhoX, rhoy and rhoZ).
 */
template <Parameters::SimulationDimension simulationDimension>
void KSpaceFirstOrderSolver::computeDensityNonliner() {
  SolverCudaKernels::computeDensityNonlinear<simulationDimension>(mMatrixContainer);
  ;

} // end of computeDensityNonliner
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate new values of acoustic density for linear case (rhoX, rhoy and rhoZ).
 */
template <Parameters::SimulationDimension simulationDimension>
void KSpaceFirstOrderSolver::computeDensityLinear() {
  SolverCudaKernels::computeDensityLinear<simulationDimension>(mMatrixContainer);

} // end of computeDensityLinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute acoustic pressure for non-linear case.
 */
template <Parameters::SimulationDimension simulationDimension>
void KSpaceFirstOrderSolver::computePressureNonlinear() {
  if (mParameters.getAbsorbingFlag()) { // absorbing case
    RealMatrix& densitySum = getTemp1RealND();
    RealMatrix& nonlinearTerm = getTemp2RealND();
    RealMatrix& velocityGradientSum = getTemp3RealND();

    // reusing of the temp variables
    RealMatrix& absorbTauTerm = velocityGradientSum;
    RealMatrix& absorbEtaTerm = densitySum;

    // Compute three temporary sums in the new pressure formula, non-linear absorbing case.
    SolverCudaKernels::computePressureTermsNonlinear<simulationDimension>(densitySum,
                                                                          nonlinearTerm,
                                                                          velocityGradientSum,
                                                                          mMatrixContainer);

    getTempCufftX().computeR2CFftND(velocityGradientSum);
    getTempCufftY().computeR2CFftND(densitySum);

    SolverCudaKernels::computeAbsorbtionTerm(getTempCufftX(),
                                             getTempCufftY(),
                                             getAbsorbNabla1(),
                                             getAbsorbNabla2());

    getTempCufftX().computeC2RFftND(absorbTauTerm);
    getTempCufftY().computeC2RFftND(absorbEtaTerm);

    SolverCudaKernels::sumPressureTermsNonlinear(nonlinearTerm, absorbTauTerm, absorbEtaTerm, mMatrixContainer);
  } else {
    SolverCudaKernels::sumPressureNonlinearLossless<simulationDimension>(mMatrixContainer);
  }
} // end of computePressureNonlinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute new p for linear case.
 */
template <Parameters::SimulationDimension simulationDimension>
void KSpaceFirstOrderSolver::computePressureLinear() {
  if (mParameters.getAbsorbingFlag()) { // absorbing case
    RealMatrix& densitySum = getTemp1RealND();
    RealMatrix& velocityGradientTerm = getTemp2RealND();

    RealMatrix& absorbTauTerm = getTemp2RealND();
    RealMatrix& absorbEtaTerm = getTemp3RealND();

    SolverCudaKernels::computePressureTermsLinear<simulationDimension>(densitySum,
                                                                       velocityGradientTerm,
                                                                       mMatrixContainer);

    // ifftn ( absorbNabla1 * fftn (rho0 * (duxdx + duydy + duzdz))
    getTempCufftX().computeR2CFftND(velocityGradientTerm);
    getTempCufftY().computeR2CFftND(densitySum);

    SolverCudaKernels::computeAbsorbtionTerm(getTempCufftX(),
                                             getTempCufftY(),
                                             getAbsorbNabla1(),
                                             getAbsorbNabla2());

    getTempCufftX().computeC2RFftND(absorbTauTerm);
    getTempCufftY().computeC2RFftND(absorbEtaTerm);

    SolverCudaKernels::sumPressureTermsLinear(absorbTauTerm, absorbEtaTerm, densitySum, mMatrixContainer);
  } else {
    SolverCudaKernels::sumPressureLinearLossless<simulationDimension>(mMatrixContainer);
  }
} // end of computePressureLinear()
//----------------------------------------------------------------------------------------------------------------------

/**
 * Add velocity source to the particle velocity.
 */
void KSpaceFirstOrderSolver::addVelocitySource() {
  size_t timeIndex = mParameters.getTimeIndex();

  if (mParameters.getVelocitySourceMode() != Parameters::SourceMode::kAdditive) { // executed Dirichlet and AdditiveNoCorrection source
    if (mParameters.getVelocityXSourceFlag() > timeIndex) {
      SolverCudaKernels::addVelocitySource(getUxSgx(),
                                           getVelocityXSourceInput(),
                                           getVelocitySourceIndex());
    }
    if (mParameters.getVelocityYSourceFlag() > timeIndex) {
      SolverCudaKernels::addVelocitySource(getUySgy(),
                                           getVelocityYSourceInput(),
                                           getVelocitySourceIndex());
    }

    if ((mParameters.isSimulation3D()) && (mParameters.getVelocityZSourceFlag() > timeIndex)) {
      SolverCudaKernels::addVelocitySource(getUzSgz(),
                                           getVelocityZSourceInput(),
                                           getVelocitySourceIndex());
    }
  } else { // execute Additive source
    RealMatrix& scaledSource = getTemp1RealND();

    if (mParameters.getVelocityXSourceFlag() > timeIndex) {
      scaleSource(scaledSource,
                  getVelocityXSourceInput(),
                  getVelocitySourceIndex(),
                  mParameters.getVelocitySourceMany());

      // Insert source
      SolverCudaKernels::addVelocityScaledSource(getUxSgx(), scaledSource);
    }

    if (mParameters.getVelocityYSourceFlag() > timeIndex) {
      scaleSource(scaledSource,
                  getVelocityYSourceInput(),
                  getVelocitySourceIndex(),
                  mParameters.getVelocitySourceMany());

      // Insert source
      SolverCudaKernels::addVelocityScaledSource(getUySgy(), scaledSource);
    }

    if ((mParameters.isSimulation3D()) && (mParameters.getVelocityZSourceFlag() > timeIndex)) {
      scaleSource(scaledSource,
                  getVelocityZSourceInput(),
                  getVelocitySourceIndex(),
                  mParameters.getVelocitySourceMany());

      // Insert source
      SolverCudaKernels::addVelocityScaledSource(getUzSgz(), scaledSource);
    }
  }
} // end of addVelocitySource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Add in pressure source.
 */
template <Parameters::SimulationDimension simulationDimension>
void KSpaceFirstOrderSolver::addPressureSource() {
  size_t timeIndex = mParameters.getTimeIndex();

  if (mParameters.getPressureSourceFlag() > timeIndex) {
    if (mParameters.getPressureSourceMode() != Parameters::SourceMode::kAdditive) { // executed Dirichlet and AdditiveNoCorrection source
      SolverCudaKernels::addPressureSource<simulationDimension>(mMatrixContainer);
    } else { // execute Additive source
      RealMatrix& scaledSource = getTemp1RealND();

      scaleSource(scaledSource,
                  getPressureSourceInput(),
                  getPressureSourceIndex(),
                  mParameters.getPressureSourceMany());

      // Insert source
      SolverCudaKernels::addPressureScaledSource<simulationDimension>(mMatrixContainer, scaledSource);

    } // Additive source
  }   // apply source
} // end of AddPressureSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Scale source signal
 */
void KSpaceFirstOrderSolver::scaleSource(RealMatrix& scaledSource,
                                         const RealMatrix& sourceInput,
                                         const IndexMatrix& sourceIndex,
                                         const size_t manyFlag) {
  // Zero source scaling matrix on GPU.
  scaledSource.zeroDeviceMatrix();
  // Inject source to scaling matrix
  SolverCudaKernels::insertSourceIntoScalingMatrix(scaledSource,
                                                   sourceInput,
                                                   sourceIndex,
                                                   manyFlag);
  // Compute FFT
  getTempCufftX().computeR2CFftND(scaledSource);
  // Calculate gradient
  SolverCudaKernels::computeSourceGradient(getTempCufftX(), getSourceKappa());
  // Compute iFFT
  getTempCufftX().computeC2RFftND(scaledSource);
} // end of scaleSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate p0 source.
 */
template <Parameters::SimulationDimension simulationDimension>
void KSpaceFirstOrderSolver::addInitialPressureSource() {
  // add the initial pressure to rho as a mass source
  SolverCudaKernels::addInitialPressureSource<simulationDimension>(mMatrixContainer);

  //-----------------------------------------------------------------------//
  //--compute u(t = t1 + dt/2) based on the assumption u(dt/2) = -u(-dt/2)-//
  //--    which forces u(t = t1) = 0                                      -//
  //-----------------------------------------------------------------------//
  getTempCufftX().computeR2CFftND(getP());

  SolverCudaKernels::computePressureGradient<simulationDimension>(mMatrixContainer);

  getTempCufftX().computeC2RFftND(getUxSgx());
  getTempCufftY().computeC2RFftND(getUySgy());
  if (simulationDimension == SD::k3D) {
    getTempCufftZ().computeC2RFftND(getUzSgz());
  }

  if (mParameters.getRho0ScalarFlag()) {
    if (mParameters.getNonUniformGridFlag()) { // non uniform grid, homogeneous
      SolverCudaKernels::computeInitialVelocityHomogeneousNonuniform<simulationDimension>(mMatrixContainer);
    } else { // uniform grid, homogeneous
      SolverCudaKernels::computeInitialVelocityHomogeneousUniform<simulationDimension>(mMatrixContainer);
    }
  } else {
    // heterogeneous, uniform grid
    // divide the matrix by 2 and multiply with st./rho0_sg
    SolverCudaKernels::computeInitialVelocityHeterogeneous<simulationDimension>(mMatrixContainer);
  }
} // end of addInitialPressureSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Generate kappa matrix for lossless medium.
 * For 2D simulation, the zPart == 0.
 */
void KSpaceFirstOrderSolver::generateKappa() {
  const float dx2Rec = 1.0f / (mParameters.getDx() * mParameters.getDx());
  const float dy2Rec = 1.0f / (mParameters.getDy() * mParameters.getDy());
  // For 2D simulation set dz to 0
  const float dz2Rec = (mParameters.isSimulation3D()) ? 1.0f / (mParameters.getDz() * mParameters.getDz()) : 0.0f;

  const float cRefDtPi = mParameters.getCRef() * mParameters.getDt() * static_cast<float>(M_PI);

  const float nxRec = 1.0f / static_cast<float>(mParameters.getFullDimensionSizes().nx);
  const float nyRec = 1.0f / static_cast<float>(mParameters.getFullDimensionSizes().ny);
  // For 2D simulation, nzRec remains 1
  const float nzRec = 1.0f / static_cast<float>(mParameters.getFullDimensionSizes().nz);

  const DimensionSizes& reducedDimensionSizes = mParameters.getReducedDimensionSizes();

  float* kappa = getKappa().getHostData();

#pragma omp parallel for schedule(static) if (mParameters.isSimulation3D())
  for (size_t z = 0; z < reducedDimensionSizes.nz; z++) {
    const float zf = static_cast<float>(z);
    float zPart = 0.5f - fabs(0.5f - zf * nzRec);
    zPart = (zPart * zPart) * dz2Rec;

#pragma omp parallel for schedule(static) if (mParameters.isSimulation2D())
    for (size_t y = 0; y < reducedDimensionSizes.ny; y++) {
      const float yf = static_cast<float>(y);
      float yPart = 0.5f - fabs(0.5f - yf * nyRec);
      yPart = (yPart * yPart) * dy2Rec;

      const float yzPart = zPart + yPart;
      for (size_t x = 0; x < reducedDimensionSizes.nx; x++) {
        const float xf = static_cast<float>(x);
        float xPart = 0.5f - fabs(0.5f - xf * nxRec);
        xPart = (xPart * xPart) * dx2Rec;

        float k = cRefDtPi * sqrt(xPart + yzPart);

        const size_t i = get1DIndex(z, y, x, reducedDimensionSizes);

        // kappa element
        kappa[i] = (k == 0.0f) ? 1.0f : sin(k) / k;
      } // x
    }   // y
  }     // z
} // end of generateKappa
//----------------------------------------------------------------------------------------------------------------------

/**
 * Generate sourceKappa matrix for additive sources.
 * For 2D simulation, the zPart == 0.
 */
void KSpaceFirstOrderSolver::generateSourceKappa() {
  const float dx2Rec = 1.0f / (mParameters.getDx() * mParameters.getDx());
  const float dy2Rec = 1.0f / (mParameters.getDy() * mParameters.getDy());
  const float dz2Rec = (mParameters.isSimulation3D()) ? 1.0f / (mParameters.getDz() * mParameters.getDz()) : 0.0f;

  const float cRefDtPi = mParameters.getCRef() * mParameters.getDt() * static_cast<float>(M_PI);

  const float nxRec = 1.0f / static_cast<float>(mParameters.getFullDimensionSizes().nx);
  const float nyRec = 1.0f / static_cast<float>(mParameters.getFullDimensionSizes().ny);
  const float nzRec = 1.0f / static_cast<float>(mParameters.getFullDimensionSizes().nz);

  const DimensionSizes& reducedDimensionSizes = mParameters.getReducedDimensionSizes();

  float* sourceKappa = getSourceKappa().getHostData();

#pragma omp parallel for schedule(static) if (mParameters.isSimulation3D())
  for (size_t z = 0; z < reducedDimensionSizes.nz; z++) {
    const float zf = static_cast<float>(z);
    float zPart = 0.5f - fabs(0.5f - zf * nzRec);
    zPart = (zPart * zPart) * dz2Rec;

#pragma omp parallel for schedule(static) if (mParameters.isSimulation2D())
    for (size_t y = 0; y < reducedDimensionSizes.ny; y++) {
      const float yf = static_cast<float>(y);
      float yPart = 0.5f - fabs(0.5f - yf * nyRec);
      yPart = (yPart * yPart) * dy2Rec;

      const float yzPart = zPart + yPart;
      for (size_t x = 0; x < reducedDimensionSizes.nx; x++) {
        const float xf = static_cast<float>(x);
        float xPart = 0.5f - fabs(0.5f - xf * nxRec);
        xPart = (xPart * xPart) * dx2Rec;

        float k = cRefDtPi * sqrt(xPart + yzPart);

        const size_t i = get1DIndex(z, y, x, reducedDimensionSizes);

        // sourceKappa element
        sourceKappa[i] = cos(k);
      } // x
    }   // y
  }     // z
} // end of generateSourceKappa
//----------------------------------------------------------------------------------------------------------------------

/**
 * Generate kappa, absorb_nabla1, absorb_nabla2 for absorbing medium.
 * For the 2D simulation the zPart == 0
 */
void KSpaceFirstOrderSolver::generateKappaAndNablas() {
  const float dxSqRec = 1.0f / (mParameters.getDx() * mParameters.getDx());
  const float dySqRec = 1.0f / (mParameters.getDy() * mParameters.getDy());
  const float dzSqRec = (mParameters.isSimulation3D()) ? 1.0f / (mParameters.getDz() * mParameters.getDz()) : 0.0f;

  const float cRefDt2 = mParameters.getCRef() * mParameters.getDt() * 0.5f;
  const float pi2 = static_cast<float>(M_PI) * 2.0f;

  const size_t nx = mParameters.getFullDimensionSizes().nx;
  const size_t ny = mParameters.getFullDimensionSizes().ny;
  const size_t nz = mParameters.getFullDimensionSizes().nz;

  const float nxRec = 1.0f / static_cast<float>(nx);
  const float nyRec = 1.0f / static_cast<float>(ny);
  const float nzRec = 1.0f / static_cast<float>(nz);

  const DimensionSizes& reducedDimensionSizes = mParameters.getReducedDimensionSizes();

  float* kappa = getKappa().getHostData();
  float* absorbNabla1 = getAbsorbNabla1().getHostData();
  float* absorbNabla2 = getAbsorbNabla2().getHostData();
  const float alphaPower = mParameters.getAlphaPower();

#pragma omp parallel for schedule(static) if (mParameters.isSimulation3D())
  for (size_t z = 0; z < reducedDimensionSizes.nz; z++) {
    const float zf = static_cast<float>(z);
    float zPart = 0.5f - fabs(0.5f - zf * nzRec);
    zPart = (zPart * zPart) * dzSqRec;

#pragma omp parallel for schedule(static) if (mParameters.isSimulation2D())
    for (size_t y = 0; y < reducedDimensionSizes.ny; y++) {
      const float yf = static_cast<float>(y);
      float yPart = 0.5f - fabs(0.5f - yf * nyRec);
      yPart = (yPart * yPart) * dySqRec;

      const float yzPart = zPart + yPart;

      for (size_t x = 0; x < reducedDimensionSizes.nx; x++) {
        const float xf = static_cast<float>(x);
        float xPart = 0.5f - fabs(0.5f - xf * nxRec);
        xPart = (xPart * xPart) * dxSqRec;

        float k = pi2 * sqrt(xPart + yzPart);
        float cRefK = cRefDt2 * k;

        const size_t i = get1DIndex(z, y, x, reducedDimensionSizes);

        kappa[i] = (cRefK == 0.0f) ? 1.0f : sin(cRefK) / cRefK;

        absorbNabla1[i] = pow(k, alphaPower - 2.0f);
        absorbNabla2[i] = pow(k, alphaPower - 1.0f);

        if (absorbNabla1[i] == std::numeric_limits<float>::infinity())
          absorbNabla1[i] = 0.0f;
        if (absorbNabla2[i] == std::numeric_limits<float>::infinity())
          absorbNabla2[i] = 0.0f;
      } // x
    }   // y
  }     // z
} // end of generateKappaAndNablas
//----------------------------------------------------------------------------------------------------------------------

/**
 * Generate absorbTau and absorbEta in for heterogenous medium.
 */
void KSpaceFirstOrderSolver::generateTauAndEta() {
  if ((mParameters.getAlphaCoeffScalarFlag()) && (mParameters.getC0ScalarFlag())) { // scalar values
    const float alphaPower = mParameters.getAlphaPower();
    const float tanPi2AlphaPower = tan(static_cast<float>(M_PI_2) * alphaPower);
    const float alphaNeperCoeff = (100.0f * pow(1.0e-6f / (2.0f * static_cast<float>(M_PI)), alphaPower)) / (20.0f * static_cast<float>(M_LOG10E));

    const float alphaCoeff2 = 2.0f * mParameters.getAlphaCoeffScalar() * alphaNeperCoeff;

    mParameters.setAbsorbTauScalar((-alphaCoeff2) * pow(mParameters.getC0Scalar(), alphaPower - 1));
    mParameters.setAbsorbEtaScalar(alphaCoeff2 * pow(mParameters.getC0Scalar(), alphaPower) * tanPi2AlphaPower);
  } else { // matrix

    const DimensionSizes& dimensionSizes = mParameters.getFullDimensionSizes();

    float* absorbTau = getAbsorbTau().getHostData();
    float* absorbEta = getAbsorbEta().getHostData();

    const bool alphaCoeffScalarFlag = mParameters.getAlphaCoeffScalarFlag();
    const float alphaCoeffScalar = (alphaCoeffScalarFlag) ? mParameters.getAlphaCoeffScalar() : 0.0f;
    const float* alphaCoeffMatrix = (alphaCoeffScalarFlag) ? nullptr
                                                           : getTemp1RealND().getHostData();

    const bool c0ScalarFlag = mParameters.getC0ScalarFlag();
    const float c0Scalar = (c0ScalarFlag) ? mParameters.getC0Scalar() : 0.0f;
    // here c2 still holds just c0!
    const float* cOMatrix = (c0ScalarFlag) ? nullptr : getC2().getHostData();

    const float alphaPower = mParameters.getAlphaPower();
    const float tanPi2AlphaPower = tan(static_cast<float>(M_PI_2) * alphaPower);

    // alpha = 100*alpha.*(1e-6/(2*pi)).^y./
    //                   (20*log10(exp(1)));
    const float alphaNeperCoeff = (100.0f * pow(1.0e-6f / (2.0f * static_cast<float>(M_PI)), alphaPower)) / (20.0f * static_cast<float>(M_LOG10E));

#pragma omp parallel for schedule(static) if (mParameters.isSimulation3D())
    for (size_t z = 0; z < dimensionSizes.nz; z++) {
#pragma omp parallel for schedule(static) if (mParameters.isSimulation2D())
      for (size_t y = 0; y < dimensionSizes.ny; y++) {
        for (size_t x = 0; x < dimensionSizes.nx; x++) {
          const size_t i = get1DIndex(z, y, x, dimensionSizes);

          const float alphaCoeff2 = 2.0f * alphaNeperCoeff * ((alphaCoeffScalarFlag) ? alphaCoeffScalar : alphaCoeffMatrix[i]);

          absorbTau[i] = (-alphaCoeff2) * pow(((c0ScalarFlag) ? c0Scalar : cOMatrix[i]), alphaPower - 1.0f);
          absorbEta[i] = alphaCoeff2 * pow(((c0ScalarFlag) ? c0Scalar : cOMatrix[i]), alphaPower) * tanPi2AlphaPower;

        } // x
      }   // y
    }     // z
  }       // matrix
} // end of generateTauAndEta
//----------------------------------------------------------------------------------------------------------------------

/**
 * Prepare dt./ rho0  for non-uniform grid.
 */
template <Parameters::SimulationDimension simulationDimension>
void KSpaceFirstOrderSolver::generateInitialDenisty() {
  float* dtRho0Sgx = getDtRho0Sgx().getHostData();
  float* dtRho0Sgy = getDtRho0Sgy().getHostData();
  float* dtRho0Sgz = (simulationDimension == SD::k3D) ? getDtRho0Sgz().getHostData() : nullptr;

  const float dt = mParameters.getDt();

  const float* duxdxnSgx = getDxudxnSgx().getHostData();
  const float* duydynSgy = getDyudynSgy().getHostData();
  const float* duzdznSgz = (simulationDimension == SD::k3D) ? getDzudznSgz().getHostData() : nullptr;

  const DimensionSizes& dimensionSizes = mParameters.getFullDimensionSizes();

#pragma omp parallel for schedule(static) if (simulationDimension == SD::k3D)
  for (size_t z = 0; z < dimensionSizes.nz; z++) {
#pragma omp parallel for schedule(static) if (simulationDimension == SD::k2D)
    for (size_t y = 0; y < dimensionSizes.ny; y++) {
      for (size_t x = 0; x < dimensionSizes.nx; x++) {
        const size_t i = get1DIndex(z, y, x, dimensionSizes);

        dtRho0Sgx[i] = (dt * duxdxnSgx[x]) / dtRho0Sgx[i];
        dtRho0Sgy[i] = (dt * duydynSgy[y]) / dtRho0Sgy[i];
        if (simulationDimension == SD::k3D) {
          dtRho0Sgz[i] = (dt * duzdznSgz[z]) / dtRho0Sgz[i];
        }
      } // x
    }   // y
  }     // z
} // end of generateInitialDenisty
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute c^2 on the CPU side.
 */
void KSpaceFirstOrderSolver::computeC2() {
  if (!mParameters.getC0ScalarFlag()) { // matrix
    float* c2 = getC2().getHostData();
    const auto size = getC2().size();

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) {
      c2[i] = c2[i] * c2[i];
    }
  } // matrix
} // computeC2
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculated shifted velocities.
 * \n
 * ux_shifted = real(ifft(bsxfun(\@times, x_shift_neg, fft(ux_sgx, [], 1)), [], 1)); \n
 * uy_shifted = real(ifft(bsxfun(\@times, y_shift_neg, fft(uy_sgy, [], 2)), [], 2)); \n
 * uz_shifted = real(ifft(bsxfun(\@times, z_shift_neg, fft(uz_sgz, [], 3)), [], 3)); \n
 */
template <Parameters::SimulationDimension simulationDimension>
void KSpaceFirstOrderSolver::computeShiftedVelocity() {
  auto& tempCufftShift = getTempCufftShift();

  // uxShifted
  tempCufftShift.computeR2CFft1DX(getUxSgx());
  SolverCudaKernels::computeVelocityShiftInX(tempCufftShift, getXShiftNegR());
  tempCufftShift.computeC2RFft1DX(getUxShifted());

  // uyShifted
  tempCufftShift.computeR2CFft1DY(getUySgy());
  SolverCudaKernels::computeVelocityShiftInY(tempCufftShift, getYShiftNegR());
  tempCufftShift.computeC2RFft1DY(getUyShifted());

  if (mParameters.isSimulation3D()) {
    // uzShifted
    tempCufftShift.computeR2CFft1DZ(getUzSgz());
    SolverCudaKernels::computeVelocityShiftInZ(tempCufftShift, getZShiftNegR());
    tempCufftShift.computeC2RFft1DZ(getUzShifted());
  }
} // end of computeShiftedVelocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print progress statistics.
 */
void KSpaceFirstOrderSolver::printStatistics() {
  const float nt = float(mParameters.getNt());
  const float timeIndex = float(mParameters.getTimeIndex());

  if (timeIndex + 1 > (mActPercent * nt * 0.01f)) {
    mActPercent += mParameters.getProgressPrintInterval();
    mIterationTime.stop();

    if (mParameters.getTimeIndex() >= mParameters.getSamplingStartTimeIndex()) {
      mSamplingIterationTimes.push_back(mIterationTime.getElapsedTime() - mIterationTime.getElapsedTimeOverPreviousLegs());
    } else {
      mNotSamplingIterationTimes.push_back(mIterationTime.getElapsedTime() - mIterationTime.getElapsedTimeOverPreviousLegs());
    }

    const double elTime = mIterationTime.getElapsedTime();
    const double elTimeWithLegs = mIterationTime.getElapsedTime() + mSimulationTime.getElapsedTimeOverPreviousLegs();
    const double toGo = ((elTimeWithLegs / static_cast<double>((timeIndex + 1)) * nt)) - elTimeWithLegs;

    struct tm* current;
    time_t now;
    time(&now);
    now += time_t(toGo);
    current = localtime(&now);

    Logger::log(Logger::LogLevel::kBasic,
                kOutFmtSimulationProgress,
                (timeIndex + 1) / (nt * 0.01f), '%',
                elTime, toGo,
                current->tm_mday, current->tm_mon + 1, current->tm_year - 100,
                current->tm_hour, current->tm_min, current->tm_sec,
                mIterationTime.getElapsedTime() - mIterationTime.getElapsedTimeOverPreviousLegs(),
                size_t(timeIndex) + 1);
    Logger::flush(Logger::LogLevel::kBasic);

    mIterationTime.SetElapsedTimeOverPreviousLegs(mIterationTime.getElapsedTime());
  }
} // end of printStatistics
//----------------------------------------------------------------------------------------------------------------------

/**
 * Was the loop interrupted to checkpoint?
 */
bool KSpaceFirstOrderSolver::isCheckpointInterruption() const {
  return (mParameters.getTimeIndex() != mParameters.getNt());
} // end of isCheckpointInterruption
//----------------------------------------------------------------------------------------------------------------------

/**
 * Check the output file has the correct format and version.
 */
void KSpaceFirstOrderSolver::checkOutputFile() {
  // The header has already been read
  Hdf5FileHeader& fileHeader = mParameters.getFileHeader();
  Hdf5File& outputFile = mParameters.getOutputFile();

  // test file type
  if (fileHeader.getFileType() != Hdf5FileHeader::FileType::kOutput) {
    throw ios::failure(Logger::formatMessage(kErrFmtBadOutputFileFormat,
                                             mParameters.getOutputFileName().c_str()));
  }

  // test file major version
  if (!fileHeader.checkMajorFileVersion()) {
    throw ios::failure(Logger::formatMessage(kErrFmtBadMajorFileVersion,
                                             mParameters.getOutputFileName().c_str(),
                                             fileHeader.getFileMajorVersion().c_str()));
  }

  // test file minor version
  if (!fileHeader.checkMinorFileVersion()) {
    throw ios::failure(Logger::formatMessage(kErrFmtBadMinorFileVersion,
                                             mParameters.getOutputFileName().c_str(),
                                             fileHeader.getFileMinorVersion().c_str()));
  }

  // Check dimension sizes
  DimensionSizes outputDimSizes;
  outputFile.readScalarValue(outputFile.getRootGroup(), kNxName, outputDimSizes.nx);
  outputFile.readScalarValue(outputFile.getRootGroup(), kNyName, outputDimSizes.ny);
  outputFile.readScalarValue(outputFile.getRootGroup(), kNzName, outputDimSizes.nz);

  if (mParameters.getFullDimensionSizes() != outputDimSizes) {
    throw ios::failure(Logger::formatMessage(kErrFmtOutputDimensionsMismatch,
                                             outputDimSizes.nx,
                                             outputDimSizes.ny,
                                             outputDimSizes.nz,
                                             mParameters.getFullDimensionSizes().nx,
                                             mParameters.getFullDimensionSizes().ny,
                                             mParameters.getFullDimensionSizes().nz));
  }
} // end of checkOutputFile
//----------------------------------------------------------------------------------------------------------------------

/**
 * Check the file type and the version of the checkpoint file.
 */
void KSpaceFirstOrderSolver::checkCheckpointFile() {
  // read the header and check the file version
  Hdf5FileHeader fileHeader;
  Hdf5File& checkpointFile = mParameters.getCheckpointFile();

  fileHeader.readHeaderFromCheckpointFile(checkpointFile);

  // test file type
  if (fileHeader.getFileType() != Hdf5FileHeader::FileType::kCheckpoint) {
    throw ios::failure(Logger::formatMessage(kErrFmtBadCheckpointFileFormat,
                                             mParameters.getCheckpointFileName().c_str()));
  }

  // test file major version
  if (!fileHeader.checkMajorFileVersion()) {
    throw ios::failure(Logger::formatMessage(kErrFmtBadMajorFileVersion,
                                             mParameters.getCheckpointFileName().c_str(),
                                             fileHeader.getFileMajorVersion().c_str()));
  }

  // test file minor version
  if (!fileHeader.checkMinorFileVersion()) {
    throw ios::failure(Logger::formatMessage(kErrFmtBadMinorFileVersion,
                                             mParameters.getCheckpointFileName().c_str(),
                                             fileHeader.getFileMinorVersion().c_str()));
  }

  // Check dimension sizes
  DimensionSizes checkpointDimSizes;
  checkpointFile.readScalarValue(checkpointFile.getRootGroup(), kNxName, checkpointDimSizes.nx);
  checkpointFile.readScalarValue(checkpointFile.getRootGroup(), kNyName, checkpointDimSizes.ny);
  checkpointFile.readScalarValue(checkpointFile.getRootGroup(), kNzName, checkpointDimSizes.nz);

  if (mParameters.getFullDimensionSizes() != checkpointDimSizes) {
    throw ios::failure(Logger::formatMessage(kErrFmtCheckpointDimensionsMismatch,
                                             checkpointDimSizes.nx,
                                             checkpointDimSizes.ny,
                                             checkpointDimSizes.nz,
                                             mParameters.getFullDimensionSizes().nx,
                                             mParameters.getFullDimensionSizes().ny,
                                             mParameters.getFullDimensionSizes().nz));
  }
} // end of checkCheckpointFile
//----------------------------------------------------------------------------------------------------------------------

/**
 * Restore cumulated elapsed time from the output file.
 */
void KSpaceFirstOrderSolver::loadElapsedTimeFromOutputFile() {
  double totalTime, dataLoadTime, preProcessingTime, simulationTime, postProcessingTime;

  // Get execution times stored in the output file header
  mParameters.getFileHeader().getExecutionTimes(totalTime,
                                                dataLoadTime,
                                                preProcessingTime,
                                                simulationTime,
                                                postProcessingTime);

  mTotalTime.SetElapsedTimeOverPreviousLegs(totalTime);
  mDataLoadTime.SetElapsedTimeOverPreviousLegs(dataLoadTime);
  mPreProcessingTime.SetElapsedTimeOverPreviousLegs(preProcessingTime);
  mSimulationTime.SetElapsedTimeOverPreviousLegs(simulationTime);
  mPostProcessingTime.SetElapsedTimeOverPreviousLegs(postProcessingTime);

} // end of loadElapsedTimeFromOutputFile
//----------------------------------------------------------------------------------------------------------------------

inline size_t KSpaceFirstOrderSolver::get1DIndex(const size_t z,
                                                 const size_t y,
                                                 const size_t x,
                                                 const DimensionSizes& dimensionSizes) {
  return (z * dimensionSizes.ny + y) * dimensionSizes.nx + x;
} // end of get1DIndex
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//
