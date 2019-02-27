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
 * @version   kspaceFirstOrder3D 3.6
 *
 * @date      12 July      2012, 10:27 (created)\n
 *            24 February  2019, 11:22 (revised)
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

#include <KSpaceSolver/KSpaceFirstOrderSolver.h>

#include <Hdf5/Hdf5FileHeader.h>
#include <Hdf5/Hdf5File.h>

#include <Logger/ErrorMessages.h>
#include <Logger/Logger.h>

#include <KSpaceSolver/SolverCudaKernels.cuh>
#include <Containers/MatrixContainer.h>

using std::string;
using std::ios;


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
  : mMatrixContainer(), mOutputStreamContainer(), mParameters(Parameters::getInstance()),
    mActPercent(0), mIsTimestepRightAfterRestore(false),
    mTotalTime(), mPreProcessingTime(), mDataLoadTime(), mSimulationTime(),
    mPostProcessingTime(), mIterationTime()
{
  mTotalTime.start();

  //Switch off default HDF5 error messages
  H5Eset_auto(H5E_DEFAULT, NULL, NULL);
}// end of KSpaceFirstOrderSolver
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor of the class.
 */
KSpaceFirstOrderSolver::~KSpaceFirstOrderSolver()
{
  // Delete CUDA FFT plans and related data
  CufftComplexMatrix::destroyAllPlansAndStaticData();

  // Free memory
  freeMemory();

  //Reset device after the run - recommended by CUDA SDK
  cudaDeviceReset();
}// end of ~KSpaceFirstOrderSolver
//----------------------------------------------------------------------------------------------------------------------


/**
 * The method allocates the matrix container and create all matrices and creates all output streams.
 */
void KSpaceFirstOrderSolver::allocateMemory()
{
  Logger::log(Logger::LogLevel::kBasic, kOutFmtMemoryAllocation);
  Logger::flush(Logger::LogLevel::kBasic);

  // create container, then all matrices
  mMatrixContainer.init();
  mMatrixContainer.createMatrices();

  // add output streams into container
  mOutputStreamContainer.init(mMatrixContainer);

  Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
}// end of allocateMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * The method frees all memory allocated by the class.
 */
void KSpaceFirstOrderSolver::freeMemory()
{
  mMatrixContainer.freeMatrices();
  mOutputStreamContainer.freeStreams();
}// end of freeMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Load data from the input file provided by the parameter class and creates the output time
 * series streams.
 */
void KSpaceFirstOrderSolver::loadInputData()
{
  // Load data from disk
  Logger::log(Logger::LogLevel::kBasic, kOutFmtDataLoading);
  Logger::flush(Logger::LogLevel::kBasic);

  mDataLoadTime.start();

  // open and load input file
  Hdf5File& inputFile      = mParameters.getInputFile(); // file is opened (in Parameters)
  Hdf5File& outputFile     = mParameters.getOutputFile();
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

  // The simulation does not use check pointing or this is the first turn
  bool recoverFromCheckpoint = (mParameters.isCheckpointEnabled() &&
                                Hdf5File::canAccess(mParameters.getCheckpointFileName()));


  if (recoverFromCheckpoint)
  {
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

    checkpointFile.close();
    Logger::log(Logger::LogLevel::kFull, kOutFmtDone);

    //--------------------------------------- Read data from the output file -----------------------------------------//
    // Reopen output file for RW access
    Logger::log(Logger::LogLevel::kFull, kOutFmtReadingOutputFile);
    Logger::flush(Logger::LogLevel::kFull);

    outputFile.open(mParameters.getOutputFileName(), H5F_ACC_RDWR);

    //Read file header of the output file
    mParameters.getFileHeader().readHeaderFromOutputFile(outputFile);

    // Restore elapsed time
    loadElapsedTimeFromOutputFile();

    // Reopen streams
    mOutputStreamContainer.reopenStreams();
    Logger::log(Logger::LogLevel::kFull, kOutFmtDone);
  }
  else
  {
    //------------------------------------ First round of multi-leg simulation ---------------------------------------//
    // Create the output file
    Logger::log(Logger::LogLevel::kFull, kOutFmtCreatingOutputFile);
    Logger::flush(Logger::LogLevel::kFull);

    outputFile.create(mParameters.getOutputFileName());
    Logger::log(Logger::LogLevel::kFull, kOutFmtDone);

    // Create the steams, link them with the sampled matrices
    // however DO NOT allocate memory!
    mOutputStreamContainer.createStreams();
  }

  mDataLoadTime.stop();

  if (Logger::getLevel() != Logger::LogLevel::kFull)
  {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
  }
}// end of loadInputData
//----------------------------------------------------------------------------------------------------------------------

/**
 * This method computes k-space First Order 2D/3D simulation.
 */
void KSpaceFirstOrderSolver::compute()
{
  mPreProcessingTime.start();

  Logger::log(Logger::LogLevel::kBasic, kOutFmtFftPlans);
  Logger::flush(Logger::LogLevel::kBasic);

  CudaParameters& cudaParameters = mParameters.getCudaParameters();

  // fft initialisation and preprocessing
  try
  {
    // initilaise all cuda FFT plans
    initializeCufftPlans();
    Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);

    Logger::log(Logger::LogLevel::kBasic,kOutFmtPreProcessing);
    Logger::flush(Logger::LogLevel::kBasic);

    // preprocessing is done on CPU and must pretend the CUDA configuration
    preProcessing();

    mPreProcessingTime.stop();
    // Set kernel configurations
    cudaParameters.setKernelConfiguration();

    // Set up constant memory - copy over to GPU
    // Constant memory uses some variables calculated during preprocessing
    cudaParameters.setUpDeviceConstants();

    Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
  }
  catch (const std::exception& e)
  {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtFailed);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtLastSeparator);

    Logger::errorAndTerminate(Logger::wordWrapString(e.what(),kErrFmtPathDelimiters, 9));
  }

  // Logger header for simulation
  Logger::log(Logger::LogLevel::kBasic, kOutFmtElapsedTime, mPreProcessingTime.getElapsedTime());
  Logger::log(Logger::LogLevel::kBasic, kOutFmtCompResourcesHeader);
  Logger::log(Logger::LogLevel::kBasic, kOutFmtCurrentHostMemory,   getHostMemoryUsage());
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
  try
  {
    mSimulationTime.start();

    computeMainLoop();

    mSimulationTime.stop();

    Logger::log(Logger::LogLevel::kBasic,kOutFmtSimulationEndSeparator);
  }
  catch (const std::exception& e)
  {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtSimulatoinFinalSeparator);
    Logger::errorAndTerminate(Logger::wordWrapString(e.what(),kErrFmtPathDelimiters, 9));
  }

  // Post processing region
  mPostProcessingTime.start();

  try
  {
    if (isCheckpointInterruption())
    { // Checkpoint
      Logger::log(Logger::LogLevel::kBasic, kOutFmtElapsedTime, mSimulationTime.getElapsedTime());
      Logger::log(Logger::LogLevel::kBasic, kOutFmtCheckpointTimeSteps, mParameters.getTimeIndex());
      Logger::log(Logger::LogLevel::kBasic, kOutFmtCheckpointHeader);
      Logger::log(Logger::LogLevel::kBasic, kOutFmtCreatingCheckpoint);
      Logger::flush(Logger::LogLevel::kBasic);

      if (Logger::getLevel() == Logger::LogLevel::kFull)
      {
        Logger::log(Logger::LogLevel::kBasic, kOutFmtNoDone);
      }

      saveCheckpointData();

      if (Logger::getLevel() != Logger::LogLevel::kFull)
      {
        Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
      }
    }
    else
    { // Finish
      Logger::log(Logger::LogLevel::kBasic, kOutFmtElapsedTime, mSimulationTime.getElapsedTime());
      Logger::log(Logger::LogLevel::kBasic, kOutFmtSeparator);
      Logger::log(Logger::LogLevel::kBasic, kOutFmtPostProcessing);
      Logger::flush(Logger::LogLevel::kBasic);

      postProcessing();

      // if checkpointing is enabled and the checkpoint file was created in the past, delete it
      if (mParameters.isCheckpointEnabled())
      {
        std::remove(mParameters.getCheckpointFileName().c_str());
      }
      Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
    }
  }
  catch (const std::exception &e)
  {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtFailed);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtLastSeparator);

    Logger::errorAndTerminate(Logger::wordWrapString(e.what(), kErrFmtPathDelimiters,9));
  }
  mPostProcessingTime.stop();

  // Final data written
  try
  {
    writeOutputDataInfo();
    mParameters.getOutputFile().close();

    Logger::log(Logger::LogLevel::kBasic, kOutFmtElapsedTime, mPostProcessingTime.getElapsedTime());
    }
  catch (const std::exception &e)
  {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtLastSeparator);
    Logger::errorAndTerminate(Logger::wordWrapString(e.what(), kErrFmtPathDelimiters, 9));
  }
}// end of compute()
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get peak CPU memory usage.
 */
size_t KSpaceFirstOrderSolver::getHostMemoryUsage() const
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
}// end of getHostMemoryUsage
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get peak GPU memory usage.
 */
size_t KSpaceFirstOrderSolver::getDeviceMemoryUsage() const
{
  size_t free, total;
  cudaMemGetInfo(&free,&total);

  return ((total - free) >> 20);
}// end of getDeviceMemoryUsage
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get release code version.
 */
const string KSpaceFirstOrderSolver::getCodeName() const
{
  return string(kOutFmtKWaveVersion);
}// end of getCodeName
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print full code name and the license.
 */
void KSpaceFirstOrderSolver::printFullCodeNameAndLicense() const
{
  Logger::log(Logger::LogLevel::kBasic,
              kOutFmtBuildNoDataTime,
              10, 11, __DATE__,
              8, 8, __TIME__);

  if (mParameters.getGitHash() != "")
  {
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
  #if (defined (__AVX2__))
    Logger::log(Logger::LogLevel::kBasic, kOutFmtAVX2);
  #elif (defined (__AVX__))
    Logger::log(Logger::LogLevel::kBasic, kOutFmtAVX);
  #elif (defined (__SSE4_2__))
    Logger::log(Logger::LogLevel::kBasic, kOutFmtSSE42);
  #elif (defined (__SSE4_1__))
    Logger::log(Logger::LogLevel::kBasic, kOutFmtSSE41);
  #elif (defined (__SSE3__))
    Logger::log(Logger::LogLevel::kBasic, kOutFmtSSE3);
  #elif (defined (__SSE2__))
    Logger::log(Logger::LogLevel::kBasic, kOutFmtSSE2);
  #endif

  Logger::log(Logger::LogLevel::kBasic, kOutFmtSeparator);

 // CUDA detection
  int cudaRuntimeVersion;
  if (cudaRuntimeGetVersion(&cudaRuntimeVersion) != cudaSuccess)
  {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtCudaRuntimeNA);
  }
  else
  {
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
  if (cudaParameters.getDeviceIdx() == CudaParameters::kDefaultDeviceIdx)
  {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtCudaDeviceInfoNA);
  }
  else
  {
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
}// end of getFullCodeAndLincence
//----------------------------------------------------------------------------------------------------------------------

 /**
  * Get total simulation time.
  */
double KSpaceFirstOrderSolver::getTotalTime() const
{
  return mTotalTime.getElapsedTime();
}// end of getTotalTime()
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get pre-processing time.
 */
double KSpaceFirstOrderSolver::getPreProcessingTime() const
{
  return mPreProcessingTime.getElapsedTime();
}// end of getPreProcessingTime
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get data load time.
 */
double KSpaceFirstOrderSolver::getDataLoadTime() const
{
  return mDataLoadTime.getElapsedTime();
}// end of getDataLoadTime
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get simulation time (time loop).
 */
double KSpaceFirstOrderSolver::getSimulationTime() const
{
  return mSimulationTime.getElapsedTime();
}// end of getSimulationTime
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get post-processing time.
 */
double KSpaceFirstOrderSolver::getPostProcessingTime() const
{
  return mPostProcessingTime.getElapsedTime();
}// end of getPostProcessingTime
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get total simulation time cumulated over all legs.
 */
double KSpaceFirstOrderSolver::getCumulatedTotalTime() const
{
  return mTotalTime.getElapsedTimeOverAllLegs();
}// end of getCumulatedTotalTime
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get pre-processing time cumulated over all legs.
 */
double KSpaceFirstOrderSolver::getCumulatedPreProcessingTime() const
{
  return mPreProcessingTime.getElapsedTimeOverAllLegs();
} // end of getCumulatedPreProcessingTime
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get data load time cumulated over all legs.
 */
double KSpaceFirstOrderSolver::getCumulatedDataLoadTime() const
{
  return mDataLoadTime.getElapsedTimeOverAllLegs();
}// end of getCumulatedDataLoadTime
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get simulation time (time loop) cumulated over all legs.
 */
double KSpaceFirstOrderSolver::getCumulatedSimulationTime() const
{
  return mSimulationTime.getElapsedTimeOverAllLegs();
}// end of getCumulatedSimulationTime
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get post-processing time cumulated over all legs.
 */
double KSpaceFirstOrderSolver::getCumulatedPostProcessingTime() const
{
  return mPostProcessingTime.getElapsedTimeOverAllLegs();
}// end of getCumulatedPostProcessingTime
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


/**
 * Initialize cuda FFT plans.
 */
void KSpaceFirstOrderSolver::initializeCufftPlans()
{
  // create real to complex plans
  CufftComplexMatrix::createR2CFftPlan3D(mParameters.getFullDimensionSizes());

 // create complex to real plans
  CufftComplexMatrix::createC2RFftPlan3D(mParameters.getFullDimensionSizes());

  // if necessary, create 1D shift plans.
  // in this case, the matrix has a bit bigger dimensions to be able to store shifted matrices.
  if (Parameters::getInstance().getStoreVelocityNonStaggeredRawFlag())
  {
    // X shifts
    CufftComplexMatrix::createR2CFftPlan1DX(mParameters.getFullDimensionSizes());
    CufftComplexMatrix::createC2RFftPlan1DX(mParameters.getFullDimensionSizes());

    // Y shifts
    CufftComplexMatrix::createR2CFftPlan1DY(mParameters.getFullDimensionSizes());
    CufftComplexMatrix::createC2RFftPlan1DY(mParameters.getFullDimensionSizes());

    // Z shifts
    CufftComplexMatrix::createR2CFftPlan1DZ(mParameters.getFullDimensionSizes());
    CufftComplexMatrix::createC2RFftPlan1DZ(mParameters.getFullDimensionSizes());
  }// end u_non_staggered
}// end of initializeCufftPlans
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute pre-processing phase.
 */
void KSpaceFirstOrderSolver::preProcessing()
{
  // get the correct sensor mask and recompute indices
  if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kIndex)
  {
    getSensorMaskIndex().recomputeIndicesToCPP();
  }

  if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kCorners)
  {
    getSensorMaskCorners().recomputeIndicesToCPP();
  }

  if ((mParameters.getTransducerSourceFlag() != 0) ||
      (mParameters.getVelocityXSourceFlag() != 0)  ||
      (mParameters.getVelocityYSourceFlag() != 0)  ||
      (mParameters.getVelocityZSourceFlag() != 0)
     )
  {
    getVelocitySourceIndex().recomputeIndicesToCPP();
  }

  if (mParameters.getTransducerSourceFlag() != 0)
  {
    getDelayMask().recomputeIndicesToCPP();
  }

  if (mParameters.getPressureSourceFlag() != 0)
  {
    getPressureSourceIndex().recomputeIndicesToCPP();
  }

  // compute dt / rho0_sg...
  if (!mParameters.getRho0ScalarFlag())
  { // non-uniform grid cannot be pre-calculated :-(
    // rho is matrix
    if (mParameters.getNonUniformGridFlag())
    {
      generateInitialDenisty();
    }
    else
    {
      getDtRho0Sgx().scalarDividedBy(mParameters.getDt());
      getDtRho0Sgy().scalarDividedBy(mParameters.getDt());
      getDtRho0Sgz().scalarDividedBy(mParameters.getDt());
    }
  }

  // generate different matrices
  if (mParameters.getAbsorbingFlag() != 0)
  {
    generateKappaAndNablas();
    generateTauAndEta();
  }
  else
  {
    generateKappa();
  }

  // Generate sourceKappa
  if (((mParameters.getVelocitySourceMode() == Parameters::SourceMode::kAdditive) ||
       (mParameters.getPressureSourceMode() == Parameters::SourceMode::kAdditive)) &&
      (mParameters.getPressureSourceFlag()  ||
       mParameters.getVelocityXSourceFlag() ||
       mParameters.getVelocityYSourceFlag() ||
       mParameters.getVelocityZSourceFlag()))
  {
    generateSourceKappa();
  }

  // calculate c^2. It has to be after kappa gen... because of c modification
  computeC2();

}// end of preProcessing
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute the main time loop of KSpaceFirstOrder solver.
 */
void KSpaceFirstOrderSolver::computeMainLoop()
{
  mActPercent = 0;

  // if resuming from a checkpoint,
  // set ActPercent to correspond the t_index after recovery
  if (mParameters.getTimeIndex() > 0)
  {
    // We're restarting after checkpoint
    mIsTimestepRightAfterRestore = true;
    mActPercent = (100 * mParameters.getTimeIndex()) / mParameters.getNt();
  }

  // Progress header
  Logger::log(Logger::LogLevel::kBasic,kOutFmtSimulationHeader);

  // Initial copy of data to the GPU
  mMatrixContainer.copyMatricesToDevice();

  mIterationTime.start();

  // execute main loop
  while ((mParameters.getTimeIndex() < mParameters.getNt()) && (!isTimeToCheckpoint()))
  {
    const size_t timeIndex = mParameters.getTimeIndex();

    // compute velocity
    computeVelocity();

    // add in the velocity source term
    addVelocitySource();

    // add in the transducer source term (t = t1) to ux
    if (mParameters.getTransducerSourceFlag() > timeIndex)
    {
      SolverCudaKernels::addTransducerSource(getUxSgx(),
                                             getVelocitySourceIndex(),
                                             getTransducerSourceInput(),
                                             getDelayMask(),
                                             timeIndex);
    }

    // compute gradient of velocity
    computeVelocityGradient();

    // compute density
    if (mParameters.getNonLinearFlag())
    {
      computeDensityNonliner();
    }
    else
    {
      computeDensityLinear();
    }

    // add in the source pressure term
    addPressureSource();

    if (mParameters.getNonLinearFlag())
    {
      computePressureNonlinear();
    }
    else
    {
      computePressureLinear();
    }

    //-- calculate initial pressure
    if ((timeIndex == 0) && (mParameters.getInitialPressureSourceFlag() == 1))  addInitialPressureSource();

    storeSensorData();
    printStatistics();

    mParameters.incrementTimeIndex();
    mIsTimestepRightAfterRestore = false;
  }

    // Since disk operations are one step delayed, we have to do the last one here.
    // However we need to check if the loop wasn't skipped due to very short checkpoint interval
    if (mParameters.getTimeIndex() > mParameters.getSamplingStartTimeIndex() && (!mIsTimestepRightAfterRestore))
    {
      mOutputStreamContainer.flushRawStreams();
    }
}// end of computeMainLoop()
//----------------------------------------------------------------------------------------------------------------------

/**
 * Post processing, and closing the output streams.
 */
void KSpaceFirstOrderSolver::postProcessing()
{
  if (mParameters.getStorePressureFinalAllFlag())
  {
    getP().copyFromDevice();
    getP().writeData(mParameters.getOutputFile(),
                     kPressureFinalName,
                     mParameters.getCompressionLevel());
  }// p_final

  if (mParameters.getStoreVelocityFinalAllFlag())
  {
    getUxSgx().copyFromDevice();
    getUySgy().copyFromDevice();
    getUzSgz().copyFromDevice();

    getUxSgx().writeData(mParameters.getOutputFile(),
                         kUxFinalName,
                         mParameters.getCompressionLevel());
    getUySgy().writeData(mParameters.getOutputFile(),
                         kUyFinalName,
                         mParameters.getCompressionLevel());
    getUzSgz().writeData(mParameters.getOutputFile(),
                         kUzFinalName,
                         mParameters.getCompressionLevel());
  }// u_final

  // Apply post-processing, flush data on disk/
  mOutputStreamContainer.postProcessStreams();
  mOutputStreamContainer.closeStreams();

  // store sensor mask if wanted
  if (mParameters.getCopySensorMaskFlag())
  {
    if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kIndex)
    {
      getSensorMaskIndex().recomputeIndicesToMatlab();
      getSensorMaskIndex().writeData(mParameters.getOutputFile(),kSensorMaskIndexName,
                                     mParameters.getCompressionLevel());
    }

    if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kCorners)
    {
      getSensorMaskCorners().recomputeIndicesToMatlab();
      getSensorMaskCorners().writeData(mParameters.getOutputFile(),kSensorMaskCornersName,
                                       mParameters.getCompressionLevel());
    }
  }
}// end of postProcessing
//----------------------------------------------------------------------------------------------------------------------

/**
 * Store sensor data.
 */
void KSpaceFirstOrderSolver::storeSensorData()
{
  // Unless the time for sampling has come, exit.
  if (mParameters.getTimeIndex() >= mParameters.getSamplingStartTimeIndex())
  {
    // Read event for t_index-1. If sampling did not occur by then, ignored it.
    // if it did store data on disk (flush) - the GPU is running asynchronously.
    // But be careful, flush has to be one step delayed to work correctly.
    // when restoring from checkpoint we have to skip the first flush
    if (mParameters.getTimeIndex() > mParameters.getSamplingStartTimeIndex() && !mIsTimestepRightAfterRestore)
    {
      mOutputStreamContainer.flushRawStreams();
    }

    // if --u_non_staggered is switched on, calculate unstaggered velocity.
    if (mParameters.getStoreVelocityNonStaggeredRawFlag())
    {
      computeShiftedVelocity();
    }

    // Sample data for step t  (store event for sampling in next turn)
    mOutputStreamContainer.sampleStreams();
    // the last step (or data after) checkpoint are flushed in the main loop
  }
}// end of storeSensorData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write statistics and the header into the output file.
 */
void KSpaceFirstOrderSolver::writeOutputDataInfo()
{
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
  fileHeader.setExecutionTimes(getTotalTime(),
                               getDataLoadTime(),
                               getPreProcessingTime(),
                               getSimulationTime(),
                               getPostProcessingTime());

  fileHeader.setNumberOfCores();

  fileHeader.writeHeaderToOutputFile(mParameters.getOutputFile());
}// end of writeOutputDataInfo
//----------------------------------------------------------------------------------------------------------------------

/**
 * Save checkpoint data into the checkpoint file, flush aggregated outputs into the output file.
 */
void KSpaceFirstOrderSolver::saveCheckpointData()
{
  // Create Checkpoint file
  Hdf5File& checkpointFile = mParameters.getCheckpointFile();
  // if it happens and the file is opened (from the recovery, close it)
  if (checkpointFile.isOpen()) checkpointFile.close();

  Logger::log(Logger::LogLevel::kFull,kOutFmtStoringCheckpointData);
  Logger::flush(Logger::LogLevel::kFull);

  // Create the new file (overwrite the old one)
  checkpointFile.create(mParameters.getCheckpointFileName());

  //-------------------------------------- Store Matrices ----------------------------------------//

  // Store all necessary matrices in Checkpoint file
  mMatrixContainer.storeDataIntoCheckpointFile();
  // Write t_index
  checkpointFile.writeScalarValue(checkpointFile.getRootGroup(), kTimeIndexName, mParameters.getTimeIndex());

  // store basic dimension sizes (nx, ny, nz) - time index is not necessary
  checkpointFile.writeScalarValue(checkpointFile.getRootGroup(), kNxName, mParameters.getFullDimensionSizes().nx);
  checkpointFile.writeScalarValue(checkpointFile.getRootGroup(), kNyName, mParameters.getFullDimensionSizes().ny);
  checkpointFile.writeScalarValue(checkpointFile.getRootGroup(), kNzName, mParameters.getFullDimensionSizes().nz);

  // Write checkpoint file header
  Hdf5FileHeader fileHeader = mParameters.getFileHeader();

  fileHeader.setFileType(Hdf5FileHeader::FileType::kCheckpoint);
  fileHeader.setCodeName(getCodeName());
  fileHeader.setActualCreationTime();

  fileHeader.writeHeaderToCheckpointFile(checkpointFile);

  checkpointFile.close();
  Logger::log(Logger::LogLevel::kFull, kOutFmtDone);

  // checkpoint output streams only if necessary (t_index > start_index), we're here one step ahead!
  if (mParameters.getTimeIndex() > mParameters.getSamplingStartTimeIndex())
  {
    Logger::log(Logger::LogLevel::kFull,kOutFmtStoringSensorData);
    Logger::flush(Logger::LogLevel::kFull);

    mOutputStreamContainer.checkpointStreams();

    Logger::log(Logger::LogLevel::kFull, kOutFmtDone);
  }

  mOutputStreamContainer.closeStreams();
}// end of saveCheckpointData()
//----------------------------------------------------------------------------------------------------------------------


/**
 * Compute new values of acoustic velocity in all three dimensions (UxSgx, UySgy, UzSgz).
 *
 * <b>Matlab code:</b> \n
 *
 * \verbatim
   p_l = fftn(p);
   ux_sgx = bsxfun(@times, pml_x_sgx, ...
       bsxfun(@times, pml_x_sgx, ux_sgx) ...
       - dt .* rho0_sgx_inv .* real(ifftn( bsxfun(@times, ddx_k_shift_pos, kappa .* p_k) )) ...
       );
   uy_sgy = bsxfun(@times, pml_y_sgy, ...
       bsxfun(@times, pml_y_sgy, uy_sgy) ...
       - dt .* rho0_sgy_inv .* real(ifftn( bsxfun(@times, ddy_k_shift_pos, kappa .* p_k) )) ...
       );
   uz_sgz = bsxfun(@times, pml_z_sgz, ...
       bsxfun(@times, pml_z_sgz, uz_sgz) ...
       - dt .* rho0_sgz_inv .* real(ifftn( bsxfun(@times, ddz_k_shift_pos, kappa .* p_k) )) ...
       );
 \endverbatim
 */
void KSpaceFirstOrderSolver::computeVelocity()
{
  // fftn(p);
  getTempCufftX().computeR2CFft3D(getP());
  // bsxfun(@times, ddx_k_shift_pos, kappa .* pre_result) , for all 3 dims
  SolverCudaKernels::computePressureGradient(getTempCufftX(),
                                             getTempCufftY(),
                                             getTempCufftZ(),
                                             getKappa(),
                                             getDdxKShiftPos(),
                                             getDdyKShiftPos(),
                                             getDdzKShiftPos());

  // ifftn(pre_result)
  getTempCufftX().computeC2RFft3D(getTemp1RealND());
  getTempCufftY().computeC2RFft3D(getTemp2RealND());
  getTempCufftZ().computeC2RFft3D(getTemp3RealND());

  // bsxfun(@times, pml_x_sgx, bsxfun(@times, pml_x_sgx, ux_sgx) - dt .* rho0_sgx_inv .* (pre_result))
  if (mParameters.getRho0ScalarFlag())
  { // scalars
    if (mParameters.getNonUniformGridFlag())
    {
      SolverCudaKernels::computeVelocityHomogeneousNonuniform(getUxSgx(),
                                                              getUySgy(),
                                                              getUzSgz(),
                                                              getTemp1RealND(),
                                                              getTemp2RealND(),
                                                              getTemp3RealND(),
                                                              getDxudxnSgx(),
                                                              getDyudynSgy(),
                                                              getDzudznSgz(),
                                                              getPmlXSgx(),
                                                              getPmlYSgy(),
                                                              getPmlZSgz());
    }
    else
    {
      SolverCudaKernels::computeVelocityHomogeneousUniform(getUxSgx(),
                                                           getUySgy(),
                                                           getUzSgz(),
                                                           getTemp1RealND(),
                                                           getTemp2RealND(),
                                                           getTemp3RealND(),
                                                           getPmlXSgx(),
                                                           getPmlYSgy(),
                                                           getPmlZSgz());
    }
  }
  else
  {// matrices
    SolverCudaKernels::computeVelocity(getUxSgx(),
                                       getUySgy(),
                                       getUzSgz(),
                                       getTemp1RealND(),
                                       getTemp2RealND(),
                                       getTemp3RealND(),
                                       getDtRho0Sgx(),
                                       getDtRho0Sgy(),
                                       getDtRho0Sgz(),
                                       getPmlXSgx(),
                                       getPmlYSgy(),
                                       getPmlZSgz());
  }
}// end of computeVelocity
//----------------------------------------------------------------------------------------------------------------------


/**
 * Compute new gradient of velocity (Duxdx, Duydy, Duzdz).
 *
 * <b>Matlab code:</b> \n
 *
 * \verbatim
   duxdx = real(ifftn( bsxfun(@times, ddx_k_shift_neg, kappa .* fftn(ux_sgx)) ));
   duydy = real(ifftn( bsxfun(@times, ddy_k_shift_neg, kappa .* fftn(uy_sgy)) ));
   duzdz = real(ifftn( bsxfun(@times, ddz_k_shift_neg, kappa .* fftn(uz_sgz)) ));
 \endverbatim
 */
void KSpaceFirstOrderSolver::computeVelocityGradient()
{
  getTempCufftX().computeR2CFft3D(getUxSgx());
  getTempCufftY().computeR2CFft3D(getUySgy());
  getTempCufftZ().computeR2CFft3D(getUzSgz());

  // calculate Duxyz on uniform grid
  SolverCudaKernels::computeVelocityGradient(getTempCufftX(),
                                             getTempCufftY(),
                                             getTempCufftZ(),
                                             getKappa(),
                                             getDdxKShiftNeg(),
                                             getDdyKShiftNeg(),
                                             getDdzKShiftNeg());

  getTempCufftX().computeC2RFft3D(getDuxdx());
  getTempCufftY().computeC2RFft3D(getDuydy());
  getTempCufftZ().computeC2RFft3D(getDuzdz());

  // Non-uniform grid
  if (mParameters.getNonUniformGridFlag() != 0)
  {
    SolverCudaKernels::computeVelocityGradientShiftNonuniform(getDuxdx(),
                                                              getDuydy(),
                                                              getDuzdz(),
                                                              getDxudxn(),
                                                              getDyudyn(),
                                                              getDzudzn());
  }// non-uniform grid
}// end of computeVelocityGradient
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate new values of acoustic density for non-linear case (rhoX, rhoy and rhoZ).
 *
 * <b>Matlab code:</b> \n
 *
 *\verbatim
    rho0_plus_rho = 2 .* (rhox + rhoy + rhoz) + rho0;
    rhox = bsxfun(@times, pml_x, bsxfun(@times, pml_x, rhox) - dt .* rho0_plus_rho .* duxdx);
    rhoy = bsxfun(@times, pml_y, bsxfun(@times, pml_y, rhoy) - dt .* rho0_plus_rho .* duydy);
    rhoz = bsxfun(@times, pml_z, bsxfun(@times, pml_z, rhoz) - dt .* rho0_plus_rho .* duzdz);
 \endverbatim
 */
void KSpaceFirstOrderSolver::computeDensityNonliner()
{
  SolverCudaKernels::computeDensityNonlinear(getRhoX(),
                                             getRhoY(),
                                             getRhoZ(),
                                             getPmlX(),
                                             getPmlY(),
                                             getPmlZ(),
                                             getDuxdx(),
                                             getDuydy(),
                                             getDuzdz(),
                                             mParameters.getRho0ScalarFlag(),
                                             (mParameters.getRho0ScalarFlag()) ? nullptr :getRho0().getDeviceData());

}// end of computeDensityNonliner
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate new values of acoustic density for linear case (rhoX, rhoy and rhoZ).
 *
 * <b>Matlab code:</b> \n
 *
 *\verbatim
    rhox = bsxfun(@times, pml_x, bsxfun(@times, pml_x, rhox) - dt .* rho0 .* duxdx);
    rhoy = bsxfun(@times, pml_y, bsxfun(@times, pml_y, rhoy) - dt .* rho0 .* duydy);
    rhoz = bsxfun(@times, pml_z, bsxfun(@times, pml_z, rhoz) - dt .* rho0 .* duzdz);
\endverbatim
 *
 */
void KSpaceFirstOrderSolver::computeDensityLinear()
{
  SolverCudaKernels::computeDensityLinear(getRhoX(),
                                          getRhoY(),
                                          getRhoZ(),
                                          getPmlX(),
                                          getPmlY(),
                                          getPmlZ(),
                                          getDuxdx(),
                                          getDuydy(),
                                          getDuzdz(),
                                          mParameters.getRho0ScalarFlag(),
                                          (mParameters.getRho0ScalarFlag()) ? nullptr :getRho0().getDeviceData());

}// end of computeDensityLinear
//----------------------------------------------------------------------------------------------------------------------


/**
 * Compute acoustic pressure for non-linear case.
 *
 * <b>Matlab code:</b> \n
 *
 *\verbatim
    case 'lossless'
        % calculate p using a nonlinear adiabatic equation of state
        p = c.^2 .* (rhox + rhoy + rhoz + medium.BonA .* (rhox + rhoy + rhoz).^2 ./ (2 .* rho0));

    case 'absorbing'
        % calculate p using a nonlinear absorbing equation of state
        p = c.^2 .* (...
            (rhox + rhoy + rhoz) ...
            + absorb_tau .* real(ifftn( absorb_nabla1 .* fftn(rho0 .* (duxdx + duydy + duzdz)) ))...
            - absorb_eta .* real(ifftn( absorb_nabla2 .* fftn(rhox + rhoy + rhoz) ))...
            + medium.BonA .*(rhox + rhoy + rhoz).^2 ./ (2 .* rho0) ...
            );

 \endverbatim
 */
void KSpaceFirstOrderSolver::computePressureNonlinear()
{
  if (mParameters.getAbsorbingFlag())
  { // absorbing case
    RealMatrix& densitySum         = getTemp1RealND();
    RealMatrix& nonlinearTerm      = getTemp2RealND();
    RealMatrix& velocitGradientSum = getTemp3RealND();

    // reusing of the temp variables
    RealMatrix& absorbTauTerm = velocitGradientSum;
    RealMatrix& absorbEtaTerm = densitySum;

    computePressureTermsNonlinear(densitySum, nonlinearTerm, velocitGradientSum);

    getTempCufftX().computeR2CFft3D(velocitGradientSum);
    getTempCufftY().computeR2CFft3D(densitySum);

    SolverCudaKernels::computeAbsorbtionTerm(getTempCufftX(), getTempCufftY(), getAbsorbNabla1(), getAbsorbNabla2());

    getTempCufftX().computeC2RFft3D(absorbTauTerm);
    getTempCufftY().computeC2RFft3D(absorbEtaTerm);

    sumPressureTermsNonlinear(absorbTauTerm, absorbEtaTerm, nonlinearTerm);
  }
  else
  {
    sumPressureTermsNonlinearLossless();
  }
}// end of computePressureNonlinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute new p for linear case.
 *
 * <b>Matlab code:</b> \n
 *
 *\verbatim
    case 'lossless'

        % calculate p using a linear adiabatic equation of state
        p = c.^2 .* (rhox + rhoy + rhoz);

    case 'absorbing'

        % calculate p using a linear absorbing equation of state
        p = c.^2 .* ( ...
            (rhox + rhoy + rhoz) ...
            + absorb_tau .* real(ifftn( absorb_nabla1 .* fftn(rho0 .* (duxdx + duydy + duzdz)) )) ...
            - absorb_eta .* real(ifftn( absorb_nabla2 .* fftn(rhox + rhoy + rhoz) )) ...
            );
 \endverbatim
 */
void KSpaceFirstOrderSolver::computePressureLinear()
{
  if (mParameters.getAbsorbingFlag())
  { // absorbing case
    RealMatrix& densitySum           = getTemp1RealND();
    RealMatrix& velocityGradientTerm = getTemp2RealND();

    RealMatrix& absorbTauTerm        = getTemp2RealND();
    RealMatrix& absorbEtaTerm        = getTemp3RealND();

    computePressureTermsLinear(densitySum, velocityGradientTerm);

    // ifftn ( absorbNabla1 * fftn (rho0 * (duxdx + duydy + duzdz))
    getTempCufftX().computeR2CFft3D(velocityGradientTerm);
    getTempCufftY().computeR2CFft3D(densitySum);

    SolverCudaKernels::computeAbsorbtionTerm(getTempCufftX(), getTempCufftY(), getAbsorbNabla1(), getAbsorbNabla2());

    getTempCufftX().computeC2RFft3D(absorbTauTerm);
    getTempCufftY().computeC2RFft3D(absorbEtaTerm);

    sumPressureTermsLinear(absorbTauTerm, absorbEtaTerm, densitySum);
  }
  else
  {
    // lossless case
    sumPressureTermsLinearLossless();
  }
}// end of computePressureLinear()
//----------------------------------------------------------------------------------------------------------------------

/**
 * Add velocity source to the particle velocity.
 */
void KSpaceFirstOrderSolver::addVelocitySource()
{
  size_t timeIndex = mParameters.getTimeIndex();

  if (mParameters.getVelocitySourceMode() != Parameters::SourceMode::kAdditive)
  { // executed Dirichlet and AdditiveNoCorrection source
    if (mParameters.getVelocityXSourceFlag() > timeIndex)
    {
      SolverCudaKernels::addVelocitySource(getUxSgx(),
                                           getVelocityXSourceInput(),
                                           getVelocitySourceIndex(),
                                           timeIndex);
    }
    if (mParameters.getVelocityYSourceFlag() > timeIndex)
    {
      SolverCudaKernels::addVelocitySource(getUySgy(),
                                           getVelocityYSourceInput(),
                                           getVelocitySourceIndex(),
                                           timeIndex);
    }
    if (mParameters.getVelocityZSourceFlag() > timeIndex)
    {
      SolverCudaKernels::addVelocitySource(getUzSgz(),
                                           getVelocityZSourceInput(),
                                           getVelocitySourceIndex(),
                                           timeIndex);
    }
  }
  else
  { // execute Additive source
    if (mParameters.getVelocityXSourceFlag() > timeIndex)
    {
      RealMatrix& scaledSource = getTemp1RealND();

      scaleSource(scaledSource,
                  getVelocityXSourceInput(),
                  getVelocitySourceIndex(),
                  mParameters.getVelocitySourceMany());

      // Insert source
      SolverCudaKernels::addVelocityScaledSource(getUxSgx(), scaledSource);
    }

    if (mParameters.getVelocityYSourceFlag() > timeIndex)
    {
      RealMatrix& scaledSource = getTemp1RealND();

      scaleSource(scaledSource,
                  getVelocityYSourceInput(),
                  getVelocitySourceIndex(),
                  mParameters.getVelocitySourceMany());

      // Insert source
      SolverCudaKernels::addVelocityScaledSource(getUySgy(), scaledSource);
    }

    if (mParameters.getVelocityZSourceFlag() > timeIndex)
    {
      RealMatrix& scaledSource = getTemp1RealND();

      scaleSource(scaledSource,
                  getVelocityZSourceInput(),
                  getVelocitySourceIndex(),
                  mParameters.getVelocitySourceMany());

      // Insert source
      SolverCudaKernels::addVelocityScaledSource(getUzSgz(), scaledSource);
    }
  }
}// end of addVelocitySource
//----------------------------------------------------------------------------------------------------------------------

/*
 * Add in pressure source.
 */
void KSpaceFirstOrderSolver::addPressureSource()
{
  size_t timeIndex = mParameters.getTimeIndex();

  if (mParameters.getPressureSourceFlag() > timeIndex)
  {
    if (mParameters.getPressureSourceMode() != Parameters::SourceMode::kAdditive)
    { // executed Dirichlet and AdditiveNoCorrection source
      SolverCudaKernels::addPressureSource(getRhoX(),
                                           getRhoY(),
                                           getRhoZ(),
                                           getPressureSourceInput(),
                                           getPressureSourceIndex(),
                                           timeIndex);
    }
    else
    { // execute Additive source
      RealMatrix& scaledSource = getTemp1RealND();

      scaleSource(scaledSource,
                  getPressureSourceInput(),
                  getPressureSourceIndex(),
                  mParameters.getPressureSourceMany());

      // Insert source
      SolverCudaKernels::addPressureScaledSource(getRhoX(),
                                                 getRhoY(),
                                                 getRhoZ(),
                                                 scaledSource);
    } // Additive source
  } // apply source
}// end of AddPressureSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Scale source signal
 */
void KSpaceFirstOrderSolver::scaleSource(RealMatrix&        scaledSource,
                                           const RealMatrix&  sourceInput,
                                           const IndexMatrix& sourceIndex,
                                           const size_t       manyFlag)
{
  CufftComplexMatrix& cufftMatrix  = getTempCufftX();

  // Zero source scaling matrix on GPU.
  scaledSource.zeroDeviceMatrix();
  // Inject source to scaling matrix
  SolverCudaKernels::insertSourceIntoScalingMatrix(scaledSource,
                                                   sourceInput,
                                                   sourceIndex,
                                                   manyFlag,
                                                   mParameters.getTimeIndex());
  // Compute FFT
  cufftMatrix.computeR2CFft3D(scaledSource);
  // Calculate gradient
  SolverCudaKernels::computeSourceGradient(cufftMatrix, getSourceKappa());
  // Compute iFFT
  cufftMatrix.computeC2RFft3D(scaledSource);
}// end of scaleSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate p0 source when necessary.
 *
 * <b>Matlab code:</b> \n
 *
 *\verbatim
    % add the initial pressure to rho as a mass source
    p = source.p0;
    rhox = source.p0 ./ (3 .* c.^2);
    rhoy = source.p0 ./ (3 .* c.^2);
    rhoz = source.p0 ./ (3 .* c.^2);

    % compute u(t = t1 + dt/2) based on the assumption u(dt/2) = -u(-dt/2)
    % which forces u(t = t1) = 0
    ux_sgx = dt .* rho0_sgx_inv .* real(ifftn( bsxfun(@times, ddx_k_shift_pos, kappa .* fftn(p)) )) / 2;
    uy_sgy = dt .* rho0_sgy_inv .* real(ifftn( bsxfun(@times, ddy_k_shift_pos, kappa .* fftn(p)) )) / 2;
    uz_sgz = dt .* rho0_sgz_inv .* real(ifftn( bsxfun(@times, ddz_k_shift_pos, kappa .* fftn(p)) )) / 2;
 \endverbatim
 */
void KSpaceFirstOrderSolver::addInitialPressureSource()
{
  // get over the scalar problem
  bool isSoundScalar = mParameters.getC0ScalarFlag();
  const float* c2 = (isSoundScalar) ? nullptr : getC2().getDeviceData();

  //-- add the initial pressure to rho as a mass source --//
  SolverCudaKernels::addInitialPressureSource(getP(),
                                              getRhoX(),
                                              getRhoY(),
                                              getRhoZ(),
                                              getInitialPressureSourceInput(),
                                              isSoundScalar,
                                              c2);

  //-----------------------------------------------------------------------//
  //--compute u(t = t1 + dt/2) based on the assumption u(dt/2) = -u(-dt/2)-//
  //--    which forces u(t = t1) = 0                                      -//
  //-----------------------------------------------------------------------//
  getTempCufftX().computeR2CFft3D(getP());

  SolverCudaKernels::computePressureGradient(getTempCufftX(),
                                             getTempCufftY(),
                                             getTempCufftZ(),
                                             getKappa(),
                                             getDdxKShiftPos(),
                                             getDdyKShiftPos(),
                                             getDdzKShiftPos());

  getTempCufftX().computeC2RFft3D(getUxSgx());
  getTempCufftY().computeC2RFft3D(getUySgy());
  getTempCufftZ().computeC2RFft3D(getUzSgz());

  if (mParameters.getRho0ScalarFlag())
  {
    if (mParameters.getNonUniformGridFlag())
    { // non uniform grid, homogeneous
      SolverCudaKernels::computeInitialVelocityHomogeneousNonuniform(getUxSgx(),
                                                                     getUySgy(),
                                                                     getUzSgz(),
                                                                     getDxudxnSgx(),
                                                                     getDyudynSgy(),
                                                                     getDzudznSgz());
    }
    else
    { //uniform grid, homogeneous
      SolverCudaKernels::computeInitialVelocityHomogeneousUniform(getUxSgx(), getUySgy(), getUzSgz());
    }
  }
  else
  {
    // heterogeneous, uniform grid
    // divide the matrix by 2 and multiply with st./rho0_sg
    SolverCudaKernels::computeInitialVelocity(getUxSgx(),
                                              getUySgy(),
                                              getUzSgz(),
                                              getDtRho0Sgx(),
                                              getDtRho0Sgy(),
                                              getDtRho0Sgz());
  }
}// end of addInitialPressureSource
//----------------------------------------------------------------------------------------------------------------------


/**
 * Generate kappa matrix for lossless medium.
 */
void KSpaceFirstOrderSolver::generateKappa()
{
  #pragma omp parallel
  {
    const float dx2Rec = 1.0f / (mParameters.getDx() * mParameters.getDx());
    const float dy2Rec = 1.0f / (mParameters.getDy() * mParameters.getDy());
    const float dz2Rec = 1.0f / (mParameters.getDz() * mParameters.getDz());

    const float cRefDtPi = mParameters.getCRef() * mParameters.getDt() * static_cast<float>(M_PI);

    const float nxRec = 1.0f / static_cast<float>(mParameters.getFullDimensionSizes().nx);
    const float nyRec = 1.0f / static_cast<float>(mParameters.getFullDimensionSizes().ny);
    const float nzRec = 1.0f / static_cast<float>(mParameters.getFullDimensionSizes().nz);

    const size_t nx = mParameters.getReducedDimensionSizes().nx;
    const size_t ny = mParameters.getReducedDimensionSizes().ny;
    const size_t nz = mParameters.getReducedDimensionSizes().nz;

    float* kappa = getKappa().getHostData();

    #pragma omp for schedule (static)
    for (size_t z = 0; z < nz; z++)
    {
      const float zf    = static_cast<float>(z);
            float zPart = 0.5f - fabs(0.5f - zf * nzRec);
                  zPart = (zPart * zPart) * dz2Rec;

      for (size_t y = 0; y < ny; y++)
      {
        const float yf    = static_cast<float>(y);
              float yPart = 0.5f - fabs(0.5f - yf * nyRec);
                    yPart = (yPart * yPart) * dy2Rec;

        const float yzPart = zPart + yPart;
        for (size_t x = 0; x < nx; x++)
        {
          const float xf = static_cast<float>(x);
                float xPart = 0.5f - fabs(0.5f - xf * nxRec);
                      xPart = (xPart * xPart) * dx2Rec;

                float k = cRefDtPi * sqrt(xPart + yzPart);

          // kappa element
          kappa[(z * ny + y) * nx + x] = (k == 0.0f) ? 1.0f : sin(k) / k;
        }//x
      }//y
    }// z
  }// parallel
}// end of generateKappa
//----------------------------------------------------------------------------------------------------------------------


/**
 * Generate sourceKappa matrix for additive sources.
 * For 2D simulation, the zPart == 0.
 */
void KSpaceFirstOrderSolver::generateSourceKappa()
{
  const float dx2Rec = 1.0f / (mParameters.getDx() * mParameters.getDx());
  const float dy2Rec = 1.0f / (mParameters.getDy() * mParameters.getDy());
  const float dz2Rec = 1.0f / (mParameters.getDz() * mParameters.getDz());

  const float cRefDtPi = mParameters.getCRef() * mParameters.getDt() * static_cast<float>(M_PI);

  const float nxRec = 1.0f / static_cast<float>(mParameters.getFullDimensionSizes().nx);
  const float nyRec = 1.0f / static_cast<float>(mParameters.getFullDimensionSizes().ny);
  const float nzRec = 1.0f / static_cast<float>(mParameters.getFullDimensionSizes().nz);

  const size_t nx = mParameters.getReducedDimensionSizes().nx;
  const size_t ny = mParameters.getReducedDimensionSizes().ny;
  const size_t nz = mParameters.getReducedDimensionSizes().nz;


  float* sourceKappa = getSourceKappa().getHostData();

  #pragma omp parallel for schedule(static)
  for (size_t z = 0; z < nz; z++)
  {
    const float zf    = static_cast<float>(z);
          float zPart = 0.5f - fabs(0.5f - zf * nzRec);
                zPart = (zPart * zPart) * dz2Rec;

    for (size_t y = 0; y < ny; y++)
    {
      const float yf    = static_cast<float>(y);
            float yPart = 0.5f - fabs(0.5f - yf * nyRec);
                  yPart = (yPart * yPart) * dy2Rec;

      const float yzPart = zPart + yPart;
      for (size_t x = 0; x < nx; x++)
      {
        const float xf = static_cast<float>(x);
              float xPart = 0.5f - fabs(0.5f - xf * nxRec);
                    xPart = (xPart * xPart) * dx2Rec;

              float k = cRefDtPi * sqrt(xPart + yzPart);

        // sourceKappa element
        sourceKappa[(z * ny + y) * nx + x] = cos(k);
      }//x
    }//y
  }// z
}// end of generateSourceKappa
//----------------------------------------------------------------------------------------------------------------------

/**
 * Generate kappa, absorb_nabla1, absorb_nabla2 for absorbing medium.
 */
void KSpaceFirstOrderSolver::generateKappaAndNablas()
{
  #pragma omp parallel
  {
    const float dxSqRec    = 1.0f / (mParameters.getDx() * mParameters.getDx());
    const float dySqRec    = 1.0f / (mParameters.getDy() * mParameters.getDy());
    const float dzSqRec    = 1.0f / (mParameters.getDz() * mParameters.getDz());

    const float cRefDt2    = mParameters.getCRef() * mParameters.getDt() * 0.5f;
    const float pi2        = static_cast<float>(M_PI) * 2.0f;

    const size_t nx        = mParameters.getFullDimensionSizes().nx;
    const size_t ny        = mParameters.getFullDimensionSizes().ny;
    const size_t nz        = mParameters.getFullDimensionSizes().nz;

    const float nxRec      = 1.0f / static_cast<float>(nx);
    const float nyRec      = 1.0f / static_cast<float>(ny);
    const float nzRec      = 1.0f / static_cast<float>(nz);

    const size_t nxComplex = mParameters.getReducedDimensionSizes().nx;
    const size_t nyComplex = mParameters.getReducedDimensionSizes().ny;
    const size_t nzComplex = mParameters.getReducedDimensionSizes().nz;

    float* kappa           = getKappa().getHostData();
    float* absorbNabla1    = getAbsorbNabla1().getHostData();
    float* absorbNabla2    = getAbsorbNabla2().getHostData();
    const float alphaPower = mParameters.getAlphaPower();

    #pragma omp for schedule (static)
    for (size_t z = 0; z < nzComplex; z++)
    {
      const float zf    = static_cast<float>(z);
            float zPart = 0.5f - fabs(0.5f - zf * nzRec);
                  zPart = (zPart * zPart) * dzSqRec;

      for (size_t y = 0; y < nyComplex; y++)
      {
        const float yf    = static_cast<float>(y);
              float yPart = 0.5f - fabs(0.5f - yf * nyRec);
                    yPart = (yPart * yPart) * dySqRec;

        const float yzPart = zPart + yPart;

        size_t i = (z * nyComplex + y) * nxComplex;

        for (size_t x = 0; x < nxComplex; x++)
        {
          const float xf    = static_cast<float>(x);
                float xPart = 0.5f - fabs(0.5f - xf * nxRec);
                      xPart = (xPart * xPart) * dxSqRec;

                float k     = pi2 * sqrt(xPart + yzPart);
                float cRefK = cRefDt2 * k;

          absorbNabla1[i]   = pow(k, alphaPower - 2);
          absorbNabla2[i]   = pow(k, alphaPower - 1);

          kappa[i]          = (cRefK == 0.0f) ? 1.0f : sin(cRefK) / cRefK;

          if (absorbNabla1[i] == std::numeric_limits<float>::infinity()) absorbNabla1[i] = 0.0f;
          if (absorbNabla2[i] == std::numeric_limits<float>::infinity()) absorbNabla2[i] = 0.0f;

          i++;
        }//x
      }//y
    }// z
  }// parallel
}// end of generateKappaAndNablas
//----------------------------------------------------------------------------------------------------------------------

/**
 * Generate absorbTau and absorbEta in for heterogenous medium.
 */
void KSpaceFirstOrderSolver::generateTauAndEta()
{

  if ((mParameters.getAlphaCoeffScalarFlag()) && (mParameters.getC0ScalarFlag()))
  { // scalar values
    const float alphaPower       = mParameters.getAlphaPower();
    const float tanPi2AlphaPower = tan(static_cast<float> (M_PI_2) * alphaPower);
    const float alphaNeperCoeff  = (100.0f * pow(1.0e-6f / (2.0f * static_cast<float>(M_PI)), alphaPower)) /
                                   (20.0f * static_cast<float>(M_LOG10E));

    const float alphaCoeff2      = 2.0f * mParameters.getAlphaCoeffScalar() * alphaNeperCoeff;

    mParameters.setAbsorbTauScalar((-alphaCoeff2) * pow(mParameters.getC0Scalar(), alphaPower - 1));
    mParameters.setAbsorbEtaScalar(  alphaCoeff2  * pow(mParameters.getC0Scalar(), alphaPower) * tanPi2AlphaPower);
  }
  else
  { // matrix
    #pragma omp parallel
    {
      const size_t nx  = mParameters.getFullDimensionSizes().nx;
      const size_t ny  = mParameters.getFullDimensionSizes().ny;
      const size_t nz  = mParameters.getFullDimensionSizes().nz;

      float* absorbTau = getAbsorbTau().getHostData();
      float* absorbEta = getAbsorbEta().getHostData();

      const bool   alphaCoeffScalarFlag = mParameters.getAlphaCoeffScalarFlag();
      const float  alphaCoeffScalar     = (alphaCoeffScalarFlag) ? mParameters.getAlphaCoeffScalar() : 0;
      const float* alphaCoeffMatrix     = (alphaCoeffScalarFlag) ? nullptr : getTemp1RealND().getHostData();

     // here the c2 hold just c0!
      const bool   c0ScalarFlag = mParameters.getC0ScalarFlag();
      const float  c0Scalar     = (c0ScalarFlag) ? mParameters.getC0Scalar() : 0;
      const float* cOMatrix     = (c0ScalarFlag) ? nullptr : getC2().getHostData();


      const float alphaPower       = mParameters.getAlphaPower();
      const float tanPi2AlphaPower = tan(static_cast<float>(M_PI_2) * alphaPower);

      //alpha = 100*alpha.*(1e-6/(2*pi)).^y./
      //                  (20*log10(exp(1)));
      const float alphaNeperCoeff = (100.0f * pow(1.0e-6f / (2.0f * static_cast<float>(M_PI)), alphaPower)) /
                                    (20.0f * static_cast<float>(M_LOG10E));


      #pragma omp for schedule (static)
      for (size_t z = 0; z < nz; z++)
      {
        for (size_t y = 0; y < ny; y++)
        {
          size_t i = (z * ny + y) * nx;
          for (size_t x = 0; x < nx; x++)
          {
            const float alphaCoeff2 = 2.0f * alphaNeperCoeff *
                                      ((alphaCoeffScalarFlag) ? alphaCoeffScalar : alphaCoeffMatrix[i]);

            absorbTau[i] = (-alphaCoeff2) * pow((c0ScalarFlag) ? c0Scalar : cOMatrix[i], alphaPower - 1);
            absorbEta[i] =   alphaCoeff2  * pow((c0ScalarFlag) ? c0Scalar : cOMatrix[i], alphaPower) * tanPi2AlphaPower;

            i++;
          }//x
        }//y
      }// z
    }// parallel
  } // matrix
}// end of generateTauAndEta
//----------------------------------------------------------------------------------------------------------------------

/**
 * Prepare dt./ rho0  for non-uniform grid.
 */
void KSpaceFirstOrderSolver::generateInitialDenisty()
{
  #pragma omp parallel
  {
    float* dtRho0Sgx   = getDtRho0Sgx().getHostData();
    float* dtRho0Sgy   = getDtRho0Sgy().getHostData();
    float* dtRho0Sgz   = getDtRho0Sgz().getHostData();

    const float dt = mParameters.getDt();

    const float* duxdxnSgx = getDxudxnSgx().getHostData();
    const float* duydynSgy = getDyudynSgy().getHostData();
    const float* duzdznSgz = getDzudznSgz().getHostData();

    const size_t nz = getDtRho0Sgx().getDimensionSizes().nz;
    const size_t ny = getDtRho0Sgx().getDimensionSizes().ny;
    const size_t nx = getDtRho0Sgx().getDimensionSizes().nx;

    const size_t sliceSize = (nx * ny);

    #pragma omp for schedule (static)
    for (size_t z = 0; z < nz; z++)
    {
      register size_t i = z * sliceSize;
      for (size_t y = 0; y < ny; y++)
      {
        for (size_t x = 0; x < nx; x++)
        {
          dtRho0Sgx[i] = (dt * duxdxnSgx[x]) / dtRho0Sgx[i];
          i++;
        } // x
      } // y
    } // z

    #pragma omp for schedule (static)
    for (size_t z = 0; z < nz; z++)
    {
      register size_t i = z * sliceSize;
      for (size_t y = 0; y < ny; y++)
      {
        const float duydynEl = duydynSgy[y];
        for (size_t x = 0; x < nx; x++)
        {
          dtRho0Sgy[i] = (dt * duydynEl) / dtRho0Sgy[i];
          i++;
        } // x
      } // y
    } // z


    #pragma omp for schedule (static)
    for (size_t z = 0; z < nz; z++)
    {
      register size_t i = z * sliceSize;
      const float duzdznEl = duzdznSgz[z];
      for (size_t y = 0; y < ny; y++)
      {
        for (size_t x = 0; x < nx; x++)
        {
          dtRho0Sgz[i] = (dt * duzdznEl) / dtRho0Sgz[i];
          i++;
        } // x
      } // y
    } // z
  } // parallel
}// end of generateInitialDenisty
//----------------------------------------------------------------------------------------------------------------------


/**
 * Compute c^2 on the CPU side.
 */
void KSpaceFirstOrderSolver::computeC2()
{
  if (!mParameters.getC0ScalarFlag())
  { // matrix
    float* c2 =  getC2().getHostData();

    #pragma omp parallel for schedule (static)
    for (size_t i=0; i < getC2().size(); i++)
    {
      c2[i] = c2[i] * c2[i];
    }
  }// matrix
}// computeC2
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute three temporary sums in the new pressure formula, non-linear absorbing case.
 */
void KSpaceFirstOrderSolver::computePressureTermsNonlinear(RealMatrix& densitySum,
                                                             RealMatrix& nonlinearTerm,
                                                             RealMatrix& velocityGradientSum)
{
  const float* bOnA = (mParameters.getBOnAScalarFlag()) ? nullptr : getBOnA().getDeviceData();
  const float* rho0 = (mParameters.getRho0ScalarFlag()) ? nullptr : getRho0().getDeviceData();

  SolverCudaKernels::computePressureTermsNonlinear(densitySum,
                                                   nonlinearTerm,
                                                   velocityGradientSum,
                                                   getRhoX(),
                                                   getRhoY(),
                                                   getRhoZ(),
                                                   getDuxdx(),
                                                   getDuydy(),
                                                   getDuzdz(),
                                                   mParameters.getBOnAScalarFlag(),
                                                   bOnA,
                                                   mParameters.getRho0ScalarFlag(),
                                                   rho0);

}// end of computePressureTermsNonlinear
//----------------------------------------------------------------------------------------------------------------------


/**
 * Calculate two temporary sums in the new pressure formula, linear absorbing case.
 */
void KSpaceFirstOrderSolver::computePressureTermsLinear(RealMatrix& densitySum,
                                                          RealMatrix& velocityGradientSum)
{
  const float* rho0 = (mParameters.getRho0ScalarFlag()) ? nullptr : getRho0().getDeviceData();

  SolverCudaKernels::computePressureTermsLinear(densitySum,
                                                velocityGradientSum,
                                                getRhoX(),
                                                getRhoY(),
                                                getRhoZ(),
                                                getDuxdx(),
                                                getDuydy(),
                                                getDuzdz(),
                                                mParameters.getRho0ScalarFlag(),
                                                rho0);
}// end of computePressurePartsLinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Sum sub-terms to calculate new pressure, non-linear case.
 */
void KSpaceFirstOrderSolver::sumPressureTermsNonlinear(const RealMatrix& absorbTauTerm,
                                                         const RealMatrix& absorbEtaTerm,
                                                         const RealMatrix& nonlinearTerm)
{
  const bool isC2Scalar         = mParameters.getC0ScalarFlag();
  // in sound speed and alpha coeff are sclars, then both AbsorbTau and absorbEta must be scalar.
  const bool areTauAndEtaScalars = mParameters.getC0ScalarFlag() && mParameters.getAlphaCoeffScalarFlag();

  const float* c2        = (isC2Scalar)          ? nullptr : getC2().getDeviceData();
  const float* absorbTau = (areTauAndEtaScalars) ? nullptr : getAbsorbTau().getDeviceData();
  const float* absorbEta = (areTauAndEtaScalars) ? nullptr : getAbsorbEta().getDeviceData();

  SolverCudaKernels::sumPressureTermsNonlinear(getP(),
                                               nonlinearTerm,
                                               absorbTauTerm,
                                               absorbEtaTerm,
                                               isC2Scalar,
                                               c2,
                                               areTauAndEtaScalars,
                                               absorbTau,
                                               absorbEta);
}// end of sumPressureTermsNonlinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Sum sub-terms to calculate new pressure, linear case.
 */
void KSpaceFirstOrderSolver::sumPressureTermsLinear(const RealMatrix& absorbTauTerm,
                                                      const RealMatrix& absorbEtaTerm,
                                                      const RealMatrix& densitySum)
{
  const bool isC2Scalar          = mParameters.getC0ScalarFlag();
  const bool areTauAndEtaScalars = mParameters.getC0ScalarFlag() && mParameters.getAlphaCoeffScalarFlag();

  const float* c2        = (isC2Scalar)          ? nullptr : getC2().getDeviceData();
  const float* absorbTau = (areTauAndEtaScalars) ? nullptr : getAbsorbTau().getDeviceData();
  const float* absorbEta = (areTauAndEtaScalars) ? nullptr : getAbsorbEta().getDeviceData();

  SolverCudaKernels::sumPressureTermsLinear(getP(),
                                            absorbTauTerm,
                                            absorbEtaTerm,
                                            densitySum,
                                            isC2Scalar,
                                            c2,
                                            areTauAndEtaScalars,
                                            absorbTau,
                                            absorbEta);
}// end of sumPressureTermsLinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Sum sub-terms for new pressure, non-linear lossless case.
 */
void KSpaceFirstOrderSolver::sumPressureTermsNonlinearLossless()
{
  const bool   isC2Scalar   = mParameters.getC0ScalarFlag();
  const bool   isBOnAScalar = mParameters.getBOnAScalarFlag();
  const bool   isRho0Scalar = mParameters.getRho0ScalarFlag();

  const float* c2   = (isC2Scalar)   ? nullptr : getC2().getDeviceData();
  const float* bOnA = (isBOnAScalar) ? nullptr : getBOnA().getDeviceData();
  const float* rho0 = (isRho0Scalar) ? nullptr : getRho0().getDeviceData();

  SolverCudaKernels::sumPressureNonlinearLossless(getP(),
                                                  getRhoX(),
                                                  getRhoY(),
                                                  getRhoZ(),
                                                  isC2Scalar,
                                                  c2,
                                                  isBOnAScalar,
                                                  bOnA,
                                                  isRho0Scalar,
                                                  rho0);

}// end of sumPressureTermsNonlinearLossless
//----------------------------------------------------------------------------------------------------------------------

/**
 * Sum sub-terms for new pressure, linear lossless case.
 */
void KSpaceFirstOrderSolver::sumPressureTermsLinearLossless()
{
  const float* c2  = (mParameters.getC0ScalarFlag()) ? nullptr : getC2().getDeviceData();

  SolverCudaKernels::sumPressureLinearLossless(getP(),
                                               getRhoX(),
                                               getRhoY(),
                                               getRhoZ(),
                                               mParameters.getC0ScalarFlag(),
                                               c2);

}// end of sumPressureTermsLinearLossless
//----------------------------------------------------------------------------------------------------------------------


/**
 * Calculated shifted velocities.
 * \n
 * ux_shifted = real(ifft(bsxfun(\@times, x_shift_neg, fft(ux_sgx, [], 1)), [], 1)); \n
 * uy_shifted = real(ifft(bsxfun(\@times, y_shift_neg, fft(uy_sgy, [], 2)), [], 2)); \n
 * uz_shifted = real(ifft(bsxfun(\@times, z_shift_neg, fft(uz_sgz, [], 3)), [], 3)); \n
 */

void KSpaceFirstOrderSolver::computeShiftedVelocity()
{
  // uxShifted
  getTempCufftShift().computeR2CFft1DX(getUxSgx());
  SolverCudaKernels::computeVelocityShiftInX(getTempCufftShift(), getXShiftNegR());
  getTempCufftShift().computeC2RFft1DX(getUxShifted());

  // uyShifted
  getTempCufftShift().computeR2CFft1DY(getUySgy());
  SolverCudaKernels::computeVelocityShiftInY(getTempCufftShift(), getYShiftNegR());
  getTempCufftShift().computeC2RFft1DY(getUyShifted());

  // uzShifted
  getTempCufftShift().computeR2CFft1DZ(getUzSgz());
  SolverCudaKernels::computeVelocityShiftInZ(getTempCufftShift(), getZShiftNegR());
  getTempCufftShift().computeC2RFft1DZ(getUzShifted());

}// end of computeShiftedVelocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print progress statistics.
 */
void KSpaceFirstOrderSolver::printStatistics()
{
  const float  nt = static_cast<float>(mParameters.getNt());
  const size_t timeIndex = mParameters.getTimeIndex();

  if (timeIndex > (mActPercent * nt * 0.01f))
  {
    mActPercent += mParameters.getProgressPrintInterval();

    mIterationTime.stop();

    const double elTime = mIterationTime.getElapsedTime();
    const double elTimeWithLegs = mIterationTime.getElapsedTime() + mSimulationTime.getElapsedTimeOverPreviousLegs();
    const double toGo   = ((elTimeWithLegs / static_cast<double>((timeIndex + 1)) *  nt)) - elTimeWithLegs;

    struct tm* current;
    time_t now;
    time(&now);
    now += toGo;
    current = localtime(&now);

    Logger::log(Logger::LogLevel::kBasic,
                kOutFmtSimulationProgress,
                static_cast<size_t>(((timeIndex) / (nt * 0.01f))),'%',
                elTime, toGo,
                current->tm_mday, current->tm_mon+1, current->tm_year-100,
                current->tm_hour, current->tm_min, current->tm_sec);
    Logger::flush(Logger::LogLevel::kBasic);
  }
}// end of printStatistics
//----------------------------------------------------------------------------------------------------------------------

/**
 * Is time to checkpoint?
 */
bool KSpaceFirstOrderSolver::isTimeToCheckpoint()
{
  if (!mParameters.isCheckpointEnabled()) return false;

  mTotalTime.stop();

  return (mTotalTime.getElapsedTime() > static_cast<float>(mParameters.getCheckpointInterval()));
}// end of isTimeToCheckpoint
//----------------------------------------------------------------------------------------------------------------------


/**
 * Was the loop interrupted to checkpoint?
 */
bool KSpaceFirstOrderSolver::isCheckpointInterruption() const
{
  return (mParameters.getTimeIndex() != mParameters.getNt());
}// end of isCheckpointInterruption
//----------------------------------------------------------------------------------------------------------------------

/**
 * Check the output file has the correct format and version.
 */
void KSpaceFirstOrderSolver::checkOutputFile()
{
  // The header has already been read
  Hdf5FileHeader& fileHeader = mParameters.getFileHeader();
  Hdf5File&       outputFile = mParameters.getOutputFile();

  // test file type
  if (fileHeader.getFileType() != Hdf5FileHeader::FileType::kOutput)
  {
    throw ios::failure(Logger::formatMessage(kErrFmtBadOutputFileFormat,
                                             mParameters.getOutputFileName().c_str()));
  }

  // test file major version
  if (!fileHeader.checkMajorFileVersion())
  {
    throw ios::failure(Logger::formatMessage(kErrFmtBadMajorFileVersion,
                                             mParameters.getOutputFileName().c_str(),
                                             fileHeader.getFileMajorVersion().c_str()));
  }

  // test file minor version
  if (!fileHeader.checkMinorFileVersion())
  {
    throw ios::failure(Logger::formatMessage(kErrFmtBadMinorFileVersion,
                                             mParameters.getOutputFileName().c_str(),
                                             fileHeader.getFileMinorVersion().c_str()));
  }

  // Check dimension sizes
  DimensionSizes outputDimSizes;
  outputFile.readScalarValue(outputFile.getRootGroup(), kNxName, outputDimSizes.nx);
  outputFile.readScalarValue(outputFile.getRootGroup(), kNyName, outputDimSizes.ny);
  outputFile.readScalarValue(outputFile.getRootGroup(), kNzName, outputDimSizes.nz);

 if (mParameters.getFullDimensionSizes() != outputDimSizes)
 {
    throw ios::failure(Logger::formatMessage(kErrFmtOutputDimensionsMismatch,
                                             outputDimSizes.nx,
                                             outputDimSizes.ny,
                                             outputDimSizes.nz,
                                             mParameters.getFullDimensionSizes().nx,
                                             mParameters.getFullDimensionSizes().ny,
                                             mParameters.getFullDimensionSizes().nz));
 }
}// end of checkOutputFile
//----------------------------------------------------------------------------------------------------------------------


/**
 * Check the file type and the version of the checkpoint file.
 */
void KSpaceFirstOrderSolver::checkCheckpointFile()
{
  // read the header and check the file version
  Hdf5FileHeader fileHeader;
  Hdf5File&      checkpointFile = mParameters.getCheckpointFile();

  fileHeader.readHeaderFromCheckpointFile(checkpointFile);

  // test file type
  if (fileHeader.getFileType() != Hdf5FileHeader::FileType::kCheckpoint)
  {
    throw ios::failure(Logger::formatMessage(kErrFmtBadCheckpointFileFormat,
                                             mParameters.getCheckpointFileName().c_str()));
  }

  // test file major version
  if (!fileHeader.checkMajorFileVersion())
  {
    throw ios::failure(Logger::formatMessage(kErrFmtBadMajorFileVersion,
                                             mParameters.getCheckpointFileName().c_str(),
                                             fileHeader.getFileMajorVersion().c_str()));
  }

  // test file minor version
  if (!fileHeader.checkMinorFileVersion())
  {
    throw ios::failure(Logger::formatMessage(kErrFmtBadMinorFileVersion,
                                             mParameters.getCheckpointFileName().c_str(),
                                             fileHeader.getFileMinorVersion().c_str()));
  }

  // Check dimension sizes
  DimensionSizes checkpointDimSizes;
  checkpointFile.readScalarValue(checkpointFile.getRootGroup(), kNxName, checkpointDimSizes.nx);
  checkpointFile.readScalarValue(checkpointFile.getRootGroup(), kNyName, checkpointDimSizes.ny);
  checkpointFile.readScalarValue(checkpointFile.getRootGroup(), kNzName, checkpointDimSizes.nz);

 if (mParameters.getFullDimensionSizes() != checkpointDimSizes)
 {
   throw ios::failure(Logger::formatMessage(kErrFmtCheckpointDimensionsMismatch,
                                            checkpointDimSizes.nx,
                                            checkpointDimSizes.ny,
                                            checkpointDimSizes.nz,
                                            mParameters.getFullDimensionSizes().nx,
                                            mParameters.getFullDimensionSizes().ny,
                                            mParameters.getFullDimensionSizes().nz));
 }
}// end of checkCheckpointFile
//----------------------------------------------------------------------------------------------------------------------

/**
 * Restore cumulated elapsed time from the output file.
 */
void KSpaceFirstOrderSolver::loadElapsedTimeFromOutputFile()
{
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

}// end of loadElapsedTimeFromOutputFile
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//
