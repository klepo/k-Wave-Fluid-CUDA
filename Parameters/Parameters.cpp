/**
 * @file      Parameters.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file containing parameters of the simulation.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      09 August    2012, 13:39 (created) \n
 *            14 March     2019, 09:47 (revised)
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

#ifdef _OPENMP
  #include <omp.h>
#endif

// Windows build needs to undefine macro MINMAX to support std::limits
#ifdef _WIN64
  #ifndef NOMINMAX
    # define NOMINMAX
  #endif
  
  #include <windows.h>
#endif

#include <string>
#include <exception>
#include <stdexcept>
#include <limits>

#include <Parameters/Parameters.h>
#include <Parameters/CudaParameters.h>
#include <Utils/MatrixNames.h>
#include <Logger/Logger.h>

#include "Utils/TimeMeasure.h"

using std::ios;
using std::string;

//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Constants -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Variables -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

// initialization of the singleton instance flag
bool Parameters::sParametersInstanceFlag   = false;

// initialization of the instance
Parameters* Parameters::sPrametersInstance = nullptr;

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Destructor.
 */
Parameters::~Parameters()
{
  sParametersInstanceFlag = false;
  if (sPrametersInstance)
  {
    delete sPrametersInstance;
  }
  sPrametersInstance = nullptr;
};
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get instance of singleton class.
 */
Parameters& Parameters::getInstance()
{
  if(!sParametersInstanceFlag)
  {
    sPrametersInstance = new Parameters();
    sParametersInstanceFlag = true;
    return *sPrametersInstance;
  }
  else
  {
    return *sPrametersInstance;
  }
}// end of getInstance()
//----------------------------------------------------------------------------------------------------------------------

/**
 * Parse command line and read scalar values from the input file to initialise the class and the simulation.
 */
void Parameters::init(int argc, char** argv)
{
  mCommandLineParameters.parseCommandLine(argc, argv);

  if (getGitHash() != "")
  {
    Logger::log(Logger::LogLevel::kFull, kOutFmtGitHashLeft, getGitHash().c_str());
    Logger::log(Logger::LogLevel::kFull, kOutFmtSeparator);
  }
  if (mCommandLineParameters.isPrintVersionOnly())
  {
    return;
  }

  Logger::log(Logger::LogLevel::kBasic, kOutFmtReadingConfiguration);
  readScalarsFromInputFile();

  if (mCommandLineParameters.isBenchmarkEnabled())
  {
    mNt = mCommandLineParameters.getBenchmarkTimeStepsCount();
  }

  if ((mCommandLineParameters.getSamplingStartTimeIndex() >= mNt ) ||
      (mCommandLineParameters.getSamplingStartTimeIndex() < 0))
  {
    throw std::invalid_argument(Logger::formatMessage(kErrFmtIllegalSamplingStartTimeStep, 1l, mNt));
  }

  // Checkpoint by number of time steps
  if (mCommandLineParameters.getCheckpointTimeSteps() > 0)
  {
    mTimeStepsToCheckpoint = mCommandLineParameters.getCheckpointTimeSteps();
  }

  Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
}// end of parseCommandLine
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print parameters of the simulation based in the actual level of verbosity.
 */
void Parameters::printSimulatoinSetup()
{
  Logger::log(Logger::LogLevel::kBasic, kOutFmtNumberOfThreads, getNumberOfThreads());

  Logger::log(Logger::LogLevel::kBasic, kOutFmtSimulationDetailsTitle);


  const string domainsSizes = (isSimulation3D()) ? Logger::formatMessage(kOutFmt3DDomainSizeFormat,
                                                                         getFullDimensionSizes().nx,
                                                                         getFullDimensionSizes().ny,
                                                                         getFullDimensionSizes().nz)
                                                 : Logger::formatMessage(kOutFmt2DDomainSizeFormat,
                                                                         getFullDimensionSizes().nx,
                                                                         getFullDimensionSizes().ny);

  // Print simulation size
  Logger::log(Logger::LogLevel::kBasic, kOutFmtDomainSize, domainsSizes.c_str());

  Logger::log(Logger::LogLevel::kBasic, kOutFmtSimulatoinLenght, getNt());

  // Print all command line parameters
  mCommandLineParameters.printComandlineParamers();

  if (getSensorMaskType() == SensorMaskType::kIndex)
  {
    Logger::log(Logger::LogLevel::kAdvanced, kOutFmtSensorMaskIndex);
  }
  if (getSensorMaskType() == SensorMaskType::kCorners)
  {
    Logger::log(Logger::LogLevel::kAdvanced, kOutFmtSensorMaskCuboid);
  }
}// end of printSimulatoinSetup
//----------------------------------------------------------------------------------------------------------------------

/**
 * Read scalar values from the input HDF5 file.
 */
void Parameters::readScalarsFromInputFile()
{
  DimensionSizes scalarSizes(1, 1, 1);

  if (!mInputFile.isOpen())
  {
    // Open file -- exceptions handled in main
    mInputFile.open(mCommandLineParameters.getInputFileName(), H5F_ACC_RDWR);
  }

  mFileHeader.readHeaderFromInputFile(mInputFile);

  // check file type
  if (mFileHeader.getFileType() != Hdf5FileHeader::FileType::kInput)
  {
    throw ios::failure(Logger::formatMessage(kErrFmtBadInputFileFormat, getInputFileName().c_str()));
  }

  // check version
  if (!mFileHeader.checkMajorFileVersion())
  {
    throw ios::failure(Logger::formatMessage(kErrFmtBadMajorFileVersion,
                                             getInputFileName().c_str(),
                                             mFileHeader.getFileMajorVersion().c_str()));
  }

  if (!mFileHeader.checkMinorFileVersion())
  {
    throw ios::failure(Logger::formatMessage(kErrFmtBadMinorFileVersion,
                                             getInputFileName().c_str(),
                                             mFileHeader.getFileMinorVersion().c_str()));
  }

  const hid_t rootGroup = mInputFile.getRootGroup();

  size_t x, y, z;
  mInputFile.readScalarValue(rootGroup, kNxName, x);
  mInputFile.readScalarValue(rootGroup, kNyName, y);
  mInputFile.readScalarValue(rootGroup, kNzName, z);

  mFullDimensionSizes    = DimensionSizes(x, y, z);
  mReducedDimensionSizes = DimensionSizes(((x / 2) + 1), y, z);

  mInputFile.readScalarValue(rootGroup, kNtName, mNt);

  mInputFile.readScalarValue(rootGroup, kDtName, mDt);
  mInputFile.readScalarValue(rootGroup, kDxName, mDx);
  mInputFile.readScalarValue(rootGroup, kDyName, mDy);
  if (isSimulation3D())
  {
   mInputFile.readScalarValue(rootGroup, kDzName, mDz);
  }

  mInputFile.readScalarValue(rootGroup, kCRefName,     mCRef);
  mInputFile.readScalarValue(rootGroup, kPmlXSizeName, mPmlXSize);
  mInputFile.readScalarValue(rootGroup, kPmlYSizeName, mPmlYSize);
  if (isSimulation3D())
  {
    mInputFile.readScalarValue(rootGroup, kPmlZSizeName, mPmlZSize);
  }

  mInputFile.readScalarValue(rootGroup, kPmlXAlphaName, mPmlXAlpha);
  mInputFile.readScalarValue(rootGroup, kPmlYAlphaName, mPmlYAlpha);
  if (isSimulation3D())
  {
    mInputFile.readScalarValue(rootGroup, kPmlZAlphaName, mPmlZAlpha);
  }

  // if the file is of version 1.0, there must be a sensor mask index (backward compatibility)
  if (mFileHeader.getFileVersion() == Hdf5FileHeader::FileVersion::kVersion10)
  {
    mSensorMaskIndexSize = mInputFile.getDatasetSize(rootGroup, kSensorMaskIndexName);

    //if -u_non_staggered_raw enabled, throw an error - not supported
    if (getStoreVelocityNonStaggeredRawFlag() || getStoreVelocityNonStaggeredCFlag())
    {
      throw ios::failure(kErrFmtNonStaggeredVelocityNotSupportedFileVersion);
    }
  }// version 1.0

  // This is the current version 1.1
  if (mFileHeader.getFileVersion() == Hdf5FileHeader::FileVersion::kVersion11)
  {
    // read sensor mask type as a size_t value to enum
    size_t sensorMaskTypeNumericValue = 0;
    mInputFile.readScalarValue(rootGroup, kSensorMaskTypeName, sensorMaskTypeNumericValue);

    // convert the long value on
    switch (sensorMaskTypeNumericValue)
    {
      case 0:
      {
        mSensorMaskType = SensorMaskType::kIndex;
        break;
      }
      case 1:
      {
        mSensorMaskType = SensorMaskType::kCorners;
        break;
      }
      default:
      {
        throw ios::failure(kErrFmtBadSensorMaskType);
        break;
      }
    }//case

    // read the input mask size
    switch (mSensorMaskType)
    {
      case SensorMaskType::kIndex:
      {
        mSensorMaskIndexSize = mInputFile.getDatasetSize(rootGroup, kSensorMaskIndexName);
        break;
      }
      case SensorMaskType::kCorners:
      {
        // mask dimensions are [6, N, 1] - I want to know N
        mSensorMaskCornersSize = mInputFile.getDatasetDimensionSizes(rootGroup, kSensorMaskCornersName).ny;
        break;
      }
    }// switch
  }// version 1.1

  // flags

  mInputFile.readScalarValue(rootGroup, kPressureSourceFlagName,        mPressureSourceFlag);
  mInputFile.readScalarValue(rootGroup, kInitialPressureSourceFlagName, mInitialPressureSourceFlag);

  mInputFile.readScalarValue(rootGroup, kTransducerSourceFlagName,      mTransducerSourceFlag);

  mInputFile.readScalarValue(rootGroup, kVelocityXSourceFlagName,  mVelocityXSourceFlag);
  mInputFile.readScalarValue(rootGroup, kVelocityYSourceFlagName,  mVelocityYSourceFlag);
  if (isSimulation3D())
  {
  mInputFile.readScalarValue(rootGroup, kVelocityZSourceFlagName,  mVelocityZSourceFlag);
  }

  mInputFile.readScalarValue(rootGroup, kNonUniformGridFlagName, mNonUniformGridFlag);
  mInputFile.readScalarValue(rootGroup, kAbsorbingFlagName,      mAbsorbingFlag);
  mInputFile.readScalarValue(rootGroup, kNonLinearFlagName,      mNonLinearFlag);

  // Vector sizes.
  if (mTransducerSourceFlag == 0)
  {
    mTransducerSourceInputSize = 0;
  }
  else
  {
    mTransducerSourceInputSize =mInputFile.getDatasetSize(rootGroup, kTransducerSourceInputName);
  }

  // in 2D mVelocityZSourceFlag is always 0
  if ((mTransducerSourceFlag > 0) || (mVelocityXSourceFlag > 0) ||
      (mVelocityYSourceFlag > 0)  || (mVelocityZSourceFlag > 0))
  {
    mVelocitySourceIndexSize = mInputFile.getDatasetSize(rootGroup, kVelocitySourceIndexName);
  }

  // uxyz_source_flags,  in 2D mVelocityZSourceFlag is always 0
  if ((mVelocityXSourceFlag > 0) || (mVelocityYSourceFlag > 0) || (mVelocityZSourceFlag > 0))
  {
    mInputFile.readScalarValue(rootGroup, kVelocitySourceManyName, mVelocitySourceMany);

    size_t sourceModeNumericValue = 0;
    mInputFile.readScalarValue(rootGroup, kVelocitySourceModeName, sourceModeNumericValue);

    // convert the size_t value to enum
    switch (sourceModeNumericValue)
    {
      case 0: mVelocitySourceMode = SourceMode::kDirichlet;
        break;
      case 1: mVelocitySourceMode = SourceMode::kAdditiveNoCorrection;
        break;
      case 2: mVelocitySourceMode = SourceMode::kAdditive;
        break;
      default:
      {
        throw ios::failure(kErrFmtBadVelocitySourceMode);
        break;
      }
    }//case
  }
  else
  {
    mVelocitySourceMany = 0;
    mVelocitySourceMode = SourceMode::kDirichlet;
  }

  // p_source_flag
  if (mPressureSourceFlag != 0)
  {
    mInputFile.readScalarValue(rootGroup, kPressureSourceManyName, mPressureSourceMany);

    size_t sourceModeNumericValue = 0;
    mInputFile.readScalarValue(rootGroup, kPressureSourceModeName, sourceModeNumericValue);

    // convert the size_t value to enum
    switch (sourceModeNumericValue)
    {
      case 0: mPressureSourceMode = SourceMode::kDirichlet;
        break;
      case 1: mPressureSourceMode = SourceMode::kAdditiveNoCorrection;
        break;
      case 2: mPressureSourceMode = SourceMode::kAdditive;
        break;
      default:
      {
        throw ios::failure(kErrFmtBadPressureSourceMode);
        break;
      }
    }//case

    mPressureSourceIndexSize = mInputFile.getDatasetSize(rootGroup, kPressureSourceIndexName);
  }
  else
  {
    mPressureSourceMode = SourceMode::kDirichlet;
    mPressureSourceMany = 0;
    mPressureSourceIndexSize = 0;
  }

  // absorb flag.
  if (mAbsorbingFlag != 0)
  {
    mInputFile.readScalarValue(rootGroup, kAlphaPowerName, mAlphaPower);
    if (mAlphaPower == 1.0f)
    {
      throw std::invalid_argument(kErrFmtIllegalAlphaPowerValue);
    }

    mAlphaCoeffScalarFlag = mInputFile.getDatasetDimensionSizes(rootGroup, kAlphaCoeffName) == scalarSizes;

    if (mAlphaCoeffScalarFlag)
    {
      mInputFile.readScalarValue(rootGroup, kAlphaCoeffName, mAlphaCoeffScalar);
    }
  }

  mC0ScalarFlag = mInputFile.getDatasetDimensionSizes(rootGroup, kC0Name) == scalarSizes;
  if (mC0ScalarFlag)
  {
    mInputFile.readScalarValue(rootGroup, kC0Name, mC0Scalar);
  }

  if (mNonLinearFlag)
  {
    mBOnAScalarFlag = mInputFile.getDatasetDimensionSizes(rootGroup, kBonAName) == scalarSizes;
    if (mBOnAScalarFlag)
    {
      mInputFile.readScalarValue(rootGroup, kBonAName, mBOnAScalar);
    }
  }

  mRho0ScalarFlag = mInputFile.getDatasetDimensionSizes(rootGroup, kRho0Name) == scalarSizes;
  if (mRho0ScalarFlag)
  {
    mInputFile.readScalarValue(rootGroup, kRho0Name,      mRho0Scalar);
    mInputFile.readScalarValue(rootGroup, kRho0SgxName,   mRho0SgxScalar);
    mInputFile.readScalarValue(rootGroup, kRho0SgyName,   mRho0SgyScalar);
    if (isSimulation3D())
    {
      mInputFile.readScalarValue(rootGroup, kRho0SgzName, mRho0SgzScalar);
    }
  }
  // Init compression stuff
  if (mCommandLineParameters.getStorePressureCFlag() || mCommandLineParameters.getStoreVelocityCFlag() || mCommandLineParameters.getStoreVelocityNonStaggeredCFlag())
  {
    if (mCommandLineParameters.getPeriod() > 0 && mCommandLineParameters.getFrequency() > 0)
    {
      throw ios::failure(Logger::formatMessage(kErrFmtBadPeriodAndFrequencyValue));
    }

    if (mCommandLineParameters.getFrequency() > 0)
    {
      try
      {
        mCommandLineParameters.mPeriod = 1.0f / (mCommandLineParameters.getFrequency() * mDt);
        mInputFile.writeFloatAttribute(rootGroup, kPressureSourceInputName, Hdf5File::kPeriodName, mCommandLineParameters.getPeriod());
      }
      catch (...)
      {
        throw ios::failure(Logger::formatMessage(kErrFmtCannotComputePeriodValue));
      }
    }

    if (!(mCommandLineParameters.getPeriod() > 0) && mPressureSourceFlag)
    {
      /*try
      {
        mCommandLineParameters.mPeriod = mInputFile.readFloatAttribute(rootGroup, kPressureSourceInputName, Hdf5File::kPeriodName);
      }
      catch (...)*/
      {
        mCommandLineParameters.mPeriod = 0;
        DimensionSizes size = mInputFile.getDatasetDimensionSizes(rootGroup, kPressureSourceInputName);
        hsize_t length = 0;
        hsize_t limit = 500;
        length = size.ny > limit ? limit : size.ny;
        float *data = static_cast<float *>(_mm_malloc(length * sizeof(float), kDataAlignment));
        // Read only part of the dataset
        hid_t dataset = mInputFile.openDataset(rootGroup, kPressureSourceInputName);
        mInputFile.readHyperSlab(dataset, DimensionSizes(size.nx / 2, size.ny - length, 0), DimensionSizes(1, length, 1), data);
        mInputFile.closeDataset(dataset);
        // Compute period
        mCommandLineParameters.mPeriod = CompressHelper::findPeriod(data, length);
        _mm_free(data);
        mInputFile.writeFloatAttribute(rootGroup, kPressureSourceInputName, Hdf5File::kPeriodName, mCommandLineParameters.getPeriod());
      }
    }

    if (!(mCommandLineParameters.getPeriod() > 0))
    {
      throw ios::failure(Logger::formatMessage(kErrFmtMissingPeriodValue));
    }


    if (!(mCommandLineParameters.getFrequency() > 0))
    {
      mCommandLineParameters.mFrequency = 1.0f / (mCommandLineParameters.getPeriod() * mDt);
    }

    // Create compression helper
    CompressHelper &compressHelper = CompressHelper::getInstance();
    compressHelper.init(mCommandLineParameters.getPeriod(), mCommandLineParameters.getMOS(), mCommandLineParameters.getHarmonics(), true);

    // Set compression sizes for temp buffers
    // complex number -> * 2
    if (mSensorMaskType == SensorMaskType::kIndex)
    {
      mCompressedDimensionSizes = DimensionSizes(mSensorMaskIndexSize * compressHelper.getHarmonics() * 2, 1, 1);
    }
    else if (mSensorMaskType == SensorMaskType::kCorners)
    {
      DimensionSizes size = mInputFile.getDatasetDimensionSizes(rootGroup, kSensorMaskCornersName);
      size_t *data = static_cast<size_t *>(_mm_malloc(size.nElements() * sizeof(size_t), kDataAlignment));
      mInputFile.readCompleteDataset(rootGroup, kSensorMaskCornersName, size, data);
      size_t cuboidsSize = 0;
      for (size_t i = 0; i < mSensorMaskCornersSize; i++)
      {
        DimensionSizes corner1(data[i * 6], data[i * 6 + 1], data[i * 6 + 2]);
        DimensionSizes corner2(data[i * 6 + 3], data[i * 6 + 4], data[i * 6 + 5]);
        cuboidsSize += (corner2 - corner1).nElements();
      }
      _mm_free(data);

      mCompressedDimensionSizes = DimensionSizes(cuboidsSize * compressHelper.getHarmonics() * 2, 1, 1);
    }
  }
}// end of readScalarsFromInputFile
//----------------------------------------------------------------------------------------------------------------------

/**
 * Save scalars into the output HDF5 file.
 */
void Parameters::saveScalarsToOutputFile()
{
  const hid_t rootGroup = mOutputFile.getRootGroup();

  // Write dimension sizes
  mOutputFile.writeScalarValue(rootGroup, kNxName, mFullDimensionSizes.nx);
  mOutputFile.writeScalarValue(rootGroup, kNyName, mFullDimensionSizes.ny);
  mOutputFile.writeScalarValue(rootGroup, kNzName, mFullDimensionSizes.nz);

  mOutputFile.writeScalarValue(rootGroup, kNtName, mNt);

  mOutputFile.writeScalarValue(rootGroup, kDtName, mDt);
  mOutputFile.writeScalarValue(rootGroup, kDxName, mDx);
  mOutputFile.writeScalarValue(rootGroup, kDyName, mDy);
  if (isSimulation3D())
  {
    mOutputFile.writeScalarValue(rootGroup, kDzName, mDz);
  }

  mOutputFile.writeScalarValue(rootGroup, kCRefName, mCRef);

  mOutputFile.writeScalarValue(rootGroup, kPmlXSizeName,    mPmlXSize);
  mOutputFile.writeScalarValue(rootGroup, kPmlYSizeName,    mPmlYSize);
  if (isSimulation3D())
  {
    mOutputFile.writeScalarValue(rootGroup, kPmlZSizeName,  mPmlZSize);
  }

  mOutputFile.writeScalarValue(rootGroup, kPmlXAlphaName,   mPmlXAlpha);
  mOutputFile.writeScalarValue(rootGroup, kPmlYAlphaName,   mPmlYAlpha);
  if (isSimulation3D())
  {
    mOutputFile.writeScalarValue(rootGroup, kPmlZAlphaName, mPmlZAlpha);
  }

  mOutputFile.writeScalarValue(rootGroup, kVelocityXSourceFlagName,   mVelocityXSourceFlag);
  mOutputFile.writeScalarValue(rootGroup, kVelocityYSourceFlagName,   mVelocityYSourceFlag);
  if (isSimulation3D())
  {
    mOutputFile.writeScalarValue(rootGroup, kVelocityZSourceFlagName, mVelocityZSourceFlag);
  }

  mOutputFile.writeScalarValue(rootGroup, kTransducerSourceFlagName,  mTransducerSourceFlag);

  mOutputFile.writeScalarValue(rootGroup, kPressureSourceFlagName,        mPressureSourceFlag);
  mOutputFile.writeScalarValue(rootGroup, kInitialPressureSourceFlagName, mInitialPressureSourceFlag);

  mOutputFile.writeScalarValue(rootGroup, kNonUniformGridFlagName, mNonUniformGridFlag);
  mOutputFile.writeScalarValue(rootGroup, kAbsorbingFlagName,      mAbsorbingFlag);
  mOutputFile.writeScalarValue(rootGroup, kNonLinearFlagName,      mNonLinearFlag);

  // uxyz_source_flags.
  if ((mVelocityXSourceFlag > 0) || (mVelocityYSourceFlag > 0) || (mVelocityZSourceFlag > 0))
  {
    mOutputFile.writeScalarValue(rootGroup, kVelocitySourceManyName, mVelocitySourceMany);
    mOutputFile.writeScalarValue(rootGroup, kVelocitySourceModeName, static_cast<size_t>(mVelocitySourceMode));
  }

  // p_source_flag.
  if (mPressureSourceFlag != 0)
  {
    mOutputFile.writeScalarValue(rootGroup, kPressureSourceManyName, mPressureSourceMany);
    mOutputFile.writeScalarValue(rootGroup, kPressureSourceModeName, static_cast<size_t>(mPressureSourceMode));
  }

  // absorb flag
  if (mAbsorbingFlag != 0)
  {
    mOutputFile.writeScalarValue(rootGroup, kAlphaPowerName, mAlphaPower);
  }

  // if copy sensor mask, then copy the mask type
  if (getCopySensorMaskFlag())
  {
    size_t sensorMaskTypeNumericValue = 0;

    switch (mSensorMaskType)
    {
      case SensorMaskType::kIndex: sensorMaskTypeNumericValue = 0;
        break;
      case SensorMaskType::kCorners: sensorMaskTypeNumericValue = 1;
        break;
    }// switch

    mOutputFile.writeScalarValue(rootGroup, kSensorMaskTypeName, sensorMaskTypeNumericValue);
  }
}// end of saveScalarsToFile
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get GitHash of the code
 */
string Parameters::getGitHash() const
{
#if (defined (__KWAVE_GIT_HASH__))
  return string(__KWAVE_GIT_HASH__);
#else
  return "";
#endif
}// end of getGitHash
//----------------------------------------------------------------------------------------------------------------------

/**
 * Select a device device for execution.
 */
void Parameters::selectDevice()
{
  Logger::log(Logger::LogLevel::kBasic, kOutFmtSelectedDevice);
  Logger::flush(Logger::LogLevel::kBasic);

  int deviceIdx = mCommandLineParameters.getCudaDeviceIdx();
  mCudaParameters.selectDevice(deviceIdx); // throws an exception when wrong

  Logger::log(Logger::LogLevel::kBasic, kOutFmtDeviceId, mCudaParameters.getDeviceIdx());

  Logger::log(Logger::LogLevel::kBasic, kOutFmtDeviceName, mCudaParameters.getDeviceName().c_str());
}// end of selectDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Is time to checkpoint?
 */
bool Parameters::isTimeToCheckpoint(TimeMeasure timer) const
{
  timer.stop();

  const auto checkpointInterval = mCommandLineParameters.getCheckpointInterval();

  return (isCheckpointEnabled() &&
          ((mTimeStepsToCheckpoint == 0) ||
           (( checkpointInterval > 0) && (timer.getElapsedTime() > float(checkpointInterval)))
          )
         );
}// end of isTimeToCheckpoint
//----------------------------------------------------------------------------------------------------------------------

/**
 * Increment simulation time step and decrement steps to checkpoint.
 */
void Parameters::incrementTimeIndex()
{
  mTimeIndex++;
  mTimeStepsToCheckpoint--;
 } // end of incrementTimeIndex
//----------------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


/**
 * Constructor.
 */
Parameters::Parameters()
  : mCudaParameters(),
    mCommandLineParameters(),
    mInputFile(), mOutputFile(), mCheckpointFile(), mFileHeader(),
    mFullDimensionSizes(0,0,0), mReducedDimensionSizes(0,0,0),
    mNt(0), mTimeIndex(0), mTimeStepsToCheckpoint(std::numeric_limits<size_t>::max()),
    mDt(0.0f), mDx(0.0f), mDy(0.0f), mDz(0.0f),
    mCRef(0.0f), mC0ScalarFlag(false), mC0Scalar(0.0f),
    mRho0ScalarFlag(false), mRho0Scalar(0.0f),
    mRho0SgxScalar(0.0f),   mRho0SgyScalar(0.0f), mRho0SgzScalar(0.0f),
    mNonUniformGridFlag(0), mAbsorbingFlag(0), mNonLinearFlag(0),
    mAlphaCoeffScalarFlag(false), mAlphaCoeffScalar(0.0f), mAlphaPower(0.0f),
    mAbsorbEtaScalar(0.0f), mAbsorbTauScalar(0.0f),
    mBOnAScalarFlag(false), mBOnAScalar (0.0f),
    mPmlXSize(0), mPmlYSize(0), mPmlZSize(0),
    mPmlXAlpha(0.0f), mPmlYAlpha(0.0f), mPmlZAlpha(0.0f),
    mPressureSourceFlag(0), mInitialPressureSourceFlag(0), mTransducerSourceFlag(0),
    mVelocityXSourceFlag(0), mVelocityYSourceFlag(0), mVelocityZSourceFlag(0),
    mPressureSourceIndexSize(0), mTransducerSourceInputSize(0),mVelocitySourceIndexSize(0),
    mPressureSourceMode(SourceMode::kDirichlet), mPressureSourceMany(0),
    mVelocitySourceMode(SourceMode::kDirichlet), mVelocitySourceMany(0),
    mSensorMaskType(SensorMaskType::kIndex), mSensorMaskIndexSize (0), mSensorMaskCornersSize(0)
{

}// end of Parameters()
//----------------------------------------------------------------------------------------------------------------------

size_t Parameters::getCompressedSteps() const
{
    return mCompressedSteps;
}

DimensionSizes Parameters::getCompressedDimensionSizes() const
{
    return mCompressedDimensionSizes;
}
