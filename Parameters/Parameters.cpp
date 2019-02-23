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
 * @version   kspaceFirstOrder3D 3.6
 *
 * @date      09 August    2012, 13:39 (created) \n
 *            22 February  2019, 22:04 (revised)
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


#include <string>
#include <exception>
#include <stdexcept>

#include <Parameters/Parameters.h>
#include <Parameters/CudaParameters.h>
#include <Utils/MatrixNames.h>
#include <Logger/Logger.h>


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

  Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
}// end of parseCommandLine
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
 * Print parameters of the simulation based in the actual level of verbosity.
 */
void Parameters::printSimulatoinSetup()
{
  Logger::log(Logger::LogLevel::kBasic, kOutFmtNumberOfThreads, getNumberOfThreads());

  Logger::log(Logger::LogLevel::kBasic, kOutFmtSimulationDetailsTitle);


  const string domainsSizes = Logger::formatMessage(kOutFmtDomainSizeFormat,
                                                    getFullDimensionSizes().nx,
                                                    getFullDimensionSizes().ny,
                                                    getFullDimensionSizes().nz);
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
  DimensionSizes scalarSizes(1,1,1);

  if (!mInputFile.isOpen())
  {
    // Open file -- exceptions handled in main
    mInputFile.open(mCommandLineParameters.getInputFileName());
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

  mInputFile.readScalarValue(rootGroup, kNtName, mNt);

  mInputFile.readScalarValue(rootGroup, kDtName, mDt);
  mInputFile.readScalarValue(rootGroup, kDxName, mDx);
  mInputFile.readScalarValue(rootGroup, kDyName, mDy);
  mInputFile.readScalarValue(rootGroup, kDzName, mDz);

  mInputFile.readScalarValue(rootGroup, kCRefName,     mCRef);
  mInputFile.readScalarValue(rootGroup, kPmlXSizeName, mPmlXSize);
  mInputFile.readScalarValue(rootGroup, kPmlYSizeName, mPmlYSize);
  mInputFile.readScalarValue(rootGroup, kPmlZSizeName, mPmlZSize);

  mInputFile.readScalarValue(rootGroup, kPmlXAlphaName, mPmlXAlpha);
  mInputFile.readScalarValue(rootGroup, kPmlYAlphaName, mPmlYAlpha);
  mInputFile.readScalarValue(rootGroup, kPmlZAlphaName, mPmlZAlpha);

  size_t x, y, z;
	mInputFile.readScalarValue(rootGroup, kNxName, x);
  mInputFile.readScalarValue(rootGroup, kNyName, y);
  mInputFile.readScalarValue(rootGroup, kNzName, z);


  mFullDimensionSizes.nx = x;
  mFullDimensionSizes.ny = y;
  mFullDimensionSizes.nz = z;

  mReducedDimensionSizes.nx = ((x / 2) + 1);
  mReducedDimensionSizes.ny = y;
  mReducedDimensionSizes.nz = z;

  // if the file is of version 1.0, there must be a sensor mask index (backward compatibility)
  if (mFileHeader.getFileVersion() == Hdf5FileHeader::FileVersion::kVersion10)
  {
    mSensorMaskIndexSize = mInputFile.getDatasetSize(rootGroup, kSensorMaskIndexName);

    //if -u_non_staggered_raw enabled, throw an error - not supported
    if (getStoreVelocityNonStaggeredRawFlag())
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
  mInputFile.readScalarValue(rootGroup, kVelocityXSourceFlagName,  mVelocityXSourceFlag);
  mInputFile.readScalarValue(rootGroup, kVelocityYSourceFlagName,  mVelocityYSourceFlag);
  mInputFile.readScalarValue(rootGroup, kVelocityZSourceFlagName,  mVelocityZSourceFlag);
  mInputFile.readScalarValue(rootGroup, kTransducerSourceFlagName, mTransducerSourceFlag);

  mInputFile.readScalarValue(rootGroup, kPressureSourceFlagName,        mPressureSourceFlag);
  mInputFile.readScalarValue(rootGroup, kInitialPressureSourceFlagName, mInitialPressureSourceFlag);

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

  if ((mTransducerSourceFlag > 0) || (mVelocityXSourceFlag > 0) ||
      (mVelocityYSourceFlag > 0)  || (mVelocityZSourceFlag > 0))
  {
    mVelocitySourceIndexSize = mInputFile.getDatasetSize(rootGroup, kVelocitySourceIndexName);
  }

  // uxyz_source_flags.
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
    mVelocitySourceMode = SourceMode::kDirichlet;;
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
    mInputFile.readScalarValue(rootGroup, kRho0Name,    mRho0Scalar);
    mInputFile.readScalarValue(rootGroup, kRho0SgxName, mRho0SgxScalar);
    mInputFile.readScalarValue(rootGroup, kRho0SgyName, mRho0SgyScalar);
    mInputFile.readScalarValue(rootGroup, kRho0SgzName, mRho0SgzScalar);
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
  mOutputFile.writeScalarValue(rootGroup, kDzName, mDz);

  mOutputFile.writeScalarValue(rootGroup, kCRefName, mCRef);

  mOutputFile.writeScalarValue(rootGroup, kPmlXSizeName, mPmlXSize);
  mOutputFile.writeScalarValue(rootGroup, kPmlYSizeName, mPmlYSize);
  mOutputFile.writeScalarValue(rootGroup, kPmlZSizeName, mPmlZSize);

  mOutputFile.writeScalarValue(rootGroup, kPmlXAlphaName, mPmlXAlpha);
  mOutputFile.writeScalarValue(rootGroup, kPmlYAlphaName, mPmlYAlpha);
  mOutputFile.writeScalarValue(rootGroup, kPmlZAlphaName, mPmlZAlpha);

  mOutputFile.writeScalarValue(rootGroup, kVelocityXSourceFlagName,  mVelocityXSourceFlag);
  mOutputFile.writeScalarValue(rootGroup, kVelocityYSourceFlagName,  mVelocityYSourceFlag);
  mOutputFile.writeScalarValue(rootGroup, kVelocityZSourceFlagName,  mVelocityZSourceFlag);
  mOutputFile.writeScalarValue(rootGroup, kTransducerSourceFlagName, mTransducerSourceFlag);

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
    mNt(0), mTimeIndex(0),
    mDt(0.0f), mDx(0.0f), mDy(0.0f), mDz(0.0f),
    mCRef(0.0f), mC0ScalarFlag(false),   mC0Scalar(0.0f),
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
