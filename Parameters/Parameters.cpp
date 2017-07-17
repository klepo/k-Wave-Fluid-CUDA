/**
 * @file        Parameters.cpp
 *
 * @author      Jiri Jaros   \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing parameters of the simulation.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        09 August    2012, 13:39 (created) \n
 *              17 July      2017, 16:16 (revised)
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

#ifdef _OPENMP
  #include <omp.h>
#endif

#include <iostream>
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

  if ((mNt <= mCommandLineParameters.getSamplingStartTimeIndex()) ||
      (0 > mCommandLineParameters.getSamplingStartTimeIndex()))
  {
    throw std::invalid_argument(Logger::formatMessage(kErrFmtIllegalSamplingStartTimeStep, 1l, mNt));
  }

  Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
}// end of parseCommandLine
//----------------------------------------------------------------------------------------------------------------------


/**
 * Select a GPU device for execution.
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
}// end of printParametersOfTask
//----------------------------------------------------------------------------------------------------------------------


/**
 * Read scalar values from the input HDF5 file.
 */
void Parameters::readScalarsFromInputFile()
{
  DimensionSizes scalarSizes(1,1,1);

  if (!mInputFile.IsOpen())
  {
    // Open file -- exceptions handled in main
    mInputFile.Open(mCommandLineParameters.getInputFileName());
  }

  mFileHeader.ReadHeaderFromInputFile(mInputFile);

  // check file type
  if (mFileHeader.GetFileType() != THDF5_FileHeader::TFileType::INPUT)
  {
    throw ios::failure(Logger::formatMessage(kErrFmtBadInputFileFormat, getInputFileName().c_str()));
  }

  // check version
  if (!mFileHeader.CheckMajorFileVersion())
  {
    throw ios::failure(Logger::formatMessage(kErrFmtBadMajorFileVersion,
                                             getInputFileName().c_str(),
                                             mFileHeader.GetCurrentHDF5_MajorVersion().c_str()));
  }

  if (!mFileHeader.CheckMinorFileVersion())
  {
    throw ios::failure(Logger::formatMessage(kErrFmtBadMinorFileVersion,
                                             getInputFileName().c_str(),
                                             mFileHeader.GetCurrentHDF5_MinorVersion().c_str()));
  }

  const hid_t rootGroup = mInputFile.GetRootGroup();

  mInputFile.ReadScalarValue(rootGroup, kNtName, mNt);

  mInputFile.ReadScalarValue(rootGroup, kDtName, mDt);
  mInputFile.ReadScalarValue(rootGroup, kDxName, mDx);
  mInputFile.ReadScalarValue(rootGroup, kDyName, mDy);
  mInputFile.ReadScalarValue(rootGroup, kDzName, mDz);

  mInputFile.ReadScalarValue(rootGroup, kCRefName,     mCRef);
  mInputFile.ReadScalarValue(rootGroup, kPmlXSizeName, mPmlXSize);
  mInputFile.ReadScalarValue(rootGroup, kPmlYSizeName, mPmlYSize);
  mInputFile.ReadScalarValue(rootGroup, kPmlZSizeName, mPmlZSize);

  mInputFile.ReadScalarValue(rootGroup, kPmlXAlphaName, mPmlXAlpha);
  mInputFile.ReadScalarValue(rootGroup, kPmlYAlphaName, mPmlYAlpha);
  mInputFile.ReadScalarValue(rootGroup, kPmlZAlphaName, mPmlZAlpha);

  size_t x, y, z;
	mInputFile.ReadScalarValue(rootGroup, kNxName, x);
  mInputFile.ReadScalarValue(rootGroup, kNyName, y);
  mInputFile.ReadScalarValue(rootGroup, kNzName, z);


  mFullDimensionSizes.nx = x;
  mFullDimensionSizes.ny = y;
  mFullDimensionSizes.nz = z;

  mReducedDimensionSizes.nx = ((x/2) + 1);
  mReducedDimensionSizes.ny = y;
  mReducedDimensionSizes.nz = z;

  // if the file is of version 1.0, there must be a sensor mask index (backward compatibility)
  if (mFileHeader.GetFileVersion() == THDF5_FileHeader::TFileVersion::VERSION_10)
  {
    mSensorMaskIndexSize = mInputFile.GetDatasetElementCount(rootGroup, kSensorMaskIndexName);

    //if -u_non_staggered_raw enabled, throw an error - not supported
    if (getStoreVelocityNonStaggeredRaw())
    {
      throw ios::failure(kErrFmtNonStaggeredVelocityNotSupportedFileVersion);
    }
  }// version 1.0

  // This is the current version 1.1
  if (mFileHeader.GetFileVersion() == THDF5_FileHeader::TFileVersion::VERSION_11)
  {
    // read sensor mask type as a size_t value to enum
    size_t sensorMaskTypeNumericValue = 0;
    mInputFile.ReadScalarValue(rootGroup, kSensorMaskTypeName, sensorMaskTypeNumericValue);

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
        mSensorMaskIndexSize = mInputFile.GetDatasetElementCount(rootGroup, kSensorMaskIndexName);
        break;
      }
      case SensorMaskType::kCorners:
      {
        // mask dimensions are [6, N, 1] - I want to know N
        mSensorMaskCornersSize = mInputFile.GetDatasetDimensionSizes(rootGroup, kSensorMaskCornersName).ny;
        break;
      }
    }// switch
  }// version 1.1

  // flags
  mInputFile.ReadScalarValue(rootGroup, kVelocityXSourceFlagName,  mVelocityXSourceFlag);
  mInputFile.ReadScalarValue(rootGroup, kVelocityYSourceFlagName,  mVelocityYSourceFlag);
  mInputFile.ReadScalarValue(rootGroup, kVelocityZSourceFlagName,  mVelocityZSourceFlag);
  mInputFile.ReadScalarValue(rootGroup, kTransducerSourceFlagName, mTransducerSourceFlag);

  mInputFile.ReadScalarValue(rootGroup, kPressureSourceFlagName,        mPressureSourceFlag);
  mInputFile.ReadScalarValue(rootGroup, kInitialPressureSourceFlagName, mInitialPressureSourceFlag);

  mInputFile.ReadScalarValue(rootGroup, kNonUniformGridFlagName, mNonUniformGridFlag);
  mInputFile.ReadScalarValue(rootGroup, kAbsorbingFlagName,      mAbsorbingFlag);
  mInputFile.ReadScalarValue(rootGroup, kNonLinearFlagName,      mNonLinearFlag);

  // Vector sizes.
  if (mTransducerSourceFlag == 0)
  {
    mTransducerSourceInputSize = 0;
  }
  else
  {
    mTransducerSourceInputSize =mInputFile.GetDatasetElementCount(rootGroup, kTransducerSourceInputName);
  }

  if ((mTransducerSourceFlag > 0) || (mVelocityXSourceFlag > 0) || (mVelocityYSourceFlag > 0) || (mVelocityZSourceFlag > 0))
  {
    mVelocitySourceIndexSize = mInputFile.GetDatasetElementCount(rootGroup, kVelocitySourceIndexName);
  }

  // uxyz_source_flags.
  if ((mVelocityXSourceFlag > 0) || (mVelocityYSourceFlag > 0) || (mVelocityZSourceFlag > 0))
  {
    mInputFile.ReadScalarValue(rootGroup, kVelocitySourceManyName, mVelocitySourceMany);
    mInputFile.ReadScalarValue(rootGroup, kVelocitySourceModeName, mVelocitySourceMode);
  }
  else
  {
    mVelocitySourceMany = 0;
    mVelocitySourceMode = 0;
  }

  // p_source_flag
  if (mPressureSourceFlag != 0)
  {
    mInputFile.ReadScalarValue(rootGroup, kPressureSourceManyName, mPressureSourceMany);
    mInputFile.ReadScalarValue(rootGroup, kPressureSourceModeName, mPressureSourceMode);

    mPressureSourceIndexSize = mInputFile.GetDatasetElementCount(rootGroup, kPressureSourceIndexName);
  }
  else
  {
    mPressureSourceMode = 0;
    mPressureSourceMany = 0;
    mPressureSourceIndexSize = 0;
  }

  // absorb flag.
  if (mAbsorbingFlag != 0)
  {
    mInputFile.ReadScalarValue(rootGroup, kAlphaPowerName, mAlphaPower);
    if (mAlphaPower == 1.0f)
    {
      throw std::invalid_argument(kErrFmtIllegalAlphaPowerValue);
    }

    mAlphaCoeffScalarFlag = mInputFile.GetDatasetDimensionSizes(rootGroup, kAlphaCoeffName) == scalarSizes;

    if (mAlphaCoeffScalarFlag)
    {
      mInputFile.ReadScalarValue(rootGroup, kAlphaCoeffName, mAlphaCoeffScalar);
    }
  }

  mC0ScalarFlag = mInputFile.GetDatasetDimensionSizes(rootGroup, kC0Name) == scalarSizes;
  if (mC0ScalarFlag)
  {
    mInputFile.ReadScalarValue(rootGroup, kC0Name, mC0Scalar);
  }

  if (mNonLinearFlag)
  {
    mBOnAScalarFlag = mInputFile.GetDatasetDimensionSizes(rootGroup, kBonAName) == scalarSizes;
    if (mBOnAScalarFlag)
    {
      mInputFile.ReadScalarValue(rootGroup, kBonAName, mBOnAScalar);
    }
  }

  mRho0ScalarFlag = mInputFile.GetDatasetDimensionSizes(rootGroup, kRho0Name) == scalarSizes;
  if (mRho0ScalarFlag)
  {
    mInputFile.ReadScalarValue(rootGroup, kRho0Name,     mRho0Scalar);
    mInputFile.ReadScalarValue(rootGroup, kRho0SgxName, mRho0SgxScalar);
    mInputFile.ReadScalarValue(rootGroup, kRho0SgyName, mRho0SgyScalar);
    mInputFile.ReadScalarValue(rootGroup, kRho0SgzName, mRho0SgzScalar);
    }
}// end of readScalarsFromInputFile
//----------------------------------------------------------------------------------------------------------------------

/**
 * Save scalars into the output HDF5 file.
 */
void Parameters::saveScalarsToOutputFile()
{
    const hid_t rootGroup = mOutputFile.GetRootGroup();

  // Write dimension sizes
  mOutputFile.WriteScalarValue(rootGroup, kNxName, mFullDimensionSizes.nx);
  mOutputFile.WriteScalarValue(rootGroup, kNyName, mFullDimensionSizes.ny);
  mOutputFile.WriteScalarValue(rootGroup, kNzName, mFullDimensionSizes.nz);

  mOutputFile.WriteScalarValue(rootGroup, kNtName, mNt);

  mOutputFile.WriteScalarValue(rootGroup, kDtName, mDt);
  mOutputFile.WriteScalarValue(rootGroup, kDxName, mDx);
  mOutputFile.WriteScalarValue(rootGroup, kDyName, mDy);
  mOutputFile.WriteScalarValue(rootGroup, kDzName, mDz);

  mOutputFile.WriteScalarValue(rootGroup, kCRefName, mCRef);

  mOutputFile.WriteScalarValue(rootGroup, kPmlXSizeName, mPmlXSize);
  mOutputFile.WriteScalarValue(rootGroup, kPmlYSizeName, mPmlYSize);
  mOutputFile.WriteScalarValue(rootGroup, kPmlZSizeName, mPmlZSize);

  mOutputFile.WriteScalarValue(rootGroup, kPmlXAlphaName, mPmlXAlpha);
  mOutputFile.WriteScalarValue(rootGroup, kPmlYAlphaName, mPmlYAlpha);
  mOutputFile.WriteScalarValue(rootGroup, kPmlZAlphaName, mPmlZAlpha);

  mOutputFile.WriteScalarValue(rootGroup, kVelocityXSourceFlagName,  mVelocityXSourceFlag);
  mOutputFile.WriteScalarValue(rootGroup, kVelocityYSourceFlagName,  mVelocityYSourceFlag);
  mOutputFile.WriteScalarValue(rootGroup, kVelocityZSourceFlagName,  mVelocityZSourceFlag);
  mOutputFile.WriteScalarValue(rootGroup, kTransducerSourceFlagName, mTransducerSourceFlag);

  mOutputFile.WriteScalarValue(rootGroup, kPressureSourceFlagName,        mPressureSourceFlag);
  mOutputFile.WriteScalarValue(rootGroup, kInitialPressureSourceFlagName, mInitialPressureSourceFlag);

  mOutputFile.WriteScalarValue(rootGroup, kNonUniformGridFlagName, mNonUniformGridFlag);
  mOutputFile.WriteScalarValue(rootGroup, kAbsorbingFlagName,      mAbsorbingFlag);
  mOutputFile.WriteScalarValue(rootGroup, kNonLinearFlagName,      mNonLinearFlag);

  // uxyz_source_flags.
  if ((mVelocityXSourceFlag > 0) || (mVelocityYSourceFlag > 0) || (mVelocityZSourceFlag > 0))
  {
    mOutputFile.WriteScalarValue(rootGroup, kVelocitySourceManyName, mVelocitySourceMany);
    mOutputFile.WriteScalarValue(rootGroup, kVelocitySourceModeName, mVelocitySourceMode);
  }

  // p_source_flag.
  if (mPressureSourceFlag != 0)
  {
    mOutputFile.WriteScalarValue(rootGroup, kPressureSourceManyName, mPressureSourceMany);
    mOutputFile.WriteScalarValue(rootGroup, kPressureSourceModeName, mPressureSourceMode);
  }

  // absorb flag
  if (mAbsorbingFlag != 0)
  {
    mOutputFile.WriteScalarValue(rootGroup, kAlphaPowerName, mAlphaPower);
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

    mOutputFile.WriteScalarValue(rootGroup, kSensorMaskTypeName, sensorMaskTypeNumericValue);
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
Parameters::Parameters() :
  mCudaParameters(),
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
  mAbsorbEtaScalar(0.0f), mAbsorbTauScalar (0.0f),
  mBOnAScalarFlag(false), mBOnAScalar (0.0f),
  mPmlXSize(0), mPmlYSize(0), mPmlZSize(0),
  mPmlXAlpha(0.0f), mPmlYAlpha(0.0f), mPmlZAlpha(0.0f),
  mPressureSourceFlag(0), mInitialPressureSourceFlag(0), mTransducerSourceFlag(0),
  mVelocityXSourceFlag(0), mVelocityYSourceFlag(0), mVelocityZSourceFlag(0),
  mPressureSourceIndexSize(0), mTransducerSourceInputSize(0),mVelocitySourceIndexSize(0),
  mPressureSourceMode(0), mPressureSourceMany(0), mVelocitySourceMany(0), mVelocitySourceMode(0),
  mSensorMaskType(SensorMaskType::kIndex), mSensorMaskIndexSize (0), mSensorMaskCornersSize(0)
{

}// end of Parameters()
//----------------------------------------------------------------------------------------------------------------------


