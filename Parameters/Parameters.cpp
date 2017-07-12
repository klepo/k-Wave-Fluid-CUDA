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
 *              12 July      2017, 11:03 (revised)
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
#include <Parameters/CUDAParameters.h>
#include <Utils/MatrixNames.h>
#include <Logger/Logger.h>


using std::ios;
using std::string;

//------------------------------------------------------------------------------------------------//
//------------------------------------------ Constants -------------------------------------------//
//------------------------------------------------------------------------------------------------//


//------------------------------------------------------------------------------------------------//
//------------------------------------------ VARIABLES -------------------------------------------//
//------------------------------------------------------------------------------------------------//

bool TParameters::parametersInstanceFlag = false;

TParameters* TParameters::parametersSingleInstance = NULL;

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Get instance of singleton class.
 */
TParameters& TParameters::GetInstance()
{
  if(!parametersInstanceFlag)
  {
      parametersSingleInstance = new TParameters();
      parametersInstanceFlag = true;
      return *parametersSingleInstance;
  }
  else
  {
      return *parametersSingleInstance;
  }
}// end of GetInstance()
//--------------------------------------------------------------------------------------------------

/**
 * Destructor.
 */
TParameters::~TParameters()
{
  parametersInstanceFlag = false;
  if (parametersSingleInstance)
  {
    delete parametersSingleInstance;
  }
  parametersSingleInstance = nullptr;
};
//--------------------------------------------------------------------------------------------------

/**
 * Parse command line and read scalar values from the input file to initialise the class and
 * the simulation.
 *
 * @param [in] argc - Number of commandline parameters
 * @param [in] argv - Commandline parameters
 */
void TParameters::Init(int argc, char** argv)
{
  commandLineParameters.ParseCommandLine(argc, argv);

  if (GetGitHash() != "")
  {
    TLogger::Log(TLogger::TLogLevel::FULL, OUT_FMT_GIT_HASH_LEFT, GetGitHash().c_str());
    TLogger::Log(TLogger::TLogLevel::FULL, OUT_FMT_SEPARATOR);
  }
  if (commandLineParameters.IsVersion())
  {
    return;
  }

  TLogger::Log(TLogger::TLogLevel::BASIC, OUT_FMT_READING_CONFIGURATION);
  ReadScalarsFromInputFile(inputFile);

  if (commandLineParameters.IsBenchmarkFlag())
  {
    nt = commandLineParameters.GetBenchmarkTimeStepsCount();
  }

  if ((nt <= commandLineParameters.GetStartTimeIndex()) ||
      (0 > commandLineParameters.GetStartTimeIndex()))
  {
    throw std::invalid_argument(TLogger::FormatMessage(ERR_FMT_ILLEGAL_START_TIME_VALUE,
                                                       1l,
                                                       nt));
  }

  TLogger::Log(TLogger::TLogLevel::BASIC, OUT_FMT_DONE);
}// end of ParseCommandLine
//--------------------------------------------------------------------------------------------------


/**
 * Select a GPU device for execution.
 */
void TParameters::SelectDevice()
{
  TLogger::Log(TLogger::TLogLevel::BASIC,
               OUT_FMT_SELECTED_DEVICE);
  TLogger::Flush(TLogger::TLogLevel::BASIC);

  int deviceIdx = commandLineParameters.GetCUDADeviceIdx();
  cudaParameters.SelectDevice(deviceIdx); // throws an exception when wrong

  TLogger::Log(TLogger::TLogLevel::BASIC,
               OUT_FMT_DEVICE_ID,
               cudaParameters.GetDeviceIdx());

  TLogger::Log(TLogger::TLogLevel::BASIC,
               OUT_FMT_DEVICE_NAME,
               cudaParameters.GetDeviceName().c_str());
}// end of SelectDevice
//--------------------------------------------------------------------------------------------------


/**
 * Print parameters of the simulation, based in the actual level of verbosity.
 */
void TParameters::PrintSimulatoinSetup()
{
  TLogger::Log(TLogger::TLogLevel::BASIC,
               OUT_FMT_NUMBER_OF_THREADS,
               GetNumberOfThreads());

  TLogger::Log(TLogger::TLogLevel::BASIC,  OUT_FMT_SIMULATION_DETAIL_TITLE);


  const string domainsSizes = TLogger::FormatMessage(OUT_FMT_DOMAIN_SIZE_FORMAT,
                                                     GetFullDimensionSizes().nx,
                                                     GetFullDimensionSizes().ny,
                                                     GetFullDimensionSizes().nz);
  // Print simulation size
  TLogger::Log(TLogger::TLogLevel::BASIC, OUT_FMT_DOMAIN_SIZE, domainsSizes.c_str());

  TLogger::Log(TLogger::TLogLevel::BASIC, OUT_FMT_SIMULATION_LENGTH, Get_nt());

  // Print all command line parameters
  commandLineParameters.PrintComandlineParamers();

  if (Get_sensor_mask_type() == TSensorMaskType::INDEX)
  {
    TLogger::Log(TLogger::TLogLevel::ADVANCED, OUT_FMT_SENSOR_MASK_INDEX);
  }
  if (Get_sensor_mask_type() == TSensorMaskType::CORNERS)
  {
    TLogger::Log(TLogger::TLogLevel::ADVANCED, OUT_FMT_SENSOR_MASK_CUBOID);
  }
}// end of PrintParametersOfTask
//--------------------------------------------------------------------------------------------------


/**
 * Read scalar values from the input HDF5 file.
 *
 * @param [in] inputFile - Handle to an opened input file.
 *
 * @throw ios:failure if the file cannot be open or is of a wrong type or version.
 */
void TParameters::ReadScalarsFromInputFile(THDF5_File& inputFile)
{
  DimensionSizes scalarSizes(1,1,1);

  if (!inputFile.IsOpen())
  {
    // Open file -- exceptions handled in main
    inputFile.Open(commandLineParameters.GetInputFileName());
  }

  fileHeader.ReadHeaderFromInputFile(inputFile);

  // check file type
  if (fileHeader.GetFileType() != THDF5_FileHeader::TFileType::INPUT)
  {
    throw ios::failure(TLogger::FormatMessage(ERR_FMT_BAD_INPUT_FILE_FORMAT,
                                              GetInputFileName().c_str()));
  }

  // check version
  if (!fileHeader.CheckMajorFileVersion())
  {
    throw ios::failure(TLogger::FormatMessage(ERR_FMT_BAD_MAJOR_File_Version,
                                              GetInputFileName().c_str(),
                                              fileHeader.GetCurrentHDF5_MajorVersion().c_str()));
  }

  if (!fileHeader.CheckMinorFileVersion())
  {
    throw ios::failure(TLogger::FormatMessage(ERR_FMT_BAD_MINOR_FILE_VERSION,
                                              GetInputFileName().c_str(),
                                              fileHeader.GetCurrentHDF5_MinorVersion().c_str()));
  }

  const hid_t rootGroup = inputFile.GetRootGroup();

  inputFile.ReadScalarValue(rootGroup, kNtName, nt);

  inputFile.ReadScalarValue(rootGroup, kDtName, dt);
  inputFile.ReadScalarValue(rootGroup, kDxName, dx);
  inputFile.ReadScalarValue(rootGroup, kDyName, dy);
  inputFile.ReadScalarValue(rootGroup, kDzName, dz);

  inputFile.ReadScalarValue(rootGroup, kCRefName,      c_ref);
  inputFile.ReadScalarValue(rootGroup, kPmlXSizeName, pml_x_size);
  inputFile.ReadScalarValue(rootGroup, kPmlYSizeName, pml_y_size);
  inputFile.ReadScalarValue(rootGroup, kPmlZSizeName, pml_z_size);

  inputFile.ReadScalarValue(rootGroup, kPmlXAlphaName, pml_x_alpha);
  inputFile.ReadScalarValue(rootGroup, kPmlYAlphaName, pml_y_alpha);
  inputFile.ReadScalarValue(rootGroup, kPmlZAlphaName, pml_z_alpha);

  size_t x, y, z;
	inputFile.ReadScalarValue(rootGroup, kNxName, x);
  inputFile.ReadScalarValue(rootGroup, kNyName, y);
  inputFile.ReadScalarValue(rootGroup, kNzName, z);


  fullDimensionSizes.nx = x;
  fullDimensionSizes.ny = y;
  fullDimensionSizes.nz = z;

  reducedDimensionSizes.nx = ((x/2) + 1);
  reducedDimensionSizes.ny = y;
  reducedDimensionSizes.nz = z;

  // if the file is of version 1.0, there must be a sensor mask index (backward compatibility)
  if (fileHeader.GetFileVersion() == THDF5_FileHeader::TFileVersion::VERSION_10)
  {
    sensor_mask_ind_size = inputFile.GetDatasetElementCount(rootGroup, kSensorMaskIndexName);

    //if -u_non_staggered_raw enabled, throw an error - not supported
    if (IsStore_u_non_staggered_raw())
    {
      throw ios::failure(ERR_FMT_U_NON_STAGGERED_NOT_SUPPORTED_FILE_VERSION);
    }
  }// version 1.0

  // This is the current version 1.1
  if (fileHeader.GetFileVersion() == THDF5_FileHeader::TFileVersion::VERSION_11)
  {
    // read sensor mask type as a size_t value to enum
    size_t sensorMaskTypeNumericValue = 0;
    inputFile.ReadScalarValue(rootGroup, kSensorMaskTypeName, sensorMaskTypeNumericValue);

    // convert the long value on
    switch (sensorMaskTypeNumericValue)
    {
      case 0:
      {
        sensor_mask_type = TSensorMaskType::INDEX;
        break;
      }
      case 1:
      {
        sensor_mask_type = TSensorMaskType::CORNERS;
        break;
      }
      default:
      {
        throw ios::failure(ERR_FMT_BAD_SENSOR_MASK_TYPE);
        break;
      }
    }//case

    // read the input mask size
    switch (sensor_mask_type)
    {
      case TSensorMaskType::INDEX:
      {
        sensor_mask_ind_size = inputFile.GetDatasetElementCount(rootGroup, kSensorMaskIndexName);
        break;
      }
      case TSensorMaskType::CORNERS:
      {
        // mask dimensions are [6, N, 1] - I want to know N
        sensor_mask_corners_size = inputFile.GetDatasetDimensionSizes(rootGroup, kSensorMaskCornersName).ny;
        break;
      }
    }// switch
  }// version 1.1

  // flags
  inputFile.ReadScalarValue(rootGroup, kUxSourceFlagName, ux_source_flag);
  inputFile.ReadScalarValue(rootGroup, kUySourceFlagName, uy_source_flag);
  inputFile.ReadScalarValue(rootGroup, kUzSourceFlagName, uz_source_flag);
  inputFile.ReadScalarValue(rootGroup, kTransducerSourceFlagName, transducer_source_flag);

  inputFile.ReadScalarValue(rootGroup, kPressureSourceFlagName, p_source_flag);
  inputFile.ReadScalarValue(rootGroup, kP0SourceFlagName,p0_source_flag);

  inputFile.ReadScalarValue(rootGroup, kNonUniformGridFlagName, nonuniform_grid_flag);
  inputFile.ReadScalarValue(rootGroup, kAbsorbingFlagName,       absorbing_flag);
  inputFile.ReadScalarValue(rootGroup, kNonLinearFlagName,       nonlinear_flag);

  // Vector sizes.
  if (transducer_source_flag == 0)
  {
    transducer_source_input_size = 0;
  }
  else
  {
    transducer_source_input_size =inputFile.GetDatasetElementCount(rootGroup, kTransducerSourceInputName);
  }

  if ((transducer_source_flag > 0) || (ux_source_flag > 0) || (uy_source_flag > 0) || (uz_source_flag > 0))
  {
    u_source_index_size = inputFile.GetDatasetElementCount(rootGroup, kVelocitySourceIndexName);
  }

  // uxyz_source_flags.
  if ((ux_source_flag > 0) || (uy_source_flag > 0) || (uz_source_flag > 0))
  {
    inputFile.ReadScalarValue(rootGroup, kVelocitySourceManyName, u_source_many);
    inputFile.ReadScalarValue(rootGroup, kVelocitySourceModeName, u_source_mode);
  }
  else
  {
    u_source_many = 0;
    u_source_mode = 0;
  }

  // p_source_flag
  if (p_source_flag != 0)
  {
    inputFile.ReadScalarValue(rootGroup, kPressureSourceManyName, p_source_many);
    inputFile.ReadScalarValue(rootGroup, kPressureSourceModeName, p_source_mode);

    p_source_index_size = inputFile.GetDatasetElementCount(rootGroup, kPressureSourceIndexName);
  }
  else
  {
    p_source_mode = 0;
    p_source_many = 0;
    p_source_index_size = 0;
  }

  // absorb flag.
  if (absorbing_flag != 0)
  {
    inputFile.ReadScalarValue(rootGroup, kAlphaPowerName, alpha_power);
    if (alpha_power == 1.0f)
    {
      throw std::invalid_argument(ERR_FMT_ILLEGAL_ALPHA_POWER_VALUE);
    }

    alpha_coeff_scalar_flag = inputFile.GetDatasetDimensionSizes(rootGroup, kAlphaCoeffName) == scalarSizes;

    if (alpha_coeff_scalar_flag)
    {
      inputFile.ReadScalarValue(rootGroup, kAlphaCoeffName, alpha_coeff_scalar);
    }
  }

  c0_scalar_flag = inputFile.GetDatasetDimensionSizes(rootGroup, kC0Name) == scalarSizes;
  if (c0_scalar_flag)
  {
    inputFile.ReadScalarValue(rootGroup, kC0Name, c0_scalar);
  }

  if (nonlinear_flag)
  {
    BonA_scalar_flag = inputFile.GetDatasetDimensionSizes(rootGroup, kBonAName) == scalarSizes;
    if (BonA_scalar_flag)
    {
      inputFile.ReadScalarValue(rootGroup, kBonAName, BonA_scalar);
    }
  }

  rho0_scalar_flag = inputFile.GetDatasetDimensionSizes(rootGroup, kRho0Name) == scalarSizes;
  if (rho0_scalar_flag)
  {
    inputFile.ReadScalarValue(rootGroup, kRho0Name,     rho0_scalar);
    inputFile.ReadScalarValue(rootGroup, kRho0SgxName, rho0_sgx_scalar);
    inputFile.ReadScalarValue(rootGroup, kRho0SgyName, rho0_sgy_scalar);
    inputFile.ReadScalarValue(rootGroup, kRho0SgzName, rho0_sgz_scalar);
    }
}// end of ReadScalarsFromInputFile
//--------------------------------------------------------------------------------------------------

/**
 * Save scalars into the output HDF5 file.
 *
 * @param [in] outputFile - Handle to an opened output file where to store
 */
void TParameters::SaveScalarsToFile(THDF5_File& outputFile)
{
  const hid_t HDF5RootGroup = outputFile.GetRootGroup();

  // Write dimension sizes
  outputFile.WriteScalarValue(HDF5RootGroup, kNxName, fullDimensionSizes.nx);
  outputFile.WriteScalarValue(HDF5RootGroup, kNyName, fullDimensionSizes.ny);
  outputFile.WriteScalarValue(HDF5RootGroup, kNzName, fullDimensionSizes.nz);

  outputFile.WriteScalarValue(HDF5RootGroup, kNtName, nt);

  outputFile.WriteScalarValue(HDF5RootGroup, kDtName, dt);
  outputFile.WriteScalarValue(HDF5RootGroup, kDxName, dx);
  outputFile.WriteScalarValue(HDF5RootGroup, kDyName, dy);
  outputFile.WriteScalarValue(HDF5RootGroup, kDzName, dz);

  outputFile.WriteScalarValue(HDF5RootGroup, kCRefName, c_ref);

  outputFile.WriteScalarValue(HDF5RootGroup, kPmlXSizeName, pml_x_size);
  outputFile.WriteScalarValue(HDF5RootGroup, kPmlYSizeName, pml_y_size);
  outputFile.WriteScalarValue(HDF5RootGroup, kPmlZSizeName, pml_z_size);

  outputFile.WriteScalarValue(HDF5RootGroup, kPmlXAlphaName, pml_x_alpha);
  outputFile.WriteScalarValue(HDF5RootGroup, kPmlYAlphaName, pml_y_alpha);
  outputFile.WriteScalarValue(HDF5RootGroup, kPmlZAlphaName, pml_z_alpha);

  outputFile.WriteScalarValue(HDF5RootGroup, kUxSourceFlagName, ux_source_flag);
  outputFile.WriteScalarValue(HDF5RootGroup, kUySourceFlagName, uy_source_flag);
  outputFile.WriteScalarValue(HDF5RootGroup, kUzSourceFlagName, uz_source_flag);
  outputFile.WriteScalarValue(HDF5RootGroup, kTransducerSourceFlagName, transducer_source_flag);

  outputFile.WriteScalarValue(HDF5RootGroup, kPressureSourceFlagName,  p_source_flag);
  outputFile.WriteScalarValue(HDF5RootGroup, kP0SourceFlagName, p0_source_flag);

  outputFile.WriteScalarValue(HDF5RootGroup, kNonUniformGridFlagName, nonuniform_grid_flag);
  outputFile.WriteScalarValue(HDF5RootGroup, kAbsorbingFlagName,       absorbing_flag);
  outputFile.WriteScalarValue(HDF5RootGroup, kNonLinearFlagName,       nonlinear_flag);

  // uxyz_source_flags.
  if ((ux_source_flag > 0) || (uy_source_flag > 0) || (uz_source_flag > 0))
  {
    outputFile.WriteScalarValue(HDF5RootGroup, kVelocitySourceManyName, u_source_many);
      outputFile.WriteScalarValue(HDF5RootGroup, kVelocitySourceModeName, u_source_mode);
  }

  // p_source_flag.
  if (p_source_flag != 0)
  {
    outputFile.WriteScalarValue(HDF5RootGroup, kPressureSourceManyName, p_source_many);
    outputFile.WriteScalarValue(HDF5RootGroup, kPressureSourceModeName, p_source_mode);

    }

  // absorb flag
  if (absorbing_flag != 0)
  {
    outputFile.WriteScalarValue(HDF5RootGroup, kAlphaPowerName, alpha_power);
  }

  // if copy sensor mask, then copy the mask type
  if (IsCopySensorMask())
  {
    size_t SensorMaskTypeNumericValue = 0;

    switch (sensor_mask_type)
    {
      case TSensorMaskType::INDEX: SensorMaskTypeNumericValue = 0;
        break;
      case TSensorMaskType::CORNERS: SensorMaskTypeNumericValue = 1;
        break;
    }// switch

    outputFile.WriteScalarValue(HDF5RootGroup, kSensorMaskTypeName, SensorMaskTypeNumericValue);
  }
}// end of SaveScalarsToFile
//--------------------------------------------------------------------------------------------------

/**
 * Get GitHash of the code
 * @return githash
 */
string TParameters::GetGitHash() const
{
#if (defined (__KWAVE_GIT_HASH__))
  return string(__KWAVE_GIT_HASH__);
#else
  return "";
#endif
}// end of GetGitHash
//--------------------------------------------------------------------------------------------------



//------------------------------------------------------------------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//


/**
 * Constructor.
 */
TParameters::TParameters() :
        cudaParameters(),
        commandLineParameters(),
        inputFile(), outputFile(), checkpointFile(), fileHeader(),
        nt(0), t_index(0), dt(0.0f),
        dx(0.0f), dy(0.0f), dz(0.0f),
        c_ref(0.0f), alpha_power(0.0f),
        fullDimensionSizes(0,0,0), reducedDimensionSizes(0,0,0),
        sensor_mask_ind_size (0), u_source_index_size(0), p_source_index_size(0),
        transducer_source_input_size(0),
        ux_source_flag(0), uy_source_flag(0), uz_source_flag(0),
        p_source_flag(0), p0_source_flag(0), transducer_source_flag(0),
        u_source_many(0), u_source_mode(0), p_source_mode(0), p_source_many(0),
        nonuniform_grid_flag(0), absorbing_flag(0), nonlinear_flag(0),
        pml_x_size(0), pml_y_size(0), pml_z_size(0),
        alpha_coeff_scalar_flag(false), alpha_coeff_scalar(0.0f),
        c0_scalar_flag(false), c0_scalar(0.0f),
        absorb_eta_scalar(0.0f), absorb_tau_scalar (0.0f),
        BonA_scalar_flag(false), BonA_scalar (0.0f),
        rho0_scalar_flag(false), rho0_scalar(0.0f),
        rho0_sgx_scalar(0.0f), rho0_sgy_scalar(0.0f), rho0_sgz_scalar(0.0f)
{

}// end of TParameters()
//--------------------------------------------------------------------------------------------------


