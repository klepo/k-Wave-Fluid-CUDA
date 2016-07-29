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
 *              29 July      2016, 16:58 (revised)
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
#include <sstream>
#include <exception>
#include <stdexcept>

#include <Parameters/Parameters.h>
#include <Parameters/CUDAParameters.h>
#include <Utils/MatrixNames.h>
#include <Logger/ErrorMessages.h>
#include <Logger/OutputMessages.h>
#include <Logger/Logger.h>


using std::ios;
using std::string;

//------------------------------------------------------------------------------------------------//
//------------------------------------------ CONSTANTS -------------------------------------------//
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
    TLogger::Log(TLogger::FULL, OUT_FMT_GIT_HASH_LEFT, GetGitHash().c_str());
    TLogger::Log(TLogger::FULL, OUT_FMT_SEPARATOR);
  }
  if (commandLineParameters.IsVersion())
  {
    return;
  }

  TLogger::Log(TLogger::BASIC, OUT_FMT_READING_CONFIGURATION);
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

  TLogger::Log(TLogger::BASIC, OUT_FMT_DONE);
}// end of ParseCommandLine
//--------------------------------------------------------------------------------------------------


/**
 * Select a GPU device for execution.
 */
void TParameters::SelectDevice()
{
  TLogger::Log(TLogger::BASIC,
               OUT_FMT_SELECTED_DEVICE);
  TLogger::Flush(TLogger::BASIC);

  int deviceIdx = commandLineParameters.GetCUDADeviceIdx();
  cudaParameters.SelectDevice(deviceIdx); // throws an exception when wrong

  TLogger::Log(TLogger::BASIC,
               OUT_FMT_DEVICE_ID,
               cudaParameters.GetDeviceIdx());

  TLogger::Log(TLogger::BASIC,
               OUT_FMT_DEVICE_NAME,
               cudaParameters.GetDeviceName().c_str());
}// end of SelectDevice
//--------------------------------------------------------------------------------------------------


/**
 * Print parameters of the simulation, based in the actual level of verbosity.
 */
void TParameters::PrintSimulatoinSetup()
{
  TLogger::Log(TLogger::BASIC,
               OUT_FMT_NUMBER_OF_THREADS,
               GetNumberOfThreads());

  TLogger::Log(TLogger::BASIC,  OUT_FMT_SIMULATION_DETAIL_TITLE);


  const string domainsSizes = TLogger::FormatMessage(OUT_FMT_DOMAIN_SIZE_FORMAT,
                                                     GetFullDimensionSizes().nx,
                                                     GetFullDimensionSizes().ny,
                                                     GetFullDimensionSizes().nz);
  // Print simulation size
  TLogger::Log(TLogger::BASIC, OUT_FMT_DOMAIN_SIZE, domainsSizes.c_str());

  TLogger::Log(TLogger::BASIC, OUT_FMT_SIMULATION_LENGTH, Get_nt());

  // Print all command line parameters
  commandLineParameters.PrintComandlineParamers();

  if (Get_sensor_mask_type() == INDEX)
  {
    TLogger::Log(TLogger::ADVANCED, OUT_FMT_SENSOR_MASK_INDEX);
  }
  if (Get_sensor_mask_type() == CORNERS)
  {
    TLogger::Log(TLogger::ADVANCED, OUT_FMT_SENSOR_MASK_CUBOID);
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
  TDimensionSizes scalarSizes(1,1,1);

  if (!inputFile.IsOpen())
  {
    // Open file -- exceptions handled in main
    inputFile.Open(commandLineParameters.GetInputFileName());
  }

  fileHeader.ReadHeaderFromInputFile(inputFile);

  // check file type
  if (fileHeader.GetFileType() != THDF5_FileHeader::INPUT)
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

  inputFile.ReadScalarValue(rootGroup, Nt_NAME, nt);

  inputFile.ReadScalarValue(rootGroup, dt_NAME, dt);
  inputFile.ReadScalarValue(rootGroup, dx_NAME, dx);
  inputFile.ReadScalarValue(rootGroup, dy_NAME, dy);
  inputFile.ReadScalarValue(rootGroup, dz_NAME, dz);

  inputFile.ReadScalarValue(rootGroup, c_ref_NAME,      c_ref);
  inputFile.ReadScalarValue(rootGroup, pml_x_size_NAME, pml_x_size);
  inputFile.ReadScalarValue(rootGroup, pml_y_size_NAME, pml_y_size);
  inputFile.ReadScalarValue(rootGroup, pml_z_size_NAME, pml_z_size);

  inputFile.ReadScalarValue(rootGroup, pml_x_alpha_NAME, pml_x_alpha);
  inputFile.ReadScalarValue(rootGroup, pml_y_alpha_NAME, pml_y_alpha);
  inputFile.ReadScalarValue(rootGroup, pml_z_alpha_NAME, pml_z_alpha);

  size_t x, y, z;
	inputFile.ReadScalarValue(rootGroup, Nx_NAME, x);
  inputFile.ReadScalarValue(rootGroup, Ny_NAME, y);
  inputFile.ReadScalarValue(rootGroup, Nz_NAME, z);


  fullDimensionSizes.nx = x;
  fullDimensionSizes.ny = y;
  fullDimensionSizes.nz = z;

  reducedDimensionSizes.nx = ((x/2) + 1);
  reducedDimensionSizes.ny = y;
  reducedDimensionSizes.nz = z;

  // if the file is of version 1.0, there must be a sensor mask index (backward compatibility)
  if (fileHeader.GetFileVersion() == THDF5_FileHeader::VERSION_10)
  {
    sensor_mask_ind_size = inputFile.GetDatasetElementCount(rootGroup, sensor_mask_index_NAME);

    //if -u_non_staggered_raw enabled, throw an error - not supported
    if (IsStore_u_non_staggered_raw())
    {
      throw ios::failure(ERR_FMT_U_NON_STAGGERED_NOT_SUPPORTED_FILE_VERSION);
    }
  }// version 1.0

  // This is the current version 1.1
  if (fileHeader.GetFileVersion() == THDF5_FileHeader::VERSION_11)
  {
    // read sensor mask type as a size_t value to enum
    size_t sensorMaskTypeNumericValue = 0;
    inputFile.ReadScalarValue(rootGroup, sensor_mask_type_NAME, sensorMaskTypeNumericValue);

    // convert the long value on
    switch (sensorMaskTypeNumericValue)
    {
      case 0:
      {
        sensor_mask_type = INDEX;
        break;
      }
      case 1:
      {
        sensor_mask_type = CORNERS;
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
      case INDEX:
      {
        sensor_mask_ind_size = inputFile.GetDatasetElementCount(rootGroup, sensor_mask_index_NAME);
        break;
      }
      case CORNERS:
      {
        // mask dimensions are [6, N, 1] - I want to know N
        sensor_mask_corners_size = inputFile.GetDatasetDimensionSizes(rootGroup, sensor_mask_corners_NAME).ny;
        break;
      }
    }// switch
  }// version 1.1

  // flags
  inputFile.ReadScalarValue(rootGroup, ux_source_flag_NAME, ux_source_flag);
  inputFile.ReadScalarValue(rootGroup, uy_source_flag_NAME, uy_source_flag);
  inputFile.ReadScalarValue(rootGroup, uz_source_flag_NAME, uz_source_flag);
  inputFile.ReadScalarValue(rootGroup, transducer_source_flag_NAME, transducer_source_flag);

  inputFile.ReadScalarValue(rootGroup, p_source_flag_NAME, p_source_flag);
  inputFile.ReadScalarValue(rootGroup, p0_source_flag_NAME,p0_source_flag);

  inputFile.ReadScalarValue(rootGroup, nonuniform_grid_flag_NAME, nonuniform_grid_flag);
  inputFile.ReadScalarValue(rootGroup, absorbing_flag_NAME,       absorbing_flag);
  inputFile.ReadScalarValue(rootGroup, nonlinear_flag_NAME,       nonlinear_flag);

  // Vector sizes.
  if (transducer_source_flag == 0)
  {
    transducer_source_input_size = 0;
  }
  else
  {
    transducer_source_input_size =inputFile.GetDatasetElementCount(rootGroup, transducer_source_input_NAME);
  }

  if ((transducer_source_flag > 0) || (ux_source_flag > 0) || (uy_source_flag > 0) || (uz_source_flag > 0))
  {
    u_source_index_size = inputFile.GetDatasetElementCount(rootGroup, u_source_index_NAME);
  }

  // uxyz_source_flags.
  if ((ux_source_flag > 0) || (uy_source_flag > 0) || (uz_source_flag > 0))
  {
    inputFile.ReadScalarValue(rootGroup, u_source_many_NAME, u_source_many);
    inputFile.ReadScalarValue(rootGroup, u_source_mode_NAME, u_source_mode);
  }
  else
  {
    u_source_many = 0;
    u_source_mode = 0;
  }

  // p_source_flag
  if (p_source_flag != 0)
  {
    inputFile.ReadScalarValue(rootGroup, p_source_many_NAME, p_source_many);
    inputFile.ReadScalarValue(rootGroup, p_source_mode_NAME, p_source_mode);

    p_source_index_size = inputFile.GetDatasetElementCount(rootGroup, p_source_index_NAME);
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
    inputFile.ReadScalarValue(rootGroup, alpha_power_NAME, alpha_power);
    if (alpha_power == 1.0f)
    {
      throw std::invalid_argument(ERR_FMT_ILLEGAL_ALPHA_POWER_VALUE);
    }

    alpha_coeff_scalar_flag = inputFile.GetDatasetDimensionSizes(rootGroup, alpha_coeff_NAME) == scalarSizes;

    if (alpha_coeff_scalar_flag)
    {
      inputFile.ReadScalarValue(rootGroup, alpha_coeff_NAME, alpha_coeff_scalar);
    }
  }

  c0_scalar_flag = inputFile.GetDatasetDimensionSizes(rootGroup, c0_NAME) == scalarSizes;
  if (c0_scalar_flag)
  {
    inputFile.ReadScalarValue(rootGroup, c0_NAME, c0_scalar);
  }

  if (nonlinear_flag)
  {
    BonA_scalar_flag = inputFile.GetDatasetDimensionSizes(rootGroup, BonA_NAME) == scalarSizes;
    if (BonA_scalar_flag)
    {
      inputFile.ReadScalarValue(rootGroup, BonA_NAME, BonA_scalar);
    }
  }

  rho0_scalar_flag = inputFile.GetDatasetDimensionSizes(rootGroup, rho0_NAME) == scalarSizes;
  if (rho0_scalar_flag)
  {
    inputFile.ReadScalarValue(rootGroup, rho0_NAME,     rho0_scalar);
    inputFile.ReadScalarValue(rootGroup, rho0_sgx_NAME, rho0_sgx_scalar);
    inputFile.ReadScalarValue(rootGroup, rho0_sgy_NAME, rho0_sgy_scalar);
    inputFile.ReadScalarValue(rootGroup, rho0_sgz_NAME, rho0_sgz_scalar);
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
  outputFile.WriteScalarValue(HDF5RootGroup, Nx_NAME, fullDimensionSizes.nx);
  outputFile.WriteScalarValue(HDF5RootGroup, Ny_NAME, fullDimensionSizes.ny);
  outputFile.WriteScalarValue(HDF5RootGroup, Nz_NAME, fullDimensionSizes.nz);

  outputFile.WriteScalarValue(HDF5RootGroup, Nt_NAME, nt);

  outputFile.WriteScalarValue(HDF5RootGroup, dt_NAME, dt);
  outputFile.WriteScalarValue(HDF5RootGroup, dx_NAME, dx);
  outputFile.WriteScalarValue(HDF5RootGroup, dy_NAME, dy);
  outputFile.WriteScalarValue(HDF5RootGroup, dz_NAME, dz);

  outputFile.WriteScalarValue(HDF5RootGroup, c_ref_NAME, c_ref);

  outputFile.WriteScalarValue(HDF5RootGroup, pml_x_size_NAME, pml_x_size);
  outputFile.WriteScalarValue(HDF5RootGroup, pml_y_size_NAME, pml_y_size);
  outputFile.WriteScalarValue(HDF5RootGroup, pml_z_size_NAME, pml_z_size);

  outputFile.WriteScalarValue(HDF5RootGroup, pml_x_alpha_NAME, pml_x_alpha);
  outputFile.WriteScalarValue(HDF5RootGroup, pml_y_alpha_NAME, pml_y_alpha);
  outputFile.WriteScalarValue(HDF5RootGroup, pml_z_alpha_NAME, pml_z_alpha);

  outputFile.WriteScalarValue(HDF5RootGroup, ux_source_flag_NAME, ux_source_flag);
  outputFile.WriteScalarValue(HDF5RootGroup, uy_source_flag_NAME, uy_source_flag);
  outputFile.WriteScalarValue(HDF5RootGroup, uz_source_flag_NAME, uz_source_flag);
  outputFile.WriteScalarValue(HDF5RootGroup, transducer_source_flag_NAME, transducer_source_flag);

  outputFile.WriteScalarValue(HDF5RootGroup, p_source_flag_NAME,  p_source_flag);
  outputFile.WriteScalarValue(HDF5RootGroup, p0_source_flag_NAME, p0_source_flag);

  outputFile.WriteScalarValue(HDF5RootGroup, nonuniform_grid_flag_NAME, nonuniform_grid_flag);
  outputFile.WriteScalarValue(HDF5RootGroup, absorbing_flag_NAME,       absorbing_flag);
  outputFile.WriteScalarValue(HDF5RootGroup, nonlinear_flag_NAME,       nonlinear_flag);

  // uxyz_source_flags.
  if ((ux_source_flag > 0) || (uy_source_flag > 0) || (uz_source_flag > 0))
  {
    outputFile.WriteScalarValue(HDF5RootGroup, u_source_many_NAME, u_source_many);
      outputFile.WriteScalarValue(HDF5RootGroup, u_source_mode_NAME, u_source_mode);
  }

  // p_source_flag.
  if (p_source_flag != 0)
  {
    outputFile.WriteScalarValue(HDF5RootGroup, p_source_many_NAME, p_source_many);
    outputFile.WriteScalarValue(HDF5RootGroup, p_source_mode_NAME, p_source_mode);

    }

  // absorb flag
  if (absorbing_flag != 0)
  {
    outputFile.WriteScalarValue(HDF5RootGroup, alpha_power_NAME, alpha_power);
  }

  // if copy sensor mask, then copy the mask type
  if (IsCopySensorMask())
  {
    size_t SensorMaskTypeNumericValue = 0;

    switch (sensor_mask_type)
    {
      case INDEX: SensorMaskTypeNumericValue = 0;
        break;
      case CORNERS: SensorMaskTypeNumericValue = 1;
        break;
    }// switch

    outputFile.WriteScalarValue(HDF5RootGroup, sensor_mask_type_NAME, SensorMaskTypeNumericValue);
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
        sensor_mask_ind_size (0), u_source_index_size(0), p_source_index_size(0), transducer_source_input_size(0),
        ux_source_flag(0), uy_source_flag(0), uz_source_flag(0),
        p_source_flag(0), p0_source_flag(0), transducer_source_flag(0),
        u_source_many(0), u_source_mode(0), p_source_mode(0), p_source_many(0),
        nonuniform_grid_flag(0), absorbing_flag(0), nonlinear_flag(0),
        pml_x_size(0), pml_y_size(0), pml_z_size(0),
        alpha_coeff_scalar_flag(false), alpha_coeff_scalar(0.0f),
        c0_scalar_flag(false), c0_scalar(0.0f),
        absorb_eta_scalar(0.0f), absorb_tau_scalar (0.0f),
        BonA_scalar_flag(false), BonA_scalar (0.0f),
        rho0_scalar_flag(false), rho0_scalar(0.0f), rho0_sgx_scalar(0.0f), rho0_sgy_scalar(0.0f), rho0_sgz_scalar(0.0f)
{

}// end of TParameters()
//--------------------------------------------------------------------------------------------------


