/**
 * @file        Parameters.cpp
 * @author      Jiri Jaros   \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing parameters of the simulation.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        09 August    2012, 13:39 (created) \n
 *              17 December  2014, 20:37 (revised)
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
#include <Utils/ErrorMessages.h>


using namespace std;

//----------------------------------------------------------------------------//
//                              Constants                                     //
//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//
//                              Definitions                                   //
//----------------------------------------------------------------------------//

bool TParameters::ParametersInstanceFlag = false;

TParameters* TParameters::ParametersSingleInstance = NULL;

//--------------------------------------------------------------------------//
//                            Implementation                                //
//                            public methods                                //
//--------------------------------------------------------------------------//

/**
 * Get instance of singleton class.
 */
TParameters* TParameters::GetInstance()
{
  if(!ParametersInstanceFlag)
  {
      ParametersSingleInstance = new TParameters();
      ParametersInstanceFlag = true;
      return ParametersSingleInstance;
  }
  else
  {
      return ParametersSingleInstance;
  }
}// end of GetInstance()
//----------------------------------------------------------------------------

/**
 * Parse command line.
 * @param [in] argc
 * @param [in] argv
 */
void TParameters::ParseCommandLine(int argc, char** argv)
{

  CommandLinesParameters.ParseCommandLine(argc, argv);

  // this must be here to read the GPU parameters and report a potential error
  int DeviceIdx = CommandLinesParameters.GetGPUDeviceIdx();
  CUDAParameters.SelectDevice(DeviceIdx); // throws an exception when wrong

  if (CommandLinesParameters.IsVersion())
  {
    return;
  }

  ReadScalarsFromHDF5InputFile(HDF5_InputFile);

  if (CommandLinesParameters.IsBenchmarkFlag())
  {
    Nt = CommandLinesParameters.GetBenchmarkTimeStepsCount();
  }

  if ((Nt <= CommandLinesParameters.GetStartTimeIndex()) ||
      (0 > CommandLinesParameters.GetStartTimeIndex()))
  {
    fprintf(stderr,
            Parameters_ERR_FMT_Illegal_StartTime_value,
            static_cast<size_t> (1),
            Nt);
    CommandLinesParameters.PrintUsageAndExit();
  }

}// end of ParseCommandLine
//----------------------------------------------------------------------------

/**
 * Read scalar values from the input HDF5 file.
 *
 * @param [in] HDF5_InputFile - Handle to an opened input file.
 * @throw ios:failure if the file cannot be open or is of a wrong type or version.
 */
void TParameters::ReadScalarsFromHDF5InputFile(THDF5_File & HDF5_InputFile)
{
  TDimensionSizes ScalarSizes(1,1,1);

  if (!HDF5_InputFile.IsOpened())
  {
    // Open file
    try
    {
      HDF5_InputFile.Open(CommandLinesParameters.GetInputFileName().c_str());
    }
    catch (ios::failure e)
    {
      fprintf(stderr, "%s", e.what());
      PrintUsageAndExit();
    }
  }

  HDF5_FileHeader.ReadHeaderFromInputFile(HDF5_InputFile);

  // check file type
  if (HDF5_FileHeader.GetFileType() != THDF5_FileHeader::hdf5_ft_input)
  {
    char ErrorMessage[256] = "";
    sprintf(ErrorMessage,
            Parameters_ERR_FMT_IncorrectInputFileFormat,
            GetInputFileName().c_str());
    throw ios::failure(ErrorMessage);
  }

  // check version
  if (!HDF5_FileHeader.CheckMajorFileVersion())
  {
    char ErrorMessage[256] = "";
    sprintf(ErrorMessage,
            Parameters_ERR_FMT_IncorrectMajorHDF5FileVersion,
            GetInputFileName().c_str(),
            HDF5_FileHeader.GetCurrentHDF5_MajorVersion().c_str());
    throw ios::failure(ErrorMessage);
  }

  if (!HDF5_FileHeader.CheckMinorFileVersion())
  {
    char ErrorMessage[256] = "";
    sprintf(ErrorMessage,
            Parameters_ERR_FMT_IncorrectMinorHDF5FileVersion,
            GetInputFileName().c_str(),
            HDF5_FileHeader.GetCurrentHDF5_MinorVersion().c_str());
    throw ios::failure(ErrorMessage);
  }

  const hid_t HDF5RootGroup = HDF5_InputFile.GetRootGroup();

  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, Nt_Name, Nt);

  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, dt_Name, dt);
  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, dx_Name, dx);
  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, dy_Name, dy);
  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, dz_Name, dz);

  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, c_ref_Name,      c_ref);
  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, pml_x_size_Name, pml_x_size);
  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, pml_y_size_Name, pml_y_size);
  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, pml_z_size_Name, pml_z_size);

  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, pml_x_alpha_Name, pml_x_alpha);
  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, pml_y_alpha_Name, pml_y_alpha);
  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, pml_z_alpha_Name, pml_z_alpha);

  size_t X, Y, Z;
	HDF5_InputFile.ReadScalarValue(HDF5RootGroup, Nx_Name, X);
  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, Ny_Name, Y);
  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, Nz_Name, Z);


  FullDimensionSizes.X = X;
  FullDimensionSizes.Y = Y;
  FullDimensionSizes.Z = Z;

  ReducedDimensionSizes.X = ((X/2) + 1);
  ReducedDimensionSizes.Y = Y;
  ReducedDimensionSizes.Z = Z;

  // if the file is of version 1.0, there must be a sensor mask index (backward compatibility)
  if (HDF5_FileHeader.GetFileVersion() == THDF5_FileHeader::hdf5_fv_10)
  {
    sensor_mask_ind_size = HDF5_InputFile.GetDatasetElementCount(HDF5RootGroup, sensor_mask_index_Name);

    //if -u_non_staggered_raw enabled, throw an error - not supported
    if (IsStore_u_non_staggered_raw())
    {
      throw ios::failure(Parameters_ERR_FMT_UNonStaggeredNotSupportedForFile10);
    }
  }// version 1.0

  // This is the current version 1.1
  if (HDF5_FileHeader.GetFileVersion() == THDF5_FileHeader::hdf5_fv_11)
  {
    // read sensor mask type as a size_t value to enum
    size_t SensorMaskTypeNumericValue = 0;
    HDF5_InputFile.ReadScalarValue(HDF5RootGroup, sensor_mask_type_Name, SensorMaskTypeNumericValue);

    // convert the long value on
    switch (SensorMaskTypeNumericValue)
    {
      case 0:
      {
        sensor_mask_type = smt_index;
        break;
      }
      case 1:
      {
        sensor_mask_type = smt_corners;
        break;
      }
      default:
      {
        throw ios::failure(Parameters_ERR_FMT_WrongSensorMaskType);
        break;
      }
    }//case

    // read the input mask size
    switch (sensor_mask_type)
    {
      case smt_index:
      {
        sensor_mask_ind_size = HDF5_InputFile.GetDatasetElementCount(HDF5RootGroup, sensor_mask_index_Name);
        break;
      }
      case smt_corners:
      {
        // mask dimensions are [6, N, 1] - I want to know N
        sensor_mask_corners_size = HDF5_InputFile.GetDatasetDimensionSizes(HDF5RootGroup, sensor_mask_corners_Name).Y;
        break;
      }
    }// switch
  }// version 1.1

  // flags
  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, ux_source_flag_Name, ux_source_flag);
  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, uy_source_flag_Name, uy_source_flag);
  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, uz_source_flag_Name, uz_source_flag);
  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, transducer_source_flag_Name, transducer_source_flag);

  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, p_source_flag_Name, p_source_flag);
  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, p0_source_flag_Name,p0_source_flag);

  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, nonuniform_grid_flag_Name, nonuniform_grid_flag);
  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, absorbing_flag_Name,       absorbing_flag);
  HDF5_InputFile.ReadScalarValue(HDF5RootGroup, nonlinear_flag_Name,       nonlinear_flag);

  // Vector sizes.
  if (transducer_source_flag == 0)
  {
    transducer_source_input_size = 0;
  }
  else
  {
    transducer_source_input_size =HDF5_InputFile.GetDatasetElementCount(HDF5RootGroup, transducer_source_input_Name);
  }

  if ((transducer_source_flag > 0) || (ux_source_flag > 0) || (uy_source_flag > 0) || (uz_source_flag > 0))
  {
    u_source_index_size = HDF5_InputFile.GetDatasetElementCount(HDF5RootGroup, u_source_index_Name);
  }

  // uxyz_source_flags.
  if ((ux_source_flag > 0) || (uy_source_flag > 0) || (uz_source_flag > 0))
  {
    HDF5_InputFile.ReadScalarValue(HDF5RootGroup, u_source_many_Name, u_source_many);
    HDF5_InputFile.ReadScalarValue(HDF5RootGroup, u_source_mode_Name, u_source_mode);
  }
  else
  {
    u_source_many = 0;
    u_source_mode = 0;
  }

  // p_source_flag
  if (p_source_flag != 0)
  {
    HDF5_InputFile.ReadScalarValue(HDF5RootGroup, p_source_many_Name, p_source_many);
    HDF5_InputFile.ReadScalarValue(HDF5RootGroup, p_source_mode_Name, p_source_mode);

    p_source_index_size = HDF5_InputFile.GetDatasetElementCount(HDF5RootGroup, p_source_index_Name);
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
    HDF5_InputFile.ReadScalarValue(HDF5RootGroup, alpha_power_Name, alpha_power);
    if (alpha_power == 1.0f)
    {
      fprintf(stderr, "%s", Parameters_ERR_FMT_Illegal_alpha_power_value);
      PrintUsageAndExit();
    }

    alpha_coeff_scalar_flag = HDF5_InputFile.GetDatasetDimensionSizes(HDF5RootGroup, alpha_coeff_Name) == ScalarSizes;

    if (alpha_coeff_scalar_flag)
    {
      HDF5_InputFile.ReadScalarValue(HDF5RootGroup, alpha_coeff_Name, alpha_coeff_scalar);
    }
  }

  c0_scalar_flag = HDF5_InputFile.GetDatasetDimensionSizes(HDF5RootGroup, c0_Name) == ScalarSizes;
  if (c0_scalar_flag)
  {
    HDF5_InputFile.ReadScalarValue(HDF5RootGroup, c0_Name, c0_scalar);
  }

  if (nonlinear_flag)
  {
    BonA_scalar_flag = HDF5_InputFile.GetDatasetDimensionSizes(HDF5RootGroup, BonA_Name) == ScalarSizes;
    if (BonA_scalar_flag)
    {
      HDF5_InputFile.ReadScalarValue(HDF5RootGroup, BonA_Name, BonA_scalar);
    }
  }

  rho0_scalar_flag = HDF5_InputFile.GetDatasetDimensionSizes(HDF5RootGroup, rho0_Name) == ScalarSizes;
  if (rho0_scalar_flag)
  {
    HDF5_InputFile.ReadScalarValue(HDF5RootGroup, rho0_Name,     rho0_scalar);
    HDF5_InputFile.ReadScalarValue(HDF5RootGroup, rho0_sgx_Name, rho0_sgx_scalar);
    HDF5_InputFile.ReadScalarValue(HDF5RootGroup, rho0_sgy_Name, rho0_sgy_scalar);
    HDF5_InputFile.ReadScalarValue(HDF5RootGroup, rho0_sgz_Name, rho0_sgz_scalar);
    }
}// end of ReadScalarsFromHDF5InputFile
//----------------------------------------------------------------------------

/**
 * Save scalars into the output HDF5 file.
 * @param [in] HDF5_OutputFile - Handle to an opened output file where to store
 */
void TParameters::SaveScalarsToHDF5File(THDF5_File & HDF5_OutputFile)
{
  const hid_t HDF5RootGroup = HDF5_OutputFile.GetRootGroup();

  // Write dimension sizes
  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, Nx_Name, FullDimensionSizes.X);
  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, Ny_Name, FullDimensionSizes.Y);
  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, Nz_Name, FullDimensionSizes.Z);

  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, Nt_Name,  Nt);

  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, dt_Name, dt);
  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, dx_Name, dx);
  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, dy_Name, dy);
  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, dz_Name, dz);

  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, c_ref_Name, c_ref);

  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, pml_x_size_Name, pml_x_size);
  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, pml_y_size_Name, pml_y_size);
  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, pml_z_size_Name, pml_z_size);

  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, pml_x_alpha_Name, pml_x_alpha);
  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, pml_y_alpha_Name, pml_y_alpha);
  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, pml_z_alpha_Name, pml_z_alpha);

  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, ux_source_flag_Name, ux_source_flag);
  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, uy_source_flag_Name, uy_source_flag);
  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, uz_source_flag_Name, uz_source_flag);
  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, transducer_source_flag_Name, transducer_source_flag);

  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, p_source_flag_Name,  p_source_flag);
  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, p0_source_flag_Name, p0_source_flag);

  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, nonuniform_grid_flag_Name, nonuniform_grid_flag);
  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, absorbing_flag_Name,       absorbing_flag);
  HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, nonlinear_flag_Name,       nonlinear_flag);

  // uxyz_source_flags.
  if ((ux_source_flag > 0) || (uy_source_flag > 0) || (uz_source_flag > 0))
  {
    HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, u_source_many_Name, u_source_many);
      HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, u_source_mode_Name, u_source_mode);
  }

  // p_source_flag.
  if (p_source_flag != 0)
  {
    HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, p_source_many_Name, p_source_many);
    HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, p_source_mode_Name, p_source_mode);

    }

  // absorb flag
  if (absorbing_flag != 0)
  {
    HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, alpha_power_Name, alpha_power);
  }

  // if copy sensor mask, then copy the mask type
  if (IsCopySensorMask())
  {
    size_t SensorMaskTypeNumericValue = 0;

    switch (sensor_mask_type)
    {
      case smt_index: SensorMaskTypeNumericValue = 0;
        break;
      case smt_corners: SensorMaskTypeNumericValue = 1;
        break;
    }// switch

    HDF5_OutputFile.WriteScalarValue(HDF5RootGroup, sensor_mask_type_Name, SensorMaskTypeNumericValue);
  }

}// end of SaveScalarsToHDF5File

//------------------------------------------------------------------------------

//----------------------------------------------------------------------------//
//                              Implementation                                //
//                            protected methods                               //
//----------------------------------------------------------------------------//

/**
 * Constructor
 */
TParameters::TParameters() :
        CUDAParameters(),
        HDF5_InputFile(), HDF5_OutputFile(), HDF5_CheckpointFile(), HDF5_FileHeader(),
        CommandLinesParameters(),
        Nt(0), t_index(0), dt(0.0f),
        dx(0.0f), dy(0.0f), dz(0.0f),
        c_ref(0.0f), alpha_power(0.0f),
        FullDimensionSizes(0,0,0), ReducedDimensionSizes(0,0,0),
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

}// end of TFFT1DParameters
//----------------------------------------------------------------------------

//--------------------------------------------------------------------------//
//                              Implementation                              //
//                              private methods                             //
//--------------------------------------------------------------------------//

/**
 * Print usage end exit.
 */
void TParameters::PrintUsageAndExit()
{
  CommandLinesParameters.PrintUsageAndExit();
}// end of PrintUsage
//------------------------------------------------------------------------------

