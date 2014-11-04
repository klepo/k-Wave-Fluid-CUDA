/**
 * @file        Parameters.cpp
 * @author      Jiri Jaros   \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing parameters of the simulation.
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        09 August    2012, 13:39 (created) \n
 *              04 November  2014, 17:26 (revised)
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

#include <omp.h>
#include <iostream>
#include <string>
#include <sstream>
#include <exception>
#include <stdexcept>

#include "../Parameters/Parameters.h"
#include "../Utils/MatrixNames.h"
#include "../Utils/ErrorMessages.h"

#if CUDA_VERSION
#include "../CUDA/CUDATuner.h"
#elif OPENCL_VERSION
#include "../OpenCL/OpenCLTuner.h"
#endif

using namespace std;

//--------------------------------------------------------------------------//
//                            Constants                                     //
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
//                            Definitions                                   //
//--------------------------------------------------------------------------//

bool TParameters::ParametersInstanceFlag = false;

TParameters* TParameters::ParametersSingle = NULL;

//--------------------------------------------------------------------------//
//                            Implementation                                //
//                            public methods                                //
//--------------------------------------------------------------------------//

/*
 * Get instance of singleton class.
 */
TParameters* TParameters::GetInstance()
{

    if(!ParametersInstanceFlag)
    {
        ParametersSingle = new TParameters();
        ParametersInstanceFlag = true;
        return ParametersSingle;
    }
    else
    {
        return ParametersSingle;
    }

}// end of Create
//----------------------------------------------------------------------------

/*
 * Parse command line
 * @param [in] argc
 * @param [in] argv
 */
void TParameters::ParseCommandLine(int argc, char** argv)
{

    command_line_parameters.ParseCommandLine(argc, argv);

    if (command_line_parameters.IsVersion()){
        return;
    }

    ReadScalarsFromHDF5InputFile(HDF5_InputFile);

    if (command_line_parameters.IsBenchmarkFlag()){
        Nt = command_line_parameters.GetBenchmarkTimeStepsCount();
    }

    if ((Nt <= command_line_parameters.GetStartTimeIndex()) ||
        (0  > command_line_parameters.GetStartTimeIndex())){
        fprintf(stderr,
                Parameters_ERR_FMT_Illegal_StartTime_value,
                static_cast<size_t>(1),
                Nt);
        command_line_parameters.PrintUsageAndExit();
    }

    #if CUDA_VERSION
    //cuboid corners is currently not supported in CUDA version
    //thus check for it and notify the user accordingly
    if(this->Get_sensor_mask_type() == TParameters::smt_corners){
        fprintf(stderr,
                CUDA_ERR_FMT_OptionNotSupported,
                "SMT Cuboid corners");
        command_line_parameters.PrintUsageAndExit();
    }
    //__u_non_staggered is also not currently supported in the CUDA version so
    //notify the user and fail.
    if(command_line_parameters.IsStore_u_non_staggered_raw()){
        fprintf(stderr,
                CUDA_ERR_FMT_OptionNotSupported,
                "--u_non_staggered_raw");
        command_line_parameters.PrintUsageAndExit();
    }
#endif

#if CUDA_VERSION
    CUDATuner* tuner = CUDATuner::GetInstance();
#elif OPENCL_VERSION
    OpenCLTuner* tuner = OpenCLTuner::GetInstance();
#endif

#if CUDA_VERSION || OPENCL_VERSION
    int pGPU_ID = command_line_parameters.GetGPUDeviceID();
    bool did_set_device = tuner->SetDevice(pGPU_ID);
    if(!did_set_device){
        fprintf(stderr, "Error: couldn't setup the device!\n");
        exit(EXIT_FAILURE);
    }
    tuner->Set1DBlockSize(command_line_parameters.Get1DBlockSize());

    tuner->Set3DBlockSize(command_line_parameters.Get3DBlockSize_X(),
                          command_line_parameters.Get3DBlockSize_Y(),
                          command_line_parameters.Get3DBlockSize_Z());

    bool device_can_use_block_sizes = tuner->CanDeviceHandleBlockSizes();
    if(!device_can_use_block_sizes){
        fprintf(stderr, "Error: the device %i cannot handle either the \n"
                        "\t1D blocksize of %i\n"
                        "\tor\n"
                        "\t3D blocksize of %i,%i,%i\n",
                        pGPU_ID,
                        command_line_parameters.Get1DBlockSize(),
                        command_line_parameters.Get3DBlockSize_X(),
                        command_line_parameters.Get3DBlockSize_Y(),
                        command_line_parameters.Get3DBlockSize_Z());
        exit(EXIT_FAILURE);
    }
#endif

}// end of ParseCommandLine
//----------------------------------------------------------------------------

/*
 * Read scalar values from the input HDF5 file.
 *
 * @param [in] HDF5_InputFile - Handle to an opened input file
 */
void TParameters::ReadScalarsFromHDF5InputFile(THDF5_File & HDF5_InputFile)
{

    TDimensionSizes ScalarSizes(1,1,1);

    if (!HDF5_InputFile.IsOpened()) {
        // Open file
        try{
            HDF5_InputFile.Open(
                    command_line_parameters.GetInputFileName().c_str());
        } catch (ios::failure e){
            fprintf(stderr,"%s",e.what());
            PrintUsageAndExit();
        }
    }

    HDF5_FileHeader.ReadHeaderFromInputFile(HDF5_InputFile);

    if (HDF5_FileHeader.GetFileType() != THDF5_FileHeader::hdf5_ft_input) {
        char ErrorMessage[256] = "";
        sprintf(ErrorMessage,
                Parameters_ERR_FMT_IncorrectInputFileFormat,
                GetInputFileName().c_str());
        throw ios::failure(ErrorMessage);
    }

    if (!HDF5_FileHeader.CheckMajorFileVersion()) {
        char ErrorMessage[256] = "";
        sprintf(ErrorMessage,
                Parameters_ERR_FMT_IncorrectMajorHDF5FileVersion,
                GetInputFileName().c_str(),
                HDF5_FileHeader.GetCurrentHDF5_MajorVersion().c_str());
        throw ios::failure(ErrorMessage);
    }

    if (!HDF5_FileHeader.CheckMinorFileVersion()) {
        char ErrorMessage[256] = "";
        sprintf(ErrorMessage,
                Parameters_ERR_FMT_IncorrectMinorHDF5FileVersion,
                GetInputFileName().c_str(),
                HDF5_FileHeader.GetCurrentHDF5_MinorVersion().c_str());
        throw ios::failure(ErrorMessage);
    }

    const hid_t HDF5RootGroup = HDF5_InputFile.GetRootGroup();

    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup, Nt_Name, ScalarSizes, &Nt);

    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup, dt_Name, ScalarSizes, &dt);
    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup, dx_Name, ScalarSizes, &dx);
    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup, dy_Name, ScalarSizes, &dy);
    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup, dz_Name, ScalarSizes, &dz);

    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                       c_ref_Name,
                                       ScalarSizes,
                                       &c_ref);

    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                       pml_x_size_Name,
                                       ScalarSizes,
                                       &pml_x_size);
    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                       pml_y_size_Name,
                                       ScalarSizes,
                                       &pml_y_size);
    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                       pml_z_size_Name,
                                       ScalarSizes,
                                       &pml_z_size);

    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                       pml_x_alpha_Name,
                                       ScalarSizes,
                                       &pml_x_alpha);
    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                       pml_y_alpha_Name,
                                       ScalarSizes,
                                       &pml_y_alpha);
    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                       pml_z_alpha_Name,
                                       ScalarSizes,
                                       &pml_z_alpha);

    size_t X, Y, Z;
	HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup, Nx_Name, ScalarSizes, &X);
    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup, Ny_Name, ScalarSizes, &Y);
    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup, Nz_Name, ScalarSizes, &Z);


    FullDimensionSizes.X = X;
    FullDimensionSizes.Y = Y;
    FullDimensionSizes.Z = Z;

    ReducedDimensionSizes.X = ((X/2) + 1);
    ReducedDimensionSizes.Y = Y;
    ReducedDimensionSizes.Z = Z;

    // if the file is of version 1.0, there must be a sensor mask index
    // (backward compatibility)
    if (HDF5_FileHeader.GetFileVersion() == THDF5_FileHeader::hdf5_fv_10) {
        sensor_mask_ind_size =
            HDF5_InputFile.GetDatasetElementCount(HDF5RootGroup,
                                                  sensor_mask_index_Name);

        //if -u_non_staggered_raw enabled, throw an error - not supported
        if (IsStore_u_non_staggered_raw()) {
            throw ios::failure(Parameters_ERR_FMT_UNonStaggeredNotSupportedForFile10);
        }
    }

    // This is the current version 1.1
    if (HDF5_FileHeader.GetFileVersion() == THDF5_FileHeader::hdf5_fv_11) {

        // read sensor mask type as a size_t value to enum
        size_t SensorMaskTypeLongValue = 0;
        HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                           sensor_mask_type_Name,
                                           ScalarSizes,
                                           &SensorMaskTypeLongValue);

        // convert the long value on
        switch (SensorMaskTypeLongValue) {
            case 0: sensor_mask_type = smt_index;
                    break;
            case 1: sensor_mask_type = smt_corners;
                    break;
            default: throw ios::failure(Parameters_ERR_FMT_WrongSensorMaskType);
                     break;
        }//case

        // read the input mask size
        switch (sensor_mask_type) {
            case smt_index:
                {
                    sensor_mask_ind_size =
                        HDF5_InputFile.GetDatasetElementCount(HDF5RootGroup,
                                                              sensor_mask_index_Name);
                    break;
                }
            case smt_corners:
                {
                    // mask dimensions are [6, N, 1] - I want to know N
                    sensor_mask_corners_size = HDF5_InputFile.GetDatasetDimensionSizes(
                            HDF5RootGroup,
                            sensor_mask_corners_Name).Y;
                    break;
                }
        }// switch
    }// version 1.1

    // flags
    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                       ux_source_flag_Name,
                                       ScalarSizes,
                                       &ux_source_flag);
    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                       uy_source_flag_Name,
                                       ScalarSizes,
                                       &uy_source_flag);
    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                       uz_source_flag_Name,
                                       ScalarSizes,
                                       &uz_source_flag);
    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                       transducer_source_flag_Name,
                                       ScalarSizes,
                                       &transducer_source_flag);

    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                       p_source_flag_Name,
                                       ScalarSizes,
                                       &p_source_flag);
    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                       p0_source_flag_Name,
                                       ScalarSizes,
                                       &p0_source_flag);

    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                       nonuniform_grid_flag_Name,
                                       ScalarSizes,
                                       &nonuniform_grid_flag);
    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                       absorbing_flag_Name,
                                       ScalarSizes,
                                       &absorbing_flag);
    HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                       nonlinear_flag_Name,
                                       ScalarSizes,
                                       &nonlinear_flag);

    //--- Vector sizes ---//
    if (transducer_source_flag == 0)
        transducer_source_input_size = 0;
    else {
        transducer_source_input_size =
            HDF5_InputFile.GetDatasetElementCount(HDF5RootGroup,
                                                  transducer_source_input_Name);
    }

    if ((transducer_source_flag > 0) ||
        (ux_source_flag > 0)         ||
        (uy_source_flag > 0)         ||
        (uz_source_flag > 0)){
        u_source_index_size =
            HDF5_InputFile.GetDatasetElementCount(HDF5RootGroup,
                                                  u_source_index_Name);
    }

    //-- uxyz_source_flags --//
    if ((ux_source_flag > 0) ||
        (uy_source_flag > 0) ||
        (uz_source_flag > 0)){
        HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                           u_source_many_Name,
                                           ScalarSizes,
                                           &u_source_many);
        HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                           u_source_mode_Name,
                                           ScalarSizes,
                                           &u_source_mode);
    } else{
        u_source_many = 0;
        u_source_mode = 0;
    }

    //-- p_source_flag --//
    if (p_source_flag != 0) {
        HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                           p_source_many_Name,
                                           ScalarSizes,
                                           &p_source_many);
        HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                           p_source_mode_Name,
                                           ScalarSizes,
                                           &p_source_mode);

        p_source_index_size =
            HDF5_InputFile.GetDatasetElementCount(HDF5RootGroup,
                                                  p_source_index_Name);

    } else{
        p_source_mode = 0;
        p_source_many = 0;
        p_source_index_size = 0;
    }

    // absorb flag
    if (absorbing_flag != 0) {
        HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                           alpha_power_Name,
                                           ScalarSizes,
                                           &alpha_power);
        if (alpha_power == 1.0f){
            fprintf(stderr, "%s", Parameters_ERR_FMT_Illegal_alpha_power_value);
            PrintUsageAndExit();
        }

        alpha_coeff_scalar_flag =
            HDF5_InputFile.GetDatasetDimensionSizes(HDF5RootGroup, alpha_coeff_Name)
            == ScalarSizes;

        if (alpha_coeff_scalar_flag){
            HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                               alpha_coeff_Name,
                                               ScalarSizes,
                                               &alpha_coeff_scalar);
        }
    }

    c0_scalar_flag =
        HDF5_InputFile.GetDatasetDimensionSizes(HDF5RootGroup, c0_Name) == ScalarSizes;
    if (c0_scalar_flag){
        HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                           c0_Name,
                                           ScalarSizes,
                                           &c0_scalar);
    }

    if (nonlinear_flag){
        BonA_scalar_flag =
            HDF5_InputFile.GetDatasetDimensionSizes(HDF5RootGroup, BonA_Name) == ScalarSizes;
        if (BonA_scalar_flag){
            HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                               BonA_Name,
                                               ScalarSizes,
                                               &BonA_scalar);
        }
    }

    rho0_scalar_flag =
        HDF5_InputFile.GetDatasetDimensionSizes(HDF5RootGroup, rho0_Name) == ScalarSizes;
    if (rho0_scalar_flag) {
        HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                           rho0_Name,
                                           ScalarSizes,
                                           &rho0_scalar);
        HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                           rho0_sgx_Name,
                                           ScalarSizes,
                                           &rho0_sgx_scalar);
        HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                           rho0_sgy_Name,
                                           ScalarSizes,
                                           &rho0_sgy_scalar);
        HDF5_InputFile.ReadCompleteDataset(HDF5RootGroup,
                                           rho0_sgz_Name,
                                           ScalarSizes,
                                           &rho0_sgz_scalar);
    }
}// end of ReadScalarsFromMatlabInputFile
//----------------------------------------------------------------------------

/*
 * Save scalars into the output HDF5 file.
 * @param [in] HDF5_OutputFile - Handle to an opened output file where to store
 */
void TParameters::SaveScalarsToHDF5File(THDF5_File & HDF5_OutputFile)
{
    const hid_t HDF5RootGroup = HDF5_OutputFile.GetRootGroup();

    // Write dimension sizes
    HDF5_OutputFile.WriteScalarValue(HDF5RootGroup,
                                     Nx_Name,
                                     static_cast<size_t>(FullDimensionSizes.X));
    HDF5_OutputFile.WriteScalarValue(HDF5RootGroup,
                                     Ny_Name,
                                     static_cast<size_t>(FullDimensionSizes.Y));
    HDF5_OutputFile.WriteScalarValue(HDF5RootGroup,
                                     Nz_Name,
                                     static_cast<size_t>(FullDimensionSizes.Z));

    HDF5_OutputFile.WriteScalarValue(HDF5RootGroup,
                                     Nt_Name,
                                     static_cast<size_t>(Nt));

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

    HDF5_OutputFile.WriteScalarValue(HDF5RootGroup,
                                     ux_source_flag_Name,
                                     ux_source_flag);
    HDF5_OutputFile.WriteScalarValue(HDF5RootGroup,
                                     uy_source_flag_Name,
                                     uy_source_flag);
    HDF5_OutputFile.WriteScalarValue(HDF5RootGroup,
                                     uz_source_flag_Name,
                                     uz_source_flag);
    HDF5_OutputFile.WriteScalarValue(HDF5RootGroup,
                                     transducer_source_flag_Name,
                                     transducer_source_flag);

    HDF5_OutputFile.WriteScalarValue(HDF5RootGroup,
                                     p_source_flag_Name,
                                     p_source_flag);
    HDF5_OutputFile.WriteScalarValue(HDF5RootGroup,
                                     p0_source_flag_Name,
                                     p0_source_flag);

    HDF5_OutputFile.WriteScalarValue(HDF5RootGroup,
                                     nonuniform_grid_flag_Name,
                                     nonuniform_grid_flag);
    HDF5_OutputFile.WriteScalarValue(HDF5RootGroup,
                                     absorbing_flag_Name,
                                     absorbing_flag);
    HDF5_OutputFile.WriteScalarValue(HDF5RootGroup,
                                     nonlinear_flag_Name,
                                     nonlinear_flag);

    //-- uxyz_source_flags --//
    if ((ux_source_flag > 0) || (uy_source_flag > 0) || (uz_source_flag > 0)){
        HDF5_OutputFile.WriteScalarValue(HDF5RootGroup,
                                         u_source_many_Name,
                                         u_source_many);
        HDF5_OutputFile.WriteScalarValue(HDF5RootGroup,
                                         u_source_mode_Name,
                                         u_source_mode);
    }

    //-- p_source_flag --//
    if (p_source_flag != 0) {

        HDF5_OutputFile.WriteScalarValue(HDF5RootGroup,
                                         p_source_many_Name,
                                         p_source_many);
        HDF5_OutputFile.WriteScalarValue(HDF5RootGroup,
                                         p_source_mode_Name,
                                         p_source_mode);

    }

    // absorb flag
    if (absorbing_flag != 0) {
        HDF5_OutputFile.WriteScalarValue(HDF5RootGroup,
                                         alpha_power_Name,
                                         alpha_power);
    }

}// end of SaveScalarsToHDF5File

//--------------------------------------------------------------------------//
//                              Implementation                              //
//                            protected methods                             //
//--------------------------------------------------------------------------//

/*
 * Constructor
 */
TParameters::TParameters()
{
    Nt = 0;
    dt = 0.0f;

    dx, dy, dz = 0.0f;

    c_ref = 0.0f;
    alpha_power = 0.0f;

    FullDimensionSizes    = TDimensionSizes(0,0,0);
    ReducedDimensionSizes = TDimensionSizes(0,0,0);

    sensor_mask_ind_size, u_source_index_size, p_source_index_size = 0;

    transducer_source_input_size = 0;

    ux_source_flag, uy_source_flag, uz_source_flag = 0;

    p_source_flag, p0_source_flag, transducer_source_flag = 0;

    u_source_many, u_source_mode = 0;

    p_source_mode, p_source_many = 0;

    nonuniform_grid_flag, absorbing_flag, nonlinear_flag = 0;

    pml_x_size, pml_y_size, pml_z_size = 0;

    alpha_coeff_scalar_flag = false;
    alpha_coeff_scalar = 0.0f;

    c0_scalar_flag = false;
    c0_scalar = 0.0f;

    absorb_eta_scalar, absorb_tau_scalar = 0.0f;

    BonA_scalar_flag = false;
    BonA_scalar = 0.0f;

    rho0_scalar_flag = false;
    rho0_scalar = 0.0f;

    rho0_sgx_scalar, rho0_sgy_scalar, rho0_sgz_scalar = 0.0f;

}// end of TFFT1DParameters
//----------------------------------------------------------------------------

//--------------------------------------------------------------------------//
//                              Implementation                              //
//                              private methods                             //
//--------------------------------------------------------------------------//

/*
 * print usage end exit
 */
void TParameters::PrintUsageAndExit(){

    command_line_parameters.PrintUsageAndExit();

}// end of PrintUsage
//------------------------------------------------------------------------------

