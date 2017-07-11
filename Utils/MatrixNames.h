/**
 * @file        MatrixNames.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file storing names of all variables/matrices/output streams used in the
 *              simulation
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        14 September 2012, 17:28 (created) \n
 *              11 July      2017, 14:32 (revised)
 *
 * @section License
 * This file is part of the C++ extension of thq-Wave Toolbox
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

#ifndef TMatrixNamesH
#define	TMatrixNamesH

//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------- Constants ------------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief   Datatype for matrix names.
 * @details Datatype for matrix names.
 */
using MatrixName = const std::string;

/// Nt variable name
MatrixName kNtName         = "Nt";
/// t_index name
MatrixName kTIndexName     = "t_index";
/// dt variable name
MatrixName kDtName         = "dt";
/// dx variable name
MatrixName kDxName         = "dx";
/// dy variable name
MatrixName kDyName         = "dy";
/// dz variable name
MatrixName kDzName         = "dz";

/// c_ref variable name
MatrixName kCRefName       = "c_ref";
/// c0 variable name
MatrixName kC0Name         = "c0";

/// alpha_power variable name
MatrixName kAlphaPowerName = "alpha_power";
/// alpha_coeff variable name
MatrixName kAlphaCoeffName = "alpha_coeff";

/// Nx variable name
MatrixName kNxName         = "Nx";
/// Ny variable name
MatrixName kNyName         = "Ny";
/// Nz variable name
MatrixName kNzName         = "Nz";

/// x_shift_neg_r variable name
MatrixName kXShifyNegRName = "x_shift_neg_r";
/// y_shift_neg_r variable name
MatrixName kYShiftNegRName = "y_shift_neg_r";
/// z_shift_neg_r variable name
MatrixName kZShiftNegRName = "z_shift_neg_r";

/// ux_shifted variable name
MatrixName kUxShiftedName  = "ux_shifted";
/// uy_shifted variable name
MatrixName kUyShiftedName  = "uy_shifted";
/// uz_shifted variable name
MatrixName kUzShiftedName  = "uz_shifted";

/// pml_x_size variable name
MatrixName kPmlXSizeName   = "pml_x_size";
/// pml_y_size variable name
MatrixName kPmlYSizeName   = "pml_z_size";
/// pml_z_size variable name
MatrixName kPmlZSizeName   = "pml_y_size";

/// pml_x_sgx variable name
MatrixName kPmlXSgxName    = "pml_x_sgx";
/// pml_y_sgy variable name
MatrixName kPmlYSgyName    = "pml_y_sgy";
/// pml_z_sgz variable name
MatrixName  kPmlZSgzName   = "pml_z_sgz";

/// pml_x variable name
MatrixName kPmlXName       = "pml_x";
/// pml_y variable name
MatrixName kPmlYName       = "pml_y";
/// pml_z variable name
MatrixName kPmlZName       = "pml_z";


/// pml_x_alpha variable name
MatrixName kPmlXAlphaName    = "pml_x_alpha";
/// pml_y_alpha variable name
MatrixName kPmlYAlphaName    = "pml_y_alpha";
/// pml_z_alpha variable name
MatrixName kPmlZAlphaName    = "pml_z_alpha";

/// ux_source_flag variable name
MatrixName kUxSourceFlagName = "ux_source_flag";
/// uy_source_flag variable name
MatrixName kUySourceFlagName = "uy_source_flag";
/// uz_source_flag variable name
MatrixName kUzSourceFlagName = "uz_source_flag";

/// u_source_many variable name
MatrixName kUSourceManyName  = "u_source_many";
/// p_source_many variable name
MatrixName kPSourceManyName  = "p_source_many";

/// p_source_flag variable name
MatrixName kPSourceFlagName  = "p_source_flag";
/// p0_source_flag variable name
MatrixName kP0SourceFlagName = "p0_source_flag";

/// u_source_mode variable name
MatrixName kUSourceModeName  = "u_source_mode";
/// p_source_mode variable name
MatrixName kPSourceModeName  = "p_source_mode";

/// p_source_input variable name
MatrixName kPSourceInputName = "p_source_input";
/// p_source_index variable name
MatrixName kPSourceIndexName = "p_source_index";

/// u_source_index variable name
MatrixName kUSourceIndexName  = "u_source_index";
/// ux_source_input variable name
MatrixName kUxSourceInputName = "ux_source_input";
/// uy_source_input variable name
MatrixName kUySourceInputName = "uy_source_input";
/// uz_source_input variable name
MatrixName kUzSourceInputName = "uz_source_input";

/// nonuniform_grid_flag variable name
MatrixName kNonUniformGridFlagName   = "nonuniform_grid_flag";
/// absorbing_flag variable name
MatrixName kAbsorbingFlagName        = "absorbing_flag";
/// nonlinear_flag variable name
MatrixName kNonLinearFlagName        = "nonlinear_flag";

/// transducer_source_flag variable name
MatrixName kTransducerSourceFlagName = "transducer_source_flag";
/// sensor_mask_index variable name
MatrixName kSensorMaskIndexName      = "sensor_mask_index";
/// sensor_mask_type variable name
MatrixName kSensorMaskTypeName       = "sensor_mask_type";
/// sensor_mask_corners variable name
MatrixName kSensorMaskCornersName    = "sensor_mask_corners";

/// transducer_source_input variable name
MatrixName kTransducerSourceInputName = "transducer_source_input";

/// p0_source_input variable name
MatrixName kP0SourceInputName = "p0_source_input";
/// delay_mask variable name
MatrixName kDelayMaskName     = "delay_mask";


/// kappa_r variable name
MatrixName kKappaRName = "kappa_r";
/// BonA variable name
MatrixName kBonAName   = "BonA";
/// p variable name
MatrixName kPName      = "p";
/// rhox variable name
MatrixName kRhoxName   = "rhox";
/// rhoy variable name
MatrixName kRhoyName   = "rhoy";
/// rhoz variable name
MatrixName kRhozName   = "rhoz";

/// ux variable name
MatrixName kUxName     = "ux";
/// uy variable name
MatrixName kUyName     = "uy";
/// uz variable name
MatrixName kUzName     = "uz";

/// ux_sgx variable name
MatrixName kUxSgxName  = "ux_sgx";
/// uy_sgy variable name
MatrixName kUySgyName  = "uy_sgy";
/// uz_sgz variable name
MatrixName kUzSgzName  = "uz_sgz";

/// ux_non_staggered variable name
MatrixName kUxNonStaggeredName = "ux_non_staggered";
/// uy_non_staggered variable name
MatrixName kUyNonStaggeredName = "uy_non_staggered";
/// uz_non_staggered variable name
MatrixName kUzNonStaggeredName = "uz_non_staggered";

/// duxdx variable name
MatrixName kDuxdxName          = "duxdx";
/// duydy variable name
MatrixName kDuydyName          = "duydy";
/// duzdz variable name
MatrixName kDuzdzName          = "duzdz";

/// dxudxn variable name
MatrixName kDxudxnName         = "dxudxn";
/// dyudyn variable name
MatrixName kDyudynName         = "dyudyn";
/// dzudzn variable name
MatrixName kDzudznName         = "dzudzn";

/// dxudxn_sgx variable name
MatrixName kDxudxnSgxName      = "dxudxn_sgx";
/// dyudyn_sgy variable name
MatrixName kDyudynSgyName      = "dyudyn_sgy";
/// dzudzn_sgz variable name
MatrixName kDzudznSgzName      = "dzudzn_sgz";

/// ddx_k_shift_pos_r variable name
MatrixName kDdxKShiftPosRName  = "ddx_k_shift_pos_r";
/// ddy_k_shift_pos variable name
MatrixName kDdyKShiftPosName   = "ddy_k_shift_pos";
/// ddz_k_shift_pos variable name
MatrixName kDdzKShiftPosName   = "ddz_k_shift_pos";

/// ddx_k_shift_neg_r variable name
MatrixName kDdxKShiftNegRName  = "ddx_k_shift_neg_r";
/// ddy_k_shift_neg variable name
MatrixName kDdyKShiftNegName   = "ddy_k_shift_neg";
/// ddz_k_shift_neg variable name
MatrixName kDdzKShiftNegName   = "ddz_k_shift_neg";

/// rho0 variable name
MatrixName kRho0Name           = "rho0";
/// rho0_sgx variable name
MatrixName kRho0SgxName        = "rho0_sgx";
/// rho0_sgy variable name
MatrixName kRho0SgyName        = "rho0_sgy";
/// rho0_sgz variable name
MatrixName kRho0SgzName        = "rho0_sgz";

/// absorb_tau variable name
MatrixName kAbsorbTauName      = "absorb_tau";
/// absorb_eta variable name
MatrixName kAbsorbEtaName      = "absorb_eta";
/// absorb_nabla1_r variable name
MatrixName kAbsorbNabla1RName  = "absorb_nabla1_r";
/// absorb_nabla2_r variable name
MatrixName kAbsorbNabla2RName  = "absorb_nabla2_r";

/// p_rms variable name
MatrixName kPRmsName    = "p_rms";
/// p_max variable name
MatrixName kPMaxName    = "p_max";
/// p_min variable name
MatrixName kPminName    = "p_min";
/// p_max_all variable name
MatrixName kPMaxAllName = "p_max_all";
/// p_min_all variable name
MatrixName kPMinAllName = "p_min_all";
/// p_final variable name
MatrixName kPFinalName  = "p_final";

/// ux_rms variable name
MatrixName kUxRmsName = "ux_rms";
/// uy_rms variable name
MatrixName kUyRmsName = "uy_rms";
/// uz_rms variable name
MatrixName kUzRmsName = "uz_rms";

/// ux_max variable name
MatrixName kUxMaxName = "ux_max";
/// uy_max variable name
MatrixName kUyMaxName = "uy_max";
/// uz_max variable name
MatrixName kUzMaxName = "uz_max";
/// ux_min variable name
MatrixName kUxMinName = "ux_min";
/// uy_min variable name
MatrixName kUyMinName = "uy_min";
/// uz_min variable name
MatrixName kUzMinName = "uz_min";

/// ux_max_all variable name
MatrixName kUxMaxAllName = "ux_max_all";
/// uy_max_all variable name
MatrixName kUyMaxAllName = "uy_max_all";
/// uz_max_all variable name
MatrixName kUzMaxAllName = "uz_max_all";
/// ux_min_all variable name
MatrixName kUxMinAllName = "ux_min_all";
/// uy_min_all variable name
MatrixName kUyMinAllName = "uy_min_all";
/// uz_min_all variable name
MatrixName kUzMinAllName = "uz_min_all";

/// ux_final variable name
MatrixName kUxFinalName = "ux_final";
/// uy_final variable name
MatrixName kUyFinalName = "uy_final";
/// uz_final variable name
MatrixName kUzFinalName = "uz_final";


/// Temp_1_RS3D variable name
MatrixName kTemp1Real3DName = "Temp_1_RS3D";
/// Temp_2_RS3D variable name
MatrixName kTemp2Real3DName = "Temp_2_RS3D";
/// Temp_3_RS3D variable name
MatrixName kTemp3Real3DName = "Temp_3_RS3D";


/// CUFFT_shift_temp variable name
MatrixName kCufftShiftTempName = "CUFFT_shift_temp";
/// CUFFT_X_temp variable name
MatrixName kCufftXTempName     = "CUFFT_X_temp";
/// CUFFT_Y_temp variable name
MatrixName kCufftYTempName     = "CUFFT_Y_temp";
/// CUFFT_Z_temp variable name
MatrixName kCufftZTempName     = "CUFFT_Z_temp";
//----------------------------------------------------------------------------------------------------------------------

#endif	/* TMatrixNamesH */
