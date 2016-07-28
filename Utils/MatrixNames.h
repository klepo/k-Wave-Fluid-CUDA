/**
 * @file        MatrixNames.h
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
 *              25 July      2016, 10:02 (revised)
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

#ifndef MATRIX_NAMES_H
#define	MATRIX_NAMES_H

//------------------------------------------------------------------------------------------------//
//------------------------------------------ CONSTANTS -------------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * @typedef TMatrixName
 * @brief   Datatype for matrix names.
 * @details Datatype for matrix names.
 */
typedef const std::string TMatrixName;

/// Nt variable name
TMatrixName  Nt_NAME            = "Nt";
/// t_index name
TMatrixName  t_index_NAME       = "t_index";
/// dt variable name
TMatrixName  dt_NAME            = "dt";
/// dx variable name
TMatrixName  dx_NAME            = "dx";
/// dy variable name
TMatrixName  dy_NAME            = "dy";
/// dz variable name
TMatrixName  dz_NAME            = "dz";

/// c_ref variable name
TMatrixName  c_ref_NAME         = "c_ref";
/// c0 variable name
TMatrixName  c0_NAME            = "c0";

/// alpha_power variable name
TMatrixName  alpha_power_NAME   = "alpha_power";
/// alpha_coeff variable name
TMatrixName  alpha_coeff_NAME   = "alpha_coeff";

/// Nx variable name
TMatrixName  Nx_NAME            = "Nx";
/// Ny variable name
TMatrixName  Ny_NAME            = "Ny";
/// Nz variable name
TMatrixName  Nz_NAME            = "Nz";

/// x_shift_neg_r variable name
TMatrixName x_shift_neg_r_NAME  = "x_shift_neg_r";
/// y_shift_neg_r variable name
TMatrixName y_shift_neg_r_NAME  = "y_shift_neg_r";
/// z_shift_neg_r variable name
TMatrixName z_shift_neg_r_NAME  = "z_shift_neg_r";

/// ux_shifted variable name
TMatrixName ux_shifted_NAME     = "ux_shifted";
/// uy_shifted variable name
TMatrixName uy_shifted_NAME     = "uy_shifted";
/// uz_shifted variable name
TMatrixName uz_shifted_NAME     = "uz_shifted";

/// pml_x_size variable name
TMatrixName  pml_x_size_NAME    = "pml_x_size";
/// pml_y_size variable name
TMatrixName  pml_y_size_NAME    = "pml_z_size";
/// pml_z_size variable name
TMatrixName  pml_z_size_NAME    = "pml_y_size";

/// pml_x_sgx variable name
TMatrixName  pml_x_sgx_NAME     = "pml_x_sgx";
/// pml_y_sgy variable name
TMatrixName  pml_y_sgy_NAME     = "pml_y_sgy";
/// pml_z_sgz variable name
TMatrixName  pml_z_sgz_NAME     = "pml_z_sgz";

/// pml_x variable name
TMatrixName  pml_x_NAME         = "pml_x";
/// pml_y variable name
TMatrixName  pml_y_NAME         = "pml_y";
/// pml_z variable name
TMatrixName  pml_z_NAME         = "pml_z";


/// pml_x_alpha variable name
TMatrixName  pml_x_alpha_NAME   = "pml_x_alpha";
/// pml_y_alpha variable name
TMatrixName  pml_y_alpha_NAME   = "pml_y_alpha";
/// pml_z_alpha variable name
TMatrixName  pml_z_alpha_NAME   = "pml_z_alpha";

/// ux_source_flag variable name
TMatrixName ux_source_flag_NAME = "ux_source_flag";
/// uy_source_flag variable name
TMatrixName uy_source_flag_NAME = "uy_source_flag";
/// uz_source_flag variable name
TMatrixName uz_source_flag_NAME = "uz_source_flag";

/// u_source_many variable name
TMatrixName u_source_many_NAME  = "u_source_many";
/// p_source_many variable name
TMatrixName p_source_many_NAME  = "p_source_many";

/// p_source_flag variable name
TMatrixName p_source_flag_NAME  = "p_source_flag";
/// p0_source_flag variable name
TMatrixName p0_source_flag_NAME = "p0_source_flag";

/// u_source_mode variable name
TMatrixName u_source_mode_NAME  = "u_source_mode";
/// p_source_mode variable name
TMatrixName p_source_mode_NAME  = "p_source_mode";

/// p_source_input variable name
TMatrixName p_source_input_NAME = "p_source_input";
/// p_source_index variable name
TMatrixName p_source_index_NAME = "p_source_index";

/// u_source_index variable name
TMatrixName u_source_index_NAME  = "u_source_index";
/// ux_source_input variable name
TMatrixName ux_source_input_NAME = "ux_source_input";
/// uy_source_input variable name
TMatrixName uy_source_input_NAME = "uy_source_input";
/// uz_source_input variable name
TMatrixName uz_source_input_NAME = "uz_source_input";

/// nonuniform_grid_flag variable name
TMatrixName nonuniform_grid_flag_NAME    = "nonuniform_grid_flag";
/// absorbing_flag variable name
TMatrixName absorbing_flag_NAME          = "absorbing_flag";
/// nonlinear_flag variable name
TMatrixName nonlinear_flag_NAME          = "nonlinear_flag";

/// transducer_source_flag variable name
TMatrixName transducer_source_flag_NAME  = "transducer_source_flag";
/// sensor_mask_index variable name
TMatrixName sensor_mask_index_NAME       = "sensor_mask_index";
/// sensor_mask_type variable name
TMatrixName sensor_mask_type_NAME        = "sensor_mask_type";
/// sensor_mask_corners variable name
TMatrixName sensor_mask_corners_NAME     = "sensor_mask_corners";

/// transducer_source_input variable name
TMatrixName transducer_source_input_NAME = "transducer_source_input";

/// p0_source_input variable name
TMatrixName p0_source_input_NAME = "p0_source_input";
/// delay_mask variable name
TMatrixName delay_mask_NAME      = "delay_mask";


/// kappa_r variable name
TMatrixName  kappa_r_NAME        = "kappa_r";
/// BonA variable name
TMatrixName  BonA_NAME           = "BonA";
/// p variable name
TMatrixName  p_NAME              = "p";
/// rhox variable name
TMatrixName  rhox_NAME           = "rhox";
/// rhoy variable name
TMatrixName  rhoy_NAME           = "rhoy";
/// rhoz variable name
TMatrixName  rhoz_NAME           = "rhoz";

/// ux variable name
TMatrixName  ux_NAME             = "ux";
/// uy variable name
TMatrixName  uy_NAME             = "uy";
/// uz variable name
TMatrixName  uz_NAME             = "uz";

/// ux_sgx variable name
TMatrixName  ux_sgx_NAME         = "ux_sgx";
/// uy_sgy variable name
TMatrixName  uy_sgy_NAME         = "uy_sgy";
/// uz_sgz variable name
TMatrixName  uz_sgz_NAME         = "uz_sgz";

/// ux_non_staggered variable name
TMatrixName  ux_non_staggered_NAME = "ux_non_staggered";
/// uy_non_staggered variable name
TMatrixName  uy_non_staggered_NAME = "uy_non_staggered";
/// uz_non_staggered variable name
TMatrixName  uz_non_staggered_NAME = "uz_non_staggered";

/// duxdx variable name
TMatrixName  duxdx_NAME          = "duxdx";
/// duydy variable name
TMatrixName  duydy_NAME          = "duydy";
/// duzdz variable name
TMatrixName  duzdz_NAME          = "duzdz";

/// dxudxn variable name
TMatrixName  dxudxn_NAME         = "dxudxn";
/// dyudyn variable name
TMatrixName  dyudyn_NAME         = "dyudyn";
/// dzudzn variable name
TMatrixName  dzudzn_NAME         = "dzudzn";

/// dxudxn_sgx variable name
TMatrixName  dxudxn_sgx_NAME     = "dxudxn_sgx";
/// dyudyn_sgy variable name
TMatrixName  dyudyn_sgy_NAME     = "dyudyn_sgy";
/// dzudzn_sgz variable name
TMatrixName  dzudzn_sgz_NAME     = "dzudzn_sgz";

/// ddx_k_shift_pos_r variable name
TMatrixName  ddx_k_shift_pos_r_NAME = "ddx_k_shift_pos_r";
/// ddy_k_shift_pos variable name
TMatrixName  ddy_k_shift_pos_NAME   = "ddy_k_shift_pos";
/// ddz_k_shift_pos variable name
TMatrixName  ddz_k_shift_pos_NAME   = "ddz_k_shift_pos";

/// ddx_k_shift_neg_r variable name
TMatrixName  ddx_k_shift_neg_r_NAME = "ddx_k_shift_neg_r";
/// ddy_k_shift_neg variable name
TMatrixName  ddy_k_shift_neg_NAME   = "ddy_k_shift_neg";
/// ddz_k_shift_neg variable name
TMatrixName  ddz_k_shift_neg_NAME   = "ddz_k_shift_neg";

/// rho0 variable name
TMatrixName  rho0_NAME            = "rho0";
/// rho0_sgx variable name
TMatrixName  rho0_sgx_NAME        = "rho0_sgx";
/// rho0_sgy variable name
TMatrixName  rho0_sgy_NAME        = "rho0_sgy";
/// rho0_sgz variable name
TMatrixName  rho0_sgz_NAME        = "rho0_sgz";

/// absorb_tau variable name
TMatrixName  absorb_tau_NAME      = "absorb_tau";
/// absorb_eta variable name
TMatrixName  absorb_eta_NAME      = "absorb_eta";
/// absorb_nabla1_r variable name
TMatrixName  absorb_nabla1_r_NAME = "absorb_nabla1_r";
/// absorb_nabla2_r variable name
TMatrixName  absorb_nabla2_r_NAME = "absorb_nabla2_r";

/// p_rms variable name
TMatrixName  p_rms_NAME     = "p_rms";
/// p_max variable name
TMatrixName  p_max_NAME     = "p_max";
/// p_min variable name
TMatrixName  p_min_NAME     = "p_min";
/// p_max_all variable name
TMatrixName  p_max_all_NAME = "p_max_all";
/// p_min_all variable name
TMatrixName  p_min_all_NAME = "p_min_all";
/// p_final variable name
TMatrixName  p_final_NAME   = "p_final";

/// ux_rms variable name
TMatrixName  ux_rms_NAME = "ux_rms";
/// uy_rms variable name
TMatrixName  uy_rms_NAME = "uy_rms";
/// uz_rms variable name
TMatrixName  uz_rms_NAME = "uz_rms";

/// ux_max variable name
TMatrixName  ux_max_NAME = "ux_max";
/// uy_max variable name
TMatrixName  uy_max_NAME = "uy_max";
/// uz_max variable name
TMatrixName  uz_max_NAME = "uz_max";
/// ux_min variable name
TMatrixName  ux_min_NAME = "ux_min";
/// uy_min variable name
TMatrixName  uy_min_NAME = "uy_min";
/// uz_min variable name
TMatrixName  uz_min_NAME = "uz_min";

/// ux_max_all variable name
TMatrixName  ux_max_all_NAME = "ux_max_all";
/// uy_max_all variable name
TMatrixName  uy_max_all_NAME = "uy_max_all";
/// uz_max_all variable name
TMatrixName  uz_max_all_NAME = "uz_max_all";
/// ux_min_all variable name
TMatrixName  ux_min_all_NAME = "ux_min_all";
/// uy_min_all variable name
TMatrixName  uy_min_all_NAME = "uy_min_all";
/// uz_min_all variable name
TMatrixName  uz_min_all_NAME = "uz_min_all";

/// ux_final variable name
TMatrixName  ux_final_NAME = "ux_final";
/// uy_final variable name
TMatrixName  uy_final_NAME = "uy_final";
/// uz_final variable name
TMatrixName  uz_final_NAME = "uz_final";


/// Temp_1_RS3D variable name
TMatrixName temp_1_real_3D_NAME = "Temp_1_RS3D";
/// Temp_2_RS3D variable name
TMatrixName temp_2_real_3D_NAME = "Temp_2_RS3D";
/// Temp_3_RS3D variable name
TMatrixName temp_3_real_3D_NAME = "Temp_3_RS3D";


/// CUFFT_shift_temp variable name
TMatrixName cufft_shift_temp_NAME = "CUFFT_shift_temp";
/// CUFFT_X_temp variable name
TMatrixName cufft_X_temp_NAME     = "CUFFT_X_temp";
/// CUFFT_Y_temp variable name
TMatrixName cufft_Y_temp_NAME     = "CUFFT_Y_temp";
/// CUFFT_Z_temp variable name
TMatrixName cufft_z_temp_NAME     = "CUFFT_Z_temp";

#endif	/* MATRIX_NAMES_H */

