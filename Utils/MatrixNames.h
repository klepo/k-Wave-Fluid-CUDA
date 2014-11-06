/**
 * @file        MatrixNames.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file storing names of all variables.
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        14 September 2012, 17:28 (created) \n
 *              04 November  2014, 15:08W (revised)
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

#ifndef MATRIXNAMES_H
#define	MATRIXNAMES_H

//----------------------------------------------------------------------------//
//                              Constants                                     //
//----------------------------------------------------------------------------//

/// Nt variable name
const char * const  Nt_Name                     = "Nt";
/// t_index name
const char * const  t_index_Name                = "t_index";
/// dt variable name
const char * const  dt_Name                     = "dt";
/// dx variable name
const char * const  dx_Name                     = "dx";
/// dy variable name
const char * const  dy_Name                     = "dy";
/// dz variable name
const char * const  dz_Name                     = "dz";

/// c_ref variable name
const char * const  c_ref_Name                  = "c_ref";
/// c0 variable name
const char * const  c0_Name                     = "c0";

/// alpha_power variable name
const char * const  alpha_power_Name            = "alpha_power";
/// alpha_coeff variable name
const char * const  alpha_coeff_Name            = "alpha_coeff";

/// Nx variable name
const char * const  Nx_Name                     = "Nx";
/// Ny variable name
const char * const  Ny_Name                     = "Ny";
/// Nz variable name
const char * const  Nz_Name                     = "Nz";

/// x_shift_neg_r variable name
const char * const x_shift_neg_r_Name           = "x_shift_neg_r";
/// y_shift_neg_r variable name
const char * const y_shift_neg_r_Name           = "y_shift_neg_r";
/// z_shift_neg_r variable name
const char * const z_shift_neg_r_Name           = "z_shift_neg_r";

/// ux_shifted variable name
const char * const ux_shifted_Name             = "ux_shifted";
/// uy_shifted variable name
const char * const uy_shifted_Name             = "uy_shifted";
/// uz_shifted variable name
const char * const uz_shifted_Nam  e           = "uz_shifted";

/// pml_x_size variable name
const char * const  pml_x_size_Name             = "pml_x_size";
/// pml_y_size variable name
const char * const  pml_y_size_Name             = "pml_z_size";
/// pml_z_size variable name
const char * const  pml_z_size_Name             = "pml_y_size";

/// pml_x_sgx variable name
const char * const  pml_x_sgx_Name             = "pml_x_sgx";
/// pml_y_sgy variable name
const char * const  pml_y_sgy_Name             = "pml_y_sgy";
/// pml_z_sgz variable name
const char * const  pml_z_sgz_Name             = "pml_z_sgz";

/// pml_x variable name
const char * const  pml_x_Name                 = "pml_x";
/// pml_y variable name
const char * const  pml_y_Name                 = "pml_y";
/// pml_z variable name
const char * const  pml_z_Name                 = "pml_z";


/// pml_x_alpha variable name
const char * const  pml_x_alpha_Name           = "pml_x_alpha";
/// pml_y_alpha variable name
const char * const  pml_y_alpha_Name           = "pml_y_alpha";
/// pml_z_alpha variable name
const char * const  pml_z_alpha_Name           = "pml_z_alpha";

/// ux_source_flag variable name
const char * const ux_source_flag_Name         = "ux_source_flag";
/// uy_source_flag variable name
const char * const uy_source_flag_Name         = "uy_source_flag";
/// uz_source_flag variable name
const char * const uz_source_flag_Name         = "uz_source_flag";

/// u_source_many variable name
const char * const u_source_many_Name          = "u_source_many";
/// p_source_many variable name
const char * const p_source_many_Name          = "p_source_many";

/// p_source_flag variable name
const char * const p_source_flag_Name          = "p_source_flag";
/// p0_source_flag variable name
const char * const p0_source_flag_Name         = "p0_source_flag";

/// u_source_mode variable name
const char * const u_source_mode_Name          = "u_source_mode";
/// p_source_mode variable name
const char * const p_source_mode_Name          = "p_source_mode";

/// p_source_input variable name
const char * const p_source_input_Name         = "p_source_input";
/// p_source_index variable name
const char * const p_source_index_Name         = "p_source_index";

/// u_source_index variable name
const char * const u_source_index_Name         = "u_source_index";
/// ux_source_input variable name
const char * const ux_source_input_Name        = "ux_source_input";
/// uy_source_input variable name
const char * const uy_source_input_Name        = "uy_source_input";
/// uz_source_input variable name
const char * const uz_source_input_Name        = "uz_source_input";

/// nonuniform_grid_flag variable name
const char * const nonuniform_grid_flag_Name   = "nonuniform_grid_flag";
/// absorbing_flag variable name
const char * const absorbing_flag_Name         = "absorbing_flag";
/// nonlinear_flag variable name
const char * const nonlinear_flag_Name         = "nonlinear_flag";

/// transducer_source_flag variable name
const char * const transducer_source_flag_Name = "transducer_source_flag";
/// sensor_mask_index variable name
const char * const sensor_mask_index_Name      = "sensor_mask_index";
/// sensor_mask_type variable name
const char * const sensor_mask_type_Name       = "sensor_mask_type";
/// sensor_mask_corners variable name
const char * const sensor_mask_corners_Name    = "sensor_mask_corners";

/// transducer_source_input variable name
const char * const transducer_source_input_Name= "transducer_source_input";

/// p0_source_input variable name
const char * const p0_source_input_Name = "p0_source_input";
/// delay_mask variable name
const char * const delay_mask_Name      = "delay_mask";


/// kappa_r variable name
const char * const  kappa_r_Name        = "kappa_r";
/// BonA variable name
const char * const  BonA_Name           = "BonA";
/// p variable name
const char * const  p_Name              = "p";
/// rhox variable name
const char * const  rhox_Name           = "rhox";
/// rhoy variable name
const char * const  rhoy_Name           = "rhoy";
/// rhoz variable name
const char * const  rhoz_Name           = "rhoz";

/// ux variable name
const char * const  ux_Name             = "ux";
/// uy variable name
const char * const  uy_Name             = "uy";
/// uz variable name
const char * const  uz_Name             = "uz";

/// ux_sgx variable name
const char * const  ux_sgx_Name         = "ux_sgx";
/// uy_sgy variable name
const char * const  uy_sgy_Name         = "uy_sgy";
/// uz_sgz variable name
const char * const  uz_sgz_Name         = "uz_sgz";

/// duxdx variable name
const char * const  duxdx_Name          = "duxdx";
/// duydy variable name
const char * const  duydy_Name          = "duydy";
/// duzdz variable name
const char * const  duzdz_Name          = "duzdz";

/// dxudxn variable name
const char * const  dxudxn_Name         = "dxudxn";
/// dyudyn variable name
const char * const  dyudyn_Name         = "dyudyn";
/// dzudzn variable name
const char * const  dzudzn_Name         = "dzudzn";

/// dxudxn_sgx variable name
const char * const  dxudxn_sgx_Name     = "dxudxn_sgx";
/// dyudyn_sgy variable name
const char * const  dyudyn_sgy_Name     = "dyudyn_sgy";
/// dzudzn_sgz variable name
const char * const  dzudzn_sgz_Name     = "dzudzn_sgz";

/// ddx_k_shift_pos_r variable name
const char * const  ddx_k_shift_pos_r_Name = "ddx_k_shift_pos_r";
/// ddy_k_shift_pos variable name
const char * const  ddy_k_shift_pos_Name   = "ddy_k_shift_pos";
/// ddz_k_shift_pos variable name
const char * const  ddz_k_shift_pos_Name   = "ddz_k_shift_pos";

/// ddx_k_shift_neg_r variable name
const char * const  ddx_k_shift_neg_r_Name = "ddx_k_shift_neg_r";
/// ddy_k_shift_neg variable name
const char * const  ddy_k_shift_neg_Name   = "ddy_k_shift_neg";
/// ddz_k_shift_neg variable name
const char * const  ddz_k_shift_neg_Name   = "ddz_k_shift_neg";

/// rho0 variable name
const char * const  rho0_Name           = "rho0";
/// rho0_sgx variable name
const char * const  rho0_sgx_Name       = "rho0_sgx";
/// rho0_sgy variable name
const char * const  rho0_sgy_Name       = "rho0_sgy";
/// rho0_sgz variable name
const char * const  rho0_sgz_Name       = "rho0_sgz";

/// absorb_tau variable name
const char * const  absorb_tau_Name     = "absorb_tau";
/// absorb_eta variable name
const char * const  absorb_eta_Name     = "absorb_eta";
/// absorb_nabla1_r variable name
const char * const  absorb_nabla1_r_Name= "absorb_nabla1_r";
/// absorb_nabla2_r variable name
const char * const  absorb_nabla2_r_Name= "absorb_nabla2_r";

/// p_rms variable name
const char * const  p_rms_Name  = "p_rms";
/// p_max variable name
const char * const  p_max_Name  = "p_max";
/// p_min variable name
const char * const  p_min_Name  = "p_min";
/// p_max_all variable name
const char * const  p_max_all_Name  = "p_max_all";
/// p_min_all variable name
const char * const  p_min_all_Name  = "p_min_all";
/// p_final variable name
const char * const  p_final_Name= "p_final";

/// ux_rms variable name
const char * const  ux_rms_Name = "ux_rms";
/// uy_rms variable name
const char * const  uy_rms_Name = "uy_rms";
/// uz_rms variable name
const char * const  uz_rms_Name = "uz_rms";

/// ux_max variable name
const char * const  ux_max_Name = "ux_max";
/// uy_max variable name
const char * const  uy_max_Name = "uy_max";
/// uz_max variable name
const char * const  uz_max_Name = "uz_max";
/// ux_min variable name
const char * const  ux_min_Name = "ux_min";
/// uy_min variable name
const char * const  uy_min_Name = "uy_min";
/// uz_min variable name
const char * const  uz_min_Name = "uz_min";

/// ux_max_all variable name
const char * const  ux_max_all_Name = "ux_max_all";
/// uy_max_all variable name
const char * const  uy_max_all_Name = "uy_max_all";
/// uz_max_all variable name
const char * const  uz_max_all_Name = "uz_max_all";
/// ux_min_all variable name
const char * const  ux_min_all_Name = "ux_min_all";
/// uy_min_all variable name
const char * const  uy_min_all_Name = "uy_min_all";
/// uz_min_all variable name
const char * const  uz_min_all_Name = "uz_min_all";

/// ux_final variable name
const char * const  ux_final_Name = "ux_final";
/// uy_final variable name
const char * const  uy_final_Name = "uy_final";
/// uz_final variable name
const char * const  uz_final_Name = "uz_final";

/// Ix_avg variable name
const char * const  Ix_avg_Name = "Ix_avg";
/// Iy_avg variable name
const char * const  Iy_avg_Name = "Iy_avg";
/// Iz_avg variable name
const char * const  Iz_avg_Name = "Iz_avg";

/// Ix_max variable name
const char * const  Ix_max_Name = "Ix_max";
/// Iy_max variable name
const char * const  Iy_max_Name = "Iy_max";
/// Iz_max variable name
const char * const  Iz_max_Name = "Iz_max";

/// Temp_1_RS3D variable name
const char * const Temp_1_RS3D_Name = "Temp_1_RS3D";
/// Temp_2_RS3D variable name
const char * const Temp_2_RS3D_Name = "Temp_2_RS3D";
/// Temp_3_RS3D variable name
const char * const Temp_3_RS3D_Name = "Temp_3_RS3D";


/// CUFFT_shift_temp variable name
const char * const CUFFT_shift_temp_Name = "CUFFT_shift_temp";
/// CUFFT_X_temp variable name
const char * const CUFFT_X_temp_Name     = "CUFFT_X_temp";
/// CUFFT_Y_temp variable name
const char * const CUFFT_Y_temp_Name     = "CUFFT_Y_temp";
/// CUFFT_Z_temp variable name
const char * const CUFFT_Z_temp_Name     = "CUFFT_Z_temp";

#endif	/* MATRIXNAMES_H */

