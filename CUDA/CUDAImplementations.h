/**
 * @file        CUDAImplementations.h
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the all CUDA kernels
 *              for the GPU implementation
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        11 March    2013, 13:10 (created) \n
 *              04 November 2014, 14:47 (revised)
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

#ifndef __kwave_opencl__CUDAImplementations__
#define __kwave_opencl__CUDAImplementations__

#include <iostream>

#include "../MatrixClasses/RealMatrix.h"
#include "../MatrixClasses/ComplexMatrix.h"
#include "../MatrixClasses/LongMatrix.h"
#include "../Utils/DimensionSizes.h"

#include "MatrixClasses/CUFFTComplexMatrix.h"
#include "CUDATuner.h"

class CUDAImplementations{

    private:
        //private variables required for instance variable
        static bool instance_flag;
        static CUDAImplementations *single_cuda_implementation;
        CUDATuner* tuner;

        //functions
        CUDAImplementations();

    public:
        //new data types
        struct device_constants_t{
            size_t max_x;
            size_t max_y;
            size_t max_z;
            size_t total_element_count;
            float  divider;
            size_t complex_max_x;
            size_t complex_max_y;
            size_t complex_max_z;
            size_t complex_total_element_count;
        };

        //access instance (there can be only one)
        static CUDAImplementations* GetInstance();
        ~CUDAImplementations();

        //setup functions
        void SetUpExecutionModelWithTuner(size_t x,
                                          size_t y,
                                          size_t z);

        void SetUpDeviceConstants(size_t max_x,
                                  size_t max_y,
                                  size_t max_z,
                                  size_t complex_max_x,
                                  size_t complex_max_y,
                                  size_t complex_max_z);

        //computational functions
        void ExtractDataFromPressureMatrix(TRealMatrix& SourceMatrix,
                                           TLongMatrix& Index,
                                           TRealMatrix& TempBuffer);

        void ScalarDividedBy(TRealMatrix& matrix,
                             const float scalar);

        void ZeroMatrix(TRealMatrix& matrix);

        void Compute_ux_sgx_normalize(TRealMatrix& uxyz_sgxyz,
                                      TRealMatrix& FFT_p,
                                      TRealMatrix& dt_rho0,
                                      TRealMatrix& pml);

        void Compute_ux_sgx_normalize_scalar_uniform(TRealMatrix& uxyz_sgxyz,
                                                     TRealMatrix& FFT_p,
                                                     float dt_rho0,
                                                     TRealMatrix& pml);

        void Compute_ux_sgx_normalize_scalar_nonuniform(
                TRealMatrix& uxyz_sgxyz,
                TRealMatrix& FFT_p,
                float dt_rho0,
                TRealMatrix& dxudxn_sgx,
                TRealMatrix& pml);

        void Compute_uy_sgy_normalize(TRealMatrix& uxyz_sgxyz,
                                      TRealMatrix& FFT_p,
                                      TRealMatrix& dt_rho0,
                                      TRealMatrix& pml);

        void Compute_uy_sgy_normalize_scalar_uniform(TRealMatrix& uxyz_sgxyz,
                                                     TRealMatrix& FFT_p,
                                                     float dt_rho0,
                                                     TRealMatrix& pml);

        void Compute_uy_sgy_normalize_scalar_nonuniform(
                TRealMatrix& uxyz_sgxyz,
                TRealMatrix& FFT_p,
                float dt_rho0,
                TRealMatrix& dyudyn_sgy,
                TRealMatrix& pml);

        void Compute_uz_sgz_normalize(TRealMatrix& uxyz_sgxyz,
                TRealMatrix& FFT_p,
                TRealMatrix& dt_rho0,
                TRealMatrix& pml);

        void Compute_uz_sgz_normalize_scalar_uniform(TRealMatrix& uxyz_sgxyz,
                                                     TRealMatrix& FFT_p,
                                                     float dt_rho0,
                                                     TRealMatrix& pml);

        void Compute_uz_sgz_normalize_scalar_nonuniform(
                TRealMatrix& uxyz_sgxyz,
                TRealMatrix& FFT_p,
                float dt_rho0,
                TRealMatrix& dzudzn_sgz,
                TRealMatrix& pml);

        void AddTransducerSource(TRealMatrix& uxyz_sgxyz,
                                 TLongMatrix& us_index,
                                 TLongMatrix& delay_mask,
                                 TRealMatrix& transducer_signal);

        void Add_u_source(TRealMatrix& uxyz_sgxyz,
                          TRealMatrix& u_source_input,
                          TLongMatrix& us_index,
                          int t_index,
                          size_t u_source_mode,
                          size_t u_source_many);

        /*
         * uxyz_sgxyzMatrix functions (fourier)
         */
        void Compute_dt_rho_sg_mul_ifft_div_2(
                TRealMatrix& uxyz_sgxyz,
                TRealMatrix& dt_rho0_sg,
                TCUFFTComplexMatrix& FFT);

        void Compute_dt_rho_sg_mul_ifft_div_2(
                TRealMatrix& uxyz_sgxyz,
                float dt_rho_0_sgx,
                TCUFFTComplexMatrix& FFT);

        void Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_x(
                TRealMatrix& uxyz_sgxyz,
                float dt_rho_0_sgx,
                TRealMatrix& dxudxn_sgx,
                TCUFFTComplexMatrix& FFT);

        void Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_y(
                TRealMatrix& uxyz_sgxyz,
                float dt_rho_0_sgy,
                TRealMatrix& dyudyn_sgy,
                TCUFFTComplexMatrix& FFT);

        void Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_z(
                TRealMatrix& uxyz_sgxyz,
                float dt_rho_0_sgz,
                TRealMatrix& dzudzn_sgz,
                TCUFFTComplexMatrix& FFT);

        /*
         * KSpaceFirstOrder3DSolver functions
         */
        void Compute_ddx_kappa_fft_p(TRealMatrix& X_Matrix,
                                     TCUFFTComplexMatrix& FFT_X,
                                     TCUFFTComplexMatrix& FFT_Y,
                                     TCUFFTComplexMatrix& FFT_Z,
                                     TRealMatrix& kappa,
                                     TComplexMatrix & ddx,
                                     TComplexMatrix & ddy,
                                     TComplexMatrix & ddz);

        void Compute_duxyz_initial(TCUFFTComplexMatrix& Temp_FFT_X,
                                   TCUFFTComplexMatrix& Temp_FFT_Y,
                                   TCUFFTComplexMatrix& Temp_FFT_Z,
                                   TRealMatrix& kappa,
                                   TRealMatrix& ux_sgx,
                                   TComplexMatrix& ddx_k_shift_neg,
                                   TComplexMatrix& ddy_k_shift_neg,
                                   TComplexMatrix& ddz_k_shift_neg);

        void Compute_duxyz_non_linear(TRealMatrix& duxdx,
                                      TRealMatrix& duydy,
                                      TRealMatrix& duzdz,
                                      TRealMatrix& dxudxn,
                                      TRealMatrix& dyudyn,
                                      TRealMatrix& dzudzn);

        void ComputeC2_matrix(TRealMatrix& c2);

        void Calculate_p0_source_add_initial_pressure(TRealMatrix& rhox,
                                                      TRealMatrix& rhoy,
                                                      TRealMatrix& rhoz,
                                                      TRealMatrix& p0,
                                                      size_t c2_shift,
                                                      float* c2);

        void Compute_rhoxyz_nonlinear_scalar(TRealMatrix& rhox,
                                             TRealMatrix& rhoy,
                                             TRealMatrix& rhoz,
                                             TRealMatrix& pml_x,
                                             TRealMatrix& pml_y,
                                             TRealMatrix& pml_z,
                                             TRealMatrix& duxdx,
                                             TRealMatrix& duydy,
                                             TRealMatrix& duzdz,
                                             float dt_el,
                                             float rho0_scalar);

        void Compute_rhoxyz_nonlinear_matrix(TRealMatrix& rhox,
                                             TRealMatrix& rhoy,
                                             TRealMatrix& rhoz,
                                             TRealMatrix& pml_x,
                                             TRealMatrix& pml_y,
                                             TRealMatrix& pml_z,
                                             TRealMatrix& duxdx,
                                             TRealMatrix& duydy,
                                             TRealMatrix& duzdz,
                                             float dt_el,
                                             TRealMatrix& rho0);

        void Compute_rhoxyz_linear_scalar(TRealMatrix& rhox,
                                          TRealMatrix& rhoy,
                                          TRealMatrix& rhoz,
                                          TRealMatrix& pml_x,
                                          TRealMatrix& pml_y,
                                          TRealMatrix& pml_z,
                                          TRealMatrix& duxdx,
                                          TRealMatrix& duydy,
                                          TRealMatrix& duzdz,
                                          const float dt_el,
                                          const float rho0_scalar);

        void Compute_rhoxyz_linear_matrix(TRealMatrix& rhox,
                                          TRealMatrix& rhoy,
                                          TRealMatrix& rhoz,
                                          TRealMatrix& pml_x,
                                          TRealMatrix& pml_y,
                                          TRealMatrix& pml_z,
                                          TRealMatrix& duxdx,
                                          TRealMatrix& duydy,
                                          TRealMatrix& duzdz,
                                          const float dt_el,
                                          TRealMatrix& rho0);

        void Add_p_source(TRealMatrix& rhox,
                          TRealMatrix& rhoy,
                          TRealMatrix& rhoz,
                          TRealMatrix& p_source_input,
                          TLongMatrix& p_source_index,
                          size_t p_source_many,
                          size_t p_source_mode,
                          size_t t_index);

        void Calculate_SumRho_BonA_SumDu(TRealMatrix& RHO_Temp,
                                         TRealMatrix& BonA_Temp,
                                         TRealMatrix& Sum_du,
                                         TRealMatrix& rhox,
                                         TRealMatrix& rhoy,
                                         TRealMatrix& rhoz,
                                         TRealMatrix& duxdx,
                                         TRealMatrix& duydy,
                                         TRealMatrix& duzdz,
                                         const float  BonA_data_scalar,
                                         const float* BonA_data_matrix,
                                         const size_t BonA_shift,
                                         const float  rho0_data_scalar,
                                         const float* rho0_data_matrix,
                                         const size_t rho0_shift);

        void Compute_Absorb_nabla1_2(TRealMatrix& absorb_nabla1,
                                     TRealMatrix& absorb_nabla2,
                                     TCUFFTComplexMatrix& FFT_1,
                                     TCUFFTComplexMatrix& FFT_2);

        void Sum_Subterms_nonlinear(TRealMatrix& BonA_temp,
                                    TRealMatrix& p,
                                    const float  c2_data_scalar,
                                    const float* c2_data_matrix,
                                    const size_t c2_shift,
                                    const float* Absorb_tau_data,
                                    const float  tau_data_scalar,
                                    const float* tau_data_matrix,
                                    const float* Absorb_eta_data,
                                    const float  eta_data_scalar,
                                    const float* eta_data_matrix,
                                    const size_t tau_eta_shift);

        void Sum_new_p_nonlinear_lossless(const size_t TotalElementCount,
                                          TRealMatrix& p,
                                          const float* rhox_data,
                                          const float* rhoy_data,
                                          const float* rhoz_data,
                                          const float  c2_data_scalar,
                                          const float* c2_data_matrix,
                                          const size_t c2_shift,
                                          const float  BonA_data_scalar,
                                          const float* BonA_data_matrix,
                                          const size_t BonA_shift,
                                          const float  rho0_data_scalar,
                                          const float* rho0_data_matrix,
                                          const size_t rho0_shift);

        void Calculate_SumRho_SumRhoDu(TRealMatrix& Sum_rhoxyz,
                                       TRealMatrix& Sum_rho0_du,
                                       TRealMatrix& rhox,
                                       TRealMatrix& rhoy,
                                       TRealMatrix& rhoz,
                                       TRealMatrix& duxdx,
                                       TRealMatrix& duydy,
                                       TRealMatrix& duzdz,
                                       const float* rho0_data,
                                       const float  rho0_data_el,
                                       const size_t TotalElementCount,
                                       const bool   rho0_scalar_flag);

        void Sum_Subterms_linear(TRealMatrix& Absorb_tau_temp,
                                 TRealMatrix& Absorb_eta_temp,
                                 TRealMatrix& Sum_rhoxyz,
                                 TRealMatrix& p,
                                 const size_t total_element_count,
                                 const size_t c2_shift,
                                 const size_t tau_eta_shift,
                                 const float  tau_data_scalar,
                                 const float* tau_data_matrix,
                                 const float  eta_data_scalar,
                                 const float* eta_data_matrix,
                                 const float  c2_data_scalar,
                                 const float* c2_data_matrix);

        void Sum_new_p_linear_lossless_scalar(TRealMatrix& p,
                                              TRealMatrix& rhox,
                                              TRealMatrix& rhoy,
                                              TRealMatrix& rhoz,
                                              const size_t total_element_count,
                                              const float c2_element);
        void Sum_new_p_linear_lossless_matrix(TRealMatrix& p,
                                              TRealMatrix& rhox,
                                              TRealMatrix& rhoy,
                                              TRealMatrix& rhoz,
                                              const size_t total_element_count,
                                              TRealMatrix& c2);

        void StoreSensorData_store_p_max(TRealMatrix& p,
                                         TRealMatrix& p_sensor_max,
                                         TLongMatrix& sensor_mask_index);

        void StoreSensorData_store_p_rms(TRealMatrix& p,
                                         TRealMatrix& p_sensor_rms,
                                         TLongMatrix& sensor_mask_index);

        void StoreSensorData_store_u_max(TRealMatrix& ux_sgx,
                                         TRealMatrix& uy_sgy,
                                         TRealMatrix& uz_sgz,
                                         TRealMatrix& ux_sensor_max,
                                         TRealMatrix& uy_sensor_max,
                                         TRealMatrix& uz_sensor_max,
                                         TLongMatrix& sensor_mask_index);

        void StoreSensorData_store_u_rms(TRealMatrix& ux_sgx,
                                         TRealMatrix& uy_sgy,
                                         TRealMatrix& uz_sgz,
                                         TRealMatrix& ux_sensor_rms,
                                         TRealMatrix& uy_sensor_rms,
                                         TRealMatrix& uz_sensor_rms,
                                         TLongMatrix& sensor_mask_index);

        void StoreIntensityData_first_step(const size_t sensor_size,
                                           const TDimensionSizes Dims,
                                           const size_t * index,
                                           const float* ux,
                                           const float* uy,
                                           const float* uz,
                                           const float* p,
                                           float * ux_i_1,
                                           float * uy_i_1,
                                           float * uz_i_1,
                                           float * p_i_1);

        void StoreIntensityData_other_step(const size_t sensor_size,
                                           const TDimensionSizes Dims,
                                           const size_t * index,
                                           const float* ux,
                                           const float* uy,
                                           const float* uz,
                                           const float* p,
                                           float* ux_i_1,
                                           float* uy_i_1,
                                           float* uz_i_1,
                                           float* p_i_1,
                                           float* Ix_avg,
                                           float* Iy_avg,
                                           float* Iz_avg,
                                           float* Ix_max,
                                           float* Iy_max,
                                           float* Iz_max,
                                           bool store_I_avg,
                                           bool store_I_max,
                                           size_t Nt,
                                           size_t start_time_index);

};

#endif /* defined(__kwave_opencl__CImplementations__) */
