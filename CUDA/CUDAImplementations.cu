/**
 * @file        CUDAImplementations.cpp
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the all CUDA kernels
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

#include "CUDAImplementations.h"

__constant__ CUDAImplementations::device_constants_t constants;

//singleton stuff
bool CUDAImplementations::instance_flag = false;
CUDAImplementations* CUDAImplementations::single_cuda_implementation = NULL;

CUDAImplementations::CUDAImplementations(){

}

CUDAImplementations* CUDAImplementations::GetInstance()
{
    if(!instance_flag){
        single_cuda_implementation = new CUDAImplementations();
        instance_flag = true;
        return single_cuda_implementation;
    }
    else{
        return single_cuda_implementation;
    }
}

CUDAImplementations::~CUDAImplementations()
{
    delete single_cuda_implementation;
    instance_flag = false;
}

void CUDAImplementations::SetUpExecutionModelWithTuner(const TDimensionSizes & FullDimensionSizes,
                                                       const TDimensionSizes & ReducedDimensionSizes)
{
    tuner = TCUDATuner::GetInstance();

    tuner->GenerateExecutionModelForMatrixSize(FullDimensionSizes, ReducedDimensionSizes);

}

void CUDAImplementations::SetUpDeviceConstants(size_t max_x,
                                               size_t max_y,
                                               size_t max_z,
                                               size_t complex_max_x,
                                               size_t complex_max_y,
                                               size_t complex_max_z)
{
    size_t total_element_count = max_x * max_y * max_z;
    size_t complex_total_element_count = complex_max_x *
                                         complex_max_y * complex_max_z;
    float divider = 1.0f / total_element_count;

    device_constants_t host_constants;

    host_constants.max_x = max_x;
    host_constants.max_y = max_y;
    host_constants.max_z = max_z;
    host_constants.total_element_count = total_element_count;
    host_constants.divider = divider;
    host_constants.complex_max_x = complex_max_x;
    host_constants.complex_max_y = complex_max_y;
    host_constants.complex_max_z = complex_max_z;
    host_constants.complex_total_element_count = complex_total_element_count;

    cudaMemcpyToSymbol(constants,
                       &host_constants,
                       sizeof(device_constants_t));
}

__global__ void CUDAExtractDataFromPressureMatrix(const float*  p_matrix,
                                                        float*  local_data,
                                                  const size_t*   indices_vector,
                                                  const size_t  indices_size)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    while (i < indices_size) {
        local_data[i] = p_matrix[indices_vector[i]];
        i += stride;
    }
}

void CUDAImplementations::ExtractDataFromPressureMatrix(
        TRealMatrix& SourceMatrix,
        TIndexMatrix& Index,
        TRealMatrix& TempBuffer)
{
    const float*  p_matrix = SourceMatrix.GetRawDeviceData();
          float*  local_data = TempBuffer.GetRawDeviceData();
    const size_t*   indices_vector = Index.GetRawDeviceData();
    const size_t  indices_size = Index.GetTotalElementCount();

    CUDAExtractDataFromPressureMatrix
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
    (p_matrix, local_data, indices_vector, indices_size);
}

__global__ void CudaScalarDividedBy(float* matrix,
                                    float scalar,
                                    size_t matrix_size)
{

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    float scalar_reg = scalar;
    while (i < matrix_size) {
        matrix[i] = scalar_reg / matrix[i];
        i += stride;
    }

}

void CUDAImplementations::ScalarDividedBy(TRealMatrix& matrix,
                                          const float scalar)
{
    float* matrix_data = matrix.GetRawDeviceData();
    size_t matrix_size = matrix.GetTotalAllocatedElementCount();

    CudaScalarDividedBy
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
        (matrix_data,
         scalar,
         matrix_size);
}

__global__ void CudaZeroMatrix(float* pMatrixData,
                               size_t pTotalAllocatedElementCount)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    while (i < pTotalAllocatedElementCount){
        pMatrixData[i] = 0.0f;
        i+=stride;
    }
}

void CUDAImplementations::ZeroMatrix(TRealMatrix& matrix)
{
    float* matrix_data = matrix.GetRawDeviceData();
    size_t matrix_size = matrix.GetTotalAllocatedElementCount();

    CudaZeroMatrix
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
            (matrix_data,
             matrix_size);
}

__global__ void CudaCompute_ux_sgx_normalize(//const size_t max_x,
                                             //const size_t max_y,
                                             //const size_t max_z,
                                                   float* matrix_data,
                                             const float* FFT_p,
                                             const float* dt_rho0,
                                             const float* pml)
{

    const size_t max_x = constants.max_x;
    const size_t max_y = constants.max_y;
    const size_t max_z = constants.max_z;
    const float  divider = constants.divider;

    //const size_t total_element_count = max_x * max_y * max_z;
    //const float divider = 1.0f / total_element_count;

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t x_stride = blockDim.x * gridDim.x;
    size_t y_stride = blockDim.y * gridDim.y;
    size_t z_stride = blockDim.z * gridDim.z;

    while(z < max_z){
        while(y < max_y){
            while(x < max_x){
                int i = z*max_x*max_y + y*max_x + x;

                register float matrix_element = matrix_data[i];
                const float FFT_p_el = divider * FFT_p[i] * dt_rho0[i];

                matrix_element *= pml[x];
                matrix_element -= FFT_p_el;
                matrix_data[i] = matrix_element * pml[x];
                x+=x_stride;
            }
            y+=y_stride;
        }
        z+=z_stride;
    }
}

void CUDAImplementations::Compute_ux_sgx_normalize(TRealMatrix& uxyz_sgxyz,
                                                   TRealMatrix& FFT_p,
                                                   TRealMatrix& dt_rho0,
                                                   TRealMatrix& pml)
{

    //const size_t max_x = uxyz_sgxyz.GetDimensionSizes().X;
    //const size_t max_y = uxyz_sgxyz.GetDimensionSizes().Y;
    //const size_t max_z = uxyz_sgxyz.GetDimensionSizes().Z;
          float* matrix_data  = uxyz_sgxyz.GetRawDeviceData();
    const float* FFT_p_data   = FFT_p.GetRawDeviceData();
    const float* dt_rho0_data = dt_rho0.GetRawDeviceData();
    const float* pml_data     = pml.GetRawDeviceData();

    CudaCompute_ux_sgx_normalize
        <<< tuner->GetNumberOfBlocksFor3D(),
            tuner->GetNumberOfThreadsFor3D() >>>
            (//max_x,
             //max_y,
             //max_z,
             matrix_data,
             FFT_p_data,
             dt_rho0_data,
             pml_data);
}

__global__ void CudaCompute_ux_sgx_normalize_scalar_uniform(
        //const size_t max_x,
        //const size_t max_y,
        //const size_t max_z,
        float*       matrix_data,
        const float* FFT_p,
        const float  dt_rho0,
        const float* pml)
{
    const size_t max_x = constants.max_x;
    const size_t max_y = constants.max_y;
    const size_t max_z = constants.max_z;
    const size_t matrix_size = constants.total_element_count;

    //size_t matrix_size = max_x*max_y*max_z;

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t x_stride = blockDim.x * gridDim.x;
    size_t y_stride = blockDim.y * gridDim.y;
    size_t z_stride = blockDim.z * gridDim.z;

    const float divider = dt_rho0 / matrix_size;

    for(; z < max_z; z+=z_stride){
        for(; y < max_y; y+=y_stride){
            for(; x < max_x; x+=x_stride){
                register size_t i = z*max_y*max_x + y*max_x + x;
                register float matrix_element = matrix_data[i];
                const float FFT_p_el = divider * FFT_p[i];
                matrix_element *= pml[x];
                matrix_element -= FFT_p_el;
                matrix_data[i] = matrix_element * pml[x];
            }
        }
    }
}

void CUDAImplementations::Compute_ux_sgx_normalize_scalar_uniform(
        TRealMatrix& uxyz_sgxyz,
        TRealMatrix& FFT_p,
        float dt_rho0,
        TRealMatrix& pml)
{

    //const size_t max_x = uxyz_sgxyz.GetDimensionSizes().X;
    //const size_t max_y = uxyz_sgxyz.GetDimensionSizes().Y;
    //const size_t max_z = uxyz_sgxyz.GetDimensionSizes().Z;
    float* matrix_data = uxyz_sgxyz.GetRawDeviceData();
    const float* FFT_p_data = FFT_p.GetRawDeviceData();
    const float* pml_data = pml.GetRawDeviceData();

    CudaCompute_ux_sgx_normalize_scalar_uniform
        <<< tuner->GetNumberOfBlocksFor3D(),
            tuner->GetNumberOfThreadsFor3D() >>>
        (//max_x,
         //max_y,
         //max_z,
         matrix_data,
         FFT_p_data,
         dt_rho0,
         pml_data);
}

__global__ void CudaCompute_ux_sgx_normalize_scalar_nonuniform(
        //const size_t max_x,
        //const size_t max_y,
        //const size_t max_z,
        float* matrix_data,
        const float* FFT_p,
        const float dt_rho0,
        const float* dxudxn_sgx,
        const float* pml)
{

    const size_t max_x = constants.max_x;
    const size_t max_y = constants.max_y;
    const size_t max_z = constants.max_z;
    const size_t matrix_size = constants.total_element_count;

    //size_t matrix_size = max_x*max_y*max_z; //64

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t x_stride = blockDim.x * gridDim.x;
    size_t y_stride = blockDim.y * gridDim.y;
    size_t z_stride = blockDim.z * gridDim.z;

    const float divider = dt_rho0 / matrix_size;

    for(; z < max_z; z+=z_stride){
        for(; y < max_y; y+=y_stride){
            for(; x < max_x; x+=x_stride){
                register size_t i = z*max_y*max_x + y*max_x + x;
                register float matrix_element = matrix_data[i];
                const float FFT_p_el = (divider * dxudxn_sgx[x]) * FFT_p[i];
                matrix_element *= pml[x];
                matrix_element -= FFT_p_el;
                matrix_data[i] = matrix_element * pml[x];
            }
        }
    }
}

void CUDAImplementations::Compute_ux_sgx_normalize_scalar_nonuniform(
        TRealMatrix& uxyz_sgxyz,
        TRealMatrix& FFT_p,
        float dt_rho0,
        TRealMatrix& dxudxn_sgx,
        TRealMatrix& pml)
{

    //const size_t max_x = uxyz_sgxyz.GetDimensionSizes().X;
    //const size_t max_y = uxyz_sgxyz.GetDimensionSizes().Y;
    //const size_t max_z = uxyz_sgxyz.GetDimensionSizes().Z;
    float* matrix_data = uxyz_sgxyz.GetRawDeviceData();
    const float* FFT_p_data = FFT_p.GetRawDeviceData();
    const float* dxudxn_sgx_data = dxudxn_sgx.GetRawDeviceData();
    const float* pml_data = pml.GetRawDeviceData();

    CudaCompute_ux_sgx_normalize_scalar_nonuniform
        <<< tuner->GetNumberOfBlocksFor3D(),
            tuner->GetNumberOfThreadsFor3D() >>>
        (//max_x,
         //max_y,
         //max_z,
         matrix_data,
         FFT_p_data,
         dt_rho0,
         dxudxn_sgx_data,
         pml_data);
}

__global__ void CudaCompute_uy_sgy_normalize(//const size_t max_x,
                                             //const size_t max_y,
                                             //const size_t max_z,
                                                   float* matrix_data,
                                             const float* FFT_p,
                                             const float* dt_rho0,
                                             const float* pml)
{

    const size_t max_x = constants.max_x;
    const size_t max_y = constants.max_y;
    const size_t max_z = constants.max_z;
    const float  divider = constants.divider;

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t x_stride = blockDim.x * gridDim.x;
    size_t y_stride = blockDim.y * gridDim.y;
    size_t z_stride = blockDim.z * gridDim.z;

    //size_t total_element_count = max_x*max_y*max_z;
    //const float divider = 1.0f / total_element_count;

    while(z < max_z){
        while(y < max_y){
            register float pml_y = pml[y];
            while(x < max_x){
                int i = z*max_x*max_y + y*max_x + x;

                register float matrix_element = matrix_data[i];
                const float FFT_p_el = divider * FFT_p[i] * dt_rho0[i];

                matrix_element *= pml_y;
                matrix_element -= FFT_p_el;
                matrix_data[i] = matrix_element * pml_y;
                x+=x_stride;
            }
            y+=y_stride;
        }
        z+=z_stride;
    }
}

void CUDAImplementations::Compute_uy_sgy_normalize(TRealMatrix& uxyz_sgxyz,
                                                   TRealMatrix& FFT_p,
                                                   TRealMatrix& dt_rho0,
                                                   TRealMatrix& pml)
{

    //const size_t max_x = uxyz_sgxyz.GetDimensionSizes().X;
    //const size_t max_y = uxyz_sgxyz.GetDimensionSizes().Y;
    //const size_t max_z = uxyz_sgxyz.GetDimensionSizes().Z;
          float* matrix_data = uxyz_sgxyz.GetRawDeviceData();
    const float* FFT_p_data = FFT_p.GetRawDeviceData();
    const float* dt_rho0_data = dt_rho0.GetRawDeviceData();
    const float* pml_data = pml.GetRawDeviceData();

    CudaCompute_uy_sgy_normalize
        <<< tuner->GetNumberOfBlocksFor3D(),
            tuner->GetNumberOfThreadsFor3D() >>>
        (//max_x,
         //max_y,
         //max_z,
         matrix_data,
         FFT_p_data,
         dt_rho0_data,
         pml_data);
}

__global__ void CudaCompute_uy_sgy_normalize_scalar_uniform(
        //const size_t max_x,
        //const size_t max_y,
        //const size_t max_z,
              float* matrix_data,
        const float* FFT_p,
        const float dt_rho0,
        const float* pml)
{

    const size_t max_x = constants.max_x;
    const size_t max_y = constants.max_y;
    const size_t max_z = constants.max_z;
    const size_t matrix_size = constants.total_element_count;

    //size_t matrix_size = max_x*max_y*max_z;

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t x_stride = blockDim.x * gridDim.x;
    size_t y_stride = blockDim.y * gridDim.y;
    size_t z_stride = blockDim.z * gridDim.z;

    const float divider = dt_rho0 / matrix_size;

    for(; z < max_z; z+=z_stride){
        for(; y < max_y; y+=y_stride){
            register float pml_y = pml[y];
            for(; x < max_x; x+=x_stride){
                register size_t i = z*max_y*max_x + y*max_x + x;
                register float matrix_element = matrix_data[i];
                const float FFT_p_el = divider * FFT_p[i];
                matrix_element *= pml_y;
                matrix_element -= FFT_p_el;
                matrix_data[i] = matrix_element * pml_y;
            }
        }
    }
}

void CUDAImplementations::Compute_uy_sgy_normalize_scalar_uniform(
        TRealMatrix& uxyz_sgxyz,
        TRealMatrix& FFT_p,
        float dt_rho0,
        TRealMatrix& pml)
{

    //const size_t max_x = uxyz_sgxyz.GetDimensionSizes().X;
    //const size_t max_y = uxyz_sgxyz.GetDimensionSizes().Y;
    //const size_t max_z = uxyz_sgxyz.GetDimensionSizes().Z;
          float* matrix_data = uxyz_sgxyz.GetRawDeviceData();
    const float* FFT_p_data = FFT_p.GetRawDeviceData();
    const float* pml_data = pml.GetRawDeviceData();

    CudaCompute_uy_sgy_normalize_scalar_uniform
        <<< tuner->GetNumberOfBlocksFor3D(),
            tuner->GetNumberOfThreadsFor3D() >>>
        (//max_x,
         //max_y,
         //max_z,
         matrix_data,
         FFT_p_data,
         dt_rho0,
         pml_data);
}

__global__ void CudaCompute_uy_sgy_normalize_scalar_nonuniform(
        //const size_t max_x,
        //const size_t max_y,
        //const size_t max_z,
              float* matrix_data,
        const float* FFT_p,
        const float dt_rho0,
        const float* dyudyn_sgy,
        const float* pml)
{

    const size_t max_x = constants.max_x;
    const size_t max_y = constants.max_y;
    const size_t max_z = constants.max_z;
    const size_t matrix_size = constants.total_element_count;

    //size_t matrix_size = max_x*max_y*max_z; //64

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t x_stride = blockDim.x * gridDim.x;
    size_t y_stride = blockDim.y * gridDim.y;
    size_t z_stride = blockDim.z * gridDim.z;

    const float divider = dt_rho0 / matrix_size;

    for(; z < max_z; z+=z_stride){
        for(; y < max_y; y+=y_stride){
            register float pml_y = pml[y];
            register float dyudyn_sgy_y = dyudyn_sgy[y];
            for(; x < max_x; x+=x_stride){
                register size_t i = z*max_y*max_x + y*max_x + x;
                register float matrix_element = matrix_data[i];
                const float FFT_p_el = (divider * dyudyn_sgy_y) * FFT_p[i];
                matrix_element *= pml_y;
                matrix_element -= FFT_p_el;
                matrix_data[i] = matrix_element * pml_y;
            }
        }
    }
}

void CUDAImplementations::Compute_uy_sgy_normalize_scalar_nonuniform(
        TRealMatrix& uxyz_sgxyz,
        TRealMatrix& FFT_p,
        float dt_rho0,
        TRealMatrix& dyudyn_sgy,
        TRealMatrix& pml)
{

    //const size_t max_x = uxyz_sgxyz.GetDimensionSizes().X;
    //const size_t max_y = uxyz_sgxyz.GetDimensionSizes().Y;
    //const size_t max_z = uxyz_sgxyz.GetDimensionSizes().Z;
          float* matrix_data = uxyz_sgxyz.GetRawDeviceData();
    const float* FFT_p_data = FFT_p.GetRawDeviceData();
    const float* dyudyn_sgy_data = dyudyn_sgy.GetRawDeviceData();
    const float* pml_data = pml.GetRawDeviceData();

    CudaCompute_uy_sgy_normalize_scalar_nonuniform
        <<< tuner->GetNumberOfBlocksFor3D(),
            tuner->GetNumberOfThreadsFor3D() >>>
        (//max_x,
         //max_y,
         //max_z,
         matrix_data,
         FFT_p_data,
         dt_rho0,
         dyudyn_sgy_data,
         pml_data);
}


__global__ void CudaCompute_uz_sgz_normalize(//const size_t max_x,
                                             //const size_t max_y,
                                             //const size_t max_z,
                                                   float* matrix_data,
                                             const float* FFT_p,
                                             const float* dt_rho0,
                                             const float* pml)
{

    const size_t max_x = constants.max_x;
    const size_t max_y = constants.max_y;
    const size_t max_z = constants.max_z;
    const float  divider = constants.divider;

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t x_stride = blockDim.x * gridDim.x;
    size_t y_stride = blockDim.y * gridDim.y;
    size_t z_stride = blockDim.z * gridDim.z;

    //size_t total_element_count = max_x*max_y*max_z;
    //const float divider = 1.0f / total_element_count;

    while(z < max_z){
        register float pml_z = pml[z];
        while(y < max_y){
            while(x < max_x){
                int i = z*max_x*max_y + y*max_x + x;

                register float matrix_element = matrix_data[i];
                const float FFT_p_el = divider * FFT_p[i] * dt_rho0[i];

                matrix_element *= pml_z;
                matrix_element -= FFT_p_el;
                matrix_data[i] = matrix_element * pml_z;
                x+=x_stride;
            }
            y+=y_stride;
        }
        z+=z_stride;
    }
}

void CUDAImplementations::Compute_uz_sgz_normalize(TRealMatrix& uxyz_sgxyz,
                                                   TRealMatrix& FFT_p,
                                                   TRealMatrix& dt_rho0,
                                                   TRealMatrix& pml)
{

    //const size_t max_x = uxyz_sgxyz.GetDimensionSizes().X;
    //const size_t max_y = uxyz_sgxyz.GetDimensionSizes().Y;
    //const size_t max_z = uxyz_sgxyz.GetDimensionSizes().Z;
          float* matrix_data = uxyz_sgxyz.GetRawDeviceData();
    const float* FFT_p_data = FFT_p.GetRawDeviceData();
    const float* dt_rho0_data = dt_rho0.GetRawDeviceData();
    const float* pml_data = pml.GetRawDeviceData();

    CudaCompute_uz_sgz_normalize
        <<< tuner->GetNumberOfBlocksFor3D(),
            tuner->GetNumberOfThreadsFor3D() >>>
        (//max_x,
         //max_y,
         //max_z,
         matrix_data,
         FFT_p_data,
         dt_rho0_data,
         pml_data);
}

__global__ void CudaCompute_uz_sgz_normalize_scalar_uniform(
        //const size_t max_x,
        //const size_t max_y,
        //const size_t max_z,
              float* matrix_data,
        const float* FFT_p,
        const float dt_rho0,
        const float* pml)
{

    const size_t max_x = constants.max_x;
    const size_t max_y = constants.max_y;
    const size_t max_z = constants.max_z;
    const size_t matrix_size = constants.total_element_count;

    //size_t matrix_size = max_x*max_y*max_z;

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t x_stride = blockDim.x * gridDim.x;
    size_t y_stride = blockDim.y * gridDim.y;
    size_t z_stride = blockDim.z * gridDim.z;

    const float divider = dt_rho0 / matrix_size;

    for(; z < max_z; z+=z_stride){
        register float pml_z = pml[z];
        for(; y < max_y; y+=y_stride){
            for(; x < max_x; x+=x_stride){
                register size_t i = z*max_y*max_x + y*max_x + x;
                register float matrix_element = matrix_data[i];
                const float FFT_p_el = divider * FFT_p[i];
                matrix_element *= pml_z;
                matrix_element -= FFT_p_el;
                matrix_data[i] = matrix_element * pml_z;
            }
        }
    }
}

void CUDAImplementations::Compute_uz_sgz_normalize_scalar_uniform(
        TRealMatrix& uxyz_sgxyz,
        TRealMatrix& FFT_p,
        float dt_rho0,
        TRealMatrix& pml)
{

    //const size_t max_x = uxyz_sgxyz.GetDimensionSizes().X;
    //const size_t max_y = uxyz_sgxyz.GetDimensionSizes().Y;
    //const size_t max_z = uxyz_sgxyz.GetDimensionSizes().Z;
          float* matrix_data = uxyz_sgxyz.GetRawDeviceData();
    const float* FFT_p_data = FFT_p.GetRawDeviceData();
    const float* pml_data = pml.GetRawDeviceData();

    CudaCompute_uz_sgz_normalize_scalar_uniform
        <<< tuner->GetNumberOfBlocksFor3D(),
            tuner->GetNumberOfThreadsFor3D() >>>
        (//max_x,
         //max_y,
         //max_z,
         matrix_data,
         FFT_p_data,
         dt_rho0,
         pml_data);
}

__global__ void CudaCompute_uz_sgz_normalize_scalar_nonuniform(
        //const size_t max_x,
        //const size_t max_y,
        //const size_t max_z,
        float* matrix_data,
        const float* FFT_p,
        const float dt_rho0,
        const float* dzudzn_sgz,
        const float* pml)
{

    const size_t max_x = constants.max_x;
    const size_t max_y = constants.max_y;
    const size_t max_z = constants.max_z;
    const size_t matrix_size = constants.total_element_count;

    //size_t matrix_size = max_x*max_y*max_z;

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t x_stride	= blockDim.x * gridDim.x;
    size_t y_stride	= blockDim.y * gridDim.y;
    size_t z_stride	= blockDim.z * gridDim.z;

    const float divider = dt_rho0 / matrix_size;

    for(; z < max_z; z+=z_stride){
        register float pml_z = pml[z];
        for(; y < max_y; y+=y_stride){
            for(; x < max_x; x+=x_stride){
                register size_t i = z*max_y*max_x + y*max_x + x;
                register float matrix_element = matrix_data[i];
                const float FFT_p_el = (divider * dzudzn_sgz[z]) * FFT_p[i];
                matrix_element *= pml_z;
                matrix_element -= FFT_p_el;
                matrix_data[i] = matrix_element * pml_z;
            }
        }
    }
}

void CUDAImplementations::Compute_uz_sgz_normalize_scalar_nonuniform(
        TRealMatrix& uxyz_sgxyz,
        TRealMatrix& FFT_p,
        float dt_rho0,
        TRealMatrix& dzudzn_sgz,
        TRealMatrix& pml)
{

    //const size_t max_x = uxyz_sgxyz.GetDimensionSizes().X;
    //const size_t max_y = uxyz_sgxyz.GetDimensionSizes().Y;
    //const size_t max_z = uxyz_sgxyz.GetDimensionSizes().Z;
          float* matrix_data = uxyz_sgxyz.GetRawDeviceData();
    const float* FFT_p_data = FFT_p.GetRawDeviceData();
    const float* dzudzn_sgz_data = dzudzn_sgz.GetRawDeviceData();
    const float* pml_data = pml.GetRawDeviceData();

    CudaCompute_uz_sgz_normalize_scalar_nonuniform
        <<< tuner->GetNumberOfBlocksFor3D(),
            tuner->GetNumberOfThreadsFor3D() >>>
        (//max_x,
         //max_y,
         //max_z,
         matrix_data,
         FFT_p_data,
         dt_rho0,
         dzudzn_sgz_data,
         pml_data);
}

__global__ void CudaAddTransducerSource(const size_t us_index_size,
                                              float* ux,
                                        const size_t*  us_index,
                                              size_t*  delay_mask,
                                        const float* transducer_signal)
{

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    while (i < us_index_size) {
        ux[us_index[i]] += transducer_signal[delay_mask[i]];
        delay_mask[i] ++;

        i += stride;
    }
}

void CUDAImplementations::AddTransducerSource(TRealMatrix& uxyz_sgxyz,
                                              TIndexMatrix& us_index,
                                              TIndexMatrix& delay_mask,
                                              TRealMatrix& transducer_signal)
{

    const size_t us_index_size = us_index.GetTotalElementCount();
    float * ux = uxyz_sgxyz.GetRawDeviceData();
    const size_t* us_index_data = us_index.GetRawDeviceData();
    size_t * delay_mask_data = delay_mask.GetRawDeviceData();
    const float* transducer_signal_data = transducer_signal.GetRawDeviceData();

    CudaAddTransducerSource
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
        (us_index_size,
         ux,
         us_index_data,
         delay_mask_data,
         transducer_signal_data);
}

__global__ void CudaAdd_u_source(float* matrix_data,
                                 const float* u_source_input,
                                 const size_t* us_index,
                                 const int t_index,
                                 const size_t u_source_mode,
                                 const size_t u_source_many,
                                 const size_t us_index_total_element_count)
{

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t index2D = t_index;

    if (u_source_many != 0) { // is 2D
        index2D = t_index * us_index_total_element_count;
    }

    if (u_source_mode == 0){
        while(i < us_index_total_element_count){
            if (u_source_many != 0){
                matrix_data[us_index[i]] = u_source_input[index2D+i];
            }else{
                matrix_data[us_index[i]] = u_source_input[index2D];
            }
            i+= stride;
        }
    }// end of dirichlet

    if (u_source_mode == 1){
        while(i < us_index_total_element_count){
            if (u_source_many != 0){
                matrix_data[us_index[i]] += u_source_input[index2D+i];
            }else{
                matrix_data[us_index[i]] += u_source_input[index2D];
            }
            i+=stride;
        }
    }// end of add
}

void CUDAImplementations::Add_u_source(TRealMatrix& uxyz_sgxyz,
                                       TRealMatrix& u_source_input,
                                       TIndexMatrix& us_index,
                                       int t_index,
                                       size_t u_source_mode,
                                       size_t u_source_many)
{

    float* matrix_data = uxyz_sgxyz.GetRawDeviceData();
    float* u_source_input_data = u_source_input.GetRawDeviceData();
    size_t*  us_index_data = us_index.GetRawDeviceData();
    size_t us_index_size = us_index.GetTotalElementCount();

    CudaAdd_u_source
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
        (matrix_data,
         u_source_input_data,
         us_index_data,
         t_index,
         u_source_mode,
         u_source_many,
         us_index_size);
}

/*
 * uxyz_sgxyzMatrix functions (fourier)
 */
__global__  void CudaCompute_dt_rho_sg_mul_ifft_div_2(
              float* dt,
        const float* dt_rho0_sg)//,
        //const size_t matrix_size)
{

    const size_t matrix_size = constants.total_element_count;

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    const float divider = 1.0f/(2.0f * matrix_size);

    while (i < matrix_size) {
        dt[i] = dt_rho0_sg[i] * (dt[i] * divider);
        i += stride;
    }
}

void CUDAImplementations::Compute_dt_rho_sg_mul_ifft_div_2(
        TRealMatrix& uxyz_sgxyz,
        TRealMatrix& dt_rho0_sg,
        TCUFFTComplexMatrix& FFT)
{

    FFT.Compute_FFT_3D_C2R(uxyz_sgxyz);

    float * matrix_data = uxyz_sgxyz.GetRawDeviceData();
    float * dt_rho0_sg_data = dt_rho0_sg.GetRawDeviceData();
    //size_t matrix_size = uxyz_sgxyz.GetTotalElementCount();

    CudaCompute_dt_rho_sg_mul_ifft_div_2
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
        (matrix_data,
         dt_rho0_sg_data);//,
         //matrix_size);
}

__global__  void CudaCompute_dt_rho_sg_mul_ifft_div_2(
              float* dt,
        const float dt_rho_0_sgx)//,
        //const size_t matrix_size)
{

    const size_t matrix_size = constants.total_element_count;

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    const float divider = 1.0f/(2.0f * matrix_size) * dt_rho_0_sgx;

    while (i < matrix_size) {
        dt[i] = dt[i] * divider;
        i += stride;
    }
}

void CUDAImplementations::Compute_dt_rho_sg_mul_ifft_div_2(
        TRealMatrix& uxyz_sgxyz,
        float dt_rho_0_sg,
        TCUFFTComplexMatrix& FFT)
{
    FFT.Compute_FFT_3D_C2R(uxyz_sgxyz);

    float* matrix_data = uxyz_sgxyz.GetRawDeviceData();
    float dt_rho0_sg_data = dt_rho_0_sg;
    //size_t matrix_size = uxyz_sgxyz.GetTotalElementCount();

    CudaCompute_dt_rho_sg_mul_ifft_div_2
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
        (matrix_data,
         dt_rho0_sg_data);//,
         //matrix_size);
}

__global__ void CudaCompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_x(
        float* dt,
        const float  dt_rho_0_sgx,
        const float* dxudxn_sgx)//,
        //const size_t maxX,
        //const size_t maxY,
        //const size_t maxZ)
{

    const size_t max_x = constants.max_x;
    const size_t max_y = constants.max_y;
    const size_t max_z = constants.max_z;
    const size_t matrix_size = constants.total_element_count;

    //size_t matrix_size = maxX*maxY*maxZ;

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    const float divider = 1.0f/(2.0f * matrix_size) * dt_rho_0_sgx;

    for(; z < max_z; z+=zstride){
        for(; y < max_y; y+=ystride){
            for(; x < max_x; x+=xstride){
                register size_t i = z*max_y*max_x + y*max_x + x;
                dt[i] = dt[i] * divider * dxudxn_sgx[x];
            }
        }
    }
}

void CUDAImplementations::Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_x(
        TRealMatrix& uxyz_sgxyz,
        float dt_rho_0_sgx,
        TRealMatrix& dxudxn_sgx,
        TCUFFTComplexMatrix& FFT)
{
    FFT.Compute_FFT_3D_C2R(uxyz_sgxyz);

    float* matrix_data = uxyz_sgxyz.GetRawDeviceData();
    float  dt_rho0_sgx_data = dt_rho_0_sgx;

    //const size_t max_x = uxyz_sgxyz.GetDimensionSizes().X;
    //const size_t max_y = uxyz_sgxyz.GetDimensionSizes().Y;
    //const size_t max_z = uxyz_sgxyz.GetDimensionSizes().Z;

    float* dxudxn_sgx_data = dxudxn_sgx.GetRawDeviceData();

    CudaCompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_x
        <<< tuner->GetNumberOfBlocksFor3D(),
            tuner->GetNumberOfThreadsFor3D() >>>
        (matrix_data,
         dt_rho0_sgx_data,
         dxudxn_sgx_data);//,
         //max_x,
         //max_y,
         //max_z);
}

__global__ void CudaCompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_y(
              float* dt,
        const float  dt_rho_0_sgy,
        const float* dyudyn_sgy)//,
        //const size_t maxX,
        //const size_t maxY,
        //const size_t maxZ)
{

    const size_t max_x = constants.max_x;
    const size_t max_y = constants.max_y;
    const size_t max_z = constants.max_z;
    const size_t matrix_size = constants.total_element_count;

    //size_t matrix_size = maxX*maxY*maxZ;

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    const float divider = 1.0f/(2.0f * matrix_size) * dt_rho_0_sgy;

    for(; z < max_z; z+=zstride){
        for(; y < max_y; y+=ystride){
            const float dyudyn_sgy_data = dyudyn_sgy[y] * divider;
            for(; x < max_x; x+=xstride){
                register size_t i = z*max_y*max_x + y*max_x + x;

                dt[i] = dt[i] * dyudyn_sgy_data;
            }
        }
    }
}

void CUDAImplementations::Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_y(
        TRealMatrix& uxyz_sgxyz,
        float dt_rho_0_sgy,
        TRealMatrix& dyudyn_sgy,
        TCUFFTComplexMatrix& FFT)
{
    FFT.Compute_FFT_3D_C2R(uxyz_sgxyz);

    float * matrix_data = uxyz_sgxyz.GetRawDeviceData();
    float dt_rho0_sgy_data = dt_rho_0_sgy;

    //const size_t max_x = uxyz_sgxyz.GetDimensionSizes().X;
    //const size_t max_y = uxyz_sgxyz.GetDimensionSizes().Y;
    //const size_t max_z = uxyz_sgxyz.GetDimensionSizes().Z;

    float* dyudyn_sgy_data = dyudyn_sgy.GetRawDeviceData();

    CudaCompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_y
        <<< tuner->GetNumberOfBlocksFor3D(),
            tuner->GetNumberOfThreadsFor3D() >>>
        (matrix_data,
         dt_rho0_sgy_data,
         dyudyn_sgy_data);//,
         //max_x,
         //max_y,
         //max_z);
}

__global__ void CudaCompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_z(
        float* dt,
        float  dt_rho_0_sgz,
        float* dzudzn_sgz)//,
        //size_t max_x,
        //size_t max_y,
        //size_t max_z)
{

    const size_t max_x = constants.max_x;
    const size_t max_y = constants.max_y;
    const size_t max_z = constants.max_z;
    const size_t matrix_size = constants.total_element_count;

    //size_t matrix_size = maxX*maxY*maxZ;

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    const float divider = 1.0f/(2.0f * matrix_size) * dt_rho_0_sgz;

    for(; z < max_z; z+=zstride){
        const float dzudzn_sgz_data = dzudzn_sgz[z] * divider;
        for(; y < max_y; y+=ystride){
            for(; x < max_x; x+=xstride){
                register size_t i = z*max_y*max_x + y*max_x + x;

                dt[i] = dt[i] * dzudzn_sgz_data;
            }
        }
    }
}

void CUDAImplementations::Compute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_z(TRealMatrix& uxyz_sgxyz,
        float dt_rho_0_sgz,
        TRealMatrix& dzudzn_sgz,
        TCUFFTComplexMatrix& FFT)
{
    FFT.Compute_FFT_3D_C2R(uxyz_sgxyz);

    float * matrix_data = uxyz_sgxyz.GetRawDeviceData();
    float dt_rho0_sgz_data = dt_rho_0_sgz;

    //const size_t max_x = uxyz_sgxyz.GetDimensionSizes().X;
    //const size_t max_y = uxyz_sgxyz.GetDimensionSizes().Y;
    //const size_t max_z = uxyz_sgxyz.GetDimensionSizes().Z;

    float* dzudzn_sgz_data = dzudzn_sgz.GetRawDeviceData();

    CudaCompute_dt_rho_sg_mul_ifft_div_2_scalar_nonuniform_z
        <<< tuner->GetNumberOfBlocksFor3D(),
            tuner->GetNumberOfThreadsFor3D() >>>
        (matrix_data,
         dt_rho0_sgz_data,
         dzudzn_sgz_data);//,
         //max_x,
         //max_y,
         //max_z);
}

__global__ void CudaCompute_ddx_kappa_fft_p(float* FFT_X,
                                            float* FFT_Y,
                                            float* FFT_Z,
                                            float* kappa,
                                            float* ddx,
                                            float* ddy,
                                            float* ddz)//,
                                            //size_t max_x,
                                            //size_t max_y,
                                            //size_t max_z)
{

    const size_t max_x = constants.complex_max_x;
    const size_t max_y = constants.complex_max_y;
    const size_t max_z = constants.complex_max_z;

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    for(; z < max_z; z+=zstride){
        const float ddz_re = ddz[z<<1];
        const float ddz_im = ddz[(z<<1)+1];
        for(; y < max_y; y+=ystride){
            const float ddy_re = ddy[y<<1];
            const float ddy_im = ddy[(y<<1)+1];
            for(; x < max_x; x+=xstride){
                const float ddx_re = ddx[x<<1];
                const float ddx_im = ddx[(x<<1)+1];

                size_t i = x + y * max_x + z * (max_y * max_x);
                size_t i2 = i << 1;

                // kappa ./ p_k
                const float kappa_el  = kappa[i];
                const float p_k_el_re = FFT_X[i2] * kappa_el;
                const float p_k_el_im = FFT_X[i2+1] * kappa_el;

                //bxfun(ddx...)
                FFT_X[i2] = p_k_el_re * ddx_re - p_k_el_im * ddx_im;
                FFT_X[i2+1] = p_k_el_re * ddx_im + p_k_el_im * ddx_re;

                FFT_Y[i2] = p_k_el_re * ddy_re - p_k_el_im * ddy_im;
                FFT_Y[i2+1] = p_k_el_re * ddy_im + p_k_el_im * ddy_re;

                //bxfun(ddz...)
                FFT_Z[i2] = p_k_el_re * ddz_re - p_k_el_im * ddz_im;
                FFT_Z[i2+1] = p_k_el_re * ddz_im + p_k_el_im * ddz_re;
            }
        }
    }
}// end of Compute_ddx_kappa_fft_x
//----------------------------------------------------------------------------

/*
 * KSpaceFirstOrder3DSolver functions
 */
void CUDAImplementations::Compute_ddx_kappa_fft_p(TRealMatrix& X_Matrix,
                                                  TCUFFTComplexMatrix& FFT_X,
                                                  TCUFFTComplexMatrix& FFT_Y,
                                                  TCUFFTComplexMatrix& FFT_Z,
                                                  TRealMatrix& kappa,
                                                  TComplexMatrix& ddx,
                                                  TComplexMatrix& ddy,
                                                  TComplexMatrix& ddz)
{
    // Compute FFT of X
    FFT_X.Compute_FFT_3D_R2C(X_Matrix);

    float* p_k_x_data = FFT_X.GetRawDeviceData();
    float* p_k_y_data = FFT_Y.GetRawDeviceData();
    float* p_k_z_data = FFT_Z.GetRawDeviceData();

    float* kappa_data = kappa.GetRawDeviceData();
    float* ddx_data   = ddx.GetRawDeviceData();
    float* ddy_data   = ddy.GetRawDeviceData();
    float* ddz_data   = ddz.GetRawDeviceData();

    //const size_t Z_Size = FFT_X.GetDimensionSizes().Z;
    //const size_t Y_Size = FFT_X.GetDimensionSizes().Y;
    //const size_t X_Size = FFT_X.GetDimensionSizes().X;

    CudaCompute_ddx_kappa_fft_p
        <<< tuner->GetNumberOfBlocksFor3DComplex(),
            tuner->GetNumberOfThreadsFor3D() >>>
            (p_k_x_data,
             p_k_y_data,
             p_k_z_data,
             kappa_data,
             ddx_data,
             ddy_data,
             ddz_data);//,
             //X_Size,
             //Y_Size,
             //Z_Size);
}

__global__  void CudaCompute_duxyz_initial(      float* Temp_FFT_X_Data,
                                                 float* Temp_FFT_Y_Data,
                                                 float* Temp_FFT_Z_Data,
                                           const float* kappa_data,
                                           const float* ddx,
                                           const float* ddy,
                                           const float* ddz)//,
                                           //const size_t max_x,
                                           //const size_t max_y,
                                           //const size_t max_z,
                                           //const float divider)
{

    const size_t max_x = constants.complex_max_x;
    const size_t max_y = constants.complex_max_y;
    const size_t max_z = constants.complex_max_z;
    const float  divider = constants.divider;

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    for (; z < max_z; z+=zstride){
        //real
        const float ddz_neg_re = ddz[z*2];
        //imag
        const float ddz_neg_im = ddz[z*2+1];
        for (; y < max_y; y+=ystride){

            const float ddy_neg_re = ddy[y*2];
            const float ddy_neg_im = ddy[y*2+1];

            for (; x < max_x; x+=xstride){
                register size_t i = x + y * max_x + z * (max_y * max_x);
                i = i << 1;
                float FFT_el_x_re = Temp_FFT_X_Data[i];
                float FFT_el_x_im = Temp_FFT_X_Data[i+1];

                float FFT_el_y_re = Temp_FFT_Y_Data[i];
                float FFT_el_y_im = Temp_FFT_Y_Data[i+1];

                float FFT_el_z_re = Temp_FFT_Z_Data[i];
                float FFT_el_z_im = Temp_FFT_Z_Data[i+1];

                const float kappa_el = kappa_data[i >> 1];

                FFT_el_x_re *= kappa_el;
                FFT_el_x_im *= kappa_el;

                FFT_el_y_re *= kappa_el;
                FFT_el_y_im *= kappa_el;

                FFT_el_z_re *= kappa_el;
                FFT_el_z_im *= kappa_el;

                Temp_FFT_X_Data[i]     = ((FFT_el_x_re * ddx[x*2]) -
                        (FFT_el_x_im * ddx[x*2+1])) * divider;
                Temp_FFT_X_Data[i + 1] = ((FFT_el_x_im * ddx[x*2]) +
                        (FFT_el_x_re * ddx[x*2+1])) * divider;

                Temp_FFT_Y_Data[i]     = ((FFT_el_y_re * ddy_neg_re) -
                        (FFT_el_y_im * ddy_neg_im)) * divider;
                Temp_FFT_Y_Data[i + 1] = ((FFT_el_y_im * ddy_neg_re) +
                        (FFT_el_y_re * ddy_neg_im)) * divider;

                Temp_FFT_Z_Data[i]     = ((FFT_el_z_re * ddz_neg_re) -
                        (FFT_el_z_im * ddz_neg_im)) * divider;
                Temp_FFT_Z_Data[i + 1] = ((FFT_el_z_im * ddz_neg_re) +
                        (FFT_el_z_re * ddz_neg_im)) * divider;
            } // x
        } // y
    } // z
}

void CUDAImplementations::Compute_duxyz_initial(
        TCUFFTComplexMatrix& Temp_FFT_X,
        TCUFFTComplexMatrix& Temp_FFT_Y,
        TCUFFTComplexMatrix& Temp_FFT_Z,
        TRealMatrix& kappa,
        TRealMatrix& ux_sgx,
        TComplexMatrix& ddx_k_shift_neg,
        TComplexMatrix& ddy_k_shift_neg,
        TComplexMatrix& ddz_k_shift_neg)
{

    float * Temp_FFT_X_Data  = Temp_FFT_X.GetRawDeviceData();
    float * Temp_FFT_Y_Data  = Temp_FFT_Y.GetRawDeviceData();
    float * Temp_FFT_Z_Data  = Temp_FFT_Z.GetRawDeviceData();

    const float * kappa_data   = kappa.GetRawDeviceData();

    //const size_t FFT_Z_dim = Temp_FFT_Z.GetDimensionSizes().Z;
    //const size_t FFT_Y_dim = Temp_FFT_Y.GetDimensionSizes().Y;
    //const size_t FFT_X_dim = Temp_FFT_X.GetDimensionSizes().X;

    //const float  Divider = 1.0f / ux_sgx.GetTotalElementCount();

    const float * ddx = (float*) ddx_k_shift_neg.GetRawDeviceData();
    const float * ddy = (float*) ddy_k_shift_neg.GetRawDeviceData();
    const float * ddz = (float*) ddz_k_shift_neg.GetRawDeviceData();

    CudaCompute_duxyz_initial
        <<< tuner->GetNumberOfBlocksFor3D(),
            tuner->GetNumberOfThreadsFor3D() >>>
            (Temp_FFT_X_Data,
             Temp_FFT_Y_Data,
             Temp_FFT_Z_Data,
             kappa_data,
             ddx,
             ddy,
             ddz);//,
             //FFT_X_dim,
             //FFT_Y_dim,
             //FFT_Z_dim,
             //Divider);
}

__global__  void CudaCompute_duxyz_non_linear(float* duxdx_data,
        float* duydy_data,
        float* duzdz_data,
        const float* duxdxn_data,
        const float* duydyn_data,
        const float* duzdzn_data,
        const size_t max_x,
        const size_t max_y,
        const size_t max_z)
{

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    for (; z < max_z; z+=zstride){
        const float duzdzn_el = duzdzn_data[z];
        for (; y < max_y; y+=ystride){
            const float dyudyn_el = duydyn_data[y];
            for (; x < max_x; x+=xstride){
                register size_t i = x + y * max_x + z * (max_y * max_x);
                duxdx_data[i] *= duxdxn_data[x];
                duydy_data[i] *= dyudyn_el;
                duzdz_data[i] *= duzdzn_el;
            } // x
        } // y
    } // z
}

void CUDAImplementations::Compute_duxyz_non_linear(TRealMatrix& duxdx,
        TRealMatrix& duydy,
        TRealMatrix& duzdz,
        TRealMatrix& dxudxn,
        TRealMatrix& dyudyn,
        TRealMatrix& dzudzn)
{

    float * duxdx_data = duxdx.GetRawDeviceData();
    float * duydy_data = duydy.GetRawDeviceData();
    float * duzdz_data = duzdz.GetRawDeviceData();

    const float * duxdxn_data = dxudxn.GetRawDeviceData();
    const float * duydyn_data = dyudyn.GetRawDeviceData();
    const float * duzdzn_data = dzudzn.GetRawDeviceData();

    const size_t Z_Size = duxdx.GetDimensionSizes().Z;
    const size_t Y_Size = duxdx.GetDimensionSizes().Y;
    const size_t X_Size = duxdx.GetDimensionSizes().X;

    CudaCompute_duxyz_non_linear
        <<< tuner->GetNumberOfBlocksFor3D(),
        tuner->GetNumberOfThreadsFor3D() >>>
            (duxdx_data,
             duydy_data,
             duzdz_data,
             duxdxn_data,
             duydyn_data,
             duzdzn_data,
             X_Size,
             Y_Size,
             Z_Size);
}

__global__ void CudaComputeC2_matrix(float * c2_data, size_t max_i){
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    while (i < max_i){
        c2_data[i] = c2_data[i] * c2_data[i];
        i += istride;
    }
}

void CUDAImplementations::ComputeC2_matrix(TRealMatrix& c2)
{
    float * c2_data =  c2.GetRawDeviceData();

    CudaComputeC2_matrix
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
            (c2_data,
             c2.GetTotalElementCount());
}

__global__ void CudaCalculate_p0_source_add_initial_pressure(
        float*       rhox_data,
        float*       rhoy_data,
        float*       rhoz_data,
        const float* p0_data,
        const float  c2,
        const size_t max_i)
{

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    float tmp;

    while (i < max_i){
        tmp = p0_data[i] / (3.0f * c2);
        rhox_data[i] = tmp;
        rhoy_data[i] = tmp;
        rhoz_data[i] = tmp;

        i+=istride;
    }
}

__global__ void CudaCalculate_p0_source_add_initial_pressure(
        float*       rhox_data,
        float*       rhoy_data,
        float*       rhoz_data,
        const float* p0_data,
        const size_t c2_shift,
        const float* c2,
        const size_t max_i)
{

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    float tmp;

    while (i < max_i){
        tmp = p0_data[i] / (3.0f * c2[i * c2_shift]);
        rhox_data[i] = tmp;
        rhoy_data[i] = tmp;
        rhoz_data[i] = tmp;

        i+=istride;
    }
}

void CUDAImplementations::Calculate_p0_source_add_initial_pressure(
        TRealMatrix& rhox,
        TRealMatrix& rhoy,
        TRealMatrix& rhoz,
        TRealMatrix& p0,
        size_t c2_shift,
        float* c2_data)
{
    const float* p0_data = p0.GetRawDeviceData();

    float* rhox_data = rhox.GetRawDeviceData();
    float* rhoy_data = rhoy.GetRawDeviceData();
    float* rhoz_data = rhoz.GetRawDeviceData();

    if (!c2_shift){
        CudaCalculate_p0_source_add_initial_pressure
            <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
                (rhox_data,
                 rhoy_data,
                 rhoz_data,
                 p0_data,
                 c2_data[0],
                 rhox.GetTotalElementCount());
    }
    else {
        CudaCalculate_p0_source_add_initial_pressure
            <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
                (rhox_data,
                 rhoy_data,
                 rhoz_data,
                 p0_data,
                 c2_shift,
                 c2_data,
                 rhox.GetTotalElementCount());
    }
}

__global__ void CudaCompute_rhoxyz_nonlinear_scalar(float* rhox_data,
        float* rhoy_data,
        float* rhoz_data,
        const float* pml_x_data,
        const float* pml_y_data,
        const float* pml_z_data,
        const float* duxdx_data,
        const float* duydy_data,
        const float* duzdz_data,
        const float dt_rho0,
        const float dt2,
        const size_t max_x,
        const size_t max_y,
        const size_t max_z)
{

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    for(; z < max_z; z += zstride){
        const float pml_z = pml_z_data[z];
        for(; y < max_y; y += ystride){
            const float pml_y = pml_y_data[y];
            for(; x < max_x; x += xstride){
                register size_t i = x + y * max_x + z * (max_y * max_x);

                const float pml_x = pml_x_data[x];
                float dux = duxdx_data[i];

                rhox_data[i] = pml_x * (
                        ((pml_x * rhox_data[i]) - (dt_rho0 * dux))/
                        (1.0f + (dt2 * dux))
                        );

                float duy = duydy_data[i];
                rhoy_data[i] = pml_y * (
                        ((pml_y * rhoy_data[i]) - (dt_rho0 * duy))/
                        (1.0f + (dt2 * duy))
                        );

                float duz = duzdz_data[i];
                rhoz_data[i] = pml_z * (
                        ((pml_z * rhoz_data[i]) - (dt_rho0 * duz))/
                        (1.0f + (dt2 * duz))
                        );
            }
        }
    }
}

void CUDAImplementations::Compute_rhoxyz_nonlinear_scalar(TRealMatrix& rhox,
        TRealMatrix& rhoy,
        TRealMatrix& rhoz,
        TRealMatrix& pml_x,
        TRealMatrix& pml_y,
        TRealMatrix& pml_z,
        TRealMatrix& duxdx,
        TRealMatrix& duydy,
        TRealMatrix& duzdz,
        float dt_el,
        float rho0_scalar)
{
    const size_t max_z = rhox.GetDimensionSizes().Z;
    const size_t max_y = rhox.GetDimensionSizes().Y;
    const size_t max_x = rhox.GetDimensionSizes().X;

    float * rhox_data  = rhox.GetRawDeviceData();
    float * rhoy_data  = rhoy.GetRawDeviceData();
    float * rhoz_data  = rhoz.GetRawDeviceData();

    const float * pml_x_data = pml_x.GetRawDeviceData();
    const float * pml_y_data = pml_y.GetRawDeviceData();
    const float * pml_z_data = pml_z.GetRawDeviceData();
    const float * duxdx_data = duxdx.GetRawDeviceData();
    const float * duydy_data = duydy.GetRawDeviceData();
    const float * duzdz_data = duzdz.GetRawDeviceData();

    const float dt2 = 2.0f * dt_el;
    const float dt_rho0 = rho0_scalar * dt_el;

    CudaCompute_rhoxyz_nonlinear_scalar
        <<< tuner->GetNumberOfBlocksFor3D(),
        tuner->GetNumberOfThreadsFor3D() >>>
            (rhox_data,
             rhoy_data,
             rhoz_data,
             pml_x_data,
             pml_y_data,
             pml_z_data,
             duxdx_data,
             duydy_data,
             duzdz_data,
             dt_rho0,
             dt2,
             max_x,
             max_y,
             max_z);
}

__global__ void CudaCompute_rhoxyz_nonlinear_matrix(float* rhox_data,
        float* rhoy_data,
        float* rhoz_data,
        const float* pml_x_data,
        const float* pml_y_data,
        const float* pml_z_data,
        const float* duxdx_data,
        const float* duydy_data,
        const float* duzdz_data,
        const float dt_el,
        const float* rho0_data,
        const size_t max_x,
        const size_t max_y,
        const size_t max_z)
{

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    const float dt2 = 2.0f * dt_el;

    for(; z < max_z; z += zstride){
        const float pml_z = pml_z_data[z];

        for(; y < max_y; y += ystride){
            const float pml_y = pml_y_data[y];

            for(; x < max_x; x += xstride){
                const float pml_x = pml_x_data[x];

                register size_t i = x + y * max_x + z * (max_y * max_x);
                const float dt_rho0 = dt_el * rho0_data[i];

                float dux = duxdx_data[i];
                rhox_data[i] = pml_x * (
                        ((pml_x * rhox_data[i]) - (dt_rho0 * dux))/
                        (1.0f + (dt2 * dux))
                        );

                float duy = duydy_data[i];
                rhoy_data[i] = pml_y * (
                        ((pml_y * rhoy_data[i]) - (dt_rho0 * duy))/
                        (1.0f + (dt2 * duy))
                        );

                float duz = duzdz_data[i];
                rhoz_data[i] = pml_z * (
                        ((pml_z * rhoz_data[i]) - (dt_rho0 * duz))/
                        (1.0f + (dt2 * duz))
                        );
            }
        }
    }
}

void CUDAImplementations::Compute_rhoxyz_nonlinear_matrix(TRealMatrix& rhox,
        TRealMatrix& rhoy,
        TRealMatrix& rhoz,
        TRealMatrix& pml_x,
        TRealMatrix& pml_y,
        TRealMatrix& pml_z,
        TRealMatrix& duxdx,
        TRealMatrix& duydy,
        TRealMatrix& duzdz,
        float dt_el,
        TRealMatrix& rho0)
{

    const size_t max_z = rhox.GetDimensionSizes().Z;
    const size_t max_y = rhox.GetDimensionSizes().Y;
    const size_t max_x = rhox.GetDimensionSizes().X;

    float * rhox_data  = rhox.GetRawDeviceData();
    float * rhoy_data  = rhoy.GetRawDeviceData();
    float * rhoz_data  = rhoz.GetRawDeviceData();

    const float * pml_x_data = pml_x.GetRawDeviceData();
    const float * pml_y_data = pml_y.GetRawDeviceData();
    const float * pml_z_data = pml_z.GetRawDeviceData();
    const float * duxdx_data = duxdx.GetRawDeviceData();
    const float * duydy_data = duydy.GetRawDeviceData();
    const float * duzdz_data = duzdz.GetRawDeviceData();

    const float * rho0_data  = rho0.GetRawDeviceData();

    CudaCompute_rhoxyz_nonlinear_matrix
        <<< tuner->GetNumberOfBlocksFor3D(),
        tuner->GetNumberOfThreadsFor3D() >>>
            (rhox_data,
             rhoy_data,
             rhoz_data,
             pml_x_data,
             pml_y_data,
             pml_z_data,
             duxdx_data,
             duydy_data,
             duzdz_data,
             dt_el,
             rho0_data,
             max_x,
             max_y,
             max_z);
}

__global__ void CudaCompute_rhoxyz_linear_scalar(float* rhox_data,
        float* rhoy_data,
        float* rhoz_data,
        const float* pml_x_data,
        const float* pml_y_data,
        const float* pml_z_data,
        const float* duxdx_data,
        const float* duydy_data,
        const float* duzdz_data,
        const float dt_rho0,
        const size_t max_x,
        const size_t max_y,
        const size_t max_z)
{

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    for (; z < max_z; z += zstride){
        const float pml_z = pml_z_data[z];
        for (; y < max_y; y += ystride){
            const float pml_y = pml_y_data[y];
            for (; x < max_x; x += xstride){

                register size_t i = x + y * max_x + z * (max_y * max_x);

                const float pml_x = pml_x_data[x];
                rhox_data[i] = pml_x * (((pml_x * rhox_data[i]) -
                            (dt_rho0 * duxdx_data[i])));

                rhoy_data[i] = pml_y * (((pml_y * rhoy_data[i]) -
                            (dt_rho0 * duydy_data[i])));

                rhoz_data[i] = pml_z * (((pml_z * rhoz_data[i]) -
                            (dt_rho0 * duzdz_data[i])));
            }
        }
    }
}

void CUDAImplementations::Compute_rhoxyz_linear_scalar(TRealMatrix& rhox,
        TRealMatrix& rhoy,
        TRealMatrix& rhoz,
        TRealMatrix& pml_x,
        TRealMatrix& pml_y,
        TRealMatrix& pml_z,
        TRealMatrix& duxdx,
        TRealMatrix& duydy,
        TRealMatrix& duzdz,
        const float dt_el,
        const float rho0_scalar
        )
{
    const size_t max_z = rhox.GetDimensionSizes().Z;
    const size_t max_y = rhox.GetDimensionSizes().Y;
    const size_t max_x = rhox.GetDimensionSizes().X;

    float * rhox_data  = rhox.GetRawDeviceData();
    float * rhoy_data  = rhoy.GetRawDeviceData();
    float * rhoz_data  = rhoz.GetRawDeviceData();

    const float * pml_x_data = pml_x.GetRawDeviceData();
    const float * pml_y_data = pml_y.GetRawDeviceData();
    const float * pml_z_data = pml_z.GetRawDeviceData();
    const float * duxdx_data = duxdx.GetRawDeviceData();
    const float * duydy_data = duydy.GetRawDeviceData();
    const float * duzdz_data = duzdz.GetRawDeviceData();

    const float dt_rho0 = rho0_scalar * dt_el;

    CudaCompute_rhoxyz_linear_scalar
        <<< tuner->GetNumberOfBlocksFor3D(),
        tuner->GetNumberOfThreadsFor3D() >>>
            (rhox_data,
             rhoy_data,
             rhoz_data,
             pml_x_data,
             pml_y_data,
             pml_z_data,
             duxdx_data,
             duydy_data,
             duzdz_data,
             dt_rho0,
             max_x,
             max_y,
             max_z);
}

__global__ void CudaCompute_rhoxyz_linear_matrix(float * rhox_data,
        float * rhoy_data,
        float * rhoz_data,
        const float * pml_x_data,
        const float * pml_y_data,
        const float * pml_z_data,
        const float * duxdx_data,
        const float * duydy_data,
        const float * duzdz_data,
        const float dt_el,
        const float * rho0_data,
        const size_t max_x,
        const size_t max_y,
        const size_t max_z)
{

    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    size_t y = threadIdx.y + blockIdx.y*blockDim.y;
    size_t z = threadIdx.z + blockIdx.z*blockDim.z;

    size_t xstride = blockDim.x * gridDim.x;
    size_t ystride = blockDim.y * gridDim.y;
    size_t zstride = blockDim.z * gridDim.z;

    for (; z < max_z; z += zstride){
        const float pml_z = pml_z_data[z];

        for (; y < max_y; y += ystride){
            const float pml_y = pml_y_data[y];

            for (; x < max_x; x += xstride){
                register size_t i = x + y * max_x + z * (max_y * max_x);

                const float dt_rho0 = dt_el * rho0_data[i];
                const float pml_x   = pml_x_data[x];

                rhox_data[i] = pml_x * (((pml_x * rhox_data[i]) -
                            (dt_rho0 * duxdx_data[i])));

                rhoy_data[i] = pml_y * (((pml_y * rhoy_data[i]) -
                            (dt_rho0 * duydy_data[i])));

                rhoz_data[i] = pml_z * (((pml_z * rhoz_data[i]) -
                            (dt_rho0 * duzdz_data[i])));
            }
        }
    }
}

void CUDAImplementations::Compute_rhoxyz_linear_matrix(TRealMatrix& rhox,
        TRealMatrix& rhoy,
        TRealMatrix& rhoz,
        TRealMatrix& pml_x,
        TRealMatrix& pml_y,
        TRealMatrix& pml_z,
        TRealMatrix& duxdx,
        TRealMatrix& duydy,
        TRealMatrix& duzdz,
        const float dt_el,
        TRealMatrix& rho0)
{
    const size_t max_z = rhox.GetDimensionSizes().Z;
    const size_t max_y = rhox.GetDimensionSizes().Y;
    const size_t max_x = rhox.GetDimensionSizes().X;
    float * rhox_data  = rhox.GetRawDeviceData();
    float * rhoy_data  = rhoy.GetRawDeviceData();
    float * rhoz_data  = rhoz.GetRawDeviceData();

    const float * pml_x_data = pml_x.GetRawDeviceData();
    const float * pml_y_data = pml_y.GetRawDeviceData();
    const float * pml_z_data = pml_z.GetRawDeviceData();
    const float * duxdx_data = duxdx.GetRawDeviceData();
    const float * duydy_data = duydy.GetRawDeviceData();
    const float * duzdz_data = duzdz.GetRawDeviceData();

    const float * rho0_data  = rho0.GetRawDeviceData();

    CudaCompute_rhoxyz_linear_matrix
        <<< tuner->GetNumberOfBlocksFor3D(),
        tuner->GetNumberOfThreadsFor3D() >>>
            (rhox_data,
             rhoy_data,
             rhoz_data,
             pml_x_data,
             pml_y_data,
             pml_z_data,
             duxdx_data,
             duydy_data,
             duzdz_data,
             dt_el,
             rho0_data,
             max_x,
             max_y,
             max_z);

}

__global__ void CudaAdd_p_sourceReplacement(
        float* rhox_data,
        float* rhoy_data,
        float* rhoz_data,
        const float* p_source_input_data,
        const size_t* p_source_index_data,
        const size_t index2D,
        const size_t p_source_many,
        const size_t max_i)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    if (p_source_many != 0){
        for (; i < max_i; i += istride){

            rhox_data[p_source_index_data[i]] = p_source_input_data[index2D+i];
            rhoy_data[p_source_index_data[i]] = p_source_input_data[index2D+i];
            rhoz_data[p_source_index_data[i]] = p_source_input_data[index2D+i];

        }
    }
    else{
        for (; i < max_i; i += istride){

            rhox_data[p_source_index_data[i]] = p_source_input_data[index2D];
            rhoy_data[p_source_index_data[i]] = p_source_input_data[index2D];
            rhoz_data[p_source_index_data[i]] = p_source_input_data[index2D];

        }
    }
}

__global__ void CudaAdd_p_sourceAddition(
        float* rhox_data,
        float* rhoy_data,
        float* rhoz_data,
        const float* p_source_input_data,
        const size_t* p_source_index_data,
        const size_t index2D,
        const size_t p_source_many,
        const size_t max_i)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    if (p_source_many != 0){
        for (; i < max_i; i += istride){

            rhox_data[p_source_index_data[i]] += p_source_input_data[index2D+i];
            rhoy_data[p_source_index_data[i]] += p_source_input_data[index2D+i];
            rhoz_data[p_source_index_data[i]] += p_source_input_data[index2D+i];

        }
    }
    else{
        for (; i < max_i; i += istride){

            rhox_data[p_source_index_data[i]] += p_source_input_data[index2D];
            rhoy_data[p_source_index_data[i]] += p_source_input_data[index2D];
            rhoz_data[p_source_index_data[i]] += p_source_input_data[index2D];

        }
    }
}

void CUDAImplementations::Add_p_source(TRealMatrix& rhox,
        TRealMatrix& rhoy,
        TRealMatrix& rhoz,
        TRealMatrix& p_source_input,
        TIndexMatrix& p_source_index,
        size_t p_source_many,
        size_t p_source_mode,
        size_t t_index)
{
    float * rhox_data = rhox.GetRawDeviceData();
    float * rhoy_data = rhoy.GetRawDeviceData();
    float * rhoz_data = rhoz.GetRawDeviceData();

    const float * p_source_input_data = p_source_input.GetRawDeviceData();
    const size_t  * p_source_index_data = p_source_index.GetRawDeviceData();

    size_t index2D = t_index;

    if (p_source_many != 0) { // is 2D
        index2D = t_index * p_source_index.GetTotalElementCount();
    }

    // replacement
    if (p_source_mode == 0){
        CudaAdd_p_sourceReplacement
            <<< tuner->GetNumberOfBlocksForSubmatrixWithSize(
                            p_source_index.GetTotalElementCount()),
                //tuner->GetNumberOfBlocksFor1D(),
                tuner->GetNumberOfThreadsFor1D() >>>
                (rhox_data,
                 rhoy_data,
                 rhoz_data,
                 p_source_input_data,
                 p_source_index_data,
                 index2D,
                 p_source_many,
                 p_source_index.GetTotalElementCount());
        // Addition
    }else{
        CudaAdd_p_sourceAddition
            <<< tuner->GetNumberOfBlocksForSubmatrixWithSize(
                            p_source_index.GetTotalElementCount()),
                //tuner->GetNumberOfBlocksFor1D(),
                tuner->GetNumberOfThreadsFor1D() >>>
                (rhox_data,
                 rhoy_data,
                 rhoz_data,
                 p_source_input_data,
                 p_source_index_data,
                 index2D,
                 p_source_many,
                 p_source_index.GetTotalElementCount());
    }// type of replacement
}

__global__ void CudaCalculate_SumRho_BonA_SumDu(float*       RHO_Temp_Data,
                                                float*       BonA_Temp_Data,
                                                float*       SumDU_Temp_Data,
                                                const float* rhox_data,
                                                const float* rhoy_data,
                                                const float* rhoz_data,
                                                const float* dux_data,
                                                const float* duy_data,
                                                const float* duz_data,
                                                const float  BonA_data_scalar,
                                                const float* BonA_data_matrix,
                                                const size_t BonA_shift,
                                                const float  rho0_data_scalar,
                                                const float* rho0_data_matrix,
                                                const size_t rho0_shift,
                                                const size_t max_i)
{

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < max_i; i += istride){
        float BonA_data, rho0_data;

        if(BonA_shift == 0){
            BonA_data = BonA_data_scalar;
        }
        else{
            BonA_data = BonA_data_matrix[i];
        }

        if(rho0_shift == 0){
            rho0_data = rho0_data_scalar;
        }
        else{
            rho0_data = rho0_data_matrix[i];
        }

        register const float rho_xyz_el = rhox_data[i] +
            rhoy_data[i] +
            rhoz_data[i];
        RHO_Temp_Data[i]   = rho_xyz_el;
        BonA_Temp_Data[i]  = ((BonA_data * (rho_xyz_el * rho_xyz_el))
                / (2.0f * rho0_data)) + rho_xyz_el;
        SumDU_Temp_Data[i] = rho0_data * (dux_data[i] +
                duy_data[i] +
                duz_data[i]);
    }
}

void CUDAImplementations::Calculate_SumRho_BonA_SumDu(
        TRealMatrix& RHO_Temp,
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
        const size_t rho0_shift)
{
    const float* rhox_data = rhox.GetRawDeviceData();
    const float* rhoy_data = rhoy.GetRawDeviceData();
    const float* rhoz_data = rhoz.GetRawDeviceData();

    const float* dux_data = duxdx.GetRawDeviceData();
    const float* duy_data = duydy.GetRawDeviceData();
    const float* duz_data = duzdz.GetRawDeviceData();

          float* RHO_Temp_Data  = RHO_Temp.GetRawDeviceData();
          float* BonA_Temp_Data = BonA_Temp.GetRawDeviceData();
          float* SumDU_Temp_Data= Sum_du.GetRawDeviceData();

    CudaCalculate_SumRho_BonA_SumDu
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
            (RHO_Temp_Data,
             BonA_Temp_Data,
             SumDU_Temp_Data,
             rhox_data,
             rhoy_data,
             rhoz_data,
             dux_data,
             duy_data,
             duz_data,
             BonA_data_scalar,
             BonA_data_matrix,
             BonA_shift,
             rho0_data_scalar,
             rho0_data_matrix,
             rho0_shift,
             RHO_Temp.GetTotalElementCount());
}

__global__ void CudaCompute_Absorb_nabla1_2(float* FFT_1_data,
        float* FFT_2_data,
        const float* nabla1,
        const float* nabla2,
        const size_t max_i)
{

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < max_i; i += istride){
        float nabla_data1 = nabla1[i];
        FFT_1_data[(i<<1)]   *= nabla_data1;
        FFT_1_data[(i<<1)+1] *= nabla_data1;

        float nabla_data2 = nabla2[i];
        FFT_2_data[(i<<1)]   *= nabla_data2;
        FFT_2_data[(i<<1)+1] *= nabla_data2;
    }
}

void CUDAImplementations::Compute_Absorb_nabla1_2(TRealMatrix& absorb_nabla1,
        TRealMatrix& absorb_nabla2,
        TCUFFTComplexMatrix& FFT_1,
        TCUFFTComplexMatrix& FFT_2)
{
    const float * nabla1 = absorb_nabla1.GetRawDeviceData();
    const float * nabla2 = absorb_nabla2.GetRawDeviceData();

    float * FFT_1_data  = FFT_1.GetRawDeviceData();
    float * FFT_2_data  = FFT_2.GetRawDeviceData();

    CudaCompute_Absorb_nabla1_2
        <<< tuner->GetNumberOfBlocksFor1D(),
        tuner->GetNumberOfThreadsFor1D() >>>
            (FFT_1_data,
             FFT_2_data,
             nabla1,
             nabla2,
             FFT_1.GetTotalElementCount());
}

__global__ void CudaSum_Subterms_nonlinear(float*       p_data,
        const float* BonA_data,
        const float  c2_data_scalar,
        const float* c2_data_matrix,
        const size_t c2_shift,
        const float* Absorb_tau_data,
        const float  tau_data_scalar,
        const float* tau_data_matrix,
        const float* Absorb_eta_data,
        const float  eta_data_scalar,
        const float* eta_data_matrix,
        const size_t tau_eta_shift,
        const size_t max_i)
{
    const float divider = 1.0f / (float) max_i;

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < max_i; i += istride){

        float c2_data, eta_data, tau_data;

        if(c2_shift == 0){
            c2_data = c2_data_scalar;
        }
        else{
            c2_data = c2_data_matrix[i];
        }

        if(tau_eta_shift == 0){
            tau_data = tau_data_scalar;
            eta_data = eta_data_scalar;
        }
        else{
            tau_data = tau_data_matrix[i];
            eta_data = eta_data_matrix[i];
        }

        p_data[i] = c2_data *(BonA_data[i] +
                (divider * ((Absorb_tau_data[i] * tau_data) -
                            (Absorb_eta_data[i] * eta_data))));
    }
}

void CUDAImplementations::Sum_Subterms_nonlinear(TRealMatrix& BonA_temp,
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
        const size_t tau_eta_shift){

    const size_t TotalElementCount = p.GetTotalElementCount();

    const float * BonA_data = BonA_temp.GetRawDeviceData();
    float * p_data  = p.GetRawDeviceData();

    CudaSum_Subterms_nonlinear
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
            (p_data,
             BonA_data,
             c2_data_scalar,
             c2_data_matrix,
             c2_shift,
             Absorb_tau_data,
             tau_data_scalar,
             tau_data_matrix,
             Absorb_eta_data,
             eta_data_scalar,
             eta_data_matrix,
             tau_eta_shift,
             p.GetTotalElementCount());
}

__global__ void CudaSum_new_p_nonlinear_lossless(
        const size_t TotalElementCount,
        float*       p_data,
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
        const size_t rho0_shift
        )
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < TotalElementCount; i += istride){

        float c2_data;
        float BonA_data;
        float rho0_data;

        if (c2_shift == 0){
            c2_data = c2_data_scalar;
        }
        else {
            c2_data = c2_data_matrix[i];
        }
        if (BonA_shift == 0){
            BonA_data = BonA_data_scalar;
        }
        else {
            BonA_data = BonA_data_matrix[i];
        }
        if (rho0_shift == 0){
            rho0_data = rho0_data_scalar;
        }
        else {
            rho0_data = rho0_data_matrix[i];
        }

        const float sum_rho = rhox_data[i] + rhoy_data[i] + rhoz_data[i];
        p_data[i] = c2_data *(
                sum_rho +
                (BonA_data * (sum_rho* sum_rho) /
                 (2.0f* rho0_data))
                );
    }
}

void CUDAImplementations::Sum_new_p_nonlinear_lossless(
        size_t       TotalElementCount,
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
        const size_t rho0_shift)
{
    float * p_data  = p.GetRawDeviceData();

    CudaSum_new_p_nonlinear_lossless
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
            (TotalElementCount,
             p_data,
             rhox_data,
             rhoy_data,
             rhoz_data,
             c2_data_scalar,
             c2_data_matrix,
             c2_shift,
             BonA_data_scalar,
             BonA_data_matrix,
             BonA_shift,
             rho0_data_scalar,
             rho0_data_matrix,
             rho0_shift);
}

__global__ void CudaCalculate_SumRho_SumRhoDu(float* Sum_rhoxyz_data,
        float* Sum_rho0_du_data,
        const float* rhox_data,
        const float* rhoy_data,
        const float* rhoz_data,
        const float* dux_data,
        const float* duy_data,
        const float* duz_data,
        const float* rho0_data,
        const float rho0_data_el,
        const size_t TotalElementCount,
        bool rho0_scalar_flag)
{

    size_t istart = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (size_t i = istart; i < TotalElementCount; i += istride){
        Sum_rhoxyz_data[i] = rhox_data[i] + rhoy_data[i] + rhoz_data[i];
    }

    if (rho0_scalar_flag){ // scalar
        for (size_t i = istart; i < TotalElementCount; i += istride){
            Sum_rho0_du_data[i] = rho0_data_el *
                (dux_data[i] + duy_data[i] + duz_data[i]);
        }
    }
    else
    { // matrix
        for (size_t i = istart; i < TotalElementCount; i += istride){
            Sum_rho0_du_data[i] = rho0_data[i] *
                (dux_data[i] + duy_data[i] + duz_data[i]);
        }
    }
}

void CUDAImplementations::Calculate_SumRho_SumRhoDu(
        TRealMatrix& Sum_rhoxyz,
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
        const bool   rho0_scalar_flag)
{
    const float * rhox_data = rhox.GetRawDeviceData();
    const float * rhoy_data = rhoy.GetRawDeviceData();
    const float * rhoz_data = rhoz.GetRawDeviceData();

    const float * dux_data = duxdx.GetRawDeviceData();
    const float * duy_data = duydy.GetRawDeviceData();
    const float * duz_data = duzdz.GetRawDeviceData();

    float * Sum_rhoxyz_data  = Sum_rhoxyz.GetRawDeviceData();
    float * Sum_rho0_du_data = Sum_rho0_du.GetRawDeviceData();

    CudaCalculate_SumRho_SumRhoDu
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
            (Sum_rhoxyz_data,
             Sum_rho0_du_data,
             rhox_data,
             rhoy_data,
             rhoz_data,
             dux_data,
             duy_data,
             duz_data,
             rho0_data,
             rho0_data_el,
             TotalElementCount,
             rho0_scalar_flag);
}

__global__ void CudaSum_Subterms_linear(const float* Absorb_tau_data,
        const float* Absorb_eta_data,
        const float* Sum_rhoxyz_data,
              float* p_data,
        const size_t total_element_count,
        const size_t c2_shift,
        const size_t tau_eta_shift,
        const float  tau_data_scalar,
        const float* tau_data_matrix,
        const float  eta_data_scalar,
        const float* eta_data_matrix,
        const float  c2_data_scalar,
        const float* c2_data_matrix)
{

    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    const float divider = 1.0f / (float) total_element_count;

    for (; i < total_element_count; i += istride){

        float c2_data;
        float tau_data;
        float eta_data;

        //if c2 is a scalar use that element else use correct matrix element
        if(c2_shift == 0){
            c2_data = c2_data_scalar;
        }
        else{
            c2_data = c2_data_matrix[i];
        }
        //same as above but if tau is scalar then so too is eta
        if(tau_eta_shift == 0){
            tau_data = tau_data_scalar;
            eta_data = eta_data_scalar;
        }
        else{
            tau_data = tau_data_matrix[i];
            eta_data = eta_data_matrix[i];
        }

        p_data[i] = c2_data *(
                Sum_rhoxyz_data[i] +
                (divider * (Absorb_tau_data[i] * tau_data -
                            Absorb_eta_data[i] * eta_data)));
    }
}

void CUDAImplementations::Sum_Subterms_linear(TRealMatrix& Absorb_tau_temp,
        TRealMatrix& Absorb_eta_temp,
        TRealMatrix& Sum_rhoxyz,
        TRealMatrix& p,
        const size_t total_element_count,
        size_t       c2_shift,
        size_t       tau_eta_shift,
        const float  tau_data_scalar,
        const float* tau_data_matrix,
        const float  eta_data_scalar,
        const float* eta_data_matrix,
        const float  c2_data_scalar,
        const float* c2_data_matrix)
{
    const float *  Absorb_tau_data = Absorb_tau_temp.GetRawDeviceData();
    const float *  Absorb_eta_data = Absorb_eta_temp.GetRawDeviceData();

    const float * Sum_rhoxyz_data = Sum_rhoxyz.GetRawDeviceData();
    float * p_data  = p.GetRawDeviceData();

    CudaSum_Subterms_linear
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
            (Absorb_tau_data,
             Absorb_eta_data,
             Sum_rhoxyz_data,
             p_data,
             total_element_count,
             c2_shift,
             tau_eta_shift,
             tau_data_scalar,
             tau_data_matrix,
             eta_data_scalar,
             eta_data_matrix,
             c2_data_scalar,
             c2_data_matrix);
}

__global__ void CudaSum_new_p_linear_lossless_scalar(float* p_data,
        const float* rhox_data,
        const float* rhoy_data,
        const float* rhoz_data,
        const size_t total_element_count,
        const float  c2_element)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < total_element_count; i += istride){
        p_data[i] = c2_element * ( rhox_data[i] + rhoy_data[i] + rhoz_data[i]);
    }
}

void CUDAImplementations::Sum_new_p_linear_lossless_scalar(
        TRealMatrix& p,
        TRealMatrix& rhox,
        TRealMatrix& rhoy,
        TRealMatrix& rhoz,
        const size_t total_element_count,
        const float c2_element
        )
{
    float * p_data = p.GetRawDeviceData();
    const float * rhox_data = rhox.GetRawDeviceData();
    const float * rhoy_data = rhoy.GetRawDeviceData();
    const float * rhoz_data = rhoz.GetRawDeviceData();

    CudaSum_new_p_linear_lossless_scalar
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
        (p_data,
         rhox_data,
         rhoy_data,
         rhoz_data,
         total_element_count,
         c2_element);
}

__global__ void CudaSum_new_p_linear_lossless_matrix(float* p_data,
        const float* rhox_data,
        const float* rhoy_data,
        const float* rhoz_data,
        const size_t total_element_count,
        const float* c2_data)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < total_element_count; i += istride){
        p_data[i] = c2_data[i] * ( rhox_data[i] + rhoy_data[i] + rhoz_data[i]);
    }
}

void CUDAImplementations::Sum_new_p_linear_lossless_matrix(TRealMatrix& p,
        TRealMatrix& rhox,
        TRealMatrix& rhoy,
        TRealMatrix& rhoz,
        const size_t total_element_count,
        TRealMatrix& c2)
{
          float* p_data = p.GetRawDeviceData();
    const float* rhox_data = rhox.GetRawDeviceData();
    const float* rhoy_data = rhoy.GetRawDeviceData();
    const float* rhoz_data = rhoz.GetRawDeviceData();
    const float* c2_data = c2.GetRawDeviceData();

    CudaSum_new_p_linear_lossless_matrix
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
            (p_data,
             rhox_data,
             rhoy_data,
             rhoz_data,
             total_element_count,
             c2_data);
}

__global__ void CudaStoreSensorData_store_p_max(const float* p_data,
        float* p_max,
        const size_t* index,
        const size_t sensor_size)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < sensor_size; i += istride){
        if (p_max[i] < p_data[index[i]]) p_max[i] = p_data[index[i]];
    }
}

void CUDAImplementations::StoreSensorData_store_p_max(TRealMatrix& p,
        TRealMatrix& p_sensor_max,
        TIndexMatrix& sensor_mask_index
        )
{
    const float* p_data = p.GetRawDeviceData();
    float* p_max = p_sensor_max.GetRawDeviceData();
    const size_t* index = sensor_mask_index.GetRawDeviceData();
    const size_t sensor_size = sensor_mask_index.GetTotalElementCount();

    CudaStoreSensorData_store_p_max
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
        (p_data,
         p_max,
         index,
         sensor_size);
}

__global__ void CudaStoreSensorData_store_p_rms(const float* p_data,
        float* p_rms,
        const size_t* index,
        const size_t sensor_size)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < sensor_size; i += istride){
        p_rms[i] += (p_data[index[i]] * p_data[index[i]]);
    }
}

void CUDAImplementations::StoreSensorData_store_p_rms(TRealMatrix& p,
        TRealMatrix& p_sensor_rms,
        TIndexMatrix& sensor_mask_index
        )
{
    const float* p_data = p.GetRawDeviceData();
    float* p_rms = p_sensor_rms.GetRawDeviceData();
    const size_t* index = sensor_mask_index.GetRawDeviceData();
    const size_t sensor_size = sensor_mask_index.GetTotalElementCount();

    CudaStoreSensorData_store_p_rms
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
        (p_data,
         p_rms,
         index,
         sensor_size);
}

__global__ void CudaStoreSensorData_store_u_max(const float* ux,
        const float* uy,
        const float* uz,
        float* ux_max,
        float* uy_max,
        float* uz_max,
        const size_t* index,
        const size_t sensor_size)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < sensor_size; i += istride){
        if (ux_max[i] < ux[index[i]]) ux_max[i] = ux[index[i]];
        if (uy_max[i] < uy[index[i]]) uy_max[i] = uy[index[i]];
        if (uz_max[i] < uz[index[i]]) uz_max[i] = uz[index[i]];
    }
}

void CUDAImplementations::StoreSensorData_store_u_max(TRealMatrix& ux_sgx,
        TRealMatrix& uy_sgy,
        TRealMatrix& uz_sgz,
        TRealMatrix& ux_sensor_max,
        TRealMatrix& uy_sensor_max,
        TRealMatrix& uz_sensor_max,
        TIndexMatrix& sensor_mask_index
        )
{
    const float* ux = ux_sgx.GetRawDeviceData();
    const float* uy = uy_sgy.GetRawDeviceData();
    const float* uz = uz_sgz.GetRawDeviceData();
    float* ux_max = ux_sensor_max.GetRawDeviceData();
    float* uy_max = uy_sensor_max.GetRawDeviceData();
    float* uz_max = uz_sensor_max.GetRawDeviceData();
    const size_t* index = sensor_mask_index.GetRawDeviceData();
    const size_t sensor_size = sensor_mask_index.GetTotalElementCount();

    CudaStoreSensorData_store_u_max
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
        (ux,
         uy,
         uz,
         ux_max,
         uy_max,
         uz_max,
         index,
         sensor_size);
}

__global__ void CudaStoreSensorData_store_u_rms(const float* ux,
        const float* uy,
        const float* uz,
        float* ux_rms,
        float* uy_rms,
        float* uz_rms,
        const size_t* index,
        const size_t sensor_size)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < sensor_size; i += istride){
        ux_rms[i] += (ux[index[i]] * ux[index[i]]);
        uy_rms[i] += (uy[index[i]] * uy[index[i]]);
        uz_rms[i] += (uz[index[i]] * uz[index[i]]);
    }
}

void CUDAImplementations::StoreSensorData_store_u_rms(TRealMatrix& ux_sgx,
        TRealMatrix& uy_sgy,
        TRealMatrix& uz_sgz,
        TRealMatrix& ux_sensor_rms,
        TRealMatrix& uy_sensor_rms,
        TRealMatrix& uz_sensor_rms,
        TIndexMatrix& sensor_mask_index
        )
{
    const float* ux = ux_sgx.GetRawDeviceData();
    const float* uy = uy_sgy.GetRawDeviceData();
    const float* uz = uz_sgz.GetRawDeviceData();
    float* ux_rms = ux_sensor_rms.GetRawDeviceData();
    float* uy_rms = uy_sensor_rms.GetRawDeviceData();
    float* uz_rms = uz_sensor_rms.GetRawDeviceData();
    const size_t* index = sensor_mask_index.GetRawDeviceData();
    const size_t sensor_size = sensor_mask_index.GetTotalElementCount();

    CudaStoreSensorData_store_u_rms
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
        (ux,
         uy,
         uz,
         ux_rms,
         uy_rms,
         uz_rms,
         index,
         sensor_size);
}

__global__ void CudaStoreIntensityData_first_step(const size_t sensor_size,
        const size_t dims_x,
        const size_t dims_y,
        const size_t * index,
        const float* ux,
        const float* uy,
        const float* uz,
        const float* p,
        float * ux_i_1,
        float * uy_i_1,
        float * uz_i_1,
        float * p_i_1)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < sensor_size; i += istride){
        // calculate positions in the grid
        const size_t sensor_point_ind    = index[i];
        const size_t sensor_point_ind_1x = ((sensor_point_ind % dims_x) == 0) ?
            sensor_point_ind : sensor_point_ind - 1;
        const size_t sensor_point_ind_1y = (((sensor_point_ind /  dims_x) %  dims_y) == 0) ?
            sensor_point_ind : sensor_point_ind - dims_x;
        const size_t sensor_point_ind_1z = (((sensor_point_ind /  (dims_y * dims_x))) == 0) ?
            sensor_point_ind : sensor_point_ind - (dims_x * dims_y);

        //avg of actual values of u in staggered grid
        ux_i_1[i] = (ux[sensor_point_ind] + ux[sensor_point_ind_1x]) * 0.5f;
        uy_i_1[i] = (uy[sensor_point_ind] + uy[sensor_point_ind_1y]) * 0.5f;
        uz_i_1[i] = (uz[sensor_point_ind] + uz[sensor_point_ind_1z]) * 0.5f;
        p_i_1[i]   = p[sensor_point_ind];
    }
}

void CUDAImplementations::StoreIntensityData_first_step(
        const size_t sensor_size,
        const TDimensionSizes Dims,
        const size_t * index_data,
        const float* ux_data,
        const float* uy_data,
        const float* uz_data,
        const float* p_data,
        float * ux_i_1_data,
        float * uy_i_1_data,
        float * uz_i_1_data,
        float * p_i_1_data
        )
{

    const size_t dims_x = Dims.X;
    const size_t dims_y = Dims.Y;

    CudaStoreIntensityData_first_step
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
            (sensor_size,
             dims_x,
             dims_y,
             index_data,
             ux_data,
             uy_data,
             uz_data,
             p_data,
             ux_i_1_data,
             uy_i_1_data,
             uz_i_1_data,
             p_i_1_data);
}

__global__ void CudaStoreIntensityData_other_step(const size_t sensor_size,
                                                  const size_t dims_x,
                                                  const size_t dims_y,
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
                                                  size_t start_time_index)
{
    size_t i = threadIdx.x + blockIdx.x*blockDim.x;
    size_t istride = blockDim.x * gridDim.x;

    for (; i < sensor_size; i += istride){
        // calculate positions in the grid
        const size_t sensor_point_ind    = index[i];
        const size_t sensor_point_ind_1x = ((sensor_point_ind % dims_x) == 0) ?
            sensor_point_ind : sensor_point_ind - 1;
        const size_t sensor_point_ind_1y =
            (((sensor_point_ind /  dims_x) % dims_y) == 0) ?
                sensor_point_ind : sensor_point_ind - dims_x;
        const size_t sensor_point_ind_1z =
            (((sensor_point_ind /  (dims_y * dims_x))) == 0) ?
                sensor_point_ind : sensor_point_ind - (dims_x * dims_y);

        //avg of actual values of u in staggered grid
        const float ux_act =
            (ux[sensor_point_ind] + ux[sensor_point_ind_1x]) * 0.5f;
        const float uy_act =
            (uy[sensor_point_ind] + uy[sensor_point_ind_1y]) * 0.5f;
        const float uz_act =
            (uz[sensor_point_ind] + uz[sensor_point_ind_1z]) * 0.5f;
        const float p_data = p_i_1[i];

        // calculate actual intensity based on
        // p(n-1) * 1/4[ u(i-1)(n-1) + u(i)(n-1) + u(i-1)(n) + u(i)(n)
        const float Ix = p_data * ((ux_act + ux_i_1[i]) * 0.5f);
        const float Iy = p_data * ((uy_act + uy_i_1[i]) * 0.5f);
        const float Iz = p_data * ((uz_act + uz_i_1[i]) * 0.5f);

        ux_i_1[i] = ux_act;
        uy_i_1[i] = uy_act;
        uz_i_1[i] = uz_act;
        p_i_1[i]  = p[sensor_point_ind];

        const float Divider = 1.0f / (float) (Nt - start_time_index);

        // easily predictable...
        if (store_I_max) {
            if (Ix_max[i] < Ix) Ix_max[i] = Ix;
            if (Iy_max[i] < Iy) Iy_max[i] = Iy;
            if (Iz_max[i] < Iz) Iz_max[i] = Iz;
        }

        // easily predictable...
        if (store_I_avg) {
            Ix_avg[i] += (Ix * Divider);
            Iy_avg[i] += (Iy * Divider);
            Iz_avg[i] += (Iz * Divider);
        }
    }
}

void CUDAImplementations::StoreIntensityData_other_step(
        const size_t sensor_size,
        const TDimensionSizes Dims,
        const size_t * index_data,
        const float* ux_data,
        const float* uy_data,
        const float* uz_data,
        const float* p_data,
        float* ux_i_1_data,
        float* uy_i_1_data,
        float* uz_i_1_data,
        float* p_i_1_data,
        float* Ix_avg_data,
        float* Iy_avg_data,
        float* Iz_avg_data,
        float* Ix_max_data,
        float* Iy_max_data,
        float* Iz_max_data,
        bool store_I_avg,
        bool store_I_max,
        size_t Nt,
        size_t start_time_index)
{

    const size_t dims_x = Dims.X;
    const size_t dims_y = Dims.Y;

    CudaStoreIntensityData_other_step
        <<< tuner->GetNumberOfBlocksFor1D(),
            tuner->GetNumberOfThreadsFor1D() >>>
            (sensor_size,
             dims_x,
             dims_y,
             index_data,
             ux_data,
             uy_data,
             uz_data,
             p_data,
             ux_i_1_data,
             uy_i_1_data,
             uz_i_1_data,
             p_i_1_data,
             Ix_avg_data,
             Iy_avg_data,
             Iz_avg_data,
             Ix_max_data,
             Iy_max_data,
             Iz_max_data,
             store_I_avg,
             store_I_max,
             Nt,
             start_time_index);
}

