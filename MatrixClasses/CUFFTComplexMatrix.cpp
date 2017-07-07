/**
 * @file        CUFFTComplexMatrix.cpp
 *
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the class that implements 3D FFT using the cuFFT
 *              interface.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        09 August    2011, 13:10 (created) \n
 *              07 July      2017, 18:35 (revised)
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


#include <string>
#include <stdexcept>
#include <cufft.h>

#include <MatrixClasses/CUFFTComplexMatrix.h>
#include <MatrixClasses/RealMatrix.h>
#include <Logger/Logger.h>
#include <KSpaceSolver/SolverCUDAKernels.cuh>


//------------------------------------------------------------------------------------------------//
//------------------------------------------ Constants -------------------------------------------//
//------------------------------------------------------------------------------------------------//



//------------------------------------------------------------------------------------------------//
//----------------------------------- Static Member Variables ------------------------------------//
//------------------------------------------------------------------------------------------------//
cufftHandle TCUFFTComplexMatrix::cufftPlan_3D_R2C = cufftHandle();
cufftHandle TCUFFTComplexMatrix::cufftPlan_3D_C2R = cufftHandle();

cufftHandle TCUFFTComplexMatrix::cufftPlan_1DX_R2C = cufftHandle();
cufftHandle TCUFFTComplexMatrix::cufftPlan_1DY_R2C = cufftHandle();
cufftHandle TCUFFTComplexMatrix::cufftPlan_1DZ_R2C = cufftHandle();
cufftHandle TCUFFTComplexMatrix::cufftPlan_1DX_C2R = cufftHandle();
cufftHandle TCUFFTComplexMatrix::cufftPlan_1DY_C2R = cufftHandle();
cufftHandle TCUFFTComplexMatrix::cufftPlan_1DZ_C2R = cufftHandle();




/**
 * Error message for the CUFFT class.
 */
std::map<cufftResult, TErrorMessage> TCUFFTComplexMatrix::cuFFTErrorMessages
{
  {CUFFT_INVALID_PLAN             , ERR_FMT_CUFFT_INVALID_PLAN},
  {CUFFT_ALLOC_FAILED             , ERR_FMT_CUFFT_ALLOC_FAILED},
  {CUFFT_INVALID_TYPE             , ERR_FMT_CUFFT_INVALID_TYPE},
  {CUFFT_INVALID_VALUE            , ERR_FMT_CUFFT_INVALID_VALUE},
  {CUFFT_INTERNAL_ERROR           , ERR_FMT_CUFFT_INVALID_VALUE},
  {CUFFT_EXEC_FAILED              , ERR_FMT_CUFFT_EXEC_FAILED},
  {CUFFT_SETUP_FAILED             , eRR_FMT_CUFFT_SETUP_FAILED},
  {CUFFT_INVALID_SIZE             , ERR_FMT_CUFFT_INVALID_SIZE},
  {CUFFT_UNALIGNED_DATA           , ERR_FMT_CUFFT_UNALIGNED_DATA},
  {CUFFT_INCOMPLETE_PARAMETER_LIST, ERR_FMT_CUFFT_INCOMPLETE_PARAMETER_LIST},
  {CUFFT_INVALID_DEVICE           , ERR_FMT_CUFFT_INVALID_DEVICE},
  {CUFFT_PARSE_ERROR              , ERR_FMT_CUFFT_PARSE_ERROR},
  {CUFFT_NO_WORKSPACE             , ERR_FMT_CUFFT_NO_WORKSPACE},
  {CUFFT_NOT_IMPLEMENTED          , eRR_FMT_CUFFT_NOT_IMPLEMENTED},
  {CUFFT_LICENSE_ERROR            , ERR_FMT_CUFFT_LICENSE_ERROR},
  {CUFFT_NOT_SUPPORTED            , ERR_FMT_CUFFT_NOT_SUPPORTED}
};
//--------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Create an cuFFT plan for 3D Real-to-Complex. \n
 * This version doesn't need any scratch place for planning.
 *
 * @param [in] inMatrixDims - The dimension sizes of the input matrix
 * @throw runtime_error if the plan can't be created.
 */
void TCUFFTComplexMatrix::Create_FFT_Plan_3D_R2C(const TDimensionSizes& inMatrixDims)
{
  cufftResult cufftError;
  cufftError = cufftPlan3d(&cufftPlan_3D_R2C,
                           static_cast<int>(inMatrixDims.nz),
                           static_cast<int>(inMatrixDims.ny),
                           static_cast<int>(inMatrixDims.nx),
                           CUFFT_R2C);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Plan_3D_R2C");
}// end of CreateFFTPlan3D_RealToComplex
//-------------------------------------------------------------------------------------------------

/**
 * Create cuFFT plan for Complex-to-Real. \n
 * This version doesn't need any scratch place for planning.
 *
 * @param [in] outMatrixDims - the dimension sizes of the output matrix
 * @throw runtime_error if the plan can't be created.
 */
void TCUFFTComplexMatrix::Create_FFT_Plan_3D_C2R(const TDimensionSizes& outMatrixDims)
{
  cufftResult_t cufftError;
  cufftError = cufftPlan3d(&cufftPlan_3D_C2R,
                           static_cast<int>(outMatrixDims.nz),
                           static_cast<int>(outMatrixDims.ny),
                           static_cast<int>(outMatrixDims.nx),
                           CUFFT_C2R);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "CUDA_FFT_3D_C2R");
}//end of CreateFFTPlan3D_ComplexToReal
//--------------------------------------------------------------------------------------------------


/**
 * Create cuFFT plan for 1DX Real-to-Complex. \n
 * This version doesn't need any scratch place for planning. All 1D transforms are done in a
 * single batch (no transpose needed) and in out-of-place manner.
 *
 * @param [in] inMatrixDims - The dimension sizes of the input matrix
 * @throw runtime_error if the plan can't be created.
 */
void TCUFFTComplexMatrix::Create_FFT_Plan_1DX_R2C(const TDimensionSizes& inMatrixDims)
{
  cufftResult_t cufftError;

  // set dimensions
  const int nx   = static_cast<int> (inMatrixDims.nx);
  const int ny   = static_cast<int> (inMatrixDims.ny);
  const int nz   = static_cast<int> (inMatrixDims.nz);
  const int nx_2 = ((nx / 2) + 1);

  // set up rank and strides
  int rank = 1;
  int n[] = {nx};

  // Since runs out-of-place no padding is needed.
  int inembed[] = {nx};
  int istride   = 1;
  int idist     = nx;

  int onembed[] = {nx_2};
  int ostride   = 1;
  int odist     = nx_2;

  int batch = ny * nz;

  // plan the FFT
  cufftError = cufftPlanMany(&cufftPlan_1DX_R2C, rank, n,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_R2C, batch);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "CUDA_FFT_1DX_R2C");
}//end of Create_FFT_Plan_1DX_R2C
//--------------------------------------------------------------------------------------------------


/**
 * Create cuFFT plan for 1DY Real-to-Complex. \n
 * This version doesn't need any scratch place for planning. All 1D transforms are done in a single
 * batch. Data has to be transposed and padded according to the cuFFT data layout before the
 * transform. The FFT is done in-place.
 *
 * @param [in] inMatrixDims - The dimension sizes of the input matrix
 * @throw runtime_error if the plan can't be created.
 */
void TCUFFTComplexMatrix::Create_FFT_Plan_1DY_R2C(const TDimensionSizes& inMatrixDims)
{
  cufftResult_t cufftError;

  // set dimensions
  const int nx   = static_cast<int> (inMatrixDims.nx);
  const int ny   = static_cast<int> (inMatrixDims.ny);
  const int nz   = static_cast<int> (inMatrixDims.nz);
  const int ny_2 = ((ny / 2) + 1);

  // set up rank and strides
  int rank = 1;
  int n[] = {ny};

  // The input matrix is transposed with every row padded by a single element.
  int inembed[] = {ny +1};
  int istride   = 1;
  int idist     = ny + 1;

  int onembed[] = {ny_2};
  int ostride   = 1;
  int odist     = ny_2;

  int batch =  nx * nz;

  cufftError = cufftPlanMany(&cufftPlan_1DY_R2C, rank, n,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_R2C, batch);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "CUDA_FFT_1DY_R2C");
}//end of Create_FFT_Plan_1DY_R2C
//--------------------------------------------------------------------------------------------------

/**
 * Create cuFFT plan for 1DZ Real-to-Complex. \n
 * This version doesn't need any scratch place for planning.  All 1D transforms are done in a single
 * batch. Data has to be transposed and padded according to the cuFFT data layout before the
 * transform. The FFT is done in-place.
 *
 * @param [in] inMatrixDims - The dimension sizes of the input matrix
 * @throw runtime_error if the plan can't be created.
 */
void TCUFFTComplexMatrix::Create_FFT_Plan_1DZ_R2C(const TDimensionSizes& inMatrixDims)
{
  cufftResult_t cufftError;

  const int nx   = static_cast<int> (inMatrixDims.nx);
  const int ny   = static_cast<int> (inMatrixDims.ny);
  const int nz   = static_cast<int> (inMatrixDims.nz);
  const int nz_2 = ((nz / 2) + 1);

  // set up rank and strides
  int rank = 1;
  int n[] = {nz};

  // The input matrix is transposed with every row padded by a single element.
  int inembed[] = {nz + 1};
  int istride   = 1;
  int idist     = nz + 1;

  int onembed[] = {nz_2};
  int ostride   = 1;
  int odist     = nz_2;

  int batch =  nx * ny;

  cufftError = cufftPlanMany(&cufftPlan_1DZ_R2C, rank, n,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_R2C, batch);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "CUDA_FFT_1DZ_R2C");

}//end of Create_FFT_Plan_1DZ_R2C
//--------------------------------------------------------------------------------------------------



/**
 * Create cuFFT plan for 1DX Complex-to-Real. \n
 * This version doesn't need any scratch place for planning.  All 1D transforms are done in a
 * single batch (no transpose needed). The FFT is done out-of-place.
 *
 * @param [in] outMatrixDims - The dimension sizes of the input matrix
 * @throw runtime_error if the plan can't be created.
 */
void TCUFFTComplexMatrix::Create_FFT_Plan_1DX_C2R(const TDimensionSizes& outMatrixDims)
{
  cufftResult_t cufftError;

  // set dimensions
  const int nx   = static_cast<int> (outMatrixDims.nx);
  const int ny   = static_cast<int> (outMatrixDims.ny);
  const int nz   = static_cast<int> (outMatrixDims.nz);
  const int nx_2 = ((nx / 2) + 1);

  // set up rank and strides
  int rank = 1;
  int n[] = {nx};

  // Since runs out-of-place no padding is needed.
  int inembed[] = {nx_2};
  int istride   = 1;
  int idist     = nx_2;

  int onembed[] = {nx};
  int ostride   = 1;
  int odist     = nx;

  int batch = ny * nz;

  cufftError = cufftPlanMany(&cufftPlan_1DX_C2R, rank, n,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_C2R, batch);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "CUDA_FFT_1DX_C2R");
}//end of Create_FFT_Plan_1DX_C2R
//-------------------------------------------------------------------------------------------------


/**
 * Create cuFFT plan for 1DY Complex-to-Real. \n
 * This version doesn't need any scratch place for planning. All 1D transforms are done in a single
 * batch. The output matrix is padded and transposed to be padded according to the cuFFT data
 * layout.
 *
 * @param [in] outMatrixDims - The dimension sizes of the input matrix
 * @throw runtime_error if the plan can't be created.
 */
void TCUFFTComplexMatrix::Create_FFT_Plan_1DY_C2R(const TDimensionSizes& outMatrixDims)
{
  cufftResult_t cufftError;
  // set dimensions
  const int nx   = static_cast<int> (outMatrixDims.nx);
  const int ny   = static_cast<int> (outMatrixDims.ny);
  const int nz   = static_cast<int> (outMatrixDims.nz);
  const int ny_2 = ((ny / 2) + 1);

  // set up rank and strides
  int rank = 1;
  int n[] = {ny};

  int inembed[] = {ny_2};
  int istride   = 1;
  int idist     = ny_2;

  // The output matrix is transposed with every row padded by a single element.
  int onembed[] = {ny + 1};
  int ostride   = 1;
  int odist     = ny + 1;

  int batch =  nx * nz;

  cufftError = cufftPlanMany(&cufftPlan_1DY_C2R, rank, n,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_C2R, batch);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "CUDA_FFT_1DY_C2R");
}//end of Create_FFT_Plan_1DY_R2C
//--------------------------------------------------------------------------------------------------


/**
 * Create cuFFT plan for 1DZ Complex-to-Real. \n
 * This version doesn't need any scratch place for planning. All 1D transforms are done in a single
 * batch. The output matrix is padded and transposed to be padded according to the cuFFT data
 * layout.
 *
 * @param [in] outMatrixDims - the dimension sizes of the input matrix
 * @throw runtime_error if the plan can't be created.
 */
void TCUFFTComplexMatrix::Create_FFT_Plan_1DZ_C2R(const TDimensionSizes& outMatrixDims)
{
  cufftResult_t cufftError;

  // set dimensions
  const int nx   = static_cast<int> (outMatrixDims.nx);
  const int ny   = static_cast<int> (outMatrixDims.ny);
  const int nz   = static_cast<int> (outMatrixDims.nz);
  const int nz_2 = ((nz / 2) + 1);

  // set up rank and strides
  int rank = 1;
  int n[] = {nz};

  int inembed[] = {nz_2};
  int istride   = 1;
  int idist     = nz_2;

  // The output matrix is transposed with every row padded by a single element.
  int onembed[] = {nz + 1};
  int ostride   = 1;
  int odist     = nz + 1;

  int batch =  nx * ny;

  cufftError = cufftPlanMany(&cufftPlan_1DZ_C2R, rank, n,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_C2R, batch);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "CUDA_FFT_1DZ_C2R");
}//end of Create_FFT_Plan_1DX_R2C
//--------------------------------------------------------------------------------------------------


/**
 * Destroy all static plans created by the application.
 * @throw runtime_error if the plan can't be created.
 */
void TCUFFTComplexMatrix::DestroyAllPlansAndStaticData()
{
  cufftResult_t cufftError;

  if (cufftPlan_3D_R2C)
  {
    cufftError = cufftDestroy(cufftPlan_3D_R2C);
    cufftPlan_3D_R2C = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Destroy_3D_R2C_Plan");
  }

  if (cufftPlan_3D_C2R)
  {
    cufftError = cufftDestroy(cufftPlan_3D_C2R);
    cufftPlan_3D_C2R = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Destroy_3D_C2R_Plan");
  }

  if (cufftPlan_1DX_R2C)
  {
    cufftError = cufftDestroy(cufftPlan_1DX_R2C);
    cufftPlan_1DX_R2C = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Destroy_1DX_R2C_Plan");
  }

  if (cufftPlan_1DY_R2C)
  {
    cufftError = cufftDestroy(cufftPlan_1DY_R2C);
    cufftPlan_1DY_R2C = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Destroy_1DY_R2C_Plan");
  }

  if (cufftPlan_1DZ_R2C)
  {
    cufftError = cufftDestroy(cufftPlan_1DZ_R2C);
    cufftPlan_1DZ_R2C = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Destroy_1DZ_R2C_Plan");
  }

  if (cufftPlan_1DX_C2R)
  {
    cufftError = cufftDestroy(cufftPlan_1DX_C2R);
    cufftPlan_1DX_C2R = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Destroy_1DX_C2R_Plan");
  }

  if (cufftPlan_1DY_C2R)
  {
    cufftError = cufftDestroy(cufftPlan_1DY_C2R);
    cufftPlan_1DY_C2R = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Destroy_1DY_C2R_Plan");
  }

  if (cufftPlan_1DZ_C2R)
  {
    cufftError = cufftDestroy(cufftPlan_1DZ_C2R);
    cufftPlan_1DZ_C2R = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Destroy_1DZ_C2R_Plan");
  }

  cuFFTErrorMessages.clear();
}// end of DestroyAllPlans
//--------------------------------------------------------------------------------------------------


/**
 * Computer forward out-of-place 3D Real-to-Complex FFT.
 *
 * @param [in] inMatrix - Input data for the forward FFT
 * @throw runtime_error if the plan is not valid.
 */
void TCUFFTComplexMatrix::Compute_FFT_3D_R2C(TRealMatrix& inMatrix)
{
  //Compute forward cuFFT (if the plan does not exist, it also returns error)
  cufftResult_t cufftError = cufftExecR2C(cufftPlan_3D_R2C,
                                          static_cast<cufftReal*>(inMatrix.GetDeviceData()),
                                          reinterpret_cast<cufftComplex*>(deviceData));

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Execute_FFT_3D_R2C");
}// end of Compute_FFT_3D_R2C
//--------------------------------------------------------------------------------------------------



/**
 * Computer forward out-of-place 3D Complex-to-Real FFT.
 *
 * @param [out] outMatrix - output of the inverse FFT.
 * @throw runtime_error if the plan is not valid.
 */
void TCUFFTComplexMatrix::Compute_FFT_3D_C2R(TRealMatrix& outMatrix)
{
  //Compute forward cuFFT (if the plan does not exist, it also returns error)
  cufftResult_t cufftError = cufftExecC2R(cufftPlan_3D_C2R,
                                          reinterpret_cast<cufftComplex*>(deviceData),
                                          static_cast<cufftReal*>(outMatrix.GetDeviceData()));

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Execute_FFT_3D_RCR");
}// end of Compute_FFT_3D_C2R
//--------------------------------------------------------------------------------------------------

/**
 * Computer forward out-of-place 1DX Real-to-Complex FFT.
 *
 * @param [in] inMatrix - Input data for the forward FFT.
 * @throw runtime_error if the plan is not valid.
 */
void TCUFFTComplexMatrix::Compute_FFT_1DX_R2C(TRealMatrix& inMatrix)
{
  //Compute forward cuFFT (if the plan does not exist, it also returns error)
  cufftResult_t cufftError = cufftExecR2C(cufftPlan_1DX_R2C,
                                          static_cast<cufftReal*>(inMatrix.GetDeviceData()),
                                          reinterpret_cast<cufftComplex*>(deviceData));

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Execute_FFT_1DX_R2C");
}// end of Compute_FFT_1DX_R2C
//--------------------------------------------------------------------------------------------------


/**
 * Computer forward out-of-place 1DY Real-to-Complex FFT. The matrix is first X<->Y transposed
 * followed by the 1D FFT. The matrix is left in the transposed format.
 *
 * @param [in] inMatrix - Input data for the forward FFT.
 * @throw runtime_error if the plan is not valid.
 */
void TCUFFTComplexMatrix::Compute_FFT_1DY_R2C(TRealMatrix& inMatrix)
{
  /// Transpose a real 3D matrix in the X-Y direction
  dim3 dimSizes(inMatrix.GetDimensionSizes().nx,
                inMatrix.GetDimensionSizes().ny,
                inMatrix.GetDimensionSizes().nz);

  SolverCUDAKernels::TrasposeReal3DMatrixXY<SolverCUDAKernels::TransposePadding::kOutput>
                                           (deviceData,
                                            inMatrix.GetDeviceData(),
                                            dimSizes);

  // Compute forward cuFFT (if the plan does not exist, it also returns error).
  // the FFT is calculated in-place (may be a bit slower than out-of-place, however
  // it does not request additional transfers and memory).
  cufftResult_t cufftError = cufftExecR2C(cufftPlan_1DY_R2C,
                                          static_cast<cufftReal*>(deviceData),
                                          reinterpret_cast<cufftComplex*>(deviceData));

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Execute_FFT_1DY_R2C");
}// end of Compute_FFT_1DY_R2C
//--------------------------------------------------------------------------------------------------

/**
 * Computer forward out-of-place 1DY Real-to-Complex FFT. The matrix is first X<->Z transposed
 * followed by the 1D FFT. The matrix is left in the transposed format.
 *
 * @param [in] inMatrix - Input data for the forward FFT.
 * @throw runtime_error if the plan is not valid.
 */
void TCUFFTComplexMatrix::Compute_FFT_1DZ_R2C(TRealMatrix& inMatrix)
{
  /// Transpose a real 3D matrix in the X-Z direction
  dim3 dimSizes(inMatrix.GetDimensionSizes().nx,
                inMatrix.GetDimensionSizes().ny,
                inMatrix.GetDimensionSizes().nz);

  SolverCUDAKernels::TrasposeReal3DMatrixXZ<SolverCUDAKernels::TransposePadding::kOutput>
                                           (deviceData,
                                            inMatrix.GetDeviceData(),
                                            dimSizes);

  // Compute forward cuFFT (if the plan does not exist, it also returns error).
  // the FFT is calculated in-place (may be a bit slower than out-of-place, however
  // it does not request additional transfers and memory).
  cufftResult_t cufftError = cufftExecR2C(cufftPlan_1DZ_R2C,
                                          static_cast<cufftReal*>(deviceData),
                                          reinterpret_cast<cufftComplex*>(deviceData));

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Execute_FFT_1DZ_R2C");
}// end of Compute_FFT_1DZ_R2C
//--------------------------------------------------------------------------------------------------

/**
 * Computer inverse out-of-place 1DX Real-to-Complex FFT.
 *
 * @param [out] outMatrix - Output data for the inverse FFT.
 * @throw runtime_error if the plan is not valid.
 */
void TCUFFTComplexMatrix::Compute_FFT_1DX_C2R(TRealMatrix& outMatrix)
{
  //Compute inverse cuFFT (if the plan does not exist, it also returns error)
  cufftResult_t cufftError = cufftExecC2R(cufftPlan_1DX_C2R,
                                          reinterpret_cast<cufftComplex*>(deviceData),
                                          static_cast<cufftReal*> (outMatrix.GetDeviceData()));

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Execute_FFT_1DX_C2R");
}// end of Compute_FFT_1DX_C2R
//--------------------------------------------------------------------------------------------------


/**
 * Computer inverse out-of-place 1DY Real-to-Complex FFT. The matrix is requested to be in the
 * transposed layout. After the FFT is calculated, an Y<->X transposed follows. The matrix is
 * returned in the normal layout (z, y, x)  format.
 *
 * @param [out] outMatrix - Output data for the inverse FFT.
 * @throw runtime_error if the plan is not valid.
 */
void TCUFFTComplexMatrix::Compute_FFT_1DY_C2R(TRealMatrix& outMatrix)
{
  // Compute forward cuFFT (if the plan does not exist, it also returns error).
  // the FFT is calculated in-place (may be a bit slower than out-of-place, however
  // it does not request additional transfers and memory).
  cufftResult_t cufftError = cufftExecC2R(cufftPlan_1DY_C2R,
                                          reinterpret_cast<cufftComplex*>(deviceData),
                                          static_cast<cufftReal*>(deviceData));

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Execute_FFT_1DY_C2R");

  /// Transpose a real 3D matrix back in the X-Y direction
  dim3 dimSizes(outMatrix.GetDimensionSizes().ny,
                outMatrix.GetDimensionSizes().nx,
                outMatrix.GetDimensionSizes().nz);

  SolverCUDAKernels::TrasposeReal3DMatrixXY<SolverCUDAKernels::TransposePadding::kInput>
                                           (outMatrix.GetDeviceData(),
                                            deviceData,
                                            dimSizes);
}// end of Compute_FFT_1DY_C2R
//--------------------------------------------------------------------------------------------------


/**
 * Computer forward out-of-place 1DY Real-to-Complex FFT. The matrix is requested to  be in the
 * transposed layout. After the FFT is calculated, an Z<->X transposed follows. The matrix is
 * returned in the normal layout (z, y, x).
 *
 * @param [out] outMatrix - Output data for the inverse FFT.
 * @throw runtime_error if the plan is not valid.
 */
void TCUFFTComplexMatrix::Compute_FFT_1DZ_C2R(TRealMatrix& outMatrix)
{
  // Compute forward cuFFT (if the plan does not exist, it also returns error).
  // the FFT is calculated in-place (may be a bit slower than out-of-place, however
  // it does not request additional transfers and memory).
  cufftResult_t cufftError = cufftExecC2R(cufftPlan_1DZ_C2R,
                                          reinterpret_cast<cufftComplex*>(deviceData),
                                          static_cast<cufftReal*>(deviceData));

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Execute_FFT_1DZ_C2R");

  /// Transpose a real 3D matrix in the Z<->X direction
  dim3 DimSizes(outMatrix.GetDimensionSizes().nz,
                outMatrix.GetDimensionSizes().ny,
                outMatrix.GetDimensionSizes().nx);

  SolverCUDAKernels::TrasposeReal3DMatrixXZ<SolverCUDAKernels::TransposePadding::kInput>
                                           (outMatrix.GetDeviceData(),
                                            GetDeviceData(),
                                            DimSizes);
}// end of Compute_FFT_1DZ_C2R
//--------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//


//------------------------------------------------------------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Throw cuda FFT exception
 *
 * @param [in] cufftError        - CUDA FFT error code
 * @param [in] transformTypeName - CUDA transform type name
 * @throw runtime error if error occurs
 */
void TCUFFTComplexMatrix::ThrowCUFFTException(const cufftResult  cufftError,
                                              const std::string& transformTypeName)
{
  std::string errMsg;
  if (cuFFTErrorMessages.find(cufftError) != cuFFTErrorMessages.end())
  {
    errMsg = TLogger::FormatMessage(cuFFTErrorMessages[cufftError], transformTypeName.c_str());
  }
  else // unknown error
  {
    errMsg = TLogger::FormatMessage(ERR_FMT_CUFFT_UNKNOWN_ERROR, transformTypeName.c_str());
  }

  // Throw exception
  throw std::runtime_error(errMsg);
}// end of GetCuFFTErrorMessage
//--------------------------------------------------------------------------------------------------