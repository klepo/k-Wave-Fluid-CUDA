/**
 * @file        CUFFTComplexMatrix.cpp
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the class that implements
 *              3D FFT using the FFTW interface.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        09 August    2011, 13:10 (created) \n
 *              08 July      2015, 16:54 (revised)
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


#include <iostream>
#include <string>
#include <stdexcept>
#include <cufft.h>
#include <map>

#include <MatrixClasses/CUFFTComplexMatrix.h>
#include <MatrixClasses/RealMatrix.h>
#include <Utils/ErrorMessages.h>
#include <CUDA/CUDAImplementations.h>


using namespace std;

/**
* Check errors of the CUDA routines and print error.
 * @param [in] code  - error code of last routine
 * @param [in] file  - The name of the file, where the error was raised
 * @param [in] line  - What is the line
 * @param [in] Abort - Shall the code abort?
 */
inline void gpuAssert(cudaError_t code,
                      string file,
                      int line,
                      bool Abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,
            "GPUassert: %s %s %d\n",
            cudaGetErrorString(code),file.c_str(),line);
    if (Abort) exit(code);
  }
}

/// Define to get the usage easier
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


//----------------------------------------------------------------------------//
//                               Constants                                    //
//----------------------------------------------------------------------------//




//----------------------------------------------------------------------------//
//                         Static Member Variables                            //
//----------------------------------------------------------------------------//
cufftHandle TCUFFTComplexMatrix::cufft_plan_3D_R2C = static_cast<cufftHandle>(NULL);
cufftHandle TCUFFTComplexMatrix::cufft_plan_3D_C2R = static_cast<cufftHandle>(NULL);

cufftHandle TCUFFTComplexMatrix::cufft_plan_1DX_R2C = static_cast<cufftHandle>(NULL);
cufftHandle TCUFFTComplexMatrix::cufft_plan_1DY_R2C = static_cast<cufftHandle>(NULL);
cufftHandle TCUFFTComplexMatrix::cufft_plan_1DZ_R2C = static_cast<cufftHandle>(NULL);
cufftHandle TCUFFTComplexMatrix::cufft_plan_1DX_C2R = static_cast<cufftHandle>(NULL);
cufftHandle TCUFFTComplexMatrix::cufft_plan_1DY_C2R = static_cast<cufftHandle>(NULL);
cufftHandle TCUFFTComplexMatrix::cufft_plan_1DZ_C2R = static_cast<cufftHandle>(NULL);


//std::map<cufftResult_t, const char *> cuFFTErrorMessages;

/**
 * @variable cuFFTErrorMessages
 * @brief Error message for the CUFFT class
 */
std::map<cufftResult, TErrorMessage> TCUFFTComplexMatrix::cuFFTErrorMessages
{
  {CUFFT_INVALID_PLAN             , CUFFTComplexMatrix_ERR_FMT_INVALID_PLAN},
  {CUFFT_ALLOC_FAILED             , CUFFTComplexMatrix_ERR_FMT_CUFFT_ALLOC_FAILED},
  {CUFFT_INVALID_TYPE             , CUFFTComplexMatrix_ERR_FMT_CUFFT_INVALID_TYPE},
  {CUFFT_INVALID_VALUE            , CUFFTComplexMatrix_ERR_FMT_CUFFT_INVALID_VALUE},
  {CUFFT_INTERNAL_ERROR           , CUFFTComplexMatrix_ERR_FMT_CUFFT_INVALID_VALUE},
  {CUFFT_EXEC_FAILED              , CUFFTComplexMatrix_ERR_FMT_CUFFT_EXEC_FAILED},
  {CUFFT_SETUP_FAILED             , CUFFTComplexMatrix_ERR_FMT_CUFFT_SETUP_FAILED},
  {CUFFT_INVALID_SIZE             , CUFFTComplexMatrix_ERR_FMT_CUFFT_INVALID_SIZE},
  {CUFFT_UNALIGNED_DATA           , CUFFTComplexMatrix_ERR_FMT_CUFFT_UNALIGNED_DATA},
  {CUFFT_INCOMPLETE_PARAMETER_LIST, CUFFTComplexMatrix_ERR_FMT_CUFFT_INCOMPLETE_PARAMETER_LIST},
  {CUFFT_INVALID_DEVICE           , CUFFTComplexMatrix_ERR_FMT_CUFFT_INVALID_DEVICE},
  {CUFFT_PARSE_ERROR              , CUFFTComplexMatrix_ERR_FMT_CUFFT_PARSE_ERROR},
  {CUFFT_NO_WORKSPACE             , CUFFTComplexMatrix_ERR_FMT_CUFFT_NO_WORKSPACE},
  {CUFFT_NOT_IMPLEMENTED          , CUFFTComplexMatrix_ERR_FMT_CUFFT_NOT_IMPLEMENTED},
  {CUFFT_LICENSE_ERROR            , CUFFTComplexMatrix_ERR_FMT_CUFFT_LICENSE_ERROR}
};
//-------------------------------------------------------------------------------


//----------------------------------------------------------------------------//
//                               Public methods                               //
//----------------------------------------------------------------------------//

/**
 *
 * Create an cuFFT plan for 3D Real-to-Complex. \n
 * This version doesn't need any scratch place for planning.
 * @param [in] InMatrixDims - the dimension sizes of the input matrix
 * @throw runtime_error if the plan can't be created.
 */
void TCUFFTComplexMatrix::Create_FFT_Plan_3D_R2C(const TDimensionSizes& InMatrixDims)
{
  cufftResult cufftError;
  cufftError = cufftPlan3d(&cufft_plan_3D_R2C,
                           static_cast<int>(InMatrixDims.Z),
                           static_cast<int>(InMatrixDims.Y),
                           static_cast<int>(InMatrixDims.X),
                           CUFFT_R2C);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Plan_3D_R2C");

  // be careful, this feature is deprecated in CUDA 6.5
  cufftError = cufftSetCompatibilityMode(cufft_plan_3D_R2C,
                                         CUFFT_COMPATIBILITY_NATIVE);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "SetCompatibilty_3D_R2C_Plan");
}// end of CreateFFTPlan3D_RealToComplex
//------------------------------------------------------------------------------

/*
 * Create cuFFT plan for Complex-to-Real. \n
 * This version doesn't need any scratch place for planning.
 * @param [in] OutMatrixDims - the dimension sizes of the output matrix
 * @throw runtime_error if the plan can't be created.
 */
void TCUFFTComplexMatrix::Create_FFT_Plan_3D_C2R(const TDimensionSizes& OutMatrixDims)
{
  cufftResult_t cufftError;
  cufftError = cufftPlan3d(&cufft_plan_3D_C2R,
                           static_cast<int>(OutMatrixDims.Z),
                           static_cast<int>(OutMatrixDims.Y),
                           static_cast<int>(OutMatrixDims.X),
                           CUFFT_C2R);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "CUDA_FFT_3D_C2R");


  // be careful, this feature is deprecated in CUDA 6.5
  cufftError =  cufftSetCompatibilityMode(cufft_plan_3D_C2R,
                                          CUFFT_COMPATIBILITY_NATIVE);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "SetCompatibilty_3D_C2R_Plan");
}//end of CreateFFTPlan3D_ComplexToReal
//------------------------------------------------------------------------------


/*
 * Create cuFFT plan for 1DX Real-to-Complex. \n
 * This version doesn't need any scratch place for planning.
 * All 1D transforms are done in a single batch (no transpose needed).
 * @param [in] InMatrixDims - the dimension sizes of the input matrix
 * @throw runtime_error if the plan can't be created.
 */
void TCUFFTComplexMatrix::Create_FFT_Plan_1DX_R2C(const TDimensionSizes& InMatrixDims)
{
  cufftResult_t cufftError;

  // set dimensions
  const int X   = static_cast<int> (InMatrixDims.X);
  const int Y   = static_cast<int> (InMatrixDims.Y);
  const int Z   = static_cast<int> (InMatrixDims.Z);
  const int X_2 = ((X / 2) + 1);

  // set up rank and strides
  int rank = 1;
  int n[] = {X};

  int inembed[] = {X};
  int istride   = 1;
  int idist     = X;

  int onembed[] = {X_2};
  int ostride   = 1;
  int odist     = X_2;

  int batch = Y * Z;

  // plan the FFT
  cufftError = cufftPlanMany(&cufft_plan_1DX_R2C, rank, n,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_R2C, batch);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "CUDA_FFT_1DX_R2C");

  // be careful, this feature is deprecated in CUDA 7.0.
  // It is necessary to use this compatibility level, otherwise we would have to use an
  // out-of-place transform - (inplace transform corrupts data)
  cufftError =  cufftSetCompatibilityMode(cufft_plan_1DX_R2C,
                                          CUFFT_COMPATIBILITY_NATIVE);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "SetCompatibilty_1DX_R2C_Plan");
}//end of Create_FFT_Plan_1DX_R2C
//------------------------------------------------------------------------------


/*
 * Create cuFFT plan for 1DY Real-to-Complex. \n
 * This version doesn't need any scratch place for planning.
 * All 1D transforms are done in a single batch. Data has
 * to be transposed before the transform.
 * @param [in] InMatrixDims - the dimension sizes of the input matrix
 * @throw runtime_error if the plan can't be created.
 */
void TCUFFTComplexMatrix::Create_FFT_Plan_1DY_R2C(const TDimensionSizes& InMatrixDims)
{
  cufftResult_t cufftError;

  // set dimensions
  const int X   = static_cast<int> (InMatrixDims.X);
  const int Y   = static_cast<int> (InMatrixDims.Y);
  const int Z   = static_cast<int> (InMatrixDims.Z);
  const int Y_2 = ((Y / 2) + 1);

  // set up rank and strides
  int rank = 1;
  int n[] = {Y};

  int inembed[] = {Y};
  int istride   = 1;
  int idist     = Y;

  int onembed[] = {Y_2};
  int ostride   = 1;
  int odist     = Y_2;

  int batch =  X * Z;

  cufftError = cufftPlanMany(&cufft_plan_1DY_R2C, rank, n,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_R2C, batch);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "CUDA_FFT_1DY_R2C");

  // be careful, this feature is deprecated in CUDA 7.0.
  // It is necessary to use this compatibility level, otherwise we would have to use an
  // out-of-place transform - (inplace transforms corrupts data)
  cufftError =  cufftSetCompatibilityMode(cufft_plan_1DY_R2C,
                                          CUFFT_COMPATIBILITY_NATIVE);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "SetCompatibilty_1DY_R2C_Plan");
}//end of Create_FFT_Plan_1DY_R2C
//------------------------------------------------------------------------------

/*
 * Create cuFFT plan for 1DZ Real-to-Complex. \n
 * This version doesn't need any scratch place for planning.
 * All 1D transforms are done in a single batch. Data has
 * to be transposed before the transform.
 * @param [in] InMatrixDims - the dimension sizes of the input matrix
 * @throw runtime_error if the plan can't be created.
 */
void TCUFFTComplexMatrix::Create_FFT_Plan_1DZ_R2C(const TDimensionSizes& InMatrixDims)
{
  cufftResult_t cufftError;

  const int X   = static_cast<int> (InMatrixDims.X);
  const int Y   = static_cast<int> (InMatrixDims.Y);
  const int Z   = static_cast<int> (InMatrixDims.Z);
  const int Z_2 = ((Z / 2) + 1);

  // set up rank and strides
  int rank = 1;
  int n[] = {Z};

  int inembed[] = {Z};
  int istride   = 1;
  int idist     = Z;

  int onembed[] = {Z_2};
  int ostride   = 1;
  int odist     = Z_2;

  int batch =  X * Y;

  cufftError = cufftPlanMany(&cufft_plan_1DZ_R2C, rank, n,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_R2C, batch);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "CUDA_FFT_1DZ_R2C");

  // be careful, this feature is deprecated in CUDA 7.0.
  // It is necessary to use this compatibility level, otherwise we would have to use an
  // out-of-place transform - (inplace transforms corrupts data)
  cufftError =  cufftSetCompatibilityMode(cufft_plan_1DZ_R2C,
                                          CUFFT_COMPATIBILITY_NATIVE);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "SetCompatibilty_1DZ_R2C_Plan");
}//end of Create_FFT_Plan_1DZ_R2C
//------------------------------------------------------------------------------



/*
 * Create cuFFT plan for 1DX Complex-to-Real. \n
 * This version doesn't need any scratch place for planning.
 * All 1D transforms are done in a single batch (no transpose needed).
 * @param [in] OutMatrixDims - the dimension sizes of the input matrix
 * @throw runtime_error if the plan can't be created.
 */
void TCUFFTComplexMatrix::Create_FFT_Plan_1DX_C2R(const TDimensionSizes& OutMatrixDims)
{
  cufftResult_t cufftError;

  // set dimensions
  const int X   = static_cast<int> (OutMatrixDims.X);
  const int Y   = static_cast<int> (OutMatrixDims.Y);
  const int Z   = static_cast<int> (OutMatrixDims.Z);
  const int X_2 = ((X / 2) + 1);

  // set up rank and strides
  int rank = 1;
  int n[] = {X};

  int inembed[] = {X_2};
  int istride   = 1;
  int idist     = X_2;

  int onembed[] = {X};
  int ostride   = 1;
  int odist     = X;

  int batch = Y * Z;

  cufftError = cufftPlanMany(&cufft_plan_1DX_C2R, rank, n,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_C2R, batch);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "CUDA_FFT_1DX_C2R");

  // be careful, this feature is deprecated in CUDA 7.0.
  // It is necessary to use this compatibility level, otherwise we would have to use an
  // out-of-place transform - (inplace transforms corrupts data)
  cufftError =  cufftSetCompatibilityMode(cufft_plan_1DX_C2R,
                                          CUFFT_COMPATIBILITY_NATIVE);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "SetCompatibilty_1DX_C2R_Plan");
}//end of Create_FFT_Plan_1DX_R2C
//------------------------------------------------------------------------------


/*
 * Create cuFFT plan for 1DY Complex-to-Real. \n
 * This version doesn't need any scratch place for planning.
 * All 1D transforms are done in a single batch. Data has
 * to be transposed before the transform.
 * @param [in] OutMatrixDims - the dimension sizes of the input matrix
 * @throw runtime_error if the plan can't be created.
 */
void TCUFFTComplexMatrix::Create_FFT_Plan_1DY_C2R(const TDimensionSizes& OutMatrixDims)
{
  cufftResult_t cufftError;
  // set dimensions
  const int X   = static_cast<int> (OutMatrixDims.X);
  const int Y   = static_cast<int> (OutMatrixDims.Y);
  const int Z   = static_cast<int> (OutMatrixDims.Z);
  const int Y_2 = ((Y/ 2) + 1);

  // set up rank and strides
  int rank = 1;
  int n[] = {Y};

  int inembed[] = {Y_2};
  int istride   = 1;
  int idist     = Y_2;

  int onembed[] = {Y};
  int ostride   = 1;
  int odist     = Y;

  int batch =  X * Z;

  cufftError = cufftPlanMany(&cufft_plan_1DY_C2R, rank, n,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_C2R, batch);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "CUDA_FFT_1DY_C2R");

  // be careful, this feature is deprecated in CUDA 7.0.
  // It is necessary to use this compatibility level, otherwise we would have to use an
  // out-of-place transform - (inplace transforms corrupts data)
  cufftError =  cufftSetCompatibilityMode(cufft_plan_1DY_C2R,
                                          CUFFT_COMPATIBILITY_NATIVE);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "SetCompatibilty_1DX_C2R_Plan");
}//end of Create_FFT_Plan_1DX_R2C
//------------------------------------------------------------------------------


/*
 * Create cuFFT plan for 1DZ Complex-to-Real. \n
 * This version doesn't need any scratch place for planning.
 * All 1D transforms are done in a single batch. Data has
 * to be transposed before the transform.
 * @param [in] OutMatrixDims - the dimension sizes of the input matrix
 * @throw runtime_error if the plan can't be created.
 */
void TCUFFTComplexMatrix::Create_FFT_Plan_1DZ_C2R(const TDimensionSizes& OutMatrixDims)
{
  cufftResult_t cufftError;

  // set dimensions
  const int X   = static_cast<int> (OutMatrixDims.X);
  const int Y   = static_cast<int> (OutMatrixDims.Y);
  const int Z   = static_cast<int> (OutMatrixDims.Z);
  const int Z_2 = ((Z / 2) + 1);

  // set up rank and strides
  int rank = 1;
  int n[] = {Z};

  int inembed[] = {Z_2};
  int istride   = 1;
  int idist     = Z_2;

  int onembed[] = {Z};
  int ostride   = 1;
  int odist     = Z;

  int batch =  X * Y;

  cufftError = cufftPlanMany(&cufft_plan_1DZ_C2R, rank, n,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_C2R, batch);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "CUDA_FFT_1DZ_C2R");

  // be careful, this feature is deprecated in CUDA 7.0.
  // It is necessary to use this compatibility level, otherwise we would have to use an
  // out-of-place transform - (inplace transforms corrupts data)
  cufftError =  cufftSetCompatibilityMode(cufft_plan_1DZ_C2R,
                                          CUFFT_COMPATIBILITY_NATIVE);

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "SetCompatibilty_1DZ_R2C_Plan");
}//end of Create_FFT_Plan_1DX_R2C
//------------------------------------------------------------------------------


/**
 * Destroy all static plans created by the application.
 * @throw runtime_error if the plan can't be created.
 */
void TCUFFTComplexMatrix::DestroyAllPlansAndStaticData()
{
  cufftResult_t cufftError;

  if (cufft_plan_3D_R2C)
  {
    cufftError = cufftDestroy(cufft_plan_3D_R2C);
    cufft_plan_3D_R2C = static_cast<cufftHandle>(NULL);
    if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Destroy_3D_R2C_Plan");
  }

  if (cufft_plan_3D_C2R)
  {
    cufftError = cufftDestroy(cufft_plan_3D_C2R);
    cufft_plan_3D_C2R = static_cast<cufftHandle>(NULL);
    if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Destroy_3D_C2R_Plan");
  }

  if (cufft_plan_1DX_R2C)
  {
    cufftError = cufftDestroy(cufft_plan_1DX_R2C);
    cufft_plan_1DX_R2C = static_cast<cufftHandle>(NULL);
    if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Destroy_1DX_R2C_Plan");
  }

  if (cufft_plan_1DY_R2C)
  {
    cufftError = cufftDestroy(cufft_plan_1DY_R2C);
    cufft_plan_1DY_R2C = static_cast<cufftHandle>(NULL);
    if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Destroy_1DY_R2C_Plan");
  }

  if (cufft_plan_1DZ_R2C)
  {
    cufftError = cufftDestroy(cufft_plan_1DZ_R2C);
    cufft_plan_1DZ_R2C = static_cast<cufftHandle>(NULL);
    if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Destroy_1DZ_R2C_Plan");
  }

  if (cufft_plan_1DX_C2R)
  {
    cufftError = cufftDestroy(cufft_plan_1DX_C2R);
    cufft_plan_1DX_C2R = static_cast<cufftHandle>(NULL);
    if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Destroy_1DX_C2R_Plan");
  }

  if (cufft_plan_1DY_C2R)
  {
    cufftError = cufftDestroy(cufft_plan_1DY_C2R);
    cufft_plan_1DY_C2R = static_cast<cufftHandle>(NULL);
    if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Destroy_1DY_C2R_Plan");
  }

  if (cufft_plan_1DZ_C2R)
  {
    cufftError = cufftDestroy(cufft_plan_1DZ_C2R);
    cufft_plan_1DZ_C2R = static_cast<cufftHandle>(NULL);
    if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Destroy_1DZ_C2R_Plan");
  }

  cuFFTErrorMessages.clear();
}// end of DestroyAllPlans
//------------------------------------------------------------------------------


/**
 * Computer forward out-of-place 3D Real-to-Complex FFT.
 * @param [in] InMatrix - Input data for the forward FFT
 * @throw runtime_error if the plan is not valid.
 */
void TCUFFTComplexMatrix::Compute_FFT_3D_R2C(TRealMatrix& InMatrix)
{
  //Compute forward cuFFT (if the plan does not exist, it also returns error)
  cufftResult_t cufftError = cufftExecR2C(cufft_plan_3D_R2C,
                                          static_cast<cufftReal*>(InMatrix.GetRawDeviceData()),
                                          reinterpret_cast<cufftComplex*>(pdMatrixData));

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Execute_FFT_3D_R2C");
}// end of Compute_FFT_3D_R2C
//------------------------------------------------------------------------------



/**
 * Computer forward out-of-place 3D Complex-to-Real FFT.
 * @param [out] OutMatrix - output of the inverse FFT.
 * @throw runtime_error if the plan is not valid.
 */
void TCUFFTComplexMatrix::Compute_FFT_3D_C2R(TRealMatrix& OutMatrix)
{
  //Compute forward cuFFT (if the plan does not exist, it also returns error)
  cufftResult_t cufftError = cufftExecC2R(cufft_plan_3D_C2R,
                                          reinterpret_cast<cufftComplex*>(pdMatrixData),
                                          static_cast<cufftReal*>(OutMatrix.GetRawDeviceData()));

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Execute_FFT_3D_RCR");
}// end of Compute_FFT_3D_C2R
//------------------------------------------------------------------------------

/**
 * Computer forward out-of-place 1DX Real-to-Complex FFT.
 * @param [in] InMatrix - Input data for the forward FFT.
 * @throw runtime_error if the plan is not valid.
 */
void TCUFFTComplexMatrix::Compute_FFT_1DX_R2C(TRealMatrix& InMatrix)
{
  //Compute forward cuFFT (if the plan does not exist, it also returns error)
  cufftResult_t cufftError = cufftExecR2C(cufft_plan_1DX_R2C,
                                          static_cast<cufftReal*>(InMatrix.GetRawDeviceData()),
                                          reinterpret_cast<cufftComplex*>(pdMatrixData));

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Execute_FFT_1DX_R2C");
}// end of Compute_FFT_1DX_R2C
//------------------------------------------------------------------------------


/**
 * Computer forward out-of-place 1DY Real-to-Complex FFT. The matrix is first
 * X<->Y transposed followed by the 1D FFT. The matrix is left in the transposed
 * format.
 * @param [in] InMatrix - Input data for the forward FFT.
 * @throw runtime_error if the plan is not valid.
 */
void TCUFFTComplexMatrix::Compute_FFT_1DY_R2C(TRealMatrix& InMatrix)
{
  /// Transpose a real 3D matrix in the X-Y direction
  dim3 DimSizes(InMatrix.GetDimensionSizes().X,
                InMatrix.GetDimensionSizes().Y,
                InMatrix.GetDimensionSizes().Z);

  TCUDAImplementations::GetInstance()->TrasposeReal3DMatrixXY(pdMatrixData,
                                                              InMatrix.GetRawDeviceData(),
                                                              DimSizes);

  // Compute forward cuFFT (if the plan does not exist, it also returns error).
  // the FFT is calculated in-place (may be a bit slower than out-of-place, however
  // it does not request additional transfers and memory).
  cufftResult_t cufftError = cufftExecR2C(cufft_plan_1DY_R2C,
                                          static_cast<cufftReal*>(pdMatrixData),
                                          reinterpret_cast<cufftComplex*>(pdMatrixData));

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Execute_FFT_1DY_R2C");
}// end of Compute_FFT_1DY_R2C
//------------------------------------------------------------------------------

/**
 * Computer forward out-of-place 1DY Real-to-Complex FFT. The matrix is first
 * X<->Z transposed followed by the 1D FFT. The matrix is left in the transposed
 * format.
 * @param [in] InMatrix - Input data for the forward FFT.
 * @throw runtime_error if the plan is not valid.
 */
void TCUFFTComplexMatrix::Compute_FFT_1DZ_R2C(TRealMatrix& InMatrix)
{
  /// Transpose a real 3D matrix in the X-Z direction
  dim3 DimSizes(InMatrix.GetDimensionSizes().X, InMatrix.GetDimensionSizes().Y, InMatrix.GetDimensionSizes().Z);
  TCUDAImplementations::GetInstance()->TrasposeReal3DMatrixXZ(pdMatrixData, InMatrix.GetRawDeviceData(), DimSizes);

  // Compute forward cuFFT (if the plan does not exist, it also returns error).
  // the FFT is calculated in-place (may be a bit slower than out-of-place, however
  // it does not request additional transfers and memory).
  cufftResult_t cufftError = cufftExecR2C(cufft_plan_1DZ_R2C,
                                          static_cast<cufftReal*>(pdMatrixData),
                                          reinterpret_cast<cufftComplex*>(pdMatrixData));

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Execute_FFT_1DZ_R2C");
}// end of Compute_FFT_1DZ_R2C
//------------------------------------------------------------------------------

/**
 * Computer inverse out-of-place 1DX Real-to-Complex FFT.
 * @param [out] OutMatrix - Input data for the inverse FFT (real matrix).
 * @throw runtime_error if the plan is not valid.
 */
void TCUFFTComplexMatrix::Compute_FFT_1DX_C2R(TRealMatrix& OutMatrix)
{
  //Compute inverse cuFFT (if the plan does not exist, it also returns error)
  cufftResult_t cufftError = cufftExecC2R(cufft_plan_1DX_C2R,
                                          reinterpret_cast<cufftComplex*>(pdMatrixData),
                                          static_cast<cufftReal*> (OutMatrix.GetRawDeviceData()));

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Execute_FFT_1DX_C2R");
}// end of Compute_FFT_1DX_C2R
//------------------------------------------------------------------------------


/**
 * Computer inverse out-of-place 1DY Real-to-Complex FFT. The matrix is requested to
 * be in the transposed layout. After the FFT is calculated, an Y<->X transposed follows.
 * The matrix is returned in the normal layout (z, y, x)
 * format.
 * @param [out] OutMatrix - Input data for the forward FFT.
 * @throw runtime_error if the plan is not valid.
 */
void TCUFFTComplexMatrix::Compute_FFT_1DY_C2R(TRealMatrix& OutMatrix)
{
  // Compute forward cuFFT (if the plan does not exist, it also returns error).
  // the FFT is calculated in-place (may be a bit slower than out-of-place, however
  // it does not request additional transfers and memory).
  cufftResult_t cufftError = cufftExecC2R(cufft_plan_1DY_C2R,
                                          reinterpret_cast<cufftComplex*>(pdMatrixData),
                                          static_cast<cufftReal*>(pdMatrixData));

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Execute_FFT_1DY_C2R");

  /// Transpose a real 3D matrix back in the X-Y direction
  dim3 DimSizes(OutMatrix.GetDimensionSizes().Y,
                OutMatrix.GetDimensionSizes().X,
                OutMatrix.GetDimensionSizes().Z);

  TCUDAImplementations::GetInstance()->TrasposeReal3DMatrixXY(OutMatrix.GetRawDeviceData(),
                                                              pdMatrixData,
                                                              DimSizes);
}// end of Compute_FFT_1DY_C2R
//------------------------------------------------------------------------------


/**
 * Computer forward out-of-place 1DY Real-to-Complex FFT. The matrix is requested
 * to  be in the transposed layout. After the FFT is calculated, an Z<->X transposed follows.
 * The matrix is returned in the normal layout (z, y, x).
 * @param [out] OutMatrix - Input data for the forward FFT.
 * @throw runtime_error if the plan is not valid.
 */
void TCUFFTComplexMatrix::Compute_FFT_1DZ_C2R(TRealMatrix& OutMatrix)
{
  // Compute forward cuFFT (if the plan does not exist, it also returns error).
  // the FFT is calculated in-place (may be a bit slower than out-of-place, however
  // it does not request additional transfers and memory).
  cufftResult_t cufftError = cufftExecC2R(cufft_plan_1DZ_C2R,
                                          reinterpret_cast<cufftComplex*>(pdMatrixData),
                                          static_cast<cufftReal*>(pdMatrixData));

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Execute_FFT_1DZ_C2R");

  /// Transpose a real 3D matrix in the Z<->X direction
  dim3 DimSizes(OutMatrix.GetDimensionSizes().Z, OutMatrix.GetDimensionSizes().Y, OutMatrix.GetDimensionSizes().X);
  TCUDAImplementations::GetInstance()->TrasposeReal3DMatrixXZ(OutMatrix.GetRawDeviceData(), GetRawDeviceData(), DimSizes);
}// end of Compute_FFT_1DZ_C2R
//------------------------------------------------------------------------------


//----------------------------------------------------------------------------//
//                          Protected methods                                 //
//----------------------------------------------------------------------------//


//----------------------------------------------------------------------------//
//                              Private methods                               //
//----------------------------------------------------------------------------//

/**
 * Throw cuda FFT exception
 * @param [in] cufftError        - CUDA FFT error code
 * @param [in] TransformTypeName - CUDA transform type name
 * @throw runtime error
 */
void TCUFFTComplexMatrix::ThrowCUFFTException(const cufftResult cufftError,
                                              const char * TransformTypeName)
{
  char ErrorMessage[256];

  if (cuFFTErrorMessages.find(cufftError) != cuFFTErrorMessages.end())
  {
    sprintf(ErrorMessage, cuFFTErrorMessages[cufftError], TransformTypeName);
  }
  else // unknown error
  {
    sprintf(ErrorMessage, CUFFTComplexMatrix_ERR_FMT_CUFFT_UNKNOWN_ERROR, TransformTypeName);
  }

  // Throw exception
  throw std::runtime_error(ErrorMessage);
}// end of GetCuFFTErrorMessage
//------------------------------------------------------------------------------