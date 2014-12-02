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
 * @version     kspaceFirstOrder3D 3.3
 * @date        09 August    2011, 13:10 (created) \n
 *              28 November  2014, 15:28 (revised)
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
 * /**
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

  cuFFTErrorMessages.clear();
}// end of DestroyAllPlans
//------------------------------------------------------------------------------


/**
 * Computer forward out-of place 3D Real-to-Complex FFT.
 * @param [in] InMatrix - Input data for the forward FFT
 * @throw runtime_error if the plan is not valid.
 */
void TCUFFTComplexMatrix::Compute_FFT_3D_R2C(TRealMatrix& InMatrix)
{
  //Compute forward cuFFT (if the plan does not exist, it also returns error)
  cufftResult_t cufftError =
          cufftExecR2C(cufft_plan_3D_R2C,
                       static_cast<cufftReal*>(InMatrix.GetRawDeviceData()),
                       reinterpret_cast<cufftComplex*>(pdMatrixData));

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Execute_FFT_3D_R2C");
}// end of Compute_FFT_3D_R2C
//------------------------------------------------------------------------------



/**
 * Computer forward out-of place 3D Complex-to-Real FFT.
 * @param [out] OutMatrix - output of the inverse FFT.
 * @throw runtime_error if the plan is not valid.
 */
void TCUFFTComplexMatrix::Compute_FFT_3D_C2R(TRealMatrix& OutMatrix)
{
  //Compute forward cuFFT (if the plan does not exist, it also returns error)
  cufftResult_t cufftError =
          cufftExecC2R(cufft_plan_3D_C2R,
                       reinterpret_cast<cufftComplex*>(pdMatrixData),
                       static_cast<cufftReal*>(OutMatrix.GetRawDeviceData()));

  if (cufftError != CUFFT_SUCCESS) ThrowCUFFTException(cufftError, "Execute_FFT_3D_RCR");

}// end of Compute_iFFT_3D_c2r_GPU
//----------------------------------------------------------------------------

//--------------------------------------------------------------------------//
//                          Protected methods                               //
//--------------------------------------------------------------------------//


//----------------------------------------------------------------------------//
//                              Private methods                               //
//----------------------------------------------------------------------------//

/**
 * Throw cuda FFT exception
 * @param [in] cufftError        - cuda FFT error code
 * @param [in] TransformTypeName - Cuda transform type name
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