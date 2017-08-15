/**
 * @file        CufftComplexMatrix.cpp
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
 *              10 August    2017, 16:41 (revised)
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

#include <MatrixClasses/CufftComplexMatrix.h>
#include <MatrixClasses/RealMatrix.h>
#include <Logger/Logger.h>
#include <KSpaceSolver/SolverCudaKernels.cuh>


//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Constants -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Initialisation ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


cufftHandle CufftComplexMatrix::sR2CFftPlan3D = cufftHandle();
cufftHandle CufftComplexMatrix::sC2RFftPlan3D = cufftHandle();

cufftHandle CufftComplexMatrix::sR2CFftPlan1DX = cufftHandle();
cufftHandle CufftComplexMatrix::sR2CFftPlan1DY = cufftHandle();
cufftHandle CufftComplexMatrix::sR2CFftPlan1DZ = cufftHandle();
cufftHandle CufftComplexMatrix::sC2RFftPlan1DX = cufftHandle();
cufftHandle CufftComplexMatrix::sC2RFftPlan1DY = cufftHandle();
cufftHandle CufftComplexMatrix::sC2RFftPlan1DZ = cufftHandle();


/**
 * Error message for the CUFFT class.
 */
std::map<cufftResult, ErrorMessage> CufftComplexMatrix::sCufftErrorMessages
{
  {CUFFT_INVALID_PLAN             , kErrFmtCufftInvalidPlan},
  {CUFFT_ALLOC_FAILED             , kErrFmtCufftAllocFailed},
  {CUFFT_INVALID_TYPE             , kErrFmtCufftInvalidType},
  {CUFFT_INVALID_VALUE            , kErrFmtCufftInvalidValue},
  {CUFFT_INTERNAL_ERROR           , kErrFmtCuFFTInternalError},
  {CUFFT_EXEC_FAILED              , kErrFmtCufftExecFailed},
  {CUFFT_SETUP_FAILED             , kErrFmtCufftSetupFailed},
  {CUFFT_INVALID_SIZE             , kErrFmtCufftInvalidSize},
  {CUFFT_UNALIGNED_DATA           , kErrFmtCufftUnalignedData},
  {CUFFT_INCOMPLETE_PARAMETER_LIST, kErrFmtCufftIncompleteParaterList},
  {CUFFT_INVALID_DEVICE           , kErrFmtCufftInvalidDevice},
  {CUFFT_PARSE_ERROR              , kErrFmtCufftParseError},
  {CUFFT_NO_WORKSPACE             , kErrFmtCufftNoWorkspace},
  {CUFFT_NOT_IMPLEMENTED          , kErrFmtCufftNotImplemented},
  {CUFFT_LICENSE_ERROR            , kErrFmtCufftLicenseError},
  {CUFFT_NOT_SUPPORTED            , kErrFmtCufftNotSupported}
};
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Create an cuFFT plan for 3D Real-to-Complex.
 */
void CufftComplexMatrix::createR2CFftPlan3D(const DimensionSizes& inMatrixDims)
{
  cufftResult cufftError = cufftPlan3d(&sR2CFftPlan3D,
                                       static_cast<int>(inMatrixDims.nz),
                                       static_cast<int>(inMatrixDims.ny),
                                       static_cast<int>(inMatrixDims.nx),
                                       CUFFT_R2C);

  if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtCreateR2CFftPlan3D);
}// end of createR2CFftPlan3D
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create cuFFT plan for Complex-to-Real.
 */
void CufftComplexMatrix::createC2RFftPlan3D(const DimensionSizes& outMatrixDims)
{
  cufftResult_t cufftError = cufftPlan3d(&sC2RFftPlan3D,
                                         static_cast<int>(outMatrixDims.nz),
                                         static_cast<int>(outMatrixDims.ny),
                                         static_cast<int>(outMatrixDims.nx),
                                         CUFFT_C2R);

  if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtCreateC2RFftPlan3D);
}//end of createC2RFftPlan3D
//----------------------------------------------------------------------------------------------------------------------


/**
 * Create cuFFT plan for 1DX Real-to-Complex.
 */
void CufftComplexMatrix::createR2CFftPlan1DX(const DimensionSizes& inMatrixDims)
{
  // set dimensions
  const int nx   = static_cast<int>(inMatrixDims.nx);
  const int ny   = static_cast<int>(inMatrixDims.ny);
  const int nz   = static_cast<int>(inMatrixDims.nz);
  const int nxR = ((nx / 2) + 1);

  // set up rank and strides
  int rank = 1;
  int n[] = {nx};

  // Since runs out-of-place no padding is needed.
  int inembed[] = {nx};
  int istride   = 1;
  int idist     = nx;

  int onembed[] = {nxR};
  int ostride   = 1;
  int odist     = nxR;

  int batch = ny * nz;

  // plan the FFT
  cufftResult_t cufftError = cufftPlanMany(&sR2CFftPlan1DX, rank, n,
                                           inembed, istride, idist,
                                           onembed, ostride, odist,
                                           CUFFT_R2C, batch);

  if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtCreateR2CFftPlan1DX);
}//end of createR2CFftPlan1DX
//----------------------------------------------------------------------------------------------------------------------


/**
 * Create cuFFT plan for 1DY Real-to-Complex.
 */
void CufftComplexMatrix::createR2CFftPlan1DY(const DimensionSizes& inMatrixDims)
{
  // set dimensions
  const int nx   = static_cast<int> (inMatrixDims.nx);
  const int ny   = static_cast<int> (inMatrixDims.ny);
  const int nz   = static_cast<int> (inMatrixDims.nz);
  const int nyR = ((ny / 2) + 1);

  // set up rank and strides
  int rank = 1;
  int n[] = {ny};

  // The input matrix is transposed with every row padded by a single element.
  int inembed[] = {ny +1};
  int istride   = 1;
  int idist     = ny + 1;

  int onembed[] = {nyR};
  int ostride   = 1;
  int odist     = nyR;

  int batch =  nx * nz;

  cufftResult_t cufftError = cufftPlanMany(&sR2CFftPlan1DY, rank, n,
                                           inembed, istride, idist,
                                           onembed, ostride, odist,
                                           CUFFT_R2C, batch);

  if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtCreateR2CFftPlan1DY);
}//end of createR2CFftPlan1DY
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create cuFFT plan for 1DZ Real-to-Complex.
 */
void CufftComplexMatrix::createR2CFftPlan1DZ(const DimensionSizes& inMatrixDims)
{
  const int nx   = static_cast<int> (inMatrixDims.nx);
  const int ny   = static_cast<int> (inMatrixDims.ny);
  const int nz   = static_cast<int> (inMatrixDims.nz);
  const int nzR = ((nz / 2) + 1);

  // set up rank and strides
  int rank = 1;
  int n[] = {nz};

  // The input matrix is transposed with every row padded by a single element.
  int inembed[] = {nz + 1};
  int istride   = 1;
  int idist     = nz + 1;

  int onembed[] = {nzR};
  int ostride   = 1;
  int odist     = nzR;

  int batch =  nx * ny;

  cufftResult_t cufftError = cufftPlanMany(&sR2CFftPlan1DZ, rank, n,
                                           inembed, istride, idist,
                                           onembed, ostride, odist,
                                           CUFFT_R2C, batch);

  if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtCreateR2CFftPlan1DZ);

}//end of createR2CFftPlan1DZ
//----------------------------------------------------------------------------------------------------------------------



/**
 * Create cuFFT plan for 1DX Complex-to-Real.
 */
void CufftComplexMatrix::createC2RFftPlan1DX(const DimensionSizes& outMatrixDims)
{
  // set dimensions
  const int nx   = static_cast<int> (outMatrixDims.nx);
  const int ny   = static_cast<int> (outMatrixDims.ny);
  const int nz   = static_cast<int> (outMatrixDims.nz);
  const int nxR = ((nx / 2) + 1);

  // set up rank and strides
  int rank = 1;
  int n[] = {nx};

  // Since runs out-of-place no padding is needed.
  int inembed[] = {nxR};
  int istride   = 1;
  int idist     = nxR;

  int onembed[] = {nx};
  int ostride   = 1;
  int odist     = nx;

  int batch = ny * nz;

  cufftResult_t cufftError = cufftPlanMany(&sC2RFftPlan1DX, rank, n,
                                           inembed, istride, idist,
                                           onembed, ostride, odist,
                                           CUFFT_C2R, batch);

  if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtCreateC2RFftPlan1DX);
}//end of createC2RFftPlan1DX
//----------------------------------------------------------------------------------------------------------------------


/**
 * Create cuFFT plan for 1DY Complex-to-Real.
 */
void CufftComplexMatrix::createC2RFftPlan1DY(const DimensionSizes& outMatrixDims)
{
  // set dimensions
  const int nx   = static_cast<int> (outMatrixDims.nx);
  const int ny   = static_cast<int> (outMatrixDims.ny);
  const int nz   = static_cast<int> (outMatrixDims.nz);
  const int nyR = ((ny / 2) + 1);

  // set up rank and strides
  int rank = 1;
  int n[] = {ny};

  int inembed[] = {nyR};
  int istride   = 1;
  int idist     = nyR;

  // The output matrix is transposed with every row padded by a single element.
  int onembed[] = {ny + 1};
  int ostride   = 1;
  int odist     = ny + 1;

  int batch =  nx * nz;

  cufftResult_t cufftError = cufftPlanMany(&sC2RFftPlan1DY, rank, n,
                                           inembed, istride, idist,
                                           onembed, ostride, odist,
                                           CUFFT_C2R, batch);

  if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtCreateC2RFftPlan1DY);
}//end of Create_FFT_Plan_1DY_R2C
//----------------------------------------------------------------------------------------------------------------------


/**
 * Create cuFFT plan for 1DZ Complex-to-Real.
 */
void CufftComplexMatrix::createC2RFftPlan1DZ(const DimensionSizes& outMatrixDims)
{
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

  cufftResult_t cufftError = cufftPlanMany(&sC2RFftPlan1DZ, rank, n,
                                           inembed, istride, idist,
                                           onembed, ostride, odist,
                                           CUFFT_C2R, batch);

  if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtCreateC2RFftPlan1DZ);
}//end of Create_FFT_Plan_1DX_R2C
//----------------------------------------------------------------------------------------------------------------------


/**
 * Destroy all static plans created by the application.
 */
void CufftComplexMatrix::destroyAllPlansAndStaticData()
{
  cufftResult_t cufftError;

  if (sR2CFftPlan3D)
  {
    cufftError = cufftDestroy(sR2CFftPlan3D);
    sR2CFftPlan3D = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtDestroyR2CFftPlan3D);
  }

  if (sC2RFftPlan3D)
  {
    cufftError = cufftDestroy(sC2RFftPlan3D);
    sC2RFftPlan3D = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtDestroyC2RFftPlan3D);
  }

  if (sR2CFftPlan1DX)
  {
    cufftError = cufftDestroy(sR2CFftPlan1DX);
    sR2CFftPlan1DX = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtDestroyR2CFftPlan1DX);
  }

  if (sR2CFftPlan1DY)
  {
    cufftError = cufftDestroy(sR2CFftPlan1DY);
    sR2CFftPlan1DY = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtDestroyR2CFftPlan1DY);
  }

  if (sR2CFftPlan1DZ)
  {
    cufftError = cufftDestroy(sR2CFftPlan1DZ);
    sR2CFftPlan1DZ = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtDestroyR2CFftPlan1DZ);
  }

  if (sC2RFftPlan1DX)
  {
    cufftError = cufftDestroy(sC2RFftPlan1DX);
    sC2RFftPlan1DX = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtDestroyC2RFftPlan1DX);
  }

  if (sC2RFftPlan1DY)
  {
    cufftError = cufftDestroy(sC2RFftPlan1DY);
    sC2RFftPlan1DY = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtDestroyC2RFftPlan1DY);
  }

  if (sC2RFftPlan1DZ)
  {
    cufftError = cufftDestroy(sC2RFftPlan1DZ);
    sC2RFftPlan1DZ = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtDestroyC2RFftPlan1DZ);
  }

  // clear static data
  sCufftErrorMessages.clear();
}// end of destroyAllPlansAndStaticData
//----------------------------------------------------------------------------------------------------------------------


/**
 * Computer forward out-of-place 3D Real-to-Complex FFT.
 */
void CufftComplexMatrix::computeR2CFft3D(RealMatrix& inMatrix)
{
  //Compute forward cuFFT (if the plan does not exist, it also returns error)
  cufftResult_t cufftError = cufftExecR2C(sR2CFftPlan3D,
                                          static_cast<cufftReal*>(inMatrix.getDeviceData()),
                                          reinterpret_cast<cufftComplex*>(mDeviceData));

  if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtExecuteR2CFftPlan3D);
}// end of computeR2CFft3D
//----------------------------------------------------------------------------------------------------------------------



/**
 * Computer forward out-of-place 3D Complex-to-Real FFT.
 */
void CufftComplexMatrix::computeC2RFft3D(RealMatrix& outMatrix)
{
  //Compute forward cuFFT (if the plan does not exist, it also returns error)
  cufftResult_t cufftError = cufftExecC2R(sC2RFftPlan3D,
                                          reinterpret_cast<cufftComplex*>(mDeviceData),
                                          static_cast<cufftReal*>(outMatrix.getDeviceData()));

  if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtExecuteC2RFftPlan3D);
}// end of computeC2RFft3D
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer forward out-of-place 1DX Real-to-Complex FFT.
 */
void CufftComplexMatrix::computeR2CFft1DX(RealMatrix& inMatrix)
{
  //Compute forward cuFFT (if the plan does not exist, it also returns error)
  cufftResult_t cufftError = cufftExecR2C(sR2CFftPlan1DX,
                                          static_cast<cufftReal*>(inMatrix.getDeviceData()),
                                          reinterpret_cast<cufftComplex*>(mDeviceData));

  if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtExecuteR2CFftPlan1DX);
}// end of computeR2CFft1DX
//----------------------------------------------------------------------------------------------------------------------


/**
 * Computer forward out-of-place 1DY Real-to-Complex FFT. The matrix is first X<->Y transposed
 * followed by the 1D FFT. The matrix is left in the transposed format.
 */
void CufftComplexMatrix::computeR2CFft1DY(RealMatrix& inMatrix)
{
  /// Transpose a real 3D matrix in the X-Y direction
  dim3 dimSizes(static_cast<unsigned int>(inMatrix.getDimensionSizes().nx),
                static_cast<unsigned int>(inMatrix.getDimensionSizes().ny),
                static_cast<unsigned int>(inMatrix.getDimensionSizes().nz));

  SolverCudaKernels::trasposeReal3DMatrixXY<SolverCudaKernels::TransposePadding::kOutput>
                                           (mDeviceData,
                                            inMatrix.getDeviceData(),
                                            dimSizes);

  // Compute forward cuFFT (if the plan does not exist, it also returns error).
  // the FFT is calculated in-place (may be a bit slower than out-of-place, however
  // it does not request additional transfers and memory).
  cufftResult_t cufftError = cufftExecR2C(sR2CFftPlan1DY,
                                          static_cast<cufftReal*>(mDeviceData),
                                          reinterpret_cast<cufftComplex*>(mDeviceData));

  if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtExecuteR2CFftPlan1DY);
}// end of computeR2CFft1DY
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer forward out-of-place 1DZ Real-to-Complex FFT.
 */
void CufftComplexMatrix::computeR2CFft1DZ(RealMatrix& inMatrix)
{
  /// Transpose a real 3D matrix in the X-Z direction
  dim3 dimSizes(static_cast<unsigned int>(inMatrix.getDimensionSizes().nx),
                static_cast<unsigned int>(inMatrix.getDimensionSizes().ny),
                static_cast<unsigned int>(inMatrix.getDimensionSizes().nz));

  SolverCudaKernels::trasposeReal3DMatrixXZ<SolverCudaKernels::TransposePadding::kOutput>
                                           (mDeviceData,
                                            inMatrix.getDeviceData(),
                                            dimSizes);

  // Compute forward cuFFT (if the plan does not exist, it also returns error).
  // the FFT is calculated in-place (may be a bit slower than out-of-place, however
  // it does not request additional transfers and memory).
  cufftResult_t cufftError = cufftExecR2C(sR2CFftPlan1DZ,
                                          static_cast<cufftReal*>(mDeviceData),
                                          reinterpret_cast<cufftComplex*>(mDeviceData));

  if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtExecuteR2CFftPlan1DZ);
}// end of computeR2CFft1DZ
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer inverse out-of-place 1DX Real-to-Complex FFT.
 */
void CufftComplexMatrix::computeC2RFft1DX(RealMatrix& outMatrix)
{
  //Compute inverse cuFFT (if the plan does not exist, it also returns error)
  cufftResult_t cufftError = cufftExecC2R(sC2RFftPlan1DX,
                                          reinterpret_cast<cufftComplex*>(mDeviceData),
                                          static_cast<cufftReal*>(outMatrix.getDeviceData()));

  if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtExecuteC2RFftPlan1DX);
}// end of computeC2RFft1DX
//----------------------------------------------------------------------------------------------------------------------


/**
 * Computer inverse out-of-place 1DY Real-to-Complex FFT.
 */
void CufftComplexMatrix::computeC2RFft1DY(RealMatrix& outMatrix)
{
  // Compute forward cuFFT (if the plan does not exist, it also returns error).
  // the FFT is calculated in-place (may be a bit slower than out-of-place, however
  // it does not request additional transfers and memory).
  cufftResult_t cufftError = cufftExecC2R(sC2RFftPlan1DY,
                                          reinterpret_cast<cufftComplex*>(mDeviceData),
                                          static_cast<cufftReal*>(mDeviceData));

  if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtExecuteC2RFftPlan1DY);

  /// Transpose a real 3D matrix back in the X-Y direction
  dim3 dimSizes(static_cast<unsigned int>(outMatrix.getDimensionSizes().ny),
                static_cast<unsigned int>(outMatrix.getDimensionSizes().nx),
                static_cast<unsigned int>(outMatrix.getDimensionSizes().nz));

  SolverCudaKernels::trasposeReal3DMatrixXY<SolverCudaKernels::TransposePadding::kInput>
                                           (outMatrix.getDeviceData(),
                                            mDeviceData,
                                            dimSizes);
}// end of computeC2RFft1DY
//----------------------------------------------------------------------------------------------------------------------


/**
 * Computer forward out-of-place 1DY Real-to-Complex FFT.
 */
void CufftComplexMatrix::computeC2RFft1DZ(RealMatrix& outMatrix)
{
  // Compute forward cuFFT (if the plan does not exist, it also returns error).
  // the FFT is calculated in-place (may be a bit slower than out-of-place, however
  // it does not request additional transfers and memory).
  cufftResult_t cufftError = cufftExecC2R(sC2RFftPlan1DZ,
                                          reinterpret_cast<cufftComplex*>(mDeviceData),
                                          static_cast<cufftReal*>(mDeviceData));

  if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtExecuteC2RFftPlan1DZ);

  /// Transpose a real 3D matrix in the Z<->X direction
  dim3 DimSizes(static_cast<unsigned int>(outMatrix.getDimensionSizes().nz),
                static_cast<unsigned int>(outMatrix.getDimensionSizes().ny),
                static_cast<unsigned int>(outMatrix.getDimensionSizes().nx));

  SolverCudaKernels::trasposeReal3DMatrixXZ<SolverCudaKernels::TransposePadding::kInput>
                                           (outMatrix.getDeviceData(),
                                            getDeviceData(),
                                            DimSizes);
}// end of computeC2RFft1DZ
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Protected methods ------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Throw cuda FFT exception.
 */
void CufftComplexMatrix::throwCufftException(const cufftResult  cufftError,
                                             const std::string& transformTypeName)
{
  std::string errMsg;
  if (sCufftErrorMessages.find(cufftError) != sCufftErrorMessages.end())
  {
    errMsg = Logger::formatMessage(sCufftErrorMessages[cufftError], transformTypeName.c_str());
  }
  else // unknown error
  {
    errMsg = Logger::formatMessage(kErrFmtCufftUnknownError, transformTypeName.c_str());
  }

  // Throw exception
  throw std::runtime_error(errMsg);
}// end of throwCufftException
//----------------------------------------------------------------------------------------------------------------------