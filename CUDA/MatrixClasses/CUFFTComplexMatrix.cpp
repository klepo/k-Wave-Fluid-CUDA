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
 *              04 November  2014, 14:44 (revised)
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

#include "CUFFTComplexMatrix.h"
#include "../../MatrixClasses/RealMatrix.h"
#include "../../Utils/ErrorMessages.h"


#include <iostream>
#include <string>
#include <assert.h>

#include <malloc.h>


using namespace std;

#define confirmRun 0

inline void gpuAssert(cudaError_t code,
                      string file,
                      int line,
                      bool Abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,
                "GPUassert: %s %s %d\n",
                cudaGetErrorString(code),file.c_str(),line);
        if (Abort) exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

//--------------------------------------------------------------------------//
//                             Constants                                    //
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
//                       Static Member Variables                            //
//--------------------------------------------------------------------------//
cufftHandle TCUFFTComplexMatrix::cufft_plan_RealToComplex = NULL;
cufftHandle TCUFFTComplexMatrix::cufft_plan_ComplexToReal = NULL;

//--------------------------------------------------------------------------//
//                             Public methods                               //
//--------------------------------------------------------------------------//

/*
 * Constructor
 * @param DimensionSizes - Dimension sizes of the reduced complex matrix
 */
TCUFFTComplexMatrix::TCUFFTComplexMatrix(struct TDimensionSizes DimensionSizes)
{

    pdMatrixData = NULL;
    InitDimensions(DimensionSizes);
    AllocateMemory();

}// end of TCUFFTComplexMatrix
//----------------------------------------------------------------------------

/*
 * Destructor
 */
TCUFFTComplexMatrix::~TCUFFTComplexMatrix()
{

    //-- Destroy cufft plans --//
    FreeMemory();

    if (cufft_plan_RealToComplex){
        cufftDestroy(cufft_plan_RealToComplex);
    }

    if (cufft_plan_ComplexToReal){
        cufftDestroy(cufft_plan_ComplexToReal);
    }

}// end of ~TCUFFTComplexMatrix()
//----------------------------------------------------------------------------

/*
 * Create FFTW plan for Real-to-Complex.
 * @param [in,out] InMatrix  - RealMatrix of which to create the plan
 */
void TCUFFTComplexMatrix::CreateFFTPlan3D_R2C(TRealMatrix& InMatrix)
{
    //cout << "Creating plan... watch memory here!" << endl;
    //LogPreAlloc();

    cufftResult_t cufftError = cufftPlan3d(
            &cufft_plan_RealToComplex,
            static_cast<int>(InMatrix.GetDimensionSizes().Z),
            static_cast<int>(InMatrix.GetDimensionSizes().Y),
            static_cast<int>(InMatrix.GetDimensionSizes().X),
            CUFFT_R2C);

    if(cufftError != CUFFT_SUCCESS){
        cout << "The plan is wrong" << endl;
    }

    //cuFFT compatibility with FFTW settings
    cufftSetCompatibilityMode(cufft_plan_RealToComplex,
                              //CUFFT_COMPATIBILITY_FFTW_ALL);
                              CUFFT_COMPATIBILITY_NATIVE);
    //cout << "Created plan... watch memory here!" << endl;
    //LogPostAlloc();

}// end of CreateFFTPlan3D_RealToComplex
//----------------------------------------------------------------------------

/*
 * Create FFTW plan for Complex-to-Real.
 * @param [in, out] OutMatrix - RealMatrix of which to create the plan.
 */
void TCUFFTComplexMatrix::CreateFFTPlan3D_C2R(TRealMatrix& InMatrix)
{
    //cout << "Creating plan... watch memory here!" << endl;
    //LogPreAlloc();

    cufftResult_t cufftError = cufftPlan3d(
            &cufft_plan_ComplexToReal,
            static_cast<int>(InMatrix.GetDimensionSizes().Z),
            static_cast<int>(InMatrix.GetDimensionSizes().Y),
            static_cast<int>(InMatrix.GetDimensionSizes().X),
            CUFFT_C2R);

    if(cufftError != CUFFT_SUCCESS){
        cout << "The plan is wrong" << endl;
    }

    //cuFFT compatibility with FFTW settings
    cufftSetCompatibilityMode(cufft_plan_ComplexToReal,
                              //CUFFT_COMPATIBILITY_FFTW_ALL);
                              CUFFT_COMPATIBILITY_NATIVE);

    //cout << "Created plan... watch memory here!" << endl;
    //LogPostAlloc();

}//end of CreateFFTPlan3D_ComplexToReal
//----------------------------------------------------------------------------

void TCUFFTComplexMatrix::Compute_FFT_3D_R2C(TRealMatrix& InMatrix)
{
    //Compute the cuFFT
    cufftResult_t cufftError =
        cufftExecR2C(cufft_plan_RealToComplex,
                     static_cast<cufftReal*>(InMatrix.GetRawDeviceData()),
                     reinterpret_cast<cufftComplex*>(pdMatrixData));

    //ensure there were no errors
    if (!FlagChecker(cufftError)) {
        cerr << "TCUFFTComplexMatrix FFT error" << endl;
        throw bad_exception();
    }

}// end of Compute_FFT_3D_r2c_GPU
//----------------------------------------------------------------------------

void TCUFFTComplexMatrix::Compute_iFFT_3D_C2R(TRealMatrix& OutMatrix)
{

    //perform ifft
    cufftResult_t cufftError =
        cufftExecC2R(cufft_plan_ComplexToReal,
                     reinterpret_cast<cufftComplex*>(pdMatrixData),
                     static_cast<cufftReal*>(OutMatrix.GetRawDeviceData()));

    //check for errors
    if (!FlagChecker(cufftError)) {
        cerr << "TCUFFTComplexMatrix iFFT error" << endl;
        throw bad_exception();
    }
}// end of Compute_iFFT_3D_c2r_GPU
//----------------------------------------------------------------------------

//--------------------------------------------------------------------------//
//                          Protected methods                               //
//--------------------------------------------------------------------------//

/*
 * Allocate Memory using fftwf_malloc function to ensure correct alignment
 */
void TCUFFTComplexMatrix::AllocateMemory()
{
    assert(pMatrixData == NULL);
    assert(pdMatrixData == NULL);
    /* No memory allocated before this function*/


    //size of memory to allocate
    size_t size_in_bytes = pTotalAllocatedElementCount*sizeof(float);


    //Allocate memory
    pMatrixData = static_cast<float*>(malloc(size_in_bytes));

    if (!pMatrixData) {
        fprintf(stderr,Matrix_ERR_FMT_Not_Enough_Memory, "TBaseFloatMatrix");
        throw bad_alloc();
    }

    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&pdMatrixData),
                         size_in_bytes));

    if (!pdMatrixData) {
        cerr << "TCUFFTComplexMatrix Matrix_ERR_FMT_Not_Enough_Memory" << endl;
        throw bad_alloc();
    }


}// end of virtual bool FreeMemory
//----------------------------------------------------------------------------

/*
 * Free memory using fftwf_free
 */
void TCUFFTComplexMatrix::FreeMemory()
{

    if (pMatrixData){
        free(pMatrixData);
    }
    pMatrixData = NULL;

    if (pdMatrixData) {
        gpuErrchk(cudaFree(pdMatrixData));
    }
    pdMatrixData = NULL;

}// end of FreeMemory
//----------------------------------------------------------------------------

//--------------------------------------------------------------------------//
//                          Private methods                                 //
//--------------------------------------------------------------------------//
bool TCUFFTComplexMatrix::FlagChecker(cufftResult flag)
{

    switch (flag) {
        case CUFFT_SUCCESS:
            //cout << "successful with CUFFT_SUCCESS" << endl;
            return true;
            break;

        case CUFFT_INVALID_PLAN:
            cout << "failed with CUFFT_INVALID_PLAN" << endl;
            return false;
            break;

        case CUFFT_ALLOC_FAILED:
            cout << "failed with CUFFT_ALLOC_FAILED" << endl;
            return false;
            break;

        case CUFFT_INVALID_VALUE:
            cout << "failed with CUFFT_INVALID_VALUE" << endl;
            return false;
            break;

        case CUFFT_INTERNAL_ERROR:
            cout << "failed with CUFFT_INTERNAL_ERROR" << endl;
            return false;
            break;

        case CUFFT_EXEC_FAILED:
            cout << "failed with CUFFT_EXEC_FAILED" << endl;
            return false;
            break;

        case CUFFT_SETUP_FAILED:
            cout << "failed with CUFFT_SETUP_FAILED" << endl;
            return false;
            break;

        case CUFFT_INVALID_SIZE:
            cout << "failed with CUFFT_INVALID_SIZE" << endl;
            return false;
            break;

        case CUFFT_UNALIGNED_DATA:
            cout << "failed with CUFFT_UNALIGNED_DATA" << endl;
            return false;
            break;

        default:
            cout << "failed with SOMETHING," <<
                " but I don't recognise the flag" << endl;
            return false;
            break;
    }
}

