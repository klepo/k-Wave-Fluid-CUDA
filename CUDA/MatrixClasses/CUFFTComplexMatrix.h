//
//  CUFFTComplexMatrix.h
//  kspace
//
//  Created by Beau Johnston on 21/10/12.
//  Copyright (c) 2012 Beau Johnston. All rights reserved.
//

#ifndef kspace_CUFFTComplexMatrix_h
#define kspace_CUFFTComplexMatrix_h

#include <cuda_runtime.h>
#include <cufft.h>

#include "../../MatrixClasses/ComplexMatrix.h"

/*
 * @class TFFTWComplexMatrix
 * @brief Class implementing 3D Real-To-Complex and Complex-To-Real transforms
 *      using FFTW interface
 */
class TCUFFTComplexMatrix : public TComplexMatrix
{
    public:
        //Constructor
        TCUFFTComplexMatrix(struct TDimensionSizes DimensionSizes);
        //Destructor
        virtual ~TCUFFTComplexMatrix();

        //Create FFTW plan for Real-to-Complex
        static void CreateFFTPlan3D_R2C(TRealMatrix& InMatrix);
        //Create FFTW plan for Complex-to-Real
        static void CreateFFTPlan3D_C2R(TRealMatrix& OutMatrix);

        //Compute 3D FFT Real-to-Complex
        void Compute_FFT_3D_R2C(TRealMatrix& InMatrix);
        //Compute 3D FFT Complex-to-Real
        void Compute_iFFT_3D_C2R(TRealMatrix& OutMatrix);

        //override the default TBaseFloatMatrix::GetRawDeviceData() as that
        //returns pdMatrixData
        float* GetRawDeviceData()
        {
            return (float*)pdMatrixData;
        }

    protected:

        //Copy constructor not allowed for public
        TCUFFTComplexMatrix(const TCUFFTComplexMatrix& original);

        static cufftHandle cufft_plan_RealToComplex;
        static cufftHandle cufft_plan_ComplexToReal;

        //Allocate memory for the FFTW matrix
        virtual void AllocateMemory();
        //Free memory of the FFTW matrix
        virtual void FreeMemory();

    private:
        bool FlagChecker(cufftResult flag);

};// TCUFFTComplexMatrix

#endif

