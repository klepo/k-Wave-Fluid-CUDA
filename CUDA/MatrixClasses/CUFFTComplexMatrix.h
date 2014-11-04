/**
 * @file        CUFFTComplexMatrix.h
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the class that implements
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

