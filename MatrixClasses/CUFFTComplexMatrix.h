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
 *              28 November  2014, 15:15 (revised)
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

#ifndef CUFFT_COMPLEX_MATRIX_H
#define CUFFT_COMPLEX_MATRIX_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <map>

#include <MatrixClasses/ComplexMatrix.h>
#include <Utils/ErrorMessages.h>

/**
 * @class TCUFFTComplexMatrix
 * @brief Class implementing 3D Real-To-Complex and Complex-To-Real transforms
 *      using CUDA FFT interface.
 * @details Class implementing 3D Real-To-Complex and Complex-To-Real transforms
 *      using CUDA FFT interface.
 *
 */
class TCUFFTComplexMatrix : public TComplexMatrix
{
  public:
    /// Constructor (inherited from TComplexMatrix).
    TCUFFTComplexMatrix(const TDimensionSizes& DimensionSizes) : TComplexMatrix(DimensionSizes) {};
    /// Destructor (Inherited from TComplexMatrix).
    virtual ~TCUFFTComplexMatrix(){};

    /// Create static FFTW plan for Real-to-Complex.
    static void Create_FFT_Plan_3D_R2C(const TDimensionSizes& InMatrixDims);
    /// Create static FFTW plan for Complex-to-Real.
    static void Create_FFT_Plan_3D_C2R(const TDimensionSizes& OutMatrixDims);

    /// Destroy all static plans and error messages.
    static void DestroyAllPlansAndStaticData();

    /// Compute 3D out-of-place Real-to-Complex FFT.
    void Compute_FFT_3D_R2C(TRealMatrix& InMatrix);
    /// Compute 3D out-of-place Complex-to-Real FFT.
    void Compute_FFT_3D_C2R(TRealMatrix& OutMatrix);


  protected:

    /// Copy constructor not allowed for public.
    TCUFFTComplexMatrix(const TCUFFTComplexMatrix& src);
    /// Operator = not allowed for public.
    TCUFFTComplexMatrix & operator = (const TCUFFTComplexMatrix& src);

    /// CUFFT plan for the 3D Real-to-Complex transform.
    static cufftHandle cufft_plan_3D_R2C;
    /// CUFFT plan for the 3D Real-to-Complex transform.
    static cufftHandle cufft_plan_3D_C2R;

  private:

   /// Throw an exception with a given error message
   static void ThrowCUFFTException(const cufftResult cufftError,
                                   const char * TransformTypeName);

   /// Static map with error messages for cuFFT matrix
   static  std::map<cufftResult, TErrorMessage> cuFFTErrorMessages;

};// TCUFFTComplexMatrix

#endif /* CUFFT_COMPLEX_MATRIX_H */
