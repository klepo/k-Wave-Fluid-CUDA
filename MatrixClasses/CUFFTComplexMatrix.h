/**
 * @file        CUFFTComplexMatrix.h
 *
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the class that implements 3D FFT using the cuFFT
 *              interface.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        09 August    2011, 13:10 (created) \n
 *              17 July      2017, 16:14 (revised)
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

#ifndef CUFFT_COMPLEX_MATRIX_H
#define CUFFT_COMPLEX_MATRIX_H

#include <map>
#include <cufft.h>

#include <MatrixClasses/ComplexMatrix.h>
#include <Logger/ErrorMessages.h>

/**
 * @class   TCUFFTComplexMatrix
 * @brief   Class implementing 3D Real-To-Complex and Complex-To-Real transforms using CUDA
 *          FFT interface.
 * @details Class implementing 3D Real-To-Complex and Complex-To-Real transforms using CUDA
 *          FFT interface.
 *
 */
class TCUFFTComplexMatrix : public TComplexMatrix
{
  public:
    /// Default constructor not allowed.
    TCUFFTComplexMatrix() = delete;
    /// Constructor (inherited from TComplexMatrix).
    TCUFFTComplexMatrix(const DimensionSizes& DimensionSizes) : TComplexMatrix(DimensionSizes) {};
    /// Copy constructor not allowed.
    TCUFFTComplexMatrix(const TCUFFTComplexMatrix&) = delete;
    /// Destructor (Inherited from TComplexMatrix).
    virtual ~TCUFFTComplexMatrix(){};

    /// operator= is not allowed.
    TCUFFTComplexMatrix& operator=(const TCUFFTComplexMatrix&) = delete;

    /// Create static cuFFT plan for Real-to-Complex.
    static void Create_FFT_Plan_3D_R2C(const DimensionSizes& inMatrixDims);
    /// Create static cuFFT plan for Complex-to-Real.
    static void Create_FFT_Plan_3D_C2R(const DimensionSizes& outMatrixDims);

    /// Create static cuFFT plan for Real-to-Complex in the X dimension.
    static void Create_FFT_Plan_1DX_R2C(const DimensionSizes& inMatrixDims);
    /// Create static cuFFT plan for Real-to-Complex in the Y dimension.
    static void Create_FFT_Plan_1DY_R2C(const DimensionSizes& inMatrixDims);
    /// Create static cuFFT plan for Real-to-Complex in the Z dimension.
    static void Create_FFT_Plan_1DZ_R2C(const DimensionSizes& inMatrixDims);

    /// Create static cuFFT plan for Complex-to-Real in the X dimension.
    static void Create_FFT_Plan_1DX_C2R(const DimensionSizes& outMatrixDims);
    /// Create static cuFFT plan for Complex-to-Real in the Y dimension.
    static void Create_FFT_Plan_1DY_C2R(const DimensionSizes& outMatrixDims);
    /// Create static cuFFT plan for Complex-to-Real in the Z dimension.
    static void Create_FFT_Plan_1DZ_C2R(const DimensionSizes& outMatrixDims);

    /// Destroy all static plans and error messages.
    static void DestroyAllPlansAndStaticData();

    /// Compute 3D out-of-place Real-to-Complex FFT.
    void Compute_FFT_3D_R2C(TRealMatrix& inMatrix);
    /// Compute 3D out-of-place Complex-to-Real FFT.
    void Compute_FFT_3D_C2R(TRealMatrix& outMatrix);

    /// Compute 1D out-of-place Real-to-Complex FFT in the X dimension.
    void Compute_FFT_1DX_R2C(TRealMatrix& inMatrix);
    /// Compute 1D out-of-place Real-to-Complex FFT in the Y dimension.
    void Compute_FFT_1DY_R2C(TRealMatrix& inMatrix);
    /// Compute 1D out-of-place Real-to-Complex FFT in the Z dimension.
    void Compute_FFT_1DZ_R2C(TRealMatrix& inMatrix);

    /// Compute 1D out-of-place Complex-to-Real FFT in the X dimension.
    void Compute_FFT_1DX_C2R(TRealMatrix& outMatrix);
    /// Compute 1D out-of-place Complex-to-Real FFT in the Y dimension.
    void Compute_FFT_1DY_C2R(TRealMatrix& outMatrix);
    /// Compute 1D out-of-place Complex-to-Real FFT in the Z dimension.
    void Compute_FFT_1DZ_C2R(TRealMatrix& outMatrix);


  protected:
    /// cuFFT plan for the 3D Real-to-Complex transform.
    static cufftHandle cufftPlan_3D_R2C;
    /// cuFFT plan for the 3D Complex-to-Real transform.
    static cufftHandle cufftPlan_3D_C2R;

    /// cuFFT plan for the 1D Real-to-Complex transform in the X dimension.
    static cufftHandle cufftPlan_1DX_R2C;
    /// cuFFT plan for the 3D Real-to-Complex transform in the Y dimension.
    static cufftHandle cufftPlan_1DY_R2C;
    /// cuFFT plan for the 3D Real-to-Complex transform in the Z dimension.
    static cufftHandle cufftPlan_1DZ_R2C;

    /// cuFFT plan for the 3D Complex-to-Real transform in the X dimension.
    static cufftHandle cufftPlan_1DX_C2R;
    /// cuFFT plan for the 3D Complex-to-Real transform in the Y dimension.
    static cufftHandle cufftPlan_1DY_C2R;
    /// cuFFT plan for the 3Z Complex-to-Real transform in the Z dimension.
    static cufftHandle cufftPlan_1DZ_C2R;

  private:

   /// Throw an exception with a given error message
   static void ThrowCUFFTException(const cufftResult   cufftError,
                                   const std::string&  transformTypeName);

   static  std::map<cufftResult, ErrorMessage> cuFFTErrorMessages;

};// TCUFFTComplexMatrix

#endif /* CUFFT_COMPLEX_MATRIX_H */
