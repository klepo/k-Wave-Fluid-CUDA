/**
 * @file        ComplexMatrix.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file with the class for complex matrices.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        11 July     2011, 14:02 (created) \n
 *              21 July     2016, 15:55 (revised)
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

#ifndef COMPLEX_MATRIX_H
#define	COMPLEX_MATRIX_H

#include <complex>

#include <MatrixClasses/BaseFloatMatrix.h>
#include <MatrixClasses/RealMatrix.h>

#include <Utils/DimensionSizes.h>

/**
 * @typedef TFloatComplex
 * @brief   C++ complex values
 * @details C++ complex values
 */
typedef std::complex<float> TFloatComplex;

/**
 * @class   TComplexMatrix
 * @brief   The class for complex matrices.
 * @details The class for complex matrices.
 */
class TComplexMatrix : public TBaseFloatMatrix
{
  public:
    /// Constructor.
    TComplexMatrix(const TDimensionSizes& dimensionSizes);
    /// Destructor.
    virtual ~TComplexMatrix();


    /**
     * @brief   Operator [].
     * @details Operator [].
     * @param [in] index - 1D index into the array
     * @return An element of the matrix
     */
    inline TFloatComplex& operator[](const size_t& index)
    {
      return reinterpret_cast<TFloatComplex*>(matrixData)[index];
    };

    /**
     * @brief   Operator [], constant version.
     * @details Operator [], constant version.
     * @param [in] index - 1D index into the array
     * @return element of the matrix
     */
    inline const TFloatComplex& operator[](const size_t& index) const
    {
      return reinterpret_cast<TFloatComplex*> (matrixData)[index];
    };

    /// Load data from the HDF5_File.
    virtual void ReadDataFromHDF5File(THDF5_File& file,
                                      TMatrixName& matrixName);

    /// Write data into the HDF5_File
    virtual void WriteDataToHDF5File(THDF5_File&  file,
                                     TMatrixName& matrixName,
                                     const size_t compressionLevel);


protected:
    /// Default constructor not allowed for public.
    TComplexMatrix() : TBaseFloatMatrix() {};
    /// Copy constructor not allowed for public.
    TComplexMatrix(const TComplexMatrix& src);
    /// Operator not allowed for public.
    TComplexMatrix& operator = (const TComplexMatrix& src);

    /// Initialize dimension sizes and related structures.
    virtual void InitDimensions(const TDimensionSizes& dimensionSizes);

private:

};// end of TComplexMatrix
//--------------------------------------------------------------------------------------------------

#endif	/* COMPLEX_MATRIX_H */

