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
 * @date        11 July     2011, 14:02 (created) \n
 *              13 November 2014, 15:47 (revised)
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

#ifndef COMPLEXMATRIXDATA_H
#define	COMPLEXMATRIXDATA_H

#include <MatrixClasses/BaseFloatMatrix.h>
#include <MatrixClasses/RealMatrix.h>

#include <Utils/DimensionSizes.h>


using namespace std;


/**
 * @struct TFloatComplex
 * @brief  Structure for complex values
 * @todo:  Change to classic C++ complex (better support of vectorisation)
 */
struct TFloatComplex
{
  /// real part
  float real;
  /// imaginary part
  float imag;
};//TFloatComplex
//------------------------------------------------------------------------------


/**
 * @class   TComplexMatrix
 * @brief   The class for complex matrices.
 * @details The class for complex matrices.
 */
class TComplexMatrix : public TBaseFloatMatrix
{
  public:

    /// Constructor.
    TComplexMatrix(const TDimensionSizes & DimensionSizes);

    /// Destructor.
    virtual ~TComplexMatrix()
    {
      FreeMemory();
    };


    /* @brief   operator [].
     * @details operator [].
     * @param [in] index - 1D index into the array
     * @return           - element of the matrix
     */
    inline TFloatComplex& operator [](const size_t& index)
    {
      return ((TFloatComplex *) pMatrixData)[index];
    };

    /**
     * @brief   operator [], constant version.
     * @details operator [], constant version.
     * @param [in] index - 1D index into the array
     * @return element of the matrix
     */
    inline const TFloatComplex& operator [](const size_t& index) const
    {
      return ((TFloatComplex *) pMatrixData)[index];
    };


    /**
     * @brief   Get element from 3D matrix.
     * @details Get element from 3D matrix.
     * @param [in] X - X dimension
     * @param [in] Y - Y dimension
     * @param [in] Z - Z dimension
     * @return a complex element of the class
     */
    inline  TFloatComplex& GetElementFrom3D(const size_t X,
                                            const size_t Y,
                                            const size_t Z)
    {
      return ((TFloatComplex *) pMatrixData)[Z * (p2DDataSliceSize>>1) + Y * (pDataRowSize>>1) + X];
    };


    /// Load data from the HDF5_File.
    virtual void ReadDataFromHDF5File(THDF5_File & HDF5_File,
                                      const char * MatrixName);

    /// Write data into the HDF5_File
    virtual void WriteDataToHDF5File(THDF5_File & HDF5_File,
                                     const char * MatrixName,
                                     const size_t CompressionLevel);


protected:
    /// Default constructor not allowed for public.
    TComplexMatrix() : TBaseFloatMatrix() {};

    /// Copy constructor not allowed for public.
    TComplexMatrix(const TComplexMatrix& src);

    /// Operator not allowed for public.
    TComplexMatrix& operator = (const TComplexMatrix& src);

    /// Initialize dimension sizes and related structures.
    virtual void InitDimensions(const TDimensionSizes& DimensionSizes);

private:


};// end of TComplexMatrix
//------------------------------------------------------------------------------

#endif	/* COMPLEXMATRIXDATA_H */

