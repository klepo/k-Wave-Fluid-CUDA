/**
 * @file        BaseFloatMatrix.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the base class for
 *              single precisions floating point numbers (floats).
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        11 July      2011, 12:13 (created) \n
 *              12 November  2014, 13:58 (revised)
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

#ifndef BASEFLOATMATRIXDATA_H
#define	BASEFLOATMATRIXDATA_H

#include <MatrixClasses/BaseMatrix.h>
#include <Utils/DimensionSizes.h>

using namespace std;

/**
 * @class TBaseFloatMatrix
 * @brief Abstract base class for float based matrices defining basic interface.
 *        Higher dimensional matrices stored as 1D arrays, row-major order.
 *
 * @details Abstract base class for float based matrices defining basic interface.
 *          Higher dimensional matrices stored as 1D arrays, row-major order.
 *          Implemented both on CPU and GPU side.
 *
 */
class TBaseFloatMatrix : public TBaseMatrix
{
  public:
    /// Default constructor.
    TBaseFloatMatrix() : TBaseMatrix(),
            pTotalElementCount(0), pTotalAllocatedElementCount(0), pDimensionSizes(),
            pDataRowSize(0), p2DDataSliceSize(0),
            pMatrixData(NULL), pdMatrixData(NULL)
    {
    };

    //Destructor.
    virtual ~TBaseFloatMatrix() {};

    /// Get dimension sizes of the matrix.
    virtual TDimensionSizes GetDimensionSizes() const
    {
      return pDimensionSizes;
    }

    /// Get element count of the matrix.
    virtual size_t GetTotalElementCount() const
    {
      return pTotalElementCount;
    };

    /// Get total allocated element count (might differ from total element count used for the simulation because of padding).
    virtual size_t GetTotalAllocatedElementCount() const
    {
      return pTotalAllocatedElementCount;
    };

    /// Copy data from other matrix with the same size.
    virtual void CopyData(const TBaseFloatMatrix & src);

    /// Zero all elements of the matrix (NUMA first touch).
    virtual void ZeroMatrix();

    /// Divide scalar/ matrix_element[i].
    virtual void ScalarDividedBy(const float scalar);


    /// Get raw CPU data out of the class (for direct CPU kernel access).
    virtual float * GetRawData()
    {
      return pMatrixData;
    }

    /// Get raw CPU data out of the class (for direct CPU kernel access).
    virtual const float * GetRawData() const
    {
      return pMatrixData;
    }

    /// Get raw GPU data out of the class (for direct GPU kernel access).
    virtual float * GetRawDeviceData()
    {
      return pdMatrixData;
    }

    /// Get raw GPU data out of the class (for direct GPU kernel access).
    virtual const float * GetRawDeviceData() const
    {
      return pdMatrixData;
    }

    /// Copy data from CPU (Host) to GPU (Device).
    virtual void CopyToDevice();

    /// Copy data from GPU (Device) to CPU (Host).
    virtual void CopyFromDevice();


  protected:

    /// Total number of elements.
    size_t pTotalElementCount;
    /// Total number of allocated elements (in terms of floats).
    size_t pTotalAllocatedElementCount;

    /// Dimension sizes.
    struct TDimensionSizes pDimensionSizes;

    /// Size of a 1D row in X dimension.
    size_t pDataRowSize;
    /// Size of a 2D slab (X,Y).
    size_t p2DDataSliceSize;

    /// Raw CPU matrix data.
    float * pMatrixData;

    /// Raw GPU matrix data.
    float * pdMatrixData;


    ///Memory allocation (both on CPU and GPU).
    virtual void AllocateMemory();

    ///Memory allocation (both on CPU and GPU).
    virtual void FreeMemory();

    /// Copy constructor is not directly allowed.
    TBaseFloatMatrix(const TBaseFloatMatrix& src);
    /// operator =  is not directly allowed.
    TBaseFloatMatrix & operator =(const TBaseFloatMatrix& src);

  private:

};//end of class TBaseFloatMatrix
//----------------------------------------------------------------------------

#endif /* TBASEMATRIXDATA_H */

