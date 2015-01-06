/**
 * @file        BaseIndexMatrix.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the base class for
 *              64b-wide integers (long for Linux/ size_t for Windows).
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        26 July     2011, 14:17 (created) \n
 *              12 November 2014, 15:55 (revised)
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

#ifndef BASE_INDEX_MATRIX_H
#define BASE_INDEX_MATRIX_H

#include <MatrixClasses/BaseMatrix.h>
#include <Utils/DimensionSizes.h>

using namespace std;

/**
 * @class TBaseIndexMatrix
 * @brief Abstract base class for index based matrices defining basic interface.
 *        Higher dimensional matrices stored as 1D arrays, row-major order.

 * @details Abstract base class for index based matrices defining basic interface.
 *          Higher dimensional matrices stored as 1D arrays, row-major order.
 *          This matrix stores the data on both the CPU and GPU side.
 */
class TBaseIndexMatrix : public TBaseMatrix
{
  public:

    TBaseIndexMatrix() : TBaseMatrix(),
            pTotalElementCount(0), pTotalAllocatedElementCount(0),
            pDimensionSizes(), pDataRowSize(0), p2DDataSliceSize(0),
            pMatrixData(NULL), pdMatrixData(NULL)
    {}

    /// Get dimension sizes of the matrix.
    inline struct TDimensionSizes GetDimensionSizes() const
    {
      return pDimensionSizes;
    }

    /// Get total element count of the matrix.
    virtual size_t GetTotalElementCount() const
    {
      return pTotalElementCount;
    };

    /// Get total allocated element count (might differ from total element count used for the simulation because of padding).
    virtual size_t GetTotalAllocatedElementCount() const
    {
      return pTotalAllocatedElementCount;
    };

    /// Destructor.
    virtual ~TBaseIndexMatrix(){};

    /// Zero all elements of the matrix (NUMA first touch).
    virtual void ZeroMatrix();

    /// Get raw data out of the class (for direct CPU kernel access).
    virtual size_t * GetRawData()
    {
      return pMatrixData;
    }

    /// Get raw data out of the class (for direct CPU kernel access).
    virtual const size_t * GetRawData() const
    {
      return pMatrixData;
    }

    /// Get raw GPU data out of the class (for direct GPU kernel access).
    virtual size_t * GetRawDeviceData()
    {
      return pdMatrixData;
    }

    /// Get raw GPU data out of the class (for direct GPU kernel access).
    virtual const size_t* GetRawDeviceData() const
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
    /// Total number of allocated elements (the array size).
    size_t pTotalAllocatedElementCount;

    /// Dimension sizes.
    struct TDimensionSizes pDimensionSizes;

    /// Size of 1D row in X dimension.
    size_t pDataRowSize;
    /// Size of 2D slab (X,Y).
    size_t p2DDataSliceSize;

    /// Raw CPU matrix data.
    size_t* pMatrixData;

    /// Raw GPU matrix data.
    size_t* pdMatrixData;

        ///Memory allocation (both on CPU and GPU)
    virtual void AllocateMemory();

    ///Memory allocation (both on CPU and GPU)
    virtual void FreeMemory();

    /// Copy constructor is not directly allowed
    TBaseIndexMatrix(const TBaseIndexMatrix& src);
    /// operator =  is not directly allowed
    TBaseIndexMatrix & operator =(const TBaseIndexMatrix& src);

  private:

};// end of TBaseLongMatrix
//-----------------------------------------------------------------------------

#endif /* BASE_INDEX_MATRIX_H */

