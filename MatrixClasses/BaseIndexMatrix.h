/**
 * @file        BaseIndexMatrix.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the base class for 64b-wide integers implemented as
 *              size_t datatype.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        26 July     2011, 14:17 (created) \n
 *              11 July     2017, 16:44 (revised)
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

#ifndef BASE_INDEX_MATRIX_H
#define BASE_INDEX_MATRIX_H

#include <MatrixClasses/BaseMatrix.h>
#include <Utils/DimensionSizes.h>


/**
 * @class TBaseIndexMatrix
 * @brief Abstract base class for index based matrices defining basic interface.
 *        Higher dimensional matrices stored as 1D arrays, row-major order.

 * @details Abstract base class for index based matrices defining basic interface. Higher
 *          dimensional matrices stored as 1D arrays, row-major order. This matrix stores the data
 *          on both the CPU and GPU side.
 */
class TBaseIndexMatrix : public TBaseMatrix
{
  public:
    /// Default constructor.
    TBaseIndexMatrix();
    /// Copy constructor is not allowed.
    TBaseIndexMatrix(const TBaseIndexMatrix&) = delete;
    /// Destructor.
    virtual ~TBaseIndexMatrix(){};

    /// operator= is not allowed.
    TBaseIndexMatrix& operator=(const TBaseIndexMatrix&) = delete;

    /// Get dimension sizes of the matrix.
    virtual struct DimensionSizes GetDimensionSizes() const
    {
      return dimensionSizes;
    }

    /// Get total element count of the matrix.
    virtual size_t GetElementCount() const
    {
      return nElements;
    };

    /// Get total allocated element count (might differ from total element count used for the simulation because of padding).
    virtual size_t GetAllocatedElementCount() const
    {
      return nAllocatedElements;
    };

    /// Zero all elements of the matrix (NUMA first touch).
    virtual void ZeroMatrix();

    /// Get raw data out of the class (for direct CPU kernel access).
    virtual size_t* GetHostData()
    {
      return hostData;
    }

    /// Get raw data out of the class (for direct CPU kernel access).
    virtual const size_t* GetHostData() const
    {
      return hostData;
    }

    /// Get raw GPU data out of the class (for direct GPU kernel access).
    virtual size_t* GetDeviceData()
    {
      return deviceData;
    }

    /// Get raw GPU data out of the class (for direct GPU kernel access).
    virtual const size_t* GetDeviceData() const
    {
      return deviceData;
    }

    /// Copy data from CPU -> GPU (Host -> Device).
    virtual void CopyToDevice();

    /// Copy data from GPU -> CPU (Device -> Host).
    virtual void CopyFromDevice();

  protected:

    /// Memory allocation (both on CPU and GPU)
    virtual void AllocateMemory();
    /// Memory deallocation (both on CPU and GPU)
    virtual void FreeMemory();

    /// Total number of elements.
    size_t nElements;
    /// Total number of allocated elements (the array size).
    size_t nAllocatedElements;

    /// Dimension sizes.
    struct DimensionSizes dimensionSizes;

    /// Size of 1D row in X dimension.
    size_t rowSize;
    /// Size of 2D slab (X,Y).
    size_t slabSize;

    /// Raw CPU matrix data.
    size_t* hostData;
    /// Raw GPU matrix data.
    size_t* deviceData;

  private:

};// end of TBaseLongMatrix
//--------------------------------------------------------------------------------------------------

#endif /* BASE_INDEX_MATRIX_H */
