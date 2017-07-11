/**
 * @file        BaseFloatMatrix.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the base class for single precisions floating point
 *              numbers (floats).
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        11 July      2011, 12:13 (created) \n
 *              07 July      2017, 18:26 (revised)
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

#ifndef BASE_FLOAT_MATRIX_H
#define	BASE_FLOAT_MATRIX_H

#include <MatrixClasses/BaseMatrix.h>
#include <Utils/DimensionSizes.h>


/**
 * @class TBaseFloatMatrix
 * @brief Abstract base class for float based matrices defining basic interface. Higher dimensional
 *        matrices stored as 1D arrays, row-major order.
 *
 * @details Abstract base class for float based matrices defining basic interface. Higher
 *          dimensional matrices stored as 1D arrays, row-major order.Implemented both on
 *          CPU and GPU side.
 */
class TBaseFloatMatrix : public TBaseMatrix
{
  public:
    /// Default constructor.
    TBaseFloatMatrix();
    /// Copy constructor is not allowed.
    TBaseFloatMatrix(const TBaseFloatMatrix&) = delete;
    //Destructor.
    virtual ~TBaseFloatMatrix() {};

    /// Operator= is not allowed.
    TBaseFloatMatrix& operator=(const TBaseFloatMatrix&);

    /// Get dimension sizes of the matrix.
    virtual TDimensionSizes GetDimensionSizes() const
    {
      return dimensionSizes;
    }

    /// Get element count of the matrix.
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

    /// Divide scalar/ matrix_element[i].
    virtual void ScalarDividedBy(const float scalar);


    /// Get raw CPU data out of the class (for direct CPU kernel access).
    virtual float* GetHostData()
    {
      return hostData;
    }

    /// Get raw CPU data out of the class (for direct CPU kernel access).
    virtual const float* GetHostData() const
    {
      return hostData;
    }

    /// Get raw GPU data out of the class (for direct GPU kernel access).
    virtual float* GetDeviceData()
    {
      return deviceData;
    }

    /// Get raw GPU data out of the class (for direct GPU kernel access).
    virtual const float* GetDeviceData() const
    {
      return deviceData;
    }

    /// Copy data from CPU -> GPU (Host -> Device).
    virtual void CopyToDevice();
    /// Copy data from GPU -> CPU (Device -> Host).
    virtual void CopyFromDevice();

  protected:

    /// Memory allocation (both on CPU and GPU).
    virtual void AllocateMemory();
    /// Memory allocation (both on CPU and GPU).
    virtual void FreeMemory();

    /// Total number of elements.
    size_t nElements;
    /// Total number of allocated elements (in terms of floats).
    size_t nAllocatedElements;

    /// Dimension sizes.
    struct TDimensionSizes dimensionSizes;

    /// Size of a 1D row in X dimension.
    size_t dataRowSize;
    /// Size of a 2D slab.
    size_t dataSlabSize;

    /// Raw CPU matrix data.
    float* hostData;
    /// Raw GPU matrix data.
    float* deviceData;

  private:

};//end of class TBaseFloatMatrix
//--------------------------------------------------------------------------------------------------

#endif /* BASE_FLOAT_MATRIX_H */
