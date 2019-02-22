/**
 * @file      BaseIndexMatrix.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file containing the base class for 64b-wide integers implemented as size_t datatype.
 *
 * @version   kspaceFirstOrder3D 3.6
 *
 * @date      26 July      2011, 14:17 (created) \n
 *            22 February  2019, 12:36 (revised)
 *
 * @copyright Copyright (C) 2019 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the C++ extension of the [k-Wave Toolbox](http://www.k-wave.org).
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify it under the terms
 * of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 */

#ifndef BASE_INDEX_MATRIX_H
#define BASE_INDEX_MATRIX_H

#include <MatrixClasses/BaseMatrix.h>
#include <Utils/DimensionSizes.h>


/**
 * @class   BaseIndexMatrix
 * @brief   Abstract base class for index based matrices defining basic interface.
 *          Higher dimensional matrices stored as 1D arrays, row-major order.

 * @details Abstract base class for index based matrices defining basic interface. Higher
 *          dimensional matrices stored as 1D arrays, row-major order. This matrix stores the data
 *          on both the CPU and GPU side. The I/O is done via HDF5 files.
 */
class BaseIndexMatrix : public BaseMatrix
{
  public:
    /// Default constructor.
    BaseIndexMatrix();
    /// Copy constructor is not allowed.
    BaseIndexMatrix(const BaseIndexMatrix&) = delete;
    /// Destructor.
    virtual ~BaseIndexMatrix(){};

    /// operator= is not allowed.
    BaseIndexMatrix& operator=(const BaseIndexMatrix&) = delete;

    /**
     * @brief  Get dimension sizes of the matrix.
     * @return Dimension sizes of the matrix.
     */
    virtual const DimensionSizes& getDimensionSizes() const { return mDimensionSizes; }

    /**
     * @brief Size of the matrix.
     * @return Number of elements.
     */
    virtual size_t size()                             const { return mSize; };
    /**
     * @brief  The capacity of the matrix (this may differ from size due to padding, etc.).
     * @return Capacity of the currently allocated storage.
     */
    virtual size_t capacity()                         const { return mCapacity; };

    /// Zero all elements of the matrix (NUMA first touch).
    virtual void   zeroMatrix();

    /**
     * @brief  Get matrix data stored stored on the host side (for direct host kernels).
     * @return Pointer to matrix data stored on the host side.
     */
    virtual size_t*       getHostData()                     { return mHostData; }
    /**
     * @brief  Get matrix data stored stored on the host side (for direct host kernels).
     * @return Pointer to matrix data stored on the host side.
     */
    virtual const size_t* getHostData()               const { return mHostData; }
    /**
     * @brief  Get matrix data stored stored on the device side (for direct device kernels).
     * @return Pointer to matrix data on the device side.
     */
    virtual size_t*       getDeviceData()                   { return mDeviceData; }
    /**
     * @brief  Get matrix data stored stored on the device side (for direct device kernels).
     * @return Pointer to matrix data on the device side.
     */
    virtual const size_t* getDeviceData()             const { return mDeviceData; }

    /// Copy data from CPU -> GPU (Host -> Device).
    virtual void copyToDevice();

    /// Copy data from GPU -> CPU (Device -> Host).
    virtual void copyFromDevice();

  protected:

   /**
    * @brief Aligned memory allocation (both on CPU and GPU).
    * @throw std::bad_alloc - If there's not enough memory.
    */
    virtual void allocateMemory();
    /// Memory deallocation (both on CPU and GPU)
    virtual void freeMemory();

    /// Total number of elements.
    size_t mSize;
    /// Total number of allocated elements (in terms of size_t).
    size_t mCapacity;

    /// Dimension sizes.
    DimensionSizes mDimensionSizes;

    /// Size of 1D row in X dimension.
    size_t mRowSize;
    /// Size of a XY slab.
    size_t mSlabSize;

    /// Raw CPU matrix data.
    size_t* mHostData;
    /// Raw GPU matrix data.
    size_t* mDeviceData;

  private:

};// end of BaseIndexMatrix
//----------------------------------------------------------------------------------------------------------------------

#endif /* BASE_INDEX_MATRIX_H */
