/**
 * @file      BaseFloatMatrix.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file containing the base class for single precisions floating point numbers (floats).
 *
 * @version   kspaceFirstOrder3D 3.5
 *
 * @date      11 July      2011, 12:13 (created) \n
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

#ifndef BASE_FLOAT_MATRIX_H
#define BASE_FLOAT_MATRIX_H

#include <MatrixClasses/BaseMatrix.h>
#include <Utils/DimensionSizes.h>


/**
 * @class   BaseFloatMatrix
 * @brief   Abstract base class for float based matrices defining basic interface. Higher dimensional
 *          matrices stored as 1D arrays, row-major order.
 *
 * @details Abstract base class for float based matrices defining basic interface. Higher dimensional matrices stored
 *          as 1D arrays, row-major order.Implemented both on CPU and GPU side. The I/O is done via HDF5 files.
 */
class BaseFloatMatrix : public BaseMatrix
{
  public:
    /// Default constructor.
    BaseFloatMatrix();
    /// Copy constructor is not allowed.
    BaseFloatMatrix(const BaseFloatMatrix&) = delete;
    //Destructor.
    virtual ~BaseFloatMatrix() {};

    /// Operator= is not allowed.
    BaseFloatMatrix& operator=(const BaseFloatMatrix&);

    /**
     * @brief  Get dimension sizes of the matrix.
     * @return Dimension sizes of the matrix.
     */
    virtual const DimensionSizes& getDimensionSizes() const { return mDimensionSizes; }
    /**
     * @brief  Size of the matrix.
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
     * @brief Calculate matrix = scalar / matrix.
     * @param [in] scalar - Scalar constant
     */
    virtual void   scalarDividedBy(const float scalar);


    /**
     * @brief  Get matrix data stored stored on the host side (for direct host kernels).
     * @return Pointer to matrix data stored on the host side.
     */
    virtual float*       getHostData()                      { return mHostData; }
    /**
     * @brief  Get matrix data stored stored on the host side (for direct host kernels).
     * @return Pointer to matrix data stored on the host side.
     */
    virtual const float* getHostData()                const { return mHostData; }
    /**
     * @brief  Get matrix data stored stored on the device side (for direct device kernels).
     * @return Pointer to matrix data on the device side.
     */
    virtual float*       getDeviceData()                    { return mDeviceData; }
    /**
     * @brief  Get matrix data stored stored on the device side (for direct device kernels).
     * @return Pointer to matrix data on the device side.
     */
    virtual const float* getDeviceData()              const { return mDeviceData; }

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
    /// Memory allocation (both on CPU and GPU).
    virtual void freeMemory();

    /// Total number of used elements.
    size_t mSize;
    /// Total number of allocated elements (in terms of floats).
    size_t mCapacity;

    /// Dimension sizes.
    DimensionSizes mDimensionSizes;

    /// Size of a 1D row in X dimension.
    size_t mRowSize;
    /// Size of a XY slab.
    size_t mSlabSize;

    /// Raw CPU matrix data.
    float* mHostData;
    /// Raw GPU matrix data.
    float* mDeviceData;

  private:

};//end of BaseFloatMatrix
//----------------------------------------------------------------------------------------------------------------------

#endif /* BASE_INDEX_MATRIX_H */
