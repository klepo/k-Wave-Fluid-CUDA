/**
 * @file        BaseMatrix.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file of the common ancestor of all matrix classes.
 *              A pure abstract class.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        11 July     2012, 11:34 (created) \n
 *              20 July     2017, 17:01 (revised)
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

#ifndef BaseMatrixH
#define BaseMatrixH

#include <Utils/DimensionSizes.h>
#include <Hdf5/Hdf5File.h>

/**
 * @class BaseMatrix
 * @brief Abstract base class. The common ancestor defining the common interface and allowing derived classes to be
 *        allocated, freed and loaded from the file using the Matrix container.
 *
 * @details Abstract base class. The common ancestor defining the common interface and allowing derived classes to be
 *          allocated, freed and loaded from the file using the Matrix container. In this version of the code, It
 *          allocates memory both on the CPU and GPU side. The I/O is done via HDF5 files.
 */
class BaseMatrix
{
  public:
    /// Default constructor.
    BaseMatrix() {};
    /// Copy constructor is not allowed.
    BaseMatrix(const BaseMatrix&) = delete;
    /// Destructor
    virtual ~BaseMatrix() {};

    /// Operator= is not allowed.
    BaseMatrix& operator=(const BaseMatrix&) = delete;

    /**
     * @brief  Get dimension sizes of the matrix.
     * @return Dimension sizes of the matrix.
     */
    virtual const DimensionSizes& getDimensionSizes() const  = 0;
    /**
     * @brief Size of the matrix.
     * @return Number of elements.
     */
    virtual size_t size()     const = 0;
    /**
     * @brief  The capacity of the matrix (this may differ from size due to padding, etc.).
     * @return Capacity of the currently allocated storage.
     */
    virtual size_t capacity() const  = 0;

    /**
     * @brief   Read matrix from HDF5 file.
     * @details Read matrix from HDF5 file.
     * @param [in] file       - Handle to the HDF5 file.
     * @param [in] matrixName - HDF5 dataset name to read from.
     */
    virtual void readData(Hdf5File&   file,
                          MatrixName& matrixName) = 0;
    /**
     * @brief   Write data into HDF5 file.
     * @details Write data into HDF5 file.
     * @param [in] file             - Handle to the HDF5 file.
     * @param [in] matrixName       - HDF5 dataset name to write to.
     * @param [in] compressionLevel - Compression level for the HDF5 dataset.
     */
    virtual void writeData(Hdf5File&    file,
                           MatrixName&  matrixName,
                           const size_t compressionLevel) = 0;

    /// Copy data from CPU -> GPU (Host -> Device).
    virtual void copyToDevice()   = 0;

    /// Copy data from GPU -> CPU (Device -> Host).
    virtual void copyFromDevice() = 0;

  protected:

};// end of BaseMatrix
//----------------------------------------------------------------------------------------------------------------------

#endif /* BaseMatrixH */

