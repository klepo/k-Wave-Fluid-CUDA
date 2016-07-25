/**
 * @file        BaseMatrix.h
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
 *              12 November 2014, 13:34 (revised)
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

#ifndef BASE_MATRIX_H
#define BASE_MATRIX_H

#include <Utils/DimensionSizes.h>
#include <HDF5/HDF5_File.h>

/**
 * @class TBaseMatrix
 * @brief Abstract base class. The common ancestor defining the common interface and allowing
 *        derived classes to be allocated, freed and loaded from the file using the Matrix container.
 *
 * @details Abstract base class. The common ancestor defining the common interface and allowing
 *          derived classes to be allocated, freed and loaded from the file using the Matrix
 *          container. In this version of the code, It allocates memory both on the CPU and GPU side.
 */
class TBaseMatrix
{
  public:
    /// Default constructor.
    TBaseMatrix() {};

    /// Destructor
    virtual ~TBaseMatrix() {};

    /// Get dimension sizes of the matrix.
    virtual struct TDimensionSizes GetDimensionSizes() const  = 0;

    /// Get total element count of the matrix.
    virtual size_t GetElementCount()              const = 0;
    /// Get total allocated element count (might differ from the total element count used for the simulation because of e.g. padding).
    virtual size_t GetAllocatedElementCount()     const  = 0;

    /**
     * @brief   Read matrix from the HDF5 file.
     * @details Read matrix from the HDF5 file.
     * @param [in] file       - Handle to the HDF5 file
     * @param [in] matrixName - HDF5 dataset name to read from
     */
    virtual void ReadDataFromHDF5File(THDF5_File& file,
                                      const char* matrixName) = 0;

    /**
     * @brief   Write data into the HDF5 file.
     * @details Write data into the HDF5 file.
     * @param [in] file             - Handle to the HDF5 file
     * @param [in] matrixName       - HDF5 dataset name to write to
     * @param [in] compressionLevel - Compression level for the HDF5 dataset
     */
    virtual void WriteDataToHDF5File(THDF5_File&  file,
                                     const char*  matrixName,
                                     const size_t compressionLevel) = 0;

    /// Copy data from CPU -> GPU (Host -> Device).
    virtual void CopyToDevice()   = 0;

    /// Copy data from GPU -> CPU (Device -> Host).
    virtual void CopyFromDevice() = 0;

  protected:

};// end of TBaseMatrix
//--------------------------------------------------------------------------------------------------

#endif /* BASE_MATRIX_H */

