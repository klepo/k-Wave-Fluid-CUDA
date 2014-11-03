/**
 * @file        MatrixRecord.h
 * @author      Jiri Jaros & Beau Johnston \n
 *              CECS, ANU, Australia       \n
 *              jiri.jaros@anu.edu.au      \n
 *
 * @brief       The header file of the class for storing Matrix structural details.
 * 
 * @version     kspaceFirstOrder3D 2.13
 * @date        1 September 2014, 14:30 (created)
 *
 * @section License
 * This file is part of the C++ extension of the k-Wave Toolbox
 * (http://www.k-wave.org).\n Copyright (C) 2012 Jiri Jaros and Bradley Treeby
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
 * along with k-Wave. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef MATRIXRECORD_H
#define MATRIXRECORD_H

#include "BaseMatrix.h"

/**
 * @struct TMatrixRecord
 * @brief  A structure storing details about the matrix. The matrix container
 * stores this structures.
 */
struct TMatrixRecord
{
    /**
     * @enum TMatrixDataType
     * @brief All possible types of the matrix
     */
    enum TMatrixDataType {mdtReal,
                          mdtComplex,
                          mdtIndex,
#if VANILLA_CPP_VERSION & CUDA_VERSION
                          mdtFFTW,
                          mdtCUFFT
#elif VANILLA_CPP_VERSION
                          mdtFFTW
#elif CUDA_VERSION
                          mdtCUFFT
#elif OPENCL_VERSION
#endif
                          };

    /// Pointer to the matrix object
    TBaseMatrix   * MatrixPtr;
    /// Matrix data type
    TMatrixDataType MatrixDataType;
    /// Matrix dimension sizes
    TDimensionSizes DimensionSizes;
    /// Is the matrix content loaded from the HDF5 file
    bool            LoadData;
    /// Is the matrix necessary to be preserver when checkpoint is enabled
    bool            Checkpoint;
    /// HDF5 matrix name
    string          HDF5MatrixName;

    /// Default constructor
    TMatrixRecord() : MatrixPtr(NULL), MatrixDataType(mdtReal),
                      DimensionSizes(), LoadData(false), Checkpoint(false),
                      HDF5MatrixName("")
                      {};

    /// Copy constructor
    TMatrixRecord(const TMatrixRecord& src);

    /// operator =
    TMatrixRecord& operator = (const TMatrixRecord& src);

    /// Set all values of the record
    void SetAllValues(TBaseMatrix *          MatrixPtr,
                      const TMatrixDataType  MatrixDataType,
                      const TDimensionSizes  DimensionSizes,
                      const bool             LoadData,
                      const bool             Checkpoint,
                      const string           HDF5MatrixName);

    virtual ~TMatrixRecord() {};


};// end of TMatrixRecord
//------------------------------------------------------------------------------

#endif /* MATRIXRECORD_H */


