/**
 * @file        MatrixRecord.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing metadata about matrices stored in the matrix container.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        02 December 2014, 15:44 (created) \n
 *              19 July     2016, 16:43 (revised)
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

#ifndef MATRIX_RECORD_H
#define	MATRIX_RECORD_H

#include <MatrixClasses/BaseMatrix.h>
#include <MatrixClasses/BaseFloatMatrix.h>
#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>
#include <MatrixClasses/IndexMatrix.h>
#include <MatrixClasses/CUFFTComplexMatrix.h>


/**
 * @struct TMatrixRecord
 * @brief   A structure storing details about the matrix.
 * @details A structure storing details about the matrix. The matrix container stores the list of
 *          these records - metadata and pointer to the matrix.
 */
struct TMatrixRecord
{
  /**
   * @enum TMatrixType
   * @brief All possible types of the matrix.
   */
  enum TMatrixType
  {
    REAL, COMPLEX, INDEX, CUFFT
  };

  /// Default constructor.
  TMatrixRecord();
  /// Destructor.
  ~TMatrixRecord() {};

  /// Copy constructor.
  TMatrixRecord(const TMatrixRecord& src);
  /// operator =.
  TMatrixRecord& operator= (const TMatrixRecord& src);

  /// Set all values of the record.
  void Set(const TMatrixType      matrixType,
           const TDimensionSizes  dimensionSizes,
           const bool             loadData,
           const bool             checkpoint,
           const std::string&     matrixName);

  /// Pointer to the matrix object.
  TBaseMatrix*    matrixPtr;
  /// Matrix data type.
  TMatrixType     matrixType;
  /// Matrix dimension sizes.
  TDimensionSizes dimensionSizes;
  /// Is the matrix content loaded from the HDF5 file?
  bool            loadData;
  /// Is the matrix necessary to be preserver when checkpoint is enabled?
  bool            checkpoint;
  /// Matrix name in the HDF5 file.
  std::string     matrixName;
};// end of TMatrixRecord
//------------------------------------------------------------------------------

#endif	/* MATRIX_RECORD_H */

