/**
 * @file      MatrixRecord.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file containing metadata about matrices stored in the matrix container.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      02 December  2014, 15:44 (created) \n
 *            06 March     2019, 12:57 (revised)
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

#ifndef MATRIX_RECORD_H
#define MATRIX_RECORD_H

#include <MatrixClasses/BaseMatrix.h>
#include <MatrixClasses/BaseFloatMatrix.h>
#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>
#include <MatrixClasses/IndexMatrix.h>
#include <MatrixClasses/CufftComplexMatrix.h>
#include <Utils/MatrixNames.h>

/**
 * @struct  MatrixRecord
 * @brief   A structure storing details about the matrix.
 * @details A structure storing details about the matrix. The matrix container stores the list of
 *          these records - metadata and pointer to the matrix.
 */
struct MatrixRecord
{
  /**
   * @enum  MatrixType
   * @brief All possible types of the matrix.
   */
  enum class MatrixType
  {
    /// Matrix for real values.
    kReal,
    /// Matrix for complex values.
    kComplex,
    /// Matrix for index values.
    kIndex,
    /// Matrix for cuda fft.
    kCufft
  };

  /// Default constructor.
  MatrixRecord();
  /// Destructor.
  ~MatrixRecord() = default;

  /**
   * @brief Copy constructor of MatrixRecord.
   * @param [in] src - Source.
   */
  MatrixRecord(const MatrixRecord& src);
  /**
   * operator=
   * @param [in] src - Source object/
   * @return A new object.
   */
  MatrixRecord& operator=(const MatrixRecord& src);

  /**
   * @brief Set all values for the record.
   * @param [in] matrixType     - Matrix data type.
   * @param [in] dimensionSizes - Dimension sizes.
   * @param [in] loadData       - Load data from file?
   * @param [in] checkpoint     - Checkpoint this matrix?
   * @param [in] matrixName     - HDF5 matrix name.
   */
  void set(const MatrixType      matrixType,
           const DimensionSizes  dimensionSizes,
           const bool            loadData,
           const bool            checkpoint,
           MatrixName&           matrixName);

  /// Pointer to the matrix object.
  BaseMatrix*    matrixPtr;
  /// Matrix data type.
  MatrixType     matrixType;
  /// Matrix dimension sizes.
  DimensionSizes dimensionSizes;
  /// Is the matrix content loaded from the HDF5 file?
  bool           loadData;
  /// Is the matrix necessary to be preserver when checkpoint is enabled?
  bool           checkpoint;
  /// Matrix name in the HDF5 file.
  std::string    matrixName;
};// end of MatrixRecord
//----------------------------------------------------------------------------------------------------------------------

#endif	/* MATRIX_RECORD_H */

