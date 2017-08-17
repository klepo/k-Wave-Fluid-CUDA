/**
 * @file      MatrixRecord.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file containing metadata about matrices stored in the matrix container.
 *
 * @version   kspaceFirstOrder3D 3.5
 *
 * @date      02 December  2014, 15:44 (created) \n
 *            16 August    2017, 14:58 (revised)
 *
 * @copyright Copyright (C) 2017 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the C++ extension of the k-Wave Toolbox [k-Wave Toolbox](http://www.k-wave.org).
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify it under the terms
 * of the GNU Lesser General Public License as published by the Free Software  *Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 */

//----------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------- Constants -----------------------------------------------------//
//----------------------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------------------------
//------------------------------------------------- Public methods ---------------------------------------------------//
//----------------------------------------------------------------------------------------------------------------------

#include <Containers/MatrixRecord.h>

/**
 * Default constructor.
 */
MatrixRecord::MatrixRecord()
  : matrixPtr(nullptr),
    matrixType(MatrixType::kReal),
    dimensionSizes(),
    loadData(false),
    checkpoint(false),
    matrixName()
{

}// end of constructor
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy constructor of TMatrixRecord.
 */
MatrixRecord::MatrixRecord(const MatrixRecord& src)
  : matrixPtr(src.matrixPtr),
    matrixType(src.matrixType),
    dimensionSizes(src.dimensionSizes),
    loadData(src.loadData),
    checkpoint(src.checkpoint),
    matrixName(src.matrixName)
{

}// end of copy constructor
//----------------------------------------------------------------------------------------------------------------------

/**
 * operator =
 */
MatrixRecord& MatrixRecord::operator=(const MatrixRecord& src)
{
  if (this != &src)
  {
    matrixPtr       = src.matrixPtr;
    matrixType      = src.matrixType;
    dimensionSizes  = src.dimensionSizes;
    loadData        = src.loadData;
    checkpoint      = src.checkpoint;
    matrixName      = src.matrixName;
  }

  return *this;
}// end of operator=
//----------------------------------------------------------------------------------------------------------------------

/**
 * Set all values for the record.
 */
void MatrixRecord::set(const MatrixType     matrixType,
                       const DimensionSizes dimensionSizes,
                       const bool           loadData,
                       const bool           checkpoint,
                       MatrixName&          matrixName)
{
  this->matrixPtr        = nullptr;
  this->matrixType       = matrixType;
  this->dimensionSizes   = dimensionSizes;
  this->loadData         = loadData;
  this->checkpoint       = checkpoint;
  this->matrixName       = matrixName;
}// end of set
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//
