/**
 * @file        MatrixRecord.cpp
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing metadata about matrices stored in the matrix
 *              container.
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

//----------------------------------------------------------------------------//
//--------------------------- CONSTANTS --------------------------------------//
//----------------------------------------------------------------------------//


//----------------------------------------------------------------------------//
//--------------------------- Public methods ---------------------------------//
//----------------------------------------------------------------------------//

#include <Containers/MatrixRecord.h>

/**
 * Default constructor.
 */
TMatrixRecord::TMatrixRecord() :
        matrixPtr(nullptr),
        dataType(mdtReal),
        dimensionSizes(),
        loadData(false),
        checkpoint(false),
        matrixName("")
{

}// end of constructor
//--------------------------------------------------------------------------------------------------

/**
 * Copy constructor of TMatrixRecord.
 * @param [in] src - matrix record to be copied from.
 */
TMatrixRecord::TMatrixRecord(const TMatrixRecord& src) :
        matrixPtr(src.matrixPtr),
        dataType(src.dataType),
        dimensionSizes(src.dimensionSizes),
        loadData(src.loadData),
        checkpoint(src.checkpoint),
        matrixName(src.matrixName)
{

}// end of TMatrixRecord
//--------------------------------------------------------------------------------------------------


/**
 * operator = of TMatrixRecord.
 * @param [in] src - source object
 * @return a filled object
 */
TMatrixRecord& TMatrixRecord::operator = (const TMatrixRecord& src)
{
  if (this != &src)
  {
    matrixPtr       = src.matrixPtr;
    dataType        = src.dataType;
    dimensionSizes  = src.dimensionSizes;
    loadData        = src.loadData;
    checkpoint      = src.checkpoint;
    matrixName      = src.matrixName;
  }

  return *this;
}// end of operator =
//--------------------------------------------------------------------------------------------------

/**
 * Set all values for the record.
 * @param [in] matrixPtr      - Pointer to the MatrixClass object
 * @param [in] matrixDataType - Matrix data type
 * @param [in] dimensionSizes - Dimension sizes
 * @param [in] loadData       - Load data from file?
 * @param [in] checkpoint     - Checkpoint this matrix?
 * @param [in] matrixName     - HDF5 matrix name
 */
void TMatrixRecord::Set(const TMatrixDataType matrixDataType,
                        const TDimensionSizes dimensionSizes,
                        const bool            loadData,
                        const bool            checkpoint,
                        const string          matrixName)
{
  this->matrixPtr        = nullptr;
  this->dataType         = matrixDataType;
  this->dimensionSizes   = dimensionSizes;
  this->loadData         = loadData;
  this->checkpoint       = checkpoint;
  this->matrixName       = matrixName;
}// end of SetAllValues
//--------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//
