/**
 * @file        MatrixContainer.cpp
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the matrix container.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        02 December  2014, 16:17 (created) \n
 *              20 July      2017, 14:12 (revised)
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

#include <stdexcept>

#include <Containers/MatrixContainer.h>
#include <Parameters/Parameters.h>
#include <Logger/ErrorMessages.h>
#include <Logger/Logger.h>

//------------------------------------------------------------------------------------------------//
//------------------------------------------ Constants -------------------------------------------//
//------------------------------------------------------------------------------------------------//


//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Constructor
 */
TMatrixContainer::TMatrixContainer() : matrixContainer()
{

}// end of Constructor.
//--------------------------------------------------------------------------------------------------


/**
 * Destructor.
 * No need for virtual destructor (no polymorphism).
 */
TMatrixContainer::~TMatrixContainer()
{
  matrixContainer.clear();
}// end of ~TMatrixContainer
//--------------------------------------------------------------------------------------------------

/*
 * Create all matrix objects in the container.
 * @throw bad_alloc - usually due to out of memory.
 */
void TMatrixContainer::CreateMatrices()
{
 using MatrixType = TMatrixRecord::TMatrixType;

  for (auto& it : matrixContainer)
  {
    if (it.second.matrixPtr != nullptr)
    { // the data is already allocated
      throw std::invalid_argument(Logger::formatMessage(kErrFmtRelocationError,
                                                         it.second.matrixName.c_str()));
    }

    switch (it.second.matrixType)
    {
      case MatrixType::REAL:
      {
        it.second.matrixPtr = new RealMatrix(it.second.dimensionSizes);
        break;
      }

      case MatrixType::COMPLEX:
      {
        it.second.matrixPtr = new ComplexMatrix(it.second.dimensionSizes);
        break;
      }

      case MatrixType::INDEX:
      {
        it.second.matrixPtr = new IndexMatrix(it.second.dimensionSizes);
        break;
      }

      case MatrixType::CUFFT:
      {
        it.second.matrixPtr = new CufftComplexMatrix(it.second.dimensionSizes);
        break;
      }

      default: // unknown matrix type
      {
        throw std::invalid_argument(Logger::formatMessage(kErrFmtBadMatrixDistributionType,
                                                           it.second.matrixName.c_str()));
        break;
      }
    }// switch
  }// end for
}// end of CreateAllObjects
//--------------------------------------------------------------------------------------------------

/**
 * This function creates the list of matrices being used in the simulation. It is done based on the
 * simulation parameters. All matrices records are created here.
 */
void TMatrixContainer::AddMatrices()
{
  using MatrixType = TMatrixRecord::TMatrixType;
  using MatrixId   = TMatrixContainer::TMatrixIdx;

  const Parameters& params = Parameters::getInstance();

  DimensionSizes fullDims    = params.getFullDimensionSizes();
  DimensionSizes reducedDims = params.getReducedDimensionSizes();

  // this cannot be constexpr because of Visual studio 12.
  const bool LOAD         = true;
  const bool NOLOAD       = false;
  const bool CHECKPOINT   = true;
  const bool NOCHECKPOINT = false;

  //----------------------------------------- Allocate all matrices ------------------------------//

  matrixContainer[MatrixId::kappa] .Set(MatrixType::REAL, reducedDims, NOLOAD, NOCHECKPOINT, kKappaRName);

  if (!params.getC0ScalarFlag())
  {
    matrixContainer[MatrixId::c2]  .Set(MatrixType::REAL,    fullDims,   LOAD, NOCHECKPOINT, kC0Name);
  }

  matrixContainer[MatrixId::p]     .Set(MatrixType::REAL,    fullDims, NOLOAD,   CHECKPOINT, kPName);

  matrixContainer[MatrixId::rhox]  .Set(MatrixType::REAL,    fullDims, NOLOAD,   CHECKPOINT, kRhoxName);
  matrixContainer[MatrixId::rhoy]  .Set(MatrixType::REAL,    fullDims, NOLOAD,   CHECKPOINT, kRhoyName);
  matrixContainer[MatrixId::rhoz]  .Set(MatrixType::REAL,    fullDims, NOLOAD,   CHECKPOINT, kRhozName);

  matrixContainer[MatrixId::ux_sgx].Set(MatrixType::REAL,    fullDims, NOLOAD,   CHECKPOINT, kUxSgxName);
  matrixContainer[MatrixId::uy_sgy].Set(MatrixType::REAL,    fullDims, NOLOAD,   CHECKPOINT, kUySgyName);
  matrixContainer[MatrixId::uz_sgz].Set(MatrixType::REAL,    fullDims, NOLOAD,   CHECKPOINT, kUzSgzName);

  matrixContainer[MatrixId::duxdx] .Set(MatrixType::REAL,    fullDims, NOLOAD, NOCHECKPOINT, kDuxdxName);
  matrixContainer[MatrixId::duydy] .Set(MatrixType::REAL,    fullDims, NOLOAD, NOCHECKPOINT, kDuydyName);
  matrixContainer[MatrixId::duzdz] .Set(MatrixType::REAL,    fullDims, NOLOAD, NOCHECKPOINT, kDuzdzName);

  if (!params.getRho0ScalarFlag())
  {
    matrixContainer[MatrixId::rho0]       .Set(MatrixType::REAL, fullDims, LOAD, NOCHECKPOINT, kRho0Name);
    matrixContainer[MatrixId::dt_rho0_sgx].Set(MatrixType::REAL, fullDims, LOAD, NOCHECKPOINT, kRho0SgxName);
    matrixContainer[MatrixId::dt_rho0_sgy].Set(MatrixType::REAL, fullDims, LOAD, NOCHECKPOINT, kRho0SgyName);
    matrixContainer[MatrixId::dt_rho0_sgz].Set(MatrixType::REAL, fullDims, LOAD, NOCHECKPOINT, kRho0SgzName);
  }

  matrixContainer[MatrixId::ddx_k_shift_pos].Set(MatrixType::COMPLEX, DimensionSizes(reducedDims.nx, 1, 1), LOAD, NOCHECKPOINT, kDdxKShiftPosRName);
  matrixContainer[MatrixId::ddy_k_shift_pos].Set(MatrixType::COMPLEX, DimensionSizes(1, reducedDims.ny, 1), LOAD, NOCHECKPOINT, kDdyKShiftPosName);
  matrixContainer[MatrixId::ddz_k_shift_pos].Set(MatrixType::COMPLEX, DimensionSizes(1, 1, reducedDims.nz), LOAD, NOCHECKPOINT, kDdzKShiftPosName);

  matrixContainer[MatrixId::ddx_k_shift_neg].Set(MatrixType::COMPLEX, DimensionSizes(reducedDims.nx ,1, 1), LOAD, NOCHECKPOINT, kDdxKShiftNegRName);
  matrixContainer[MatrixId::ddy_k_shift_neg].Set(MatrixType::COMPLEX, DimensionSizes(1, reducedDims.ny, 1), LOAD, NOCHECKPOINT, kDdyKShiftNegName);
  matrixContainer[MatrixId::ddz_k_shift_neg].Set(MatrixType::COMPLEX, DimensionSizes(1, 1, reducedDims.nz), LOAD, NOCHECKPOINT, kDdzKShiftNegName);

  matrixContainer[MatrixId::pml_x_sgx].Set(MatrixType::REAL, DimensionSizes(fullDims.nx, 1, 1), LOAD, NOCHECKPOINT, kPmlXSgxName);
  matrixContainer[MatrixId::pml_y_sgy].Set(MatrixType::REAL, DimensionSizes(1, fullDims.ny, 1), LOAD, NOCHECKPOINT, kPmlYSgyName);
  matrixContainer[MatrixId::pml_z_sgz].Set(MatrixType::REAL, DimensionSizes(1, 1, fullDims.nz), LOAD, NOCHECKPOINT, kPmlZSgzName);

  matrixContainer[MatrixId::pml_x].Set(MatrixType::REAL, DimensionSizes(fullDims.nx, 1, 1), LOAD, NOCHECKPOINT, kPmlXName);
  matrixContainer[MatrixId::pml_y].Set(MatrixType::REAL, DimensionSizes(1, fullDims.ny, 1), LOAD, NOCHECKPOINT, kPmlYName);
  matrixContainer[MatrixId::pml_z].Set(MatrixType::REAL, DimensionSizes(1, 1, fullDims.nz), LOAD, NOCHECKPOINT, kPmlZName);

  if (params.getNonLinearFlag())
  {
    if (! params.getBOnAScalarFlag())
    {
      matrixContainer[MatrixId::BonA].Set(MatrixType::REAL, fullDims, LOAD, NOCHECKPOINT, kBonAName);
    }
  }

  if (params.getAbsorbingFlag() != 0)
  {
    if (!((params.getC0ScalarFlag()) && (params.getAlphaCoeffScalarFlag())))
    {
      matrixContainer[MatrixId::absorb_tau].Set(MatrixType::REAL, fullDims, NOLOAD, NOCHECKPOINT, kAbsorbTauName);
      matrixContainer[MatrixId::absorb_eta].Set(MatrixType::REAL, fullDims, NOLOAD, NOCHECKPOINT, kAbsorbEtaName);
    }

    matrixContainer[MatrixId::absorb_nabla1].Set(MatrixType::REAL, reducedDims, NOLOAD, NOCHECKPOINT, kAbsorbNabla1RName);
    matrixContainer[MatrixId::absorb_nabla2].Set(MatrixType::REAL, reducedDims, NOLOAD, NOCHECKPOINT, kAbsorbNabla2RName);
  }

  // linear sensor mask
  if (params.getSensorMaskType() == Parameters::SensorMaskType::kIndex)
  {
    matrixContainer[MatrixId::sensor_mask_index].Set(MatrixType::INDEX,
                                                     DimensionSizes(params.getSensorMaskIndexSize(), 1, 1),
                                                     LOAD, NOCHECKPOINT, kSensorMaskIndexName);
  }

  // cuboid sensor mask
  if (params.getSensorMaskType() == Parameters::SensorMaskType::kCorners)
  {
    matrixContainer[MatrixId::sensor_mask_corners].Set(MatrixType::INDEX,
                                                       DimensionSizes(6, params.getSensorMaskCornersSize(), 1),
                                                       LOAD, NOCHECKPOINT, kSensorMaskCornersName);
  }

  // if p0 source flag
  if (params.getInitialPressureSourceFlag() == 1)
  {
    matrixContainer[MatrixId::p0_source_input].Set(MatrixType::REAL, fullDims, LOAD, NOCHECKPOINT, kP0SourceInputName);
  }

  // u_source_index
  if ((params.getTransducerSourceFlag() != 0) ||
      (params.getVelocityXSourceFlag() != 0)         ||
      (params.getVelocityYSourceFlag() != 0)         ||
      (params.getVelocityZSourceFlag() != 0))
  {
    matrixContainer[MatrixId::u_source_index].Set(MatrixType::INDEX,
                                                  DimensionSizes(1, 1, params.getVelocitySourceIndexSize()),
                                                  LOAD, NOCHECKPOINT, kVelocitySourceIndexName);
  }

  //transducer source flag defined
  if (params.getTransducerSourceFlag() != 0)
  {
    matrixContainer[MatrixId::delay_mask].Set(MatrixType::INDEX,
                                              DimensionSizes(1 ,1, params.getVelocitySourceIndexSize()),
                                              LOAD, NOCHECKPOINT, kDelayMaskName);

    matrixContainer[MatrixId::transducer_source_input].Set(MatrixType::REAL,
                                                           DimensionSizes(1 ,1, params.getTransducerSourceInputSize()),
                                                           LOAD, NOCHECKPOINT, kTransducerSourceInputName);
  }

  // p variables
  if (params.getPressureSourceFlag() != 0)
  {
    if (params.getPressureSourceMany() == 0)
    { // 1D case
      matrixContainer[MatrixId::p_source_input].Set(MatrixType::REAL,
                                                    DimensionSizes(1, 1, params.getPressureSourceFlag()),
                                                    LOAD, NOCHECKPOINT, kPressureSourceInputName);
    }
    else
    { // 2D case
      matrixContainer[MatrixId::p_source_input].Set(MatrixType::REAL,
                                                    DimensionSizes(1, params.getPressureSourceIndexSize(),params.getPressureSourceFlag()),
                                                    LOAD, NOCHECKPOINT, kPressureSourceInputName);
    }

    matrixContainer[MatrixId::p_source_index].Set(MatrixType::INDEX,
                                                  DimensionSizes(1, 1, params.getPressureSourceIndexSize()),
                                                  LOAD, NOCHECKPOINT, kPressureSourceIndexName);
  }

  //------------------------------------ uxyz source flags ---------------------------------------//
  if (params.getVelocityXSourceFlag() != 0)
  {
    if (params.getVelocitySourceMany() == 0)
    { // 1D
      matrixContainer[MatrixId::ux_source_input].Set(MatrixType::REAL,
                                                     DimensionSizes(1, 1, params.getVelocityXSourceFlag()),
                                                     LOAD, NOCHECKPOINT, kVelocityXSourceInputName);
    }
    else
    { // 2D
      matrixContainer[MatrixId::ux_source_input].Set(MatrixType::REAL,
                                                     DimensionSizes(1, params.getVelocitySourceIndexSize(), params.getVelocityXSourceFlag()),
                                                     LOAD, NOCHECKPOINT, kVelocityXSourceInputName);
    }
  }// ux_source_input

  if (params.getVelocityYSourceFlag() != 0)
  {
    if (params.getVelocitySourceMany() == 0)
    { // 1D
      matrixContainer[MatrixId::uy_source_input].Set(MatrixType::REAL,
                                                     DimensionSizes(1, 1, params.getVelocityYSourceFlag()),
                                                     LOAD, NOCHECKPOINT, kVelocityYSourceInputName);
    }
    else
    { // 2D
      matrixContainer[MatrixId::uy_source_input].Set(MatrixType::REAL,
                                                     DimensionSizes(1,params.getVelocitySourceIndexSize(),params.getVelocityYSourceFlag()),
                                                     LOAD, NOCHECKPOINT, kVelocityYSourceInputName);
    }
  }// uy_source_input

  if (params.getVelocityZSourceFlag() != 0)
  {
    if (params.getVelocitySourceMany() == 0)
    { // 1D
      matrixContainer[MatrixId::uz_source_input].Set(MatrixType::REAL,
                                                     DimensionSizes(1, 1, params.getVelocityZSourceFlag()),
                                                     LOAD, NOCHECKPOINT, kVelocityZSourceInputName);
    }
    else
    { // 2D
      matrixContainer[MatrixId::uz_source_input].Set(MatrixType::REAL,
                                                     DimensionSizes(1, params.getVelocitySourceIndexSize(), params.getVelocityZSourceFlag()),
                                                     LOAD, NOCHECKPOINT, kVelocityZSourceInputName);
    }
  }// uz_source_input

  //-- Nonlinear grid
  if (params.getNonUniformGridFlag()!= 0)
  {
    matrixContainer[MatrixId::dxudxn].Set(MatrixType::REAL, DimensionSizes(fullDims.nx, 1, 1), LOAD, NOCHECKPOINT, kDxudxnName);
    matrixContainer[MatrixId::dyudyn].Set(MatrixType::REAL, DimensionSizes(1, fullDims.ny, 1), LOAD, NOCHECKPOINT, kDyudynName);
    matrixContainer[MatrixId::dzudzn].Set(MatrixType::REAL, DimensionSizes(1 ,1, fullDims.nz), LOAD, NOCHECKPOINT, kDzudznName);

    matrixContainer[MatrixId::dxudxn_sgx].Set(MatrixType::REAL, DimensionSizes(fullDims.nx, 1, 1), LOAD, NOCHECKPOINT, kDxudxnSgxName);
    matrixContainer[MatrixId::dyudyn_sgy].Set(MatrixType::REAL, DimensionSizes(1, fullDims.ny, 1), LOAD, NOCHECKPOINT, kDyudynSgyName);
    matrixContainer[MatrixId::dzudzn_sgz].Set(MatrixType::REAL, DimensionSizes(1 ,1, fullDims.nz), LOAD, NOCHECKPOINT, kDzudznSgzName);
  }

  //-- u_non_staggered_raw
  if (params.getStoreVelocityNonStaggeredRaw())
  {
    DimensionSizes shiftDims = fullDims;

    const size_t nx_2 = fullDims.nx / 2 + 1;
    const size_t ny_2 = fullDims.ny / 2 + 1;
    const size_t nz_2 = fullDims.nz / 2 + 1;

    size_t xCutSize = nx_2        * fullDims.ny * fullDims.nz;
    size_t yCutSize = fullDims.nx * ny_2        * fullDims.nz;
    size_t zCutSize = fullDims.nx * fullDims.ny * nz_2;

    if ((xCutSize >= yCutSize) && (xCutSize >= zCutSize))
    {
      // X cut is the biggest
      shiftDims.nx = nx_2;
    }
    else if ((yCutSize >= xCutSize) && (yCutSize >= zCutSize))
    {
      // Y cut is the biggest
      shiftDims.ny = ny_2;
    }
    else if ((zCutSize >= xCutSize) && (zCutSize >= yCutSize))
    {
      // Z cut is the biggest
      shiftDims.nz = nz_2;
    }
    else
    {
      //all are the same
      shiftDims.nx = nx_2;
    }

    matrixContainer[MatrixId::cufft_shift_temp].Set(MatrixType::CUFFT, shiftDims, NOLOAD, NOCHECKPOINT, kCufftShiftTempName);


    // these three are necessary only for u_non_staggered calculation now
    matrixContainer[MatrixId::ux_shifted].Set(MatrixType::REAL, fullDims, NOLOAD, NOCHECKPOINT, kUxShiftedName);
    matrixContainer[MatrixId::uy_shifted].Set(MatrixType::REAL, fullDims, NOLOAD, NOCHECKPOINT, kUyShiftedName);
    matrixContainer[MatrixId::uz_shifted].Set(MatrixType::REAL, fullDims, NOLOAD, NOCHECKPOINT, kUzShiftedName);

    // shifts from the input file
    matrixContainer[MatrixId::x_shift_neg_r].Set(MatrixType::COMPLEX, DimensionSizes(nx_2, 1, 1), LOAD,NOCHECKPOINT, kXShiftNegRName);
    matrixContainer[MatrixId::y_shift_neg_r].Set(MatrixType::COMPLEX, DimensionSizes(1, ny_2, 1), LOAD,NOCHECKPOINT, kYShiftNegRName);
    matrixContainer[MatrixId::z_shift_neg_r].Set(MatrixType::COMPLEX, DimensionSizes(1, 1, nz_2), LOAD,NOCHECKPOINT, kZShiftNegRName);
  }// u_non_staggered


  //------------------------------------- Temporary matrices -------------------------------------//
  // this matrix used to load alpha_coeff for absorb_tau pre-calculation

  if ((params.getAbsorbingFlag() != 0) && (!params.getAlphaCoeffScalarFlag()))
  {
    matrixContainer[MatrixId::temp_1_real_3D].Set(MatrixType::REAL, fullDims, LOAD, NOCHECKPOINT, kAlphaCoeffName);
  }
  else
  {
    matrixContainer[MatrixId::temp_1_real_3D].Set(MatrixType::REAL, fullDims, NOLOAD, NOCHECKPOINT, kTemp1Real3DName);
  }

  matrixContainer[MatrixId::temp_2_real_3D].Set(MatrixType::REAL, fullDims, NOLOAD, NOCHECKPOINT, kTemp2Real3DName);
  matrixContainer[MatrixId::temp_3_real_3D].Set(MatrixType::REAL, fullDims, NOLOAD, NOCHECKPOINT, kTemp3Real3DName);

  matrixContainer[MatrixId::cufft_x_temp].Set(MatrixType::CUFFT, reducedDims, NOLOAD, NOCHECKPOINT, kCufftXTempName);
  matrixContainer[MatrixId::cufft_y_temp].Set(MatrixType::CUFFT, reducedDims, NOLOAD, NOCHECKPOINT, kCufftYTempName);
  matrixContainer[MatrixId::cufft_z_temp].Set(MatrixType::CUFFT, reducedDims, NOLOAD, NOCHECKPOINT, kCufftZTempName);
}// end of AddMatricesIntoContainer
//--------------------------------------------------------------------------------------------------



/**
 * Load all marked matrices from the input HDF5 file.
 * @param [in] inputFile - HDF5 input file handle
 */
void TMatrixContainer::LoadDataFromInputFile(Hdf5File& inputFile)
{
  for (const auto& it : matrixContainer)
  {
    if (it.second.loadData)
    {
      it.second.matrixPtr->readData(inputFile, it.second.matrixName);
    }
  }
}// end of LoadDataFromInputFile
//--------------------------------------------------------------------------------------------------

/**
 * Load selected matrices from the checkpoint HDF5 file.
 * @param [in] checkpointFile - HDF5 checkpoint file handle
 */
void TMatrixContainer::LoadDataFromCheckpointFile(Hdf5File& checkpointFile)
{
  for (const auto& it : matrixContainer)
  {
    if (it.second.checkpoint)
    {
      it.second.matrixPtr->readData(checkpointFile,it.second.matrixName);
    }
  }
}// end of LoadDataFromCheckpointFile
//--------------------------------------------------------------------------------------------------

/**
 * Store selected matrices into the checkpoint file.
 * @param [in] checkpointFile - Checkpoint file
 */
void TMatrixContainer::StoreDataIntoCheckpointFile(Hdf5File& checkpointFile)
{
  for (const auto& it : matrixContainer)
  {
    if (it.second.checkpoint)
    {
      // Copy data from device first
      it.second.matrixPtr->copyFromDevice();
      // store data to the checkpoint file
      it.second.matrixPtr->writeData(checkpointFile,
                                               it.second.matrixName,
                                               Parameters::getInstance().getCompressionLevel());
    }
  }
}// end of StoreDataIntoCheckpointFile
//--------------------------------------------------------------------------------------------------

/**
 * Free all matrix objects.
 */
void TMatrixContainer::FreeMatrices()
{
  for (auto& it : matrixContainer)
  {
    if (it.second.matrixPtr)
    {
      delete it.second.matrixPtr;
      it.second.matrixPtr = nullptr;
    }
  }
}// end of FreeMatrices
//--------------------------------------------------------------------------------------------------

/**
 * Copy all matrices over to the GPU.
 */
void TMatrixContainer::CopyMatricesToDevice()
{
  for (const auto& it : matrixContainer)
  {
    it.second.matrixPtr->copyToDevice();
  }
}//end of CopyMatricesToDevice
//--------------------------------------------------------------------------------------------------

/**
 * Copy all matrices back over to CPU. Can be used for debugging purposes.
 */
void TMatrixContainer::CopyMatricesFromDevice()
{
  for (const auto& it : matrixContainer)
  {
    it.second.matrixPtr->copyFromDevice();
  }
}// end of CopyAllMatricesFromGPU
//--------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//
