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
 *              28 June      2017, 14:10 (revised)
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
TMatrixContainer::TMatrixContainer()
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
      throw std::invalid_argument(TLogger::FormatMessage(ERR_FMT_RELOCATION_ERROR,
                                                         it.second.matrixName.c_str()));
    }

    switch (it.second.matrixType)
    {
      case MatrixType::REAL:
      {
        it.second.matrixPtr = new TRealMatrix(it.second.dimensionSizes);
        break;
      }

      case MatrixType::COMPLEX:
      {
        it.second.matrixPtr = new TComplexMatrix(it.second.dimensionSizes);
        break;
      }

      case MatrixType::INDEX:
      {
        it.second.matrixPtr = new TIndexMatrix(it.second.dimensionSizes);
        break;
      }

      case MatrixType::CUFFT:
      {
        it.second.matrixPtr = new TCUFFTComplexMatrix(it.second.dimensionSizes);
        break;
      }

      default: // unknown matrix type
      {
        throw std::invalid_argument(TLogger::FormatMessage(ERR_FMT_BAD_MATRIX_DISTRIBUTION_TYPE,
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

  const TParameters& params = TParameters::GetInstance();

  TDimensionSizes fullDims    = params.GetFullDimensionSizes();
  TDimensionSizes reducedDims = params.GetReducedDimensionSizes();

  // this cannot be constexpr because of Visual studio 12.
  const bool LOAD         = true;
  const bool NOLOAD       = false;
  const bool CHECKPOINT   = true;
  const bool NOCHECKPOINT = false;

  //----------------------------------------- Allocate all matrices ------------------------------//

  matrixContainer[MatrixId::kappa] .Set(MatrixType::REAL, reducedDims, NOLOAD, NOCHECKPOINT, kappa_r_NAME);

  if (!params.Get_c0_scalar_flag())
  {
    matrixContainer[MatrixId::c2]  .Set(MatrixType::REAL,    fullDims,   LOAD, NOCHECKPOINT, c0_NAME);
  }

  matrixContainer[MatrixId::p]     .Set(MatrixType::REAL,    fullDims, NOLOAD,   CHECKPOINT, p_NAME);

  matrixContainer[MatrixId::rhox]  .Set(MatrixType::REAL,    fullDims, NOLOAD,   CHECKPOINT, rhox_NAME);
  matrixContainer[MatrixId::rhoy]  .Set(MatrixType::REAL,    fullDims, NOLOAD,   CHECKPOINT, rhoy_NAME);
  matrixContainer[MatrixId::rhoz]  .Set(MatrixType::REAL,    fullDims, NOLOAD,   CHECKPOINT, rhoz_NAME);

  matrixContainer[MatrixId::ux_sgx].Set(MatrixType::REAL,    fullDims, NOLOAD,   CHECKPOINT, ux_sgx_NAME);
  matrixContainer[MatrixId::uy_sgy].Set(MatrixType::REAL,    fullDims, NOLOAD,   CHECKPOINT, uy_sgy_NAME);
  matrixContainer[MatrixId::uz_sgz].Set(MatrixType::REAL,    fullDims, NOLOAD,   CHECKPOINT, uz_sgz_NAME);

  matrixContainer[MatrixId::duxdx] .Set(MatrixType::REAL,    fullDims, NOLOAD, NOCHECKPOINT, duxdx_NAME);
  matrixContainer[MatrixId::duydy] .Set(MatrixType::REAL,    fullDims, NOLOAD, NOCHECKPOINT, duydy_NAME);
  matrixContainer[MatrixId::duzdz] .Set(MatrixType::REAL,    fullDims, NOLOAD, NOCHECKPOINT, duzdz_NAME);

  if (!params.Get_rho0_scalar_flag())
  {
    matrixContainer[MatrixId::rho0]       .Set(MatrixType::REAL, fullDims, LOAD, NOCHECKPOINT, rho0_NAME);
    matrixContainer[MatrixId::dt_rho0_sgx].Set(MatrixType::REAL, fullDims, LOAD, NOCHECKPOINT, rho0_sgx_NAME);
    matrixContainer[MatrixId::dt_rho0_sgy].Set(MatrixType::REAL, fullDims, LOAD, NOCHECKPOINT, rho0_sgy_NAME);
    matrixContainer[MatrixId::dt_rho0_sgz].Set(MatrixType::REAL, fullDims, LOAD, NOCHECKPOINT, rho0_sgz_NAME);
  }

  matrixContainer[MatrixId::ddx_k_shift_pos].Set(MatrixType::COMPLEX, TDimensionSizes(reducedDims.nx, 1, 1), LOAD, NOCHECKPOINT, ddx_k_shift_pos_r_NAME);
  matrixContainer[MatrixId::ddy_k_shift_pos].Set(MatrixType::COMPLEX, TDimensionSizes(1, reducedDims.ny, 1), LOAD, NOCHECKPOINT, ddy_k_shift_pos_NAME);
  matrixContainer[MatrixId::ddz_k_shift_pos].Set(MatrixType::COMPLEX, TDimensionSizes(1, 1, reducedDims.nz), LOAD, NOCHECKPOINT, ddz_k_shift_pos_NAME);

  matrixContainer[MatrixId::ddx_k_shift_neg].Set(MatrixType::COMPLEX, TDimensionSizes(reducedDims.nx ,1, 1), LOAD, NOCHECKPOINT, ddx_k_shift_neg_r_NAME);
  matrixContainer[MatrixId::ddy_k_shift_neg].Set(MatrixType::COMPLEX, TDimensionSizes(1, reducedDims.ny, 1), LOAD, NOCHECKPOINT, ddy_k_shift_neg_NAME);
  matrixContainer[MatrixId::ddz_k_shift_neg].Set(MatrixType::COMPLEX, TDimensionSizes(1, 1, reducedDims.nz), LOAD, NOCHECKPOINT, ddz_k_shift_neg_NAME);

  matrixContainer[MatrixId::pml_x_sgx].Set(MatrixType::REAL, TDimensionSizes(fullDims.nx, 1, 1), LOAD, NOCHECKPOINT, pml_x_sgx_NAME);
  matrixContainer[MatrixId::pml_y_sgy].Set(MatrixType::REAL, TDimensionSizes(1, fullDims.ny, 1), LOAD, NOCHECKPOINT, pml_y_sgy_NAME);
  matrixContainer[MatrixId::pml_z_sgz].Set(MatrixType::REAL, TDimensionSizes(1, 1, fullDims.nz), LOAD, NOCHECKPOINT, pml_z_sgz_NAME);

  matrixContainer[MatrixId::pml_x].Set(MatrixType::REAL, TDimensionSizes(fullDims.nx, 1, 1), LOAD, NOCHECKPOINT, pml_x_NAME);
  matrixContainer[MatrixId::pml_y].Set(MatrixType::REAL, TDimensionSizes(1, fullDims.ny, 1), LOAD, NOCHECKPOINT, pml_y_NAME);
  matrixContainer[MatrixId::pml_z].Set(MatrixType::REAL, TDimensionSizes(1, 1, fullDims.nz), LOAD, NOCHECKPOINT, pml_z_NAME);

  if (params.Get_nonlinear_flag())
  {
    if (! params.Get_BonA_scalar_flag())
    {
      matrixContainer[MatrixId::BonA].Set(MatrixType::REAL, fullDims, LOAD, NOCHECKPOINT, BonA_NAME);
    }
  }

  if (params.Get_absorbing_flag() != 0)
  {
    if (!((params.Get_c0_scalar_flag()) && (params.Get_alpha_coeff_scalar_flag())))
    {
      matrixContainer[MatrixId::absorb_tau].Set(MatrixType::REAL, fullDims, NOLOAD, NOCHECKPOINT, absorb_tau_NAME);
      matrixContainer[MatrixId::absorb_eta].Set(MatrixType::REAL, fullDims, NOLOAD, NOCHECKPOINT, absorb_eta_NAME);
    }

    matrixContainer[MatrixId::absorb_nabla1].Set(MatrixType::REAL, reducedDims, NOLOAD, NOCHECKPOINT, absorb_nabla1_r_NAME);
    matrixContainer[MatrixId::absorb_nabla2].Set(MatrixType::REAL, reducedDims, NOLOAD, NOCHECKPOINT, absorb_nabla2_r_NAME);
  }

  // linear sensor mask
  if (params.Get_sensor_mask_type() == TParameters::TSensorMaskType::INDEX)
  {
    matrixContainer[MatrixId::sensor_mask_index].Set(MatrixType::INDEX,
                                                     TDimensionSizes(params.Get_sensor_mask_index_size(), 1, 1),
                                                     LOAD, NOCHECKPOINT, sensor_mask_index_NAME);
  }

  // cuboid sensor mask
  if (params.Get_sensor_mask_type() == TParameters::TSensorMaskType::CORNERS)
  {
    matrixContainer[MatrixId::sensor_mask_corners].Set(MatrixType::INDEX,
                                                       TDimensionSizes(6, params.Get_sensor_mask_corners_size(), 1),
                                                       LOAD, NOCHECKPOINT, sensor_mask_corners_NAME);
  }

  // if p0 source flag
  if (params.Get_p0_source_flag() == 1)
  {
    matrixContainer[MatrixId::p0_source_input].Set(MatrixType::REAL, fullDims, LOAD, NOCHECKPOINT, p0_source_input_NAME);
  }

  // u_source_index
  if ((params.Get_transducer_source_flag() != 0) ||
      (params.Get_ux_source_flag() != 0)         ||
      (params.Get_uy_source_flag() != 0)         ||
      (params.Get_uz_source_flag() != 0))
  {
    matrixContainer[MatrixId::u_source_index].Set(MatrixType::INDEX,
                                                  TDimensionSizes(1, 1, params.Get_u_source_index_size()),
                                                  LOAD, NOCHECKPOINT, u_source_index_NAME);
  }

  //transducer source flag defined
  if (params.Get_transducer_source_flag() != 0)
  {
    matrixContainer[MatrixId::delay_mask].Set(MatrixType::INDEX,
                                              TDimensionSizes(1 ,1, params.Get_u_source_index_size()),
                                              LOAD, NOCHECKPOINT, delay_mask_NAME);

    matrixContainer[MatrixId::transducer_source_input].Set(MatrixType::REAL,
                                                           TDimensionSizes(1 ,1, params.Get_transducer_source_input_size()),
                                                           LOAD, NOCHECKPOINT, transducer_source_input_NAME);
  }

  // p variables
  if (params.Get_p_source_flag() != 0)
  {
    if (params.Get_p_source_many() == 0)
    { // 1D case
      matrixContainer[MatrixId::p_source_input].Set(MatrixType::REAL,
                                                    TDimensionSizes(1, 1, params.Get_p_source_flag()),
                                                    LOAD, NOCHECKPOINT, p_source_input_NAME);
    }
    else
    { // 2D case
      matrixContainer[MatrixId::p_source_input].Set(MatrixType::REAL,
                                                    TDimensionSizes(1, params.Get_p_source_index_size(),params.Get_p_source_flag()),
                                                    LOAD, NOCHECKPOINT, p_source_input_NAME);
    }

    matrixContainer[MatrixId::p_source_index].Set(MatrixType::INDEX,
                                                  TDimensionSizes(1, 1, params.Get_p_source_index_size()),
                                                  LOAD, NOCHECKPOINT, p_source_index_NAME);
  }

  //------------------------------------ uxyz source flags ---------------------------------------//
  if (params.Get_ux_source_flag() != 0)
  {
    if (params.Get_u_source_many() == 0)
    { // 1D
      matrixContainer[MatrixId::ux_source_input].Set(MatrixType::REAL,
                                                     TDimensionSizes(1, 1, params.Get_ux_source_flag()),
                                                     LOAD, NOCHECKPOINT, ux_source_input_NAME);
    }
    else
    { // 2D
      matrixContainer[MatrixId::ux_source_input].Set(MatrixType::REAL,
                                                     TDimensionSizes(1, params.Get_u_source_index_size(), params.Get_ux_source_flag()),
                                                     LOAD, NOCHECKPOINT, ux_source_input_NAME);
    }
  }// ux_source_input

  if (params.Get_uy_source_flag() != 0)
  {
    if (params.Get_u_source_many() == 0)
    { // 1D
      matrixContainer[MatrixId::uy_source_input].Set(MatrixType::REAL,
                                                     TDimensionSizes(1, 1, params.Get_uy_source_flag()),
                                                     LOAD, NOCHECKPOINT, uy_source_input_NAME);
    }
    else
    { // 2D
      matrixContainer[MatrixId::uy_source_input].Set(MatrixType::REAL,
                                                     TDimensionSizes(1,params.Get_u_source_index_size(),params.Get_uy_source_flag()),
                                                     LOAD, NOCHECKPOINT, uy_source_input_NAME);
    }
  }// uy_source_input

  if (params.Get_uz_source_flag() != 0)
  {
    if (params.Get_u_source_many() == 0)
    { // 1D
      matrixContainer[MatrixId::uz_source_input].Set(MatrixType::REAL,
                                                     TDimensionSizes(1, 1, params.Get_uz_source_flag()),
                                                     LOAD, NOCHECKPOINT, uz_source_input_NAME);
    }
    else
    { // 2D
      matrixContainer[MatrixId::uz_source_input].Set(MatrixType::REAL,
                                                     TDimensionSizes(1, params.Get_u_source_index_size(), params.Get_uz_source_flag()),
                                                     LOAD, NOCHECKPOINT, uz_source_input_NAME);
    }
  }// uz_source_input

  //-- Nonlinear grid
  if (params.Get_nonuniform_grid_flag()!= 0)
  {
    matrixContainer[MatrixId::dxudxn].Set(MatrixType::REAL, TDimensionSizes(fullDims.nx, 1, 1), LOAD, NOCHECKPOINT, dxudxn_NAME);
    matrixContainer[MatrixId::dyudyn].Set(MatrixType::REAL, TDimensionSizes(1, fullDims.ny, 1), LOAD, NOCHECKPOINT, dyudyn_NAME);
    matrixContainer[MatrixId::dzudzn].Set(MatrixType::REAL, TDimensionSizes(1 ,1, fullDims.nz), LOAD, NOCHECKPOINT, dzudzn_NAME);

    matrixContainer[MatrixId::dxudxn_sgx].Set(MatrixType::REAL, TDimensionSizes(fullDims.nx, 1, 1), LOAD, NOCHECKPOINT, dxudxn_sgx_NAME);
    matrixContainer[MatrixId::dyudyn_sgy].Set(MatrixType::REAL, TDimensionSizes(1, fullDims.ny, 1), LOAD, NOCHECKPOINT, dyudyn_sgy_NAME);
    matrixContainer[MatrixId::dzudzn_sgz].Set(MatrixType::REAL, TDimensionSizes(1 ,1, fullDims.nz), LOAD, NOCHECKPOINT, dzudzn_sgz_NAME);
  }

  //-- u_non_staggered_raw
  if (params.IsStore_u_non_staggered_raw())
  {
    TDimensionSizes shiftDims = fullDims;

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

    matrixContainer[MatrixId::cufft_shift_temp].Set(MatrixType::CUFFT, shiftDims, NOLOAD, NOCHECKPOINT, cufft_shift_temp_NAME);


    // these three are necessary only for u_non_staggered calculation now
    matrixContainer[MatrixId::ux_shifted].Set(MatrixType::REAL, fullDims, NOLOAD, NOCHECKPOINT, ux_shifted_NAME);
    matrixContainer[MatrixId::uy_shifted].Set(MatrixType::REAL, fullDims, NOLOAD, NOCHECKPOINT, uy_shifted_NAME);
    matrixContainer[MatrixId::uz_shifted].Set(MatrixType::REAL, fullDims, NOLOAD, NOCHECKPOINT, uz_shifted_NAME);

    // shifts from the input file
    matrixContainer[MatrixId::x_shift_neg_r].Set(MatrixType::COMPLEX, TDimensionSizes(nx_2, 1, 1), LOAD,NOCHECKPOINT, x_shift_neg_r_NAME);
    matrixContainer[MatrixId::y_shift_neg_r].Set(MatrixType::COMPLEX, TDimensionSizes(1, ny_2, 1), LOAD,NOCHECKPOINT, y_shift_neg_r_NAME);
    matrixContainer[MatrixId::z_shift_neg_r].Set(MatrixType::COMPLEX, TDimensionSizes(1, 1, nz_2), LOAD,NOCHECKPOINT, z_shift_neg_r_NAME);
  }// u_non_staggered


  //------------------------------------- Temporary matrices -------------------------------------//
  // this matrix used to load alpha_coeff for absorb_tau pre-calculation

  if ((params.Get_absorbing_flag() != 0) && (!params.Get_alpha_coeff_scalar_flag()))
  {
    matrixContainer[MatrixId::temp_1_real_3D].Set(MatrixType::REAL, fullDims, LOAD, NOCHECKPOINT, alpha_coeff_NAME);
  }
  else
  {
    matrixContainer[MatrixId::temp_1_real_3D].Set(MatrixType::REAL, fullDims, NOLOAD, NOCHECKPOINT, temp_1_real_3D_NAME);
  }

  matrixContainer[MatrixId::temp_2_real_3D].Set(MatrixType::REAL, fullDims, NOLOAD, NOCHECKPOINT, temp_2_real_3D_NAME);
  matrixContainer[MatrixId::temp_3_real_3D].Set(MatrixType::REAL, fullDims, NOLOAD, NOCHECKPOINT, temp_3_real_3D_NAME);

  matrixContainer[MatrixId::cufft_x_temp].Set(MatrixType::CUFFT, reducedDims, NOLOAD, NOCHECKPOINT, cufft_X_temp_NAME);
  matrixContainer[MatrixId::cufft_y_temp].Set(MatrixType::CUFFT, reducedDims, NOLOAD, NOCHECKPOINT, cufft_Y_temp_NAME);
  matrixContainer[MatrixId::cufft_z_temp].Set(MatrixType::CUFFT, reducedDims, NOLOAD, NOCHECKPOINT, cufft_z_temp_NAME);
}// end of AddMatricesIntoContainer
//--------------------------------------------------------------------------------------------------



/**
 * Load all marked matrices from the input HDF5 file.
 * @param [in] inputFile - HDF5 input file handle
 */
void TMatrixContainer::LoadDataFromInputFile(THDF5_File& inputFile)
{
  for (const auto& it : matrixContainer)
  {
    if (it.second.loadData)
    {
      it.second.matrixPtr->ReadDataFromHDF5File(inputFile, it.second.matrixName);
    }
  }
}// end of LoadDataFromInputFile
//--------------------------------------------------------------------------------------------------

/**
 * Load selected matrices from the checkpoint HDF5 file.
 * @param [in] checkpointFile - HDF5 checkpoint file handle
 */
void TMatrixContainer::LoadDataFromCheckpointFile(THDF5_File& checkpointFile)
{
  for (const auto& it : matrixContainer)
  {
    if (it.second.checkpoint)
    {
      it.second.matrixPtr->ReadDataFromHDF5File(checkpointFile,it.second.matrixName);
    }
  }
}// end of LoadDataFromCheckpointFile
//--------------------------------------------------------------------------------------------------

/**
 * Store selected matrices into the checkpoint file.
 * @param [in] checkpointFile - Checkpoint file
 */
void TMatrixContainer::StoreDataIntoCheckpointFile(THDF5_File& checkpointFile)
{
  for (const auto& it : matrixContainer)
  {
    if (it.second.checkpoint)
    {
      // Copy data from device first
      it.second.matrixPtr->CopyFromDevice();
      // store data to the checkpoint file
      it.second.matrixPtr->WriteDataToHDF5File(checkpointFile,
                                               it.second.matrixName,
                                               TParameters::GetInstance().GetCompressionLevel());
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
    it.second.matrixPtr->CopyToDevice();
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
    it.second.matrixPtr->CopyFromDevice();
  }
}// end of CopyAllMatricesFromGPU
//--------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//
