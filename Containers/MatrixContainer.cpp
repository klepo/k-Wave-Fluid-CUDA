/**
 * @file        MatrixContainer.cpp
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the matrix container.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        02 December  2014, 16:17 (created) \n
 *              19 July      2016, 16:20 (revised)
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

//------------------------------------------------------------------------------------------------//
//------------------------------------------ CONSTANTS -------------------------------------------//
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
 * @throw errors cause an exception bad_alloc.
 */
void TMatrixContainer::CreateMatrices()
{
  for (auto& it : matrixContainer)
  {
    if (it.second.matrixPtr != nullptr)
    { // the data is already allocated
      CreateErrorAndThrowException(MatrixContainer_ERR_FMT_ReloactaionError,
                                   it.second.matrixName,
                                   __FILE__,  __LINE__);
    }

    switch (it.second.dataType)
    {
      case TMatrixRecord::mdtReal:
      {
        it.second.matrixPtr = new TRealMatrix(it.second.dimensionSizes);
        break;
      }

      case TMatrixRecord::mdtComplex:
      {
        it.second.matrixPtr = new TComplexMatrix(it.second.dimensionSizes);
        break;
      }

      case TMatrixRecord::mdtIndex:
      {
        it.second.matrixPtr = new TIndexMatrix(it.second.dimensionSizes);
        break;
      }

      case TMatrixRecord::mdtCUFFT:
      {
        it.second.matrixPtr = new TCUFFTComplexMatrix(it.second.dimensionSizes);
        break;
      }

      default: // unknown matrix type
      {
        CreateErrorAndThrowException(MatrixContainer_ERR_FMT_RecordUnknownDistributionType,
                                     it.second.matrixName,
                                     __FILE__, __LINE__);
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
  const TParameters& params = TParameters::GetInstance();

  TDimensionSizes fullDims    = params.GetFullDimensionSizes();
  TDimensionSizes reducedDims = params.GetReducedDimensionSizes();

  // this cannot be constexpr because of Visual studio 12.
  const bool LOAD         = true;
  const bool NOLOAD       = false;
  const bool CHECKPOINT   = true;
  const bool NOCHECKPOINT = false;

  //----------------------------------------- Allocate all matrices ------------------------------//

  matrixContainer[kappa] .Set(TMatrixRecord::mdtReal, reducedDims, NOLOAD, NOCHECKPOINT, kappa_r_Name);

  if (!params.Get_c0_scalar_flag())
  {
    matrixContainer[c2]  .Set(TMatrixRecord::mdtReal,    fullDims,   LOAD, NOCHECKPOINT, c0_Name);
  }

  matrixContainer[p]     .Set(TMatrixRecord::mdtReal,    fullDims, NOLOAD,   CHECKPOINT, p_Name);

  matrixContainer[rhox]  .Set(TMatrixRecord::mdtReal,    fullDims, NOLOAD,   CHECKPOINT, rhox_Name);
  matrixContainer[rhoy]  .Set(TMatrixRecord::mdtReal,    fullDims, NOLOAD,   CHECKPOINT, rhoy_Name);
  matrixContainer[rhoz]  .Set(TMatrixRecord::mdtReal,    fullDims, NOLOAD,   CHECKPOINT, rhoz_Name);

  matrixContainer[ux_sgx].Set(TMatrixRecord::mdtReal,    fullDims, NOLOAD,   CHECKPOINT, ux_sgx_Name);
  matrixContainer[uy_sgy].Set(TMatrixRecord::mdtReal,    fullDims, NOLOAD,   CHECKPOINT, uy_sgy_Name);
  matrixContainer[uz_sgz].Set(TMatrixRecord::mdtReal,    fullDims, NOLOAD,   CHECKPOINT, uz_sgz_Name);

  matrixContainer[duxdx] .Set(TMatrixRecord::mdtReal,    fullDims, NOLOAD, NOCHECKPOINT, duxdx_Name);
  matrixContainer[duydy] .Set(TMatrixRecord::mdtReal,    fullDims, NOLOAD, NOCHECKPOINT, duydy_Name);
  matrixContainer[duzdz] .Set(TMatrixRecord::mdtReal,    fullDims, NOLOAD, NOCHECKPOINT, duzdz_Name);

  if (!params.Get_rho0_scalar_flag())
  {
    matrixContainer[rho0]       .Set(TMatrixRecord::mdtReal, fullDims, LOAD, NOCHECKPOINT, rho0_Name);
    matrixContainer[dt_rho0_sgx].Set(TMatrixRecord::mdtReal, fullDims, LOAD, NOCHECKPOINT, rho0_sgx_Name);
    matrixContainer[dt_rho0_sgy].Set(TMatrixRecord::mdtReal, fullDims, LOAD, NOCHECKPOINT, rho0_sgy_Name);
    matrixContainer[dt_rho0_sgz].Set(TMatrixRecord::mdtReal, fullDims, LOAD, NOCHECKPOINT, rho0_sgz_Name);
  }

  matrixContainer[ddx_k_shift_pos].Set(TMatrixRecord::mdtComplex, TDimensionSizes(reducedDims.X, 1, 1), LOAD, NOCHECKPOINT, ddx_k_shift_pos_r_Name);
  matrixContainer[ddy_k_shift_pos].Set(TMatrixRecord::mdtComplex, TDimensionSizes(1, reducedDims.Y, 1), LOAD, NOCHECKPOINT, ddy_k_shift_pos_Name);
  matrixContainer[ddz_k_shift_pos].Set(TMatrixRecord::mdtComplex, TDimensionSizes(1, 1, reducedDims.Z), LOAD, NOCHECKPOINT, ddz_k_shift_pos_Name);

  matrixContainer[ddx_k_shift_neg].Set(TMatrixRecord::mdtComplex, TDimensionSizes(reducedDims.X ,1, 1), LOAD, NOCHECKPOINT, ddx_k_shift_neg_r_Name);
  matrixContainer[ddy_k_shift_neg].Set(TMatrixRecord::mdtComplex, TDimensionSizes(1, reducedDims.Y, 1), LOAD, NOCHECKPOINT, ddy_k_shift_neg_Name);
  matrixContainer[ddz_k_shift_neg].Set(TMatrixRecord::mdtComplex, TDimensionSizes(1, 1, reducedDims.Z), LOAD, NOCHECKPOINT, ddz_k_shift_neg_Name);

  matrixContainer[pml_x_sgx].Set(TMatrixRecord::mdtReal, TDimensionSizes(fullDims.X, 1, 1), LOAD, NOCHECKPOINT, pml_x_sgx_Name);
  matrixContainer[pml_y_sgy].Set(TMatrixRecord::mdtReal, TDimensionSizes(1, fullDims.Y, 1), LOAD, NOCHECKPOINT, pml_y_sgy_Name);
  matrixContainer[pml_z_sgz].Set(TMatrixRecord::mdtReal, TDimensionSizes(1, 1, fullDims.Z), LOAD, NOCHECKPOINT, pml_z_sgz_Name);

  matrixContainer[pml_x].Set(TMatrixRecord::mdtReal, TDimensionSizes(fullDims.X, 1, 1), LOAD, NOCHECKPOINT, pml_x_Name);
  matrixContainer[pml_y].Set(TMatrixRecord::mdtReal, TDimensionSizes(1, fullDims.Y, 1), LOAD, NOCHECKPOINT, pml_y_Name);
  matrixContainer[pml_z].Set(TMatrixRecord::mdtReal, TDimensionSizes(1, 1, fullDims.Z), LOAD, NOCHECKPOINT, pml_z_Name);

  if (params.Get_nonlinear_flag())
  {
    if (! params.Get_BonA_scalar_flag())
    {
      matrixContainer[BonA].Set(TMatrixRecord::mdtReal, fullDims, LOAD, NOCHECKPOINT, BonA_Name);
    }
  }

  if (params.Get_absorbing_flag() != 0)
  {
    if (!((params.Get_c0_scalar_flag()) && (params.Get_alpha_coeff_scalar_flag())))
    {
      matrixContainer[absorb_tau].Set(TMatrixRecord::mdtReal, fullDims, NOLOAD, NOCHECKPOINT, absorb_tau_Name);
      matrixContainer[absorb_eta].Set(TMatrixRecord::mdtReal, fullDims, NOLOAD, NOCHECKPOINT, absorb_eta_Name);
    }

    matrixContainer[absorb_nabla1].Set(TMatrixRecord::mdtReal, reducedDims, NOLOAD, NOCHECKPOINT, absorb_nabla1_r_Name);
    matrixContainer[absorb_nabla2].Set(TMatrixRecord::mdtReal, reducedDims, NOLOAD, NOCHECKPOINT, absorb_nabla2_r_Name);
  }

  // linear sensor mask
  if (params.Get_sensor_mask_type() == TParameters::smt_index)
  {
    matrixContainer[sensor_mask_index].Set(TMatrixRecord::mdtIndex,
                                           TDimensionSizes(params.Get_sensor_mask_index_size(), 1, 1),
                                           LOAD, NOCHECKPOINT, sensor_mask_index_Name);
  }

  // cuboid sensor mask
  if (params.Get_sensor_mask_type() == TParameters::smt_corners)
  {
    matrixContainer[sensor_mask_corners].Set(TMatrixRecord::mdtIndex,
                                             TDimensionSizes(6, params.Get_sensor_mask_corners_size(), 1),
                                             LOAD, NOCHECKPOINT, sensor_mask_corners_Name);
  }

  // if p0 source flag
  if (params.Get_p0_source_flag() == 1)
  {
    matrixContainer[p0_source_input].Set(TMatrixRecord::mdtReal, fullDims, LOAD, NOCHECKPOINT, p0_source_input_Name);
  }

  // u_source_index
  if ((params.Get_transducer_source_flag() != 0) ||
      (params.Get_ux_source_flag() != 0)         ||
      (params.Get_uy_source_flag() != 0)         ||
      (params.Get_uz_source_flag() != 0))
  {
    matrixContainer[u_source_index].Set(TMatrixRecord::mdtIndex,
                                        TDimensionSizes(1, 1, params.Get_u_source_index_size()),
                                        LOAD, NOCHECKPOINT, u_source_index_Name);
  }

  //transducer source flag defined
  if (params.Get_transducer_source_flag() != 0)
  {
    matrixContainer[delay_mask].Set(TMatrixRecord::mdtIndex,
                                    TDimensionSizes(1 ,1, params.Get_u_source_index_size()),
                                    LOAD, NOCHECKPOINT, delay_mask_Name);

    matrixContainer[transducer_source_input].Set(TMatrixRecord::mdtReal,
                                                 TDimensionSizes(1 ,1, params.Get_transducer_source_input_size()),
                                                 LOAD, NOCHECKPOINT, transducer_source_input_Name);
  }

  // p variables
  if (params.Get_p_source_flag() != 0)
  {
    if (params.Get_p_source_many() == 0)
    { // 1D case
      matrixContainer[p_source_input].Set(TMatrixRecord::mdtReal,
                                          TDimensionSizes(1 ,1, params.Get_p_source_flag()),
                                          LOAD, NOCHECKPOINT, p_source_input_Name);
    }
    else
    { // 2D case
      matrixContainer[p_source_input].Set(TMatrixRecord::mdtReal,
                                          TDimensionSizes(1,params.Get_p_source_index_size(),params.Get_p_source_flag()),
                                          LOAD, NOCHECKPOINT, p_source_input_Name);
    }

    matrixContainer[p_source_index].Set(TMatrixRecord::mdtIndex,
                                        TDimensionSizes(1, 1, params.Get_p_source_index_size()),
                                        LOAD, NOCHECKPOINT, p_source_index_Name);
  }

  //------------------------------------ uxyz source flags ---------------------------------------//
  if (params.Get_ux_source_flag() != 0)
  {
    if (params.Get_u_source_many() == 0)
    { // 1D
      matrixContainer[ux_source_input].Set(TMatrixRecord::mdtReal,
                                           TDimensionSizes(1, 1, params.Get_ux_source_flag()),
                                           LOAD, NOCHECKPOINT, ux_source_input_Name);
    }
    else
    { // 2D
      matrixContainer[ux_source_input].Set(TMatrixRecord::mdtReal,
                                           TDimensionSizes(1, params.Get_u_source_index_size(), params.Get_ux_source_flag()),
                                           LOAD, NOCHECKPOINT, ux_source_input_Name);
    }
  }// ux_source_input

  if (params.Get_uy_source_flag() != 0)
  {
    if (params.Get_u_source_many() == 0)
    { // 1D
      matrixContainer[uy_source_input].Set(TMatrixRecord::mdtReal,
                                           TDimensionSizes(1, 1, params.Get_uy_source_flag()),
                                           LOAD, NOCHECKPOINT, uy_source_input_Name);
    }
    else
    { // 2D
      matrixContainer[uy_source_input].Set(TMatrixRecord::mdtReal,
                                           TDimensionSizes(1,params.Get_u_source_index_size(),params.Get_uy_source_flag()),
                                           LOAD, NOCHECKPOINT, uy_source_input_Name);
    }
  }// uy_source_input

  if (params.Get_uz_source_flag() != 0)
  {
    if (params.Get_u_source_many() == 0)
    { // 1D
      matrixContainer[uz_source_input].Set(TMatrixRecord::mdtReal,
                                           TDimensionSizes(1, 1, params.Get_uz_source_flag()),
                                           LOAD, NOCHECKPOINT, uz_source_input_Name);
    }
    else
    { // 2D
      matrixContainer[uz_source_input].Set(TMatrixRecord::mdtReal,
                                           TDimensionSizes(1, params.Get_u_source_index_size(), params.Get_uz_source_flag()),
                                           LOAD, NOCHECKPOINT, uz_source_input_Name);
    }
  }// uz_source_input

  //-- Nonlinear grid
  if (params.Get_nonuniform_grid_flag()!= 0)
  {
    matrixContainer[dxudxn].Set(TMatrixRecord::mdtReal, TDimensionSizes(fullDims.X, 1, 1), LOAD, NOCHECKPOINT, dxudxn_Name);
    matrixContainer[dyudyn].Set(TMatrixRecord::mdtReal, TDimensionSizes(1, fullDims.Y, 1), LOAD, NOCHECKPOINT, dyudyn_Name);
    matrixContainer[dzudzn].Set(TMatrixRecord::mdtReal, TDimensionSizes(1 ,1, fullDims.Z), LOAD, NOCHECKPOINT, dzudzn_Name);

    matrixContainer[dxudxn_sgx].Set(TMatrixRecord::mdtReal, TDimensionSizes(fullDims.X, 1, 1), LOAD, NOCHECKPOINT, dxudxn_sgx_Name);
    matrixContainer[dyudyn_sgy].Set(TMatrixRecord::mdtReal, TDimensionSizes(1, fullDims.Y, 1), LOAD, NOCHECKPOINT, dyudyn_sgy_Name);
    matrixContainer[dzudzn_sgz].Set(TMatrixRecord::mdtReal, TDimensionSizes(1 ,1, fullDims.Z), LOAD, NOCHECKPOINT, dzudzn_sgz_Name);
  }

  //-- u_non_staggered_raw
  if (params.IsStore_u_non_staggered_raw())
  {
    TDimensionSizes shiftDims = fullDims;

    const size_t Nx_2 = fullDims.X / 2 + 1;
    const size_t Ny_2 = fullDims.Y / 2 + 1;
    const size_t Nz_2 = fullDims.Z / 2 + 1;

    size_t xCutSize = Nx_2       * fullDims.Y * fullDims.Z;
    size_t yCutSize = fullDims.X * Ny_2       * fullDims.Z;
    size_t zCutSize = fullDims.X * fullDims.Y * Nz_2;

    if ((xCutSize >= yCutSize) && (xCutSize >= zCutSize))
    {
      // X cut is the biggest
      shiftDims.X = Nx_2;
    }
    else if ((yCutSize >= xCutSize) && (yCutSize >= zCutSize))
    {
      // Y cut is the biggest
      shiftDims.Y = Ny_2;
    }
    else if ((zCutSize >= xCutSize) && (zCutSize >= yCutSize))
    {
      // Z cut is the biggest
      shiftDims.Z = Nz_2;
    }
    else
    {
      //all are the same
      shiftDims.X = Nx_2;
    }

    matrixContainer[cufft_shift_temp].Set(TMatrixRecord::mdtCUFFT, shiftDims, NOLOAD, NOCHECKPOINT, CUFFT_shift_temp_Name);


    // these three are necessary only for u_non_staggered calculation now
    matrixContainer[ux_shifted].Set(TMatrixRecord::mdtReal, fullDims, NOLOAD, NOCHECKPOINT, ux_shifted_Name);
    matrixContainer[uy_shifted].Set(TMatrixRecord::mdtReal, fullDims, NOLOAD, NOCHECKPOINT, uy_shifted_Name);
    matrixContainer[uz_shifted].Set(TMatrixRecord::mdtReal, fullDims, NOLOAD, NOCHECKPOINT, uz_shifted_Name);

    // shifts from the input file
    matrixContainer[x_shift_neg_r].Set(TMatrixRecord::mdtComplex, TDimensionSizes(Nx_2, 1, 1), LOAD,NOCHECKPOINT, x_shift_neg_r_Name);
    matrixContainer[y_shift_neg_r].Set(TMatrixRecord::mdtComplex, TDimensionSizes(1, Ny_2, 1), LOAD,NOCHECKPOINT, y_shift_neg_r_Name);
    matrixContainer[z_shift_neg_r].Set(TMatrixRecord::mdtComplex, TDimensionSizes(1, 1, Nz_2), LOAD,NOCHECKPOINT, z_shift_neg_r_Name);
  }// u_non_staggered


  //------------------------------------- Temporary matrices -------------------------------------//
  // this matrix used to load alpha_coeff for absorb_tau pre-calculation

  if ((params.Get_absorbing_flag() != 0) && (!params.Get_alpha_coeff_scalar_flag()))
  {
    matrixContainer[temp_1_real_3D].Set(TMatrixRecord::mdtReal, fullDims, LOAD, NOCHECKPOINT, alpha_coeff_Name);
  }
  else
  {
    matrixContainer[temp_1_real_3D].Set(TMatrixRecord::mdtReal, fullDims, NOLOAD, NOCHECKPOINT, Temp_1_RS3D_Name);
  }

  matrixContainer[temp_2_real_3D].Set(TMatrixRecord::mdtReal, fullDims, NOLOAD, NOCHECKPOINT, Temp_2_RS3D_Name);
  matrixContainer[temp_3_real_3D].Set(TMatrixRecord::mdtReal, fullDims, NOLOAD, NOCHECKPOINT, Temp_3_RS3D_Name);

  matrixContainer[cufft_x_temp].Set(TMatrixRecord::mdtCUFFT, reducedDims, NOLOAD, NOCHECKPOINT, CUFFT_X_temp_Name);
  matrixContainer[cufft_y_temp].Set(TMatrixRecord::mdtCUFFT, reducedDims, NOLOAD, NOCHECKPOINT, CUFFT_Y_temp_Name);
  matrixContainer[cufft_z_temp].Set(TMatrixRecord::mdtCUFFT, reducedDims, NOLOAD, NOCHECKPOINT, CUFFT_Z_temp_Name);
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
      it.second.matrixPtr->ReadDataFromHDF5File(inputFile, it.second.matrixName.c_str());
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
      it.second.matrixPtr->ReadDataFromHDF5File(checkpointFile,it.second.matrixName.c_str());
    }
  }
}// end of LoadDataFromCheckpointFile
//--------------------------------------------------------------------------------------------------

/**
 * Store selected matrices into the checkpoint file.
 * @param [in] checkpointFile
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
                                               it.second.matrixName.c_str(),
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

/*
 * Print error and and throw an exception.
 * @throw invalid_argument
 *
 * @param [in] messageFormat - format of error
 * @param [in] matrixName    - HDF5 dataset name
 * @param [in] file          - file of error
 * @param [in] line          - line of error
 */
void TMatrixContainer::CreateErrorAndThrowException(const char*   messageFormat,
                                                    const string& matrixName,
                                                    const char*   file,
                                                    const int     line)
{
  char errorMessage[256];
  snprintf(errorMessage, 256, messageFormat, matrixName.c_str(), file, line);
  throw std::invalid_argument(errorMessage);
}// CreateErrorAndThrowException
//--------------------------------------------------------------------------------------------------
