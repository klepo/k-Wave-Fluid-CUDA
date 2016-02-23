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
 *              09 February  2015, 20:22 (revised)
 *
 * @section License
 * This file is part of the C++ extension of the k-Wave Toolbox
 * (http://www.k-wave.org).\n Copyright (C) 2014 Jiri Jaros, Beau Johnston
 * and Bradley Treeby
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
 * along with k-Wave. If not, see http://www.gnu.org/licenses/.
 */

#include <stdexcept>

#include <Containers/MatrixContainer.h>

#include <Parameters/Parameters.h>
#include <Utils/ErrorMessages.h>

//----------------------------------------------------------------------------//
//----------------------------- CONSTANTS ------------------------------------//
//----------------------------------------------------------------------------//


//----------------------------------------------------------------------------//
//--------------------------- Public methods ---------------------------------//
//----------------------------------------------------------------------------//



/*
 * Destructor of TMatrixContainer.
 */
TMatrixContainer::~TMatrixContainer()
{
  MatrixContainer.clear();
}// end of ~TMatrixContainer
//----------------------------------------------------------------------------

/*
 * Create all matrix objects in the container.
 * @throw errors cause an exception bad_alloc.
 */
void TMatrixContainer::CreateAllObjects()
{
  for (auto it = MatrixContainer.begin(); it != MatrixContainer.end(); it++)
  {
    if (it->second.MatrixPtr != NULL)
    { // the data is already allocated
      PrintErrorAndThrowException(MatrixContainer_ERR_FMT_ReloactaionError,
                                  it->second.HDF5MatrixName,
                                  __FILE__,  __LINE__);
    }

    switch (it->second.MatrixDataType)
    {
      case TMatrixRecord::mdtReal:
      {
        it->second.MatrixPtr = new TRealMatrix(it->second.DimensionSizes);
        break;
      }

      case TMatrixRecord::mdtComplex:
      {
        it->second.MatrixPtr = new TComplexMatrix(it->second.DimensionSizes);
        break;
      }

      case TMatrixRecord::mdtIndex:
      {
        it->second.MatrixPtr = new TIndexMatrix(it->second.DimensionSizes);
        break;
      }

      case TMatrixRecord::mdtCUFFT:
      {
        it->second.MatrixPtr = new TCUFFTComplexMatrix(it->second.DimensionSizes);
        break;
      }

      default: // unknown matrix type
      {
        PrintErrorAndThrowException(MatrixContainer_ERR_FMT_RecordUnknownDistributionType,
                                    it->second.HDF5MatrixName,
                                    __FILE__, __LINE__);
        break;
      }
    }// switch
  }// end for
}// end of CreateAllObjects
//------------------------------------------------------------------------------

/*
 * This function defines common matrices in K-Wave.
 * All matrices records are created here.
 */
void TMatrixContainer::AddMatricesIntoContainer()
{

  TParameters * Params = TParameters::GetInstance();

  TDimensionSizes FullDims = Params->GetFullDimensionSizes();
  TDimensionSizes ReducedDims = Params->GetReducedDimensionSizes();

  const bool LOAD         = true;
  const bool NOLOAD       = false;
  const bool CHECKPOINT   = true;
  const bool NOCHECKPOINT = false;

    //----------------------Allocate all matrices ----------------------------//

  MatrixContainer[kappa] .SetAllValues(NULL, TMatrixRecord::mdtReal, ReducedDims, NOLOAD, NOCHECKPOINT, kappa_r_Name);

  if (!Params->Get_c0_scalar_flag())
  {
    MatrixContainer[c2]  .SetAllValues(NULL, TMatrixRecord::mdtReal,    FullDims,   LOAD, NOCHECKPOINT, c0_Name);
  }

  MatrixContainer[p]     .SetAllValues(NULL, TMatrixRecord::mdtReal,    FullDims, NOLOAD,   CHECKPOINT, p_Name);

  MatrixContainer[rhox]  .SetAllValues(NULL, TMatrixRecord::mdtReal,    FullDims, NOLOAD,   CHECKPOINT, rhox_Name);
  MatrixContainer[rhoy]  .SetAllValues(NULL, TMatrixRecord::mdtReal,    FullDims, NOLOAD,   CHECKPOINT, rhoy_Name);
  MatrixContainer[rhoz]  .SetAllValues(NULL, TMatrixRecord::mdtReal,    FullDims, NOLOAD,   CHECKPOINT, rhoz_Name);

  MatrixContainer[ux_sgx].SetAllValues(NULL, TMatrixRecord::mdtReal,    FullDims, NOLOAD,   CHECKPOINT, ux_sgx_Name);
  MatrixContainer[uy_sgy].SetAllValues(NULL, TMatrixRecord::mdtReal,    FullDims, NOLOAD,   CHECKPOINT, uy_sgy_Name);
  MatrixContainer[uz_sgz].SetAllValues(NULL, TMatrixRecord::mdtReal,    FullDims, NOLOAD,   CHECKPOINT, uz_sgz_Name);

  MatrixContainer[duxdx] .SetAllValues(NULL, TMatrixRecord::mdtReal,    FullDims, NOLOAD, NOCHECKPOINT, duxdx_Name);
  MatrixContainer[duydy] .SetAllValues(NULL, TMatrixRecord::mdtReal,    FullDims, NOLOAD, NOCHECKPOINT, duydy_Name);
  MatrixContainer[duzdz] .SetAllValues(NULL ,TMatrixRecord::mdtReal,    FullDims, NOLOAD, NOCHECKPOINT, duzdz_Name);

  if (!Params->Get_rho0_scalar_flag())
  {
    MatrixContainer[rho0]       .SetAllValues(NULL, TMatrixRecord::mdtReal, FullDims, LOAD, NOCHECKPOINT, rho0_Name);
    MatrixContainer[dt_rho0_sgx].SetAllValues(NULL, TMatrixRecord::mdtReal, FullDims, LOAD, NOCHECKPOINT, rho0_sgx_Name);
    MatrixContainer[dt_rho0_sgy].SetAllValues(NULL, TMatrixRecord::mdtReal, FullDims, LOAD, NOCHECKPOINT, rho0_sgy_Name);
    MatrixContainer[dt_rho0_sgz].SetAllValues(NULL, TMatrixRecord::mdtReal, FullDims, LOAD, NOCHECKPOINT, rho0_sgz_Name);
  }

  MatrixContainer[ddx_k_shift_pos].SetAllValues(NULL, TMatrixRecord::mdtComplex, TDimensionSizes(ReducedDims.X, 1, 1), LOAD, NOCHECKPOINT, ddx_k_shift_pos_r_Name);
  MatrixContainer[ddy_k_shift_pos].SetAllValues(NULL, TMatrixRecord::mdtComplex, TDimensionSizes(1, ReducedDims.Y, 1), LOAD, NOCHECKPOINT, ddy_k_shift_pos_Name);
  MatrixContainer[ddz_k_shift_pos].SetAllValues(NULL, TMatrixRecord::mdtComplex, TDimensionSizes(1, 1, ReducedDims.Z), LOAD, NOCHECKPOINT, ddz_k_shift_pos_Name);

  MatrixContainer[ddx_k_shift_neg].SetAllValues(NULL, TMatrixRecord::mdtComplex, TDimensionSizes(ReducedDims.X ,1, 1), LOAD, NOCHECKPOINT, ddx_k_shift_neg_r_Name);
  MatrixContainer[ddy_k_shift_neg].SetAllValues(NULL, TMatrixRecord::mdtComplex, TDimensionSizes(1, ReducedDims.Y, 1), LOAD, NOCHECKPOINT, ddy_k_shift_neg_Name);
  MatrixContainer[ddz_k_shift_neg].SetAllValues(NULL, TMatrixRecord::mdtComplex, TDimensionSizes(1, 1, ReducedDims.Z), LOAD, NOCHECKPOINT, ddz_k_shift_neg_Name);

  MatrixContainer[pml_x_sgx].SetAllValues(NULL, TMatrixRecord::mdtReal, TDimensionSizes(FullDims.X, 1, 1), LOAD, NOCHECKPOINT, pml_x_sgx_Name);
  MatrixContainer[pml_y_sgy].SetAllValues(NULL, TMatrixRecord::mdtReal, TDimensionSizes(1, FullDims.Y, 1), LOAD, NOCHECKPOINT, pml_y_sgy_Name);
  MatrixContainer[pml_z_sgz].SetAllValues(NULL, TMatrixRecord::mdtReal, TDimensionSizes(1, 1, FullDims.Z), LOAD, NOCHECKPOINT, pml_z_sgz_Name);

  MatrixContainer[pml_x].SetAllValues(NULL, TMatrixRecord::mdtReal, TDimensionSizes(FullDims.X, 1, 1), LOAD, NOCHECKPOINT, pml_x_Name);
  MatrixContainer[pml_y].SetAllValues(NULL, TMatrixRecord::mdtReal, TDimensionSizes(1, FullDims.Y, 1), LOAD, NOCHECKPOINT, pml_y_Name);
  MatrixContainer[pml_z].SetAllValues(NULL, TMatrixRecord::mdtReal, TDimensionSizes(1, 1, FullDims.Z), LOAD, NOCHECKPOINT, pml_z_Name);

  if (Params->Get_nonlinear_flag())
  {
    if (! Params->Get_BonA_scalar_flag())
    {
      MatrixContainer[BonA].SetAllValues(NULL, TMatrixRecord::mdtReal, FullDims, LOAD, NOCHECKPOINT, BonA_Name);
    }
  }

  if (Params->Get_absorbing_flag() != 0)
  {
    if (!((Params->Get_c0_scalar_flag()) && (Params->Get_alpha_coeff_scalar_flag())))
    {
      MatrixContainer[absorb_tau].SetAllValues(NULL, TMatrixRecord::mdtReal, FullDims, NOLOAD, NOCHECKPOINT, absorb_tau_Name);
      MatrixContainer[absorb_eta].SetAllValues(NULL, TMatrixRecord::mdtReal, FullDims, NOLOAD, NOCHECKPOINT, absorb_eta_Name);
    }

    MatrixContainer[absorb_nabla1].SetAllValues(NULL, TMatrixRecord::mdtReal, ReducedDims, NOLOAD, NOCHECKPOINT, absorb_nabla1_r_Name);
    MatrixContainer[absorb_nabla2].SetAllValues(NULL, TMatrixRecord::mdtReal, ReducedDims, NOLOAD, NOCHECKPOINT, absorb_nabla2_r_Name);
  }

  // linear sensor mask
  if (Params->Get_sensor_mask_type() == TParameters::smt_index)
  {
    MatrixContainer[sensor_mask_index].SetAllValues(NULL, TMatrixRecord::mdtIndex,
                                                    TDimensionSizes(Params->Get_sensor_mask_index_size(), 1, 1),
                                                    LOAD, NOCHECKPOINT, sensor_mask_index_Name);
  }

  // cuboid sensor mask
  if (Params->Get_sensor_mask_type() == TParameters::smt_corners)
  {
    MatrixContainer[sensor_mask_corners].SetAllValues(NULL, TMatrixRecord::mdtIndex,
                                                      TDimensionSizes(6, Params->Get_sensor_mask_corners_size(), 1),
                                                      LOAD, NOCHECKPOINT, sensor_mask_corners_Name);
  }

  // if p0 source flag
  if (Params->Get_p0_source_flag() == 1)
  {
    MatrixContainer[p0_source_input].SetAllValues(NULL, TMatrixRecord::mdtReal, FullDims, LOAD, NOCHECKPOINT, p0_source_input_Name);
  }

  // u_source_index
  if ((Params->Get_transducer_source_flag() != 0) ||
      (Params->Get_ux_source_flag() != 0)         ||
      (Params->Get_uy_source_flag() != 0)         ||
      (Params->Get_uz_source_flag() != 0))
  {
    MatrixContainer[u_source_index].SetAllValues(NULL, TMatrixRecord::mdtIndex,
                                                 TDimensionSizes(1, 1, Params->Get_u_source_index_size()),
                                                 LOAD, NOCHECKPOINT, u_source_index_Name);
  }

  //transducer source flag defined
  if (Params->Get_transducer_source_flag() != 0)
  {
    MatrixContainer[delay_mask].SetAllValues(NULL, TMatrixRecord::mdtIndex,
                                             TDimensionSizes(1 ,1, Params->Get_u_source_index_size()),
                                             LOAD, NOCHECKPOINT, delay_mask_Name);

    MatrixContainer[transducer_source_input].SetAllValues(NULL, TMatrixRecord::mdtReal,
                                                          TDimensionSizes(1 ,1, Params->Get_transducer_source_input_size()),
                                                          LOAD, NOCHECKPOINT, transducer_source_input_Name);
  }

  // p variables
  if (Params->Get_p_source_flag() != 0)
  {
    if (Params->Get_p_source_many() == 0)
    { // 1D case
      MatrixContainer[p_source_input].SetAllValues(NULL, TMatrixRecord::mdtReal,
                                                   TDimensionSizes(1 ,1, Params->Get_p_source_flag()),
                                                   LOAD, NOCHECKPOINT, p_source_input_Name);
    }
    else
    { // 2D case
      MatrixContainer[p_source_input].SetAllValues(NULL, TMatrixRecord::mdtReal,
                                                   TDimensionSizes(1,
                                                                   Params->Get_p_source_index_size(),
                                                                   Params->Get_p_source_flag()),
                                                   LOAD, NOCHECKPOINT, p_source_input_Name);
    }

    MatrixContainer[p_source_index].SetAllValues(NULL, TMatrixRecord::mdtIndex,
                                                 TDimensionSizes(1, 1, Params->Get_p_source_index_size()),
                                                 LOAD, NOCHECKPOINT, p_source_index_Name);
  }

  //----------------------------uxyz source flags---------------------------//
  if (Params->Get_ux_source_flag() != 0)
  {
    if (Params->Get_u_source_many() == 0)
    { // 1D
      MatrixContainer[ux_source_input].SetAllValues(NULL, TMatrixRecord::mdtReal,
                                                    TDimensionSizes(1, 1, Params->Get_ux_source_flag()),
                                                    LOAD, NOCHECKPOINT, ux_source_input_Name);
    }
    else
    { // 2D
      MatrixContainer[ux_source_input].SetAllValues(NULL, TMatrixRecord::mdtReal,
                                                    TDimensionSizes(1,
                                                                    Params->Get_u_source_index_size(),
                                                                    Params->Get_ux_source_flag()),
                                                    LOAD, NOCHECKPOINT, ux_source_input_Name);
    }
  }// ux_source_input

  if (Params->Get_uy_source_flag() != 0)
  {
    if (Params->Get_u_source_many() == 0)
    { // 1D
      MatrixContainer[uy_source_input].SetAllValues(NULL, TMatrixRecord::mdtReal,
                                                    TDimensionSizes(1, 1, Params->Get_uy_source_flag()),
                                                    LOAD, NOCHECKPOINT, uy_source_input_Name);
    }
    else
    { // 2D
      MatrixContainer[uy_source_input].SetAllValues(NULL, TMatrixRecord::mdtReal,
                                                    TDimensionSizes(1,
                                                                    Params->Get_u_source_index_size(),
                                                                    Params->Get_uy_source_flag()),
                                                    LOAD, NOCHECKPOINT, uy_source_input_Name);
    }
  }// uy_source_input

  if (Params->Get_uz_source_flag() != 0)
  {
    if (Params->Get_u_source_many() == 0)
    { // 1D
      MatrixContainer[uz_source_input].SetAllValues(NULL, TMatrixRecord::mdtReal,
                                                    TDimensionSizes(1, 1, Params->Get_uz_source_flag()),
                                                    LOAD, NOCHECKPOINT, uz_source_input_Name);
    }
    else
    { // 2D
      MatrixContainer[uz_source_input].SetAllValues(NULL, TMatrixRecord::mdtReal,
                                                    TDimensionSizes(1,
                                                                    Params->Get_u_source_index_size(),
                                                                    Params->Get_uz_source_flag()),
                                                    LOAD, NOCHECKPOINT, uz_source_input_Name);
    }
  }// uz_source_input

  //-- Nonlinear grid
  if (Params->Get_nonuniform_grid_flag()!= 0)
  {
    MatrixContainer[dxudxn].SetAllValues(NULL, TMatrixRecord::mdtReal, TDimensionSizes(FullDims.X, 1, 1), LOAD, NOCHECKPOINT, dxudxn_Name);
    MatrixContainer[dyudyn].SetAllValues(NULL, TMatrixRecord::mdtReal, TDimensionSizes(1, FullDims.Y, 1), LOAD, NOCHECKPOINT, dyudyn_Name);
    MatrixContainer[dzudzn].SetAllValues(NULL, TMatrixRecord::mdtReal, TDimensionSizes(1 ,1, FullDims.Z), LOAD, NOCHECKPOINT, dzudzn_Name);

    MatrixContainer[dxudxn_sgx].SetAllValues(NULL, TMatrixRecord::mdtReal, TDimensionSizes(FullDims.X, 1, 1), LOAD, NOCHECKPOINT, dxudxn_sgx_Name);
    MatrixContainer[dyudyn_sgy].SetAllValues(NULL, TMatrixRecord::mdtReal, TDimensionSizes(1, FullDims.Y, 1), LOAD, NOCHECKPOINT, dyudyn_sgy_Name);
    MatrixContainer[dzudzn_sgz].SetAllValues(NULL, TMatrixRecord::mdtReal, TDimensionSizes(1 ,1, FullDims.Z), LOAD, NOCHECKPOINT, dzudzn_sgz_Name);
  }

    //-- u_non_staggered_raw
  if (Params->IsStore_u_non_staggered_raw())
  {
    TDimensionSizes ShiftDims = FullDims;

    size_t X_2 = FullDims.X / 2 + 1;
    size_t Y_2 = FullDims.Y / 2 + 1;
    size_t Z_2 = FullDims.Z / 2 + 1;

    size_t XCutSize = X_2        * FullDims.Y * FullDims.Z;
    size_t YCutSize = FullDims.X * Y_2        * FullDims.Z;
    size_t ZCutSize = FullDims.X * FullDims.Y * Z_2;

    if ((XCutSize >= YCutSize) && (XCutSize >= ZCutSize))
    {
      // X cut is the biggest
      ShiftDims.X = X_2;
    }
    else if ((YCutSize >= XCutSize) && (YCutSize >= ZCutSize))
    {
      // Y cut is the biggest
      ShiftDims.Y = Y_2;
    }
    else if ((ZCutSize >= XCutSize) && (ZCutSize >= YCutSize))
    {
      // Z cut is the biggest
      ShiftDims.Z = Z_2;
    }
    else
    {
      //all are the same
      ShiftDims.X = X_2;
    }

    MatrixContainer[CUFFT_shift_temp].SetAllValues(NULL, TMatrixRecord::mdtCUFFT, ShiftDims, NOLOAD, NOCHECKPOINT, CUFFT_shift_temp_Name);


    // these three are necessary only for u_non_staggered calculation now
    MatrixContainer[ux_shifted].SetAllValues(NULL, TMatrixRecord::mdtReal, FullDims, NOLOAD, NOCHECKPOINT, ux_shifted_Name);
    MatrixContainer[uy_shifted].SetAllValues(NULL, TMatrixRecord::mdtReal, FullDims, NOLOAD, NOCHECKPOINT, uy_shifted_Name);
    MatrixContainer[uz_shifted].SetAllValues(NULL, TMatrixRecord::mdtReal, FullDims, NOLOAD, NOCHECKPOINT, uz_shifted_Name);

    // shifts from the input file
    MatrixContainer[x_shift_neg_r].SetAllValues(NULL, TMatrixRecord::mdtComplex, TDimensionSizes(X_2, 1, 1), LOAD,NOCHECKPOINT, x_shift_neg_r_Name);
    MatrixContainer[y_shift_neg_r].SetAllValues(NULL, TMatrixRecord::mdtComplex, TDimensionSizes(1, Y_2, 1), LOAD,NOCHECKPOINT, y_shift_neg_r_Name);
    MatrixContainer[z_shift_neg_r].SetAllValues(NULL, TMatrixRecord::mdtComplex, TDimensionSizes(1, 1, Z_2), LOAD,NOCHECKPOINT, z_shift_neg_r_Name);
  }// u_non_staggered

  //--------------------------------------------------------------------------//
  //----------------------- Temporary matrices -------------------------------//
  //--------------------------------------------------------------------------//
  // this matrix used to load alpha_coeff for absorb_tau pre-calculation

  if ((Params->Get_absorbing_flag() != 0) && (!Params->Get_alpha_coeff_scalar_flag()))
  {
    MatrixContainer[Temp_1_RS3D].SetAllValues(NULL, TMatrixRecord::mdtReal, FullDims, LOAD, NOCHECKPOINT, alpha_coeff_Name);
  }
  else
  {
    MatrixContainer[Temp_1_RS3D].SetAllValues(NULL, TMatrixRecord::mdtReal, FullDims, NOLOAD, NOCHECKPOINT, Temp_1_RS3D_Name);
  }

  MatrixContainer[Temp_2_RS3D].SetAllValues(NULL, TMatrixRecord::mdtReal, FullDims, NOLOAD, NOCHECKPOINT, Temp_2_RS3D_Name);
  MatrixContainer[Temp_3_RS3D].SetAllValues(NULL, TMatrixRecord::mdtReal, FullDims, NOLOAD, NOCHECKPOINT, Temp_3_RS3D_Name);

  MatrixContainer[CUFFT_X_temp].SetAllValues(NULL, TMatrixRecord::mdtCUFFT, ReducedDims, NOLOAD, NOCHECKPOINT, CUFFT_X_temp_Name);
  MatrixContainer[CUFFT_Y_temp].SetAllValues(NULL, TMatrixRecord::mdtCUFFT, ReducedDims, NOLOAD, NOCHECKPOINT, CUFFT_Y_temp_Name);
  MatrixContainer[CUFFT_Z_temp].SetAllValues(NULL, TMatrixRecord::mdtCUFFT, ReducedDims, NOLOAD, NOCHECKPOINT, CUFFT_Z_temp_Name);
}// end of AddMatricesIntoContainer
//------------------------------------------------------------------------------



/**
 * Load all marked matrices from the HDF5 file.
 * @param [in] HDF5_File - HDF5 file handle
 */
void TMatrixContainer::LoadDataFromInputHDF5File(THDF5_File & HDF5_File)
{
  for (auto it = MatrixContainer.begin(); it != MatrixContainer.end(); it++)
  {
    if (it->second.LoadData)
    {
      it->second.MatrixPtr->ReadDataFromHDF5File(HDF5_File,
                                                 it->second.HDF5MatrixName.c_str());
    }
  }
}// end of LoadDataFromInputHDF5File
//------------------------------------------------------------------------------

/**
 * Load selected matrices from checkpoint HDF5 file.
 * @param [in] HDF5_File - HDF5 file handle
 */
void TMatrixContainer::LoadDataFromCheckpointHDF5File(THDF5_File & HDF5_File)
{
  for (auto it = MatrixContainer.begin(); it != MatrixContainer.end(); it++)
  {
    if (it->second.Checkpoint)
    {
      it->second.MatrixPtr->ReadDataFromHDF5File(HDF5_File,
                                                 it->second.HDF5MatrixName.c_str());
    }
  }
}// end of LoadDataFromCheckpointHDF5File
//------------------------------------------------------------------------------

/**
 * Store selected matrices into the checkpoint file.
 * @param [in] HDF5_File
 */
void TMatrixContainer::StoreDataIntoCheckpointHDF5File(THDF5_File & HDF5_File)
{
  for (auto it = MatrixContainer.begin(); it != MatrixContainer.end(); it++)
  {
    if (it->second.Checkpoint)
    {
      // Copy data from device first
      it->second.MatrixPtr->CopyFromDevice();
      // store data to the checkpoint file
      it->second.MatrixPtr->WriteDataToHDF5File(HDF5_File,
                                                it->second.HDF5MatrixName.c_str(),
                                                TParameters::GetInstance()->GetCompressionLevel());
    }
  }
}// end of StoreDataIntoCheckpointHDF5File
//------------------------------------------------------------------------------

/*
 * Free all matrix objects.
 */
void TMatrixContainer::FreeAllMatrices()
{
  for (auto it = MatrixContainer.begin(); it != MatrixContainer.end(); it++)
  {
    if (it->second.MatrixPtr)
    {
      delete it->second.MatrixPtr;
      it->second.MatrixPtr = NULL;
    }
  }
}// end of FreeAllMatrices
//------------------------------------------------------------------------------

/**
 * Copy all matrices over to the GPU.
 */
void TMatrixContainer::CopyAllMatricesToDevice()
{
  for (auto it = MatrixContainer.begin(); it != MatrixContainer.end(); it++)
  {
    it->second.MatrixPtr->CopyToDevice();
  }
}//end of CopyAllMatricesToGPU
//------------------------------------------------------------------------------

/**
 * Copy all matrices back over to CPU
 * @todo Why do I need to copy all matrices back???
 */
void TMatrixContainer::CopyAllMatricesFromDevice()
{
  for (auto it = MatrixContainer.begin(); it != MatrixContainer.end(); it++)
  {
    it->second.MatrixPtr->CopyFromDevice();
  }
}// end of CopyAllMatricesFromGPU
//------------------------------------------------------------------------------

/*
 *  Prior to allocating any memory we can approximate the amount of memory
 *  required, this is useful to determine if the target device memory is large
 *  enough for running the simulation.
 *  Returns the number of megabytes that will be in use by all matrices.
 *
 * @todo - I have no idea how this was calculated - it is very suspicious!!
 */
size_t TMatrixContainer::GetSpeculatedMemoryFootprintInMegabytes()
{
  TParameters * Params = TParameters::GetInstance();

  size_t total_float_elements = 0;
  size_t total_long_elements = 0;
  size_t intermediate_element_count;

  for (auto it = MatrixContainer.begin(); it != MatrixContainer.end(); it++)
  {

  // if we expect the matrix to exist (by anticipating a non-zero
  // dimension size) use that value for the memory estimate
  intermediate_element_count = (it->second.DimensionSizes.X *
                                it->second.DimensionSizes.Y *
                                it->second.DimensionSizes.Z);

  if (intermediate_element_count != 0)
  {
    // if this data type is based off a long
    if (it->second.MatrixDataType == TMatrixRecord::mdtIndex)
      total_long_elements += intermediate_element_count;
    else
      total_float_elements += intermediate_element_count;
  }
}

//convert from bytes to megabytes and return
//also fix to a linear regression model to improve accuracy of estimation

//linear model from all data on 2 platforms
float b_0 = 168.08536;
float b_1 = 1.22851;

//linear model from trinity
//float b_0 = 78.194739;
//float b_1 = 1.233752;

//linear model from infib2
//float b_0 = 86.36;
//float b_1 = 1.146;

return b_0 + b_1 *
        ((((total_float_elements * sizeof (float)))+
          ((total_long_elements * sizeof (long)))) >> 20);
}// end of GetSpeculatedMemoryFootprintInMegabytes
//------------------------------------------------------------------------------

//--------------------------------------------------------------------------//
//----------------------- Protected methods --------------------------------//
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
//------------------------ Private methods ---------------------------------//
//--------------------------------------------------------------------------//

/*
 * Print error and and throw an exception.
 * @throw bad_alloc
 *
 * @param [in] FMT - format of error
 * @param [in] HDF5MatrixName - HDF5 dataset name
 * @param [in] File  File of error
 * @param [in] Line  Line of error
 */
void TMatrixContainer::PrintErrorAndThrowException(const char* FMT,
                                                   const string HDF5MatrixName,
                                                   const char* File,
                                                   const int Line)
{
  fprintf(stderr,FMT, HDF5MatrixName.c_str(), File, Line);
  throw bad_alloc();
}// end of PrintErrorAndAbort
//------------------------------------------------------------------------------
