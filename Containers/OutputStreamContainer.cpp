 /**
 * @file        OutputStreamContainer.cpp
 * @author      Jiri Jaros & Beau Johnston \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file for the output stream container.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        04 December  2014, 11:41 (created) \n
 *              19 July      2016, 17:17 (revised)
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
 * If not, see http://www.gnu.org/licenses/. If not, see http://www.gnu.org/licenses/.
 */

#include <Parameters/Parameters.h>
#include <Containers/OutputStreamContainer.h>

#include <OutputHDF5Streams/BaseOutputHDF5Stream.h>
#include <OutputHDF5Streams/IndexOutputHDF5Stream.h>
#include <OutputHDF5Streams/CuboidOutputHDF5Stream.h>
#include <OutputHDF5Streams/WholeDomainOutputHDF5Stream.h>


//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Destructor.
 */
TOutputStreamContainer::~TOutputStreamContainer()
{
  outputStreamContainer.clear();
}// end of Destructor
//--------------------------------------------------------------------------------------------------

/**
 * Add all streams in the simulation to the container, set all streams records here! \n
 * Please note, the Matrix container has to be populated before calling this routine.
 *
 * @param [in] matrixContainer - matrix container to link the steams with sampled matrices and
 *                               sensor masks
 */
void TOutputStreamContainer::AddStreams(TMatrixContainer& matrixContainer)
{
  TParameters& params = TParameters::GetInstance();

  //----------------------------------------- pressure  ------------------------------------------//
  if (params.IsStore_p_raw())
  {
    outputStreamContainer[p_sensor_raw] = CreateNewOutputStream(matrixContainer,
                                                                p,
                                                                p_Name,
                                                                TBaseOutputHDF5Stream::roNONE);
  }

  if (params.IsStore_p_rms())
  {
    outputStreamContainer[p_sensor_rms] = CreateNewOutputStream(matrixContainer,
                                                                p,
                                                                p_rms_Name,
                                                                TBaseOutputHDF5Stream::roRMS);
  }

  if (params.IsStore_p_max())
  {
    outputStreamContainer[p_sensor_max] = CreateNewOutputStream(matrixContainer,
                                                                p,
                                                                p_max_Name,
                                                                TBaseOutputHDF5Stream::roMAX);
  }

  if (params.IsStore_p_min())
  {
    outputStreamContainer[p_sensor_min] = CreateNewOutputStream(matrixContainer,
                                                                p,
                                                                p_min_Name,
                                                                TBaseOutputHDF5Stream::roMIN);
  }

  if (params.IsStore_p_max_all())
  {
    outputStreamContainer[p_sensor_max_all] =
            new TWholeDomainOutputHDF5Stream(params.HDF5_OutputFile,
                                             p_max_all_Name,
                                             matrixContainer.GetMatrix<TRealMatrix>(p),
                                             TBaseOutputHDF5Stream::roMAX);
  }

  if (params.IsStore_p_min_all())
  {
    outputStreamContainer[p_sensor_min_all] =
            new TWholeDomainOutputHDF5Stream(params.HDF5_OutputFile,
                                             p_min_all_Name,
                                             matrixContainer.GetMatrix<TRealMatrix>(p),
                                             TBaseOutputHDF5Stream::roMIN);
  }

  //---------------------------------------- velocity --------------------------------------------//
  if (params.IsStore_u_raw())
  {
    outputStreamContainer[ux_sensor_raw] = CreateNewOutputStream(matrixContainer,
                                                                 ux_sgx,
                                                                 ux_Name,
                                                                 TBaseOutputHDF5Stream::roNONE);
    outputStreamContainer[uy_sensor_raw] = CreateNewOutputStream(matrixContainer,
                                                                 uy_sgy,
                                                                 uy_Name,
                                                                 TBaseOutputHDF5Stream::roNONE);
    outputStreamContainer[uz_sensor_raw] = CreateNewOutputStream(matrixContainer,
                                                                 uz_sgz,
                                                                 uz_Name,
                                                                 TBaseOutputHDF5Stream::roNONE);
  }

  if (params.IsStore_u_non_staggered_raw())
  {
    outputStreamContainer[ux_shifted_sensor_raw] = CreateNewOutputStream(matrixContainer,
                                                                         ux_shifted,
                                                                         ux_non_staggered_Name,
                                                                         TBaseOutputHDF5Stream::roNONE);
    outputStreamContainer[uy_shifted_sensor_raw] = CreateNewOutputStream(matrixContainer,
                                                                         uy_shifted,
                                                                         uy_non_staggered_Name,
                                                                         TBaseOutputHDF5Stream::roNONE);
    outputStreamContainer[uz_shifted_sensor_raw] = CreateNewOutputStream(matrixContainer,
                                                                         uz_shifted,
                                                                         uz_non_staggered_Name,
                                                                         TBaseOutputHDF5Stream::roNONE);
  }

  if (params.IsStore_u_rms())
  {
    outputStreamContainer[ux_sensor_rms] = CreateNewOutputStream(matrixContainer,
                                                                 ux_sgx,
                                                                 ux_rms_Name,
                                                                 TBaseOutputHDF5Stream::roRMS);
    outputStreamContainer[uy_sensor_rms] = CreateNewOutputStream(matrixContainer,
                                                                 uy_sgy,
                                                                 uy_rms_Name,
                                                                 TBaseOutputHDF5Stream::roRMS);
    outputStreamContainer[uz_sensor_rms] = CreateNewOutputStream(matrixContainer,
                                                                 uz_sgz,
                                                                 uz_rms_Name,
                                                                 TBaseOutputHDF5Stream::roRMS);
  }

   if (params.IsStore_u_max())
  {
    outputStreamContainer[ux_sensor_max] = CreateNewOutputStream(matrixContainer,
                                                                 ux_sgx,
                                                                 ux_max_Name,
                                                                 TBaseOutputHDF5Stream::roMAX);
    outputStreamContainer[uy_sensor_max] = CreateNewOutputStream(matrixContainer,
                                                                 uy_sgy,
                                                                 uy_max_Name,
                                                                 TBaseOutputHDF5Stream::roMAX);
    outputStreamContainer[uz_sensor_max] = CreateNewOutputStream(matrixContainer,
                                                                 uz_sgz,
                                                                 uz_max_Name,
                                                                 TBaseOutputHDF5Stream::roMAX);
  }

  if (params.IsStore_u_min())
  {
    outputStreamContainer[ux_sensor_min] = CreateNewOutputStream(matrixContainer,
                                                                 ux_sgx,
                                                                 ux_min_Name,
                                                                 TBaseOutputHDF5Stream::roMIN);
    outputStreamContainer[uy_sensor_min] = CreateNewOutputStream(matrixContainer,
                                                                 uy_sgy,
                                                                 uy_min_Name,
                                                                 TBaseOutputHDF5Stream::roMIN);
    outputStreamContainer[uz_sensor_min] = CreateNewOutputStream(matrixContainer,
                                                                 uz_sgz,
                                                                 uz_min_Name,
                                                                 TBaseOutputHDF5Stream::roMIN);
  }

  if (params.IsStore_u_max_all())
  {
    outputStreamContainer[ux_sensor_max_all] =
            new TWholeDomainOutputHDF5Stream(params.HDF5_OutputFile,
                                             ux_max_all_Name,
                                             matrixContainer.GetMatrix<TRealMatrix>(ux_sgx),
                                             TBaseOutputHDF5Stream::roMAX);
    outputStreamContainer[uy_sensor_max_all] =
            new TWholeDomainOutputHDF5Stream(params.HDF5_OutputFile,
                                             uy_max_all_Name,
                                             matrixContainer.GetMatrix<TRealMatrix>(uy_sgy),
                                             TBaseOutputHDF5Stream::roMAX);
    outputStreamContainer[uz_sensor_max_all] =
            new TWholeDomainOutputHDF5Stream(params.HDF5_OutputFile,
                                             uz_max_all_Name,
                                             matrixContainer.GetMatrix<TRealMatrix>(uz_sgz),
                                             TBaseOutputHDF5Stream::roMAX);
  }

  if (params.IsStore_u_min_all())
  {
    outputStreamContainer[ux_sensor_min_all] =
            new TWholeDomainOutputHDF5Stream(params.HDF5_OutputFile,
                                             ux_min_all_Name,
                                             matrixContainer.GetMatrix<TRealMatrix>(ux_sgx),
                                             TBaseOutputHDF5Stream::roMIN);
    outputStreamContainer[uy_sensor_min_all] =
            new TWholeDomainOutputHDF5Stream(params.HDF5_OutputFile,
                                             uy_min_all_Name,
                                             matrixContainer.GetMatrix<TRealMatrix>(uy_sgy),
                                             TBaseOutputHDF5Stream::roMIN);
    outputStreamContainer[uz_sensor_min_all] =
            new TWholeDomainOutputHDF5Stream(params.HDF5_OutputFile,
                                             uz_min_all_Name,
                                             matrixContainer.GetMatrix<TRealMatrix>(uz_sgz),
                                             TBaseOutputHDF5Stream::roMIN);
  }
}// end of AddStreams
//--------------------------------------------------------------------------------------------------

/**
 * Create all streams.
 */
void TOutputStreamContainer::CreateStreams()
{
  for (const auto& it : outputStreamContainer)
  {
    if (it.second)
    {
      it.second->Create();
    }
  }
}// end of CreateStreams
//--------------------------------------------------------------------------------------------------

/**
 * Reopen all streams after restarting from checkpoint.
 */
void TOutputStreamContainer::ReopenStreams()
{
  for (const auto& it : outputStreamContainer)
  {
    if (it.second)
    {
      it.second->Reopen();
    }
  }
}// end of ReopenStreams
//--------------------------------------------------------------------------------------------------


/**
 * Sample all streams.
 * @warning In the GPU implementation, no data is flushed on disk (just data is sampled)
 */
void TOutputStreamContainer::SampleStreams()
{
  for (const auto& it : outputStreamContainer)
  {
    if (it.second)
    {
      it.second->Sample();
    }
  }
}// end of SampleStreams
//--------------------------------------------------------------------------------------------------

/**
 * Flush stream data to disk.
 * @warning In GPU implementation, data from raw streams is flushed here. Aggregated streams are
 * ignored.
 */
void TOutputStreamContainer::FlushRawStreams()
{
  for (const auto& it : outputStreamContainer)
  {
    if (it.second)
    {
      it.second->FlushRaw();
    }
  }
}// end of SampleStreams
//--------------------------------------------------------------------------------------------------

/**
 * Checkpoint streams without post-processing (flush to the file).
 */
void TOutputStreamContainer::CheckpointStreams()
{
  for (const auto& it : outputStreamContainer)
  {
    if (it.second)
    {
      it.second->Checkpoint();
    }
  }
}// end of CheckpointStreams
//--------------------------------------------------------------------------------------------------

/**
 * Post-process all streams and flush them to the file.
 */
void TOutputStreamContainer::PostProcessStreams()
{
  for (const auto& it : outputStreamContainer)
  {
    if (it.second)
    {
      it.second->PostProcess();
    }
  }
}// end of CheckpointStreams
//------------------------------------------------------------------------------


/**
 * Close all streams (apply post-processing if necessary, flush data and close).
 */
void TOutputStreamContainer::CloseStreams()
{
  for (const auto& it : outputStreamContainer)
  {
    if (it.second)
    {
      it.second->Close();
    }
  }
}// end of CloseStreams
//--------------------------------------------------------------------------------------------------

/**
 *  Free all streams - destroy them.
 */
void TOutputStreamContainer::FreeStreams()
{
  for (auto& it : outputStreamContainer)
  {
    if (it.second)
    {
      delete it.second;
      it.second = nullptr;
    }
  }
  outputStreamContainer.clear();
}// end of FreeAllStreams
//--------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//


/**
 * Create a new output stream.
 * @param [in] matrixContainer  - name of the HDF5 dataset or group
 * @param [in] sampledMatrixIdx - code id of the matrix
 * @param [in] fileDatasetName  - name of the HDF5 dataset or group
 * @param [in] reductionOp      - reduction operator
 *
 * @return new output stream with defined links
 *
 */
TBaseOutputHDF5Stream* TOutputStreamContainer::CreateNewOutputStream(TMatrixContainer& matrixContainer,
                                                                     const TMatrixIdx  sampledMatrixIdx,
                                                                     const char*       fileDatasetName,
                                                                     const TBaseOutputHDF5Stream::TReductionOperator reductionOp)
{
  TParameters& params = TParameters::GetInstance();



  if (params.Get_sensor_mask_type() == TParameters::smt_index)
  {
    return (new TIndexOutputHDF5Stream(params.HDF5_OutputFile,
                                        fileDatasetName,
                                        matrixContainer.GetMatrix<TRealMatrix>(sampledMatrixIdx),
                                        matrixContainer.GetMatrix<TIndexMatrix>(sensor_mask_index),
                                        reductionOp)
            );
  }
  else
  {
    return (new TCuboidOutputHDF5Stream(params.HDF5_OutputFile,
                                         fileDatasetName,
                                         matrixContainer.GetMatrix<TRealMatrix>(sampledMatrixIdx),
                                         matrixContainer.GetMatrix<TIndexMatrix>(sensor_mask_corners),
                                         reductionOp)
            );
  }
}// end of CreateNewOutputStream
//--------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//
