 /**
 * @file        OutputStreamContainer.cpp
  *
 * @author      Jiri Jaros & Beau Johnston \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file for the output stream container.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        04 December  2014, 11:41 (created) \n
 *              11 July      2017  14:41 (revised)
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
//-------------------------------------- Constants -----------------------------------------------//
//------------------------------------------------------------------------------------------------//


//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Default constructor.
 */
TOutputStreamContainer::TOutputStreamContainer() : outputStreamContainer()
{

}// end of TOutputStreamContainer
//--------------------------------------------------------------------------------------------------

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
 * @param [in] matrixContainer - Matrix container to link the steams with sampled matrices and
 *                               sensor masks
 */
void TOutputStreamContainer::AddStreams(TMatrixContainer& matrixContainer)
{
  TParameters& params = TParameters::GetInstance();

  using MatrixId    = TMatrixContainer::TMatrixIdx;
  using ReductionOp = TBaseOutputHDF5Stream::TReduceOperator;
  //----------------------------------------- pressure  ------------------------------------------//
  if (params.IsStore_p_raw())
  {
    outputStreamContainer[TOutputStreamIdx::p_sensor_raw]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::p,
                                    kPName,
                                    ReductionOp::NONE);
  }

  if (params.IsStore_p_rms())
  {
    outputStreamContainer[TOutputStreamIdx::p_sensor_rms]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::p,
                                    kPRmsName,
                                    ReductionOp::RMS);
  }

  if (params.IsStore_p_max())
  {
    outputStreamContainer[TOutputStreamIdx::p_sensor_max] =
            CreateNewOutputStream(matrixContainer,
                                  MatrixId::p,
                                  kPMaxName,
                                  ReductionOp::MAX);
  }

  if (params.IsStore_p_min())
  {
    outputStreamContainer[TOutputStreamIdx::p_sensor_min] =
            CreateNewOutputStream(matrixContainer,
                                  MatrixId::p,
                                  kPminName,
                                  ReductionOp::MIN);
  }

  if (params.IsStore_p_max_all())
  {
    outputStreamContainer[TOutputStreamIdx::p_sensor_max_all] =
            new TWholeDomainOutputHDF5Stream(params.GetOutputFile(),
                                             kPMaxAllName,
                                             matrixContainer.GetMatrix<TRealMatrix>(MatrixId::p),
                                             ReductionOp::MAX);
  }

  if (params.IsStore_p_min_all())
  {
    outputStreamContainer[TOutputStreamIdx::p_sensor_min_all] =
            new TWholeDomainOutputHDF5Stream(params.GetOutputFile(),
                                             kPMinAllName,
                                             matrixContainer.GetMatrix<TRealMatrix>(MatrixId::p),
                                             ReductionOp::MIN);
  }

  //---------------------------------------- velocity --------------------------------------------//
  if (params.IsStore_u_raw())
  {
    outputStreamContainer[TOutputStreamIdx::ux_sensor_raw]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::ux_sgx,
                                    kUxName,
                                    ReductionOp::NONE);
    outputStreamContainer[TOutputStreamIdx::uy_sensor_raw]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uy_sgy,
                                    kUyName,
                                    ReductionOp::NONE);
    outputStreamContainer[TOutputStreamIdx::uz_sensor_raw]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uz_sgz,
                                    kUzName,
                                    ReductionOp::NONE);
  }

  if (params.IsStore_u_non_staggered_raw())
  {
    outputStreamContainer[TOutputStreamIdx::ux_shifted_sensor_raw]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::ux_shifted,
                                    kUxNonStaggeredName,
                                    ReductionOp::NONE);
    outputStreamContainer[TOutputStreamIdx::uy_shifted_sensor_raw]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uy_shifted,
                                    kUyNonStaggeredName,
                                    ReductionOp::NONE);
    outputStreamContainer[TOutputStreamIdx::uz_shifted_sensor_raw]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uz_shifted,
                                    kUzNonStaggeredName,
                                    ReductionOp::NONE);
  }

  if (params.IsStore_u_rms())
  {
    outputStreamContainer[TOutputStreamIdx::ux_sensor_rms]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::ux_sgx,
                                    kUxRmsName,
                                    ReductionOp::RMS);
    outputStreamContainer[TOutputStreamIdx::uy_sensor_rms]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uy_sgy,
                                    kUyRmsName,
                                    ReductionOp::RMS);
    outputStreamContainer[TOutputStreamIdx::uz_sensor_rms]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uz_sgz,
                                    kUzRmsName,
                                    ReductionOp::RMS);
  }

   if (params.IsStore_u_max())
  {
    outputStreamContainer[TOutputStreamIdx::ux_sensor_max]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::ux_sgx,
                                    kUxMaxName,
                                    ReductionOp::MAX);
    outputStreamContainer[TOutputStreamIdx::uy_sensor_max]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uy_sgy,
                                    kUyMaxName,
                                    ReductionOp::MAX);
    outputStreamContainer[TOutputStreamIdx::uz_sensor_max]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uz_sgz,
                                    kUzMaxName,
                                    ReductionOp::MAX);
  }

  if (params.IsStore_u_min())
  {
    outputStreamContainer[TOutputStreamIdx::ux_sensor_min]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::ux_sgx,
                                    kUxMinName,
                                    ReductionOp::MIN);
    outputStreamContainer[TOutputStreamIdx::uy_sensor_min]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uy_sgy,
                                    kUyMinName,
                                    ReductionOp::MIN);
    outputStreamContainer[TOutputStreamIdx::uz_sensor_min]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uz_sgz,
                                    kUzMinName,
                                    ReductionOp::MIN);
  }

  if (params.IsStore_u_max_all())
  {
    outputStreamContainer[TOutputStreamIdx::ux_sensor_max_all] =
            new TWholeDomainOutputHDF5Stream(params.GetOutputFile(),
                                             kUxMaxAllName,
                                             matrixContainer.GetMatrix<TRealMatrix>(MatrixId::ux_sgx),
                                             ReductionOp::MAX);
    outputStreamContainer[TOutputStreamIdx::uy_sensor_max_all] =
            new TWholeDomainOutputHDF5Stream(params.GetOutputFile(),
                                             kUyMaxAllName,
                                             matrixContainer.GetMatrix<TRealMatrix>(MatrixId::uy_sgy),
                                             ReductionOp::MAX);
    outputStreamContainer[TOutputStreamIdx::uz_sensor_max_all] =
            new TWholeDomainOutputHDF5Stream(params.GetOutputFile(),
                                             kUzMaxAllName,
                                             matrixContainer.GetMatrix<TRealMatrix>(MatrixId::uz_sgz),
                                             ReductionOp::MAX);
  }

  if (params.IsStore_u_min_all())
  {
    outputStreamContainer[TOutputStreamIdx::ux_sensor_min_all] =
            new TWholeDomainOutputHDF5Stream(params.GetOutputFile(),
                                             kUxMinAllName,
                                             matrixContainer.GetMatrix<TRealMatrix>(MatrixId::ux_sgx),
                                             ReductionOp::MIN);
    outputStreamContainer[TOutputStreamIdx::uy_sensor_min_all] =
            new TWholeDomainOutputHDF5Stream(params.GetOutputFile(),
                                             kUyMinAllName,
                                             matrixContainer.GetMatrix<TRealMatrix>(MatrixId::uy_sgy),
                                             ReductionOp::MIN);
    outputStreamContainer[TOutputStreamIdx::uz_sensor_min_all] =
            new TWholeDomainOutputHDF5Stream(params.GetOutputFile(),
                                             kUzMinAllName,
                                             matrixContainer.GetMatrix<TRealMatrix>(MatrixId::uz_sgz),
                                             ReductionOp::MIN);
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
//--------------------------------------------------------------------------------------------------


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
 * @param [in] matrixContainer  - Name of the HDF5 dataset or group
 * @param [in] sampledMatrixIdx - Code id of the matrix
 * @param [in] fileDatasetName  - Name of the HDF5 dataset or group
 * @param [in] reduceOp         - Reduce operator
 *
 * @return New output stream with defined links
 *
 */
TBaseOutputHDF5Stream* TOutputStreamContainer::CreateNewOutputStream(TMatrixContainer&                            matrixContainer,
                                                                     const TMatrixContainer::TMatrixIdx           sampledMatrixIdx,
                                                                     const MatrixName&                           fileDatasetName,
                                                                     const TBaseOutputHDF5Stream::TReduceOperator reduceOp)
{
  TParameters& params = TParameters::GetInstance();

  using MatrixId = TMatrixContainer::TMatrixIdx;

  if (params.Get_sensor_mask_type() == TParameters::TSensorMaskType::INDEX)
  {
    return (new TIndexOutputHDF5Stream(params.GetOutputFile(),
                                       fileDatasetName,
                                       matrixContainer.GetMatrix<TRealMatrix>(sampledMatrixIdx),
                                       matrixContainer.GetMatrix<TIndexMatrix>(MatrixId::sensor_mask_index),
                                       reduceOp)
            );
  }
  else
  {
    return (new TCuboidOutputHDF5Stream(params.GetOutputFile(),
                                        fileDatasetName,
                                        matrixContainer.GetMatrix<TRealMatrix>(sampledMatrixIdx),
                                        matrixContainer.GetMatrix<TIndexMatrix>(MatrixId::sensor_mask_corners),
                                        reduceOp)
            );
  }
}// end of CreateNewOutputStream
//--------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//
