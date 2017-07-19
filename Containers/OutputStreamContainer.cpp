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
 *              19 July      2017  15:18 (revised)
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

#include <OutputStreams/BaseOutputStream.h>
#include <OutputStreams/IndexOutputStream.h>
#include <OutputStreams/CuboidOutputStream.h>
#include <OutputStreams/WholeDomainOutputStream.h>


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
  Parameters& params = Parameters::getInstance();

  using MatrixId    = TMatrixContainer::TMatrixIdx;
  using ReductionOp = BaseOutputStream::ReduceOperator;
  //----------------------------------------- pressure  ------------------------------------------//
  if (params.getStorePressureRawFlag())
  {
    outputStreamContainer[TOutputStreamIdx::p_sensor_raw]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::p,
                                    kPName,
                                    ReductionOp::kNone);
  }

  if (params.getStorePressureRmsFlag())
  {
    outputStreamContainer[TOutputStreamIdx::p_sensor_rms]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::p,
                                    kPRmsName,
                                    ReductionOp::kRms);
  }

  if (params.getStorePressureMaxFlag())
  {
    outputStreamContainer[TOutputStreamIdx::p_sensor_max] =
            CreateNewOutputStream(matrixContainer,
                                  MatrixId::p,
                                  kPMaxName,
                                  ReductionOp::kMax);
  }

  if (params.getStorePressureMinFlag())
  {
    outputStreamContainer[TOutputStreamIdx::p_sensor_min] =
            CreateNewOutputStream(matrixContainer,
                                  MatrixId::p,
                                  kPminName,
                                  ReductionOp::kMin);
  }

  if (params.getStorePressureMaxAllFlag())
  {
    outputStreamContainer[TOutputStreamIdx::p_sensor_max_all] =
            new WholeDomainOutputStream(params.getOutputFile(),
                                             kPMaxAllName,
                                             matrixContainer.GetMatrix<RealMatrix>(MatrixId::p),
                                             ReductionOp::kMax);
  }

  if (params.getStorePressureMinAllFlag())
  {
    outputStreamContainer[TOutputStreamIdx::p_sensor_min_all] =
            new WholeDomainOutputStream(params.getOutputFile(),
                                             kPMinAllName,
                                             matrixContainer.GetMatrix<RealMatrix>(MatrixId::p),
                                             ReductionOp::kMin);
  }

  //---------------------------------------- velocity --------------------------------------------//
  if (params.getStoreVelocityRawFlag())
  {
    outputStreamContainer[TOutputStreamIdx::ux_sensor_raw]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::ux_sgx,
                                    kUxName,
                                    ReductionOp::kNone);
    outputStreamContainer[TOutputStreamIdx::uy_sensor_raw]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uy_sgy,
                                    kUyName,
                                    ReductionOp::kNone);
    outputStreamContainer[TOutputStreamIdx::uz_sensor_raw]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uz_sgz,
                                    kUzName,
                                    ReductionOp::kNone);
  }

  if (params.getStoreVelocityNonStaggeredRaw())
  {
    outputStreamContainer[TOutputStreamIdx::ux_shifted_sensor_raw]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::ux_shifted,
                                    kUxNonStaggeredName,
                                    ReductionOp::kNone);
    outputStreamContainer[TOutputStreamIdx::uy_shifted_sensor_raw]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uy_shifted,
                                    kUyNonStaggeredName,
                                    ReductionOp::kNone);
    outputStreamContainer[TOutputStreamIdx::uz_shifted_sensor_raw]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uz_shifted,
                                    kUzNonStaggeredName,
                                    ReductionOp::kNone);
  }

  if (params.getStoreVelocityRmsFlag())
  {
    outputStreamContainer[TOutputStreamIdx::ux_sensor_rms]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::ux_sgx,
                                    kUxRmsName,
                                    ReductionOp::kRms);
    outputStreamContainer[TOutputStreamIdx::uy_sensor_rms]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uy_sgy,
                                    kUyRmsName,
                                    ReductionOp::kRms);
    outputStreamContainer[TOutputStreamIdx::uz_sensor_rms]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uz_sgz,
                                    kUzRmsName,
                                    ReductionOp::kRms);
  }

   if (params.getVelocityMaxFlag())
  {
    outputStreamContainer[TOutputStreamIdx::ux_sensor_max]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::ux_sgx,
                                    kUxMaxName,
                                    ReductionOp::kMax);
    outputStreamContainer[TOutputStreamIdx::uy_sensor_max]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uy_sgy,
                                    kUyMaxName,
                                    ReductionOp::kMax);
    outputStreamContainer[TOutputStreamIdx::uz_sensor_max]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uz_sgz,
                                    kUzMaxName,
                                    ReductionOp::kMax);
  }

  if (params.getStoreVelocityMinFlag())
  {
    outputStreamContainer[TOutputStreamIdx::ux_sensor_min]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::ux_sgx,
                                    kUxMinName,
                                    ReductionOp::kMin);
    outputStreamContainer[TOutputStreamIdx::uy_sensor_min]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uy_sgy,
                                    kUyMinName,
                                    ReductionOp::kMin);
    outputStreamContainer[TOutputStreamIdx::uz_sensor_min]
            = CreateNewOutputStream(matrixContainer,
                                    MatrixId::uz_sgz,
                                    kUzMinName,
                                    ReductionOp::kMin);
  }

  if (params.getStoreVelocityMaxAllFlag())
  {
    outputStreamContainer[TOutputStreamIdx::ux_sensor_max_all] =
            new WholeDomainOutputStream(params.getOutputFile(),
                                             kUxMaxAllName,
                                             matrixContainer.GetMatrix<RealMatrix>(MatrixId::ux_sgx),
                                             ReductionOp::kMax);
    outputStreamContainer[TOutputStreamIdx::uy_sensor_max_all] =
            new WholeDomainOutputStream(params.getOutputFile(),
                                             kUyMaxAllName,
                                             matrixContainer.GetMatrix<RealMatrix>(MatrixId::uy_sgy),
                                             ReductionOp::kMax);
    outputStreamContainer[TOutputStreamIdx::uz_sensor_max_all] =
            new WholeDomainOutputStream(params.getOutputFile(),
                                             kUzMaxAllName,
                                             matrixContainer.GetMatrix<RealMatrix>(MatrixId::uz_sgz),
                                             ReductionOp::kMax);
  }

  if (params.getStoreStoreVelocityMinAllFlag())
  {
    outputStreamContainer[TOutputStreamIdx::ux_sensor_min_all] =
            new WholeDomainOutputStream(params.getOutputFile(),
                                             kUxMinAllName,
                                             matrixContainer.GetMatrix<RealMatrix>(MatrixId::ux_sgx),
                                             ReductionOp::kMin);
    outputStreamContainer[TOutputStreamIdx::uy_sensor_min_all] =
            new WholeDomainOutputStream(params.getOutputFile(),
                                             kUyMinAllName,
                                             matrixContainer.GetMatrix<RealMatrix>(MatrixId::uy_sgy),
                                             ReductionOp::kMin);
    outputStreamContainer[TOutputStreamIdx::uz_sensor_min_all] =
            new WholeDomainOutputStream(params.getOutputFile(),
                                             kUzMinAllName,
                                             matrixContainer.GetMatrix<RealMatrix>(MatrixId::uz_sgz),
                                             ReductionOp::kMin);
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
      it.second->create();
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
      it.second->reopen();
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
      it.second->sample();
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
      it.second->flushRaw();
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
      it.second->checkpoint();
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
      it.second->postProcess();
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
      it.second->close();
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
BaseOutputStream* TOutputStreamContainer::CreateNewOutputStream(TMatrixContainer&                            matrixContainer,
                                                                     const TMatrixContainer::TMatrixIdx           sampledMatrixIdx,
                                                                     const MatrixName&                           fileDatasetName,
                                                                     const BaseOutputStream::ReduceOperator reduceOp)
{
  Parameters& params = Parameters::getInstance();

  using MatrixId = TMatrixContainer::TMatrixIdx;

  if (params.getSensorMaskType() == Parameters::SensorMaskType::kIndex)
  {
    return (new IndexOutputStream(params.getOutputFile(),
                                       fileDatasetName,
                                       matrixContainer.GetMatrix<RealMatrix>(sampledMatrixIdx),
                                       matrixContainer.GetMatrix<IndexMatrix>(MatrixId::sensor_mask_index),
                                       reduceOp)
            );
  }
  else
  {
    return (new CuboidOutputStream(params.getOutputFile(),
                                        fileDatasetName,
                                        matrixContainer.GetMatrix<RealMatrix>(sampledMatrixIdx),
                                        matrixContainer.GetMatrix<IndexMatrix>(MatrixId::sensor_mask_corners),
                                        reduceOp)
            );
  }
}// end of CreateNewOutputStream
//--------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//
