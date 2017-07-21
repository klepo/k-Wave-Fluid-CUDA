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
 *              21 July      2017, 16:48 (revised)
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


//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Constants -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------- Public methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Default constructor.
 */
OutputStreamContainer::OutputStreamContainer()
  : mContainer()
{

}// end of OutputStreamContainer
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor.
 */
OutputStreamContainer::~OutputStreamContainer()
{
  mContainer.clear();
}// end of ~OutputStreamContainer
//----------------------------------------------------------------------------------------------------------------------

/**
 * Add all streams in the simulation to the container, set all streams records here!

 */
void OutputStreamContainer::addStreams(MatrixContainer& matrixContainer)
{
  Parameters& params = Parameters::getInstance();

  using MatrixIdx    = MatrixContainer::MatrixIdx;
  using ReductionOp = BaseOutputStream::ReduceOperator;
  //--------------------------------------------------- pressure  ----------------------------------------------------//
  if (params.getStorePressureRawFlag())
  {
    mContainer[OutputStreamIdx::kPressureRaw]
            = createNewOutputStream(matrixContainer, MatrixIdx::kP, kPressureRawName, ReductionOp::kNone);
  }

  if (params.getStorePressureRmsFlag())
  {
    mContainer[OutputStreamIdx::kPressureRms]
            = createNewOutputStream(matrixContainer, MatrixIdx::kP, kPressureRmsName, ReductionOp::kRms);
  }

  if (params.getStorePressureMaxFlag())
  {
    mContainer[OutputStreamIdx::kPressureMax]
            = createNewOutputStream(matrixContainer, MatrixIdx::kP, kPressureMaxName, ReductionOp::kMax);
  }

  if (params.getStorePressureMinFlag())
  {
    mContainer[OutputStreamIdx::kPressureMin]
            = createNewOutputStream(matrixContainer, MatrixIdx::kP, kPressureMinName, ReductionOp::kMin);
  }

  if (params.getStorePressureMaxAllFlag())
  {
    mContainer[OutputStreamIdx::kPressureMaxAll]
            = new WholeDomainOutputStream(params.getOutputFile(),
                                          kPressureMaxAllName,
                                          matrixContainer.getMatrix<RealMatrix>(MatrixIdx::kP),
                                          ReductionOp::kMax);
  }

  if (params.getStorePressureMinAllFlag())
  {
    mContainer[OutputStreamIdx::kPressureMinAll]
            = new WholeDomainOutputStream(params.getOutputFile(),
                                          kPressureMinAllName,
                                          matrixContainer.getMatrix<RealMatrix>(MatrixIdx::kP),
                                          ReductionOp::kMin);
  }

  //-------------------------------------------------- velocity ------------------------------------------------------//
  if (params.getStoreVelocityRawFlag())
  {
    mContainer[OutputStreamIdx::kVelocityXRaw]
            = createNewOutputStream(matrixContainer, MatrixIdx::kUxSgx, kUxName, ReductionOp::kNone);
    mContainer[OutputStreamIdx::kVelocityYRaw]
            = createNewOutputStream(matrixContainer, MatrixIdx::kUySgy, kUyName, ReductionOp::kNone);
    mContainer[OutputStreamIdx::kVelocityZRaw]
            = createNewOutputStream(matrixContainer, MatrixIdx::kUzSgz, kUzName, ReductionOp::kNone);
  }

  if (params.getStoreVelocityNonStaggeredRaw())
  {
    mContainer[OutputStreamIdx::kVelocityXNonStaggeredRaw]
            = createNewOutputStream(matrixContainer, MatrixIdx::kUxShifted, kUxNonStaggeredName, ReductionOp::kNone);
    mContainer[OutputStreamIdx::kVelocityYNonStaggeredRaw]
            = createNewOutputStream(matrixContainer, MatrixIdx::kUyShifted, kUyNonStaggeredName, ReductionOp::kNone);
    mContainer[OutputStreamIdx::kVelocityZNonStaggeredRaw]
            = createNewOutputStream(matrixContainer, MatrixIdx::kUzShifted, kUzNonStaggeredName, ReductionOp::kNone);
  }

  if (params.getStoreVelocityRmsFlag())
  {
    mContainer[OutputStreamIdx::kVelocityXRms]
            = createNewOutputStream(matrixContainer, MatrixIdx::kUxSgx, kUxRmsName, ReductionOp::kRms);
    mContainer[OutputStreamIdx::kVelocityYRms]
            = createNewOutputStream(matrixContainer, MatrixIdx::kUySgy, kUyRmsName, ReductionOp::kRms);
    mContainer[OutputStreamIdx::kVelocityZRms]
            = createNewOutputStream(matrixContainer, MatrixIdx::kUzSgz, kUzRmsName, ReductionOp::kRms);
  }

   if (params.getVelocityMaxFlag())
  {
    mContainer[OutputStreamIdx::kVelocityXMax]
            = createNewOutputStream(matrixContainer, MatrixIdx::kUxSgx, kUxMaxName, ReductionOp::kMax);
    mContainer[OutputStreamIdx::kVelocityYMax]
            = createNewOutputStream(matrixContainer, MatrixIdx::kUySgy, kUyMaxName, ReductionOp::kMax);
    mContainer[OutputStreamIdx::kVelocityZMax]
            = createNewOutputStream(matrixContainer, MatrixIdx::kUzSgz, kUzMaxName, ReductionOp::kMax);
  }

  if (params.getStoreVelocityMinFlag())
  {
    mContainer[OutputStreamIdx::kVelocityXMin]
            = createNewOutputStream(matrixContainer, MatrixIdx::kUxSgx, kUxMinName, ReductionOp::kMin);
    mContainer[OutputStreamIdx::kVelocityYMin]
            = createNewOutputStream(matrixContainer, MatrixIdx::kUySgy, kUyMinName, ReductionOp::kMin);
    mContainer[OutputStreamIdx::kVelocityZMin]
            = createNewOutputStream(matrixContainer, MatrixIdx::kUzSgz, kUzMinName, ReductionOp::kMin);
  }

  if (params.getStoreVelocityMaxAllFlag())
  {
    mContainer[OutputStreamIdx::kVelocityXMaxAll] =
            new WholeDomainOutputStream(params.getOutputFile(),
                                             kUxMaxAllName,
                                             matrixContainer.getMatrix<RealMatrix>(MatrixIdx::kUxSgx),
                                             ReductionOp::kMax);
    mContainer[OutputStreamIdx::kVelocityYMaxAll] =
            new WholeDomainOutputStream(params.getOutputFile(),
                                             kUyMaxAllName,
                                             matrixContainer.getMatrix<RealMatrix>(MatrixIdx::kUySgy),
                                             ReductionOp::kMax);
    mContainer[OutputStreamIdx::kVelocityZMaxAll] =
            new WholeDomainOutputStream(params.getOutputFile(),
                                             kUzMaxAllName,
                                             matrixContainer.getMatrix<RealMatrix>(MatrixIdx::kUzSgz),
                                             ReductionOp::kMax);
  }

  if (params.getStoreStoreVelocityMinAllFlag())
  {
    mContainer[OutputStreamIdx::kVelocityXMinAll] =
            new WholeDomainOutputStream(params.getOutputFile(),
                                             kUxMinAllName,
                                             matrixContainer.getMatrix<RealMatrix>(MatrixIdx::kUxSgx),
                                             ReductionOp::kMin);
    mContainer[OutputStreamIdx::kVelocityYMinAll] =
            new WholeDomainOutputStream(params.getOutputFile(),
                                             kUyMinAllName,
                                             matrixContainer.getMatrix<RealMatrix>(MatrixIdx::kUySgy),
                                             ReductionOp::kMin);
    mContainer[OutputStreamIdx::kVelocityZMinAll] =
            new WholeDomainOutputStream(params.getOutputFile(),
                                             kUzMinAllName,
                                             matrixContainer.getMatrix<RealMatrix>(MatrixIdx::kUzSgz),
                                             ReductionOp::kMin);
  }
}// end of addStreams
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create all streams.
 */
void OutputStreamContainer::createStreams()
{
  for (const auto& it : mContainer)
  {
    if (it.second)
    {
      it.second->create();
    }
  }
}// end of createStreams
//----------------------------------------------------------------------------------------------------------------------

/**
 * Reopen all streams after restarting from checkpoint.
 */
void OutputStreamContainer::reopenStreams()
{
  for (const auto& it : mContainer)
  {
    if (it.second)
    {
      it.second->reopen();
    }
  }
}// end of reopenStreams
//----------------------------------------------------------------------------------------------------------------------


/**
 * Sample all streams.
 * @warning In the GPU implementation, no data is flushed on disk (just data is sampled)
 */
void OutputStreamContainer::sampleStreams()
{
  for (const auto& it : mContainer)
  {
    if (it.second)
    {
      it.second->sample();
    }
  }
}// end of sampleStreams
//----------------------------------------------------------------------------------------------------------------------

/**
 * Flush stream data to disk.
 */
void OutputStreamContainer::flushRawStreams()
{
  for (const auto& it : mContainer)
  {
    if (it.second)
    {
      it.second->flushRaw();
    }
  }
}// end of sampleStreams
//----------------------------------------------------------------------------------------------------------------------

/**
 * Checkpoint streams without post-processing (flush to the file).
 */
void OutputStreamContainer::checkpointStreams()
{
  for (const auto& it : mContainer)
  {
    if (it.second)
    {
      it.second->checkpoint();
    }
  }
}// end of checkpointStreams
//----------------------------------------------------------------------------------------------------------------------

/**
 * Post-process all streams and flush them to the file.
 */
void OutputStreamContainer::postProcessStreams()
{
  for (const auto& it : mContainer)
  {
    if (it.second)
    {
      it.second->postProcess();
    }
  }
}// end of postProcessStreams
//----------------------------------------------------------------------------------------------------------------------


/**
 * Close all streams (apply post-processing if necessary, flush data and close).
 */
void OutputStreamContainer::closeStreams()
{
  for (const auto& it : mContainer)
  {
    if (it.second)
    {
      it.second->close();
    }
  }
}// end of closeStreams
//----------------------------------------------------------------------------------------------------------------------

/**
 *  Free all streams - destroy them.
 */
void OutputStreamContainer::freeStreams()
{
  for (auto& it : mContainer)
  {
    if (it.second)
    {
      delete it.second;
      it.second = nullptr;
    }
  }
  mContainer.clear();
}// end of rreeStreams
//----------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//


/**
 * Create a new output stream.
 */
BaseOutputStream* OutputStreamContainer::createNewOutputStream(MatrixContainer&                       matrixContainer,
                                                               const MatrixContainer::MatrixIdx       sampledMatrixIdx,
                                                               const MatrixName&                      fileDatasetName,
                                                               const BaseOutputStream::ReduceOperator reduceOp)
{
  Parameters& params = Parameters::getInstance();

  using MatrixIdx = MatrixContainer::MatrixIdx;

  if (params.getSensorMaskType() == Parameters::SensorMaskType::kIndex)
  {
    return (new IndexOutputStream(params.getOutputFile(),
                                  fileDatasetName,
                                  matrixContainer.getMatrix<RealMatrix>(sampledMatrixIdx),
                                  matrixContainer.getMatrix<IndexMatrix>(MatrixIdx::kSensorMaskIndex),
                                  reduceOp)
            );
  }
  else
  {
    return (new CuboidOutputStream(params.getOutputFile(),
                                   fileDatasetName,
                                   matrixContainer.getMatrix<RealMatrix>(sampledMatrixIdx),
                                   matrixContainer.getMatrix<IndexMatrix>(MatrixIdx::kSensorMaskCorners),
                                   reduceOp)
            );
  }
}// end of createNewOutputStream
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//
