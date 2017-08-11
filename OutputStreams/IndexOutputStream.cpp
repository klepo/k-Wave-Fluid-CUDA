/**
 * @file        IndexOutputStream.cpp
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file of the class saving data based on index senor mask into
 *              the output HDF5 file.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        29 August    2014, 10:10 (created)
 *              11 August    2017, 15:21 (revised)
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

#include <OutputStreams/IndexOutputStream.h>
#include <OutputStreams/OutputStreamsCudaKernels.cuh>

#include <Parameters/Parameters.h>
#include <Logger/Logger.h>

//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------- Constants ------------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


/**
 * Constructor.
 */
IndexOutputStream::IndexOutputStream(Hdf5File&            file,
                                     MatrixName&          datasetName,
                                     const RealMatrix&    sourceMatrix,
                                     const IndexMatrix&   sensorMask,
                                     const ReduceOperator reduceOp)
  : BaseOutputStream(file, datasetName, sourceMatrix, reduceOp),
    mSensorMask(sensorMask),
    mDataset(H5I_BADID),
    mSampledTimeStep(0),
    mEventSamplingFinished()
{
  // Create event for sampling
  cudaCheckErrors(cudaEventCreate(&mEventSamplingFinished));
}// end of IndexOutputStream
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor.
 */
IndexOutputStream::~IndexOutputStream()
{
  // Destroy sampling event
  cudaCheckErrors(cudaEventDestroy(mEventSamplingFinished));

  close();
  // free memory
  freeMemory();
}// end of ~IndexOutputStream
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create a HDF5 stream, create a dataset, and allocate data for it.
 */
void IndexOutputStream::create()
{

  size_t nSampledElementsPerStep = mSensorMask.size();

  const Parameters& params = Parameters::getInstance();

  // Derive dataset dimension sizes
  DimensionSizes datasetSize(nSampledElementsPerStep,
                             (mReduceOp == ReduceOperator::kNone) ?
                                params.getNt() - params.getSamplingStartTimeIndex() : 1,
                              1);

  // Set HDF5 chunk size
  DimensionSizes chunkSize(nSampledElementsPerStep, 1, 1);
  // for chunks bigger than 32 MB
  if (nSampledElementsPerStep > (kChunkSize4MB * 8))
  {
      chunkSize.nx = kChunkSize4MB; // set chunk size to MB
  }

  // Create a dataset under the root group
  mDataset = mFile.createDataset(mFile.getRootGroup(),
                                 mRootObjectName,
                                 datasetSize,
                                 chunkSize,
                                 Hdf5File::MatrixDataType::kFloat,
                                 params.getCompressionLevel());

    // Write dataset parameters
  mFile.writeMatrixDomainType(mFile.getRootGroup(), mRootObjectName, Hdf5File::MatrixDomainType::kReal);
  mFile.writeMatrixDataType  (mFile.getRootGroup(), mRootObjectName, Hdf5File::MatrixDataType::kFloat);

  // Sampled time step
  mSampledTimeStep = 0;

  // Set buffer size
  mSize = nSampledElementsPerStep;

  // Allocate memory
  allocateMemory();
}// end of create
//----------------------------------------------------------------------------------------------------------------------

/**
 * Reopen the output stream after restart.
 *
 */
void IndexOutputStream::reopen()
{
  // Get parameters
  const Parameters& params = Parameters::getInstance();

  // Set buffer size
  mSize = mSensorMask.size();

  // Allocate memory
   allocateMemory();

  // Reopen the dataset
  mDataset = mFile.openDataset(mFile.getRootGroup(), mRootObjectName);


  if (mReduceOp == ReduceOperator::kNone)
  { // raw time series - just seek to the right place in the dataset
    mSampledTimeStep = (params.getTimeIndex() < params.getSamplingStartTimeIndex()) ?
                        0 : (params.getTimeIndex() - params.getSamplingStartTimeIndex());

  }
  else
  { // aggregated quantities - reload data
    mSampledTimeStep = 0;

    // Read data from disk only if there were anything stored there (t_index >= start_index)
    if (Parameters::getInstance().getTimeIndex() > Parameters::getInstance().getSamplingStartTimeIndex())
    {
      // Since there is only a single timestep in the dataset, I can read the whole dataset
      mFile.readCompleteDataset(mFile.getRootGroup(),
                                mRootObjectName,
                                DimensionSizes(mSize, 1, 1),
                                mHostBuffer);

      // Send data to device
      copyToDevice();
    }
  }
}// end of reopen
//---------------------------------------------------------------------------------------------------------------------

/**
 * Sample grid points, line them up in the buffer, if necessary a reduce operator is applied.
 */
void IndexOutputStream::sample()
{
  switch (mReduceOp)
  {
    case ReduceOperator::kNone:
    {
      OutputStreamsCudaKernels::sampleIndex<ReduceOperator::kNone>
                                           (mDeviceBuffer,
                                            mSourceMatrix.getDeviceData(),
                                            mSensorMask.getDeviceData(),
                                            mSensorMask.size());

      // Record an event when the data has been copied over.
      cudaCheckErrors(cudaEventRecord(mEventSamplingFinished));

      break;
    }// case kNone

    case ReduceOperator::kRms:
    {
      OutputStreamsCudaKernels::sampleIndex<ReduceOperator::kRms>
                                           (mDeviceBuffer,
                                            mSourceMatrix.getDeviceData(),
                                            mSensorMask.getDeviceData(),
                                            mSensorMask.size());

      break;
    }// case kRms

    case ReduceOperator::kMax:
    {
      OutputStreamsCudaKernels::sampleIndex<ReduceOperator::kMax>
                                           (mDeviceBuffer,
                                            mSourceMatrix.getDeviceData(),
                                            mSensorMask.getDeviceData(),
                                            mSensorMask.size());
      break;
    }// case kMax

    case ReduceOperator::kMin:
    {
      OutputStreamsCudaKernels::sampleIndex<ReduceOperator::kMin>
                                           (mDeviceBuffer,
                                            mSourceMatrix.getDeviceData(),
                                            mSensorMask.getDeviceData(),
                                            mSensorMask.size());
      break;
    } //case kMin
  }// switch
}// end of sample
//----------------------------------------------------------------------------------------------------------------------

/**
 * Flush data for the timestep. Only applicable on RAW data series.
 */
void IndexOutputStream::flushRaw()
{
  if (mReduceOp == ReduceOperator::kNone)
  {
    // make sure the data has been copied from the GPU
    cudaEventSynchronize(mEventSamplingFinished);

    // only raw time series are flushed down to the disk every time step
    flushBufferToFile();
  }
}// end of FlushRaw
//--------------------------------------------------------------------------------------------------

/**
 * Apply post-processing on the buffer and flush it to the file.
 */
void IndexOutputStream::postProcess()
{
  // run inherited method
  BaseOutputStream::postProcess();

  // When no reduction operator is applied, the data is flushed after every time step
  // which means it has been done before
  if (mReduceOp != ReduceOperator::kNone)
  {
    // Copy data from GPU matrix
    copyFromDevice();
    // flush to disk
    flushBufferToFile();
  }
}// end of postProcess
//----------------------------------------------------------------------------------------------------------------------

/**
 * Checkpoint the stream and close.
 */
void IndexOutputStream::checkpoint()
{
  // raw data has already been flushed, others has to be flushed here
  if (mReduceOp != ReduceOperator::kNone)
  {
    // copy data from the device
    copyFromDevice();
    // flush to disk
    flushBufferToFile();
  }
}// end of checkpoint
//----------------------------------------------------------------------------------------------------------------------

/**
 * Close stream (apply post-processing if necessary, flush data and close).
 */
void IndexOutputStream::close()
{
  // the dataset is still opened
  if (mDataset != H5I_BADID)
  {
    mFile.closeDataset(mDataset);
  }

  mDataset = H5I_BADID;
}// end of close
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Flush the buffer down to the file at the actual position
 */
void IndexOutputStream::flushBufferToFile()
{
  mFile.writeHyperSlab(mDataset,
                       DimensionSizes(0,mSampledTimeStep,0),
                       DimensionSizes(mSize,1,1),
                       mHostBuffer);
  mSampledTimeStep++;
}// end of flushToFile
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

