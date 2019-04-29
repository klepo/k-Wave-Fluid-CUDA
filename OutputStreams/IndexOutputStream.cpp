/**
 * @file      IndexOutputStream.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file of the class saving data based on index senor mask into
 *            the output HDF5 file.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      29 August    2014, 10:10 (created) \n
 *            06 March     2019, 13:19 (revised)
 *
 * @copyright Copyright (C) 2019 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the C++ extension of the [k-Wave Toolbox](http://www.k-wave.org).
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify it under the terms
 * of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
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

  allocateMinMaxMemory(1);
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

  std::string objectName = mRootObjectName;

  // Derive dataset dimension sizes
  DimensionSizes datasetSize(nSampledElementsPerStep, 1, 1);
  if (mReduceOp == ReduceOperator::kNone)
  {
    datasetSize = DimensionSizes(nSampledElementsPerStep,
                                 params.getNt() - params.getSamplingStartTimeIndex(),
                                 1);
  }
  else if (mReduceOp == ReduceOperator::kC)
  {
    // Extend "x" dimension for copression coefficients
    nSampledElementsPerStep *= mCompressHelper->getHarmonics() * 2;
    size_t steps = params.getNt() - params.getSamplingStartTimeIndex();

    // TODO minimal number of steps for compression
    size_t compressedSteps = size_t(std::max(float(floor(float(steps) / mCompressHelper->getOSize())) - 1, 1.0f));
    datasetSize = DimensionSizes(nSampledElementsPerStep, compressedSteps, 1);

    objectName = objectName + "_c";
  }

  // Set HDF5 chunk size
  DimensionSizes chunkSize(nSampledElementsPerStep, 1, 1);
  // for data bigger than 32 MB
  if (nSampledElementsPerStep > (kChunkSize4MB * 8))
  {
      chunkSize.nx = kChunkSize4MB; // set chunk size to MB
  }

  // Create a dataset under the root group
  mDataset = mFile.createDataset(mFile.getRootGroup(),
                                 objectName,
                                 datasetSize,
                                 chunkSize,
                                 Hdf5File::MatrixDataType::kFloat,
                                 params.getCompressionLevel());

    // Write dataset parameters
  mFile.writeMatrixDomainType(mFile.getRootGroup(), objectName, Hdf5File::MatrixDomainType::kReal);
  mFile.writeMatrixDataType  (mFile.getRootGroup(), objectName, Hdf5File::MatrixDataType::kFloat);

  // Write compression parameters as attributes
  if (mReduceOp == ReduceOperator::kC)
  {
    mFile.writeLongLongAttribute(mFile.getRootGroup(), objectName, "c_harmonics", ssize_t(mCompressHelper->getHarmonics()));
    mFile.writeStringAttribute(mFile.getRootGroup(), objectName, "c_type", "c");
    mFile.writeFloatAttribute(mFile.getRootGroup(), objectName, "c_period", mCompressHelper->getPeriod());
    mFile.writeLongLongAttribute(mFile.getRootGroup(), objectName, "c_mos", ssize_t(mCompressHelper->getMos()));
    mFile.writeStringAttribute(mFile.getRootGroup(), objectName, "src_dataset_name", mRootObjectName);
  }

  // Sampled time step
  mSampledTimeStep = 0;

  // Set buffer size
  mSize = mSensorMask.size();
  mCSize = nSampledElementsPerStep;

  // Allocate memory
  allocateMemory();
}// end of create
//----------------------------------------------------------------------------------------------------------------------

/**
 * Reopen the output stream after restart.
 */
void IndexOutputStream::reopen()
{
  // Get parameters
  const Parameters& params = Parameters::getInstance();

  // Set buffer size
  mSize = mSensorMask.size();

  std::string objectName = mRootObjectName;

  if (mReduceOp == ReduceOperator::kC)
  {
    mCSize = mSize * mCompressHelper->getHarmonics() * 2;
    objectName = objectName + "_c";
  }

  // Allocate memory
   allocateMemory();

  // Reopen the dataset
  mDataset = mFile.openDataset(mFile.getRootGroup(), objectName);

  if (mReduceOp == ReduceOperator::kNone || mReduceOp == ReduceOperator::kC)
  { // raw time series - just seek to the right place in the dataset
    mSampledTimeStep = (params.getTimeIndex() < params.getSamplingStartTimeIndex()) ?
                        0 : (params.getTimeIndex() - params.getSamplingStartTimeIndex());

    if (mReduceOp == ReduceOperator::kC)
    {
      mCompressedTimeStep = size_t(std::max(float(floor(float(mSampledTimeStep) / mCompressHelper->getOSize())), 0.0f));
    }
  }
  else
  { // aggregated quantities - reload data
    mSampledTimeStep = 0;

    // Read data from disk only if there were anything stored there (t_index >= start_index)
    if (params.getTimeIndex() > params.getSamplingStartTimeIndex())
    {
      // Since there is only a single timestep in the dataset, I can read the whole dataset
      mFile.readCompleteDataset(mFile.getRootGroup(),
                                objectName,
                                DimensionSizes(mSize, 1, 1),
                                mHostBuffer);

      // Send data to device
      copyToDevice();
    }
  }

  if (params.getTimeIndex() > params.getSamplingStartTimeIndex())
  {
    // Reload temp coefficients from checkpoint file
    loadCheckpointCompressionCoefficients();

    // Reload min and max values
    const std::string datasetName = (mReduceOp == ReduceOperator::kC) ? mRootObjectName + "_c" : mRootObjectName;
    loadMinMaxValues(mFile, mFile.getRootGroup(), datasetName);
  }
}// end of reopen
//----------------------------------------------------------------------------------------------------------------------

/**
 * Sample grid points, line them up in the buffer, if necessary a reduce operator is applied.
 */
void IndexOutputStream::sample()
{
  switch (mReduceOp)
  {
    case ReduceOperator::kNone:
    case ReduceOperator::kC:
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

    for (ssize_t i = 0; i < ssize_t(mSensorMask.size()); i++)
    {
      checkOrSetMinMaxValue(minValue[0], maxValue[0], mHostBuffer[i], minValueIndex[0], maxValueIndex[0], mSensorMask.size() * mSampledTimeStep + i);
    }

    // only raw time series are flushed down to the disk every time step
    flushBufferToFile();
  }
  if (mReduceOp == ReduceOperator::kC)
  {
    // make sure the data has been copied from the GPU
    cudaEventSynchronize(mEventSamplingFinished);

    // Compression
    // Compute local index and flags
    mStepLocal = (mSampledTimeStep) % (mCompressHelper->getBSize() - 1);
    mSavingFlag = ((mStepLocal + 1) % mCompressHelper->getOSize() == 0) ? true : false;
    mOddFrameFlag = ((mCompressedTimeStep + 1) % 2 == 0) ? true : false;

    // For every point
    #pragma omp parallel for
    for (ssize_t i = 0; i < ssize_t(mSensorMask.size()); i++)
    {
      // Tady je někde problém
      checkOrSetMinMaxValue(minValue[0], maxValue[0], mHostBuffer[i], minValueIndex[0], maxValueIndex[0], mSensorMask.size() * mSampledTimeStep + i);
      size_t offset = mCompressHelper->getHarmonics() * i;

      //For every harmonics
      for (size_t ih = 0; ih < mCompressHelper->getHarmonics(); ih++)
      {
          size_t pH = offset + ih;
          size_t bIndex = ih * mCompressHelper->getBSize() + mStepLocal;

          // Correlation step
          reinterpret_cast<floatC *>(mHostBuffer1)[pH] += mCompressHelper->getBE()[bIndex] * mHostBuffer[i];
          reinterpret_cast<floatC *>(mHostBuffer2)[pH] += mCompressHelper->getBE_1()[bIndex] * mHostBuffer[i];
        }
    }

    if (mSavingFlag)
    {
      // Select accumulated value
      float *data = mOddFrameFlag ? mHostBuffer1 : mHostBuffer2;

      // Store selected buffer
      if (mCompressedTimeStep > 0)
      {
        flushBufferToFile(data);
      }

      // Set zeros for next accumulation
      //memset(data, 0, mBufferSize * sizeof(float));
      #pragma omp parallel for
      for (ssize_t i = 0; i < ssize_t(mCSize); i++)
      {
        data[i] = 0.0f;
      }
      mCompressedTimeStep++;
    }
    mSampledTimeStep++;
  }
}// end of flushRaw
//----------------------------------------------------------------------------------------------------------------------

/**
 * Apply post-processing on the buffer and flush it to the file.
 */
void IndexOutputStream::postProcess()
{
  // run inherited method
  BaseOutputStream::postProcess();

  // When no reduction operator is applied, the data is flushed after every time step
  // which means it has been done before
  if (mReduceOp != ReduceOperator::kNone && mReduceOp != ReduceOperator::kC)
  {
    // Copy data from GPU matrix
    copyFromDevice();
    // flush to disk
    flushBufferToFile();
  }

  // Store min and max values
  const std::string datasetName = (mReduceOp == ReduceOperator::kC) ? mRootObjectName + "_c" : mRootObjectName;
  storeMinMaxValues(mFile, mFile.getRootGroup(), datasetName);
}// end of postProcess
//----------------------------------------------------------------------------------------------------------------------

/**
 * Checkpoint the stream and close.
 */
void IndexOutputStream::checkpoint()
{
  // raw data has already been flushed, others has to be flushed here
  if (mReduceOp != ReduceOperator::kNone && mReduceOp != ReduceOperator::kC)
  {
    // copy data from the device
    copyFromDevice();
    // flush to disk
    flushBufferToFile();
  }
  storeCheckpointCompressionCoefficients();

  // Store min and max values
  const std::string datasetName = (mReduceOp == ReduceOperator::kC) ? mRootObjectName + "_c" : mRootObjectName;
  storeMinMaxValues(mFile, mFile.getRootGroup(), datasetName);
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
void IndexOutputStream::flushBufferToFile(float *bufferToFlush)
{
  mFile.writeHyperSlab(mDataset,
                       DimensionSizes(0, (mReduceOp == ReduceOperator::kC) ? mCompressedTimeStep - 1 : mSampledTimeStep, 0),
                       DimensionSizes((mReduceOp == ReduceOperator::kC) ? mCSize : mSize, 1, 1),
                       (bufferToFlush != nullptr) ? bufferToFlush : mHostBuffer);
  if (mReduceOp != ReduceOperator::kC) mSampledTimeStep++;
}// end of flushToFile
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

