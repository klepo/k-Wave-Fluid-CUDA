/**
 * @file      CuboidOutputStream.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file of classes responsible for storing output quantities based
 *            on the cuboid sensor mask into the output HDF5 file.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      13 February  2015, 12:51 (created) \n
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

#include <OutputStreams/CuboidOutputStream.h>
#include <OutputStreams/OutputStreamsCudaKernels.cuh>

#include <Parameters/Parameters.h>
#include <Logger/Logger.h>

using std::string;

//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Constants -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor.
 */
CuboidOutputStream::CuboidOutputStream(Hdf5File&            file,
                                       MatrixName&          groupName,
                                       const RealMatrix&    sourceMatrix,
                                       const IndexMatrix&   sensorMask,
                                       const ReduceOperator reduceOp)
  : BaseOutputStream(file, groupName, sourceMatrix, reduceOp),
    mSensorMask(sensorMask),
    mGroup(H5I_BADID),
    mSampledTimeStep(0),
    mEventSamplingFinished()
{
  // Create event for sampling
  cudaCheckErrors(cudaEventCreate(&mEventSamplingFinished));

  allocateMinMaxMemory(mSensorMask.getDimensionSizes().ny);
}// end of CuboidOutputStream
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor.
 */
CuboidOutputStream::~CuboidOutputStream()
{
  // Destroy sampling event
  cudaCheckErrors(cudaEventDestroy(mEventSamplingFinished));
  // Close the stream
  close();
  // free memory
  freeMemory();
}// end ~CuboidOutputStream
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create a HDF5 stream and allocate data for it. It also creates a HDF5 group with particular
 * datasets (one per cuboid).
 */
void CuboidOutputStream::create()
{
  // Create and open or only open the HDF5 group
  if (mFile.groupExists(mFile.getRootGroup(), mRootObjectName))
  {
    mGroup = mFile.openGroup(mFile.getRootGroup(), mRootObjectName);
  }
  else
  {
    mGroup = mFile.createGroup(mFile.getRootGroup(), mRootObjectName);
  }

  // Create all datasets (sizes, chunks, and attributes)
  size_t nCuboids = mSensorMask.getDimensionSizes().ny;
  mCuboidsInfo.reserve(nCuboids);
  size_t actualPositionInBuffer = 0;

  for (size_t cuboidIdx = 0; cuboidIdx < nCuboids; cuboidIdx++)
  {
    CuboidInfo cuboidInfo;

    cuboidInfo.cuboidIdx = createCuboidDataset(cuboidIdx);
    cuboidInfo.startingPossitionInBuffer = actualPositionInBuffer;
    mCuboidsInfo.push_back(cuboidInfo);

    if (mReduceOp == ReduceOperator::kC)
    {
      actualPositionInBuffer += (mSensorMask.getBottomRightCorner(cuboidIdx) -
                                 mSensorMask.getTopLeftCorner(cuboidIdx)
                                ).nElements() * mCompressHelper->getHarmonics() * 2;
    }
    else
    {
      actualPositionInBuffer += (mSensorMask.getBottomRightCorner(cuboidIdx) -
                                 mSensorMask.getTopLeftCorner(cuboidIdx)
                                ).nElements();
    }
  }

  //we're at the beginning
  mSampledTimeStep = 0;

  // Set buffer size
  if (mReduceOp == ReduceOperator::kC)
  {
    mCSize = mSensorMask.getSizeOfAllCuboids() * mCompressHelper->getHarmonics() * 2;
    mSize = mSensorMask.getSizeOfAllCuboids();
  }
  else
  {
    mSize = mSensorMask.getSizeOfAllCuboids();
  }

  // Allocate memory
  allocateMemory();
}// end of create
//----------------------------------------------------------------------------------------------------------------------

/**
 * Reopen the output stream after restart and reload data.
 */
void CuboidOutputStream::reopen()
{
  // Get parameters
  const Parameters& params = Parameters::getInstance();

  mSampledTimeStep = 0;
  if (mReduceOp == ReduceOperator::kNone || mReduceOp == ReduceOperator::kC) // set correct sampled tim estep for raw data series
  {
    mSampledTimeStep = (params.getTimeIndex() < params.getSamplingStartTimeIndex()) ?
                        0 : (params.getTimeIndex() - params.getSamplingStartTimeIndex());
    if (mReduceOp == ReduceOperator::kC)
    {
      mCompressedTimeStep = size_t(std::max(float(floor(float(mSampledTimeStep) / mCompressHelper->getOSize())), 0.0f));
    }
  }

  // Create the memory buffer if necessary and set starting address
  if (mReduceOp == ReduceOperator::kC)
  {
    mCSize = mSensorMask.getSizeOfAllCuboids() * mCompressHelper->getHarmonics() * 2;
    mSize = mSensorMask.getSizeOfAllCuboids();
  }
  else
  {
    mSize = mSensorMask.getSizeOfAllCuboids();
  }

  // Allocate memory if needed
  allocateMemory();

  // Open all datasets (sizes, chunks, and attributes)
  size_t nCuboids = mSensorMask.getDimensionSizes().ny;
  mCuboidsInfo.reserve(nCuboids);
  size_t actualPositionInBuffer = 0;

  // Open the HDF5 group
  mGroup = mFile.openGroup(mFile.getRootGroup(), mRootObjectName);

  for (size_t cuboidIdx = 0; cuboidIdx < nCuboids; cuboidIdx++)
  {
    CuboidInfo cuboidInfo;

    // Indexed from 1
    const std::string datasetName = (mReduceOp == ReduceOperator::kC) ? std::to_string(cuboidIdx + 1) + "_c" : std::to_string(cuboidIdx + 1);

    // open the dataset
    cuboidInfo.cuboidIdx = mFile.openDataset(mGroup,datasetName);
    cuboidInfo.startingPossitionInBuffer = actualPositionInBuffer;
    mCuboidsInfo.push_back(cuboidInfo);

    // read only if there is anything to read
    if (params.getTimeIndex() > params.getSamplingStartTimeIndex())
    {
      if (mReduceOp != ReduceOperator::kNone && mReduceOp != ReduceOperator::kC)
      { // Reload data
        DimensionSizes cuboidSize((mSensorMask.getBottomRightCorner(cuboidIdx) -
                                   mSensorMask.getTopLeftCorner(cuboidIdx)).nx,
                                  (mSensorMask.getBottomRightCorner(cuboidIdx) -
                                   mSensorMask.getTopLeftCorner(cuboidIdx)).ny,
                                  (mSensorMask.getBottomRightCorner(cuboidIdx) -
                                   mSensorMask.getTopLeftCorner(cuboidIdx)).nz);

        mFile.readCompleteDataset(mGroup,
                                  datasetName,
                                  cuboidSize,
                                  mHostBuffer + actualPositionInBuffer);
      }
    }
    // move the pointer for the next cuboid beginning (this inits the locations)
    if (mReduceOp == ReduceOperator::kC)
    {
      actualPositionInBuffer += (mSensorMask.getBottomRightCorner(cuboidIdx) -
                                 mSensorMask.getTopLeftCorner(cuboidIdx)
                                ).nElements() * mCompressHelper->getHarmonics() * 2;
    }
    else
    {
      actualPositionInBuffer += (mSensorMask.getBottomRightCorner(cuboidIdx) -
                                 mSensorMask.getTopLeftCorner(cuboidIdx)
                                ).nElements();
    }
  }

  // copy data over to the GPU only if there is anything to read
  if (params.getTimeIndex() > params.getSamplingStartTimeIndex())
  {
    copyToDevice();

    // Reload temp coefficients from checkpoint file
    loadCheckpointCompressionCoefficients();

    // Reload min and max values
    for (size_t cuboidIdx = 0; cuboidIdx < mCuboidsInfo.size(); cuboidIdx++)
    {
      const std::string datasetName = (mReduceOp == ReduceOperator::kC) ? std::to_string(cuboidIdx + 1) + "_c" : std::to_string(cuboidIdx + 1);
      //Logger::log(Logger::LogLevel::kBasic, datasetName + " ");
      //Logger::log(Logger::LogLevel::kBasic, std::to_string(minValue[cuboidIdx]));
      loadMinMaxValues(mFile, mGroup, datasetName, cuboidIdx);
    }
  }

}// end of reopen
//----------------------------------------------------------------------------------------------------------------------

/**
 * Sample grid points, line them up in the buffer, if necessary a reduce operator is applied.
 */
void CuboidOutputStream::sample()
{
  size_t cuboidInBufferStart = 0;
  // dimension sizes of the matrix being sampled
  const dim3 dimSizes (static_cast<unsigned int>(mSourceMatrix.getDimensionSizes().nx),
                       static_cast<unsigned int>(mSourceMatrix.getDimensionSizes().ny),
                       static_cast<unsigned int>(mSourceMatrix.getDimensionSizes().nz));

  // Run over all cuboids - this is not a good solution as we need to run a distinct kernel for a cuboid
  for (size_t cuboidIdx = 0; cuboidIdx < mCuboidsInfo.size(); cuboidIdx++)
  {
    // copy down dim sizes
    const dim3 topLeftCorner(static_cast<unsigned int>(mSensorMask.getTopLeftCorner(cuboidIdx).nx),
                             static_cast<unsigned int>(mSensorMask.getTopLeftCorner(cuboidIdx).ny),
                             static_cast<unsigned int>(mSensorMask.getTopLeftCorner(cuboidIdx).nz));
    const dim3 bottomRightCorner(static_cast<unsigned int>(mSensorMask.getBottomRightCorner(cuboidIdx).nx),
                                 static_cast<unsigned int>(mSensorMask.getBottomRightCorner(cuboidIdx).ny),
                                 static_cast<unsigned int>(mSensorMask.getBottomRightCorner(cuboidIdx).nz));

    //get number of samples within the cuboid
    const size_t nSamples = (mSensorMask.getBottomRightCorner(cuboidIdx) -
                             mSensorMask.getTopLeftCorner(cuboidIdx)
                            ).nElements();

    switch (mReduceOp)
    {
      case ReduceOperator::kNone:
      case ReduceOperator::kC:
      {
        // Kernel to sample raw quantities inside one cuboid
        OutputStreamsCudaKernels::sampleCuboid<ReduceOperator::kNone>
                                              (mDeviceBuffer + cuboidInBufferStart,
                                               mSourceMatrix.getDeviceData(),
                                               topLeftCorner,
                                               bottomRightCorner,
                                               dimSizes,
                                               nSamples);
        break;
      }
      case ReduceOperator::kRms:
      {
        OutputStreamsCudaKernels::sampleCuboid<ReduceOperator::kRms>
                                              (mDeviceBuffer + cuboidInBufferStart,
                                               mSourceMatrix.getDeviceData(),
                                               topLeftCorner,
                                               bottomRightCorner,
                                               dimSizes,
                                               nSamples);
        break;
      }
      case ReduceOperator::kMax:
      {
        OutputStreamsCudaKernels::sampleCuboid<ReduceOperator::kMax>
                                              (mDeviceBuffer + cuboidInBufferStart,
                                               mSourceMatrix.getDeviceData(),
                                               topLeftCorner,
                                               bottomRightCorner,
                                               dimSizes,
                                               nSamples);
        break;
      }
      case ReduceOperator::kMin:
      {
        OutputStreamsCudaKernels::sampleCuboid<ReduceOperator::kMin>
                                              (mDeviceBuffer + cuboidInBufferStart,
                                               mSourceMatrix.getDeviceData(),
                                               topLeftCorner,
                                               bottomRightCorner,
                                               dimSizes,
                                               nSamples);
        break;
      }
    }

    cuboidInBufferStart += nSamples;
  }

  if (mReduceOp == ReduceOperator::kNone)
  {
    // Record an event when the data has been copied over.
    cudaCheckErrors(cudaEventRecord(mEventSamplingFinished));
  }
}// end of sample
//----------------------------------------------------------------------------------------------------------------------

/**
 * Flush data for the timestep. Only applicable on RAW data series.
 */
void CuboidOutputStream::flushRaw()
{
  // Size of the cuboid
  DimensionSizes cuboidSize(0, 0, 0, 0);

  if (mReduceOp == ReduceOperator::kNone)
  {
    // make sure the data has been copied from the GPU
    cudaEventSynchronize(mEventSamplingFinished);

    // Find min/max value
    for (size_t cuboidIdx = 0; cuboidIdx < mCuboidsInfo.size(); cuboidIdx++)
    {
      const DimensionSizes topLeftCorner     = mSensorMask.getTopLeftCorner(cuboidIdx);
      const DimensionSizes bottomRightCorner = mSensorMask.getBottomRightCorner(cuboidIdx);
      cuboidSize    = bottomRightCorner - topLeftCorner;
      cuboidSize.nt = 1;
      size_t cuboidStartIndex = mCuboidsInfo[cuboidIdx].startingPossitionInBuffer;
      for (size_t i = cuboidStartIndex; i < cuboidStartIndex + cuboidSize.nElements(); i++)
      {
        checkOrSetMinMaxValue(minValue[cuboidIdx], maxValue[cuboidIdx], mHostBuffer[i], minValueIndex[cuboidIdx], maxValueIndex[cuboidIdx], cuboidSize.nElements() * mSampledTimeStep + i);
      }
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

    // For every cuboid/point
    for (size_t cuboidIdx = 0; cuboidIdx < mCuboidsInfo.size(); cuboidIdx++)
    {
      const DimensionSizes topLeftCorner     = mSensorMask.getTopLeftCorner(cuboidIdx);
      const DimensionSizes bottomRightCorner = mSensorMask.getBottomRightCorner(cuboidIdx);
      cuboidSize    = bottomRightCorner - topLeftCorner;
      cuboidSize.nt = 1;
      size_t cuboidStartIndex = mCuboidsInfo[cuboidIdx].startingPossitionInBuffer;
      for (size_t i = cuboidStartIndex; i < cuboidStartIndex + cuboidSize.nElements(); i++)
      {
        checkOrSetMinMaxValue(minValue[cuboidIdx], maxValue[cuboidIdx], mHostBuffer[i], minValueIndex[cuboidIdx], maxValueIndex[cuboidIdx], cuboidSize.nElements() * mSampledTimeStep + i);

        size_t offset = mCompressHelper->getHarmonics() * i;

        //For every harmonics
        for (size_t ih = 0; ih < mCompressHelper->getHarmonics(); ih++)
        {
          size_t pH = offset + ih;
          size_t bIndex = ih * mCompressHelper->getBSize() + mStepLocal;

          // Correlation step
          reinterpret_cast<floatC *>(mHostBuffer1)[pH] += mCompressHelper->getBE()[bIndex]* mHostBuffer[i];
          reinterpret_cast<floatC *>(mHostBuffer2)[pH] += mCompressHelper->getBE_1()[bIndex]* mHostBuffer[i];
        }
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


/*
 * Apply post-processing on the buffer and flush it to the file.
 */
void CuboidOutputStream::postProcess()
{
  // run inherited method
  BaseOutputStream::postProcess();

  // When no reduce operator is applied, the data is flushed after every time step
  // which means it has been done before
  if (mReduceOp != ReduceOperator::kNone && mReduceOp != ReduceOperator::kC)
  {
    // Copy data from GPU matrix
    copyFromDevice();

    flushBufferToFile();
  }

  // Store min and max values
  for (size_t cuboidIdx = 0; cuboidIdx < mCuboidsInfo.size(); cuboidIdx++)
  {
    const std::string datasetName = (mReduceOp == ReduceOperator::kC) ? std::to_string(cuboidIdx + 1) + "_c" : std::to_string(cuboidIdx + 1);
    //Logger::log(Logger::LogLevel::kBasic, datasetName + " ");
    //Logger::log(Logger::LogLevel::kBasic, std::to_string(minValue[cuboidIdx]));
    storeMinMaxValues(mFile, mGroup, datasetName, cuboidIdx);
  }
}// end of postProcess
//----------------------------------------------------------------------------------------------------------------------

/**
 * Checkpoint the stream and close.
 */
void CuboidOutputStream::checkpoint()
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
  for (size_t cuboidIdx = 0; cuboidIdx < mCuboidsInfo.size(); cuboidIdx++)
  {
    const std::string datasetName = (mReduceOp == ReduceOperator::kC) ? std::to_string(cuboidIdx + 1) + "_c" : std::to_string(cuboidIdx + 1);
    //Logger::log(Logger::LogLevel::kBasic, datasetName + " ");
    //Logger::log(Logger::LogLevel::kBasic, std::to_string(minValue[cuboidIdx]));
    storeMinMaxValues(mFile, mGroup, datasetName, cuboidIdx);
  }
}// end of checkpoint
//----------------------------------------------------------------------------------------------------------------------

/**
 * Close stream (apply post-processing if necessary, flush data, close datasets and the group).
 */
void CuboidOutputStream::close()
{
  // the group is still open
  if (mGroup != H5I_BADID)
  {
    // Close all datasets and the group
    for (size_t cuboidIdx = 0; cuboidIdx < mCuboidsInfo.size(); cuboidIdx++)
    {
      mFile.closeDataset(mCuboidsInfo[cuboidIdx].cuboidIdx);
    }
    mCuboidsInfo.clear();

    mFile.closeGroup(mGroup);
    mGroup = H5I_BADID;
  }// if opened
}// end of close
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Create a new dataset for a given cuboid specified by index (order).
 */
hid_t CuboidOutputStream::createCuboidDataset(const size_t cuboidIdx)
{
  const Parameters& params = Parameters::getInstance();

  // if time series then Number of steps else 1
  size_t nSampledTimeSteps = (mReduceOp == ReduceOperator::kNone)
                             ? params.getNt() - params.getSamplingStartTimeIndex()
                             : 0; // will be a 3D dataset

  // Set cuboid dimensions (subtract two corners (add 1) and use the appropriate component)
  DimensionSizes cuboidSize((mSensorMask.getBottomRightCorner(cuboidIdx) - mSensorMask.getTopLeftCorner(cuboidIdx)).nx,
                            (mSensorMask.getBottomRightCorner(cuboidIdx) - mSensorMask.getTopLeftCorner(cuboidIdx)).ny,
                            (mSensorMask.getBottomRightCorner(cuboidIdx) - mSensorMask.getTopLeftCorner(cuboidIdx)).nz,
                             nSampledTimeSteps);

  if (mReduceOp == ReduceOperator::kC)
  {
    size_t steps = params.getNt() - params.getSamplingStartTimeIndex();
    nSampledTimeSteps = size_t(std::max(float(floor(float(steps) / mCompressHelper->getOSize())) - 1, 1.0f));

    cuboidSize.nt = nSampledTimeSteps;
    cuboidSize.nx *= mCompressHelper->getHarmonics() * 2;
  }

  // Set chunk size
  // If the size of the cuboid is bigger than 32 MB per timestep, set the chunk to approx 4MB
  size_t nSlabs = 1; //at least one slab
  DimensionSizes cuboidChunkSize(cuboidSize.nx,
                                 cuboidSize.ny,
                                 cuboidSize.nz,
                                 (mReduceOp == ReduceOperator::kNone || mReduceOp == ReduceOperator::kC) ? 1 : 0);

  if (cuboidChunkSize.nElements() > (kChunkSize4MB * 8))
  {
    while (nSlabs * cuboidSize.nx * cuboidSize.ny < kChunkSize4MB) nSlabs++;
    cuboidChunkSize.nz = nSlabs;
  }

  // Indexed from 1
  const std::string datasetName = (mReduceOp == ReduceOperator::kC) ? std::to_string(cuboidIdx + 1) + "_c" : std::to_string(cuboidIdx + 1);

  hid_t dataset = mFile.createDataset(mGroup,
                                      datasetName.c_str(),
                                      cuboidSize,
                                      cuboidChunkSize,
                                      Hdf5File::MatrixDataType::kFloat,
                                      params.getCompressionLevel());

  // Write dataset parameters
  mFile.writeMatrixDomainType(mGroup, datasetName.c_str(), Hdf5File::MatrixDomainType::kReal);
  mFile.writeMatrixDataType  (mGroup, datasetName.c_str(), Hdf5File::MatrixDataType::kFloat);

  // Write compression parameters as attributes
  if (mReduceOp == ReduceOperator::kC)
  {
    mFile.writeLongLongAttribute(mGroup, datasetName, "c_harmonics", ssize_t(mCompressHelper->getHarmonics()));
    mFile.writeStringAttribute(mGroup, datasetName, "c_type", "c");
    mFile.writeFloatAttribute(mGroup, datasetName, "c_period", mCompressHelper->getPeriod());
    mFile.writeLongLongAttribute(mGroup, datasetName, "c_mos", ssize_t(mCompressHelper->getMos()));
    mFile.writeStringAttribute(mGroup, datasetName, "src_dataset_name", "/" + mRootObjectName + "/" + std::to_string(cuboidIdx + 1));
  }

  return dataset;
}//end of createCuboidDatasets
//----------------------------------------------------------------------------------------------------------------------

/**
 * Flush the buffer to the file (to multiple datasets if necessary).
 */
void CuboidOutputStream::flushBufferToFile(float *bufferToFlush)
{
  DimensionSizes position (0, 0, 0, 0);
  DimensionSizes blockSize(0, 0, 0, 0);

  if (mReduceOp == ReduceOperator::kNone) position.nt = mSampledTimeStep;
  if (mReduceOp == ReduceOperator::kC) position.nt = mCompressedTimeStep - 1;

  for (size_t cuboidIdx = 0; cuboidIdx < mCuboidsInfo.size(); cuboidIdx++)
  {
    blockSize = mSensorMask.getBottomRightCorner(cuboidIdx) - mSensorMask.getTopLeftCorner(cuboidIdx);
    if (mReduceOp == ReduceOperator::kC) blockSize.nx *= mCompressHelper->getHarmonics() * 2;
    blockSize.nt = 1;

    mFile.writeHyperSlab(mCuboidsInfo[cuboidIdx].cuboidIdx,
                        position,
                        blockSize,
                         ((bufferToFlush != nullptr) ? bufferToFlush : mHostBuffer) + mCuboidsInfo[cuboidIdx].startingPossitionInBuffer);
  }

  if (mReduceOp != ReduceOperator::kC) mSampledTimeStep++;
}// end of flushBufferToFile
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//
