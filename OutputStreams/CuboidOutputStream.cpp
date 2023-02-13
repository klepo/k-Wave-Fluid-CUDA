/**
 * @file      CuboidOutputStream.cpp
 *
 * @author    Jiri Jaros, Petr Kleparnik \n
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
 *            08 February  2023, 12:00 (revised)
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

#include <algorithm>

#include <OutputStreams/CuboidOutputStream.h>
#include <OutputStreams/OutputStreamsCudaKernels.cuh>
#include <Parameters/Parameters.h>
#include <Containers/OutputStreamContainer.h>

//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Constants -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor.
 */
CuboidOutputStream::CuboidOutputStream(Hdf5File& file,
  MatrixName& groupName,
  const RealMatrix& sourceMatrix,
  const IndexMatrix& sensorMask,
  const ReduceOperator reduceOp,
  OutputStreamContainer* outputStreamContainer,
  bool doNotSaveFlag)
  : BaseOutputStream(file, groupName, sourceMatrix, reduceOp, outputStreamContainer, doNotSaveFlag),
    mSensorMask(sensorMask), mGroup(H5I_BADID), mSampledTimeStep(0), mEventSamplingFinished()
{
  // Create event for sampling
  cudaCheckErrors(cudaEventCreate(&mEventSamplingFinished));
} // end of CuboidOutputStream
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
} // end ~CuboidOutputStream
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create a HDF5 stream and allocate data for it. It also creates a HDF5 group with particular datasets (one per
 * cuboid).
 */
void CuboidOutputStream::create()
{
  // Set buffer size
  // Extend "x" dimension for compression coefficients
  mSize = (mReduceOp == ReduceOperator::kC)
            ? mSensorMask.getSizeOfAllCuboids(mCompressHelper->getHarmonics() * mComplexSize)
            : mSensorMask.getSizeOfAllCuboids();
  if (mReduceOp == ReduceOperator::kC)
  {
    mOSize = mSensorMask.getSizeOfAllCuboids();
  }

  // Don't create or open group for compression coefficients if only kIAvgC or kQTermC should be stored
  if (!mDoNotSaveFlag)
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
  }
  // Create all datasets (sizes, chunks, and attributes)
  size_t nCuboids = mSensorMask.getDimensionSizes().ny;
  mCuboidsInfo.reserve(nCuboids);
  size_t actualPositionInBuffer = 0;

  for (size_t cuboidIdx = 0; cuboidIdx < nCuboids; cuboidIdx++)
  {
    CuboidInfo cuboidInfo;
    // Dont create dataset for compression coefficients if only kIAvgC or kQTermC should be stored
    if (!mDoNotSaveFlag)
    {
      cuboidInfo.cuboidIdx = createCuboidDataset(cuboidIdx);
    }
    cuboidInfo.startingPossitionInBuffer = actualPositionInBuffer;
    cuboidInfo.minValue.value            = std::numeric_limits<float>::max();
    cuboidInfo.maxValue.value            = std::numeric_limits<float>::min();
    cuboidInfo.minValue.index            = 0;
    cuboidInfo.maxValue.index            = 0;
    mCuboidsInfo.push_back(cuboidInfo);

    if (mReduceOp == ReduceOperator::kC)
    {
      actualPositionInBuffer += mSensorMask.getSizeOfCuboid(cuboidIdx, mCompressHelper->getHarmonics() * mComplexSize);
    }
    else
    {
      actualPositionInBuffer += mSensorMask.getSizeOfCuboid(cuboidIdx);
    }
  }

  // we're at the beginning
  mSampledTimeStep = 0;

  // Allocate memory
  allocateMemory();
  mCurrentStoreBuffer = mHostBuffer;

  if (mReduceOp == ReduceOperator::kC)
  {
    mCurrentStoreBuffer = nullptr;
  }
} // end of create
//----------------------------------------------------------------------------------------------------------------------

/**
 * Reopen the output stream after restart and reload data.
 */
void CuboidOutputStream::reopen()
{
  // Get parameters
  const Parameters& params = Parameters::getInstance();

  mSampledTimeStep = 0;
  if (mReduceOp == ReduceOperator::kNone || mReduceOp == ReduceOperator::kC || mReduceOp == ReduceOperator::kIAvgC)
  { // set correct sampled timestep for raw data series
    mSampledTimeStep = (params.getTimeIndex() < params.getSamplingStartTimeIndex())
                         ? 0
                         : (params.getTimeIndex() - params.getSamplingStartTimeIndex());
    if (mReduceOp == ReduceOperator::kC || mReduceOp == ReduceOperator::kIAvgC)
    {
      mCompressedTimeStep = size_t(std::max(float(floor(float(mSampledTimeStep) / mCompressHelper->getOSize())), 0.0f));
    }
  }

  // Create the memory buffer if necessary and set starting address
  mSize = (mReduceOp == ReduceOperator::kC)
            ? mSensorMask.getSizeOfAllCuboids(mCompressHelper->getHarmonics() * mComplexSize)
            : mSensorMask.getSizeOfAllCuboids();
  if (mReduceOp == ReduceOperator::kC)
  {
    mOSize = mSensorMask.getSizeOfAllCuboids();
  }

  // Allocate memory if needed
  allocateMemory();
  mCurrentStoreBuffer = mHostBuffer;

  // Open all datasets (sizes, chunks, and attributes)
  size_t nCuboids = mSensorMask.getDimensionSizes().ny;
  mCuboidsInfo.reserve(nCuboids);
  size_t actualPositionInBuffer = 0;

  // Open the HDF5 group
  if (!mDoNotSaveFlag)
  {
    mGroup = mFile.openGroup(mFile.getRootGroup(), mRootObjectName);
  }

  for (size_t cuboidIdx = 0; cuboidIdx < nCuboids; cuboidIdx++)
  {
    CuboidInfo cuboidInfo;

    // Indexed from 1
    const std::string datasetName = std::to_string(cuboidIdx + 1);

    // open the dataset
    if (!mDoNotSaveFlag)
    {
      cuboidInfo.cuboidIdx = mFile.openDataset(mGroup, datasetName);
    }
    cuboidInfo.startingPossitionInBuffer = actualPositionInBuffer;
    cuboidInfo.minValue.value            = std::numeric_limits<float>::max();
    cuboidInfo.maxValue.value            = std::numeric_limits<float>::min();
    cuboidInfo.minValue.index            = 0;
    cuboidInfo.maxValue.index            = 0;
    mCuboidsInfo.push_back(cuboidInfo);

    if (mReduceOp != ReduceOperator::kIAvg && mReduceOp != ReduceOperator::kQTerm &&
        mReduceOp != ReduceOperator::kQTermC && !mDoNotSaveFlag)
    {
      // read only if there is anything to read
      if (params.getTimeIndex() > params.getSamplingStartTimeIndex())
      {
        if (mReduceOp != ReduceOperator::kNone && mReduceOp != ReduceOperator::kC)
        { // Reload data
          mFile.readCompleteDataset(mGroup,
            datasetName,
            mSensorMask.getDimensionSizesOfCuboid(cuboidIdx),
            mHostBuffer + actualPositionInBuffer);
        }
        // Reload min and max values
        loadMinMaxValues(
          mFile, mGroup, datasetName, mCuboidsInfo[cuboidIdx].minValue, mCuboidsInfo[cuboidIdx].minValue);
      }
    }
    // move the pointer for the next cuboid beginning (this inits the locations)
    if (mReduceOp == ReduceOperator::kC)
    {
      actualPositionInBuffer += mSensorMask.getSizeOfCuboid(cuboidIdx, mCompressHelper->getHarmonics() * mComplexSize);
    }
    else
    {
      actualPositionInBuffer += mSensorMask.getSizeOfCuboid(cuboidIdx);
    }
  }

  // copy data over to the GPU only if there is anything to read
  if (params.getTimeIndex() > params.getSamplingStartTimeIndex())
  {
    copyToDevice();
    // Reload temp coefficients from checkpoint file
    loadCheckpointCompressionCoefficients();
    if (mReduceOp == ReduceOperator::kC)
    {
      mCurrentStoreBuffer = mHostBuffer1;
    }
  }

} // end of reopen
//----------------------------------------------------------------------------------------------------------------------

/**
 * Sample grid points, line them up in the buffer, if necessary a reduce operator is applied.
 */
void CuboidOutputStream::sample()
{
  size_t cuboidInBufferStart = 0;

  // dimension sizes of the matrix being sampled
  const dim3 dimSizes(static_cast<unsigned int>(mSourceMatrix.getDimensionSizes().nx),
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

    // get number of samples within the cuboid
    const size_t nSamples =
      (mSensorMask.getBottomRightCorner(cuboidIdx) - mSensorMask.getTopLeftCorner(cuboidIdx)).nElements();

    switch (mReduceOp)
    {
    case ReduceOperator::kNone:
    case ReduceOperator::kC:
    {
      // Kernel to sample raw quantities inside one cuboid
      OutputStreamsCudaKernels::sampleCuboid<ReduceOperator::kNone>(mDeviceBuffer + cuboidInBufferStart,
        mSourceMatrix.getDeviceData(),
        topLeftCorner,
        bottomRightCorner,
        dimSizes,
        nSamples);
      break;
    }
    case ReduceOperator::kRms:
    {
      OutputStreamsCudaKernels::sampleCuboid<ReduceOperator::kRms>(mDeviceBuffer + cuboidInBufferStart,
        mSourceMatrix.getDeviceData(),
        topLeftCorner,
        bottomRightCorner,
        dimSizes,
        nSamples);
      break;
    }
    case ReduceOperator::kMax:
    {
      OutputStreamsCudaKernels::sampleCuboid<ReduceOperator::kMax>(mDeviceBuffer + cuboidInBufferStart,
        mSourceMatrix.getDeviceData(),
        topLeftCorner,
        bottomRightCorner,
        dimSizes,
        nSamples);
      break;
    }
    case ReduceOperator::kMin:
    {
      OutputStreamsCudaKernels::sampleCuboid<ReduceOperator::kMin>(mDeviceBuffer + cuboidInBufferStart,
        mSourceMatrix.getDeviceData(),
        topLeftCorner,
        bottomRightCorner,
        dimSizes,
        nSamples);
      break;
    }
    default:
    {
      break;
    }
    }

    cuboidInBufferStart += nSamples;
  }

  if (mReduceOp == ReduceOperator::kNone || mReduceOp == ReduceOperator::kC)
  {
    // Record an event when the data has been copied over.
    cudaCheckErrors(cudaEventRecord(mEventSamplingFinished));
  }
} // end of sample
//----------------------------------------------------------------------------------------------------------------------

/**
 * Post sampling step, can work with other filled stream buffers
 */
void CuboidOutputStream::postSample()
{
  if (mReduceOp == ReduceOperator::kIAvgC && !Parameters::getInstance().getOnlyPostProcessingFlag())
  {
    float* bufferP =
      (*mOutputStreamContainer)[OutputStreamContainer::OutputStreamIdx::kPressureC].getCurrentStoreBuffer();
    float* bufferU =
      (*mOutputStreamContainer)[static_cast<OutputStreamContainer::OutputStreamIdx>(mVelocityOutputStreamIdx)]
        .getCurrentStoreBuffer();

    uint8_t* mBufferPInt8 = reinterpret_cast<uint8_t*>(bufferP);
    uint8_t* mBufferUInt8 = reinterpret_cast<uint8_t*>(bufferU);

    // TODO check the length of bufferP == the length of bufferU
    if (bufferP && bufferU)
    {
#pragma omp parallel for
      for (size_t i = 0; i < mSize; i++)
      {
        const size_t offset = mCompressHelper->getHarmonics() * i;
        // For every harmonics
        for (size_t ih = 0; ih < mCompressHelper->getHarmonics(); ih++)
        {
          size_t pH = offset + ih;
          FloatComplex sCP;
          FloatComplex sCU;
          if (Parameters::getInstance().get40bitCompressionFlag())
          {
            pH = pH * 5;
            CompressHelper::convert40bToFloatC(&mBufferPInt8[pH], sCP, CompressHelper::kMaxExpP);
            CompressHelper::convert40bToFloatC(&mBufferUInt8[pH], sCU, CompressHelper::kMaxExpU);
          }
          else
          {
            sCP = reinterpret_cast<FloatComplex*>(bufferP)[pH];
            sCU = reinterpret_cast<FloatComplex*>(bufferU)[pH];
          }
          mHostBuffer[i] += real(sCP * conj(sCU)) / 2.0f;
        }
      }
      mCompressedTimeStep++;
    }
  }
} // end of postSample
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
    /*for (size_t cuboidIdx = 0; cuboidIdx < mCuboidsInfo.size(); cuboidIdx++) {
      cuboidSize = mSensorMask.getDimensionSizesOfCuboid(cuboidIdx);
      cuboidSize.nt = 1;

      ReducedValue minValueLocal = mCuboidsInfo[cuboidIdx].minValue;
      ReducedValue maxValueLocal = mCuboidsInfo[cuboidIdx].maxValue;

      size_t cuboidStartIndex = mCuboidsInfo[cuboidIdx].startingPossitionInBuffer;
      for (size_t i = cuboidStartIndex; i < cuboidStartIndex + cuboidSize.nElements(); i++) {
        checkOrSetMinMaxValue(minValueLocal, maxValueLocal, mHostBuffer[i], cuboidSize.nElements() * mSampledTimeStep +
    i);
      }
      checkOrSetMinMaxValueGlobal(mCuboidsInfo[cuboidIdx].minValue, mCuboidsInfo[cuboidIdx].maxValue, minValueLocal,
    maxValueLocal);
    }*/

    // only raw time series are flushed down to the disk every time step
    flushBufferToFile();
  }

  if (mReduceOp == ReduceOperator::kC)
  {
    // make sure the data has been copied from the GPU
    cudaEventSynchronize(mEventSamplingFinished);

    // Compression
    // Compute local index and flags
    mStepLocal                          = mSampledTimeStep % (mCompressHelper->getBSize() - 1);
    mSavingFlag                         = ((mStepLocal + 1) % mCompressHelper->getOSize() == 0) ? true : false;
    mOddFrameFlag                       = ((mCompressedTimeStep + 1) % 2 == 0) ? true : false;
    const bool noCompressionOverlapFlag = Parameters::getInstance().getNoCompressionOverlapFlag();
    // noCompressionOverlapFlag -> mStoreBuffer == mStoreBuffer2
    mMirrorFirstHalfFrameFlag = (mCompressedTimeStep == 0 && mSavingFlag && !noCompressionOverlapFlag) ? true : false;

    FloatComplex* mStoreBufferFloatC  = reinterpret_cast<FloatComplex*>(mHostBuffer1);
    FloatComplex* mStoreBuffer2FloatC = reinterpret_cast<FloatComplex*>(mHostBuffer2);
    uint8_t* mStoreBufferInt8         = reinterpret_cast<uint8_t*>(mHostBuffer1);
    uint8_t* mStoreBuffer2Int8        = reinterpret_cast<uint8_t*>(mHostBuffer2);

    // For every cuboid/point
    for (size_t cuboidIdx = 0; cuboidIdx < mCuboidsInfo.size(); cuboidIdx++)
    {
      cuboidSize    = mSensorMask.getDimensionSizesOfCuboid(cuboidIdx);
      cuboidSize.nt = 1;
      // ReducedValue minValueLocal = mCuboidsInfo[cuboidIdx].minValue;
      // ReducedValue maxValueLocal = mCuboidsInfo[cuboidIdx].maxValue;

      size_t cuboidStartIndex = mCuboidsInfo[cuboidIdx].startingPossitionInBuffer;
      for (size_t i = cuboidStartIndex; i < cuboidStartIndex + cuboidSize.nElements(); i++)
      {
        // checkOrSetMinMaxValue(minValueLocal, maxValueLocal, mHostBuffer[i], cuboidSize.nElements() * mSampledTimeStep
        // + i);
        const size_t storeBufferIndexC = i * mCompressHelper->getHarmonics();

        // For every harmonics
        for (size_t ih = 0; ih < mCompressHelper->getHarmonics(); ih++)
        {
          size_t pH           = storeBufferIndexC + ih;
          const size_t bIndex = ih * mCompressHelper->getBSize() + mStepLocal;

          // 40-bit complex float compression
          if (Parameters::getInstance().get40bitCompressionFlag())
          {
            pH = pH * 5;
            FloatComplex cc1;
            FloatComplex cc2;
            if (Parameters::getInstance().getNoCompressionOverlapFlag())
            {
              CompressHelper::convert40bToFloatC(&mStoreBufferInt8[pH], cc1, mE);
              cc1 += mBE[bIndex] * mHostBuffer[i] + mBE_1[bIndex] * mHostBuffer[i];
              CompressHelper::convertFloatCTo40b(cc1, &mStoreBufferInt8[pH], mE);
            }
            else
            {
              CompressHelper::convert40bToFloatC(&mStoreBufferInt8[pH], cc1, mE);
              CompressHelper::convert40bToFloatC(&mStoreBuffer2Int8[pH], cc2, mE);
              cc1 += mBE[bIndex] * mHostBuffer[i];
              cc2 += mBE_1[bIndex] * mHostBuffer[i];
              CompressHelper::convertFloatCTo40b(cc1, &mStoreBufferInt8[pH], mE);
              CompressHelper::convertFloatCTo40b(cc2, &mStoreBuffer2Int8[pH], mE);
              // Mirror first "half" frame
              if (mMirrorFirstHalfFrameFlag)
              {
                cc2 += cc1;
                CompressHelper::convertFloatCTo40b(cc2, &mStoreBuffer2Int8[pH], mE);
              }
            }
          }
          else
          {
            // Correlation step
            mStoreBufferFloatC[pH] += mBE[bIndex] * mHostBuffer[i];
            mStoreBuffer2FloatC[pH] += mBE_1[bIndex] * mHostBuffer[i];
            // Mirror first "half" frame
            if (mMirrorFirstHalfFrameFlag)
            {
              mStoreBuffer2FloatC[pH] += mStoreBufferFloatC[pH];
            }
          }
        }
      }
    }
    // checkOrSetMinMaxValueGlobal(mCuboidsInfo[cuboidIdx].minValue, mCuboidsInfo[cuboidIdx].maxValue, minValueLocal,
    // maxValueLocal);

    const size_t steps  = Parameters::getInstance().getNt() - Parameters::getInstance().getSamplingStartTimeIndex();
    const bool lastStep = ((steps - mSampledTimeStep == 1) && steps <= mCompressHelper->getOSize()) ? true : false;
    if (mSavingFlag || lastStep)
    {
      // Select accumulated value (mStoreBuffer2 is first)
      mCurrentStoreBuffer = mOddFrameFlag ? mHostBuffer1 : mHostBuffer2;

      // Store selected buffer
      if (!mDoNotSaveFlag)
      {
        flushBufferToFile(mCurrentStoreBuffer);
      }

      mCompressedTimeStep++;
    }
    mSampledTimeStep++;
  }
} // end of flushRaw
//----------------------------------------------------------------------------------------------------------------------

/*
 * Apply post-processing on the buffer and flush it to the file.
 */
void CuboidOutputStream::postProcess()
{
  // run inherited method
  BaseOutputStream::postProcess();

  if (mReduceOp == ReduceOperator::kIAvgC && !Parameters::getInstance().getOnlyPostProcessingFlag())
  {
#pragma omp parallel for
    for (size_t i = 0; i < mSize; i++)
    {
      mHostBuffer[i] = mHostBuffer[i] / mCompressedTimeStep;
    }
    mCompressedTimeStep = 0;
  }

  // When no reduce operator is applied, the data is flushed after every time step
  // which means it has been done before
  if (mReduceOp != ReduceOperator::kNone && mReduceOp != ReduceOperator::kC && mReduceOp != ReduceOperator::kQTerm &&
      mReduceOp != ReduceOperator::kQTermC && !mDoNotSaveFlag)
  {
    // Copy data from GPU matrix
    if (mReduceOp != ReduceOperator::kIAvg && mReduceOp != ReduceOperator::kIAvgC)
    {
      copyFromDevice(); // TODO
    }
    flushBufferToFile();
  }

  if (!mDoNotSaveFlag && !Parameters::getInstance().getOnlyPostProcessingFlag())
  {
    // Store min and max values
    /*for (size_t cuboidIdx = 0; cuboidIdx < mCuboidsInfo.size(); cuboidIdx++)
    {
      storeMinMaxValues(mFile, mGroup, std::to_string(cuboidIdx + 1), mCuboidsInfo[cuboidIdx].minValue,
    mCuboidsInfo[cuboidIdx].maxValue);
    }*/
  }
} // end of postProcess
//----------------------------------------------------------------------------------------------------------------------

/**
 * Apply post-processing 2 on the buffer and flush it to the file.
 */
void CuboidOutputStream::postProcess2()
{
  // run inherited method
  BaseOutputStream::postProcess2();

  if (mReduceOp == ReduceOperator::kQTerm || mReduceOp == ReduceOperator::kQTermC)
  {
    flushBufferToFile();
  }

} // end of postProcess2
//----------------------------------------------------------------------------------------------------------------------

/**
 * Checkpoint the stream and close.
 */
void CuboidOutputStream::checkpoint()
{
  // raw data has already been flushed, others has to be flushed here
  if (mReduceOp != ReduceOperator::kNone && mReduceOp != ReduceOperator::kC && mReduceOp != ReduceOperator::kQTerm &&
      mReduceOp != ReduceOperator::kQTermC && !mDoNotSaveFlag)
  {
    // copy data from the device
    if (mReduceOp != ReduceOperator::kIAvg && mReduceOp != ReduceOperator::kIAvgC)
    {
      copyFromDevice(); // TODO
    }
    // flush to disk
    flushBufferToFile();
  }
  storeCheckpointCompressionCoefficients();

  if (!mDoNotSaveFlag)
  {
    // Store min and max values
    /*for (size_t cuboidIdx = 0; cuboidIdx < mCuboidsInfo.size(); cuboidIdx++)
    {
      storeMinMaxValues(mFile, mGroup, std::to_string(cuboidIdx + 1), mCuboidsInfo[cuboidIdx].minValue,
    mCuboidsInfo[cuboidIdx].maxValue);
    }*/
  }
} // end of checkpoint
//----------------------------------------------------------------------------------------------------------------------

/**
 * Close stream (apply post-processing if necessary, flush data, close datasets and the group).
 */
void CuboidOutputStream::close()
{
  if (!mDoNotSaveFlag)
  {
    // the group is still open
    if (mGroup != H5I_BADID)
    {
      // Close all datasets and the group
      for (size_t cuboidIdx = 0; cuboidIdx < mCuboidsInfo.size(); cuboidIdx++)
      {
        mFile.closeDataset(mCuboidsInfo[cuboidIdx].cuboidIdx);
      }
    }
    mCuboidsInfo.clear();
    mFile.closeGroup(mGroup);
  } // if opened
  mGroup = H5I_BADID;
} // end of close
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
  size_t nSampledTimeSteps = (mReduceOp == ReduceOperator::kNone) ? params.getNt() - params.getSamplingStartTimeIndex()
                                                                  : 0; // will be a 3D dataset

  // Set cuboid dimensions (subtract two corners (add 1) and use the appropriate component)
  DimensionSizes cuboidSize = mSensorMask.getDimensionSizesOfCuboid(cuboidIdx);
  cuboidSize.nt             = nSampledTimeSteps;

  if (mReduceOp == ReduceOperator::kC)
  {
    // NOTE minimal useful number of steps for compression is 1 period.
    size_t steps           = params.getNt() - params.getSamplingStartTimeIndex();
    size_t compressedSteps = size_t(std::max(float(floor(float(steps) / mCompressHelper->getOSize())), 1.0f));
    cuboidSize    = mSensorMask.getDimensionSizesOfCuboid(cuboidIdx, mCompressHelper->getHarmonics() * mComplexSize);
    cuboidSize.nt = compressedSteps;
  }

  // Set chunk size
  // If the size of the cuboid is bigger than 32 MB per timestep, set the chunk to approx 4MB
  DimensionSizes cuboidChunkSize(cuboidSize.nx,
    cuboidSize.ny,
    cuboidSize.nz,
    (mReduceOp == ReduceOperator::kNone || mReduceOp == ReduceOperator::kC) ? 1 : 0);

  if (cuboidChunkSize.nElements() > (kChunkSize4MB * 8))
  {
    size_t nSlabs = 1; // At least one slab
    while (nSlabs * cuboidSize.nx * cuboidSize.ny < kChunkSize4MB)
      nSlabs++;
    cuboidChunkSize.nz = nSlabs;
  }

  // Indexed from 1
  const std::string datasetName = std::to_string(cuboidIdx + 1);

  hid_t dataset;
  if (mFile.datasetExists(mGroup, datasetName))
  {
    dataset = mFile.openDataset(mGroup, datasetName);
  }
  else
  {
    dataset = mFile.createDataset(
      mGroup, datasetName, cuboidSize, cuboidChunkSize, Hdf5File::MatrixDataType::kFloat, params.getCompressionLevel());
    // Write dataset parameters
    mFile.writeMatrixDomainType(mGroup, datasetName, Hdf5File::MatrixDomainType::kReal);
    mFile.writeMatrixDataType(mGroup, datasetName, Hdf5File::MatrixDataType::kFloat);
  }

  // Write compression parameters as attributes
  if (mReduceOp == ReduceOperator::kC)
  {
    mFile.writeLongLongAttribute(mGroup, datasetName, "c_harmonics", ssize_t(mCompressHelper->getHarmonics()));
    mFile.writeStringAttribute(mGroup, datasetName, "c_type", "c");
    mFile.writeFloatAttribute(mGroup, datasetName, "c_period", mCompressHelper->getPeriod());
    mFile.writeLongLongAttribute(mGroup, datasetName, "c_mos", ssize_t(mCompressHelper->getMos()));
    mFile.writeLongLongAttribute(mGroup, datasetName, "c_shift", ssize_t(mShiftFlag));
    mFile.writeFloatAttribute(mGroup, datasetName, "c_complex_size", mComplexSize);
    mFile.writeLongLongAttribute(mGroup, datasetName, "c_max_exp", mE);
  }

  return dataset;
} // end of createCuboidDatasets
//----------------------------------------------------------------------------------------------------------------------

/**
 * Flush the buffer to the file (to multiple datasets if necessary).
 */
void CuboidOutputStream::flushBufferToFile(float* bufferToFlush)
{
  DimensionSizes position(0, 0, 0, 0);
  DimensionSizes blockSize(0, 0, 0, 0);

  if (mReduceOp == ReduceOperator::kNone)
    position.nt = mSampledTimeStep;
  if (mReduceOp == ReduceOperator::kC)
    position.nt = mCompressedTimeStep;

  for (size_t cuboidIdx = 0; cuboidIdx < mCuboidsInfo.size(); cuboidIdx++)
  {
    blockSize = mSensorMask.getDimensionSizesOfCuboid(cuboidIdx);
    if (mReduceOp == ReduceOperator::kC)
    {
      blockSize = mSensorMask.getDimensionSizesOfCuboid(cuboidIdx, mCompressHelper->getHarmonics() * mComplexSize);
    }
    blockSize.nt = 1;

    mFile.writeHyperSlab(mCuboidsInfo[cuboidIdx].cuboidIdx,
      position,
      blockSize,
      ((bufferToFlush != nullptr) ? bufferToFlush : mHostBuffer) + mCuboidsInfo[cuboidIdx].startingPossitionInBuffer);
  }

  if (mReduceOp != ReduceOperator::kC)
    mSampledTimeStep++;
} // end of flushBufferToFile
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//
