/**
 * @file      BaseOutputStream.cpp
 *
 * @author    Jiri Jaros, Petr Kleparnik \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file of the class saving RealMatrix data into the output HDF5 file.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      11 July      2012, 10:30 (created) \n
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

#include <cmath>
#include <immintrin.h>

#include <OutputStreams/BaseOutputStream.h>
#include <OutputStreams/OutputStreamsCudaKernels.cuh>
#include <Logger/Logger.h>
#include <Parameters/Parameters.h>
#include <Containers/OutputStreamContainer.h>

//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Constants -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor - there is no sensor mask by default!
 */
BaseOutputStream::BaseOutputStream(Hdf5File& file,
  MatrixName& rootObjectName,
  const RealMatrix& sourceMatrix,
  const ReduceOperator reduceOp,
  OutputStreamContainer* outputStreamContainer,
  bool doNotSaveFlag)

  : mFile(file), mRootObjectName(rootObjectName), mSourceMatrix(sourceMatrix), mReduceOp(reduceOp), mSize(0),
    mOutputStreamContainer(outputStreamContainer), mDoNotSaveFlag(doNotSaveFlag), mOSize(0)
{
  // Set compression variables
  if (mReduceOp == ReduceOperator::kC || mReduceOp == ReduceOperator::kIAvgC)
  {
    // Set compression helper
    mCompressHelper = &CompressHelper::getInstance();

    if (mRootObjectName == kUxNonStaggeredName + kCompressSuffix ||
        mRootObjectName == kUyNonStaggeredName + kCompressSuffix ||
        mRootObjectName == kUzNonStaggeredName + kCompressSuffix)
    {
      // Time shift of velocity
      mBE        = mCompressHelper->getBEShifted();
      mBE_1      = mCompressHelper->getBE_1Shifted();
      mShiftFlag = true;
      mE         = CompressHelper::kMaxExpU;
    }
    else
    {
      mBE   = mCompressHelper->getBE();
      mBE_1 = mCompressHelper->getBE_1();
      mE    = CompressHelper::kMaxExpP;
    }

    if (mRootObjectName == kIxAvgName + kCompressSuffix)
    {
      mVelocityOutputStreamIdx = static_cast<int>(OutputStreamContainer::OutputStreamIdx::kVelocityXNonStaggeredC);
    }
    else if (mRootObjectName == kIyAvgName + kCompressSuffix)
    {
      mVelocityOutputStreamIdx = static_cast<int>(OutputStreamContainer::OutputStreamIdx::kVelocityYNonStaggeredC);
    }
    else if (mRootObjectName == kIzAvgName + kCompressSuffix)
    {
      mVelocityOutputStreamIdx = static_cast<int>(OutputStreamContainer::OutputStreamIdx::kVelocityZNonStaggeredC);
    }

    if (Parameters::getInstance().get40bitCompressionFlag())
    {
      mComplexSize = 1.25f;
    }
  }
} // end of BaseOutputStream
//----------------------------------------------------------------------------------------------------------------------

/**
 * Post sampling step, can work with other filled stream buffers.
 */
void BaseOutputStream::postSample()
{
} // end of postSample
//----------------------------------------------------------------------------------------------------------------------

/**
 * Post sampling step 2, can work with other filled stream buffers.
 */
void BaseOutputStream::postSample2()
{
  // Compression stuff
  if (mReduceOp == ReduceOperator::kC && mSavingFlag && mCurrentStoreBuffer)
  {
    // Set zeros for next accumulation
    {
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < mSize; i++)
      {
        mCurrentStoreBuffer[i] = 0.0f;
      }
    }
    mCurrentStoreBuffer = nullptr;
  }
} // end of postSample2
//----------------------------------------------------------------------------------------------------------------------

/**
 * Apply post-processing on the buffer (Done on the GPU side as well).
 */
void BaseOutputStream::postProcess()
{
  switch (mReduceOp)
  {
  case ReduceOperator::kNone:
  {
    // do nothing
    break;
  }
  case ReduceOperator::kC:
  {
    // do nothing
    break;
  }
  case ReduceOperator::kIAvg:
  {
    // do nothing
    break;
  }
  case ReduceOperator::kIAvgC:
  {
    // do nothing
    break;
  }
  case ReduceOperator::kQTerm:
  {
    // do nothing
    break;
  }
  case ReduceOperator::kQTermC:
  {
    // do nothing
    break;
  }
  case ReduceOperator::kRms:
  {
    const float scalingCoeff =
      1.0f / (Parameters::getInstance().getNt() - Parameters::getInstance().getSamplingStartTimeIndex());

    OutputStreamsCudaKernels::postProcessingRms(mDeviceBuffer, scalingCoeff, mSize);
    break;
  }
  case ReduceOperator::kMax:
  {
    // do nothing
    break;
  }
  case ReduceOperator::kMin:
  {
    // do nothing
    break;
  }
  } // switch
} // end of postProcessing
//----------------------------------------------------------------------------------------------------------------------

/**
 * Apply post-processing 2 on the buffer.
 */
void BaseOutputStream::postProcess2()
{
} // end of postProcess2
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get current store buffer.
 * @return Current store buffer.
 */
float* BaseOutputStream::getCurrentStoreBuffer()
{
  return mCurrentStoreBuffer;
} // end of getCurrentStoreBuffer
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Zero current store buffer.
 */
void BaseOutputStream::zeroCurrentStoreBuffer()
{
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < mSize; i++)
  {
    mCurrentStoreBuffer[i] = 0.0f;
  }
} // end of zeroCurrentStoreBuffer
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Protected methods ------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Allocate memory using proper memory alignment.
 */
void BaseOutputStream::allocateMemory()
{
  if (mReduceOp == ReduceOperator::kC)
  {
    mHostBuffer = (float*)_mm_malloc((mOSize) * sizeof(float), kDataAlignment);
    if (!mHostBuffer)
    {
      throw std::bad_alloc();
    }
    mHostBuffer1 = (float*)_mm_malloc((mSize) * sizeof(float), kDataAlignment);
    if (!mHostBuffer1)
    {
      throw std::bad_alloc();
    }
    if (Parameters::getInstance().getNoCompressionOverlapFlag())
    {
      mHostBuffer2 = mHostBuffer1;
    }
    else
    {
      mHostBuffer2 = (float*)_mm_malloc((mSize) * sizeof(float), kDataAlignment);
      if (!mHostBuffer2)
      {
        throw std::bad_alloc();
      }
    }
  }
  else
  {
    // Allocate memory on the CPU side (always)
    mHostBuffer = (float*)_mm_malloc(mSize * sizeof(float), kDataAlignment);
    if (!mHostBuffer)
    {
      throw std::bad_alloc();
    }
  }

  // memory allocation done on core 0 - GPU is pinned to the first sockets
  // we need different initialization for different reduce ops
  switch (mReduceOp)
  {
  case ReduceOperator::kNone:
  {
    // zero the matrix - on the CPU side and lock on core 0 (gpu pinned to 1st socket)
    for (size_t i = 0; i < mSize; i++)
    {
      mHostBuffer[i] = 0.0f;
    }
    break;
  }

  case ReduceOperator::kC:
  {
    // zero the matrix - on the core 0
    for (size_t i = 0; i < mOSize; i++)
    {
      mHostBuffer[i] = 0.0f;
    }
    for (size_t i = 0; i < mSize; i++)
    {
      mHostBuffer1[i] = 0.0f;
      mHostBuffer2[i] = 0.0f;
    }
    break;
  }

  case ReduceOperator::kIAvg:
  {
    // zero the matrix
    for (size_t i = 0; i < mSize; i++)
    {
      mHostBuffer[i] = 0.0f;
    }
    break;
  }

  case ReduceOperator::kIAvgC:
  {
    // zero the matrix
    for (size_t i = 0; i < mSize; i++)
    {
      mHostBuffer[i] = 0.0f;
    }
    break;
  }

  case ReduceOperator::kQTerm:
  {
    // zero the matrix
    for (size_t i = 0; i < mSize; i++)
    {
      mHostBuffer[i] = 0.0f;
    }
    break;
  }

  case ReduceOperator::kQTermC:
  {
    // zero the matrix
    for (size_t i = 0; i < mSize; i++)
    {
      mHostBuffer[i] = 0.0f;
    }
    break;
  }

  case ReduceOperator::kRms:
  {
    // zero the matrix - on the CPU side and lock on core 0 (gpu pinned to 1st socket)
    for (size_t i = 0; i < mSize; i++)
    {
      mHostBuffer[i] = 0.0f;
    }
    break;
  }

  case ReduceOperator::kMax:
  {
    // set the values to the highest negative float value - on the core 0
    for (size_t i = 0; i < mSize; i++)
    {
      mHostBuffer[i] = -1 * std::numeric_limits<float>::max();
    }
    break;
  }

  case ReduceOperator::kMin:
  {
    // set the values to the highest float value - on the core 0
    for (size_t i = 0; i < mSize; i++)
    {
      mHostBuffer[i] = std::numeric_limits<float>::max();
    }
    break;
  }
  } // switch

  if (mReduceOp == ReduceOperator::kC)
  {
    // Register Host memory (pin in memory only - no mapped data)
    cudaCheckErrors(
      cudaHostRegister(mHostBuffer, mOSize * sizeof(float), cudaHostRegisterPortable | cudaHostRegisterMapped));
  }
  else
  {
    // Register Host memory (pin in memory only - no mapped data)
    cudaCheckErrors(
      cudaHostRegister(mHostBuffer, mSize * sizeof(float), cudaHostRegisterPortable | cudaHostRegisterMapped));
  }
  // cudaHostAllocWriteCombined - cannot be used since GPU writes and CPU reads

  // Map CPU data to GPU memory (RAW data) or allocate a GPU data (aggregated)
  if (mReduceOp == ReduceOperator::kNone || mReduceOp == ReduceOperator::kC)
  {
    // Register CPU memory for zero-copy
    cudaCheckErrors(cudaHostGetDevicePointer<float>(&mDeviceBuffer, mHostBuffer, 0));
  }
  else
  {
    if (mReduceOp == ReduceOperator::kC)
    {
      // Allocate memory on the GPU side
      if ((cudaMalloc<float>(&mDeviceBuffer, mOSize * sizeof(float)) != cudaSuccess) || (!mDeviceBuffer))
      {
        throw std::bad_alloc();
      }
    }
    else
    {
      // Allocate memory on the GPU side
      if ((cudaMalloc<float>(&mDeviceBuffer, mSize * sizeof(float)) != cudaSuccess) || (!mDeviceBuffer))
      {
        throw std::bad_alloc();
      }
    }
    // if doing aggregation copy initialised arrays on GPU
    copyToDevice();
  }
} // end of allocateMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Free memory.
 */
void BaseOutputStream::freeMemory()
{
  // free host buffer
  if (mHostBuffer)
  {
    cudaHostUnregister(mHostBuffer);
    _mm_free(mHostBuffer);
    mHostBuffer = nullptr;
  }
  if (mHostBuffer1)
  {
    _mm_free(mHostBuffer1);
    mHostBuffer1 = nullptr;
  }
  if (mHostBuffer2 && !Parameters::getInstance().getNoCompressionOverlapFlag())
  {
    _mm_free(mHostBuffer2);
    mHostBuffer2 = nullptr;
  }

  // Free GPU memory
  if (mReduceOp != ReduceOperator::kNone && mReduceOp != ReduceOperator::kC)
  {
    cudaCheckErrors(cudaFree(mDeviceBuffer));
  }
  mDeviceBuffer = nullptr;
} // end of FreeMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Check or set local minimal and maximal value and their indices.
 */
void BaseOutputStream::checkOrSetMinMaxValue(ReducedValue& minValue, ReducedValue& maxValue, float value, hsize_t index)
{
  if (minValue.value > value)
  {
    minValue.value = value;
    minValue.index = index;
  }
  if (maxValue.value < value)
  {
    maxValue.value = value;
    maxValue.index = index;
  }
} // end of checkOrSetMinMaxValue
//----------------------------------------------------------------------------------------------------------------------

/**
 * Check or set global (#pragma omp critical) minimal and maximal value and their indices.
 */
void BaseOutputStream::checkOrSetMinMaxValueGlobal(
  ReducedValue& minValue, ReducedValue& maxValue, ReducedValue minValueLocal, ReducedValue maxValueLocal)
{
#pragma omp critical
  {
    if (minValue.value > minValueLocal.value)
    {
      minValue.value = minValueLocal.value;
      minValue.index = minValueLocal.index;
    }
  }

#pragma omp critical
  {
    if (maxValue.value < maxValueLocal.value)
    {
      maxValue.value = maxValueLocal.value;
      maxValue.index = maxValueLocal.index;
    }
  }
} // end of checkOrSetMinMaxValueGlobal
//----------------------------------------------------------------------------------------------------------------------

/**
 * Load minimal and maximal values from dataset attributes.
 */
void BaseOutputStream::loadMinMaxValues(
  Hdf5File& file, hid_t group, std::string datasetName, ReducedValue& minValue, ReducedValue& maxValue)
{
  // Reload min and max values
  if (mReduceOp == ReduceOperator::kNone || mReduceOp == ReduceOperator::kC)
  {
    // try {
    minValue.value = file.readFloatAttribute(group, datasetName, "min");
    maxValue.value = file.readFloatAttribute(group, datasetName, "max");
    minValue.index = hsize_t(file.readLongLongAttribute(group, datasetName, "min_index"));
    maxValue.index = hsize_t(file.readLongLongAttribute(group, datasetName, "max_index"));
    //} catch (std::exception &)
    //{}
  }
} // end of loadMinMaxValues
//----------------------------------------------------------------------------------------------------------------------

/**
 * Store minimal and maximal values as dataset attributes.
 */
void BaseOutputStream::storeMinMaxValues(
  Hdf5File& file, hid_t group, std::string datasetName, ReducedValue minValue, ReducedValue maxValue)
{
  if (mReduceOp == ReduceOperator::kNone || mReduceOp == ReduceOperator::kC)
  {
    file.writeFloatAttribute(group, datasetName, "min", minValue.value);
    file.writeFloatAttribute(group, datasetName, "max", maxValue.value);
    file.writeLongLongAttribute(group, datasetName, "min_index", ssize_t(minValue.index));
    file.writeLongLongAttribute(group, datasetName, "max_index", ssize_t(maxValue.index));
  }
} // end of storeMinMaxValues
//----------------------------------------------------------------------------------------------------------------------

/**
 * Load checkpoint compression coefficients and average intensity.
 */
void BaseOutputStream::loadCheckpointCompressionCoefficients()
{
  if (mReduceOp == ReduceOperator::kC)
  {
    Hdf5File& checkpointFile = Parameters::getInstance().getCheckpointFile();
    checkpointFile.readCompleteDataset(
      checkpointFile.getRootGroup(), "Temp_" + mRootObjectName + "_1", DimensionSizes(mSize, 1, 1), mHostBuffer1);
    checkpointFile.readCompleteDataset(
      checkpointFile.getRootGroup(), "Temp_" + mRootObjectName + "_2", DimensionSizes(mSize, 1, 1), mHostBuffer2);
  }
  if (mReduceOp == ReduceOperator::kIAvgC)
  {
    Hdf5File& checkpointFile = Parameters::getInstance().getCheckpointFile();
    checkpointFile.readCompleteDataset(
      checkpointFile.getRootGroup(), "Temp_" + mRootObjectName, DimensionSizes(mSize, 1, 1), mHostBuffer);
  }
} // end of loadCheckpointCompressionCoefficients
//----------------------------------------------------------------------------------------------------------------------

/**
 * Store checkpoint compression coefficients and average intensity.
 */

void BaseOutputStream::storeCheckpointCompressionCoefficients()
{
  // Store temp compression coefficients
  if (mReduceOp == ReduceOperator::kC)
  {
    Hdf5File& checkpointFile = Parameters::getInstance().getCheckpointFile();
    hid_t dataset1           = checkpointFile.createDataset(checkpointFile.getRootGroup(),
                "Temp_" + mRootObjectName + "_1",
                DimensionSizes(mSize, 1, 1),
                DimensionSizes(mSize, 1, 1),
                Hdf5File::MatrixDataType::kFloat,
                Parameters::getInstance().getCompressionLevel());

    checkpointFile.writeHyperSlab(dataset1, DimensionSizes(0, 0, 0), DimensionSizes(mSize, 1, 1), mHostBuffer1);
    checkpointFile.closeDataset(dataset1);

    hid_t dataset2 = checkpointFile.createDataset(checkpointFile.getRootGroup(),
      "Temp_" + mRootObjectName + "_2",
      DimensionSizes(mSize, 1, 1),
      DimensionSizes(mSize, 1, 1),
      Hdf5File::MatrixDataType::kFloat,
      Parameters::getInstance().getCompressionLevel());

    checkpointFile.writeHyperSlab(dataset2, DimensionSizes(0, 0, 0), DimensionSizes(mSize, 1, 1), mHostBuffer2);
    checkpointFile.closeDataset(dataset2);

    // Write data and domain type
    checkpointFile.writeMatrixDataType(
      checkpointFile.getRootGroup(), "Temp_" + mRootObjectName + "_1", Hdf5File::MatrixDataType::kFloat);
    checkpointFile.writeMatrixDomainType(
      checkpointFile.getRootGroup(), "Temp_" + mRootObjectName + "_1", Hdf5File::MatrixDomainType::kReal);
    checkpointFile.writeMatrixDataType(
      checkpointFile.getRootGroup(), "Temp_" + mRootObjectName + "_2", Hdf5File::MatrixDataType::kFloat);
    checkpointFile.writeMatrixDomainType(
      checkpointFile.getRootGroup(), "Temp_" + mRootObjectName + "_2", Hdf5File::MatrixDomainType::kReal);
  }
  // Store temp compression average intensity
  if (mReduceOp == ReduceOperator::kIAvgC)
  {
    Hdf5File& checkpointFile = Parameters::getInstance().getCheckpointFile();
    hid_t dataset1           = checkpointFile.createDataset(checkpointFile.getRootGroup(),
                "Temp_" + mRootObjectName,
                DimensionSizes(mSize, 1, 1),
                DimensionSizes(mSize, 1, 1),
                Hdf5File::MatrixDataType::kFloat,
                Parameters::getInstance().getCompressionLevel());

    checkpointFile.writeHyperSlab(dataset1, DimensionSizes(0, 0, 0), DimensionSizes(mSize, 1, 1), mHostBuffer);
    checkpointFile.closeDataset(dataset1);
    // Write data and domain type
    checkpointFile.writeMatrixDataType(
      checkpointFile.getRootGroup(), "Temp_" + mRootObjectName, Hdf5File::MatrixDataType::kFloat);
    checkpointFile.writeMatrixDomainType(
      checkpointFile.getRootGroup(), "Temp_" + mRootObjectName, Hdf5File::MatrixDomainType::kReal);
  }
} // end of storeCheckpointCompressionCoefficients
//----------------------------------------------------------------------------------------------------------------------

/**
 *  Copy data hostBuffer -> deviceBuffer
 */
void BaseOutputStream::copyToDevice()
{
  if (mReduceOp == ReduceOperator::kC)
  {
    cudaCheckErrors(cudaMemcpy(mDeviceBuffer, mHostBuffer, mOSize * sizeof(float), cudaMemcpyHostToDevice));
  }
  else
  {
    cudaCheckErrors(cudaMemcpy(mDeviceBuffer, mHostBuffer, mSize * sizeof(float), cudaMemcpyHostToDevice));
  }
} // end of copyToDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data deviceBuffer -> hostBuffer
 */
void BaseOutputStream::copyFromDevice()
{
  if (mReduceOp == ReduceOperator::kC)
  {
    cudaCheckErrors(cudaMemcpy(mHostBuffer, mDeviceBuffer, mOSize * sizeof(float), cudaMemcpyDeviceToHost));
  }
  else
  {
    cudaCheckErrors(cudaMemcpy(mHostBuffer, mDeviceBuffer, mSize * sizeof(float), cudaMemcpyDeviceToHost));
  }
} // end of copyFromDevice
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Private methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//
