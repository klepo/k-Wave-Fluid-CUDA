/**
 * @file      BaseOutputStream.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file of the class saving RealMatrix data into the output HDF5 file.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      11 July      2012, 10:30 (created) \n
 *            14 March     2019, 09:48 (revised)
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
#include <limits>
#include <immintrin.h>

// Windows build needs to undefine macro MINMAX to support std::limits
#ifdef _WIN64
  #ifndef NOMINMAX
    # define NOMINMAX
  #endif
  
  #include <windows.h>
#endif

#include <OutputStreams/BaseOutputStream.h>
#include <OutputStreams/OutputStreamsCudaKernels.cuh>

#include <Logger/Logger.h>
#include <Parameters/Parameters.h>


//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Constants -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor - there is no sensor mask by default!
 */
BaseOutputStream::BaseOutputStream(Hdf5File&            file,
                                   MatrixName&          rootObjectName,
                                   const RealMatrix&    sourceMatrix,
                                   const ReduceOperator reduceOp)
  : mFile(file),
    mRootObjectName(rootObjectName),
    mSourceMatrix(sourceMatrix),
    mReduceOp(reduceOp)
{
  // Set compression variables
  if (mReduceOp == ReduceOperator::kC)
  {
    // Set compression helper
    mCompressHelper = &CompressHelper::getInstance();
  }
 }// end of BaseOutputStream
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

    case ReduceOperator::kRms:
    {
      const float scalingCoeff = 1.0f / (Parameters::getInstance().getNt() -
                                         Parameters::getInstance().getSamplingStartTimeIndex());

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
  }// switch

}// end of postProcessing
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Protected methods ------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Allocate memory using proper memory alignment.
 */
void BaseOutputStream::allocateMemory()
{
  // Allocate memory on the CPU side (always)
  mHostBuffer = (float*) _mm_malloc(mSize * sizeof (float), kDataAlignment);

  if (!mHostBuffer)
  {
    throw std::bad_alloc();
  }

  if (mReduceOp == ReduceOperator::kC)
  {
    mHostBuffer1 = (float*) _mm_malloc(mCSize * sizeof(float), kDataAlignment);

    if (!mHostBuffer1)
    {
    throw std::bad_alloc();
    }
    mHostBuffer2 = (float*) _mm_malloc(mCSize * sizeof(float), kDataAlignment);

    if (!mHostBuffer2)
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
      for (size_t i = 0; i < mCSize; i++)
      {
        mHostBuffer1[i] = 0.0f;
        mHostBuffer2[i] = 0.0f;
      }
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
  }// switch

  // Register Host memory (pin in memory only - no mapped data)
  cudaCheckErrors(cudaHostRegister(mHostBuffer,
                                   mSize * sizeof (float),
                                   cudaHostRegisterPortable | cudaHostRegisterMapped));
  // cudaHostAllocWriteCombined - cannot be used since GPU writes and CPU reads

  // Map CPU data to GPU memory (RAW data) or allocate a GPU data (aggregated)
  if (mReduceOp == ReduceOperator::kNone || mReduceOp == ReduceOperator::kC)
  {
    // Register CPU memory for zero-copy
    cudaCheckErrors(cudaHostGetDevicePointer<float>(&mDeviceBuffer, mHostBuffer, 0));
  }
  else
  {
    // Allocate memory on the GPU side
    if ((cudaMalloc<float>(&mDeviceBuffer, mSize * sizeof (float))!= cudaSuccess) || (!mDeviceBuffer))
    {
      throw std::bad_alloc();
    }
    // if doing aggregation copy initialised arrays on GPU
    copyToDevice();
  }
}// end of allocateMemory
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
  }
  mHostBuffer = nullptr;

  if (mHostBuffer1)
  {
    _mm_free(mHostBuffer1);
  }
  mHostBuffer1 = nullptr;

  if (mHostBuffer2)
  {
    _mm_free(mHostBuffer2);
  }
  mHostBuffer2 = nullptr;

  if (minValue)
  {
    _mm_free(minValue);
    minValue = nullptr;
  }
  if (maxValue)
  {
    _mm_free(maxValue);
    maxValue = nullptr;
  }
  if (minValueIndex)
  {
    _mm_free(minValueIndex);
    minValueIndex = nullptr;
  }
  if (maxValueIndex)
  {
    _mm_free(maxValueIndex);
    maxValueIndex = nullptr;
  }

  // Free GPU memory
  if (mReduceOp != ReduceOperator::kNone && mReduceOp != ReduceOperator::kC)
  {
    cudaCheckErrors(cudaFree(mDeviceBuffer));
  }
  mDeviceBuffer = nullptr;
}// end of FreeMemory
//----------------------------------------------------------------------------------------------------------------------

void BaseOutputStream::checkOrSetMinMaxValue(float &minV, float &maxV, float value, hsize_t &minVIndex, hsize_t &maxVIndex, hsize_t index)
{
  if (minV > value)
  { // TODO: think about this
    #pragma omp critical
    {
      if (minV > value)
      {
        minV = value;
        minVIndex = index;
      }
    }
  }

  if (maxV < value)
  { // TODO: think about this
    #pragma omp critical
    {
      if (maxV < value)
      {
        maxV = value;
        maxVIndex = index;
      }
    }
  }
}



void BaseOutputStream::allocateMinMaxMemory(hsize_t items)
{
  this->items = items;
  maxValue = (float*) _mm_malloc(items * sizeof(float), kDataAlignment);
  minValue = (float*) _mm_malloc(items * sizeof(float), kDataAlignment);
  maxValueIndex = (hsize_t*) _mm_malloc(items * sizeof(hsize_t), kDataAlignment);
  minValueIndex = (hsize_t*) _mm_malloc(items * sizeof(hsize_t), kDataAlignment);

  for (size_t i = 0; i < items; i++)
  {
    maxValue[i] = std::numeric_limits<float>::min();
    minValue[i] = std::numeric_limits<float>::max();
    maxValueIndex[i] = 0;
    minValueIndex[i] = 0;
  }
}

void BaseOutputStream::loadMinMaxValues(Hdf5File &file, hid_t group, std::string datasetName, size_t index, bool checkpoint)
{
  std::string suffix = checkpoint ? "_" + std::to_string(index) : "";

  // Reload min and max values
  if (mReduceOp == ReduceOperator::kNone || mReduceOp == ReduceOperator::kC)
  {
    //try {
      minValue[index] = file.readFloatAttribute(group, datasetName, "min" + suffix);
      maxValue[index] = file.readFloatAttribute(group, datasetName, "max" + suffix);
      minValueIndex[index] = hsize_t(file.readLongLongAttribute(group, datasetName, "min_index" + suffix));
      maxValueIndex[index] = hsize_t(file.readLongLongAttribute(group, datasetName, "max_index" + suffix));
    //} catch (std::exception &) {
    //}
  }
}

void BaseOutputStream::storeMinMaxValues(Hdf5File &file, hid_t group, std::string datasetName, size_t index, bool checkpoint)
{
  std::string suffix = checkpoint ? "_" + std::to_string(index) : "";
  if (mReduceOp == ReduceOperator::kNone || mReduceOp == ReduceOperator::kC)
  {
    file.writeFloatAttribute(group, datasetName, "min" + suffix, minValue[index]);
    file.writeFloatAttribute(group, datasetName, "max" + suffix, maxValue[index]);
    file.writeLongLongAttribute(group, datasetName, "min_index" + suffix, ssize_t(minValueIndex[index]));
    file.writeLongLongAttribute(group, datasetName, "max_index" + suffix, ssize_t(maxValueIndex[index]));
  }
}

void BaseOutputStream::loadCheckpointCompressionCoefficients()
{
  if (mReduceOp == ReduceOperator::kC)
  {
    Hdf5File& checkpointFile = Parameters::getInstance().getCheckpointFile();
    checkpointFile.readCompleteDataset(checkpointFile.getRootGroup(),
                                       "Temp_" + mRootObjectName + "_1",
                                       DimensionSizes(mCSize, 1, 1),
                                       mHostBuffer1);
    checkpointFile.readCompleteDataset(checkpointFile.getRootGroup(),
                                       "Temp_" + mRootObjectName + "_2",
                                       DimensionSizes(mCSize, 1, 1),
                                       mHostBuffer2);
  }
}

void BaseOutputStream::storeCheckpointCompressionCoefficients()
{
  // Store temp compression coefficients
  if (mReduceOp == ReduceOperator::kC)
  {
    Hdf5File& checkpointFile = Parameters::getInstance().getCheckpointFile();
    hid_t dataset1 = checkpointFile.createDataset(checkpointFile.getRootGroup(),
                                       "Temp_" + mRootObjectName + "_1",
                                       DimensionSizes(mCSize, 1, 1),
                                       DimensionSizes(mCSize, 1, 1),
                                       Hdf5File::MatrixDataType::kFloat,
                                       Parameters::getInstance().getCompressionLevel());

    checkpointFile.writeHyperSlab(dataset1, DimensionSizes(0, 0, 0), DimensionSizes(mCSize, 1, 1), mHostBuffer1);
    checkpointFile.closeDataset(dataset1);

    hid_t dataset2 = checkpointFile.createDataset(checkpointFile.getRootGroup(),
                                       "Temp_" + mRootObjectName + "_2",
                                       DimensionSizes(mCSize, 1, 1),
                                       DimensionSizes(mCSize, 1, 1),
                                       Hdf5File::MatrixDataType::kFloat,
                                       Parameters::getInstance().getCompressionLevel());

    checkpointFile.writeHyperSlab(dataset2, DimensionSizes(0, 0, 0), DimensionSizes(mCSize, 1, 1), mHostBuffer2);
    checkpointFile.closeDataset(dataset2);

    // Write data and domain type
    checkpointFile.writeMatrixDataType  (checkpointFile.getRootGroup(), "Temp_" + mRootObjectName + "_1", Hdf5File::MatrixDataType::kFloat);
    checkpointFile.writeMatrixDomainType(checkpointFile.getRootGroup(), "Temp_" + mRootObjectName + "_1", Hdf5File::MatrixDomainType::kReal);
    checkpointFile.writeMatrixDataType  (checkpointFile.getRootGroup(), "Temp_" + mRootObjectName + "_2", Hdf5File::MatrixDataType::kFloat);
    checkpointFile.writeMatrixDomainType(checkpointFile.getRootGroup(), "Temp_" + mRootObjectName + "_2", Hdf5File::MatrixDomainType::kReal);
  }
}

/**
 *  Copy data hostBuffer -> deviceBuffer
 */
void BaseOutputStream::copyToDevice()
{
  cudaCheckErrors(cudaMemcpy(mDeviceBuffer, mHostBuffer, mSize * sizeof(float), cudaMemcpyHostToDevice));
}// end of copyToDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data deviceBuffer -> hostBuffer
 */
void BaseOutputStream::copyFromDevice()
{
  cudaCheckErrors(cudaMemcpy(mHostBuffer, mDeviceBuffer, mSize * sizeof(float), cudaMemcpyDeviceToHost));
}// end of copyFromDevice
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Private methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

