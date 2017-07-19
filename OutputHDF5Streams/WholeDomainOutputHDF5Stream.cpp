/**
 * @file        WholeDomainOutputHDF5Stream.cpp
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file of the class saving RealMatrix data into the output
 *              HDF5 file, e.g. p_max_all.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        28 August    2014, 11:15 (created)
 *              19 July      2017, 15:22 (revised)
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

#include <OutputHDF5Streams/WholeDomainOutputHDF5Stream.h>
#include <Parameters/Parameters.h>

#include <OutputHDF5Streams/OutputStreamsCUDAKernels.cuh>

//------------------------------------------------------------------------------------------------//
//------------------------------------------ Constants -------------------------------------------//
//------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Constructor - links the HDF5 dataset and SourceMatrix
 */
WholeDomainOutputStream::WholeDomainOutputStream(THDF5_File&           file,
                                                 MatrixName&          datasetName,
                                                 RealMatrix&          sourceMatrix,
                                                 const ReduceOperator reduceOp)
  : BaseOutputStream(file, datasetName, sourceMatrix, reduceOp),
    mDataset(H5I_BADID),
    mSampledTimeStep(0)
{

}// end of WholeDomainOutputStream
//----------------------------------------------------------------------------------------------------------------------


/**
 * Destructor.
 */
WholeDomainOutputStream::~WholeDomainOutputStream()
{
  close();
  // free memory
  freeMemory();
}// end of ~WholeDomainOutputStream
//----------------------------------------------------------------------------------------------------------------------



/**
 * Create a HDF5 stream for the whole domain and allocate data for it.
 */
void WholeDomainOutputStream::create()
{
  DimensionSizes chunkSize(mSourceMatrix.getDimensionSizes().nx,
                           mSourceMatrix.getDimensionSizes().ny,
                           1);

  // Create a dataset under the root group
  mDataset = mFile.CreateFloatDataset(mFile.GetRootGroup(),
                                      mRootObjectName,
                                      mSourceMatrix.getDimensionSizes(),
                                      chunkSize,
                                      Parameters::getInstance().getCompressionLevel());

  // Write dataset parameters
  mFile.WriteMatrixDomainType(mFile.GetRootGroup(), mRootObjectName, THDF5_File::TMatrixDomainType::REAL);
  mFile.WriteMatrixDataType  (mFile.GetRootGroup(), mRootObjectName, THDF5_File::TMatrixDataType::FLOAT);

  // Set buffer size
  mSize = mSourceMatrix.size();

  // Allocate memory
  allocateMemory();
}//end of create
//----------------------------------------------------------------------------------------------------------------------


/**
 * Reopen the output stream after restart and reload data
 */
void WholeDomainOutputStream::reopen()
{
  const Parameters& params = Parameters::getInstance();

  // Set buffer size
  mSize = mSourceMatrix.size();

  // Allocate memory
  allocateMemory();

  // Open the dataset under the root group
  mDataset = mFile.OpenDataset(mFile.GetRootGroup(), mRootObjectName);

  mSampledTimeStep = 0;
  if (mReduceOp == ReduceOperator::kNone)
  { // seek in the dataset
    mSampledTimeStep = (params.getTimeIndex() < params.getSamplingStartTimeIndex()) ?
                       0 : (params.getTimeIndex() - params.getSamplingStartTimeIndex());
  }
  else
  { // reload data
    // Read data from disk only if there were anything stored there (t_index > start_index)
    //(one step ahead)
    if (params.getTimeIndex() > params.getSamplingStartTimeIndex())
    {
      mFile.ReadCompleteDataset(mFile.GetRootGroup(),
                                mRootObjectName,
                                mSourceMatrix.getDimensionSizes(),
                                mHostBuffer);
      // Send data to device
      copyToDevice();
    }
  }
}// end of reopen
//----------------------------------------------------------------------------------------------------------------------


/**
 * Sample all grid points, line them up in the buffer an flush to the disk unless a reduce operator
 * is applied
 */
void WholeDomainOutputStream::sample()
{
  switch (mReduceOp)
  {
    case ReduceOperator::kNone:
    {
      // Copy all data from GPU to CPU (no need to use a kernel)
      // this violates the const prerequisite, however this routine is still NOT used in the code
      const_cast<RealMatrix&> (mSourceMatrix).copyFromDevice();

      // We use here direct HDF5 offload using MEMSPACE - seems to be faster for bigger datasets
      const DimensionSizes datasetPosition(0, 0, 0, mSampledTimeStep); //4D position in the dataset

      DimensionSizes cuboidSize(mSourceMatrix.getDimensionSizes());// Size of the cuboid
      cuboidSize.nt = 1;

      // iterate over all cuboid to be sampled
      mFile.WriteCuboidToHyperSlab(mDataset,
                                   datasetPosition,
                                   DimensionSizes(0,0,0,0), // position in the SourceMatrix
                                   cuboidSize,
                                   mSourceMatrix.getDimensionSizes(),
                                   mSourceMatrix.getHostData());

      mSampledTimeStep++;   // Move forward in time

      break;
    }// case kNone

    case ReduceOperator::kRms:
    {
      OutputStreamsCudaKernels::sampleAll<ReduceOperator::kRms>
                                         (mDeviceBuffer,
                                          mSourceMatrix.getDeviceData(),
                                          mSourceMatrix.size());
      break;
    }// case kRms

    case ReduceOperator::kMax:
    {
      OutputStreamsCudaKernels::sampleAll<ReduceOperator::kMax>
                                         (mDeviceBuffer,
                                          mSourceMatrix.getDeviceData(),
                                          mSourceMatrix.size());
      break;
    }//case kMax

    case ReduceOperator::kMin:
    {
      OutputStreamsCudaKernels::sampleAll<ReduceOperator::kMin>
                                         (mDeviceBuffer,
                                          mSourceMatrix.getDeviceData(),
                                          mSourceMatrix.size());
      break;
    } //case kMin
  }// switch
}// end of sample
//----------------------------------------------------------------------------------------------------------------------


/**
 * Apply post-processing on the buffer and flush it to the file.
 */
void WholeDomainOutputStream::postProcess()
{
  // run inherited method
  BaseOutputStream::postProcess();

  // When no reduce operator is applied, the data is flushed after every time step
  // which means it has been done before
  if (mReduceOp != ReduceOperator::kNone)
  {
    // Copy data from GPU matrix
    copyFromDevice();

    flushBufferToFile();
  }
}// end of postProcess
//----------------------------------------------------------------------------------------------------------------------

/**
 * Checkpoint the stream.
 */
void WholeDomainOutputStream::checkpoint()
{
  // copy data on the CPU
    copyFromDevice();

  // raw data has already been flushed, others has to be flushed here
  if (mReduceOp != ReduceOperator::kNone) flushBufferToFile();
}// end of checkpoint
//----------------------------------------------------------------------------------------------------------------------

/**
 * Close stream (apply post-processing if necessary, flush data and close).
 */
void WholeDomainOutputStream::close()
{
  // the dataset is still opened
  if (mDataset != H5I_BADID)
  {
    mFile.CloseDataset(mDataset);
  }

  mDataset = H5I_BADID;
}// end of close
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Flush the buffer down to the file at the actual position.
 */
void WholeDomainOutputStream::flushBufferToFile()
{
  DimensionSizes size = mSourceMatrix.getDimensionSizes();
  DimensionSizes position(0,0,0);

  // Not used for NONE now!
  if (mReduceOp == ReduceOperator::kNone)
  {
    position.nt = mSampledTimeStep;
    size.nt = mSampledTimeStep;
  }

  mFile.WriteHyperSlab(mDataset, position, size, mHostBuffer);
  mSampledTimeStep++;
}// end of flushBufferToFile
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

