/**
 * @file        IndexOutputHDF5Stream.cpp
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
 *              16 July      2017, 16:54 (revised)
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

#include <OutputHDF5Streams/IndexOutputHDF5Stream.h>
#include <OutputHDF5Streams/OutputStreamsCUDAKernels.cuh>

#include <Parameters/Parameters.h>
#include <Logger/Logger.h>

//------------------------------------------------------------------------------------------------//
//------------------------------------------ Constants -------------------------------------------//
//------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//


/**
 * Constructor - links the HDF5 dataset, source (sampled matrix), sensor mask and the reduce
 * operator together. The constructor DOES NOT allocate memory because the size of the sensor mask
 * is not known at the time the instance of  the class is being created.
 *
 * @param [in] file         - Handle to the HDF5 (output) file
 * @param [in] datasetName  - The dataset's name (index based sensor data stored in a single dataset)
 * @param [in] sourceMatrix - The source matrix (only real matrices are supported)
 * @param [in] sensorMask   - Index based sensor mask
 * @param [in] reduceOp     - Reduce operator
 */
TIndexOutputHDF5Stream::TIndexOutputHDF5Stream(THDF5_File&           file,
                                               MatrixName&          datasetName,
                                               const TRealMatrix&    sourceMatrix,
                                               const TIndexMatrix&   sensorMask,
                                               const TReduceOperator reduceOp)
        : TBaseOutputHDF5Stream(file, datasetName, sourceMatrix, reduceOp),
          sensorMask(sensorMask),
          dataset(H5I_BADID),
          sampledTimeStep(0),
          eventSamplingFinished()
{
  // Create event for sampling
  checkCudaErrors(cudaEventCreate(&eventSamplingFinished));
}// end of TIndexOutputHDF5Stream
//--------------------------------------------------------------------------------------------------

/**
 * Destructor.
 * If the file is still opened, it applies the post processing and flush the data.
 * Then, the object memory is freed and the object destroyed.
 */
TIndexOutputHDF5Stream::~TIndexOutputHDF5Stream()
{
  // Destroy sampling event
  checkCudaErrors(cudaEventDestroy(eventSamplingFinished));

  Close();
  // free memory
  FreeMemory();
}// end of Destructor
//--------------------------------------------------------------------------------------------------

/**
 * Create a HDF5 stream, create a dataset, and allocate data for it.
 */
void TIndexOutputHDF5Stream::Create()
{

  size_t nSampledElementsPerStep = sensorMask.GetElementCount();

  const Parameters& params = Parameters::getInstance();

  // Derive dataset dimension sizes
  DimensionSizes datasetSize(nSampledElementsPerStep,
                              (reduceOp == TReduceOperator::NONE) ?  params.getNt() - params.getSamplingStartTimeIndex() : 1,
                              1);

  // Set HDF5 chunk size
  DimensionSizes chunkSize(nSampledElementsPerStep, 1, 1);
  // for chunks bigger than 32 MB
  if (nSampledElementsPerStep > (CHUNK_SIZE_4MB * 8))
  {
      chunkSize.nx = CHUNK_SIZE_4MB; // set chunk size to MB
  }

  // Create a dataset under the root group
  dataset = file.CreateFloatDataset(file.GetRootGroup(),
                                    rootObjectName,
                                    datasetSize,
                                    chunkSize,
                                    params.getCompressionLevel());

    // Write dataset parameters
  file.WriteMatrixDomainType(file.GetRootGroup(), rootObjectName, THDF5_File::TMatrixDomainType::REAL);
  file.WriteMatrixDataType  (file.GetRootGroup(), rootObjectName, THDF5_File::TMatrixDataType::FLOAT);

  // Sampled time step
  sampledTimeStep = 0;

  // Set buffer size
  bufferSize = nSampledElementsPerStep;

  // Allocate memory
  AllocateMemory();
}// end of Create
//--------------------------------------------------------------------------------------------------

/**
 * Reopen the output stream after restart.
 *
 */
void TIndexOutputHDF5Stream::Reopen()
{
  // Get parameters
  const Parameters& params = Parameters::getInstance();

  // Set buffer size
  bufferSize = sensorMask.GetElementCount();

  // Allocate memory
   AllocateMemory();

  // Reopen the dataset
  dataset = file.OpenDataset(file.GetRootGroup(), rootObjectName);


  if (reduceOp == TReduceOperator::NONE)
  { // raw time series - just seek to the right place in the dataset
    sampledTimeStep = (params.getTimeIndex() < params.getSamplingStartTimeIndex()) ?
                        0 : (params.getTimeIndex() - params.getSamplingStartTimeIndex());

  }
  else
  { // aggregated quantities - reload data
    sampledTimeStep = 0;

    // Read data from disk only if there were anything stored there (t_index >= start_index)
    if (Parameters::getInstance().getTimeIndex() > Parameters::getInstance().getSamplingStartTimeIndex())
    {
      // Since there is only a single timestep in the dataset, I can read the whole dataset
      file.ReadCompleteDataset(file.GetRootGroup(),
                               rootObjectName,
                               DimensionSizes(bufferSize, 1, 1),
                               hostBuffer);

      // Send data to device
      CopyDataToDevice();
    }
  }
}// end of Reopen
//--------------------------------------------------------------------------------------------------


/**
 * Sample grid points, line them up in the buffer, if necessary a reduce operator is applied.
 *
 * @warning data is not flushed, there is no sync.
 */
void TIndexOutputHDF5Stream::Sample()
{
  switch (reduceOp)
  {
    case TReduceOperator::NONE :
    {
      OutputStreamsCUDAKernels::SampleIndex<TReduceOperator::NONE>
                                           (deviceBuffer,
                                            sourceMatrix.GetDeviceData(),
                                            sensorMask.GetDeviceData(),
                                            sensorMask.GetElementCount());

      // Record an event when the data has been copied over.
      checkCudaErrors(cudaEventRecord(eventSamplingFinished));

      break;
    }// case NONE

    case TReduceOperator::RMS :
    {
      OutputStreamsCUDAKernels::SampleIndex<TReduceOperator::RMS>
                                           (deviceBuffer,
                                            sourceMatrix.GetDeviceData(),
                                            sensorMask.GetDeviceData(),
                                            sensorMask.GetElementCount());

      break;
    }// case RMS

    case TReduceOperator::MAX :
    {
      OutputStreamsCUDAKernels::SampleIndex<TReduceOperator::MAX>
                                           (deviceBuffer,
                                            sourceMatrix.GetDeviceData(),
                                            sensorMask.GetDeviceData(),
                                            sensorMask.GetElementCount());
      break;
    }// case MAX

    case TReduceOperator::MIN :
    {
      OutputStreamsCUDAKernels::SampleIndex<TReduceOperator::MIN>
                                           (deviceBuffer,
                                            sourceMatrix.GetDeviceData(),
                                            sensorMask.GetDeviceData(),
                                            sensorMask.GetElementCount());
      break;
    } //case MIN
  }// switch
}// end of Sample
//--------------------------------------------------------------------------------------------------


/**
 * Flush data for the timestep. Only applicable on RAW data series.
 */
void TIndexOutputHDF5Stream::FlushRaw()
{
  if (reduceOp == TReduceOperator::NONE)
  {
    // make sure the data has been copied from the GPU
    cudaEventSynchronize(eventSamplingFinished);

    // only raw time series are flushed down to the disk every time step
    FlushBufferToFile();
  }
}// end of FlushRaw
//--------------------------------------------------------------------------------------------------


/**
 * Apply post-processing on the buffer and flush it to the file.
 */
void TIndexOutputHDF5Stream::PostProcess()
{
  // run inherited method
  TBaseOutputHDF5Stream::PostProcess();

  // When no reduction operator is applied, the data is flushed after every time step
  // which means it has been done before
  if (reduceOp != TReduceOperator::NONE)
  {
    // Copy data from GPU matrix
    CopyDataFromDevice();
    // flush to disk
    FlushBufferToFile();
  }
}// end of PostProcessing
//-------------------------------------------------------------------------------------------------


/**
 * Checkpoint the stream and close.
 *
 */
void TIndexOutputHDF5Stream::Checkpoint()
{
  // raw data has already been flushed, others has to be flushed here
  if (reduceOp != TReduceOperator::NONE)
  {
    // copy data from the device
    CopyDataFromDevice();
    // flush to disk
    FlushBufferToFile();
  }
}// end of Checkpoint
//--------------------------------------------------------------------------------------------------

/**
 * Close stream (apply post-processing if necessary, flush data and close).
 */
void TIndexOutputHDF5Stream::Close()
{
  // the dataset is still opened
  if (dataset != H5I_BADID)
  {
    file.CloseDataset(dataset);
  }

  dataset = H5I_BADID;
}// end of Close
//--------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//


/**
 * Flush the buffer down to the file at the actual position
 */
void TIndexOutputHDF5Stream::FlushBufferToFile()
{
  file.WriteHyperSlab(dataset,
                      DimensionSizes(0,sampledTimeStep,0),
                      DimensionSizes(bufferSize,1,1),
                      hostBuffer);
  sampledTimeStep++;
}// end of FlushToFile
//--------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//

