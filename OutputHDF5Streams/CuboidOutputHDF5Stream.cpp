/**
 * @file        CuboidOutputHDF5Stream.cpp
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file of classes responsible for storing output quantities based
 *              on the cuboid sensor mask into the output HDF5 file.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        13 February  2015, 12:51 (created)
 *              19 July      2017, 12:12 (revised)
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

#include <OutputHDF5Streams/CuboidOutputHDF5Stream.h>
#include <OutputHDF5Streams/OutputStreamsCUDAKernels.cuh>

#include <Parameters/Parameters.h>
#include <Logger/Logger.h>

using std::string;

//------------------------------------------------------------------------------------------------//
//------------------------------------------ Constants -------------------------------------------//
//------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Constructor - links the HDF5 dataset, source (sampled matrix), sensor mask and the reduce
 * operator together. The constructor DOES NOT allocate memory because the size of the sensor mask
 * is not known at the time the instance of the class is being created.
 *
 * @param [in] file         - HDF5 file to write the output to
 * @param [in] groupName    - The name of the HDF5 group with datasets for particular cuboids
 * @param [in] sourceMatrix - Source matrix to be sampled
 * @param [in] sensorMask   - Sensor mask with the cuboid coordinates
 * @param [in] reduceOp     - Reduce operator

 */
TCuboidOutputHDF5Stream::TCuboidOutputHDF5Stream(THDF5_File&           file,
                                                 MatrixName&          groupName,
                                                 const RealMatrix&    sourceMatrix,
                                                 const IndexMatrix&   sensorMask,
                                                 const TReduceOperator reduceOp)
        : TBaseOutputHDF5Stream(file, groupName, sourceMatrix, reduceOp),
          sensorMask(sensorMask),
          group(H5I_BADID),
          sampledTimeStep(0),
          eventSamplingFinished()
{
  // Create event for sampling
  cudaCheckErrors(cudaEventCreate(&eventSamplingFinished));
}// end of TCubodidOutputHDF5Stream
//--------------------------------------------------------------------------------------------------

/**
 * Destructor.
 * if the file is still opened, it applies the post processing and flush the data.
 * Then, the object memory is freed and the object destroyed.
 */
TCuboidOutputHDF5Stream::~TCuboidOutputHDF5Stream()
{
  // Destroy sampling event
  cudaCheckErrors(cudaEventDestroy(eventSamplingFinished));
  // Close the stream
  Close();
  // free memory
  FreeMemory();
}// end ~TCubodidOutputHDF5Stream
//--------------------------------------------------------------------------------------------------

/**
 * Create a HDF5 stream and allocate data for it. It also creates a HDF5 group with particular
 * datasets (one per cuboid).
 */
void TCuboidOutputHDF5Stream::Create()
{
  // Create the HDF5 group and open it
  group = file.CreateGroup(file.GetRootGroup(), rootObjectName);

  // Create all datasets (sizes, chunks, and attributes)
  size_t nCuboids = sensorMask.getDimensionSizes().ny;
  cuboidsInfo.reserve(nCuboids);
  size_t actualPositionInBuffer = 0;

  for (size_t cuboidIdx = 0; cuboidIdx < nCuboids; cuboidIdx++)
  {
    TCuboidInfo cuboidInfo;

    cuboidInfo.cuboidIdx = CreateCuboidDataset(cuboidIdx);
    cuboidInfo.startingPossitionInBuffer = actualPositionInBuffer;
    cuboidsInfo.push_back(cuboidInfo);

    actualPositionInBuffer += (sensorMask.getBottomRightCorner(cuboidIdx) -
                               sensorMask.getTopLeftCorner(cuboidIdx)
                              ).nElements();
  }

  //we're at the beginning
  sampledTimeStep = 0;

  // Create the memory buffer if necessary and set starting address
  bufferSize = sensorMask.getSizeOfAllCuboids();

  // Allocate memory
  AllocateMemory();
}// end of Create
//--------------------------------------------------------------------------------------------------



/**
 * Reopen the output stream after restart and reload data.
 */
void TCuboidOutputHDF5Stream::Reopen()
{
  // Get parameters
  const Parameters& params = Parameters::getInstance();

  sampledTimeStep = 0;
  if (reduceOp == TReduceOperator::NONE) // set correct sampled timestep for raw data series
  {
    sampledTimeStep = (params.getTimeIndex() < params.getSamplingStartTimeIndex()) ?
                       0 : (params.getTimeIndex() - params.getSamplingStartTimeIndex());
  }

  // Create the memory buffer if necessary and set starting address
  bufferSize = sensorMask.getSizeOfAllCuboids();

  // Allocate memory if needed
  AllocateMemory();

  // Open all datasets (sizes, chunks, and attributes)
  size_t nCuboids = sensorMask.getDimensionSizes().ny;
  cuboidsInfo.reserve(nCuboids);
  size_t actualPositionInBuffer = 0;

  // Open the HDF5 group
  group = file.OpenGroup(file.GetRootGroup(), rootObjectName);

  for (size_t CuboidIdx = 0; CuboidIdx < nCuboids; CuboidIdx++)
  {
    TCuboidInfo cuboidInfo;
    // Indexed from 1
    const string datasetName = std::to_string(CuboidIdx + 1);

    // open the dataset
    cuboidInfo.cuboidIdx = file.OpenDataset(group,datasetName.c_str());
    cuboidInfo.startingPossitionInBuffer = actualPositionInBuffer;
    cuboidsInfo.push_back(cuboidInfo);

    // read only if there is anything to read
    if (params.getTimeIndex() > params.getSamplingStartTimeIndex())
    {
      if (reduceOp != TReduceOperator::NONE)
      { // Reload data
        DimensionSizes cuboidSize((sensorMask.getBottomRightCorner(CuboidIdx) - sensorMask.getTopLeftCorner(CuboidIdx)).nx,
                                   (sensorMask.getBottomRightCorner(CuboidIdx) - sensorMask.getTopLeftCorner(CuboidIdx)).ny,
                                   (sensorMask.getBottomRightCorner(CuboidIdx) - sensorMask.getTopLeftCorner(CuboidIdx)).nz);

        file.ReadCompleteDataset(group,
                                 datasetName.c_str(),
                                 cuboidSize,
                                 hostBuffer + actualPositionInBuffer);
      }
    }
    // move the pointer for the next cuboid beginning (this inits the locations)
    actualPositionInBuffer += (sensorMask.getBottomRightCorner(CuboidIdx) -
                               sensorMask.getTopLeftCorner(CuboidIdx)).nElements();
  }

  // copy data over to the GPU only if there is anything to read
  if (params.getTimeIndex() > params.getSamplingStartTimeIndex())
  {
    CopyDataToDevice();
  }
}// end of Reopen
//--------------------------------------------------------------------------------------------------


/**
 * Sample grid points, line them up in the buffer, if necessary a reduce operator is applied.
 *
 * @warning data is not flushed, there is no sync.
 */
void TCuboidOutputHDF5Stream::Sample()
{
  size_t cuboidInBufferStart = 0;
  // dimension sizes of the matrix being sampled
  const dim3 dimSizes (static_cast<unsigned int>(sourceMatrix.getDimensionSizes().nx),
                       static_cast<unsigned int>(sourceMatrix.getDimensionSizes().ny),
                       static_cast<unsigned int>(sourceMatrix.getDimensionSizes().nz));

  // Run over all cuboids - this is not a good solution as we need to run a distinct kernel for a cuboid
  for (size_t cuboidIdx = 0; cuboidIdx < cuboidsInfo.size(); cuboidIdx++)
  {
    // copy down dim sizes
    const dim3 topLeftCorner(static_cast<unsigned int>(sensorMask.getTopLeftCorner(cuboidIdx).nx),
                             static_cast<unsigned int>(sensorMask.getTopLeftCorner(cuboidIdx).ny),
                             static_cast<unsigned int>(sensorMask.getTopLeftCorner(cuboidIdx).nz));
    const dim3 bottomRightCorner(static_cast<unsigned int>(sensorMask.getBottomRightCorner(cuboidIdx).nx),
                                 static_cast<unsigned int>(sensorMask.getBottomRightCorner(cuboidIdx).ny),
                                 static_cast<unsigned int>(sensorMask.getBottomRightCorner(cuboidIdx).nz));

    //get number of samples within the cuboid
    const size_t nSamples = (sensorMask.getBottomRightCorner(cuboidIdx) -
                             sensorMask.getTopLeftCorner(cuboidIdx)
                            ).nElements();

    switch (reduceOp)
    {
      case TReduceOperator::NONE :
      {
        // Kernel to sample raw quantities inside one cuboid
        OutputStreamsCUDAKernels::SampleCuboid<TReduceOperator::NONE>
                                              (deviceBuffer + cuboidInBufferStart,
                                               sourceMatrix.getDeviceData(),
                                               topLeftCorner,
                                               bottomRightCorner,
                                               dimSizes,
                                               nSamples);
        break;
      }
      case TReduceOperator::RMS :
      {
        OutputStreamsCUDAKernels::SampleCuboid<TReduceOperator::RMS>
                                              (deviceBuffer + cuboidInBufferStart,
                                               sourceMatrix.getDeviceData(),
                                               topLeftCorner,
                                               bottomRightCorner,
                                               dimSizes,
                                               nSamples);
        break;
      }
      case TReduceOperator::MAX :
      {
        OutputStreamsCUDAKernels::SampleCuboid<TReduceOperator::MAX>
                                              (deviceBuffer + cuboidInBufferStart,
                                               sourceMatrix.getDeviceData(),
                                               topLeftCorner,
                                               bottomRightCorner,
                                               dimSizes,
                                               nSamples);
        break;
      }
      case TReduceOperator::MIN :
      {
        OutputStreamsCUDAKernels::SampleCuboid<TReduceOperator::MIN>
                                              (deviceBuffer + cuboidInBufferStart,
                                               sourceMatrix.getDeviceData(),
                                               topLeftCorner,
                                               bottomRightCorner,
                                               dimSizes,
                                               nSamples);
        break;
      }
    }

    cuboidInBufferStart += nSamples;
  }

  if (reduceOp == TReduceOperator::NONE)
  {
    // Record an event when the data has been copied over.
    cudaCheckErrors(cudaEventRecord(eventSamplingFinished));
  }
}// end of Sample
//-------------------------------------------------------------------------------------------------

/**
 * Flush data for the timestep. Only applicable on RAW data series.
 */
void TCuboidOutputHDF5Stream::FlushRaw()
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


/*
 * Apply post-processing on the buffer and flush it to the file.
 */
void TCuboidOutputHDF5Stream::PostProcess()
{
  // run inherited method
  TBaseOutputHDF5Stream::PostProcess();

  // When no reduce operator is applied, the data is flushed after every time step
  // which means it has been done before
  if (reduceOp != TReduceOperator::NONE)
  {
    // Copy data from GPU matrix
    CopyDataFromDevice();

    FlushBufferToFile();
  }
}// end of PostProcessing
//--------------------------------------------------------------------------------------------------


/**
 * Checkpoint the stream and close.
 */
void TCuboidOutputHDF5Stream::Checkpoint()
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
//-------------------------------------------------------------------------------------------------


/**
 * Close stream (apply post-processing if necessary, flush data, close datasets and the group).
 */
void TCuboidOutputHDF5Stream::Close()
{
  // the group is still open
  if (group != H5I_BADID)
  {
    // Close all datasets and the group
    for (size_t cuboidIdx = 0; cuboidIdx < cuboidsInfo.size(); cuboidIdx++)
    {
      file.CloseDataset(cuboidsInfo[cuboidIdx].cuboidIdx);
    }
    cuboidsInfo.clear();

    file.CloseGroup(group);
    group = H5I_BADID;
  }// if opened
}// end of Close
//--------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 *  Create a new dataset for a given cuboid specified by index (order).
 *
 * @param [in] cuboidIdx - Index of the cuboid in the sensor mask
 * @return HDF5 handle to the dataset.
 */
hid_t TCuboidOutputHDF5Stream::CreateCuboidDataset(const size_t cuboidIdx)
{
  const Parameters& params = Parameters::getInstance();

  // if time series then Number of steps else 1
  const size_t nSampledTimeSteps = (reduceOp == TReduceOperator::NONE)
                                   ? params.getNt() - params.getSamplingStartTimeIndex() : 0; // will be a 3D dataset

  // Set cuboid dimensions (subtract two corners (add 1) and use the appropriate component)
  DimensionSizes cuboidSize((sensorMask.getBottomRightCorner(cuboidIdx) - sensorMask.getTopLeftCorner(cuboidIdx)).nx,
                             (sensorMask.getBottomRightCorner(cuboidIdx) - sensorMask.getTopLeftCorner(cuboidIdx)).ny,
                             (sensorMask.getBottomRightCorner(cuboidIdx) - sensorMask.getTopLeftCorner(cuboidIdx)).nz,
                             nSampledTimeSteps);

  // Set chunk size
  // If the size of the cuboid is bigger than 32 MB per timestep, set the chunk to approx 4MB
  size_t nSlabs = 1; //at least one slab
  DimensionSizes cuboidChunkSize(cuboidSize.nx, cuboidSize.ny, cuboidSize.nz,
                                  (reduceOp == TReduceOperator::NONE) ? 1 : 0);

  if (cuboidChunkSize.nElements() > (CHUNK_SIZE_4MB * 8))
  {
    while (nSlabs * cuboidSize.nx * cuboidSize.ny < CHUNK_SIZE_4MB) nSlabs++;
    cuboidChunkSize.nz = nSlabs;
  }

  // Indexed from 1
  const string datasetName = std::to_string(cuboidIdx + 1);

  hid_t dataset = file.CreateFloatDataset(group,
                                                 datasetName.c_str(),
                                                 cuboidSize,
                                                 cuboidChunkSize,
                                                 params.getCompressionLevel());

  // Write dataset parameters
  file.WriteMatrixDomainType(group, datasetName.c_str(), THDF5_File::TMatrixDomainType::REAL);
  file.WriteMatrixDataType  (group, datasetName.c_str(), THDF5_File::TMatrixDataType::FLOAT);

  return dataset;
}//end of CreateCuboidDatasets
//--------------------------------------------------------------------------------------------------

/**
 * Flush the buffer to the file (to multiple datasets if necessary).
 */
void TCuboidOutputHDF5Stream::FlushBufferToFile()
{
  DimensionSizes position (0,0,0,0);
  DimensionSizes blockSize(0,0,0,0);

  if (reduceOp == TReduceOperator::NONE) position.nt = sampledTimeStep;

  for (size_t cuboidIdx = 0; cuboidIdx < cuboidsInfo.size(); cuboidIdx++)
  {
    blockSize = sensorMask.getBottomRightCorner(cuboidIdx) - sensorMask.getTopLeftCorner(cuboidIdx);
    blockSize.nt = 1;

    file.WriteHyperSlab(cuboidsInfo[cuboidIdx].cuboidIdx,
                        position,
                        blockSize,
                        hostBuffer + cuboidsInfo[cuboidIdx].startingPossitionInBuffer);
  }

  sampledTimeStep++;
}// end of FlushBufferToFile
//--------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//