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
 * @param [in] file         - HDF5 file to write the output to
 * @param [in] datasetName  - The name of the HDF5 group containing datasets for particular cuboids
 * @param [in] sourceMatrix - Source matrix to be sampled
 * @param [in] reduceOp     - Reduce operator
 */
TWholeDomainOutputHDF5Stream::TWholeDomainOutputHDF5Stream(THDF5_File&           file,
                                                           MatrixName&          datasetName,
                                                           TRealMatrix&          sourceMatrix,
                                                           const TReduceOperator reduceOp)
        : TBaseOutputHDF5Stream(file, datasetName, sourceMatrix, reduceOp),
          dataset(H5I_BADID),
          sampledTimeStep(0)
{

}// end of TWholeDomainOutputHDF5Stream
//--------------------------------------------------------------------------------------------------


/**
 * Destructor.
 * If the file is still opened, it applies the post processing and flush the data.
 * Then, the object memory is freed and the object destroyed.
 */
TWholeDomainOutputHDF5Stream::~TWholeDomainOutputHDF5Stream()
{
  Close();
  // free memory
  FreeMemory();
}// end of Destructor
//--------------------------------------------------------------------------------------------------



/**
 * Create a HDF5 stream for the whole domain and allocate data for it.
 */
void TWholeDomainOutputHDF5Stream::Create()
{
  DimensionSizes chunkSize(sourceMatrix.GetDimensionSizes().nx,
                            sourceMatrix.GetDimensionSizes().ny,
                            1);

  // Create a dataset under the root group
  dataset = file.CreateFloatDataset(file.GetRootGroup(),
                                    rootObjectName,
                                    sourceMatrix.GetDimensionSizes(),
                                    chunkSize,
                                    Parameters::getInstance().getCompressionLevel());

  // Write dataset parameters
  file.WriteMatrixDomainType(file.GetRootGroup(), rootObjectName, THDF5_File::TMatrixDomainType::REAL);
  file.WriteMatrixDataType  (file.GetRootGroup(), rootObjectName, THDF5_File::TMatrixDataType::FLOAT);

  // Set buffer size
  bufferSize = sourceMatrix.GetElementCount();

  // Allocate memory
  AllocateMemory();
}//end of Create
//--------------------------------------------------------------------------------------------------


/**
 * Reopen the output stream after restart and reload data
 */
void TWholeDomainOutputHDF5Stream::Reopen()
{
  const Parameters& params = Parameters::getInstance();

  // Set buffer size
  bufferSize = sourceMatrix.GetElementCount();

  // Allocate memory
  AllocateMemory();

  // Open the dataset under the root group
  dataset = file.OpenDataset(file.GetRootGroup(), rootObjectName);

  sampledTimeStep = 0;
  if (reduceOp == TReduceOperator::NONE)
  { // seek in the dataset
    sampledTimeStep = (params.getTimeIndex() < params.getSamplingStartTimeIndex()) ?
                       0 : (params.getTimeIndex() - params.getSamplingStartTimeIndex());
  }
  else
  { // reload data
    // Read data from disk only if there were anything stored there (t_index > start_index)
    //(one step ahead)
    if (params.getTimeIndex() > params.getSamplingStartTimeIndex())
    {
      file.ReadCompleteDataset(file.GetRootGroup(),
                                    rootObjectName,
                                    sourceMatrix.GetDimensionSizes(),
                                    hostBuffer);
      // Send data to device
      CopyDataToDevice();
    }
  }
}// end of Reopen
//--------------------------------------------------------------------------------------------------


/**
 * Sample all grid points, line them up in the buffer an flush to the disk unless a reduce operator
 * is applied
 */
void TWholeDomainOutputHDF5Stream::Sample()
{
  switch (reduceOp)
  {
    case TReduceOperator::NONE :
    {
      // Copy all data from GPU to CPU (no need to use a kernel)
      // this violates the const prerequisite, however this routine is still NOT used in the code
      const_cast<TRealMatrix&> (sourceMatrix).CopyFromDevice();

      // We use here direct HDF5 offload using MEMSPACE - seems to be faster for bigger datasets
      const DimensionSizes datasetPosition(0, 0, 0, sampledTimeStep); //4D position in the dataset

      DimensionSizes cuboidSize(sourceMatrix.GetDimensionSizes());// Size of the cuboid
      cuboidSize.nt = 1;

      // iterate over all cuboid to be sampled
      file.WriteCuboidToHyperSlab(dataset,
                                  datasetPosition,
                                  DimensionSizes(0,0,0,0), // position in the SourceMatrix
                                  cuboidSize,
                                  sourceMatrix.GetDimensionSizes(),
                                  sourceMatrix.GetHostData());

      sampledTimeStep++;   // Move forward in time

      break;
    }// case NONE

    case TReduceOperator::RMS  :
    {
      OutputStreamsCUDAKernels::SampleAll<TReduceOperator::RMS>
                                         (deviceBuffer,
                                          sourceMatrix.GetDeviceData(),
                                          sourceMatrix.GetElementCount());
      break;
    }// case RMS

    case TReduceOperator::MAX  :
    {
      OutputStreamsCUDAKernels::SampleAll<TReduceOperator::MAX>
                                         (deviceBuffer,
                                          sourceMatrix.GetDeviceData(),
                                          sourceMatrix.GetElementCount());
      break;
    }//case MAX

    case TReduceOperator::MIN  :
    {
      OutputStreamsCUDAKernels::SampleAll<TReduceOperator::MIN>
                                         (deviceBuffer,
                                          sourceMatrix.GetDeviceData(),
                                          sourceMatrix.GetElementCount());
      break;
    } //case MIN
  }// switch
}// end of Sample
//--------------------------------------------------------------------------------------------------


/**
 * Apply post-processing on the buffer and flush it to the file.
 */
void TWholeDomainOutputHDF5Stream::PostProcess()
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
 * Checkpoint the stream.
 */
void TWholeDomainOutputHDF5Stream::Checkpoint()
{
  // copy data on the CPU
    CopyDataFromDevice();

  // raw data has already been flushed, others has to be flushed here
  if (reduceOp != TReduceOperator::NONE) FlushBufferToFile();
}// end of Checkpoint
//-------------------------------------------------------------------------------------------------

/**
 * Close stream (apply post-processing if necessary, flush data and close).
 */
void TWholeDomainOutputHDF5Stream::Close()
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
 * Flush the buffer down to the file at the actual position.
 */
void TWholeDomainOutputHDF5Stream::FlushBufferToFile()
{
  DimensionSizes size = sourceMatrix.GetDimensionSizes();
  DimensionSizes position(0,0,0);

  // Not used for NONE now!
  if (reduceOp == TReduceOperator::NONE)
  {
      position.nt = sampledTimeStep;
      size.nt = sampledTimeStep;
  }

  file.WriteHyperSlab(dataset, position, size, hostBuffer);
  sampledTimeStep++;
}// end of FlushToFile
//--------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//

