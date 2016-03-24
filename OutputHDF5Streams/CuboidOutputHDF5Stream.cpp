/**
 * @file        CuboidOutputHDF5Stream.cpp
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file of classes responsible for storing output
 *              quantities into the output HDF5 file.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        13 February 2015, 12:51 (created)
 *              24 March    2016, 17:07 (revised)
 *
 *
 * @section License
 * This file is part of the C++ extension of the k-Wave Toolbox
 * (http://www.k-wave.org).\n Copyright (C) 2014 Jiri Jaros, Beau Johnston
 * and Bradley Treeby
 *
 * This file is part of the k-Wave. k-Wave is free software: you can
 * redistribute it and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation, either version
 * 3 of the License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with k-Wave. If not, see http://www.gnu.org/licenses/.
 */

#include <OutputHDF5Streams/CuboidOutputHDF5Stream.h>
#include <OutputHDF5Streams/OutputStreamsCUDAKernels.cuh>

#include <Parameters/Parameters.h>

//--------------------------------------------------------------------------//
//                              Constants                                   //
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
//                              Definitions                                 //
//--------------------------------------------------------------------------//

//----------------------------------------------------------------------------//
//--------------------------------- Macros -----------------------------------//
//----------------------------------------------------------------------------//

/**
 * Check errors of the CUDA routines and print error.
 * @param [in] code  - error code of last routine
 * @param [in] file  - The name of the file, where the error was raised
 * @param [in] line  - What is the line
 * @param [in] Abort - Shall the code abort?
 * @todo - check this routine and do it differently!
 */
inline void gpuAssert(cudaError_t code,
                      string file,
                      int line)
{
  if (code != cudaSuccess)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage,"GPUassert: %s %s %d\n",cudaGetErrorString(code),file.c_str(),line);

    // Throw exception
     throw std::runtime_error(ErrorMessage);
  }
}// end of gpuAssert
//------------------------------------------------------------------------------

/// Define to get the usage easier
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


//--------------------------------------------------------------------------//
//                              Implementation                              //
//                              public methods                              //
//--------------------------------------------------------------------------//

/**
 * Constructor - links the HDF5 dataset, source (sampled matrix), Sensor mask
 * and the reduction operator together. The constructor DOES NOT allocate memory
 * because the size of the sensor mask is not known at the time the instance of
 * the class is being created.
 *
 * @param [in] HDF5_File       - HDF5 file to write the output to
 * @param [in] HDF5_GroupName  - The name of the HDF5 group. This group contains
 *                               datasets for particular cuboids
 * @param [in] SourceMatrix    - Source matrix to be sampled
 * @param [in] SensorMask      - Sensor mask with the cuboid coordinates
 * @param [in] ReductionOp     - Reduction operator

 */
TCuboidOutputHDF5Stream::TCuboidOutputHDF5Stream(THDF5_File&              HDF5_File,
                                                 const char*              HDF5_GroupName,
                                                 const TRealMatrix&       SourceMatrix,
                                                 const TIndexMatrix&      SensorMask,
                                                 const TReductionOperator ReductionOp)
        : TBaseOutputHDF5Stream(HDF5_File, HDF5_GroupName, SourceMatrix, ReductionOp),
          SensorMask(SensorMask),
          HDF5_GroupId(H5I_BADID),
          SampledTimeStep(0),
          EventSamplingFinished()
{
  // Create event for sampling
  gpuErrchk(cudaEventCreate(&EventSamplingFinished));
}// end of TCubodidOutputHDF5Stream
//------------------------------------------------------------------------------

/**
 * Destructor.
 * if the file is still opened, it applies the post processing and flush the data.
 * Then, the object memory is freed and the object destroyed.
 */
TCuboidOutputHDF5Stream::~TCuboidOutputHDF5Stream()
{
  // Destroy sampling event
  gpuErrchk(cudaEventDestroy(EventSamplingFinished));
  // Close the stream
  Close();
  // free memory
  FreeMemory();
}// end ~TCubodidOutputHDF5Stream
//------------------------------------------------------------------------------

/**
 * Create a HDF5 stream and allocate data for it. It also creates a HDF5 group
 * with particular datasets (one per cuboid).
 */
void TCuboidOutputHDF5Stream::Create()
{
  // Create the HDF5 group and open it
  HDF5_GroupId = HDF5_File.CreateGroup(HDF5_File.GetRootGroup(), HDF5_RootObjectName);

  // Create all datasets (sizes, chunks, and attributes)
  size_t NumberOfCuboids        = SensorMask.GetDimensionSizes().Y;
  CuboidsInfo.reserve(NumberOfCuboids);
  size_t ActualPositionInBuffer = 0;

  for (size_t CuboidIndex = 0; CuboidIndex < NumberOfCuboids; CuboidIndex++)
  {
    TCuboidInfo CuboidInfo;

    CuboidInfo.HDF5_CuboidId = CreateCuboidDataset(CuboidIndex);
    CuboidInfo.StartingPossitionInBuffer = ActualPositionInBuffer;
    CuboidsInfo.push_back(CuboidInfo);

    ActualPositionInBuffer += (SensorMask.GetBottomRightCorner(CuboidIndex) -
                               SensorMask.GetTopLeftCorner(CuboidIndex)
                              ).GetElementCount();
  }

  //we're at the beginning
  SampledTimeStep = 0;

  // Create the memory buffer if necessary and set starting address
  BufferSize = SensorMask.GetTotalNumberOfElementsInAllCuboids();

  // Allocate memory
  AllocateMemory();
}// end of Create
//------------------------------------------------------------------------------



/**
 * Reopen the output stream after restart and reload data.
 */
void TCuboidOutputHDF5Stream::Reopen()
{
  // Get parameters
  TParameters * Params = TParameters::GetInstance();

  SampledTimeStep = 0;
  if (ReductionOp == roNONE) // set correct sampled timestep for raw data series
  {
    SampledTimeStep = (Params->Get_t_index() < Params->GetStartTimeIndex()) ?
                        0 : (Params->Get_t_index() - Params->GetStartTimeIndex());
  }

  // Create the memory buffer if necessary and set starting address
  BufferSize = SensorMask.GetTotalNumberOfElementsInAllCuboids();

  // Allocate memory if needed
  AllocateMemory();

  // Open all datasets (sizes, chunks, and attributes)
  size_t NumberOfCuboids        = SensorMask.GetDimensionSizes().Y;
  CuboidsInfo.reserve(NumberOfCuboids);
  size_t ActualPositionInBuffer = 0;

  // Open the HDF5 group
  HDF5_GroupId = HDF5_File.OpenGroup(HDF5_File.GetRootGroup(), HDF5_RootObjectName);

  for (size_t CuboidIndex = 0; CuboidIndex < NumberOfCuboids; CuboidIndex++)
  {
    TCuboidInfo CuboidInfo;

    // @todo: Can be done easily with std::to_string and c++0x or c++-11
    char HDF5_DatasetName[32] = "";
    // Indexed from 1
    sprintf(HDF5_DatasetName, "%ld",CuboidIndex + 1);

    // open the dataset
    CuboidInfo.HDF5_CuboidId = HDF5_File.OpenDataset(HDF5_GroupId,
                                                     HDF5_DatasetName);
    CuboidInfo.StartingPossitionInBuffer = ActualPositionInBuffer;
    CuboidsInfo.push_back(CuboidInfo);

    // read only if there is anything to read
    if (Params->Get_t_index() > Params->GetStartTimeIndex())
    {
      if (ReductionOp != roNONE)
      { // Reload data
        TDimensionSizes CuboidSize((SensorMask.GetBottomRightCorner(CuboidIndex) - SensorMask.GetTopLeftCorner(CuboidIndex)).X,
                                   (SensorMask.GetBottomRightCorner(CuboidIndex) - SensorMask.GetTopLeftCorner(CuboidIndex)).Y,
                                   (SensorMask.GetBottomRightCorner(CuboidIndex) - SensorMask.GetTopLeftCorner(CuboidIndex)).Z);

        HDF5_File.ReadCompleteDataset(HDF5_GroupId,
                                      HDF5_DatasetName,
                                      CuboidSize,
                                      HostStoreBuffer + ActualPositionInBuffer);
      }
    }
    // move the pointer for the next cuboid beginning (this inits the locations)
    ActualPositionInBuffer += (SensorMask.GetBottomRightCorner(CuboidIndex) -
                               SensorMask.GetTopLeftCorner(CuboidIndex)).GetElementCount();
  }

  // copy data over to the GPU only if there is anything to read
  if (Params->Get_t_index() > Params->GetStartTimeIndex())
  {
    CopyDataToDevice();
  }
}// end of Reopen
//------------------------------------------------------------------------------


/**
 * Sample grid points, line them up in the buffer, if necessary a  reduction
 * operator is applied.
 * @warning data is not flushed, there is no sync.
 */
void TCuboidOutputHDF5Stream::Sample()
{

  size_t CuboidInBufferStart = 0;
  // dimension sizes of the matrix being sampled
  const dim3 DimensionSizes (SourceMatrix.GetDimensionSizes().X,
                             SourceMatrix.GetDimensionSizes().Y,
                             SourceMatrix.GetDimensionSizes().Z);

  // Run over all cuboids - this is not a good solution as we need to run a distinct kernel for a cuboid
  for (size_t CuboidIdx = 0; CuboidIdx < CuboidsInfo.size(); CuboidIdx++)
  {
    // copy down dim sizes
    const dim3 TopLeftCorner(SensorMask.GetTopLeftCorner(CuboidIdx).X,
                             SensorMask.GetTopLeftCorner(CuboidIdx).Y,
                             SensorMask.GetTopLeftCorner(CuboidIdx).Z);
    const dim3 BottomRightCorner(SensorMask.GetBottomRightCorner(CuboidIdx).X,
                                 SensorMask.GetBottomRightCorner(CuboidIdx).Y,
                                 SensorMask.GetBottomRightCorner(CuboidIdx).Z);

    //get number of samples within the cuboid
    const size_t NumberOfSamples = (SensorMask.GetBottomRightCorner(CuboidIdx) -
                                    SensorMask.GetTopLeftCorner(CuboidIdx)
                                   ).GetElementCount();

    switch (ReductionOp)
    {
      case roNONE :
      {
        // Kernel to sample raw quantities inside one cuboid
        OutputStreamsCUDAKernels::SampleCuboid<roNONE>
                                              (DeviceStoreBuffer + CuboidInBufferStart,
                                               SourceMatrix.GetRawDeviceData(),
                                               TopLeftCorner,
                                               BottomRightCorner,
                                               DimensionSizes,
                                               NumberOfSamples);
        break;
      }
      case roRMS :
      {
        OutputStreamsCUDAKernels::SampleCuboid<roRMS>
                                              (DeviceStoreBuffer + CuboidInBufferStart,
                                               SourceMatrix.GetRawDeviceData(),
                                               TopLeftCorner,
                                               BottomRightCorner,
                                               DimensionSizes,
                                               NumberOfSamples);
        break;
      }
      case roMAX :
      {
        OutputStreamsCUDAKernels::SampleCuboid<roMAX>
                                              (DeviceStoreBuffer + CuboidInBufferStart,
                                               SourceMatrix.GetRawDeviceData(),
                                               TopLeftCorner,
                                               BottomRightCorner,
                                               DimensionSizes,
                                               NumberOfSamples);
        break;
      }
      case roMIN :
      {
        OutputStreamsCUDAKernels::SampleCuboid<roMIN>
                                              (DeviceStoreBuffer + CuboidInBufferStart,
                                               SourceMatrix.GetRawDeviceData(),
                                               TopLeftCorner,
                                               BottomRightCorner,
                                               DimensionSizes,
                                               NumberOfSamples);
        break;
      }
    }

    CuboidInBufferStart += NumberOfSamples;
  }

  if (ReductionOp == roNONE)
  {
    // Record an event when the data has been copied over.
    gpuErrchk(cudaEventRecord(EventSamplingFinished));
  }
}// end of Sample
//------------------------------------------------------------------------------

/**
 * Flush data for the timestep. Only applicable on RAW data series.
 */
void TCuboidOutputHDF5Stream::FlushRaw()
{
  if (ReductionOp == roNONE)
  {
    // make sure the data has been copied from the GPU
    cudaEventSynchronize(EventSamplingFinished);

    // only raw time series are flushed down to the disk every time step
    FlushBufferToFile();
  }
}// end of FlushRaw
//------------------------------------------------------------------------------


/*
 * Apply post-processing on the buffer and flush it to the file.
 */
void TCuboidOutputHDF5Stream::PostProcess()
{
  // run inherited method
  TBaseOutputHDF5Stream::PostProcess();

  // When no reduction operator is applied, the data is flushed after every time step
  // which means it has been done before
  if (ReductionOp != roNONE)
  {
    // Copy data from GPU matrix
    CopyDataFromDevice();

    FlushBufferToFile();
  }
}// end of PostProcessing
//------------------------------------------------------------------------------


/**
 * Checkpoint the stream and close.
 *
 */
void TCuboidOutputHDF5Stream::Checkpoint()
{
  // raw data has already been flushed, others has to be flushed here
  if (ReductionOp != roNONE)
  {
    // copy data from the device
    CopyDataFromDevice();
    // flush to disk
    FlushBufferToFile();
  }
}// end of Checkpoint
//------------------------------------------------------------------------------


/**
 * Close stream (apply post-processing if necessary, flush data, close datasets
 * and the group).
 */
void TCuboidOutputHDF5Stream::Close()
{
  // the group is still open
  if (HDF5_GroupId != H5I_BADID)
  {
    // Close all datasets and the group
    for (size_t CuboidIndex = 0; CuboidIndex < CuboidsInfo.size(); CuboidIndex++)
    {
      HDF5_File.CloseDataset(CuboidsInfo[CuboidIndex].HDF5_CuboidId);
    }
    CuboidsInfo.clear();

    HDF5_File.CloseGroup(HDF5_GroupId);
    HDF5_GroupId = H5I_BADID;
  }// if opened
}// end of Close
//------------------------------------------------------------------------------


//----------------------------------------------------------------------------//
//                            protected methods                               //
//----------------------------------------------------------------------------//

/**
 *  Create a new dataset for a given cuboid specified by index (order).
 * @param [in] Index - Index of the cuboid in the sensor mask
 * @return HDF5 handle to the dataset.
 */
hid_t TCuboidOutputHDF5Stream::CreateCuboidDataset(const size_t Index)
{
  TParameters* Params = TParameters::GetInstance();

  // if time series then Number of steps else 1
  size_t NumberOfSampledTimeSteps = (ReductionOp == roNONE)
                                      ? Params->Get_Nt() - Params->GetStartTimeIndex()
                                      : 0; // will be a 3D dataset
  // Set cuboid dimensions (subtract two corners (add 1) and use the appropriate component)
  TDimensionSizes CuboidSize((SensorMask.GetBottomRightCorner(Index) - SensorMask.GetTopLeftCorner(Index)).X,
                             (SensorMask.GetBottomRightCorner(Index) - SensorMask.GetTopLeftCorner(Index)).Y,
                             (SensorMask.GetBottomRightCorner(Index) - SensorMask.GetTopLeftCorner(Index)).Z,
                             NumberOfSampledTimeSteps
                            );

  // Set chunk size
  // If the size of the cuboid is bigger than 32 MB per timestep, set the chunk to approx 4MB
  size_t NumberOfSlabs = 1; //at least one slab
  TDimensionSizes CuboidChunkSize(CuboidSize.X, CuboidSize.Y, CuboidSize.Z, (ReductionOp == roNONE) ? 1 : 0);

  if (CuboidChunkSize.GetElementCount() > (ChunkSize_4MB * 8))
  {
    while (NumberOfSlabs * CuboidSize.X * CuboidSize.Y < ChunkSize_4MB) NumberOfSlabs++;
    CuboidChunkSize.Z = NumberOfSlabs;
  }

  // @todo: Can be done easily with std::to_string and c++0x or c++-11
  char HDF5_DatasetName[32] = "";
  // Indexed from 1
  sprintf(HDF5_DatasetName, "%ld",Index+1);
  hid_t HDF5_DatasetId = HDF5_File.CreateFloatDataset(HDF5_GroupId,
                                                      HDF5_DatasetName,
                                                      CuboidSize,
                                                      CuboidChunkSize,
                                                      Params->GetCompressionLevel()
                                                     );

  // Write dataset parameters
  HDF5_File.WriteMatrixDomainType(HDF5_GroupId,
                                  HDF5_DatasetName,
                                  THDF5_File::hdf5_mdt_real);
  HDF5_File.WriteMatrixDataType  (HDF5_GroupId,
                                  HDF5_DatasetName,
                                  THDF5_File::hdf5_mdt_float);

  return HDF5_DatasetId;
}//end of CreateCuboidDatasets
//------------------------------------------------------------------------------

/**
 * Flush the buffer to the file (to multiple datasets if necessary).
 */
void TCuboidOutputHDF5Stream::FlushBufferToFile()
{
  TDimensionSizes Position (0,0,0,0);
  TDimensionSizes BlockSize(0,0,0,0);

  if (ReductionOp == roNONE) Position.T = SampledTimeStep;

  for (size_t CuboidIndex = 0; CuboidIndex < CuboidsInfo.size(); CuboidIndex++)
  {
    BlockSize = SensorMask.GetBottomRightCorner(CuboidIndex) - SensorMask.GetTopLeftCorner(CuboidIndex);
    BlockSize.T = 1;

    HDF5_File.WriteHyperSlab(CuboidsInfo[CuboidIndex].HDF5_CuboidId,
                             Position,
                             BlockSize,
                             HostStoreBuffer + CuboidsInfo[CuboidIndex].StartingPossitionInBuffer
                             );
  }

  SampledTimeStep++;
}// end of FlushBufferToFile
//------------------------------------------------------------------------------