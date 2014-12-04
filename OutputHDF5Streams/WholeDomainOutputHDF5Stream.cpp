/**
 * @file        WholeDomainOutputHDF5Stream.cpp
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file of the class saving RealMatrix data
 *              into the output HDF5 file.
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        28 August   2014, 11:15 (created)
 *              04 December 2014, 18:29 (revised)
 *
 * @todo        review the whole class!!
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

#include <OutputHDF5Streams/WholeDomainOutputHDF5Stream.h>
#include <Parameters/Parameters.h>

using namespace std;

//--------------------------------------------------------------------------//
//                              Constants                                   //
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
//                              Definitions                                 //
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
//                              Implementation                              //
//                              public methods                              //
//--------------------------------------------------------------------------//

/**
 * Constructor - links the HDF5 dataset and SourceMatrix
 * @param HDF5_File        - HDF5 file to write the output to
 * @param HDF5_DatasetName - The name of the HDF5 group. This group contains datasets for particular cuboids
 * @param SourceMatrix     - Source matrix to be sampled
 * @param ReductionOp      - Reduction operator
 * @param BufferToReuse    - If there is a memory space to be reused, provide a pointer
 */
TWholeDomainOutputHDF5Stream::TWholeDomainOutputHDF5Stream(THDF5_File &             HDF5_File,
                                                           const char *             HDF5_DatasetName,
                                                           TRealMatrix &            SourceMatrix,
                                                           const TReductionOperator ReductionOp,
                                                           float *                  BufferToReuse)
        : TBaseOutputHDF5Stream(HDF5_File, HDF5_DatasetName, SourceMatrix, ReductionOp, BufferToReuse),
          HDF5_DatasetId(H5I_BADID),
          SampledTimeStep(0)
{

}// end of TWholeDomainOutputHDF5Stream
//------------------------------------------------------------------------------


/**
 * Destructor
 * if the file is still opened, it applies the post processing and flush the data.
 * Then, the object memory is freed and the object destroyed.
 */
TWholeDomainOutputHDF5Stream::~TWholeDomainOutputHDF5Stream()
{
  Close();
  // free memory only if it was allocated
  if (!BufferReuse) FreeMemory();
}// end of Destructor
//------------------------------------------------------------------------------



/**
 * Create a HDF5 stream for the whole domain and allocate data for it.
 */
void TWholeDomainOutputHDF5Stream::Create()
{
  TDimensionSizes ChunkSize(SourceMatrix.GetDimensionSizes().X, SourceMatrix.GetDimensionSizes().Y, 1);

  // Create a dataset under the root group
  HDF5_DatasetId = HDF5_File.CreateFloatDataset(HDF5_File.GetRootGroup(),
                                                HDF5_RootObjectName,
                                                SourceMatrix.GetDimensionSizes(),
                                                ChunkSize,
                                                TParameters::GetInstance()->GetCompressionLevel());

  // Write dataset parameters
  HDF5_File.WriteMatrixDomainType(HDF5_File.GetRootGroup(),
                                  HDF5_RootObjectName,
                                  THDF5_File::hdf5_mdt_real);
  HDF5_File.WriteMatrixDataType  (HDF5_File.GetRootGroup(),
                                  HDF5_RootObjectName,
                                  THDF5_File::hdf5_mdt_float);

  // Set buffer size
  BufferSize = SourceMatrix.GetTotalElementCount();

  // Allocate memory if needed
  if (!BufferReuse) AllocateMemory();
}//end of Create
//------------------------------------------------------------------------------


/**
 * Reopen the output stream after restart and reload data
 */
void TWholeDomainOutputHDF5Stream::Reopen()
{
  TParameters * Params = TParameters::GetInstance();

  // Set buffer size
  BufferSize = SourceMatrix.GetTotalElementCount();

  // Allocate memory if needed
  if (!BufferReuse) AllocateMemory();

  // Open the dataset under the root group
  HDF5_DatasetId = HDF5_File.OpenDataset(HDF5_File.GetRootGroup(),
                                         HDF5_RootObjectName);

  SampledTimeStep = 0;
  if (ReductionOp == roNONE)
  { // seek in the dataset
    SampledTimeStep = ((Params->Get_t_index() - Params->GetStartTimeIndex()) < 0) ?
                          0 : (Params->Get_t_index() - Params->GetStartTimeIndex());
  }
  else
  { // reload data
    HDF5_File.ReadCompleteDataset(HDF5_File.GetRootGroup(),
                                  HDF5_RootObjectName,
                                  SourceMatrix.GetDimensionSizes(),
                                  StoreBuffer);
  }
}// end of Reopen
//------------------------------------------------------------------------------


/**
 * Sample all grid points, line them up in the buffer an flush to the disk unless
 * a reduction operator is applied
 */
void TWholeDomainOutputHDF5Stream::Sample()
{

  SourceMatrix.CopyFromDevice();

  const float * SourceData = SourceMatrix.GetRawData();

  switch (ReductionOp)
  {
    case roNONE :
    {
      /* We use here direct HDF5 offload using MEMSPACE - seems to be faster for bigger datasets*/
      const TDimensionSizes DatasetPosition(0,0,0,SampledTimeStep); //4D position in the dataset

      TDimensionSizes CuboidSize(SourceMatrix.GetDimensionSizes());// Size of the cuboid
      CuboidSize.T = 1;

      // iterate over all cuboid to be sampled
      HDF5_File.WriteCuboidToHyperSlab(HDF5_DatasetId,
                                       DatasetPosition,
                                       TDimensionSizes(0,0,0,0), // position in the SourceMatrix
                                       CuboidSize,
                                       SourceMatrix.GetDimensionSizes(),
                                       SourceMatrix.GetRawData());

      SampledTimeStep++;   // Move forward in time

      break;
    }// case roNONE

    case roRMS  :
    {
      #pragma omp parallel for if (BufferSize > MinGridpointsToSampleInParallel)
      for (size_t i = 0; i < BufferSize; i++)
      {
        StoreBuffer[i] += (SourceData[i] * SourceData[i]);
      }
      break;
    }// case roRMS

    case roMAX  :
    {
      #pragma omp parallel for if (BufferSize > MinGridpointsToSampleInParallel)
      for (size_t i = 0; i < BufferSize; i++)
      {
        if (StoreBuffer[i] < SourceData[i])  StoreBuffer[i] = SourceData[i];
      }
      break;
    }//case roMAX

    case roMIN  :
    {
      #pragma omp parallel for if (BufferSize > MinGridpointsToSampleInParallel)
      for (size_t i = 0; i < BufferSize; i++)
      {
        if (StoreBuffer[i] > SourceData[i]) StoreBuffer[i] = SourceData[i];
      }
      break;
    } //case roMIN
  }// switch
}// end of Sample
//------------------------------------------------------------------------------


/**
 * Apply post-processing on the buffer and flush it to the file.
 */
void TWholeDomainOutputHDF5Stream::PostProcess()
{
  // run inherited method
  TBaseOutputHDF5Stream::PostProcess();
  // When no reduction operator is applied, the data is flushed after every time step
  if (ReductionOp != roNONE) FlushBufferToFile();
}// end of PostProcessing
//------------------------------------------------------------------------------

/**
 * Checkpoint the stream.
 */
void TWholeDomainOutputHDF5Stream::Checkpoint()
{
  // raw data has already been flushed, others has to be flushed here
  if (ReductionOp != roNONE) FlushBufferToFile();
}// end of Checkpoint
//------------------------------------------------------------------------------

/**
 * Close stream (apply post-processing if necessary, flush data and close).
 */
void TWholeDomainOutputHDF5Stream::Close()
{
  // the dataset is still opened
  if (HDF5_DatasetId != H5I_BADID)
  {
    HDF5_File.CloseDataset(HDF5_DatasetId);
  }

  HDF5_DatasetId = H5I_BADID;
}// end of Close
//------------------------------------------------------------------------------


//----------------------------------------------------------------------------//
//                TWholeDomainOutputHDF5Stream implementation                 //
//                            protected methods                               //
//----------------------------------------------------------------------------//


/**
 * Flush the buffer down to the file at the actual position.
 */
void TWholeDomainOutputHDF5Stream::FlushBufferToFile()
{
  SourceMatrix.CopyFromDevice();

  TDimensionSizes Size = SourceMatrix.GetDimensionSizes();
  TDimensionSizes Position(0,0,0);

  // Not used for roNONE now!
  if (ReductionOp == roNONE)
  {
      Position.T = SampledTimeStep;
      Size.T = SampledTimeStep;
  }

  HDF5_File.WriteHyperSlab(HDF5_DatasetId,
                           Position,
                           Size,
                           StoreBuffer);
  SampledTimeStep++;
}// end of FlushToFile
//------------------------------------------------------------------------------


