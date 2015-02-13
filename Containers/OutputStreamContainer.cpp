 /**
 * @file        OutputStreamContainer.cpp
 * @author      Jiri Jaros & Beau Johnston \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file for the output stream container.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        04 December  2014, 11:41 (created)
 *              09 February  2015, 19:56 (revised)
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

#include <Parameters/Parameters.h>
#include <Containers/OutputStreamContainer.h>

#include <OutputHDF5Streams/BaseOutputHDF5Stream.h>
#include <OutputHDF5Streams/IndexOutputHDF5Stream.h>
#include <OutputHDF5Streams/WholeDomainOutputHDF5Stream.h>


//============================================================================//
//                        TOutputStreamContainer                              //
//============================================================================//

//----------------------------------------------------------------------------//
//--------------------------- Public methods ---------------------------------//
//----------------------------------------------------------------------------//


/**
 * Destructor.
 */
TOutputStreamContainer::~TOutputStreamContainer()
{
  OutputStreamContainer.clear();
}// end of Destructor
//------------------------------------------------------------------------------

/**
 * Add all streams in simulation in the container, set all streams records here!
 * Please note, the Matrix container has to be populated before calling this routine.
 *
 * @param [in] MatrixContainer - matrix container to link the steams with
 *                               sampled matrices and sensor masks
 */
void TOutputStreamContainer::AddStreamsIntoContainer(TMatrixContainer & MatrixContainer)
{
  TParameters * Params = TParameters::GetInstance();

  //----------------------------- pressure  ----------------------------------//
  if (Params->IsStore_p_raw())
  {
    OutputStreamContainer[p_sensor_raw] = CreateNewOutputStream(MatrixContainer,
                                                                p,
                                                                p_Name,
                                                                TBaseOutputHDF5Stream::roNONE);
  }// IsStore_p_raw

  if (Params->IsStore_p_rms())
  {
    OutputStreamContainer[p_sensor_rms] = CreateNewOutputStream(MatrixContainer,
                                                                p,
                                                                p_rms_Name,
                                                                TBaseOutputHDF5Stream::roRMS);
  }

  if (Params->IsStore_p_max())
  {
    OutputStreamContainer[p_sensor_max] = CreateNewOutputStream(MatrixContainer,
                                                                p,
                                                                p_max_Name,
                                                                TBaseOutputHDF5Stream::roMAX);
  }

  if (Params->IsStore_p_min())
  {
    OutputStreamContainer[p_sensor_min] = CreateNewOutputStream(MatrixContainer,
                                                                p,
                                                                p_min_Name,
                                                                TBaseOutputHDF5Stream::roMIN);
  }

  if (Params->IsStore_p_max_all())
  {
    OutputStreamContainer[p_sensor_max_all] =
            new TWholeDomainOutputHDF5Stream(Params->HDF5_OutputFile,
                                             p_max_all_Name,
                                             MatrixContainer.GetMatrix<TRealMatrix>(p),
                                             TBaseOutputHDF5Stream::roMAX);
  }

  if (Params->IsStore_p_min_all())
  {
    OutputStreamContainer[p_sensor_min_all] =
            new TWholeDomainOutputHDF5Stream(Params->HDF5_OutputFile,
                                             p_min_all_Name,
                                             MatrixContainer.GetMatrix<TRealMatrix>(p),
                                             TBaseOutputHDF5Stream::roMIN);
  }

  //----------------------------- velocity  ----------------------------------//
  if (Params->IsStore_u_raw())
  {
    OutputStreamContainer[ux_sensor_raw] = CreateNewOutputStream(MatrixContainer,
                                                                 ux_sgx,
                                                                 ux_Name,
                                                                 TBaseOutputHDF5Stream::roNONE);
    OutputStreamContainer[uy_sensor_raw] = CreateNewOutputStream(MatrixContainer,
                                                                 uy_sgy,
                                                                 uy_Name,
                                                                 TBaseOutputHDF5Stream::roNONE);
    OutputStreamContainer[uz_sensor_raw] = CreateNewOutputStream(MatrixContainer,
                                                                 uz_sgz,
                                                                 uz_Name,
                                                                 TBaseOutputHDF5Stream::roNONE);
  }

  if (Params->IsStore_u_rms())
  {
    OutputStreamContainer[ux_sensor_rms] = CreateNewOutputStream(MatrixContainer,
                                                                 ux_sgx,
                                                                 ux_rms_Name,
                                                                 TBaseOutputHDF5Stream::roRMS);
    OutputStreamContainer[uy_sensor_rms] = CreateNewOutputStream(MatrixContainer,
                                                                 uy_sgy,
                                                                 uy_rms_Name,
                                                                 TBaseOutputHDF5Stream::roRMS);
    OutputStreamContainer[uz_sensor_rms] = CreateNewOutputStream(MatrixContainer,
                                                                 uz_sgz,
                                                                 uz_rms_Name,
                                                                 TBaseOutputHDF5Stream::roRMS);
  }

   if (Params->IsStore_u_max())
  {
    OutputStreamContainer[ux_sensor_max] = CreateNewOutputStream(MatrixContainer,
                                                                 ux_sgx,
                                                                 ux_max_Name,
                                                                 TBaseOutputHDF5Stream::roMAX);
    OutputStreamContainer[uy_sensor_max] = CreateNewOutputStream(MatrixContainer,
                                                                 uy_sgy,
                                                                 uy_max_Name,
                                                                 TBaseOutputHDF5Stream::roMAX);
    OutputStreamContainer[uz_sensor_max] = CreateNewOutputStream(MatrixContainer,
                                                                 uz_sgz,
                                                                 uz_max_Name,
                                                                 TBaseOutputHDF5Stream::roMAX);
  }

  if (Params->IsStore_u_min())
  {
    OutputStreamContainer[ux_sensor_min] = CreateNewOutputStream(MatrixContainer,
                                                                 ux_sgx,
                                                                 ux_min_Name,
                                                                 TBaseOutputHDF5Stream::roMIN);
    OutputStreamContainer[uy_sensor_min] = CreateNewOutputStream(MatrixContainer,
                                                                 uy_sgy,
                                                                 uy_min_Name,
                                                                 TBaseOutputHDF5Stream::roMIN);
    OutputStreamContainer[uz_sensor_min] = CreateNewOutputStream(MatrixContainer,
                                                                 uz_sgz,
                                                                 uz_min_Name,
                                                                 TBaseOutputHDF5Stream::roMIN);
  }

  if (Params->IsStore_u_max_all())
  {
    OutputStreamContainer[ux_sensor_max_all] =
            new TWholeDomainOutputHDF5Stream(Params->HDF5_OutputFile,
                                             ux_max_all_Name,
                                             MatrixContainer.GetMatrix<TRealMatrix>(ux_sgx),
                                             TBaseOutputHDF5Stream::roMAX);
    OutputStreamContainer[uy_sensor_max_all] =
            new TWholeDomainOutputHDF5Stream(Params->HDF5_OutputFile,
                                             uy_max_all_Name,
                                             MatrixContainer.GetMatrix<TRealMatrix>(uy_sgy),
                                             TBaseOutputHDF5Stream::roMAX);
    OutputStreamContainer[uz_sensor_max_all] =
            new TWholeDomainOutputHDF5Stream(Params->HDF5_OutputFile,
                                             uz_max_all_Name,
                                             MatrixContainer.GetMatrix<TRealMatrix>(uz_sgz),
                                             TBaseOutputHDF5Stream::roMAX);
  }

  if (Params->IsStore_u_min_all())
  {
    OutputStreamContainer[ux_sensor_min_all] =
            new TWholeDomainOutputHDF5Stream(Params->HDF5_OutputFile,
                                             ux_min_all_Name,
                                             MatrixContainer.GetMatrix<TRealMatrix>(ux_sgx),
                                             TBaseOutputHDF5Stream::roMIN);
    OutputStreamContainer[uy_sensor_min_all] =
            new TWholeDomainOutputHDF5Stream(Params->HDF5_OutputFile,
                                             uy_min_all_Name,
                                             MatrixContainer.GetMatrix<TRealMatrix>(uy_sgy),
                                             TBaseOutputHDF5Stream::roMIN);
    OutputStreamContainer[uz_sensor_min_all] =
            new TWholeDomainOutputHDF5Stream(Params->HDF5_OutputFile,
                                             uz_min_all_Name,
                                             MatrixContainer.GetMatrix<TRealMatrix>(uz_sgz),
                                             TBaseOutputHDF5Stream::roMIN);
  }
}// end of AddStreamsdIntoContainer
//------------------------------------------------------------------------------

/**
 * Create all streams.
 */
void TOutputStreamContainer::CreateStreams()
{
  for (auto it = OutputStreamContainer.begin(); it != OutputStreamContainer.end(); it++)
  {
    if (it->second)
    {
      (it->second)->Create();
    }
  }
}// end of CreateStreams
//------------------------------------------------------------------------------

/**
 * Reopen all streams after restarting form checkpoint.
 */
void TOutputStreamContainer::ReopenStreams()
{
  for (auto it = OutputStreamContainer.begin(); it != OutputStreamContainer.end(); it++)
  {
    if (it->second)
    {
      (it->second)->Reopen();
    }
  }
}// end of ReopenStreams
//------------------------------------------------------------------------------


/**
 * Sample all streams.
 * @warning In GPU implementation, no data is flushed on disk (just data is sampled)
 */
void TOutputStreamContainer::SampleStreams()
{
  for (auto it = OutputStreamContainer.begin(); it != OutputStreamContainer.end(); it++)
  {
    if (it->second)
    {
      (it->second)->Sample();
    }
  }
}// end of SampleStreams
//------------------------------------------------------------------------------

/**
 * Flush stream data to disk
 * @warning In GPU implementation, data from raw streams is flushed here. Aggregated
 * streams are ignored.
 */
void TOutputStreamContainer::FlushRawStreams()
{
  for (auto it = OutputStreamContainer.begin(); it != OutputStreamContainer.end(); it++)
  {
    if (it->second)
    {
      (it->second)->FlushRaw();
    }
  }
}// end of SampleStreams
//------------------------------------------------------------------------------

/**
 * Checkpoint streams without post-processing (flush to the file).
 */
void TOutputStreamContainer::CheckpointStreams()
{
  for (auto it = OutputStreamContainer.begin(); it != OutputStreamContainer.end(); it++)
  {
    if (it->second)
    {
      (it->second)->Checkpoint();
    }
  }
}// end of CheckpointStreams
//------------------------------------------------------------------------------

/**
 * /// Post-process all streams and flush them to the file
 */
void TOutputStreamContainer::PostProcessStreams()
{
  for (auto it = OutputStreamContainer.begin(); it != OutputStreamContainer.end(); it++)
  {
    if (it->second)
    {
      (it->second)->PostProcess();
    }
  }
}// end of CheckpointStreams
//------------------------------------------------------------------------------


/**
 * Close all streams (apply post-processing if necessary, flush data and close).
 */
void TOutputStreamContainer::CloseStreams()
{
  for (auto it = OutputStreamContainer.begin(); it != OutputStreamContainer.end(); it++)
  {
    if (it->second)
    {
      (it->second)->Close();
    }
  }
}// end of CloseStreams
//------------------------------------------------------------------------------

/**
 *  Free all streams - destroy them.
 */
void TOutputStreamContainer::FreeStreams()
{
  for (auto it = OutputStreamContainer.begin(); it != OutputStreamContainer.end(); it++)
  {
    if (it->second)
    {
      delete it->second;
    }
  }
  OutputStreamContainer.clear();
}// end of FreeAllStreams
//------------------------------------------------------------------------------


//----------------------------------------------------------------------------//
//--------------------------- Protected methods ------------------------------//
//----------------------------------------------------------------------------//


/**
 * Create a new output stream
 * @param [in] MatrixContainer  - name of the HDF5 dataset or group
 * @param [in] SampledMatrixID  - code id of the matrix
 * @param [in] HDF5_DatasetName - name of the HDF5 dataset or group
 * @param [in] ReductionOp      - reduction operator
 *
 * @return - new output stream with defined links
 *
 * @todo implement CUBOID streams!!
 */
TBaseOutputHDF5Stream * TOutputStreamContainer::CreateNewOutputStream(TMatrixContainer & MatrixContainer,
                                                                      const TMatrixID    SampledMatrixID,
                                                                      const char *       HDF5_DatasetName,
                                                                      const TBaseOutputHDF5Stream::TReductionOperator  ReductionOp)
{
  TParameters * Params = TParameters::GetInstance();

  TBaseOutputHDF5Stream * Stream = NULL;

  if (Params->Get_sensor_mask_type() == TParameters::smt_index)
  {
    Stream = new TIndexOutputHDF5Stream(Params->HDF5_OutputFile,
                                        HDF5_DatasetName,
                                        MatrixContainer.GetMatrix<TRealMatrix>(SampledMatrixID),
                                        MatrixContainer.GetMatrix<TIndexMatrix>(sensor_mask_index),
                                        ReductionOp);
  }
    /*
    }
    else
    {
        Stream = new TCuboidOutputHDF5Stream(Params->HDF5_OutputFile,
                HDF5_DatasetName,
                MatrixContainer.GetRealMatrix(SampledMatrixID),
                MatrixContainer.GetLongMatrix(sensor_mask_corners),
                ReductionOp,
                BufferToReuse);
    }
    */
  return Stream;
}// end of CreateNewOutputStream
//------------------------------------------------------------------------------




//----------------------------------------------------------------------------//
//--------------------------- Private methods --------------------------------//
//----------------------------------------------------------------------------//
