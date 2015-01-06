/**
 * @file        HDF5_File.cpp
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the HDF5 related classes.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        27 July     2012, 14:14 (created) \n
 *              06 November 2014, 16:30 (revised)
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


#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <time.h>

// Linux build
#ifdef __linux__
  #include <unistd.h>
#endif

//Windows 64 build
#ifdef _WIN64
  #include<stdio.h>
  #include<Winsock2.h>
  #pragma comment(lib, "Ws2_32.lib")
#endif

#include <HDF5/HDF5_File.h>
#include <Parameters/Parameters.h>
#include <Utils/ErrorMessages.h>

//----------------------------------------------------------------------------//
//-----------------------------    Constants----------------------------------//
//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//

const char * THDF5_File::HDF5_MatrixDomainTypeName    = "domain_type";
const char * THDF5_File::HDF5_MatrixDataTypeName      = "data_type";

const string THDF5_File::HDF5_MatrixDomainTypeNames[] = {"real","complex"};
const string THDF5_File::HDF5_MatrixDataTypeNames[]   = {"float","long"};

const string THDF5_FileHeader::HDF5_FileTypesNames[]  = {"input", "output", "checkpoint", "unknown"};

const string THDF5_FileHeader::HDF5_MajorFileVersionsNames[] = {"1"};
const string THDF5_FileHeader::HDF5_MinorFileVersionsNames[] = {"0","1"};

//----------------------------------------------------------------------------//
//----------------------------    THDF5_File    ------------------------------//
//------------------------------    Public     -------------------------------//
//----------------------------------------------------------------------------//

/**
 * Constructor.
 */
THDF5_File::THDF5_File() :
    HDF5_FileId(H5I_BADID), FileName("")
{

}// end of constructor
//------------------------------------------------------------------------------


/**
 * Create the HDF5 file.
 * @param [in] FileName - File name
 * @param [in] Flags    - Flags for the HDF5 runtime
 * @throw ios:failure if error happened.
 *
 */
void THDF5_File::Create(const char * FileName,
                        unsigned int Flags)
{
  // file is opened
  if (IsOpened())
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_FileCannotRecreated, FileName);
    throw ios::failure(ErrorMessage);
  }

  // Create a new file using default properties.
  this->FileName = FileName;

  HDF5_FileId = H5Fcreate(FileName, Flags, H5P_DEFAULT, H5P_DEFAULT);

  if (HDF5_FileId < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_FileNotCreated, FileName);
    throw ios::failure(ErrorMessage);
  }
}// end of Create
//------------------------------------------------------------------------------


/**
 * Open the HDF5 file.
 * @param [in] FileName
 * @param [in] Flags    - flags for the HDF5 runtime
 * @throw ios:failure if error happened.
 *
 */
void THDF5_File::Open(const char * FileName,
                      unsigned int Flags)
{
  if (IsOpened())
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_FileCannotReopen, FileName);
    throw ios::failure(ErrorMessage);
  };

  this->FileName = FileName;

  if (H5Fis_hdf5(FileName) == 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_NotHDF5File, FileName);
    throw ios::failure(ErrorMessage);
  }

  HDF5_FileId = H5Fopen(FileName, Flags, H5P_DEFAULT);

  if (HDF5_FileId < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_FileNotOpened, FileName);
    throw ios::failure(ErrorMessage);
  }
}// end of Open
//------------------------------------------------------------------------------

/**
 * Close the HDF5 file.
 */
void THDF5_File::Close()
{
  // Terminate access to the file.
  herr_t status = H5Fclose(HDF5_FileId);
  if (status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_FileNotClosed, FileName.c_str());

    throw ios::failure(ErrorMessage);
  }

  FileName = "";
  HDF5_FileId = H5I_BADID;
}// end of Close
//------------------------------------------------------------------------------

/**
 * Destructor.
 */
THDF5_File::~THDF5_File()
{
  if (IsOpened()) Close();
}//end of ~THDF5_File
//------------------------------------------------------------------------------


/**
 * Create a HDF5 group at a specified place in the file tree.
 * @param [in] ParentGroup  - Where to link the group at
 * @param [in] GroupName - Group name
 * @return a handle to the new group
 */
hid_t THDF5_File::CreateGroup(const hid_t  ParentGroup,
                              const char * GroupName)
{
  hid_t HDF5_group_id = H5Gcreate(ParentGroup,
                                  GroupName,
                                  H5P_DEFAULT,
                                  H5P_DEFAULT,
                                  H5P_DEFAULT);

  //if error
  if (HDF5_group_id == H5I_INVALID_HID)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage,HDF5_ERR_FMT_GroupNotCreated,GroupName, FileName.c_str());
    throw ios::failure(ErrorMessage);
  }

  return HDF5_group_id;
};// end of CreateGroup


/**
 * Open a HDF5 group at a specified place in the file tree.
 * @param [in] ParentGroup - parent group
 * @param [in] GroupName
 * @return
 */
hid_t THDF5_File::OpenGroup(const hid_t ParentGroup,
                            const char * GroupName)
{
  hid_t HDF5_group_id = H5Gopen(ParentGroup, GroupName, H5P_DEFAULT);

  //if error
  if (HDF5_group_id == H5I_INVALID_HID)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage,HDF5_ERR_FMT_GroupNotOpened,GroupName, FileName.c_str());
    throw ios::failure(ErrorMessage);
  }

  return HDF5_group_id;
}// end of OpenGroup
//------------------------------------------------------------------------------


/**
 * Close a group.
 * @param[in] Group
 */
void THDF5_File::CloseGroup(const hid_t HDF5_group_id)
{
  H5Gclose(HDF5_group_id);
}// end of CloseGroup
//------------------------------------------------------------------------------

/**
 * Open the dataset  at a specified place in the file tree.
 * @param [in] ParentGroup
 * @param [in] DatasetName
 * @return     Dataset id
 * @throw ios::failure
 */
hid_t THDF5_File::OpenDataset(const hid_t  ParentGroup,
                              const char * DatasetName)
{
  // Open dataset
  hid_t HDF5_dataset_id = H5Dopen(ParentGroup, DatasetName, H5P_DEFAULT);

  if (HDF5_dataset_id == H5I_INVALID_HID)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_DatasetNotOpened, FileName.c_str(), DatasetName);
    throw ios::failure(ErrorMessage);
  }

  return HDF5_dataset_id;
}// end of OpenDataset
//------------------------------------------------------------------------------

/**
 * Create the HDF5 dataset at a specified place in the file tree.
 * @param [in] ParentGroup       - Parent group
 * @param [in] DatasetName       - Dataset name
 * @param [in] DimensionSizes    - Dimension sizes
 * @param [in] ChunkSizes        - Chunk sizes
 * @param [in] CompressionLevel  - Compression level
 * @return a handle to the new dataset
 */
hid_t THDF5_File::CreateFloatDataset(const hid_t             ParentGroup,
                                     const char            * DatasetName,
                                     const TDimensionSizes & DimensionSizes,
                                     const TDimensionSizes & ChunkSizes,
                                     const size_t            CompressionLevel)
{
  const int RANK = (DimensionSizes.Is3D()) ? 3 : 4;

// a windows hack
  hsize_t Dims [4];
  hsize_t Chunk[4];

  // 3D dataset
  if (DimensionSizes.Is3D())
  {
    Dims[0] = DimensionSizes.Z;
    Dims[1] = DimensionSizes.Y;
    Dims[2] = DimensionSizes.X;

    Chunk[0] = ChunkSizes.Z;
    Chunk[1] = ChunkSizes.Y;
    Chunk[2] = ChunkSizes.X;
  }
  else // 4D dataset
  {
    Dims[0] = DimensionSizes.T;
    Dims[1] = DimensionSizes.Z;
    Dims[2] = DimensionSizes.Y;
    Dims[3] = DimensionSizes.X;

    Chunk[0] = ChunkSizes.T;
    Chunk[1] = ChunkSizes.Z;
    Chunk[2] = ChunkSizes.Y;
    Chunk[3] = ChunkSizes.X;
  }

  hid_t Property_list;
  herr_t Status;

  hid_t dataspace_id = H5Screate_simple(RANK, Dims, NULL);

  // set chunk size
  Property_list = H5Pcreate(H5P_DATASET_CREATE);

  Status = H5Pset_chunk(Property_list, RANK, Chunk);
  if (Status < 0)
  {
      char ErrorMessage[256];
      sprintf(ErrorMessage, HDF5_ERR_FMT_DatasetNotOpened, FileName.c_str(), DatasetName);
      throw ios::failure(ErrorMessage);
  }

  // set compression level
  Status = H5Pset_deflate(Property_list, CompressionLevel);
  if (Status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotSetCompression, FileName.c_str(), DatasetName, CompressionLevel);
    throw ios::failure(ErrorMessage);
  }

  // create dataset
  hid_t HDF5_dataset_id = H5Dcreate(ParentGroup, DatasetName, H5T_NATIVE_FLOAT, dataspace_id,
                                    H5P_DEFAULT, Property_list, H5P_DEFAULT);

  if (HDF5_dataset_id == H5I_INVALID_HID)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_DatasetNotOpened, FileName.c_str(), DatasetName);
    throw ios::failure(ErrorMessage);
  }

  H5Pclose(Property_list);

  return HDF5_dataset_id;
}// end of CreateFloatDataset
//------------------------------------------------------------------------------


/**
 * Create the HDF5 dataset at a specified place in the file tree (always 3D).
 *
 * @param [in] ParentGroup       - Parent group
 * @param [in] DatasetName       - Dataset name
 * @param [in] DimensionSizes    - Dimension sizes
 * @param [in] ChunkSizes        - Chunk sizes
 * @param [in] CompressionLevel  - Compression level
 * @return a handle to the new dataset
 */
hid_t THDF5_File::CreateIndexDataset(const hid_t ParentGroup,
                                     const char            * DatasetName,
                                     const TDimensionSizes & DimensionSizes,
                                     const TDimensionSizes & ChunkSizes,
                                     const size_t            CompressionLevel)
{

  const int RANK = 3;

  hsize_t Dims [RANK] = {DimensionSizes.Z, DimensionSizes.Y, DimensionSizes.X};
  hsize_t Chunk[RANK] = {ChunkSizes.Z, ChunkSizes.Y, ChunkSizes.X};

  hid_t Property_list;
  herr_t Status;

  hid_t Dataspace_id = H5Screate_simple(RANK, Dims, NULL);

  // set chunk size
  Property_list = H5Pcreate(H5P_DATASET_CREATE);

  Status = H5Pset_chunk(Property_list, RANK, Chunk);
  if (Status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_DatasetNotOpened, FileName.c_str(), DatasetName);
    throw ios::failure(ErrorMessage);
  }

  // set compression level
  Status = H5Pset_deflate(Property_list, CompressionLevel);
  if (Status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotSetCompression, FileName.c_str(), DatasetName, CompressionLevel);
    throw ios::failure(ErrorMessage);
  }

  // create dataset
  hid_t HDF5_dataset_id = H5Dcreate(ParentGroup,
                                    DatasetName,
                                    H5T_STD_U64LE,
                                    Dataspace_id,
                                    H5P_DEFAULT,
                                    Property_list,
                                    H5P_DEFAULT);

  if (HDF5_dataset_id == H5I_INVALID_HID)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_DatasetNotOpened, FileName.c_str(), DatasetName);
    throw ios::failure(ErrorMessage);
  }

  H5Pclose(Property_list);

  return HDF5_dataset_id;
}// end of CreateIndexDataset
//------------------------------------------------------------------------------


/**
 * Close dataset.
 * @param [in] HDF5_Dataset_id
 *
 */
void  THDF5_File::CloseDataset(const hid_t HDF5_Dataset_id)
{
  H5Dclose (HDF5_Dataset_id);
}// end of CloseDataset
//------------------------------------------------------------------------------


/**
 * Write a hyperslab into the dataset.
 *
 * @param [in] HDF5_Dataset_id - Dataset id
 * @param [in] Position - Position in the dataset
 * @param [in] Size - Size of the hyperslab
 * @param [in] Data
 * @throw ios::failure
 */
void THDF5_File::WriteHyperSlab(const hid_t HDF5_Dataset_id,
                                const TDimensionSizes & Position,
                                const TDimensionSizes & Size,
                                const float * Data)
{
  herr_t status;
  hid_t HDF5_Filespace, HDF5_Memspace;

  // Get File Space, to find out number of dimensions
  HDF5_Filespace = H5Dget_space(HDF5_Dataset_id);
  const int Rank = H5Sget_simple_extent_ndims(HDF5_Filespace);


  // Select sizes and positions, windows hack
  hsize_t ElementCount[4];
  hsize_t Offset [4];

  // 3D dataset
  if (Rank == 3)
  {
    ElementCount[0] = Size.Z;
    ElementCount[1] = Size.Y;
    ElementCount[2] = Size.X;

    Offset[0] = Position.Z;
    Offset[1] = Position.Y;
    Offset[2] = Position.X;
  }
  else // 4D dataset
  {
    ElementCount[0] = Size.T;
    ElementCount[1] = Size.Z;
    ElementCount[2] = Size.Y;
    ElementCount[3] = Size.X;

    Offset[0] = Position.T;
    Offset[1] = Position.Z;
    Offset[2] = Position.Y;
    Offset[3] = Position.X;
  }

  // select hyperslab
  status = H5Sselect_hyperslab(HDF5_Filespace,
                               H5S_SELECT_SET,
                               Offset,
                               NULL,
                               ElementCount,
                               NULL);
  if (status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(ErrorMessage);
  }

  // assign memspace
  HDF5_Memspace = H5Screate_simple(Rank, ElementCount, NULL);

  status = H5Dwrite(HDF5_Dataset_id,
                    H5T_NATIVE_FLOAT,
                    HDF5_Memspace,
                    HDF5_Filespace,
                    H5P_DEFAULT,
                    Data);
  if (status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(ErrorMessage);
  }

  H5Sclose(HDF5_Memspace);
  H5Sclose(HDF5_Filespace);
}// end of WriteHyperSlab
//------------------------------------------------------------------------------


/**
 * Write hyperslab.
 *
 * @param [in] HDF5_Dataset_id - Dataset id
 * @param [in] Position - Position in the dataset
 * @param [in] Size - Size of the hyperslab
 * @param [in] Data
 * @throw ios::failure
 */
void THDF5_File::WriteHyperSlab(const hid_t HDF5_Dataset_id,
                                const TDimensionSizes & Position,
                                const TDimensionSizes & Size,
                                const size_t * Data)
{
  herr_t status;
  hid_t HDF5_Filespace, HDF5_Memspace;

  // Get File Space, to find out number of dimensions
  HDF5_Filespace = H5Dget_space(HDF5_Dataset_id);
  const int Rank = H5Sget_simple_extent_ndims(HDF5_Filespace);

  // Set sizes and offsets, windows hack
  hsize_t ElementCount[4];
  hsize_t Offset [4];

  // 3D dataset
  if (Rank == 3)
  {
    ElementCount[0] = Size.Z;
    ElementCount[1] = Size.Y;
    ElementCount[2] = Size.X;

    Offset[0] = Position.Z;
    Offset[1] = Position.Y;
    Offset[2] = Position.X;
  }
  else // 4D dataset
  {
    ElementCount[0] = Size.T;
    ElementCount[1] = Size.Z;
    ElementCount[2] = Size.Y;
    ElementCount[3] = Size.X;

    Offset[0] = Position.T;
    Offset[1] = Position.Z;
    Offset[2] = Position.Y;
    Offset[3] = Position.X;
  }

  // select hyperslab
  status = H5Sselect_hyperslab(HDF5_Filespace,
                               H5S_SELECT_SET,
                               Offset,
                               NULL,
                               ElementCount,
                               NULL);
  if (status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(ErrorMessage);
  }


  // assign memspace
  HDF5_Memspace = H5Screate_simple(Rank, ElementCount, NULL);


  status = H5Dwrite(HDF5_Dataset_id,
                    H5T_STD_U64LE,
                    HDF5_Memspace,
                    HDF5_Filespace,
                    H5P_DEFAULT,
                    Data);
  if (status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(ErrorMessage);
  }

  H5Sclose(HDF5_Memspace);
  H5Sclose(HDF5_Filespace);
}// end of WriteHyperSlab
//------------------------------------------------------------------------------


/**
 * Write a cuboid selected inside MatrixData into a Hyperslab.
 * The routine writes 3D cuboid into a 4D dataset (only intended for raw time series).
 * @param [in] HDF5_Dataset_id   - Dataset to write MatrixData into
 * @param [in] HyperslabPosition - Position in the dataset (hyperslab) - may be 3D/4D
 * @param [in] CuboidPosition    - Position of the cuboid in MatrixData (what to sample) - must be 3D
 * @param [in] CuboidSize        - Cuboid size (size of data being sampled) - must by 3D
 * @param [in] MatrixDimensions  - Size of the original matrix (the sampled one)
 * @param [in] MatrixData        - C array of MatrixData
 */
void THDF5_File::WriteCuboidToHyperSlab(const hid_t             HDF5_Dataset_id,
                                        const TDimensionSizes & HyperslabPosition,
                                        const TDimensionSizes & CuboidPosition,
                                        const TDimensionSizes & CuboidSize,
                                        const TDimensionSizes & MatrixDimensions,
                                        const float           * MatrixData)
{
  herr_t status;
  hid_t HDF5_Filespace, HDF5_Memspace;

  const int Rank = 4;

  // Select sizes and positions
  // The T here is always 1 (only one timestep)
  hsize_t SlabSize[Rank] = {1, CuboidSize.Z, CuboidSize.Y, CuboidSize.X};
  hsize_t OffsetInDataset[Rank] = {HyperslabPosition.T, HyperslabPosition.Z, HyperslabPosition.Y, HyperslabPosition.X};
  hsize_t OffsetInMatrixData[] = {CuboidPosition.Z, CuboidPosition.Y, CuboidPosition.X};
  hsize_t MatrixSize [] = {MatrixDimensions.Z, MatrixDimensions.Y, MatrixDimensions.X};


  // select hyperslab in the HDF5 dataset
  HDF5_Filespace = H5Dget_space(HDF5_Dataset_id);
  status = H5Sselect_hyperslab(HDF5_Filespace,
                               H5S_SELECT_SET,
                               OffsetInDataset,
                               NULL,
                               SlabSize,
                               NULL);
  if (status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(ErrorMessage);
  }


  // assign memspace and select the cuboid in the sampled matrix
  HDF5_Memspace = H5Screate_simple(3, MatrixSize, NULL);
  status = H5Sselect_hyperslab(HDF5_Memspace,
                               H5S_SELECT_SET,
                               OffsetInMatrixData,
                               NULL,
                               SlabSize + 1, // Slab size has to be 3D in this case (done by skipping the T dimension)
                               NULL);
  if (status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(ErrorMessage);
  }

  // Write the data
  status = H5Dwrite(HDF5_Dataset_id,
                    H5T_NATIVE_FLOAT,
                    HDF5_Memspace,
                    HDF5_Filespace,
                    H5P_DEFAULT,
                    MatrixData);
  if (status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(ErrorMessage);
  }

  // close memspace and filespace
  H5Sclose(HDF5_Memspace);
  H5Sclose(HDF5_Filespace);
}// end of WriteCuboidToHyperSlab
//------------------------------------------------------------------------------

/**
 * Write sensor data selected by the sensor mask.
 * A routine pick elements from the MatixData based on the Sensor Data and store
 * them into a single hyperslab of size [Nsens, 1, 1].
 * @param [in] HDF5_Dataset_id   - Dataset to write MaatrixData into
 * @param [in] HyperslabPosition - 3D position in the dataset (hyperslab)
 * @param [in] IndexSensorSize   - Size of the index based sensor mask
 * @param [in] IndexSensorData   - Index based sensor mask
 * @param [in] MatrixDimensions  - Size of the sampled matrix
 * @param [in] MatrixData        - Matrix data
 * @warning  - very slow at this version of HDF5 for orthogonal planes-> DO NOT USE
 */
void THDF5_File::WriteSensorByMaskToHyperSlab(const hid_t             HDF5_Dataset_id,
                                              const TDimensionSizes & HyperslabPosition,
                                              const size_t            IndexSensorSize,
                                              const size_t          * IndexSensorData,
                                              const TDimensionSizes & MatrixDimensions,
                                              const float           * MatrixData)
{
  herr_t status;
  hid_t HDF5_Filespace, HDF5_Memspace;

  const int Rank = 3;

  // Select sizes and positions
  // Only one timestep
  hsize_t SlabSize[Rank] = {1, 1, IndexSensorSize};
  hsize_t OffsetInDataset[Rank] = {HyperslabPosition.Z, HyperslabPosition.Y, HyperslabPosition.X};
  // treat as a 1D array
  //hsize_t MatrixSize        []  = {MatrixDimensions.Z  * MatrixDimensions.Y * MatrixDimensions.X};
  hsize_t MatrixSize = MatrixDimensions.Z * MatrixDimensions.Y * MatrixDimensions.X;


  // select hyperslab in the HDF5 dataset
  HDF5_Filespace = H5Dget_space(HDF5_Dataset_id);
  status = H5Sselect_hyperslab(HDF5_Filespace,
                               H5S_SELECT_SET,
                               OffsetInDataset,
                               NULL,
                               SlabSize,
                               NULL);
  if (status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(ErrorMessage);
  }

  // assign 1D memspace and select the elements within the array
  HDF5_Memspace = H5Screate_simple(1, &MatrixSize, NULL);
  status = H5Sselect_elements(HDF5_Memspace,
                              H5S_SELECT_SET,
                              IndexSensorSize,
                              (hsize_t *) (IndexSensorData));
  if (status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(ErrorMessage);
  }

  // Write the data
  status = H5Dwrite(HDF5_Dataset_id,
                    H5T_NATIVE_FLOAT,
                    HDF5_Memspace,
                    HDF5_Filespace,
                    H5P_DEFAULT,
                    MatrixData);
  if (status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotWriteTo, "");

    throw ios::failure(ErrorMessage);
  }

  // close memspace and filespace
  H5Sclose(HDF5_Memspace);
  H5Sclose(HDF5_Filespace);
}// end of WriteSensorbyMaskToHyperSlab
//------------------------------------------------------------------------------
/**
 * Write the scalar value at a specified place in the file tree
 * (no chunks, no compression).
 * @param [in] ParentGroup
 * @param [in] DatasetName
 * @param [in] Value
 */
void THDF5_File::WriteScalarValue(const hid_t  ParentGroup,
                                  const char * DatasetName,
                                  const float  Value)
{
  const int Rank = 3;
  const hsize_t Dims[] = {1, 1, 1};

  hid_t Dataset_id = H5I_INVALID_HID;
  hid_t Dataspace_id = H5I_INVALID_HID;
  herr_t Status;


  if (H5LTfind_dataset(ParentGroup, DatasetName) == 1)
  { // dataset already exists (from previous leg) open it
    Dataset_id = OpenDataset(ParentGroup,DatasetName);
  }
  else
  { // dataset does not exist yet -> create it
    Dataspace_id = H5Screate_simple(Rank, Dims, NULL);
    Dataset_id = H5Dcreate(ParentGroup,
                           DatasetName,
                           H5T_NATIVE_FLOAT,
                           Dataspace_id,
                           H5P_DEFAULT,
                           H5P_DEFAULT,
                           H5P_DEFAULT);
  }

  // was created correctly?
  if (Dataset_id == H5I_INVALID_HID)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotWriteTo, DatasetName);
    throw ios::failure(ErrorMessage);
  }

  Status = H5Dwrite(Dataset_id,
                    H5T_NATIVE_FLOAT,
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    &Value);

  // was written correctly?
  if (Status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotWriteTo, DatasetName);
    throw ios::failure(ErrorMessage);
  }

  WriteMatrixDataType(ParentGroup, DatasetName, hdf5_mdt_float);
  WriteMatrixDomainType(ParentGroup, DatasetName, hdf5_mdt_real);
} // end of WriteScalarValue (float)
//------------------------------------------------------------------------------

/**
 * Write a scalar value at a specified place in the file tree.
 * (no chunks, no compression)
 * @param [in] ParentGroup
 * @param [in] DatasetName
 * @param [in] Value
 *
 */
void THDF5_File::WriteScalarValue(const hid_t  ParentGroup,
                                  const char * DatasetName,
                                  const size_t Value)
{
  const int Rank = 3;
  const hsize_t Dims[] = {1, 1, 1};

  hid_t Dataset_id = H5I_INVALID_HID;
  hid_t Dataspace_id = H5I_INVALID_HID;
  herr_t Error;


  if (H5LTfind_dataset(ParentGroup, DatasetName) == 1)
  { // dataset already exists (from previous leg) open it
    Dataset_id = OpenDataset(ParentGroup,
                             DatasetName);
  }
  else
  { // dataset does not exist yet -> create it
    Dataspace_id = H5Screate_simple(Rank, Dims, NULL);
    Dataset_id = H5Dcreate(ParentGroup,
                           DatasetName,
                           H5T_STD_U64LE,
                           Dataspace_id,
                           H5P_DEFAULT,
                           H5P_DEFAULT,
                           H5P_DEFAULT);
  }

    // was created correctly?
  if (Dataset_id == H5I_INVALID_HID)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotWriteTo, DatasetName);
    throw ios::failure(ErrorMessage);
  }

  Error = H5Dwrite(Dataset_id,
                   H5T_STD_U64LE,
                   H5S_ALL,
                   H5S_ALL,
                   H5P_DEFAULT,
                   &Value);

  // was written correctly?
  if (Error < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotWriteTo, DatasetName);
    throw ios::failure(ErrorMessage);
  }

  WriteMatrixDataType(ParentGroup, DatasetName, hdf5_mdt_long);
  WriteMatrixDomainType(ParentGroup, DatasetName, hdf5_mdt_real);
}// end of WriteScalarValue
//------------------------------------------------------------------------------

/**
 * Read the scalar value under a specified group - float value.
 * @param[in]  ParentGroup
 * @param[in]  DatasetName
 * @param[out] Value
 */
void THDF5_File::ReadScalarValue(const hid_t ParentGroup,
                                 const char * DatasetName,
                                 float      & Value)
{
  ReadCompleteDataset(ParentGroup, DatasetName, TDimensionSizes(1,1,1), &Value);
} // end of ReadScalarValue
//------------------------------------------------------------------------------


/**
 * Read the scalar value under a specified group - index value.
 * @param[in] ParentGroup
 * @param[in] DatasetName
 * @param[out] Value
 */
void THDF5_File::ReadScalarValue(const hid_t  ParentGroup,
                                 const char * DatasetName,
                                 size_t     & Value)
{
  ReadCompleteDataset(ParentGroup, DatasetName, TDimensionSizes(1,1,1), &Value);
}// end of ReadScalarValue
//------------------------------------------------------------------------------



/**
 * Read data from the dataset at a specified place in the file tree.
 *
 * @param [in] ParentGroup
 * @param [in] DatasetName
 * @param [in] DimensionSizes
 * @param [out] Data
 * @throw ios::failure
 */
void THDF5_File::ReadCompleteDataset (const hid_t             ParentGroup,
                                      const char            * DatasetName,
                                      const TDimensionSizes & DimensionSizes,
                                      float                 * Data)
{
    // Check Dimensions sizes
  if (GetDatasetDimensionSizes(ParentGroup, DatasetName).GetElementCount() !=
      DimensionSizes.GetElementCount())
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_WrongDimensionSizes, DatasetName);
    throw ios::failure(ErrorMessage);
  }

  // read dataset
  herr_t status = H5LTread_dataset_float(ParentGroup, DatasetName, Data);
  if (status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotReadFrom, DatasetName);
    throw ios::failure(ErrorMessage);
  }
}// end of ReadDataset (float)
//------------------------------------------------------------------------------


/**
 * Read data from the dataset at a specified place in the file tree.
 *
 * @param [in] ParentGroup
 * @param [in] DatasetName
 * @param [in] DimensionSizes
 * @param [out] Data
 * @throw ios::failure
 */
void THDF5_File::ReadCompleteDataset(const hid_t             ParentGroup,
                                     const char            * DatasetName,
                                     const TDimensionSizes & DimensionSizes,
                                     size_t                * Data)
{
  if (GetDatasetDimensionSizes(ParentGroup, DatasetName).GetElementCount() !=
      DimensionSizes.GetElementCount())
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_WrongDimensionSizes, DatasetName);
    throw ios::failure(ErrorMessage);
  }

  /* read dataset */
  herr_t status = H5LTread_dataset(ParentGroup, DatasetName, H5T_STD_U64LE, Data);
  if (status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotReadFrom, DatasetName);
    throw ios::failure(ErrorMessage);
  }
}// end of ReadCompleteDataset
//------------------------------------------------------------------------------


/**
 * Get dimension sizes of the dataset at a specified place in the file tree.
 *
 * @param [in] ParentGroup
 * @param [in] DatasetName
 * @return DimensionSizes
 * @throw ios::failure
 */
TDimensionSizes THDF5_File::GetDatasetDimensionSizes(const hid_t  ParentGroup,
                                                     const char * DatasetName)
{
  const size_t ndims = GetDatasetNumberOfDimensions(ParentGroup, DatasetName);
  hsize_t dims[4]; //windows hack

  for (size_t i = 0; i < ndims; i++) dims[i] = 0;

  herr_t status = H5LTget_dataset_info(ParentGroup, DatasetName, dims, NULL, NULL);
  if (status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotReadFrom, DatasetName);
    throw ios::failure(ErrorMessage);
  }

  if (ndims == 3)
  {
    return TDimensionSizes(dims[2], dims[1], dims[0]);
  }
  else
  {
    return TDimensionSizes(dims[3], dims[2], dims[1], dims[0]);
  }
}// end of GetDatasetDimensionSizes
//------------------------------------------------------------------------------

/**
 * Get number of dimensions of the dataset  under a specified group.
 * @param [in] ParentGroup
 * @param [in] DatasetName
 * @return  - Number of dimensions
 */
size_t THDF5_File::GetDatasetNumberOfDimensions(const hid_t  ParentGroup,
                                                const char * DatasetName)
{
  int dims = 0;

  herr_t status = H5LTget_dataset_ndims(ParentGroup, DatasetName, &dims);
  if (status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotReadFrom, DatasetName);
    throw ios::failure(ErrorMessage);
  }

  return dims;
}// end of GetDatasetNumberOfDimensions
//------------------------------------------------------------------------------


/**
 * Get dataset element count at a specified place in the file tree.
 *
 * @param [in] ParentGroup
 * @param [in] DatasetName
 * @return Number of elements
 * @throw ios::failure
 */
size_t THDF5_File::GetDatasetElementCount(const hid_t  ParentGroup,
                                          const char * DatasetName)
{
  hsize_t dims[3] = {0, 0, 0};

  herr_t status = H5LTget_dataset_info(ParentGroup, DatasetName, dims, NULL, NULL);
  if (status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotReadFrom, DatasetName);
    throw ios::failure(ErrorMessage);
  }

  return dims[0] * dims[1] * dims[2];
}// end of GetDatasetElementCount
//-----------------------------------------------------------------------------


/**
 * Write matrix data type into the dataset at a specified place in the file tree.
 *
 * @param [in] ParentGroup
 * @param [in] DatasetName
 * @param [in] MatrixDataType
 */
void THDF5_File::WriteMatrixDataType(const hid_t                  ParentGroup,
                                     const char                 * DatasetName,
                                     const THDF5_MatrixDataType & MatrixDataType)
{
  WriteStringAttribute(ParentGroup,
                       DatasetName,
                       HDF5_MatrixDataTypeName,
                       HDF5_MatrixDataTypeNames[MatrixDataType]);
}// end of WriteMatrixDataType
//------------------------------------------------------------------------------


/**
 * Write matrix data type into the dataset at a specified place in the file tree.
 *
 * @param [in] ParentGroup
 * @param [in] DatasetName
 * @param [in] MatrixDomainType
 */
void THDF5_File::WriteMatrixDomainType(const hid_t                    ParentGroup,
                                       const char                   * DatasetName,
                                       const THDF5_MatrixDomainType & MatrixDomainType)
{
  WriteStringAttribute(ParentGroup,
                       DatasetName,
                       HDF5_MatrixDomainTypeName,
                       HDF5_MatrixDomainTypeNames[MatrixDomainType]);
}// end of WriteMatrixDomainType
//------------------------------------------------------------------------------


/**
 * Read matrix data type from the dataset at a specified place in the file tree.
 *
 * @param [in] ParentGroup
 * @param [in] DatasetName
 * @return     MatrixDataType
 * @throw ios::failure
 */
THDF5_File::THDF5_MatrixDataType THDF5_File::ReadMatrixDataType(const hid_t  ParentGroup,
                                                                const char * DatasetName)
{
  string ParamValue;

  ParamValue = ReadStringAttribute(ParentGroup, DatasetName, HDF5_MatrixDataTypeName);

  if (ParamValue == HDF5_MatrixDataTypeNames[0])
  {
    return static_cast<THDF5_MatrixDataType>(0);
  }
  if (ParamValue == HDF5_MatrixDataTypeNames[1])
  {
    return static_cast<THDF5_MatrixDataType>(1);
  }

  char ErrorMessage[256];
  sprintf(ErrorMessage, HDF5_ERR_FMT_BadAttributeValue, DatasetName, HDF5_MatrixDataTypeName, ParamValue.c_str());
  throw ios::failure(ErrorMessage);

  // this will never be executed (just to prevent warning)
  return (THDF5_MatrixDataType) 0;

}// end of ReadMatrixDataType
//------------------------------------------------------------------------------


/**
 * Read matrix dataset domain type at a specified place in the file tree.
 *
 * @param [in] ParentGroup
 * @param [in] DatasetName
 * @return     DomainType
 * @throw ios::failure
 */
THDF5_File::THDF5_MatrixDomainType THDF5_File::ReadMatrixDomainType(const hid_t  ParentGroup,
                                                                    const char * DatasetName)
{
  string ParamValue;

  ParamValue = ReadStringAttribute(ParentGroup, DatasetName, HDF5_MatrixDomainTypeName);

  if (ParamValue == HDF5_MatrixDomainTypeNames[0])
  {
    return static_cast<THDF5_MatrixDomainType> (0);
  }
  if (ParamValue == HDF5_MatrixDomainTypeNames[1])
  {
    return static_cast<THDF5_MatrixDomainType> (1);
  }


  char ErrorMessage[256];
  sprintf(ErrorMessage, HDF5_ERR_FMT_BadAttributeValue, DatasetName, HDF5_MatrixDomainTypeName, ParamValue.c_str());
  throw ios::failure(ErrorMessage);

  // This line will never be executed (just to prevent warning)
  return (THDF5_MatrixDomainType) 0;
}// end of ReadMatrixDomainType
//------------------------------------------------------------------------------


/**
 * Write integer attribute at a specified place in the file tree.
 *
 * @param [in] ParentGroup
 * @param [in] DatasetName
 * @param [in] AttributeName
 * @param [in] Value
 * @throw ios::failure
 */
inline void THDF5_File::WriteStringAttribute(const hid_t    ParentGroup,
                                             const char   * DatasetName,
                                             const char   * AttributeName,
                                             const string & Value)
{
  herr_t status = H5LTset_attribute_string(ParentGroup,
                                           DatasetName,
                                           AttributeName,
                                           Value.c_str());
  if (status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotWriteToAttribute, AttributeName, DatasetName);
    throw ios::failure(ErrorMessage);
  }
}// end of WriteIntAttribute
//------------------------------------------------------------------------------



/**
 * Read integer attribute  at a specified place in the file tree.
 *
 * @param  [in] ParentGroup
 * @param  [in] DatasetName
 * @param  [in] AttributeName
 * @return  Attribute value
 * @throw ios::failure
 */
inline string THDF5_File::ReadStringAttribute(const hid_t ParentGroup,
                                              const char * DatasetName,
                                              const char * AttributeName)
{
  char Value[256] = "";
  herr_t status = H5LTget_attribute_string(ParentGroup,
                                           DatasetName,
                                           AttributeName,
                                           Value);

  if (status < 0)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage, HDF5_ERR_FMT_CouldNotReadFromAttribute, AttributeName, DatasetName);

    throw ios::failure(ErrorMessage);
  }

  return Value;
}// end of ReadIntAttribute
//------------------------------------------------------------------------------


//----------------------------------------------------------------------------//
//----------------------------    THDF5_File    ------------------------------//
//----------------------------    Protected     ------------------------------//
//----------------------------------------------------------------------------//








//----------------------------------------------------------------------------//
//------------------------    THDF5_File_Header ------------------------------//
//-----------------------------    Public     --------------------------------//
//----------------------------------------------------------------------------//

/**
 * Constructor.
 */
THDF5_FileHeader::THDF5_FileHeader()
{
  HDF5_FileHeaderValues.clear();
  PopulateHeaderFileMap();
}// end of constructor
//------------------------------------------------------------------------------


/**
 * Copy constructor.
 * @param [in] other
 */
THDF5_FileHeader::THDF5_FileHeader(const THDF5_FileHeader & other)
{
  HDF5_FileHeaderValues = other.HDF5_FileHeaderValues;
  HDF5_FileHeaderNames = other.HDF5_FileHeaderNames;
}// end of copy constructor
//------------------------------------------------------------------------------


/**
 * Destructor.
 *
 */
THDF5_FileHeader::~THDF5_FileHeader()
{
  HDF5_FileHeaderValues.clear();
  HDF5_FileHeaderNames.clear();
}// end of destructor
//------------------------------------------------------------------------------



/**
 * Read header from the input file.
 * @param [in] InputFile - Input file to read from
 * @throw ios:failure when en error happens
 */
void THDF5_FileHeader::ReadHeaderFromInputFile(THDF5_File & InputFile)
{
  // Get file root handle
  hid_t FileRootHandle = InputFile.GetRootGroup();
  // read file type
  HDF5_FileHeaderValues[hdf5_fhi_file_type] =
          InputFile.ReadStringAttribute(FileRootHandle,"/",
                                        HDF5_FileHeaderNames[hdf5_fhi_file_type].c_str());

  if (GetFileType() == hdf5_ft_input)
  {
    HDF5_FileHeaderValues[hdf5_fhi_created_by]
            = InputFile.ReadStringAttribute(FileRootHandle, "/",
                                            HDF5_FileHeaderNames[hdf5_fhi_created_by].c_str());
    HDF5_FileHeaderValues[hdf5_fhi_creation_date]
            = InputFile.ReadStringAttribute(FileRootHandle, "/",
                                            HDF5_FileHeaderNames[hdf5_fhi_creation_date].c_str());
    HDF5_FileHeaderValues[hdf5_fhi_file_description]
            = InputFile.ReadStringAttribute(FileRootHandle, "/",
                                            HDF5_FileHeaderNames[hdf5_fhi_file_description].c_str());
    HDF5_FileHeaderValues[hdf5_fhi_major_version]
            = InputFile.ReadStringAttribute(FileRootHandle, "/",
                                            HDF5_FileHeaderNames[hdf5_fhi_major_version].c_str());
    HDF5_FileHeaderValues[hdf5_fhi_minor_version]
            = InputFile.ReadStringAttribute(FileRootHandle, "/",
                                            HDF5_FileHeaderNames[hdf5_fhi_minor_version].c_str());
  }
  else
  {
    throw ios::failure(HDF5_ERR_FMT_BadInputFileType);
  }
}// end of ReadHeaderFromInputFile
//------------------------------------------------------------------------------


/**
 * Read header from output file (necessary for checkpoint-restart).
 * Read only execution times (the others are read from the input file, or
 * calculated based on the very last leg of the simulation).
 * This function is called only if checkpoint-restart is enabled.
 * @param [in] OutputFile
 */
void THDF5_FileHeader::ReadHeaderFromOutputFile(THDF5_File & OutputFile)
{
  // Get file root handle
  hid_t FileRootHandle = OutputFile.GetRootGroup();

  HDF5_FileHeaderValues[hdf5_fhi_file_type]
          = OutputFile.ReadStringAttribute(FileRootHandle, "/",
                                           HDF5_FileHeaderNames[hdf5_fhi_file_type].c_str());

  if (GetFileType() == hdf5_ft_output)
  {
    HDF5_FileHeaderValues[hdf5_fhi_total_execution_time]
            = OutputFile.ReadStringAttribute(FileRootHandle, "/",
                                             HDF5_FileHeaderNames[hdf5_fhi_total_execution_time].c_str());
    HDF5_FileHeaderValues[hdf5_fhi_data_load_time]
            = OutputFile.ReadStringAttribute(FileRootHandle, "/",
                                             HDF5_FileHeaderNames[hdf5_fhi_data_load_time].c_str());
    HDF5_FileHeaderValues[hdf5_fhi_preprocessing_time]
            = OutputFile.ReadStringAttribute(FileRootHandle, "/",
                                             HDF5_FileHeaderNames[hdf5_fhi_preprocessing_time].c_str());
    HDF5_FileHeaderValues[hdf5_fhi_simulation_time]
            = OutputFile.ReadStringAttribute(FileRootHandle, "/",
                                             HDF5_FileHeaderNames[hdf5_fhi_simulation_time].c_str());
    HDF5_FileHeaderValues[hdf5_fhi_postprocessing_time]
            = OutputFile.ReadStringAttribute(FileRootHandle, "/",
                                             HDF5_FileHeaderNames[hdf5_fhi_postprocessing_time].c_str());
  }
  else
  {
    throw ios::failure(HDF5_ERR_FMT_BadOutputFileType);
  }
}// end of ReadHeaderFromOutputFile
//------------------------------------------------------------------------------


/**
 * Read the file header form the checkpoint file. We need the header to verify
 * the file version and type
 * @param [in] CheckpointFile
 */
void THDF5_FileHeader::ReadHeaderFromCheckpointFile(THDF5_File & CheckpointFile)
{
  // Get file root handle
  hid_t FileRootHandle = CheckpointFile.GetRootGroup();
  // read file type
  HDF5_FileHeaderValues[hdf5_fhi_file_type] =
          CheckpointFile.ReadStringAttribute(FileRootHandle,"/",
                                             HDF5_FileHeaderNames[hdf5_fhi_file_type].c_str());

  if (GetFileType() == hdf5_ft_checkpoint)
  {
    HDF5_FileHeaderValues[hdf5_fhi_created_by]
            = CheckpointFile.ReadStringAttribute(FileRootHandle, "/",
                                                 HDF5_FileHeaderNames[hdf5_fhi_created_by].c_str());
    HDF5_FileHeaderValues[hdf5_fhi_creation_date]
            = CheckpointFile.ReadStringAttribute(FileRootHandle, "/",
                                                 HDF5_FileHeaderNames[hdf5_fhi_creation_date].c_str());
    HDF5_FileHeaderValues[hdf5_fhi_file_description]
            = CheckpointFile.ReadStringAttribute(FileRootHandle, "/",
                                                 HDF5_FileHeaderNames[hdf5_fhi_file_description].c_str());
    HDF5_FileHeaderValues[hdf5_fhi_major_version]
            = CheckpointFile.ReadStringAttribute(FileRootHandle, "/",
                                                 HDF5_FileHeaderNames[hdf5_fhi_major_version].c_str());
    HDF5_FileHeaderValues[hdf5_fhi_minor_version]
            = CheckpointFile.ReadStringAttribute(FileRootHandle, "/",
                                                 HDF5_FileHeaderNames[hdf5_fhi_minor_version].c_str());
  }
  else
  {
    throw ios::failure(HDF5_ERR_FMT_BadCheckpointFileType);
  }
}// end of ReadHeaderFromCheckpointFile
//------------------------------------------------------------------------------

/**
 * Write header into the output file.
 * @param [in] OutputFile
 */
void THDF5_FileHeader::WriteHeaderToOutputFile(THDF5_File & OutputFile)
{
  // Get file root handle
  hid_t FileRootHandle = OutputFile.GetRootGroup();

  for (map<THDF5_FileHeaderItems, string>::iterator it = HDF5_FileHeaderNames.begin();
       it != HDF5_FileHeaderNames.end();
       it++)
  {
    OutputFile.WriteStringAttribute(FileRootHandle,
                                    "/",
                                    it->second.c_str(),
                                    HDF5_FileHeaderValues[it->first]);
  }
}// end of WriteHeaderToOutputFile
//------------------------------------------------------------------------------

/**
 * Write header to the output file (only a subset of all possible fields are written).
 * @param [in] CheckpointFile
 */
void THDF5_FileHeader::WriteHeaderToCheckpointFile(THDF5_File & CheckpointFile)
{
  // Get file root handle
  hid_t FileRootHandle = CheckpointFile.GetRootGroup();

  // Write header
  CheckpointFile.WriteStringAttribute(FileRootHandle,
                                      "/",
                                      HDF5_FileHeaderNames [hdf5_fhi_file_type].c_str(),
                                      HDF5_FileHeaderValues[hdf5_fhi_file_type].c_str());

  CheckpointFile.WriteStringAttribute(FileRootHandle,
                                      "/",
                                      HDF5_FileHeaderNames [hdf5_fhi_created_by].c_str(),
                                      HDF5_FileHeaderValues[hdf5_fhi_created_by].c_str());

  CheckpointFile.WriteStringAttribute(FileRootHandle,
                                      "/",
                                      HDF5_FileHeaderNames [hdf5_fhi_creation_date].c_str(),
                                      HDF5_FileHeaderValues[hdf5_fhi_creation_date].c_str());

  CheckpointFile.WriteStringAttribute(FileRootHandle,
                                      "/",
                                      HDF5_FileHeaderNames [hdf5_fhi_file_description].c_str(),
                                      HDF5_FileHeaderValues[hdf5_fhi_file_description].c_str());

  CheckpointFile.WriteStringAttribute(FileRootHandle,                                      "/",
                                      HDF5_FileHeaderNames [hdf5_fhi_major_version].c_str(),
                                      HDF5_FileHeaderValues[hdf5_fhi_major_version].c_str());

  CheckpointFile.WriteStringAttribute(FileRootHandle,
                                      "/",
                                      HDF5_FileHeaderNames [hdf5_fhi_minor_version].c_str(),
                                      HDF5_FileHeaderValues[hdf5_fhi_minor_version].c_str());
}// end of WriteHeaderToCheckpointFile
//------------------------------------------------------------------------------


/**
 * Get File type.
 * @return FileType
 */
THDF5_FileHeader::THDF5_FileType  THDF5_FileHeader::GetFileType()
{
  for (int i = hdf5_ft_input; i < hdf5_ft_unknown ; i++)
  {
    if (HDF5_FileHeaderValues[hdf5_fhi_file_type] == HDF5_FileTypesNames[static_cast<THDF5_FileType >(i)])
    {
      return static_cast<THDF5_FileType >(i);
    }
  }

  return THDF5_FileHeader::hdf5_ft_unknown;
}// end of GetFileType
//------------------------------------------------------------------------------

/**
 * Set File type.
 * @param FileType
 */
void THDF5_FileHeader::SetFileType(const THDF5_FileHeader::THDF5_FileType FileType)
{
  HDF5_FileHeaderValues[hdf5_fhi_file_type] = HDF5_FileTypesNames[FileType];
}// end of SetFileType
//------------------------------------------------------------------------------


/**
 * Get File Version an enum
 * @return file version as an enum
 */
THDF5_FileHeader::THDF5_FileVersion THDF5_FileHeader::GetFileVersion()
{
  if ((HDF5_FileHeaderValues[hdf5_fhi_major_version] == HDF5_MajorFileVersionsNames[0]) &&
      (HDF5_FileHeaderValues[hdf5_fhi_minor_version] == HDF5_MinorFileVersionsNames[0]))
  {
    return hdf5_fv_10;
  }

  if ((HDF5_FileHeaderValues[hdf5_fhi_major_version] == HDF5_MajorFileVersionsNames[0]) &&
      (HDF5_FileHeaderValues[hdf5_fhi_minor_version] == HDF5_MinorFileVersionsNames[1]))
  {
    return hdf5_fv_11;
  }

  return hdf5_fv_unknown;
}// end of GetFileVersion
//------------------------------------------------------------------------------

/**
 * Set actual date and time.
 *
 */
void THDF5_FileHeader::SetActualCreationTime()
{
    struct tm *current;
  time_t now;
  time(&now);
  current = localtime(&now);

  char DateString[20];

  sprintf(DateString, "%02i/%02i/%02i, %02i:%02i:%02i",
          current->tm_mday, current->tm_mon + 1, current->tm_year - 100,
          current->tm_hour, current->tm_min, current->tm_sec);

  HDF5_FileHeaderValues[hdf5_fhi_creation_date] = DateString;
}// end of SetCreationTime
//------------------------------------------------------------------------------

/**
 * Set Host name.
 *
 */
void THDF5_FileHeader::SetHostName()
{
  char   HostName[256];

  //Linux build
  #ifdef __linux__
    gethostname(HostName, 256);
  #endif

  //Windows build
  #ifdef _WIN64
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
	  gethostname(HostName, 256);

    WSACleanup();
  #endif

  HDF5_FileHeaderValues[hdf5_fhi_host_name] = HostName;
}// end of SetHostName
//------------------------------------------------------------------------------


/**
 * Set memory consumption.
 * @param [in] TotalMemory
 */
void THDF5_FileHeader::SetMemoryConsumption(const size_t TotalMemory)
{
  char Text[20] = "";
  sprintf(Text, "%ld MB",TotalMemory);

  HDF5_FileHeaderValues[hdf5_fhi_total_memory_consumption]     = Text;

  sprintf(Text, "%ld MB",TotalMemory / TParameters::GetInstance()->GetNumberOfThreads());
  HDF5_FileHeaderValues[hdf5_fhi_peak_core_memory_consumption] = Text;
}// end of SetMemoryConsumption
//------------------------------------------------------------------------------


/**
 * Set execution times in file header.
 * @param [in] TotalTime
 * @param [in] LoadTime
 * @param [in] PreProcessingTime
 * @param [in] SimulationTime
 * @param [in] PostprocessingTime
 */
void THDF5_FileHeader::SetExecutionTimes(const double TotalTime,
                                         const double LoadTime,
                                         const double PreProcessingTime,
                                         const double SimulationTime,
                                         const double PostprocessingTime)
{
    char Text [30] = "";

  sprintf(Text, "%8.2fs", TotalTime);
  HDF5_FileHeaderValues[hdf5_fhi_total_execution_time] = Text;

  sprintf(Text, "%8.2fs", LoadTime);
  HDF5_FileHeaderValues[hdf5_fhi_data_load_time] = Text;

  sprintf(Text, "%8.2fs", PreProcessingTime);
  HDF5_FileHeaderValues[hdf5_fhi_preprocessing_time] = Text;


  sprintf(Text, "%8.2fs", SimulationTime);
  HDF5_FileHeaderValues[hdf5_fhi_simulation_time] = Text;

  sprintf(Text, "%8.2fs", PostprocessingTime);
  HDF5_FileHeaderValues[hdf5_fhi_postprocessing_time] = Text;
}// end of SetExecutionTimes
//------------------------------------------------------------------------------

/**
 * Get execution times stored in the output file header
 * @param TotalTime
 * @param LoadTime
 * @param PreProcessingTime
 * @param SimulationTime
 * @param PostprocessingTime
 */
void THDF5_FileHeader::GetExecutionTimes(double& TotalTime,
                                         double& LoadTime,
                                         double& PreProcessingTime,
                                         double& SimulationTime,
                                         double& PostprocessingTime)
{
  TotalTime          = atof(HDF5_FileHeaderValues[hdf5_fhi_total_execution_time].c_str());
  LoadTime           = atof(HDF5_FileHeaderValues[hdf5_fhi_data_load_time].c_str());
  PreProcessingTime  = atof(HDF5_FileHeaderValues[hdf5_fhi_preprocessing_time].c_str());
  SimulationTime     = atof(HDF5_FileHeaderValues[hdf5_fhi_simulation_time].c_str());
  PostprocessingTime = atof(HDF5_FileHeaderValues[hdf5_fhi_postprocessing_time].c_str());
}// end of GetExecutionTimes
//------------------------------------------------------------------------------

/**
 * Set Number of cores.
 *
 */
void THDF5_FileHeader::SetNumberOfCores()
{
  char Text[12] = "";
  sprintf(Text, "%ld", TParameters::GetInstance()->GetNumberOfThreads());

  HDF5_FileHeaderValues[hdf5_fhi_number_of_cores] = Text;
}// end of SetNumberOfCores
//------------------------------------------------------------------------------

//----------------------------------------------------------------------------//
//------------------------    THDF5_File_Header ------------------------------//
//---------------------------    Protected     --------------------------------//
//----------------------------------------------------------------------------//


/**
 * Create map with names for the header.
 *
 */
void THDF5_FileHeader::PopulateHeaderFileMap()
{
  HDF5_FileHeaderNames.clear();

  HDF5_FileHeaderNames[hdf5_fhi_created_by] = "created_by";
  HDF5_FileHeaderNames[hdf5_fhi_creation_date] = "creation_date";
  HDF5_FileHeaderNames[hdf5_fhi_file_description] = "file_description";
  HDF5_FileHeaderNames[hdf5_fhi_major_version] = "major_version";
  HDF5_FileHeaderNames[hdf5_fhi_minor_version] = "minor_version";
  HDF5_FileHeaderNames[hdf5_fhi_file_type] = "file_type";

  HDF5_FileHeaderNames[hdf5_fhi_host_name] = "host_names";
  HDF5_FileHeaderNames[hdf5_fhi_number_of_cores] = "number_of_cpu_cores";
  HDF5_FileHeaderNames[hdf5_fhi_total_memory_consumption] = "total_memory_in_use";
  HDF5_FileHeaderNames[hdf5_fhi_peak_core_memory_consumption] = "peak_core_memory_in_use";

  HDF5_FileHeaderNames[hdf5_fhi_total_execution_time] = "total_execution_time";
  HDF5_FileHeaderNames[hdf5_fhi_data_load_time] = "data_loading_phase_execution_time";
  HDF5_FileHeaderNames[hdf5_fhi_preprocessing_time] = "pre-processing_phase_execution_time";
  HDF5_FileHeaderNames[hdf5_fhi_simulation_time] = "simulation_phase_execution_time";
  HDF5_FileHeaderNames[hdf5_fhi_postprocessing_time] = "post-processing_phase_execution_time";
}// end of PopulateHeaderFileMap
//------------------------------------------------------------------------------
