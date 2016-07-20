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
 *
 * @date        27 July     2012, 14:14 (created) \n
 *              20 July     2016, 16:57 (revised)
 *
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


#include <iostream>
#include <stdexcept>
#include <ctime>

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
#include <Logger/ErrorMessages.h>

//------------------------------------------------------------------------------------------------//
//------------------------------------------ CONSTANTS -------------------------------------------//
//------------------------------------------------------------------------------------------------//


const char* THDF5_File::matrixDomainTypeName    = "domain_type";
const char* THDF5_File::matrixDataTypeName      = "data_type";

const string THDF5_File::matrixDomainTypeNames[] = {"real","complex"};
const string THDF5_File::matrixDataTypeNames[]   = {"float","long"};

const string THDF5_FileHeader::hdf5_FileTypesNames[]  = {"input", "output", "checkpoint", "unknown"};

const string THDF5_FileHeader::hdf5_MajorFileVersionsNames[] = {"1"};
const string THDF5_FileHeader::hdf5_MinorFileVersionsNames[] = {"0","1"};

//------------------------------------------------------------------------------------------------//
//--------------------------------------    THDF5_File    ----------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Constructor.
 */
THDF5_File::THDF5_File() :
    file(H5I_BADID), fileName("")
{

}// end of constructor
//--------------------------------------------------------------------------------------------------


/**
 * Create the HDF5 file.
 *
 * @param [in] fileName - File name
 * @param [in] flags    - Flags for the HDF5 runtime
 * @throw ios:failure if error happened.
 */
void THDF5_File::Create(const char*  fileName,
                        unsigned int flags)
{
  // file is opened
  if (IsOpen())
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_FileCannotRecreated, fileName);
    throw ios::failure(errMsg);
  }

  // Create a new file using default properties.
  this->fileName = fileName;

  file = H5Fcreate(fileName, flags, H5P_DEFAULT, H5P_DEFAULT);

  if (file < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_FileNotCreated, fileName);
    throw ios::failure(errMsg);
  }
}// end of Create
//--------------------------------------------------------------------------------------------------


/**
 * Open the HDF5 file.
 *
 * @param [in] fileName - File name
 * @param [in] flags    - Flags for the HDF5 runtime
 * @throw ios:failure if error happened.
 *
 */
void THDF5_File::Open(const char * fileName,
                      unsigned int flags)
{
  if (IsOpen())
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_FileCannotReopen, fileName);
    throw ios::failure(errMsg);
  };

  this->fileName = fileName;

  if (H5Fis_hdf5(fileName) == 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_NotHDF5File, fileName);
    throw ios::failure(errMsg);
  }

  file = H5Fopen(fileName, flags, H5P_DEFAULT);

  if (file < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_FileNotOpened, fileName);
    throw ios::failure(errMsg);
  }
}// end of Open
//--------------------------------------------------------------------------------------------------

/**
 * Close the HDF5 file.
 *
 * @throw ios::failure
 */
void THDF5_File::Close()
{
  // Terminate access to the file.
  herr_t status = H5Fclose(file);

  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_FileNotClosed, fileName.c_str());
    throw ios::failure(errMsg);
  }

  fileName = "";
  file = H5I_BADID;
}// end of Close
//--------------------------------------------------------------------------------------------------

/**
 * Destructor.
 */
THDF5_File::~THDF5_File()
{
  if (IsOpen()) Close();
}//end of ~THDF5_File
//--------------------------------------------------------------------------------------------------


/**
 * Create a HDF5 group at a specified place in the file tree.
 *
 * @param [in] parentGroup  - Where to link the group at
 * @param [in] groupName    - Group name
 * @return A handle to the new group
 * @throw ios::failure
 */
hid_t THDF5_File::CreateGroup(const hid_t parentGroup,
                              const char* groupName)
{
  hid_t group = H5Gcreate(parentGroup, groupName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  if (group == H5I_INVALID_HID)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_GroupNotCreated, groupName, fileName.c_str());
    throw ios::failure(errMsg);
  }

  return group;
}// end of CreateGroup
//--------------------------------------------------------------------------------------------------

/**
 * Open a HDF5 group at a specified place in the file tree.
 *
 * @param [in] parentGroup - Parent group
 * @param [in] groupName   - Group name
 * @return A handle to the group
 * @throw ios::failure
 */
hid_t THDF5_File::OpenGroup(const hid_t parentGroup,
                            const char* groupName)
{
  hid_t group = H5Gopen(parentGroup, groupName, H5P_DEFAULT);

  if (group == H5I_INVALID_HID)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_GroupNotOpened, groupName, fileName.c_str());
    throw ios::failure(errMsg);
  }

  return group;
}// end of OpenGroup
//--------------------------------------------------------------------------------------------------


/**
 * Close a group.
 *
 * @param[in] group - Group to close
 */
void THDF5_File::CloseGroup(const hid_t group)
{
  H5Gclose(group);
}// end of CloseGroup
//--------------------------------------------------------------------------------------------------

/**
 * Open a dataset at a specified place in the file tree.
 *
 * @param [in] parentGroup - Parent group Id (can be the file Id for root).
 * @param [in] datasetName - Dataset name
 * @return A handle to open dataset
 * @throw ios::failure
 */
hid_t THDF5_File::OpenDataset(const hid_t parentGroup,
                              const char* datasetName)
{
  // Open dataset
  hid_t dataset = H5Dopen(parentGroup, datasetName, H5P_DEFAULT);

  if (dataset == H5I_INVALID_HID)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_DatasetNotOpened, fileName.c_str(), datasetName);
    throw ios::failure(errMsg);
  }

  return dataset;
}// end of OpenDataset
//--------------------------------------------------------------------------------------------------

/**
 * Create a float HDF5 dataset at a specified place in the file tree.
 *
 * @param [in] parentGroup       - Parent group Id
 * @param [in] datasetName       - Dataset name
 * @param [in] dimensionSizes    - Dimension sizes
 * @param [in] chunkSizes        - Chunk sizes
 * @param [in] compressionLevel  - Compression level
 * @return A handle to the new dataset
 * @throw ios::failure
 */
hid_t THDF5_File::CreateFloatDataset(const hid_t            parentGroup,
                                     const char*            datasetName,
                                     const TDimensionSizes& dimensionSizes,
                                     const TDimensionSizes& chunkSizes,
                                     const size_t           compressionLevel)
{
  const int rank = (dimensionSizes.Is3D()) ? 3 : 4;

// a windows hack
  hsize_t dims [4];
  hsize_t chunk[4];

  // 3D dataset
  if (dimensionSizes.Is3D())
  {
    dims[0] = dimensionSizes.Z;
    dims[1] = dimensionSizes.Y;
    dims[2] = dimensionSizes.X;

    chunk[0] = chunkSizes.Z;
    chunk[1] = chunkSizes.Y;
    chunk[2] = chunkSizes.X;
  }
  else // 4D dataset
  {
    dims[0] = dimensionSizes.T;
    dims[1] = dimensionSizes.Z;
    dims[2] = dimensionSizes.Y;
    dims[3] = dimensionSizes.X;

    chunk[0] = chunkSizes.T;
    chunk[1] = chunkSizes.Z;
    chunk[2] = chunkSizes.Y;
    chunk[3] = chunkSizes.X;
  }

  hid_t  propertyList;
  herr_t status;

  hid_t dataspace = H5Screate_simple(rank, dims, NULL);

  // set chunk size
  propertyList = H5Pcreate(H5P_DATASET_CREATE);

  status = H5Pset_chunk(propertyList, rank, chunk);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_DatasetNotOpened, fileName.c_str(), datasetName);
    throw ios::failure(errMsg);
  }

  // set compression level
  status = H5Pset_deflate(propertyList, compressionLevel);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotSetCompression,
             fileName.c_str(), datasetName, compressionLevel);
    throw ios::failure(errMsg);
  }

  // create dataset
  hid_t dataset = H5Dcreate(parentGroup,
                            datasetName,
                            H5T_NATIVE_FLOAT,
                            dataspace,
                            H5P_DEFAULT,
                            propertyList,
                            H5P_DEFAULT);

  if (dataset == H5I_INVALID_HID)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_DatasetNotOpened, fileName.c_str(), datasetName);
    throw ios::failure(errMsg);
  }

  H5Pclose(propertyList);

  return dataset;
}// end of CreateFloatDataset
//--------------------------------------------------------------------------------------------------


/**
 * Create an index HDF5 dataset at a specified place in the file tree (always 3D).
 *
 * @param [in] parentGroup       - Parent group
 * @param [in] datasetName       - Dataset name
 * @param [in] dimensionSizes    - Dimension sizes
 * @param [in] chunkSizes        - Chunk sizes
 * @param [in] compressionLevel  - Compression level
 * @return A handle to the new dataset
 * @throw ios::failure
 */
hid_t THDF5_File::CreateIndexDataset(const hid_t            parentGroup,
                                     const char*            datasetName,
                                     const TDimensionSizes& dimensionSizes,
                                     const TDimensionSizes& chunkSizes,
                                     const size_t           compressionLevel)
{
  const int rank = 3;

  hsize_t dims [rank] = {dimensionSizes.Z, dimensionSizes.Y, dimensionSizes.X};
  hsize_t chunk[rank] = {chunkSizes.Z, chunkSizes.Y, chunkSizes.X};

  hid_t propertyList;
  herr_t status;

  hid_t dataspace = H5Screate_simple(rank, dims, NULL);

  // set chunk size
  propertyList = H5Pcreate(H5P_DATASET_CREATE);

  status = H5Pset_chunk(propertyList, rank, chunk);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_DatasetNotOpened, fileName.c_str(), datasetName);
    throw ios::failure(errMsg);
  }

  // set compression level
  status = H5Pset_deflate(propertyList, compressionLevel);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotSetCompression,
             fileName.c_str(), datasetName, compressionLevel);
    throw ios::failure(errMsg);
  }

  // create dataset
  hid_t dataset = H5Dcreate(parentGroup,
                            datasetName,
                            H5T_STD_U64LE,
                            dataspace,
                            H5P_DEFAULT,
                            propertyList,
                            H5P_DEFAULT);

  if (dataset == H5I_INVALID_HID)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_DatasetNotOpened, fileName.c_str(), datasetName);
    throw ios::failure(errMsg);
  }

  H5Pclose(propertyList);

  return dataset;
}// end of CreateIndexDataset
//--------------------------------------------------------------------------------------------------


/**
 * Close dataset.
 *
 * @param [in] dataset - Dataset to close
 */
void  THDF5_File::CloseDataset(const hid_t dataset)
{
  H5Dclose (dataset);
}// end of CloseDataset
//--------------------------------------------------------------------------------------------------


/**
 * Write a hyperslab into the dataset, float version.
 *
 * @param [in] dataset   - Dataset id
 * @param [in] position  - Position in the dataset
 * @param [in] size      - Size of the hyperslab
 * @param [in] data      - Data to be written
 * @throw ios::failure
 */
void THDF5_File::WriteHyperSlab(const hid_t            dataset,
                                const TDimensionSizes& position,
                                const TDimensionSizes& size,
                                const float*           data)
{
  herr_t status;
  hid_t filespace, memspace;

  // Get file space to find out number of dimensions
  filespace = H5Dget_space(dataset);
  const int rank = H5Sget_simple_extent_ndims(filespace);

  // Select sizes and positions, windows hack
  hsize_t nElement[4];
  hsize_t offset[4];

  // 3D dataset
  if (rank == 3)
  {
    nElement[0] = size.Z;
    nElement[1] = size.Y;
    nElement[2] = size.X;

    offset[0] = position.Z;
    offset[1] = position.Y;
    offset[2] = position.X;
  }
  else // 4D dataset
  {
    nElement[0] = size.T;
    nElement[1] = size.Z;
    nElement[2] = size.Y;
    nElement[3] = size.X;

    offset[0] = position.T;
    offset[1] = position.Z;
    offset[2] = position.Y;
    offset[3] = position.X;
  }

  // select hyperslab
  status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, nElement, NULL);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(errMsg);
  }

  // assign memspace
  memspace = H5Screate_simple(rank, nElement, NULL);

  status = H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, data);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(errMsg);
  }

  H5Sclose(memspace);
  H5Sclose(filespace);
}// end of WriteHyperSlab
//--------------------------------------------------------------------------------------------------


/**
 * Write a hyperslab into the dataset, index version.
 *
 * @param [in] dataset  - Dataset id
 * @param [in] position - Position in the dataset
 * @param [in] size     - Size of the hyperslab
 * @param [in] data     - Data to be written
 * @throw ios::failure
 */
void THDF5_File::WriteHyperSlab(const hid_t            dataset,
                                const TDimensionSizes& position,
                                const TDimensionSizes& size,
                                const size_t*          data)
{
  herr_t status;
  hid_t filespace, memspace;

  // Get File Space, to find out number of dimensions
  filespace = H5Dget_space(dataset);
  const int rank = H5Sget_simple_extent_ndims(filespace);

  // Set sizes and offsets, windows hack
  hsize_t nElement[4];
  hsize_t offset[4];

  // 3D dataset
  if (rank == 3)
  {
    nElement[0] = size.Z;
    nElement[1] = size.Y;
    nElement[2] = size.X;

    offset[0] = position.Z;
    offset[1] = position.Y;
    offset[2] = position.X;
  }
  else // 4D dataset
  {
    nElement[0] = size.T;
    nElement[1] = size.Z;
    nElement[2] = size.Y;
    nElement[3] = size.X;

    offset[0] = position.T;
    offset[1] = position.Z;
    offset[2] = position.Y;
    offset[3] = position.X;
  }

  // select hyperslab
  status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, nElement, NULL);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(errMsg);
  }

  // assign memspace
  memspace = H5Screate_simple(rank, nElement, NULL);

  status = H5Dwrite(dataset, H5T_STD_U64LE, memspace, filespace, H5P_DEFAULT, data);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(errMsg);
  }

  H5Sclose(memspace);
  H5Sclose(filespace);
}// end of WriteHyperSlab
//--------------------------------------------------------------------------------------------------


/**
 * Write a cuboid selected within the matrixData into a hyperslab.
 * The routine writes 3D cuboid into a 4D dataset (only intended for raw time series).
 *
 * @param [in] dataset           - Dataset to write MatrixData into
 * @param [in] hyperslabPosition - Position in the dataset (hyperslab) - may be 3D/4D
 * @param [in] cuboidPosition    - Position of the cuboid in MatrixData (what to sample) - must be 3D
 * @param [in] cuboidSize        - Cuboid size (size of data being sampled) - must by 3D
 * @param [in] matrixDimensions  - Size of the original matrix (the sampled one)
 * @param [in] matrixData        - C array of MatrixData
 * @throw ios::failure
 */
void THDF5_File::WriteCuboidToHyperSlab(const hid_t            dataset,
                                        const TDimensionSizes& hyperslabPosition,
                                        const TDimensionSizes& cuboidPosition,
                                        const TDimensionSizes& cuboidSize,
                                        const TDimensionSizes& matrixDimensions,
                                        const float*           matrixData)
{
  herr_t status;
  hid_t filespace, memspace;

  const int rank = 4;

  // Select sizes and positions
  // The T here is always 1 (only one timestep)
  hsize_t slabSize[rank]        = {1, cuboidSize.Z, cuboidSize.Y, cuboidSize.X};
  hsize_t offsetInDataset[rank] = {hyperslabPosition.T, hyperslabPosition.Z, hyperslabPosition.Y, hyperslabPosition.X};
  hsize_t offsetInMatrixData[]  = {cuboidPosition.Z, cuboidPosition.Y, cuboidPosition.X};
  hsize_t matrixSize []         = {matrixDimensions.Z, matrixDimensions.Y, matrixDimensions.X};


  // select hyperslab in the HDF5 dataset
  filespace = H5Dget_space(dataset);
  status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsetInDataset, NULL, slabSize, NULL);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(errMsg);
  }

  // assign memspace and select the cuboid in the sampled matrix
  memspace = H5Screate_simple(3, matrixSize, NULL);
  status = H5Sselect_hyperslab(memspace,
                               H5S_SELECT_SET,
                               offsetInMatrixData,
                               NULL,
                               slabSize + 1, // Slab size has to be 3D in this case (done by skipping the T dimension)
                               NULL);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(errMsg);
  }

  // Write the data
  status = H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, matrixData);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(errMsg);
  }

  // close memspace and filespace
  H5Sclose(memspace);
  H5Sclose(filespace);
}// end of WriteCuboidToHyperSlab
//--------------------------------------------------------------------------------------------------

/**
 * Write sensor data selected by the sensor mask.
 * A routine pick elements from the MatixData based on the sensor data and store.
 * them into a single hyperslab of size [Nsens, 1, 1].
 *
 * @param [in] dataset           - Dataset to write MaatrixData into
 * @param [in] hyperslabPosition - 3D position in the dataset (hyperslab)
 * @param [in] indexSensorSize   - Size of the index based sensor mask
 * @param [in] indexSensorData   - Index based sensor mask
 * @param [in] matrixDimensions  - Size of the sampled matrix
 * @param [in] matrixData        - Matrix data
 * @warning  - Very slow at this version of HDF5 for orthogonal planes-> DO NOT USE
 * @throw ios::failure
 */
void THDF5_File::WriteDataByMaskToHyperSlab(const hid_t            dataset,
                                            const TDimensionSizes& hyperslabPosition,
                                            const size_t           indexSensorSize,
                                            const size_t*          indexSensorData,
                                            const TDimensionSizes& matrixDimensions,
                                            const float*           matrixData)
{
  herr_t status;
  hid_t filespace, memspace;

  const int Rank = 3;

  // Select sizes and positions
  // Only one timestep
  hsize_t slabSize[Rank]        = {1, 1, indexSensorSize};
  hsize_t offsetInDataset[Rank] = {hyperslabPosition.Z, hyperslabPosition.Y, hyperslabPosition.X};

  // treat as a 1D array
  hsize_t matrixSize = matrixDimensions.Z * matrixDimensions.Y * matrixDimensions.X;


  // select hyperslab in the HDF5 dataset
  filespace = H5Dget_space(dataset);
  status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsetInDataset, NULL,  slabSize, NULL);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(errMsg);
  }

  // assign 1D memspace and select the elements within the array
  memspace = H5Screate_simple(1, &matrixSize, NULL);
  status = H5Sselect_elements(memspace,
                              H5S_SELECT_SET,
                              indexSensorSize,
                              (hsize_t*) (indexSensorData));
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(errMsg);
  }

  // Write the data
  status = H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, matrixData);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotWriteTo, "");
    throw ios::failure(errMsg);
  }

  // close memspace and filespace
  H5Sclose(memspace);
  H5Sclose(filespace);
}// end of WriteSensorbyMaskToHyperSlab
//--------------------------------------------------------------------------------------------------

/**
 * Write a scalar value at a specified place in the file tree
 * (no chunks, no compression). Float value.
 *
 * @param [in] parentGroup - Where to link the scalar dataset
 * @param [in] datasetName - HDF5 dataset name
 * @param [in] value       - data to be written
 * @throw ios::failure
 */
void THDF5_File::WriteScalarValue(const hid_t parentGroup,
                                  const char* datasetName,
                                  const float value)
{
  const int rank = 3;
  const hsize_t dims[] = {1, 1, 1};

  hid_t dataset = H5I_INVALID_HID;
  hid_t dataspace = H5I_INVALID_HID;
  herr_t status;


  if (H5LTfind_dataset(parentGroup, datasetName) == 1)
  { // dataset already exists (from previous simulation leg) open it
    dataset = OpenDataset(parentGroup,datasetName);
  }
  else
  { // dataset does not exist yet -> create it
    dataspace = H5Screate_simple(rank, dims, NULL);
    dataset = H5Dcreate(parentGroup,
                        datasetName,
                        H5T_NATIVE_FLOAT,
                        dataspace,
                        H5P_DEFAULT,
                        H5P_DEFAULT,
                        H5P_DEFAULT);
  }

  // was created correctly?
  if (dataset == H5I_INVALID_HID)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotWriteTo, datasetName);
    throw ios::failure(errMsg);
  }

  status = H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotWriteTo, datasetName);
    throw ios::failure(errMsg);
  }

  WriteMatrixDataType(parentGroup, datasetName, FLOAT);
  WriteMatrixDomainType(parentGroup, datasetName, REAL);
} // end of WriteScalarValue (float)
//--------------------------------------------------------------------------------------------------

/**
 * Write a scalar value at a specified place in the file tree
 * (no chunks, no compression). Index value.
 *
 * @param [in] parentGroup - Where to link the scalar dataset
 * @param [in] datasetName - HDF5 dataset name
 * @param [in] value       - Data to be written
 * @throw ios::failure
 */
void THDF5_File::WriteScalarValue(const hid_t  parentGroup,
                                  const char*  datasetName,
                                  const size_t value)
{
  const int rank = 3;
  const hsize_t dims[] = {1, 1, 1};

  hid_t  dataset = H5I_INVALID_HID;
  hid_t  dataspace = H5I_INVALID_HID;
  herr_t status;


  if (H5LTfind_dataset(parentGroup, datasetName) == 1)
  { // dataset already exists (from previous leg) open it
    dataset = OpenDataset(parentGroup, datasetName);
  }
  else
  { // dataset does not exist yet -> create it
    dataspace = H5Screate_simple(rank, dims, NULL);
    dataset = H5Dcreate(parentGroup,
                        datasetName,
                        H5T_STD_U64LE,
                        dataspace,
                        H5P_DEFAULT,
                        H5P_DEFAULT,
                        H5P_DEFAULT);
  }

    // was created correctly?
  if (dataset == H5I_INVALID_HID)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotWriteTo, datasetName);
    throw ios::failure(errMsg);
  }

  status = H5Dwrite(dataset, H5T_STD_U64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotWriteTo, datasetName);
    throw ios::failure(errMsg);
  }

  WriteMatrixDataType(parentGroup, datasetName, LONG);
  WriteMatrixDomainType(parentGroup, datasetName, REAL);
}// end of WriteScalarValue
//--------------------------------------------------------------------------------------------------

/**
 * Read the scalar value under a specified group, float value.
 *
 * @param [in]  parentGroup - Where to link the scalar dataset
 * @param [in]  datasetName - HDF5 dataset name
 * @param [out] value       - Data to be read
 */
void THDF5_File::ReadScalarValue(const hid_t parentGroup,
                                 const char* datasetName,
                                 float&      value)
{
  ReadCompleteDataset(parentGroup, datasetName, TDimensionSizes(1,1,1), &value);
} // end of ReadScalarValue
//--------------------------------------------------------------------------------------------------

/**
 * Read the scalar value under a specified group, index value.
 *
 * @param [in]  parentGroup - Where to link the scalar dataset
 * @param [in]  datasetName - HDF5 dataset name
 * @param [out] value       - Data to be read
 */
void THDF5_File::ReadScalarValue(const hid_t parentGroup,
                                 const char* datasetName,
                                 size_t&     value)
{
  ReadCompleteDataset(parentGroup, datasetName, TDimensionSizes(1,1,1), &value);
}// end of ReadScalarValue
//--------------------------------------------------------------------------------------------------

/**
 * Read data from the dataset at a specified place in the file tree, float version.
 *
 * @param [in] parentGroup     - Where is the dataset situated
 * @param [in] datasetName     - Dataset name
 * @param [in] dimensionSizes  - Dimension sizes
 * @param [out] data           - Pointer to data
 * @throw ios::failure
 */
void THDF5_File::ReadCompleteDataset (const hid_t            parentGroup,
                                      const char*            datasetName,
                                      const TDimensionSizes& dimensionSizes,
                                      float*                 data)
{
    // Check Dimensions sizes
  if (GetDatasetDimensionSizes(parentGroup, datasetName).GetElementCount() !=
      dimensionSizes.GetElementCount())
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_WrongDimensionSizes, datasetName);
    throw ios::failure(errMsg);
  }

  // read dataset
  herr_t status = H5LTread_dataset_float(parentGroup, datasetName, data);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotReadFrom, datasetName);
    throw ios::failure(errMsg);
  }
}// end of ReadDataset (float)
//-------------------------------------------------------------------------------------------------


/**
 * Read data from the dataset at a specified place in the file tree, index version.
 *
 * @param [in] parentGroup     - Where is the dataset situated
 * @param [in] datasetName     - Dataset name
 * @param [in] dimensionSizes  - Dimension sizes
 * @param [out] data           - Pointer to data
 * @throw ios::failure
 */
void THDF5_File::ReadCompleteDataset(const hid_t            parentGroup,
                                     const char*            datasetName,
                                     const TDimensionSizes& dimensionSizes,
                                     size_t*                data)
{
  if (GetDatasetDimensionSizes(parentGroup, datasetName).GetElementCount() !=
      dimensionSizes.GetElementCount())
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_WrongDimensionSizes, datasetName);
    throw ios::failure(errMsg);
  }

  // read dataset
  herr_t status = H5LTread_dataset(parentGroup, datasetName, H5T_STD_U64LE, data);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotReadFrom, datasetName);
    throw ios::failure(errMsg);
  }
}// end of ReadCompleteDataset
//--------------------------------------------------------------------------------------------------


/**
 * Get dimension sizes of the dataset at a specified place in the file tree.
 *
 * @param [in] parentGroup - Where the dataset is
 * @param [in] datasetName - Dataset name
 * @return Dimension sizes of the dataset
 * @throw ios::failure
 */
TDimensionSizes THDF5_File::GetDatasetDimensionSizes(const hid_t parentGroup,
                                                     const char* datasetName)
{
  const size_t ndims = GetDatasetNumberOfDimensions(parentGroup, datasetName);

  hsize_t dims[4] = {0, 0, 0, 0};

  herr_t status = H5LTget_dataset_info(parentGroup, datasetName, dims, NULL, NULL);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotReadFrom, datasetName);
    throw ios::failure(errMsg);
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
//--------------------------------------------------------------------------------------------------

/**
 * Get number of dimensions of the dataset  under a specified group.
 *
 * @param [in] parentGroup - Where the dataset is
 * @param [in] datasetName - Dataset name
 * @return Number of dimensions
 * @throw ios::failure
 */
size_t THDF5_File::GetDatasetNumberOfDimensions(const hid_t  parentGroup,
                                                const char* datasetName)
{
  int dims = 0;

  herr_t status = H5LTget_dataset_ndims(parentGroup, datasetName, &dims);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotReadFrom, datasetName);
    throw ios::failure(errMsg);
  }

  return dims;
}// end of GetDatasetNumberOfDimensions
//--------------------------------------------------------------------------------------------------


/**
 * Get dataset element count at a specified place in the file tree.
 *
 * @param [in] parentGroup - Where the dataset is
 * @param [in] datasetName - Dataset name
 * @return Number of elements
 * @throw ios::failure
 */
size_t THDF5_File::GetDatasetElementCount(const hid_t  parentGroup,
                                          const char* datasetName)
{
  hsize_t dims[3] = {0, 0, 0};

  herr_t status = H5LTget_dataset_info(parentGroup, datasetName, dims, NULL, NULL);
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotReadFrom, datasetName);
    throw ios::failure(errMsg);
  }

  return dims[0] * dims[1] * dims[2];
}// end of GetDatasetElementCount
//--------------------------------------------------------------------------------------------------


/**
 * Write matrix data type into the dataset at a specified place in the file tree.
 *
 * @param [in] parentGroup    - Where the dataset is
 * @param [in] datasetName    - Dataset name
 * @param [in] matrixDataType - Matrix data type in the file
 */
void THDF5_File::WriteMatrixDataType(const hid_t                parentGroup,
                                     const char*                datasetName,
                                     const TMatrixDataType& matrixDataType)
{
  WriteStringAttribute(parentGroup,
                       datasetName,
                       matrixDataTypeName,
                       matrixDataTypeNames[matrixDataType]);
}// end of WriteMatrixDataType
//--------------------------------------------------------------------------------------------------


/**
 * Write matrix data type into the dataset at a specified place in the file tree.
 *
 * @param [in] parentGroup      - Where the dataset is
 * @param [in] datasetName      - Dataset name
 * @param [in] matrixDomainType - Matrix domain type
 */
void THDF5_File::WriteMatrixDomainType(const hid_t                  parentGroup,
                                       const char*                  datasetName,
                                       const TMatrixDomainType& matrixDomainType)
{
  WriteStringAttribute(parentGroup,
                       datasetName,
                       matrixDomainTypeName,
                       matrixDomainTypeNames[matrixDomainType]);
}// end of WriteMatrixDomainType
//--------------------------------------------------------------------------------------------------


/**
 * Read matrix data type from the dataset at a specified place in the file tree.
 *
 * @param [in] parentGroup - Where the dataset is
 * @param [in] datasetName - Dataset name
 * @return Matrix data type
 * @throw ios::failure
 */
THDF5_File::TMatrixDataType THDF5_File::ReadMatrixDataType(const hid_t parentGroup,
                                                           const char* datasetName)
{
  string paramValue = ReadStringAttribute(parentGroup, datasetName, matrixDataTypeName);

  if (paramValue == matrixDataTypeNames[0])
  {
    return static_cast<TMatrixDataType>(0);
  }
  if (paramValue == matrixDataTypeNames[1])
  {
    return static_cast<TMatrixDataType>(1);
  }

  char errMsg[256];
  snprintf(errMsg, 256, HDF5_ERR_FMT_BadAttributeValue, datasetName,
           matrixDataTypeName, paramValue.c_str());
  throw ios::failure(errMsg);

  // this will never be executed (just to prevent warning)
  return  static_cast<TMatrixDataType> (0);
}// end of ReadMatrixDataType
//--------------------------------------------------------------------------------------------------


/**
 * Read matrix dataset domain type at a specified place in the file tree.
 *
 * @param [in] parentGroup - Where the dataset is
 * @param [in] datasetName - Dataset name
 * @return Matrix domain type
 * @throw ios::failure
 */
THDF5_File::TMatrixDomainType THDF5_File::ReadMatrixDomainType(const hid_t parentGroup,
                                                               const char* datasetName)
{
  string paramValue = ReadStringAttribute(parentGroup, datasetName, matrixDomainTypeName);

  if (paramValue == matrixDomainTypeNames[0])
  {
    return static_cast<TMatrixDomainType> (0);
  }
  if (paramValue == matrixDomainTypeNames[1])
  {
    return static_cast<TMatrixDomainType> (1);
  }

  char errMsg[256];
  snprintf(errMsg, 256, HDF5_ERR_FMT_BadAttributeValue, datasetName,
           matrixDomainTypeName, paramValue.c_str());
  throw ios::failure(errMsg);

  // This line will never be executed (just to prevent warning)
  return static_cast<TMatrixDomainType> (0);
}// end of ReadMatrixDomainType
//--------------------------------------------------------------------------------------------------


/**
 * Write integer attribute at a specified place in the file tree.
 *
 * @param [in] parentGroup   - Where the dataset is
 * @param [in] datasetName   - Dataset name
 * @param [in] attributeName - Attribute name
 * @param [in] value         - Data to write
 * @throw ios::failure
 */
inline void THDF5_File::WriteStringAttribute(const hid_t   parentGroup,
                                             const char*   datasetName,
                                             const char*   attributeName,
                                             const string& value)
{
  herr_t status = H5LTset_attribute_string(parentGroup,
                                           datasetName,
                                           attributeName,
                                           value.c_str());
  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotWriteToAttribute, attributeName, datasetName);
    throw ios::failure(errMsg);
  }
}// end of WriteIntAttribute
//--------------------------------------------------------------------------------------------------


/**
 * Read integer attribute  at a specified place in the file tree.
 *
 * @param [in] parentGroup   - Where the dataset is
 * @param [in] datasetName   - Dataset name
 * @param [in] attributeName - Attribute name
 * @return Attribute value
 * @throw ios::failure
 */
inline string THDF5_File::ReadStringAttribute(const hid_t parentGroup,
                                              const char* datasetName,
                                              const char* attributeName)
{
  char value[256] = "";
  herr_t status = H5LTget_attribute_string(parentGroup,
                                           datasetName,
                                           attributeName,
                                           value);

  if (status < 0)
  {
    char errMsg[256];
    snprintf(errMsg, 256, HDF5_ERR_FMT_CouldNotReadFromAttribute, attributeName, datasetName);
    throw ios::failure(errMsg);
  }

  return value;
}// end of ReadIntAttribute
//--------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------//
//------------------------------------------ THDF5_File ------------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//


//------------------------------------------------------------------------------------------------//
//------------------------------------------ THDF5_File ------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//








//------------------------------------------------------------------------------------------------//
//-------------------------------------- THDF5_File_Header ---------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Constructor.
 */
THDF5_FileHeader::THDF5_FileHeader()
{
  headerValues.clear();
  PopulateHeaderFileMap();
}// end of constructor
//--------------------------------------------------------------------------------------------------

/**
 * Copy constructor.
 * @param [in] src - Source object
 */
THDF5_FileHeader::THDF5_FileHeader(const THDF5_FileHeader& src)
{
  headerValues = src.headerValues;
  headerNames = src.headerNames;
}// end of copy constructor
//--------------------------------------------------------------------------------------------------

/**
 * Destructor.
 *
 */
THDF5_FileHeader::~THDF5_FileHeader()
{
  headerValues.clear();
  headerNames.clear();
}// end of destructor
//--------------------------------------------------------------------------------------------------


/**
 * Read header from the input file.
 *
 * @param [in, out] inputFile - Input file handle
 * @throw ios:failure when en error happens
 */
void THDF5_FileHeader::ReadHeaderFromInputFile(THDF5_File& inputFile)
{
  // Get file root handle
  hid_t rootGroup = inputFile.GetRootGroup();
  // read file type
  headerValues[FILE_TYPE] =
          inputFile.ReadStringAttribute(rootGroup,
                                        "/",
                                        headerNames[FILE_TYPE].c_str());

  if (GetFileType() == INPUT)
  {
    headerValues[CREATED_BY]
            = inputFile.ReadStringAttribute(rootGroup,
                                            "/",
                                            headerNames[CREATED_BY].c_str());
    headerValues[CREATION_DATA]
            = inputFile.ReadStringAttribute(rootGroup,
                                            "/",
                                            headerNames[CREATION_DATA].c_str());
    headerValues[FILE_DESCRIPTION]
            = inputFile.ReadStringAttribute(rootGroup,
                                            "/",
                                            headerNames[FILE_DESCRIPTION].c_str());
    headerValues[MAJOR_VERSION]
            = inputFile.ReadStringAttribute(rootGroup,
                                            "/",
                                            headerNames[MAJOR_VERSION].c_str());
    headerValues[MINOR_VERSION]
            = inputFile.ReadStringAttribute(rootGroup,
                                            "/",
                                            headerNames[MINOR_VERSION].c_str());
  }
  else
  {
    throw ios::failure(HDF5_ERR_FMT_BadInputFileType);
  }
}// end of ReadHeaderFromInputFile
//--------------------------------------------------------------------------------------------------


/**
 * Read header from output file (necessary for checkpoint-restart). Read only execution times
 * (the others are read from the input file, or calculated based on the very last leg of the
 * simulation).
 * This function is called only if checkpoint-restart is enabled.
 *
 * @param [in, out] outputFile - Output file handle
 */
void THDF5_FileHeader::ReadHeaderFromOutputFile(THDF5_File& outputFile)
{
  // Get file root handle
  hid_t rootGroup = outputFile.GetRootGroup();

  headerValues[FILE_TYPE]
          = outputFile.ReadStringAttribute(rootGroup,
                                           "/",
                                           headerNames[FILE_TYPE].c_str());

  if (GetFileType() == OUTPUT)
  {
    headerValues[TOTAL_EXECUTION_TIME]
            = outputFile.ReadStringAttribute(rootGroup,
                                             "/",
                                             headerNames[TOTAL_EXECUTION_TIME].c_str());
    headerValues[DATA_LOAD_TIME]
            = outputFile.ReadStringAttribute(rootGroup,
                                             "/",
                                             headerNames[DATA_LOAD_TIME].c_str());
    headerValues[PREPROCESSING_TIME]
            = outputFile.ReadStringAttribute(rootGroup,
                                             "/",
                                             headerNames[PREPROCESSING_TIME].c_str());
    headerValues[SIMULATION_TIME]
            = outputFile.ReadStringAttribute(rootGroup,
                                             "/",
                                             headerNames[SIMULATION_TIME].c_str());
    headerValues[POST_PROCESSING_TIME]
            = outputFile.ReadStringAttribute(rootGroup,
                                             "/",
                                             headerNames[POST_PROCESSING_TIME].c_str());
  }
  else
  {
    throw ios::failure(HDF5_ERR_FMT_BadOutputFileType);
  }
}// end of ReadHeaderFromOutputFile
//--------------------------------------------------------------------------------------------------


/**
 * Read the file header form the checkpoint file. We need the header to verify the file version
 * and type.
 * @param [in, out] checkpointFile - Checkpoint file handle
 */
void THDF5_FileHeader::ReadHeaderFromCheckpointFile(THDF5_File & checkpointFile)
{
  // Get file root handle
  hid_t rootGroup = checkpointFile.GetRootGroup();
  // read file type
  headerValues[FILE_TYPE] =
          checkpointFile.ReadStringAttribute(rootGroup,
                                             "/",
                                             headerNames[FILE_TYPE].c_str());

  if (GetFileType() == CHECKPOINT)
  {
    headerValues[CREATED_BY]
            = checkpointFile.ReadStringAttribute(rootGroup,
                                                 "/",
                                                 headerNames[CREATED_BY].c_str());
    headerValues[CREATION_DATA]
            = checkpointFile.ReadStringAttribute(rootGroup,
                                                 "/",
                                                 headerNames[CREATION_DATA].c_str());
    headerValues[FILE_DESCRIPTION]
            = checkpointFile.ReadStringAttribute(rootGroup,
                                                 "/",
                                                 headerNames[FILE_DESCRIPTION].c_str());
    headerValues[MAJOR_VERSION]
            = checkpointFile.ReadStringAttribute(rootGroup,
                                                 "/",
                                                 headerNames[MAJOR_VERSION].c_str());
    headerValues[MINOR_VERSION]
            = checkpointFile.ReadStringAttribute(rootGroup,
                                                 "/",
                                                 headerNames[MINOR_VERSION].c_str());
  }
  else
  {
    throw ios::failure(HDF5_ERR_FMT_BadCheckpointFileType);
  }
}// end of ReadHeaderFromCheckpointFile
//--------------------------------------------------------------------------------------------------

/**
 * Write header into the output file.
 *
 * @param [in,out] outputFile - Output file handle
 */
void THDF5_FileHeader::WriteHeaderToOutputFile(THDF5_File& outputFile)
{
  // Get file root handle
  hid_t rootGroup = outputFile.GetRootGroup();

  for (const auto& it : headerNames)
  {
    outputFile.WriteStringAttribute(rootGroup,
                                    "/",
                                    it.second.c_str(),
                                    headerValues[it.first]);
  }
}// end of WriteHeaderToOutputFile
//--------------------------------------------------------------------------------------------------

/**
 * Write header to the output file (only a subset of all possible fields are written).
 *
 * @param [in, out] checkpointFile - Checkpoint file handle
 */
void THDF5_FileHeader::WriteHeaderToCheckpointFile(THDF5_File& checkpointFile)
{
  // Get file root handle
  hid_t rootGroup = checkpointFile.GetRootGroup();

  // Write header
  checkpointFile.WriteStringAttribute(rootGroup,
                                      "/",
                                      headerNames [FILE_TYPE].c_str(),
                                      headerValues[FILE_TYPE].c_str());

  checkpointFile.WriteStringAttribute(rootGroup,
                                      "/",
                                      headerNames [CREATED_BY].c_str(),
                                      headerValues[CREATED_BY].c_str());

  checkpointFile.WriteStringAttribute(rootGroup,
                                      "/",
                                      headerNames [CREATION_DATA].c_str(),
                                      headerValues[CREATION_DATA].c_str());

  checkpointFile.WriteStringAttribute(rootGroup,
                                      "/",
                                      headerNames [FILE_DESCRIPTION].c_str(),
                                      headerValues[FILE_DESCRIPTION].c_str());

  checkpointFile.WriteStringAttribute(rootGroup,                                      "/",
                                      headerNames [MAJOR_VERSION].c_str(),
                                      headerValues[MAJOR_VERSION].c_str());

  checkpointFile.WriteStringAttribute(rootGroup,
                                      "/",
                                      headerNames [MINOR_VERSION].c_str(),
                                      headerValues[MINOR_VERSION].c_str());
}// end of WriteHeaderToCheckpointFile
//--------------------------------------------------------------------------------------------------


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

  char dateString[21];

  snprintf(dateString, 20, "%02i/%02i/%02i, %02i:%02i:%02i",
          current->tm_mday, current->tm_mon + 1, current->tm_year - 100,
          current->tm_hour, current->tm_min, current->tm_sec);

  headerValues[CREATION_DATA] = dateString;
}// end of SetCreationTime
//--------------------------------------------------------------------------------------------------

/**
 * Get file version as an enum.
 * @return File version as an enum
 */
THDF5_FileHeader::TFileVersion THDF5_FileHeader::GetFileVersion()
{
  if ((headerValues[MAJOR_VERSION] == hdf5_MajorFileVersionsNames[0]) &&
      (headerValues[MINOR_VERSION] == hdf5_MinorFileVersionsNames[0]))
  {
    return VERSION_10;
  }

  if ((headerValues[MAJOR_VERSION] == hdf5_MajorFileVersionsNames[0]) &&
      (headerValues[MINOR_VERSION] == hdf5_MinorFileVersionsNames[1]))
  {
    return VERSION_11;
  }

  return VERSION_UNKNOWN;
}// end of GetFileVersion
//--------------------------------------------------------------------------------------------------



/**
 * Get File type.
 *
 * @return File type
 */
THDF5_FileHeader::TFileType  THDF5_FileHeader::GetFileType()
{
  for (int i = INPUT; i < UNKNOWN ; i++)
  {
    if (headerValues[FILE_TYPE] == hdf5_FileTypesNames[static_cast<TFileType >(i)])
    {
      return static_cast<TFileType >(i);
    }
  }

  return THDF5_FileHeader::UNKNOWN;
}// end of GetFileType
//--------------------------------------------------------------------------------------------------

/**
 * Set File type.
 *
 * @param [in] fileType - File type
 */
void THDF5_FileHeader::SetFileType(const THDF5_FileHeader::TFileType fileType)
{
  headerValues[FILE_TYPE] = hdf5_FileTypesNames[fileType];
}// end of SetFileType
//--------------------------------------------------------------------------------------------------



/**
 * Set Host name.
 *
 */
void THDF5_FileHeader::SetHostName()
{
  char hostName[256];

  //Linux build
  #ifdef __linux__
    gethostname(hostName, 256);
  #endif

  //Windows build
  #ifdef _WIN64
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
	  gethostname(hostName, 256);

    WSACleanup();
  #endif

  headerValues[HOST_NAME] = hostName;
}// end of SetHostName
//--------------------------------------------------------------------------------------------------


/**
 * Set memory consumption.
 *
 * @param [in] totalMemory - Total memory consumption
 */
void THDF5_FileHeader::SetMemoryConsumption(const size_t totalMemory)
{
  char text[20] = "";
  snprintf(text, 20, "%ld MB",totalMemory);

  headerValues[TOTAL_MEMORY_CONSUMPTION]     = text;

  snprintf(text, 20, "%ld MB",totalMemory / TParameters::GetInstance().GetNumberOfThreads());
  headerValues[PEAK_CORE_MEMORY_CONSUMPTION] = text;
}// end of SetMemoryConsumption
//--------------------------------------------------------------------------------------------------


/**
 * Set execution times in file header.
 *
 * @param [in] totalTime          - Total time
 * @param [in] loadTime           - Time to load data
 * @param [in] preProcessingTime  - Preprocessing time
 * @param [in] simulationTime     - Simulation time
 * @param [in] postprocessingTime - Post processing time
 */
void THDF5_FileHeader::SetExecutionTimes(const double totalTime,
                                         const double loadTime,
                                         const double preProcessingTime,
                                         const double simulationTime,
                                         const double postprocessingTime)
{
  char text [30] = "";

  snprintf(text, 30, "%8.2fs", totalTime);
  headerValues[TOTAL_EXECUTION_TIME] = text;

  snprintf(text, 30,  "%8.2fs", loadTime);
  headerValues[DATA_LOAD_TIME] = text;

  snprintf(text, 30, "%8.2fs", preProcessingTime);
  headerValues[PREPROCESSING_TIME] = text;


  snprintf(text, 30, "%8.2fs", simulationTime);
  headerValues[SIMULATION_TIME] = text;

  snprintf(text, 30, "%8.2fs", postprocessingTime);
  headerValues[POST_PROCESSING_TIME] = text;
}// end of SetExecutionTimes
//--------------------------------------------------------------------------------------------------

/**
 * Get execution times stored in the output file header.
 *
 * @param [out] totalTime          - Total time
 * @param [out] loadTime           - Time to load data
 * @param [out] preProcessingTime  - Preprocessing time
 * @param [out] simulationTime     - Simulation time
 * @param [out] postprocessingTime - Post processing time
 */
void THDF5_FileHeader::GetExecutionTimes(double& totalTime,
                                         double& loadTime,
                                         double& preProcessingTime,
                                         double& simulationTime,
                                         double& postprocessingTime)
{
  totalTime          = std::stof(headerValues[TOTAL_EXECUTION_TIME]);
  loadTime           = std::stof(headerValues[DATA_LOAD_TIME]);
  preProcessingTime  = std::stof(headerValues[PREPROCESSING_TIME]);
  simulationTime     = std::stof(headerValues[SIMULATION_TIME]);
  postprocessingTime = std::stof(headerValues[POST_PROCESSING_TIME]);
}// end of GetExecutionTimes
//--------------------------------------------------------------------------------------------------

/**
 * Set Number of cores.
 *
 */
void THDF5_FileHeader::SetNumberOfCores()
{
  char text[12] = "";
  snprintf(text, 12, "%ld", TParameters::GetInstance().GetNumberOfThreads());

  headerValues[NUMBER_OF_CORES] = text;
}// end of SetNumberOfCores
//--------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------//
//-------------------------------------- THDF5_File_Header ---------------------------------------//
//-------------------------------------- Protected methods ---------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Create map with names for the header.
 *
 */
void THDF5_FileHeader::PopulateHeaderFileMap()
{
  headerNames.clear();

  headerNames[CREATED_BY] = "created_by";
  headerNames[CREATION_DATA] = "creation_date";
  headerNames[FILE_DESCRIPTION] = "file_description";
  headerNames[MAJOR_VERSION] = "major_version";
  headerNames[MINOR_VERSION] = "minor_version";
  headerNames[FILE_TYPE] = "file_type";

  headerNames[HOST_NAME] = "host_names";
  headerNames[NUMBER_OF_CORES] = "number_of_cpu_cores";
  headerNames[TOTAL_MEMORY_CONSUMPTION] = "total_memory_in_use";
  headerNames[PEAK_CORE_MEMORY_CONSUMPTION] = "peak_core_memory_in_use";

  headerNames[TOTAL_EXECUTION_TIME] = "total_execution_time";
  headerNames[DATA_LOAD_TIME] = "data_loading_phase_execution_time";
  headerNames[PREPROCESSING_TIME] = "pre-processing_phase_execution_time";
  headerNames[SIMULATION_TIME] = "simulation_phase_execution_time";
  headerNames[POST_PROCESSING_TIME] = "post-processing_phase_execution_time";
}// end of PopulateHeaderFileMap
//--------------------------------------------------------------------------------------------------
