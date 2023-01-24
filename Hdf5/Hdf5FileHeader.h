/**
 * @file      Hdf5FileHeader.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file containing the class processing file headers.
 *            Detail about the file header are described below.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      20 July      2017, 14:42 (created) \n
 *            06 March     2019, 12:57 (revised)
 *
 * @section   Hdf5FileHeader HDF5 File Header Structure
 *
 *The header includes following information
 *
 * \verbatim
+----------------------------------------------------------------------------------------------------------------------+
|                                           Input File / Checkpoint File Header                                        |
+----------------------------------------------------------------------------------------------------------------------+
| created_by                              Short description of the tool that created this file                         |
| creation_date                           Date when the file was created                                               |
| file_description                        Short description of the content of the file (e.g. simulation name)          |
| file_type                               Type of the file (input)                                                     |
| major_version                           Major version of the file definition (1)                                     |
| minor_version                           Minor version of the file definition (1)                                     |
+----------------------------------------------------------------------------------------------------------------------+
 \endverbatim
 *
 * \verbatim
+----------------------------------------------------------------------------------------------------------------------+
|                                                    Output File Header                                                |
+----------------------------------------------------------------------------------------------------------------------+
| created_by                              Short description of the tool that created this file                         |
| creation_date                           Date when the file was created                                               |
| file_description                        Short description of the content of the file (e.g. simulation name)          |
| file_type                               Type of the file (output)                                                    |
| major_version                           Major version of the file definition (1)                                     |
| minor_version                           Minor version of the file definition (1)                                     |
+----------------------------------------------------------------------------------------------------------------------+
| host_names                              List of hosts (computer names) the simulation was executed on                |
| number_of_cpu_cores                     Number of CPU cores used for the simulation                                  |
| data_loading_phase_execution_time       Time taken to load data from the file                                        |
| pre-processing_phase_execution_time     Time taken to pre-process data                                               |
| simulation_phase_execution_time         Time taken to run the simulation                                             |
| post-processing_phase_execution_time    Time taken to complete the post-processing phase                             |
| total_execution_time                    Total execution time                                                         |
| peak_core_memory_in_use                 Peak memory required per core during the simulation                          |
| total_memory_in_use Total               Peak memory in use                                                           |
+----------------------------------------------------------------------------------------------------------------------+
 \endverbatim
 *
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

#ifndef HDF5_FILE_HEADER_H
#define HDF5_FILE_HEADER_H

#include <map>

#include <Hdf5/Hdf5File.h>
#include <Utils/DimensionSizes.h>
#include <Utils/MatrixNames.h>

/**
 * @class   Hdf5FileHeader
 * @brief   Class for HDF5 file header.
 * @details This class manages all information that can be stored in the input output or checkpoint
 * file header.
 */
class Hdf5FileHeader {
public:
  /**
   * @enum    FileHeaderItems
   * @brief   List of all header items.
   * @details List of all header items.
   * @todo    In the future, we should add number of GPUs, peak GPU memory.
   */
  enum class FileHeaderItems {
    /// Code that created the file.
    kCreatedBy = 0,
    /// When the file was created.
    kCreationDate = 1,
    /// Description of the file (e.g. simulation).
    kFileDescription = 2,
    /// Major file version.
    kMajorVersion = 3,
    /// Minor file version.
    kMinorVersion = 4,
    /// File type.
    kFileType = 5,
    /// Machines the code were executed on.
    kHostName = 6,
    /// Total amount of memory consumed by the code.
    kTotalMemoryConsumption = 7,
    /// Peak memory consumption (by process).
    kPeakMemoryConsumption = 8,
    /// Total execution time.
    kTotalExecutionTime = 9,
    /// Time to load data in.
    kDataLoadTime = 10,
    /// Time to preprocess data.
    kPreProcessingTime = 11,
    /// Simulation time.
    kSimulationTime = 12,
    /// Time to postprocess data.
    kPostProcessingTime = 13,
    /// Number of cores the simulation was executed.
    kNumberOfCores = 14
  };

  /**
   * @enum    FileType
   * @brief   HDF5 file type.
   * @details HDF5 file type.
   */
  enum class FileType {
    /// Input file.
    kInput = 0,
    /// Output file.
    kOutput = 1,
    /// Checkpoint file.
    kCheckpoint = 2,
    /// Unknown file.
    kUnknown = 3
  };

  /**
   * @enum    FileVersion
   * @brief   HDF5 file version.
   * @details HDF5 file version.
   */
  enum class FileVersion {
    /// Version 1.0.
    kVersion10 = 0,
    /// Version 1.1.
    kVersion11 = 1,
    /// Version unknown.
    kVersionUnknown = 2
  };

  /// Constructor.
  Hdf5FileHeader();
  /**
   * @brief Copy constructor.
   * @param [in] src - Source object.
   */
  Hdf5FileHeader(const Hdf5FileHeader& src);
  /// Destructor.
  ~Hdf5FileHeader();

  /**
   * @brief Read header from the input file.
   *
   * @param [in, out] inputFile - Input file handle.
   * @throw ios:failure         - If error happens.
   */
  void readHeaderFromInputFile(Hdf5File& inputFile);
  /**
   * @brief Read header from output file (necessary for checkpoint-restart).
   *
   * Read only execution times (the others are read from the input file, or calculated based on the very last
   * leg of the simulation). This function is called only if checkpoint-restart is enabled.
   *
   * @param [in, out] outputFile - Output file handle.
   * @throw ios:failure          - If error happens.
   */
  void readHeaderFromOutputFile(Hdf5File& outputFile);
  /**
   * @brief Read the file header form the checkpoint file.
   *
   * We need the header to verify the file version and type.
   * @param [in, out] checkpointFile - Checkpoint file handle.
   * @throw ios:failure              - If error happens.
   */
  void readHeaderFromCheckpointFile(Hdf5File& checkpointFile);

  /**
   * @brief Write header into the output file.
   *
   * @param [in,out] outputFile - Output file handle.
   * @throw ios:failure         - If error happens.
   */
  void writeHeaderToOutputFile(Hdf5File& outputFile);
  /**
   * @brief Write header to the output file (only a subset of all possible fields are written).
   *
   * @param [in, out] checkpointFile - Checkpoint file handle.
   * @throw ios:failure              - If error happens.
   */
  void writeHeaderToCheckpointFile(Hdf5File& checkpointFile);

  /**
   * @brief Set code name.
   * @param [in] codeName - Code version.
   */
  void setCodeName(const std::string& codeName) {
    mHeaderValues[FileHeaderItems::kCreatedBy] = codeName;
  };

  /// Set creation time.
  void setActualCreationTime();

  /**
   * @brief   Get string representing of current Major version of the file.
   * @return  Current major file version.
   */
  static std::string getFileMajorVersion() {
    return kMajorFileVersionsNames[0];
  };

  /**
   * @brief  Get string representing of current Minor version of the file.
   * @return Current minor file version.
   */
  static std::string getFileMinorVersion() {
    return kMinorFileVersionsNames[1];
  };

  /// Set major file version.
  void setMajorFileVersion() {
    mHeaderValues[FileHeaderItems::kMajorVersion] = getFileMajorVersion();
  };

  /// Set minor file version.
  void setMinorFileVersion() {
    mHeaderValues[FileHeaderItems::kMinorVersion] = getFileMinorVersion();
  };

  /**
   * @brief  Get file version as an enum.
   * @return File version as an enum.
   */
  FileVersion getFileVersion();

  /**
   * @brief  Check major file version.
   * @return true - If the file version is supported.
   */
  bool checkMajorFileVersion() {
    return (mHeaderValues[FileHeaderItems::kMajorVersion] == getFileMajorVersion());
  };
  /**
   * @brief  Check minor file version.
   * @return true - If the file version is supported.
   */
  bool checkMinorFileVersion() {
    return (mHeaderValues[FileHeaderItems::kMinorVersion] <= getFileMinorVersion());
  };

  /**
   * @brief  Get File type.
   * @return File type.
   */
  Hdf5FileHeader::FileType getFileType();
  /**
   * @brief Set File type.
   * @param [in] fileType - File type.
   */
  void setFileType(const Hdf5FileHeader::FileType fileType);

  /// Set host name.
  void setHostName();
  /**
   * @brief Set memory consumption.
   * @param [in] totalMemory - Total memory consumption.
   */
  void setMemoryConsumption(const size_t totalMemory);
  /**
   * @brief Set execution times in file header.
   *
   * @param [in] totalTime          - Total time.
   * @param [in] loadTime           - Time to load data.
   * @param [in] preProcessingTime  - Preprocessing time.
   * @param [in] simulationTime     - Simulation time.
   * @param [in] postprocessingTime - Post processing time.
   */
  void setExecutionTimes(const double totalTime,
                         const double loadTime,
                         const double preProcessingTime,
                         const double simulationTime,
                         const double postprocessingTime);
  /**
   * @brief Get execution times stored in the output file header.
   *
   * @param [out] totalTime          - Total time.
   * @param [out] loadTime           - Time to load data.
   * @param [out] preProcessingTime  - Preprocessing time.
   * @param [out] simulationTime     - Simulation time.
   * @param [out] postprocessingTime - Post processing time.
   */
  void getExecutionTimes(double& totalTime,
                         double& loadTime,
                         double& preProcessingTime,
                         double& simulationTime,
                         double& postprocessingTime);
  /// Set number of cores.
  void setNumberOfCores();

private:
  /// map for the header values.
  std::map<FileHeaderItems, std::string> mHeaderValues;
  /// map for the header names.
  static std::map<FileHeaderItems, std::string> sHeaderNames;

  ///String representation of different file types.
  static const std::string kFileTypesNames[];
  /// String representations of Major file versions.
  static const std::string kMajorFileVersionsNames[];
  /// String representations of Major file versions.
  static const std::string kMinorFileVersionsNames[];

}; // Hdf5FileHeader
//----------------------------------------------------------------------------------------------------------------------

#endif /* HDF5_FILE_HEADER_H */
