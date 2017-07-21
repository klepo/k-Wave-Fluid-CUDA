/**
 * @file        Hdf5FileHeader.cpp
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the HDF5 related classes.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        20 July      2017, 14:54 (created) \n
 *              21 July      2017, 16:48 (revised)
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

#include <Hdf5/Hdf5FileHeader.h>
#include <Parameters/Parameters.h>

#include <Logger/ErrorMessages.h>
#include <Logger/Logger.h>

using std::ios;
using std::string;

//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Constants -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

const string Hdf5FileHeader::kFileTypesNames[]  = {"input", "output", "checkpoint", "unknown"};

const string Hdf5FileHeader::kMajorFileVersionsNames[] = {"1"};
const string Hdf5FileHeader::kMinorFileVersionsNames[] = {"0","1"};

/**
 * Initialisation of static map with header attribute names.
 */
std::map<Hdf5FileHeader::FileHeaderItems, std::string> Hdf5FileHeader::sHeaderNames
{
  {Hdf5FileHeader::FileHeaderItems::kCreatedBy             , "created_by"},
  {Hdf5FileHeader::FileHeaderItems::kCreationDate          , "creation_date"},
  {Hdf5FileHeader::FileHeaderItems::kFileDescription       , "file_description"},
  {Hdf5FileHeader::FileHeaderItems::kMajorVersion          , "major_version"},
  {Hdf5FileHeader::FileHeaderItems::kMinorVersion          , "minor_version"},
  {Hdf5FileHeader::FileHeaderItems::kFileType              , "file_type"},

  {Hdf5FileHeader::FileHeaderItems::kHostName              , "host_names"},
  {Hdf5FileHeader::FileHeaderItems::kNumberofCores         , "number_of_cpu_cores"},
  {Hdf5FileHeader::FileHeaderItems::kTotalMemoryConsumption, "total_memory_in_use"},
  {Hdf5FileHeader::FileHeaderItems::kPeakMemoryConsumption , "peak_core_memory_in_use"},

  {Hdf5FileHeader::FileHeaderItems::kTotalExecutionTime    , "total_execution_time"},
  {Hdf5FileHeader::FileHeaderItems::kDataLoadTime          , "data_loading_phase_execution_time"},
  {Hdf5FileHeader::FileHeaderItems::kPreProcessingTime     , "pre-processing_phase_execution_time"},
  {Hdf5FileHeader::FileHeaderItems::kSimulationTime        , "simulation_phase_execution_time"},
  {Hdf5FileHeader::FileHeaderItems::kPostProcessingTime    , "post-processing_phase_execution_time"}
};// mHeaderNames
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor.
 */
Hdf5FileHeader::Hdf5FileHeader()
  : mHeaderValues()
{

}// end of constructor
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy constructor.
 */
Hdf5FileHeader::Hdf5FileHeader(const Hdf5FileHeader& src)
  : mHeaderValues(src.mHeaderValues)
{

}// end of copy constructor
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor.
 */
Hdf5FileHeader::~Hdf5FileHeader()
{
  mHeaderValues.clear();
}// end of destructor
//----------------------------------------------------------------------------------------------------------------------


/**
 * Read header from the input file.
 */
void Hdf5FileHeader::readHeaderFromInputFile(Hdf5File& inputFile)
{
  // Get file root handle
  hid_t rootGroup = inputFile.getRootGroup();
  // read file type
  mHeaderValues[FileHeaderItems::kFileType] =
          inputFile.readStringAttribute(rootGroup,
                                        "/",
                                        sHeaderNames[FileHeaderItems::kFileType].c_str());

  if (getFileType() == FileType::kInput)
  {
    mHeaderValues[FileHeaderItems::kCreatedBy]
            = inputFile.readStringAttribute(rootGroup,
                                            "/",
                                            sHeaderNames[FileHeaderItems::kCreatedBy].c_str());
    mHeaderValues[FileHeaderItems::kCreationDate]
            = inputFile.readStringAttribute(rootGroup,
                                            "/",
                                            sHeaderNames[FileHeaderItems::kCreationDate].c_str());
    mHeaderValues[FileHeaderItems::kFileDescription]
            = inputFile.readStringAttribute(rootGroup,
                                            "/",
                                            sHeaderNames[FileHeaderItems::kFileDescription].c_str());
    mHeaderValues[FileHeaderItems::kMajorVersion]
            = inputFile.readStringAttribute(rootGroup,
                                            "/",
                                            sHeaderNames[FileHeaderItems::kMajorVersion].c_str());
    mHeaderValues[FileHeaderItems::kMinorVersion]
            = inputFile.readStringAttribute(rootGroup,
                                            "/",
                                            sHeaderNames[FileHeaderItems::kMinorVersion].c_str());
  }
  else
  {
    throw ios::failure(kErrFmtBadInputFileType);
  }
}// end of readHeaderFromInputFile
//----------------------------------------------------------------------------------------------------------------------


/**
 * Read header from output file (necessary for checkpoint-restart).
 */
void Hdf5FileHeader::readHeaderFromOutputFile(Hdf5File& outputFile)
{
  // Get file root handle
  hid_t rootGroup = outputFile.getRootGroup();

  mHeaderValues[FileHeaderItems::kFileType]
          = outputFile.readStringAttribute(rootGroup,
                                           "/",
                                           sHeaderNames[FileHeaderItems::kFileType].c_str());

  if (getFileType() == FileType::kOutput)
  {
    mHeaderValues[FileHeaderItems::kTotalExecutionTime]
            = outputFile.readStringAttribute(rootGroup,
                                             "/",
                                             sHeaderNames[FileHeaderItems::kTotalExecutionTime].c_str());
    mHeaderValues[FileHeaderItems::kDataLoadTime]
            = outputFile.readStringAttribute(rootGroup,
                                             "/",
                                             sHeaderNames[FileHeaderItems::kDataLoadTime].c_str());
    mHeaderValues[FileHeaderItems::kPreProcessingTime]
            = outputFile.readStringAttribute(rootGroup,
                                             "/",
                                             sHeaderNames[FileHeaderItems::kPreProcessingTime].c_str());
    mHeaderValues[FileHeaderItems::kSimulationTime]
            = outputFile.readStringAttribute(rootGroup,
                                             "/",
                                             sHeaderNames[FileHeaderItems::kSimulationTime].c_str());
    mHeaderValues[FileHeaderItems::kPostProcessingTime]
            = outputFile.readStringAttribute(rootGroup,
                                             "/",
                                             sHeaderNames[FileHeaderItems::kPostProcessingTime].c_str());
  }
  else
  {
    throw ios::failure(kErrFmtBadOutputFIleType);
  }
}// end of readHeaderFromOutputFile
//----------------------------------------------------------------------------------------------------------------------


/**
 * Read the file header form the checkpoint file.
 */
void Hdf5FileHeader::readHeaderFromCheckpointFile(Hdf5File& checkpointFile)
{
  // Get file root handle
  hid_t rootGroup = checkpointFile.getRootGroup();
  // read file type
  mHeaderValues[FileHeaderItems::kFileType] =
          checkpointFile.readStringAttribute(rootGroup, "/", sHeaderNames[FileHeaderItems::kFileType]);

  if (getFileType() == FileType::kCheckpoint)
  {
    mHeaderValues[FileHeaderItems::kCreatedBy]
            = checkpointFile.readStringAttribute(rootGroup, "/", sHeaderNames[FileHeaderItems::kCreatedBy]);
    mHeaderValues[FileHeaderItems::kCreationDate]
            = checkpointFile.readStringAttribute(rootGroup, "/", sHeaderNames[FileHeaderItems::kCreationDate]);
    mHeaderValues[FileHeaderItems::kFileDescription]
            = checkpointFile.readStringAttribute(rootGroup, "/", sHeaderNames[FileHeaderItems::kFileDescription]);
    mHeaderValues[FileHeaderItems::kMajorVersion]
            = checkpointFile.readStringAttribute(rootGroup, "/", sHeaderNames[FileHeaderItems::kMajorVersion]);
    mHeaderValues[FileHeaderItems::kMinorVersion]
            = checkpointFile.readStringAttribute(rootGroup, "/", sHeaderNames[FileHeaderItems::kMinorVersion]);
  }
  else
  {
    throw ios::failure(kErrFmtBadCheckpointFileType);
  }
}// end of readHeaderFromCheckpointFile
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write header into the output file.
 */
void Hdf5FileHeader::writeHeaderToOutputFile(Hdf5File& outputFile)
{
  // Get file root handle
  hid_t rootGroup = outputFile.getRootGroup();

  for (const auto& it : sHeaderNames)
  {
    outputFile.writeStringAttribute(rootGroup, "/", it.second, mHeaderValues[it.first]);
  }
}// end of writeHeaderToOutputFile
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write header to the output file (only a subset of all possible fields are written).
 */
void Hdf5FileHeader::writeHeaderToCheckpointFile(Hdf5File& checkpointFile)
{
  // Get file root handle
  hid_t rootGroup = checkpointFile.getRootGroup();

  // Write header
  checkpointFile.writeStringAttribute(rootGroup,
                                      "/",
                                      sHeaderNames [FileHeaderItems::kFileType].c_str(),
                                      mHeaderValues[FileHeaderItems::kFileType].c_str());

  checkpointFile.writeStringAttribute(rootGroup,
                                      "/",
                                      sHeaderNames [FileHeaderItems::kCreatedBy].c_str(),
                                      mHeaderValues[FileHeaderItems::kCreatedBy].c_str());

  checkpointFile.writeStringAttribute(rootGroup,
                                      "/",
                                      sHeaderNames [FileHeaderItems::kCreationDate].c_str(),
                                      mHeaderValues[FileHeaderItems::kCreationDate].c_str());

  checkpointFile.writeStringAttribute(rootGroup,
                                      "/",
                                      sHeaderNames [FileHeaderItems::kFileDescription].c_str(),
                                      mHeaderValues[FileHeaderItems::kFileDescription].c_str());

  checkpointFile.writeStringAttribute(rootGroup,
                                      "/",
                                      sHeaderNames [FileHeaderItems::kMajorVersion].c_str(),
                                      mHeaderValues[FileHeaderItems::kMajorVersion].c_str());

  checkpointFile.writeStringAttribute(rootGroup,
                                      "/",
                                      sHeaderNames [FileHeaderItems::kMinorVersion].c_str(),
                                      mHeaderValues[FileHeaderItems::kMinorVersion].c_str());
}// end of writeHeaderToCheckpointFile
//----------------------------------------------------------------------------------------------------------------------


/**
 * Set actual date and time.
 */
void Hdf5FileHeader::setActualCreationTime()
{
  struct tm *current;
  time_t now;
  time(&now);
  current = localtime(&now);

  mHeaderValues[FileHeaderItems::kCreationDate] = Logger::formatMessage("%02i/%02i/%02i, %02i:%02i:%02i",
                                                  current->tm_mday, current->tm_mon + 1, current->tm_year - 100,
                                                  current->tm_hour, current->tm_min, current->tm_sec);

}// end of setCreationTime
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get file version as an enum.
 */
Hdf5FileHeader::FileVersion Hdf5FileHeader::getFileVersion()
{
  if ((mHeaderValues[FileHeaderItems::kMajorVersion] == kMajorFileVersionsNames[0]) &&
      (mHeaderValues[FileHeaderItems::kMinorVersion] == kMinorFileVersionsNames[0]))
  {
    return FileVersion::kVersion10;
  }

  if ((mHeaderValues[FileHeaderItems::kMajorVersion] == kMajorFileVersionsNames[0]) &&
      (mHeaderValues[FileHeaderItems::kMinorVersion] == kMinorFileVersionsNames[1]))
  {
    return FileVersion::kVersion11;
  }

  return FileVersion::kVersionUnknown;
}// end of getFileVersion
//----------------------------------------------------------------------------------------------------------------------


/**
 * Get File type.
 */
Hdf5FileHeader::FileType  Hdf5FileHeader::getFileType()
{
  for (int i = static_cast<int>(FileType::kInput); i < static_cast<int>(FileType::kUnknown); i++)
  {
    if (mHeaderValues[FileHeaderItems::kFileType] == kFileTypesNames[i])
    {
      return static_cast<FileType >(i);
    }
  }

  return Hdf5FileHeader::FileType::kUnknown;
}// end of getFileType
//----------------------------------------------------------------------------------------------------------------------

/**
 * Set File type.
 */
void Hdf5FileHeader::setFileType(const Hdf5FileHeader::FileType fileType)
{
  mHeaderValues[FileHeaderItems::kFileType] = kFileTypesNames[static_cast<int>(fileType)];
}// end of setFileType
//----------------------------------------------------------------------------------------------------------------------


/**
 * Set Host name.
 */
void Hdf5FileHeader::setHostName()
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

  mHeaderValues[FileHeaderItems::kHostName] = hostName;
}// end of setHostName
//----------------------------------------------------------------------------------------------------------------------


/**
 * Set memory consumption.
 */
void Hdf5FileHeader::setMemoryConsumption(const size_t totalMemory)
{
  mHeaderValues[FileHeaderItems::kTotalMemoryConsumption] = Logger::formatMessage("%ld MB", totalMemory);

  mHeaderValues[FileHeaderItems::kPeakMemoryConsumption]
          = Logger::formatMessage("%ld MB", totalMemory / Parameters::getInstance().getNumberOfThreads());

}// end of setMemoryConsumption
//----------------------------------------------------------------------------------------------------------------------


/**
 * Set execution times in file header.
 */
void Hdf5FileHeader::setExecutionTimes(const double totalTime,
                                       const double loadTime,
                                       const double preProcessingTime,
                                       const double simulationTime,
                                       const double postprocessingTime)
{
  mHeaderValues[FileHeaderItems::kTotalExecutionTime] = Logger::formatMessage("%8.2fs", totalTime);
  mHeaderValues[FileHeaderItems::kDataLoadTime]       = Logger::formatMessage("%8.2fs", loadTime);
  mHeaderValues[FileHeaderItems::kPreProcessingTime]  = Logger::formatMessage("%8.2fs", preProcessingTime);
  mHeaderValues[FileHeaderItems::kSimulationTime]     = Logger::formatMessage("%8.2fs", simulationTime);
  mHeaderValues[FileHeaderItems::kPostProcessingTime] = Logger::formatMessage("%8.2fs", postprocessingTime);
}// end of setExecutionTimes
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get execution times stored in the output file header.
 */
void Hdf5FileHeader::getExecutionTimes(double& totalTime,
                                       double& loadTime,
                                       double& preProcessingTime,
                                       double& simulationTime,
                                       double& postprocessingTime)
{
  totalTime          = std::stof(mHeaderValues[FileHeaderItems::kTotalExecutionTime]);
  loadTime           = std::stof(mHeaderValues[FileHeaderItems::kDataLoadTime]);
  preProcessingTime  = std::stof(mHeaderValues[FileHeaderItems::kPreProcessingTime]);
  simulationTime     = std::stof(mHeaderValues[FileHeaderItems::kSimulationTime]);
  postprocessingTime = std::stof(mHeaderValues[FileHeaderItems::kPostProcessingTime]);
}// end of getExecutionTimes
//----------------------------------------------------------------------------------------------------------------------

/**
 * Set Number of cores.
 */
void Hdf5FileHeader::setNumberOfCores()
{
  mHeaderValues[FileHeaderItems::kNumberofCores]
          = Logger::formatMessage("%ld", Parameters::getInstance().getNumberOfThreads());
}// end of setNumberOfCores
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Protected methods ------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//
