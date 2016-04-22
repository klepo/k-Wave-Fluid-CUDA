/**
 * @file        OutputMessages.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing all messages going to the standard output.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        19 April    2016, 12:52 (created) \n
 *              22 April    2016, 15:24 (revised)
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

#ifndef OUTPUT_MESSAGES_H
#define OUTPUT_MESSAGES_H

/**
 * @typedef TOutputMessage
 * @brief Datatype for output messages
 */
typedef const char* const TOutputMessage;

// colors are defined here
//http://stackoverflow.com/questions/3585846/color-text-in-terminal-aplications-in-unix

//----------------------------- Main module outputs --------------------------//

/// Main file output message
TOutputMessage Main_OUT_FMT_SmallSeparator  = "--------------------------------------\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_CodeName  = "     %s\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_NumberOfThreads = "Number of CPU threads:       %9zu\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_Initialisatoin  = "........... Initialization ...........\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_MemoryAllocation= "Memory allocation                 ";
/// Main file output message
TOutputMessage Main_OUT_FMT_DataLoading     = "Data loading                      ";
/// Main file output message
TOutputMessage Main_OUT_FMT_InitElapsedTime = "Elapsed time:             %11.2fs\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_RecoveredForm = "Recovered from t_index:       %8ld\n\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_Computation     = "............. Computation ............\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_Summary         = "............... Summary ..............\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_HostMemoryUsage  = "Peak host memory in use:    %8zuMB\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_DeviceMemoryUsage= "Peak device memory in use:  %8zuMB\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_TotalExecutionTime= "Total execution time:        %8.2fs\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_LegExecutionTime  = "This leg execution time:     %8.2fs\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_EndOfComputation  = "         End of computation  \n";


/// Main file output message
TOutputMessage Main_OUT_FMT_Done = "Done\n";
/// Main file output message
//TOutputMessage Main_OUT_FMT_Failed = "\x1B[31m\b\b\b Failed\033[0m\n" ;
TOutputMessage Main_OUT_FMT_Failed = "\b\bFailed\n" ;


//----------------------- Parameters module outputs --------------------------//
/// Parameter module log message
TOutputMessage Parameters_OUT_FMT_SelectedGPUDeviceID = "Selected GPU device id:      ";
/// Parameter module log message
TOutputMessage Parameters_OUT_FMT_PrintOutGPUDevice   = "%9d\n";
/// Parameter module log message
TOutputMessage Parameters_OUT_FMT_GPUDeviceInfo       = "GPU Device info: %21s\n";
/// Parameter module log message
TOutputMessage Parameters_OUT_FMT_DomainSize          = "Domain dims:        [%4lu, %4lu, %4lu]\n";
/// Parameter module log message
TOutputMessage Parameters_OUT_FMT_Length              = "Simulation time steps:       %9lu\n";





#endif /* OUTPUTMESSAGES_H */

