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
 *              03 May      2016, 17:33 (revised)
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
TOutputMessage Main_OUT_FMT_NumberOfThreads = "Number of CPU threads:       %9lu\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_Initialisatoin  = "........... Initialization ...........\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_MemoryAllocation= "Memory allocation:                ";
/// Main file output message
TOutputMessage Main_OUT_FMT_DataLoading     = "Data loading:                     ";
/// Main file output message
TOutputMessage Main_OUT_FMT_InitElapsedTime = "Elapsed time:             %11.2fs\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_RecoveredForm = "Recovered from t_index:       %8ld\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_Computation     = "............. Computation ............\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_Summary         = "............... Summary ..............\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_HostMemoryUsage  = "Peak host memory in use:    %8luMB\n";
/// Main file output message
TOutputMessage Main_OUT_FMT_DeviceMemoryUsage= "Peak device memory in use:  %8luMB\n";
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
/// ComandlineParamerers module log message
TOutputMessage TParamereres_OUT_FMT_SensorMaskTypeIndex  = "Sensor mask type:                Index\n";
/// ComandlineParamerers module log message
TOutputMessage TParamereres_OUT_FMT_SensorMaskTypeCuboid = "Sensor mask type:               Cuboid\n";
/// ComandlineParamerers module log message
TOutputMessage TParamereres_OUT_FMT_GitHashCentered = "               Git Hash     \n";


/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_ReadingInputFile       = "  Reading input file              ";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_ReadingCheckpointFile  = "  Reading checkpoint file         ";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_ReadingOuptutFile      = "  Reading output file             ";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_CreatingOutputFile     = "  Creating output file            ";



//------------------ TKSpaceFirstOrder3DSolver module outputs ----------------//
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_Version       = "kspaceFirstOrder3D-CUDA v1.1";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_FFTPlans      = "FFT plans creation:               ";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_PreProcessing = "Pre-processing phase:             ";


/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_CurrentHostMemory   = "Current host memory in use:   %6luMB\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_CurrentDeviceMemory = "Current device memory in use: %6luMB\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_PreProcessingTime   = "Elapsed time:                %8.2fs\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_LongSeparator        = "-------------------------------------------------------------\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_CheckpointInterrupt  = ".............. Interrupted to checkpoint! ...................\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_Simulation           = "....................... Simulation ..........................\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_ProgressHeader       = "Progress...ElapsedTime........TimeToGo......TimeOfTermination\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_Progress             = "%5li%c      %9.3fs      %9.3fs      %02i/%02i/%02i %02i:%02i:%02i\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_CheckpointTimeSteps  = "Number of time steps completed:                    %10u\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_ElapsedTimeLong    = "Elapsed time:                                       %8.2fs\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_CheckpointProgress = "Checkpoint in progress            ";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_PostProcessing     = "Post-processing phase             ";


//------------------ TKSpaceFirstOrder3DSolver module Print code version ----------------//
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_BuildNoDateTime =
    "\n"
    "+----------------------------------------------------+\n"
    "| Build Number:     kspaceFirstOrder3D v3.4          |\n"
    "| Build date:       %*.*s                      |\n"
    "| Build time:       %*.*s                         |\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_GitHash = "| Git hash: %s |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_LicenseEmptyLine = "|                                                    |\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_CodeLinux    = "| Operating System: Linux x64                        |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_CodeWindows  = "| Operating System: Windows x64                      |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_CodeMacOS    = "| Operating System: Mac OS X x64                     |\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_GNUCompiler   = "| Compiler name:    GNU C++ %.19s                    |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_IntelCompiler = "| Compiler name:    Intel C++ %d                   |\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_AVX2  = "| Instruction set:  Intel AVX 2                      |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_AVX   = "| Instruction set:  Intel AVX                        |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_SSE42 = "| Instruction set:  Intel SSE 4.2                    |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_SSE41 = "| Instruction set:  Intel SSE 4.1                    |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_SSE3  = "| Instruction set:  Intel SSE 3                      |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_SSE2  = "| Instruction set:  Intel SSE 2                      |\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_GPURuntimeNA = "| GPU Runtime:      N/A                              |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_GPURuntime   = "| GPU Runtime:      %d.%d                              |\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADriver    = "| CUDA Driver:      %d.%d                              |\n";

/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADeviceNA  =
    "| CUDA code arch:   N/A                              |\n"
    "|                                                    |\n"
    "| CUDA Device Idx:  N/A                              |\n"
    "| CUDA Device Name: N/A                              |\n"
    "| CUDA Capability:  N/A                              |\n";


/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_CUDACodeArch   = "| CUDA code arch:   %1.1f                              |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADevice     = "| CUDA Device Idx:  %d                                |\n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADeviceName = "| CUDA Device Name: %s %.*s| \n";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADeviceNamePadding =  "                                        ";
/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADCapability = "| CUDA Capability:  %d.%d                              |\n";


/// KSpaceFirstOrder3DSolver module log message
TOutputMessage TKSpaceFirstOrder3DSolver_OUT_FMT_Licence =
    "+----------------------------------------------------+\n"
    "| Copyright (C) 2016 Jiri Jaros, Bradley Treeby and  |\n"
    "|                    Beau Johnston                   |\n"
    "| http://www.k-wave.org                              |\n"
    "+----------------------------------------------------+\n";


//------------ TCommandlineParamerers module Print code version --------------//
/// ComandlineParamerers module log message
TOutputMessage TCommandlineParamereres_OUT_FMT_InputFile  = "Input file:  ";

/// ComandlineParamerers module log message
TOutputMessage TCommandlineParamereres_OUT_FMT_OutputFile = "Output file: ";

/// ComandlineParamerers module log message
TOutputMessage TCommandlineParamereres_OUT_FMT_CheckpointFile = "Check file:  ";

/// ComandlineParamerers module log message
TOutputMessage TCommandlineParamereres_OUT_FMT_CheckpointInterval     = "Checkpoint interval:         %8lus\n";

/// ComandlineParamerers module log message
TOutputMessage TCommandlineParamereres_OUT_FMT_PrintProgressInterval  = "Print progress interval:     %8lu%%\n";
/// ComandlineParamerers module log message
TOutputMessage TCommandlineParamereres_OUT_FMT_CompressionLevel       = "Compression level:            %8lu\n";
/// ComandlineParamerers module log message
TOutputMessage TCommandlineParamereres_OUT_FMT_BenchmarkTimeStepCount = "Benchmark time steps:         %8lu\n";
/// ComandlineParamerers module log message
TOutputMessage TCommandlineParamereres_OUT_FMT_QuantitySampling       = "Sampling flag: %.*s %s \n";
/// ComandlineParamerers module log message
TOutputMessage TCommandlineParamereres_OUT_FMT_CollectionBeginsAt     = "Collection begins at timestep:%8lu\n";
/// ComandlineParamerers module log message
TOutputMessage TCommandlineParamereres_OUT_FMT_CopySensorMaskYes      = "Copy sensor mask to output file:   Yes\n";

TOutputMessage OUT_FMT_NewLine = "\n";

#endif /* OUTPUTMESSAGES_H */

