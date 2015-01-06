/**
 * @file        CommandLineParameters.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the command line parameters.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        29 August   2012, 11:25 (created) \n
 *              17 December 2014, 12:40 (revised)
 *
 * @section Params Command Line Parameters
 * The CUDA/C++ code requires two mandatory parameters and accepts a few optional
 * parameters and flags.
 *
 * * The mandatory parameters \c -i and \c -o specify the input and output file. The
 * file names respect the path conventions for particular operating system.
 * If any of the files is not specified, cannot be found or created, an error
 * message is shown.
 *
 * The \c -t parameter sets the number of threads used, which defaults the system maximum.
 * On CPUs with Intel Hyper-Threading (HT), the performance is sometimes better if HT
 * is disabled in the BIOS settings. If HT is switched on, the default will be to
 * spawn twice as many threads as there are physical processor cores, which may but again
 * may not slow down the code execution. Therefore, if the HT is on, try specifying the
 * number of threads manually for best performance (e.g. 4 for Intel i7). We
 * recommend experimenting with this parameter to find the best configuration.
 * Note, if there are other tasks being executed on the system, it might be useful
 * to further limit the number of threads to prevent system overload.
 *
 * The \c -r parameter specifies how often information about the simulation progress
 * is printed out to the command line. By default, the C++ code prints out the
 * progress of the simulation, the elapsed time, and the estimated time of
 * completion in intervals corresponding to 5% of the total number of times steps.
 *
 * The \c -c parameter specifies the compression level used by the ZIP library to
 * reduce the size of the output file. The actual compression rate is highly dependent
 * on the shape of the sensor mask and the range of stored quantities. In general,
 * the output data is very hard to compress, and using higher compression levels
 * can greatly increase the time to save data while not having a large impact on
 * the final file size. That's why we decided to disable compression in default settings.
 *
 * The \c <tt>--benchmark</tt> parameter enables the total length of simulation (i.e.,
 * the number of time steps) to be overridden by setting a new number of time
 * steps to simulate. This is particularly useful for performance evaluation and
 * benchmarking. As the code performance is relatively stable, 50-100 time steps is
 * usually enough to predict the simulation duration. This parameter can also be
 * used to quickly find the ideal number of CPU threads to use.
 *
 *
 *
 * For jobs that are expected to run for a very long time, it may be useful to
 * checkpoint and restart the execution. One motivation is the wall clock limit
 * per task on clusters where jobs must fit within a given time span
 * (e.g. 24 hours). The second motivation is a level of fault-tolerance, where
 * you can back up the state of the simulation after a predefined period.
 * To enable checkpoint-restart, the user is asked to specify a file to store the
 * actual state of the simulation by  <tt>--checkpoint_file</tt> and the
 * period in seconds after which the simulation will be interrupted by <tt>--checkpoint_interval</tt>.
 * When running on a cluster, please allocate enough time for the checkpoint procedure
 * that can take a non-negligible amount of time (7 matrices have to be stored in
 * the checkpoint file and all aggregated quantities are flushed into the output file).
 *
 * When controlling a multi-leg simulation by a script loop, the parameters of the code
 * remains the same in all legs. The first leg of the simulation creates a checkpoint
 * file while the last one deletes it. If the checkpoint file is not found the
 * simulation starts from the beginning. In order to find out how many steps have been
 * finished, please open the output file and read the variable <tt>t_index</tt>.
 *
 *
 * The \c -h and \c --help parameters print all the parameters of the C++ code.
 * The <tt> --version </tt>parameter reports detail information about the code useful for
 * debugging and bug reports. It prints out the internal version, the build date and time, the
 * git hash allowing us to track the version of the source code, the operating system,
 * the compiler name and version and the instruction set used.
 *
 *
 * The remaining flags specify the output quantities to be recorded during the
 * simulation and stored on disk analogous to  the sensor.record input. If
 * the \c -p or \c --p\_raw flags are set (these are equivalent), a time series of
 * the acoustic pressure at the grid points specified by the sensor mask is
 * recorded. If the \c --p_rms, \c --p_max, \c --p_min flags are set,
 * the root mean square and/or maximum and/or minimum values of the pressure at
 * the grid points specified by the sensor mask are recorded. If the
 * \c --p_final flag is set, the values for the entire acoustic pressure field
 * in the final time step of the simulation is stored (this will always include
 * the PML, regardless of  the setting for <tt> `PMLInside'</tt>).
 * The flags \c --p_max_all and \c --p_min_all allow to calculate the maximum and
 * minimum values over the entire acoustic pressure field, regardless on the shape
 * of the sensor mask.
 * Flags to record the acoustic particle velocity are defined in an analogous fashion.
 * For proper calculation of acoustic intensity, the particle velocity has to be
 * shifted onto the same grid as the acoustic pressure. This can be done by setting
 * \c --u_non_staggered_raw flag, that first shifts the particle velocity and then
 * samples the grid points specified by the sensor mask. Since the shift operation
 * requires additional FFTs, the impact on the simulation time may be significant.
 *
 * Any combination of <tt>p</tt> and <tt>u</tt> flags is admissible. If no output flag is set,
 * a time-series for the acoustic pressure is recorded. If it is not necessary
 * to collect the output quantities over the entire simulation, the starting time
 * step when the collection begins can be specified using the -s parameter.
 * Note, the index for the first time step is 1 (this follows the MATLAB indexing convention).
 *
 * The \c --copy_sensor_mask will copy the sensor from the input file to the output
 * one at the end of the simulation. This helps in post-processing and visualisation of
 * the outputs.
 *
\verbatim
---------------------------------- Usage ---------------------------------
Mandatory parameters:
  -i <input_file_name>            : HDF5 input file
  -o <output_file_name>           : HDF5 output file

Optional parameters:
  -t <num_threads>                : Number of CPU threads
                                      (default = all )
  -g <device_number>              : GPU device to run on
                                      (default = device with most  memory)

  --BlockSize1D <num_threads>     : Run 1D CUDA kernels with block size
                                      (default is 256 threads per block)
  --BlockSize3D <num_threads_x>   : Run 3D CUDA kernels with block size
                <num_threads_y>        in x, y and z dimensions");
                <num_threads_z>       (default is x=256 y=1 z=1 threads
                                       per block)");

  -r <interval_in_%>              : Progress print interval
                                      (default = 5%)
  -c <comp_level>                 : Output file compression level <0,9>
                                      (default = 0)
  --benchmark <steps>             : Run a specified number of time steps

  --checkpoint_file <file_name>   : HDF5 checkpoint file
  --checkpoint_interval <seconds> : Stop after a given number of seconds and
                                      store the actual state

  -h                              : Print help
  --help                          : Print help
  --version                       : Print version

Output flags:
  -p                              : Store acoustic pressure
                                      (default if nothing else is on)
                                      (the same as --p_raw)
  --p_raw                         : Store raw time series of p (default)
  --p_rms                         : Store rms of p
  --p_max                         : Store max of p
  --p_min                         : Store min of p
  --p_max_all                     : Store max of p (whole domain)
  --p_min_all                     : Store min of p (whole domain)
  --p_final                       : Store final pressure field

  -u                              : Store ux, uy, uz
                                      (the same as --u_raw)
  --u_raw                         : Store raw time series of ux, uy, uz
  --u_non_staggered_raw           : Store non-staggered raw time series of
                                      ux, uy, uz
  --u_rms                         : Store rms of ux, uy, uz
  --u_max                         : Store max of ux, uy, uz
  --u_min                         : Store min of ux, uy, uz
  --u_max_all                     : Store max of ux, uy, uz (whole domain)
  --u_min_all                     : Store min of ux, uy, uz (whole domain)
  --u_final                       : Store final acoustic velocity

  --copy_sensor_mask              : Copy sensor mask to the output file

  -s <timestep>                   : Time step when data collection begins
                                      (default = 1)
--------------------------------------------------------------------------
\endverbatim
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

#ifndef TCOMMANDLINESPARAMETERS_H
#define TCOMMANDLINESPARAMETERS_H

#include <cstdlib>
#include <string>

using std::string;

/**
 * @class TCommandLineParameters
 * @brief The class to parse and store command line parameters.
 * @details The class to parse and store command line parameters.
 */
class TCommandLineParameters
{
  public:

    /// Constructor
    TCommandLineParameters();
    /// Destructor
    virtual ~TCommandLineParameters() {};

    /// Get input file name
    string GetInputFileName()           const {return InputFileName;};
    /// Get output file name
    string GetOutputFileName()          const {return OutputFileName;};
    /// Get Checkpoint file name
    std::string GetCheckpointFileName() const {return CheckpointFileName;};


    /// Get GPU device ID
    int GetGPUDeviceIdx()               const {return GPUDeviceIdx;};
    /// Get size of 1D block
    int GetBlockSize1D()                const {return BlockSize1D;};
    /// Get X size of 3D block
    int GetBlockSize3DX()               const {return BlockSize3DX;};
    /// Get Y size of 3D block
    int GetBlockSize3DY()               const {return BlockSize3DY;};
    /// Get Z size of 3D block
    int GetBlockSize3DZ()               const {return BlockSize3DZ;};

    /// Is --benchmark flag set?
    bool IsBenchmarkFlag()              const {return BenchmarkFlag;};
    /// Is --version flag set?
    bool IsVersion()                    const {return PrintVersion; };
    /// Get benchmark time step count
    size_t GetBenchmarkTimeStepsCount() const {return BenchmarkTimeStepsCount;};

    /// Get compression level
    size_t GetCompressionLevel()        const {return CompressionLevel;};
    /// Get number of threads
    size_t GetNumberOfThreads()         const {return NumberOfThreads;};
    /// Get verbose interval
    size_t GetVerboseInterval()         const {return VerboseInterval;};
    /// Get start time index when sensor data collection begins
    size_t GetStartTimeIndex()          const {return StartTimeStep;};

    /// Is checkpoint enabled?
    bool IsCheckpointEnabled()          const {return (CheckpointInterval > 0); };
    /// Get checkpoint interval
    size_t GetCheckpointInterval()      const {return CheckpointInterval; };

    /// Is --p_raw set?
    bool IsStore_p_raw()                const {return Store_p_raw;};
    /// Is --p_rms set?
    bool IsStore_p_rms()                const {return Store_p_rms;};
    /// Is --p_max set?
    bool IsStore_p_max()                const {return Store_p_max;};
    /// Is --p_min set?
    bool IsStore_p_min()                const {return Store_p_min;};
    /// Is --p_max_all set?
    bool IsStore_p_max_all()            const {return Store_p_max_all;};
    /// Is --p_min_all set?
    bool IsStore_p_min_all()            const {return Store_p_min_all;};
    /// Is --p_final set?
    bool IsStore_p_final()              const {return Store_p_final;};

    /// Is --u_raw set?
    bool IsStore_u_raw()                const {return Store_u_raw;};
    /// Is --u_non_staggered_raw set?
    bool IsStore_u_non_staggered_raw()  const {return Store_u_non_staggered_raw;};
    /// Is --u_rms set?
    bool IsStore_u_rms()                const {return Store_u_rms;};
    /// Is --u_max set?
    bool IsStore_u_max()                const {return Store_u_max;};
    /// Is --u_min set?
    bool IsStore_u_min()                const {return Store_u_min;};
    /// Is --u_max_all set?
    bool IsStore_u_max_all()            const {return Store_u_max_all;};
    /// Is --u_min set?
    bool IsStore_u_min_all()            const {return Store_u_min_all;};
    /// Is --u_final set?
    bool IsStore_u_final()              const {return Store_u_final;};

    /// is --copy_mask set
    bool IsCopySensorMask()             const {return CopySensorMask;};

    /// Print usage and exit
    void PrintUsageAndExit();
    /// Print setup
    void PrintSetup();
    /// Parse command line
    void ParseCommandLine(int argc, char** argv);

  protected:
    /// Copy constructor not allowed for public.
    TCommandLineParameters(const TCommandLineParameters& src);

    /// operator = not allowed for public.
    TCommandLineParameters& operator = (const TCommandLineParameters& src);

  private:
    /// Input file name.
    string InputFileName;
    /// Output file name.
    string OutputFileName;
    /// Checkpoint file name.
    string CheckpointFileName;

    /// NumberOfThreads value.
    size_t NumberOfThreads;


    /// id of selected GPU devices.
    int GPUDeviceIdx;
    /// size of 1D block.
    int BlockSize1D;
    /// x size of 3D block.
    int BlockSize3DX;
    /// y size of 3D block.
    int BlockSize3DY;
    /// z size of 3D block.
    int BlockSize3DZ;

    /// VerboseInterval value.
    size_t VerboseInterval;
    // CompressionLevel value.
    size_t CompressionLevel;

    /// BenchmarkFlag value.
    bool BenchmarkFlag;
    /// BenchmarkTimeStepsCount value.
    size_t BenchmarkTimeStepsCount;
    /// Checkpoint interval in seconds
    size_t CheckpointInterval;

    /// PrintVersion value.
    bool PrintVersion;

    /// Store_p_raw value.
    bool Store_p_raw;
    /// Store_p_rms value.
    bool Store_p_rms;
    /// Store_p_max value.
    bool Store_p_max;
    /// Store_p_min value.
    bool Store_p_min;
    /// Store_p_max_all value.
    bool Store_p_max_all;
    /// Store_p_min_all value.
    bool Store_p_min_all;
    /// Store_p_final value.
    bool Store_p_final;

    /// Store_u_raw value.
    bool Store_u_raw;
    /// Store_u_non_staggered_raw value.
    bool Store_u_non_staggered_raw;
    /// Store_u_rms value.
    bool Store_u_rms;
    /// Store_u_max value.
    bool Store_u_max;
    /// Store_u_min value.
    bool Store_u_min;
    /// Store_u_max_all value.
    bool Store_u_max_all;
    /// Store_u_min_all value.
    bool Store_u_min_all;
    /// Store_u_final value.
    bool Store_u_final;

    /// Copy sensor mask to the output file.
    bool CopySensorMask;
    /// StartTimeStep value.
    size_t StartTimeStep;

    /// Default compression level.
    static const size_t DefaultCompressionLevel = 0;
    /// Default verbose interval.
    static const size_t DefaultVerboseInterval  = 5;
};// end of class TCommandLineParameters

#endif /* TCOMMANDLINESPARAMETERS_H */

