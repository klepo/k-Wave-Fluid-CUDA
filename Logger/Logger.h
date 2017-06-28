/**
 * @file        Logger.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing a class responsible for printing out
 *              info and error messages (stdout, and stderr).
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        19 April    2016, 12:52 (created) \n
 *              10 August   2016, 16:44 (revised)
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


#ifndef TLOGGER_H
#define TLOGGER_H

#include <memory>
#include <iostream>
#include <cuda_runtime.h>

#include <Logger/OutputMessages.h>
#include <Logger/ErrorMessages.h>

/**
 * @class TLogger
 * @brief Static class implementing the user interface by info messages.
 * @details StaticClass used for printing out info and error message based on the
 *          verbose level. This is a static class.
 */
class TLogger
{
  public:

   /**
    * @enum  TLogLevel
    * @brief Log level of the message.
    * @details A enum to specify at which log level the message should be displayed, or the level
    * set.
    */
    enum TLogLevel
    {
      /// Basic (default) level of verbosity
      BASIC    = 0,
      /// Advanced level of verbosity
      ADVANCED = 1,
      /// Full level of verbosity
      FULL     = 2,
    };


    /// Set the log level.
    static void SetLevel(const TLogLevel actualLogLevel);
    /// Get the log level.
    static TLogLevel GetLevel() {return logLevel;};

    /**
     * @brief   Log desired activity for a given log level, version with string format.
     * @details Log desired activity and format it using format message.
     *
     * @param [in] queryLevel - What level to use
     * @param [in] format     - Format string
     * @param [in] args       - Arguments, std::string is not accepted
     */
    template<typename... Args>
    static void Log(const TLogLevel queryLevel,
                    const std::string& format,
                    Args ... args)
    {
      if (queryLevel <= TLogger::logLevel)
      {
        std::cout << FormatMessage(format, args ...);
      }
     }// end of Log


    /// Log desired activity for a given log level.
    static void Log(const TLogLevel    queryLevel,
                    const std::string& message);

    /// Log an error.
    static void Error(const std::string& errorMessage);

    /// Log an error and terminate the execution
    static void ErrorAndTerminate(const std::string& errorMessage);

    /// Flush output messages.
    static void Flush(const TLogLevel queryLevel);

    /// Wrap the line based on logger conventions
    static std::string WordWrapString(const std::string& inputString,
                                      const std::string& delimiters,
                                      const int          indentation = 0,
                                      const int          lineSize    = 65);

    /**
     * @brief   C++-11 replacement for sprintf that works with std::string instead of char *
     * @details The routine was proposed at
     *          http://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
     *          and should work with both Linux and VS 2013.
     *          However it still does not support string in formated arguments
     * @param [in] format - Format string
     * @param [in] args   - Arguments, std::string is not accepted
     * @return formated string
     */
    template<typename ... Args>
    static std::string FormatMessage(const std::string& format, Args ... args)
    {
	  	/// when the size is 0, the routine returns the size of the formated string
      size_t size = snprintf(nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'

      std::unique_ptr<char[]> buf(new char[size]);
      snprintf(buf.get(), size, format.c_str(), args ... );
      return std::string(buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
    }
  private:
    /// Default constructor is not allowed, static class
    TLogger();
    /// Copy constructor is not allowed, static class
    TLogger(const TLogger& orig);
    /// Destructor is not allowed, static class
    ~TLogger();

    /// Log level of the logger
    static TLogLevel logLevel;

 private:
  /// Extract a word (string between two delimiters)
  static std::string GetWord(std::istringstream& textStream,
                             const std::string&  delimiters);

}; // TLogger



//------------------------------------------------------------------------------------------------//
//------------------------------------ Routines --------------------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 *@brief  Checks CUDA errors, create an error message and throw an exception.
 *@details Checks CUDA errors, create an error message and throw an exception. The template
 * parameter should be set to true for the whole code when debugging  kernel related errors.
 * Setting it to true for production run will cause IO sampling and storing not to be overlapped.
 *
 * @param [in] errorCode   - Error produced by a cuda routine
 * @param [in] routineName - Function where the error happened
 * @param [in] fileName    - File where the error happened
 * @param [in] lineNumber  - Line where the error happened
 */
template <bool forceSynchronisation = false>
inline void CheckErrors(const cudaError_t errorCode,
                        const char*       routineName,
                        const char*       fileName,
                        const int         lineNumber)
{
  if (forceSynchronisation)
  {
    cudaDeviceSynchronize();
  }

  if (errorCode != cudaSuccess)
  {
    // Throw exception
     throw std::runtime_error(TLogger::FormatMessage(ERR_FMT_GPU_ERROR,
                                                     cudaGetErrorString(errorCode),
                                                     routineName,
                                                     fileName,
                                                     lineNumber));
  }
}// end of cudaCheckErrors
//--------------------------------------------------------------------------------------------------

/**
 * @brief Macro checking cuda errors and printing the file name and line. Inspired by CUDA common
 *        checking routines.
 */
#define checkCudaErrors(val) CheckErrors ( (val), #val, __FILE__, __LINE__ )


#endif /* TLOGGER_H */

