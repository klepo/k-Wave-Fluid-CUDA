/**
 * @file        Logger.h
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
 *              21 July     2016, 11:10 (revised)
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

    /// Log desired activity.
    static void Log(const TLogLevel queryLevel,
                    const char*     format,
                    ...);

    /// Log an error.
    static void Error(const char* format,
                      ...);

    /// Log an error and terminate the execution
    static void ErrorAndTerminate(const char* Format,
                                  ...);

    /// Flush output messages.
    static void Flush(const TLogLevel queryLevel);

    /// Wrap the line based on logger conventions
    static std::string WordWrapString(const std::string& inputString,
                                      const std::string& delimiters,
                                      const int          indentation = 0,
                                      const int          lineSize    = 65);

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

#endif /* TLOGGER_H */

