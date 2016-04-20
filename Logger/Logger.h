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
 * @date        19 April    2016, 12:52 (created) \n
 *              19 April    2016, 12:52 (revised)
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

#ifndef TLOGGER_H
#define TLOGGER_H



/**
 * @class TLogger
 * @brief Static class implementing user interface by info messages
 *
 * @details StaticClass used for printing out info and error message based on the
 *          verbose level. This is a static class.
 *
 */
class TLogger
{
  public:

   /**
    * @enum  TLogLEvel
    * @brief Current log level
    */
    enum TLogLevel
    {
      // Basic (default) level of verbosity
      Basic    = 0,
      // Advanced level of verbosity
      Advanced = 1,
      // Full level of verbosity
      Full     = 2,
    };


    /// Change the log level
    static void SetLevel(const TLogLevel ActualLogLevel);

    /// Log desired activity
    static void Log(const TLogLevel QueryLevel,
                    const char *    Format,
                    ...);

  private:
    /// Default constructor is not allowed, static class
    TLogger();
    /// Copy constructor is not allowed, static class
    TLogger(const TLogger& orig);
    /// Destructor is not allowed, static class
    ~TLogger();

    static TLogLevel LogLevel;

}; // TLogger

#endif /* TLOGGER_H */

