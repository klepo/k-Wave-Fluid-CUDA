/**
 * @file        ErrorMessagesWindows.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing windows specific error messages.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        13 July     2016, 12:27 (created) \n
 *              29 July     2016, 16:43 (revised)
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


#ifndef ERROR_MESSAGES_WINDOWS_H
#define	ERROR_MESSAGES_WINDOWS_H

/**
 * @typedef TErrorMessage
 * @brief   Datatype for error messages.
 * @details Datatype for error messages.
 */
typedef const std::string TErrorMessage;

/// Error message header
TErrorMessage ERR_FMT_HEAD =
        "+---------------------------------------------------------------+\n"
        "|            !!! K-Wave experienced a fatal error !!!           |\n"
        "+---------------------------------------------------------------+\n";

/// Error message tailer
TErrorMessage ERR_FMT_TAIL =
        "+---------------------------------------------------------------+\n"
        "|                      Execution terminated                     |\n"
        "+---------------------------------------------------------------+\n";


#endif	/* ERROR_MESSAGES_WINDOWS_H */




