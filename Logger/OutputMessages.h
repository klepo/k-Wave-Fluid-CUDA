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

#ifndef OUTPUT_MESSAGES_H
#define OUTPUT_MESSAGES_H

/**
 * @typedef TOutputMessage
 * @brief Datatype for output messages
 */
typedef const char * const TOutputMessage;


TOutputMessage MAIN_OUT_FMT_SmallSeparator = "--------------------------------------\n";

#endif /* OUTPUTMESSAGES_H */

