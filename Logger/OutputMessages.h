/**
 * @file      OutputMessages.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file including output messages based on the operating system.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      19 April     2016, 12:52 (created) \n
 *            06 March     2019, 13:15 (revised)
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

#ifndef OUTPUT_MESSAGES_H
#define OUTPUT_MESSAGES_H

#ifdef __linux__
  #include <Logger/OutputMessagesLinux.h>
#endif

// Windows build
#ifdef _WIN64
  #include <Logger/OutputMessagesWindows.h>
#endif

#endif /* OUTPUT_MESSAGES_H */

