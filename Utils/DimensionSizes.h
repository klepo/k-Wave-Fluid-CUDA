/**
 * @file      DimensionSizes.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file containing the structure with 3D dimension sizes.
 *
 * @version   kspaceFirstOrder3D 3.5
 *
 * @date      09 August    2011, 12:34 (created) \n
 *            16 August    2017, 15:45 (revised)
 *
 * @copyright Copyright (C) 2017 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the C++ extension of the k-Wave Toolbox [k-Wave Toolbox](http://www.k-wave.org).
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify it under the terms
 * of the GNU Lesser General Public License as published by the Free Software  *Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 */


#ifndef DIMENSION_SIZES_H
#define DIMENSION_SIZES_H

#include <cstdlib>


#ifdef __AVX2__
/**
 * @var kDataAlignment
 * @brief memory alignment for AVX2 (32B)
 */
constexpr int kDataAlignment  = 32;
#elif __AVX__
/**
 * @var kDataAlignment
 * @brief memory alignment for AVX (32B)
 */
constexpr int kDataAlignment  = 32;
#else

/**
 * @var kDataAlignment
 * @brief memory alignment for SSE, SSE2, SSE3, SSE4 (16B)
 */
constexpr int kDataAlignment  = 16;
#endif

/**
 * @struct  DimensionSizes
 * @brief   Structure with 4D dimension sizes (3 in space and 1 in time).
 * @details Structure with 4D dimension sizes (3 in space and 1 in time).
 * The structure can be used for 3D (the time is then set to 1). \n
 * The structure contains only POD, so no C++ stuff is necessary.
 */
struct DimensionSizes
{
  /// Default constructor.
  DimensionSizes() : nx(0), ny(0), nz(0), nt(0) {};

  /**
   * @brief   Constructor.
   * @details Constructor.
   * @param [in] x, y, z, t - Three spatial dimensions and time.
   */
  DimensionSizes(size_t x, size_t y, size_t z, size_t t = 0)
    : nx(x), ny(y), nz(z), nt(t)
  {};

  /**
   * @brief Get the number of elements, in 3D only spatial domain, in 4D with time.
   * @details Get element count, in 3D only spatial domain, in 4D with time.
   * @return the number of elements the domain holds.
   */
  inline size_t nElements() const
  {
    return (is3D()) ? nx * ny * nz : nx * ny * nz * nt;
  };

  /// Does the object include spatial dimensions only?
  inline bool is3D() const
  {
    return (nt == 0);
  };

  /// Does the object include spatial and temporal dimensions?
  inline bool is4D() const
  {
    return (nt > 0);
  };

  /**
   * @brief Operator ==
   * @param [in] other  - The second operand to compare with.
   * @return true if the dimension sizes are equal.
   */
  inline bool operator==(const DimensionSizes& other) const
  {
    return ((nx == other.nx) && (ny == other.ny) && (nz == other.nz) && (nt == other.nt));
  };

  /**
   * @brief Operator !=
   * @param [in] other    - the second operand to compare with.
   * @return true if !=
   */
  inline bool operator!=(const DimensionSizes& other) const
  {
    return ((nx != other.nx) || (ny != other.ny) || (nz != other.nz) || (nt != other.nt));
  };

  /**
   * @brief Operator -
   *
   * Get the size of the cube defined by two corners.
   *
   * @param [in] op1 - Usually bottom right corner.
   * @param [in] op2 - Usually top left corner.
   * @return the size of the inner cuboid
   */
  inline friend DimensionSizes operator-(const DimensionSizes& op1,
                                         const DimensionSizes& op2)
  {
    // +1 because of planes (10.10.1 - 60.40.1)
    if (op1.is3D() && op2.is3D())
    {
      return DimensionSizes(op1.nx - op2.nx + 1, op1.ny - op2.ny + 1, op1.nz - op2.nz + 1);
    }
    else
    {
      return DimensionSizes(op1.nx - op2.nx + 1, op1.ny - op2.ny + 1,
                            op1.nz - op2.nz + 1, op1.nt - op2.nt + 1);
    }
  };

  /// Number of elements in the x direction.
  size_t nx;
  /// Number of elements in the y direction.
  size_t ny;
  /// Number of elements in the z direction.
  size_t nz;
  /// Number of time steps (for time series datasets).
  size_t nt;
}; // end of TDimensionSizes
//----------------------------------------------------------------------------------------------------------------------

#endif	/* #define	DIMENSION_SIZES_H */
