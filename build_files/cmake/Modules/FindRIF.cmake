# - Find Radeon Image Filters (RIF) library
# Find the native RIF includes and library
# This module defines
#  RIF_INCLUDE_DIRS, where to find RadeonImageFilters.h, Set when
#                         RIF_INCLUDE_DIR is found.
#  RIF_LIBRARIES, libraries to link against to use RIF.
#  RIF_ROOT_DIR, The base directory to search for RIF.
#                     This can also be an environment variable.
#  RIF_FOUND, If false, do not try to use RIF.

#=============================================================================
# Copyright 2019 Blender Foundation.
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================

# If RIF_ROOT_DIR was defined in the environment, use it.
IF(NOT RIF_ROOT_DIR AND NOT $ENV{RIF_ROOT_DIR} STREQUAL "")
  SET(RIF_ROOT_DIR $ENV{RIF_ROOT_DIR})
ENDIF()

SET(_rif_SEARCH_DIRS
  ${RIF_ROOT_DIR}
)

FIND_PATH(RIF_INCLUDE_DIR
  NAMES
    RadeonImageFilters.h
  HINTS
    ${_rif_SEARCH_DIRS}
  PATH_SUFFIXES
    include
)

FIND_LIBRARY(RIF_LIBRARY
  NAMES
    RadeonImageFilters
  HINTS
    ${_rif_SEARCH_DIRS}
  PATH_SUFFIXES
    lib64 lib
  )

# handle the QUIETLY and REQUIRED arguments and set RIF_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(RIF DEFAULT_MSG
    RIF_LIBRARY RIF_INCLUDE_DIR)

IF(RIF_FOUND)
  SET(RIF_INCLUDE_DIRS ${RIF_INCLUDE_DIR})
  SET(RIF_LIBRARIES ${RIF_LIBRARY})
ENDIF(RIF_FOUND)

MARK_AS_ADVANCED(
  RIF_INCLUDE_DIR
  RIF_LIBRARY
)

UNSET(_rif_SEARCH_DIRS)
