# pynest/CMakeLists.txt
#
# This file is part of NEST GPU.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST GPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST GPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.

include( UseCython )
set_source_files_properties(
    nestgpukernel_api.pyx
    PROPERTIES CYTHON_IS_CXX TRUE )

cython_add_module( nestgpukernel_api nestgpukernel_api.pyx )

target_link_libraries( nestgpukernel_api
    nestgpukernel
  )

set_source_files_properties(
    llapi_helpers.pyx
    PROPERTIES CYTHON_IS_CXX TRUE )

cython_add_module( llapi_helpers llapi_helpers.pyx )

message(STATUS "PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
target_include_directories( nestgpukernel_api PRIVATE
  ${PROJECT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}/src
  ${Python_INCLUDE_DIRS}
  )

install( TARGETS nestgpukernel_api llapi_helpers
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${PYEXECDIR} )
install( FILES ${PROJECT_SOURCE_DIR}/pythonlib/nestgpu.py
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${PYEXECDIR} )
