#!/bin/bash

# mlsgpu: surface reconstruction from point clouds
# Copyright (C) 2013  University of Cape Town
#
# This file is part of mlsgpu.
#
# mlsgpu is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

set -e
rm -rf build/coverage
mkdir -p build/coverage
lcov --directory build --zerocounters
build/testmain --test=commit
mpirun -n 2 build/testmpi --test=commit
lcov --capture --directory build --base $PWD/build -o build/coverage_full.info
lcov --remove build/coverage_full.info \
    '/usr/include/*' \
    '/usr/local/include/*' \
    '*/test.cpp' \
    '*/CL/*.h*' -o build/coverage.info
genhtml --prefix="$(dirname $PWD)" --demangle-cpp -o build/coverage build/coverage.info
