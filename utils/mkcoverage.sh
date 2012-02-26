#!/bin/bash
set -e
rm -rf build/coverage
mkdir -p build/coverage
lcov --directory build --zerocounters
build/testmain --test=commit
lcov --capture --directory build --base $PWD/build -o build/coverage_full.info
lcov --remove build/coverage_full.info \
    '/usr/include/*' \
    '*/test.cpp' \
    '*/CL/*.h*' -o build/coverage.info
genhtml --prefix="$(dirname $PWD)" --demangle-cpp -o build/coverage build/coverage.info
