#!/usr/bin/env python

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

from __future__ import print_function, division
import sys
import re
from textwrap import dedent

escape_re = re.compile(r'[\\"]')

def escape(s):
    return escape_re.sub(r'\\\g<0>', s)

def main(argv):
    if len(argv) < 2:
        print("Usage: {0} <input.cl>... <output.cpp>".format(sys.argv[0]), file = sys.stderr)
        return 2

    with open(sys.argv[-1], 'w') as outf:
        print(dedent('''
            #include <map>
            #include <string>

            static std::map<std::string, std::string> g_sources;

            namespace CLH
            {
            namespace detail
            {

            const std::map<std::string, std::string> &getSourceMap() { return g_sources; }

            }} // namespace CLH::detail

            namespace
            {

            struct Init
            {
                Init();
            };

            Init::Init()
            {'''), file = outf)
        for i in sys.argv[1:-1]:
            label = re.sub(r'\\', '/', i) # Fix up Windows separators
            label = re.sub(r'\.\./', '', label)
            with open(i, 'r') as inf:
                lines = inf.readlines()
                lines = [escape(line.rstrip('\n')) for line in lines]
                print('    g_sources["{0}"] ='.format(escape(label)), file = outf)
                for line in lines:
                    print('        "{0}\\n"'.format(line), file = outf)
                print('        ;', file = outf)
        print(dedent('''
            }

            static Init init;

            } // namespace'''), file = outf)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
