#!/usr/bin/env python
from __future__ import print_function, division
import sys
import re
from textwrap import dedent

escape_re = re.compile(r'[\\"]')

def escape(s):
    return escape_re.sub(r'\\\g<0>', s)

def main(argv):
    if len(argv) < 2:
        print("Usage: {} <input.cl>... <output.cpp>".format(sys.argv[0]), file = sys.stderr)
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
            with open(i, 'r') as inf:
                lines = inf.readlines()
                lines = [escape(line.rstrip('\n')) for line in lines]
                print('    g_sources["{}"] ='.format(escape(i)), file = outf)
                for line in lines:
                    print('        "{}"'.format(line), file = outf)
                print('        ;', file = outf)
        print(dedent('''
            }

            static Init init;

            } // namespace'''), file = outf)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
