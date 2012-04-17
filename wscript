import os.path

import waflib.Errors

APPNAME = 'mlsgpu'
VERSION = '0.99'
out = 'build'

variants = {
    'debug':
    {
        'debuginfo': True,
        'symbols': True,
        'optimize': False,
        'assertions': True,
        'expensive_assertions': False,
        'unit_tests': True,
        'coverage': False,
    },
    'coverage':
    {
        'debuginfo': True,
        'symbols': True,
        'optimize': False,
        'assertions': True,
        'expensive_assertions': True,
        'unit_tests': True,
        'coverage': True,
    },
    'optimized':
    {
        'debuginfo': True,
        'symbols': True,
        'optimize': True,
        'assertions': True,
        'expensive_assertions': False,
        'unit_tests': True,
        'coverage': False,
    },
    'checked':
    {
        'debuginfo': True,
        'symbols': True,
        'optimize': False,
        'assertions': True,
        'expensive_assertions': True,
        'unit_tests': True,
        'coverage': False,
    },
    'release':
    {
        'debuginfo': False,
        'symbols': False,
        'optimize': True,
        'assertions': False,
        'expensive_assertions': False,
        'unit_tests': False,
        'coverage': False,
    }
}

libs = [
    'boost_program_options-mt',
    'boost_iostreams-mt',
    'boost_thread-mt',
    'boost_math_c99-mt', 'boost_math_c99f-mt'
]

def options(opt):
    opt.load('gnu_dirs')
    opt.load('compiler_cxx')
    opt.load('waf_unit_test')
    opt.load('provenance', tooldir = '../../waf-tools')
    opt.add_option('--variant', type = 'choice', dest = 'variant', default = 'release', action = 'store', help = 'build variant', choices = variants.keys())
    opt.add_option('--lto', dest = 'lto', default = False, action = 'store_true', help = 'use link-time optimization')
    opt.add_option('--cl-headers', action = 'store', default = None, help = 'Include path for OpenCL')
    opt.add_option('--no-tests', action = 'store_true', default = False, help = 'Do not run unit tests')

def configure_variant(conf):
    if conf.env['assertions']:
        conf.define('DEBUG', '1', quote = False)
    else:
        conf.define('NDEBUG', '1', quote = False)
        conf.define('BOOST_DISABLE_ASSERTS', '1', quote = False)
    if conf.env['unit_tests']:
        conf.define('UNIT_TESTS', 1, quote = False)

def configure_variant_gcc(conf):
    ccflags = ['-Wall', '-W', '-pthread']
    conf.env.append_value('LINKFLAGS', '-pthread')
    if conf.env['optimize']:
        ccflags.append('-O2')
    else:
        pass # ccflags.append('-fno-inline')
    if conf.env['debuginfo']:
        ccflags.append('-g')
    if conf.env['coverage']:
        ccflags.extend(['-O0', '-fno-inline', '-fno-inline-functions', '-fno-default-inline', '-fno-elide-constructors'])
        ccflags.append('--coverage')
        conf.env.append_value('LINKFLAGS', '--coverage')
    if not conf.env['symbols']:
        conf.env.append_value('LINKFLAGS', '-s')
    if conf.env['DEST_CPU'] == 'x86':
        # Avoids precision weirdness due to 80-bit 8087 registers
        ccflags.extend(['-mfpmath=sse', '-msse2'])
    lto = conf.env['lto']
    if lto:
        ccflags.extend(['-flto', '-B/usr/lib/gold-ld'])
    conf.env.append_value('CFLAGS', ccflags)
    conf.env.append_value('CXXFLAGS', ccflags)
    if lto:
        # -flto requires compilation flags to be provided at link time
        conf.env.append_value('LINKFLAGS', ccflags)

def configure(conf):
    conf.load('waf_unit_test')
    conf.load('gnu_dirs')
    conf.load('compiler_cxx')
    conf.load('provenance', tooldir = '../../waf-tools')

    for (key, value) in variants[conf.options.variant].items():
        conf.env[key] = value
    conf.env['lto'] = conf.options.lto
    configure_variant(conf)

    cgal_cxxflags = []
    if conf.env['CXX_NAME'] == 'gcc':
        configure_variant_gcc(conf)

    conf.define('PROVENANCE_VARIANT', conf.options.variant)

    if conf.env['unit_tests']:
        conf.check_cxx(
            features = ['cxx', 'cxxprogram'],
            header_name = 'cppunit/Test.h', lib = 'cppunit', uselib_store = 'CPPUNIT',
            msg = 'Checking for cppunit')
    if conf.options.cl_headers:
        conf.env.append_value('INCLUDES_OPENCL', [conf.options.cl_headers])
    else:
        conf.env.append_value('INCLUDES_OPENCL', [os.path.abspath('../../khronos_headers')])
    conf.env.append_value('LIB_OPENCL', ['OpenCL'])
    conf.check_cxx(
        features = ['cxx', 'cxxprogram'],
        header_name = 'CL/cl.hpp',
        use = 'OPENCL',
        msg = 'Checking for OpenCL')
    conf.check_cxx(
        features = ['cxx', 'cxxprogram'],
        header_name = 'clogs/clogs.h', lib = 'clogs',
        use = 'OPENCL',
        uselib_store = 'CLOGS',
        msg = 'Checking for clogs')
    conf.check_cxx(
        features = ['cxx', 'cxxprogram'],
        header_name = 'stxxl.h',
        lib = 'stxxl',
        uselib_store = 'STXXL',
        msg = 'Checking for STXXL')

    # Detect which timer implementation to use
    # We have to provide a fragment because with the default one the
    # compiler can (and does) eliminate the symbol.
    timer_test = '''
#ifndef _POSIX_C_SOURCE
# define _POSIX_C_SOURCE 200112L
#endif
#include <time.h>

int main() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_nsec;
}'''
    try:
        conf.check_cxx(
            # features = ['cxx', 'cxxprogram'],
            fragment = timer_test,
            function_name = 'clock_gettime',
            uselib_store = 'TIMER',
            msg = 'Checking for clock_gettime')
    except waflib.Errors.ConfigurationError:
        conf.check_cxx(
            features = ['cxx', 'cxxprogram'],
            function_name = 'clock_gettime', header_name = 'time.h', lib = 'rt',
            fragment = timer_test,
            uselib_store = 'TIMER',
            msg = 'Checking for clock_gettime in -lrt',
            mandatory = False)

    conf.check_cxx(
        features = ['cxx', 'cxxprogram'],
        function_name = 'QueryPerformanceCounter', header_name = 'windows.h',
        uselib_store = 'TIMER',
        msg = 'Checking for QueryPerformanceCounter',
        mandatory = False)

    for l in libs:
        conf.check_cxx(lib = l)
    conf.find_program('xsltproc')

    conf.write_config_header('config.h')
    conf.env.append_value('DEFINES', 'HAVE_CONFIG_H=1')
    conf.env.append_value('INCLUDES', '.')

def empty(s):
    '''Determine whether s contains non-whitespace characters'''
    import re
    return re.match('^\s*$', s)

def print_unit_tests(bld):
    from waflib.Tools import waf_unit_test
    from waflib import Logs
    waf_unit_test.summary(bld)
    for (f, code, out, err) in getattr(bld, 'utest_results', []):
        color = 'GREEN'
        if code:
            color = 'RED'
        if not empty(out):
            Logs.pprint(color, 'Standard output from %s' % f)
            Logs.pprint(color, out)
        if not empty(err):
            Logs.pprint(color, 'Standard error from %s' % f)
            Logs.pprint(color, err)

def build(bld):
    bld(
            rule = 'python ${SRC} ${TGT}',
            source = ['utils/clc2cpp.py'] + bld.path.ant_glob('kernels/*.cl'),
            target = 'src/kernels.cpp')
    sources = bld.path.ant_glob('src/*.cpp', excl = 'src/provenance.cpp') + ['src/kernels.cpp']
    bld(
            features = ['cxx', 'provenance'],
            source = 'src/provenance.cpp',
            name = 'provenance')
    bld(
            features = ['cxx', 'cxxstlib'],
            source = sources,
            target = 'mls',
            use = 'OPENCL CLOGS STXXL TIMER',
            name = 'libmls')
    bld.program(
            source = ['mlsgpu.cpp'],
            target = 'mlsgpu',
            use = ['libmls', 'provenance', 'OPENCL'],
            lib = libs)
    bld.program(
            source = ['plymanifold.cpp', 'src/ply.cpp', 'test/manifold.cpp'],
            target = 'plymanifold',
            lib = ['boost_math_c99f-mt', 'boost_math_c99-mt'],
            install_path = None)

    bld(
            name = 'manual',
            rule = '${XSLTPROC} --xinclude --stringparam mlsgpu.version ' + VERSION + ' -o ${TGT} ${SRC}',
            source = ['doc/mlsgpu-user-manual-xml.xsl', 'doc/mlsgpu-user-manual.xml'],
            target = 'doc/mlsgpu-user-manual.html')
    output_dir = bld.bldnode.find_or_declare('doc')
    output_dir.mkdir()
    bld.install_files('${HTMLDIR}',
            output_dir.ant_glob('mlsgpu-user-manual.*'),
            cwd = bld.bldnode.find_dir('doc'),
            relative_trick = True)

    if bld.env['unit_tests']:
        test_features = 'cxx cxxprogram'
        if not bld.options.no_tests:
            test_features += ' test'
        bld.program(
                features = test_features,
                source = bld.path.ant_glob('test/*.cpp'),
                target = 'testmain',
                use = ['CPPUNIT', 'GMP', 'libmls'],
                lib = libs,
                install_path = None)
        def print_env(bld):
            print bld.env
        bld.add_post_fun(print_unit_tests)


