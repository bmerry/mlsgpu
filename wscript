import os.path

import waflib.Errors
import waflib.Tools.waf_unit_test

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

def options(opt):
    opt.load('gnu_dirs')
    opt.load('compiler_cxx')
    opt.load('waf_unit_test')
    opt.load('provenance', tooldir = '../../waf-tools')
    opt.add_option('--variant', type = 'choice', dest = 'variant', default = 'release', action = 'store', help = 'build variant', choices = list(variants.keys()))
    opt.add_option('--lto', dest = 'lto', default = False, action = 'store_true', help = 'use link-time optimization')
    opt.add_option('--cl-headers', action = 'store', default = None, help = 'Include path for OpenCL')
    opt.add_option('--enable-extras', action = 'store_true', default = False, help = 'Build extra internal tools')
    opt.add_option('--no-tests', action = 'store_true', default = False, help = 'Do not run unit tests')
    opt.add_option('--without-mpi', dest = 'mpi', action = 'store_false', default = True, help = 'Do not build for MPI')

def configure_variant(conf):
    if conf.env['assertions']:
        conf.define('DEBUG', '1', quote = False)
    else:
        conf.define('NDEBUG', '1', quote = False)
        conf.define('BOOST_DISABLE_ASSERTS', '1', quote = False)
    if conf.env['unit_tests']:
        conf.define('UNIT_TESTS', 1, quote = False)

def configure_mpi(conf):
    if conf.options.mpi:
        try:
            conf.check_cfg(path = 'mpicxx', args = '--showme:compile',
                    package = '', uselib_store = 'MPI')
            conf.check_cfg(path = 'mpicxx', args = '--showme:link',
                    package = '', uselib_store = 'MPI')
            have_mpi = True
        except waflib.Errors.ConfigurationError:
            have_mpi = False
    else:
        have_mpi = False

    conf.env['mpi'] = have_mpi

def configure_variant_gcc(conf):
    configure_mpi(conf)
    ccflags = ['-Wall', '-W', '-pthread', '-fopenmp']
    conf.env.append_value('LINKFLAGS', ['-pthread', '-fopenmp'])
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
        # Avoids precision weirdness due to 80-bit 8087 registers, and makes SSE2 available
        ccflags.extend(['-mfpmath=sse', '-msse2'])
    lto = conf.env['lto']
    if lto:
        ccflags.extend(['-flto', '-B/usr/lib/gold-ld'])
    conf.env.append_value('CFLAGS', ccflags)
    conf.env.append_value('CXXFLAGS', ccflags)
    if lto:
        # -flto requires compilation flags to be provided at link time
        conf.env.append_value('LINKFLAGS', ccflags)

    # The -Wno-unknown-pragmas is because we also use #pragma STDC FENV_ACCESS
    conf.env.append_value('CXXFLAGS_ROUNDING_MATH', ['-frounding-math', '-Wno-unknown-pragmas'])

def configure_variant_msvc(conf):
    # Wall is not enable since boost vomits up zillions of warnings
    ccflags = ['/W1', '/EHsc', '/MD']

    # Autolinked, so no need to detect or link
    conf.env['LIB_BOOST'] = []
    conf.env['LIB_BOOST_MATH'] = []
    conf.env['LIB_BOOST_TEST'] = []

    # Reduces legacy stuff
    conf.env.append_value('DEFINES', 'WIN32_LEAN_AND_MEAN')
    # windows.h defines macros called min and max by default. EVIL!
    conf.env.append_value('DEFINES', 'NOMINMAX')

    if conf.env['optimize']:
        ccflags.extend(['/O2', '/Ob2'])
        conf.env.append_value('LINKFLAGS', '/OPT:REF')
    if conf.env['debuginfo']:
        ccflags.append('/Zi')
        conf.env.append_value('LINKFLAGS', '/DEBUG')
    if conf.env['lto']:
        ccflag.append('/Og')
    conf.env.append_value('CFLAGS', ccflags)
    conf.env.append_value('CXXFLAGS', ccflags)
    if 'LIBPATH' in os.environ:
        for item in os.environ['LIBPATH'].split(os.pathsep):
            conf.env.append_value('LIBPATH', item)

def configure(conf):
    conf.load('waf_unit_test')
    conf.load('gnu_dirs')
    conf.load('compiler_cxx')
    conf.load('provenance', tooldir = '../../waf-tools')

    for (key, value) in variants[conf.options.variant].items():
        conf.env[key] = value
    conf.env['lto'] = conf.options.lto
    configure_variant(conf)

    # Defaults that may be overridden per compiler
    conf.env['LIB_BOOST_MATH'] = [
        'boost_math_c99-mt',
        'boost_math_c99f-mt']
    conf.env['LIB_BOOST'] = conf.env['LIB_BOOST_MATH'] + [
        'boost_program_options-mt',
        'boost_iostreams-mt',
        'boost_thread-mt',
        'boost_system-mt',
        'boost_serialization-mt',
        'boost_filesystem-mt'
    ]
    conf.env['LIB_BOOST_TEST'] = []

    if conf.env['CXX_NAME'] == 'gcc':
        configure_variant_gcc(conf)
    elif conf.env['CXX_NAME'] == 'msvc':
        configure_variant_msvc(conf)

    conf.define('PROVENANCE_VARIANT', conf.options.variant)

    if conf.env['unit_tests']:
        try:
            conf.check_cxx(
                features = ['cxx', 'cxxprogram'],
                header_name = 'cppunit/Test.h', lib = 'cppunit', uselib_store = 'CPPUNIT',
                msg = 'Checking for cppunit')
        except waflib.Errors.ConfigurationError:
            # A cppunit installed from source uses symbols from libdl but does not link
            # against it. The Ubuntu package does link correctly.
            conf.check_cxx(
                features = ['cxx', 'cxxprogram'],
                header_name = 'cppunit/Test.h', lib = ['cppunit', 'dl'], uselib_store = 'CPPUNIT',
                msg = 'Checking for cppunit')

    conf.define('CL_USE_DEPRECATED_OPENCL_1_1_APIS', 1, quote = False)
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

    conf.env['extras'] = conf.options.enable_extras

    conf.check_cxx(header_name = 'tr1/cstdint', mandatory = False)
    conf.check_cxx(header_name = 'tr1/unordered_map', mandatory = False)
    conf.check_cxx(header_name = 'tr1/unordered_set', mandatory = False)
    conf.check_cxx(header_name = 'xmmintrin.h', mandatory = False)
    conf.check_cxx(header_name = 'emmintrin.h', mandatory = False)

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
            features = ['cxx', 'cxxprogram'],
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

    # Apparently MacOS has a function with the same name but different
    # signature, which is why we need a fragment of code to test it.
    pthread_setname_np_test = '''
#ifndef _GNU_SOURCE
# define _GNU_SOURCE 1
#endif
#include <pthread.h>

int main() {
    char buffer[1024];
    pthread_getname_np(pthread_self(), buffer, sizeof(buffer));
    pthread_setname_np(pthread_self(), "Test thread");
    return 0;
}'''
    conf.check_cxx(
        features = ['cxx', 'cxxprogram'],
        fragment = pthread_setname_np_test,
        function_name = 'pthread_setname_np',
        msg = 'Checking for pthread_setname_np',
        mandatory = False)

    conf.check_cxx(
        features = ['cxx', 'cxxprogram'],
        function_name = 'QueryPerformanceCounter', header_name = 'windows.h',
        uselib_store = 'TIMER',
        msg = 'Checking for QueryPerformanceCounter',
        mandatory = False)

    for f in ['CreateFile', 'ReadFile', 'CloseHandle']:
        conf.check_cxx(
            features = ['cxx', 'cxxprogram'],
            function_name = f, header_name = 'windows.h',
            msg = 'Checking for ' + f,
            mandatory = False)
    for f in ['open', 'pread', 'close', 'posix_fadvise']:
        conf.check_cxx(
            features = ['cxx', 'cxxprogram'],
            function_name = f, header_name = ['fcntl.h', 'sys/types.h', 'unistd.h'],
            defines = ['_POSIX_C_SOURCE=200809L'],
            msg = 'Checking for ' + f,
            mandatory = False)

    conf.check_cxx(fragment = '''
#include <CL/cl.hpp>

static int dummy = sizeof(cl::Local);
''',
        features = ['cxx'], msg = 'Checking for cl::Local',
        define = 'HAVE_CL_LOCAL',
        mandatory = False)

    for l in conf.env['LIB_BOOST'] + conf.env['LIB_BOOST_TEST']:
        conf.check_cxx(lib = l)
    conf.find_program('xsltproc', mandatory = False)

    conf.write_config_header('config.h')
    conf.env.append_value('DEFINES', 'HAVE_CONFIG_H=1')
    conf.env.append_value('INCLUDES', '.')

def empty(s):
    '''Determine whether s contains non-whitespace characters'''
    import re
    return re.match(b'^\s*$', s)

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
            Logs.pprint(color, out.decode('utf-8'))
        if not empty(err):
            Logs.pprint(color, 'Standard error from %s' % f)
            Logs.pprint(color, err.decode('utf-8'))

def build(bld):
    make_kernels = bld(
            rule = 'python ${SRC} ${TGT}',
            source = ['utils/clc2cpp.py'] + bld.path.ant_glob('kernels/*.cl'),
            target = 'src/kernels.cpp')
    make_kernels.post() # To allow dep tracker to find the target

    splat_set_sse = bld(
        features = ['cxx'],
        source = 'src/splat_set_sse.cpp',
        name = 'splat_set_sse',
        use = 'ROUNDING_MATH')

    core_sources = [
            'src/bucket.cpp',
            'src/bucket_collector.cpp',
            'src/circular_buffer.cpp',
            'src/decache.cpp',
            'src/diskstats.cpp',
            'src/fast_ply.cpp',
            'src/grid.cpp',
            'src/logging.cpp',
            'src/misc.cpp',
            'src/options.cpp',
            'src/progress.cpp',
            'src/statistics.cpp',
            'src/splat.cpp',
            'src/splat_set.cpp',
            'src/thread_name.cpp',
            'src/timeplot.cpp',
            'src/timer.cpp']
    cl_sources = [
            'src/bucket_loader.cpp',
            'src/clh.cpp',
            'src/kernels.cpp',
            'src/marching.cpp',
            'src/mesh.cpp',
            'src/mesh_filter.cpp',
            'src/mesher.cpp',
            'src/mls.cpp',
            'src/splat_tree.cpp',
            'src/splat_tree_host.cpp',
            'src/splat_tree_cl.cpp',
            'src/statistics_cl.cpp',
            'src/workers.cpp',
            'src/mlsgpu_core.cpp']
    mpi_sources = [
            'src/serialize.cpp',
            'src/progress_mpi.cpp']
    bld(
            features = ['cxx', 'provenance'],
            source = 'src/provenance.cpp',
            name = 'provenance')
    bld(
            features = ['cxx', 'cxxstlib'],
            source = core_sources,
            target = 'mls_core',
            use = 'TIMER BOOST splat_set_sse',
            name = 'libmls_core')
    bld(
            features = ['cxx', 'cxxstlib'],
            source = cl_sources,
            target = 'mls_cl',
            use = 'OPENCL CLOGS BOOST libmls_core',
            name = 'libmls_cl')
    bld.program(
            source = 'mlsgpu.cpp',
            target = 'mlsgpu',
            use = ['libmls_cl', 'libmls_core', 'provenance'])
    if bld.env['mpi']:
        bld(
                features = ['cxx', 'cxxstlib'],
                source = mpi_sources,
                target = 'mls_mpi',
                use = 'OPENCL BOOST MPI',
                name = 'libmls_mpi')
        bld.program(
                source = 'mlsgpu-mpi.cpp',
                target = 'mlsgpu-mpi',
                use = ['libmls_cl', 'libmls_core', 'libmls_mpi', 'provenance', 'MPI'])

    if bld.env['extras']:
        bld.program(
                source = [
                    'extras/plymanifold.cpp', 
                    'extras/ply.cpp',
                    'test/manifold.cpp'],
                target = 'plymanifold',
                use = 'BOOST_MATH libmls_core',
                install_path = None)

    if bld.env['XSLTPROC']:
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
        nonmpi_sources = bld.path.ant_glob('test/test_*.cpp') + ['test/testmain.cpp']
        mpi_sources = bld.path.ant_glob('test/mpi/*.cpp')
        common_sources = [
                'test/manifold.cpp',
                'test/memory_reader.cpp',
                'test/memory_writer.cpp',
                'test/testutil.cpp']

        test_use = ['CPPUNIT', 'BOOST_TEST', 'libmls_core', 'libmls_cl']
        if bld.env['extras']:
            nonmpi_sources.extend(bld.path.ant_glob('extras/test_*.cpp'))
            nonmpi_sources.append('extras/ply.cpp')

        # TODO: use a static library for common_sources
        bld.program(
                features = test_features,
                source = nonmpi_sources + common_sources,
                target = 'testmain',
                use = test_use,
                install_path = None)
        if bld.env['mpi']:
            gen = bld.program(
                    features = test_features,
                    source = mpi_sources + common_sources,
                    target = 'testmpi',
                    use = test_use + ['libmls_mpi', 'MPI'],
                    install_path = None)
            # Find the unit test task and modify it to run through mpirun
            gen.post() # Forces the task to be generated
            for task in gen.tasks:
                if isinstance(task, waflib.Tools.waf_unit_test.utest):
                    task.ut_exec = ['mpirun', '-n', '4', task.inputs[0].abspath()]
        bld.add_post_fun(print_unit_tests)
