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
        'debuginfo': True,
        'symbols': False,
        'optimize': True,
        'assertions': False,
        'expensive_assertions': False,
        'unit_tests': False,
        'coverage': False,
    }
}

def options(opt):
    opt.load('compiler_cxx')
    opt.load('waf_unit_test')
    opt.load('provenance', tooldir = '../../waf-tools')
    opt.add_option('--variant', type = 'choice', dest = 'variant', default = 'debug', action = 'store', help = 'build variant', choices = variants.keys())
    opt.add_option('--lto', dest = 'lto', default = False, action = 'store_true', help = 'use link-time optimization')
    opt.add_option('--cl-headers', action = 'store', default = None, help = 'Include path for OpenCL')
    opt.add_option('--use-images', action = 'store_true', default = False, help = 'Use OpenCL images for start table')

def configure_variant(conf):
    if conf.env['assertions']:
        conf.define('DEBUG', '1', quote = False)
    else:
        conf.define('NDEBUG', '1', quote = False)
        conf.define('BOOST_DISABLE_ASSERTS', '1', quote = False)
    if conf.env['unit_tests']:
        conf.define('UNIT_TESTS', 1, quote = False)

def configure_variant_gcc(conf):
    ccflags = ['-Wall', '-W']
    if conf.env['optimize']:
        ccflags.append('-O2')
    else:
        pass # ccflags.append('-fno-inline')
    if conf.env['debuginfo']:
        ccflags.append('-g')
    if conf.env['coverage']:
        ccflags.append('-fprofile-arcs')
        ccflags.append('-ftest-coverage')
        conf.env.append_value('LINKFLAGS', '-fprofile-arcs')
        conf.env.append_value('LINKFLAGS', '-ftest-coverage')
    if not conf.env['symbols']:
        conf.env.append_value('LINKFLAGS', '-s')
    if conf.env['DEST_CPU'] == 'x86':
        # Avoids precision weirdness due to 80-bit 8087 registers
        ccflags.extend(['-mfpmath=sse', '-msse2'])
    if conf.env['lto']:
        ccflags.extend(['-flto', '-B/usr/lib/gold-ld'])
    conf.env.append_value('CFLAGS', ccflags)
    conf.env.append_value('CXXFLAGS', ccflags)
    if conf.env['lto']:
        # -flto requires compilation flags to be provided at link time
        conf.env.append_value('LINKFLAGS', ccflags)

def configure(conf):
    conf.load('waf_unit_test')
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
    conf.define('USE_IMAGES', int(conf.options.use_images))

    conf.check_cxx(header_name = 'cppunit/Test.h', lib = 'cppunit', uselib_store = 'CPPUNIT')
    if conf.options.cl_headers:
        conf.env.append_value('INCLUDES_OPENCL', [conf.options.cl_headers])
    conf.env.append_value('LIB_OPENCL', ['OpenCL'])
    conf.check_cxx(header_name = 'CL/cl.hpp', use = 'OPENCL')

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
    bld.read_shlib('clcpp', paths = ['../clcpp/build'])
    bld(
            name = 'CLCPP',
            use = 'clcpp',
            export_includes = '../clcpp/include'
        )

    bld(
            rule = 'python ${SRC} ${TGT}',
            source = ['clc2cpp.py'] + bld.path.ant_glob('kernels/*.cl'),
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
            use = 'OPENCL CLCPP',
            name = 'libmls')
    bld.program(
            source = ['mlsgpu.cpp'],
            target = 'mlsgpu',
            use = ['libmls', 'provenance', 'OPENCL'],
            lib = ['boost_program_options-mt', 'rt'])
    if bld.env['unit_tests']:
        bld.program(
                features = 'test',
                source = bld.path.ant_glob('test/*.cpp'),
                target = 'testmain',
                use = ['CPPUNIT', 'GMP', 'libmls'],
                lib = ['boost_program_options-mt'])
        def print_env(bld):
            print bld.env
        bld.add_post_fun(print_unit_tests)
