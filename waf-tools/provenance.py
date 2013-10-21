#!/usr/bin/env python
import subprocess

def options(opt):
    pass

def configure(opt):
    pass

from waflib.TaskGen import feature, before_method

@feature('provenance')
@before_method('process_source')
def add_provenance(self):
    provenance = b'unknown'
    try:
        try:
            check_output = subprocess.check_output
        except AttributeError:
            pass  # Happens if Python version is too old
        else:
            provenance = subprocess.check_output(['git', 'describe', '--dirty', '--always']).strip()
    except subprocess.CalledProcessError:
        pass
    except WindowsError:
        pass
    self.env.append_value('DEFINES', 'PROVENANCE_VERSION="' + str(provenance.decode()) + '"')
