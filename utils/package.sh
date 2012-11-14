#!/bin/bash
set -e
./waf
PKGDIR=mlsgpu-package
rm -rf $PKGDIR
mkdir $PKGDIR
cp build/mlsgpu build/testmain build/testmpi $PKGDIR
cp $HOME/src/stxxl-1.3.1-chpc/lib/libstxxl.so.1 $PKGDIR
for i in \
    libclogs.so.1 \
    libboost_math_c99.so.1.46.1 \
    libboost_math_c99f.so.1.46.1 \
    libboost_program_options.so.1.46.1 \
    libboost_iostreams.so.1.46.1 \
    libboost_thread.so.1.46.1 \
    libstdc++.so.6 \
    libgcc_s.so.1 \
    libcppunit-1.12.so.1 \
    libbz2.so.1.0 \
    libz.so.1 \
    libc.so.6 \
    libpthread.so.0 \
    libm.so.6 \
    libdl.so.2 \
    librt.so.1 \
    ld-linux-x86-64.so.2; do

    found=0
    for path in /usr/local/lib /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/lib64 /usr/lib /lib64 /lib; do
        if [ -e $path/$i ]; then
            cp $path/$i $PKGDIR
            chmod +x $PKGDIR/$i
            found=1
            break
        fi
    done
    if [ $found == 0 ]; then
        echo "Could not find $i" 1>&2
        exit 1
    fi
done
