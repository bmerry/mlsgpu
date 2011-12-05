#ifndef ELEMENT_T
# error "ELEMENT_T is required"
#endif

#ifndef WORK_GROUP_SIZE
#error "WORK_GROUP_SIZE is required"
#endif

#ifndef ITERATIONS
#error "ITERATIONS is required"
#endif

/**
 * Shorthand for defining a kernel with a fixed work group size.
 * This is needed to unconfuse Doxygen's parser.
 */
#define KERNEL(xsize, ysize, zsize) __kernel __attribute__((reqd_work_group_size(xsize, ysize, zsize)))

KERNEL(WORK_GROUP_SIZE, 1, 1)
void copy(__global ELEMENT_T * restrict out, __global const ELEMENT_T * restrict in, uint limit)
{
    uint p = get_group_id(0) * (WORK_GROUP_SIZE * ITERATIONS) + get_local_id(0);
    for (uint i = 0; i < ITERATIONS; i++)
    {
        if (p < limit)
            out[p] = in[p];
        p += WORK_GROUP_SIZE;
    }
}
