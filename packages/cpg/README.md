# Connor's Paramfile Generator (cpg)

`cpg` is a command-line utility for generating paramfiles for GalIC and GIZMO,
as well as creating and maintaining a coherent directory structure for all the
corresponding files. It is built using `argparse` and `jinja2`.

## GalIC disk fix

By default, GalIC specifies the disk parameters by taking as input the halo spin
parameter, the disk spin fraction, the disk mass fraction, and the disk scale
height (as a fraction of the disk scale length). It then uses these quantities
to compute the disk scale length, the internally multiplies the given scale
height by the computed scale length to obtain the actual scale height. `cpg`, on
the other hand, attempts to allow you to specify the disk structure by directly
specifying the disk scale parameters in units of kpc. It does so by attempting
to compute the necessary values for the spin parameter, spin fraction, and scale
height as fraction of scale length such that GalIC's computations will reproduce
the desired values. 

However, the scale length in GalIC is computed using an iterative process,
making it difficult to predict the final scale length or height from the input
parameters alone. Just using the values computed by `cpg` to set the disk scale
parameters will result in the GalIC-computed parameters being off by ~25% for
our typical cases. This can be easily fixed by disabling the iterative scale
length computation. To do so, go to `src/structure.c` in GalIC and comment out
the do-while loop at around line 125. (Don't forget to recompile!) Then, the
disk scale parameters that you give to `cpg` will be the exact values computed
and used internally in GalIC.

