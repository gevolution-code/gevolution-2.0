# gevolution-2.0

Copyright (c) 2015-2026 Julian Adamek
(Université de Genève & Observatoire de Paris & Queen Mary University of London & Universität Zürich & ETH Zürich)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
  
**The Software is provided "as is", without warranty of any kind, expressed or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising from,
out of or in connection with the Software or the use or other dealings in the
Software.**

## Compilation and usage

Before compilation, make sure that all required external libraries are
installed:

* LATfield2 [version 1.1 with GPU support](https://gitlab.uzh.ch/julian.adamek/LATfield2.git)
* FFTW version 3
* GNU Scientific Library (GSL) including CBLAS
* HDF5

Create a local build configuration and set the CUDA architecture and dependency
paths for the target system:

    cp config/local.mk.example config/local.mk
    make print-config
    make -j

The ignored `config/local.mk` file is intended for workstation- or
cluster-specific settings. The tracked CSCS Alps profile can be selected with:

    make CONFIG=config/alps.mk

A typical command to run a simulation looks like this:

    mpirun -np 16 ./gevolution -n 4 -m 4 -s settings.ini

For further information, please refer to the online documentation at [gevolution-code.net](https://gevolution-code.net/docs/gevolution-gpu)

## Credits

If you use gevolution for scientific work, we kindly ask you to cite
*Adamek J., Daverio D., Durrer R., and Kunz M., Nature Phys. 12, 346 (2016)*
in your publications.

For bug reports and other important feedback you can contact the authors,
adamekj@ethz.ch
