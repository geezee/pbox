# Simulating a particle in a square quantum box using CUDA

## Description
As a personal project to practice my knowledge with parallel algorithms, GPGPU
(General-purpose computing on graphics processing units) and CUDA I've decided
to simulate a problem that I've found hard to grasp in the course Modern
Physics (PHYS 212); the particle in a box problem. Basically this simulation
highlights the probabilistic nature of quantum mechanics by visualizing the
probability of a particle to exist in a very small box.

## Screenshots
![](https://raw.github.com/geezee/pbox/master/paper/graphics/result1.png)
![](https://raw.github.com/geezee/pbox/master/paper/graphics/result2.png)
![](https://raw.github.com/geezee/pbox/master/paper/graphics/result3.png)
![](https://raw.github.com/geezee/pbox/master/paper/graphics/result4.png)
![](https://raw.github.com/geezee/pbox/master/paper/graphics/result5.png)
![](https://raw.github.com/geezee/pbox/master/paper/graphics/result6.png)

## Documentation
The documentation can be found in `/paper` as LaTeX and PDF format with all
the graphics used. The documentation includes the API of the code, the
mathematics of the problem and the expected result with explanation about the
implementation.

## Install and run
```
git clone https://github.com/geezee/pbox.git
cd pbox/src
make && make run
```

## Dependencies
This program uses `nvcc` to compile the source code, therefore it requires you
to install `CUDA` and the proprietary driver for your Nvidia graphics card.

## Code license
Copyright (C) 2013 George Zakhour

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see http://www.gnu.org/licenses/.
