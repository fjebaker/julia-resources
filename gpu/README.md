# GPU processing with Julia 

This directory contains notes and scripts relating to writing GPU accelerated code, and best practices.

I am using [CUDA.jl](https://juliagpu.gitlab.io/CUDA.jl/), with an NVIDIA GTX 980 graphics card.


## Geodesic Tracing
In another project I am trying to write a geodesic intersection tracing algorithm with parallelization support for the GPU.

The included notebook here explains my initial research procedure in achieving this.