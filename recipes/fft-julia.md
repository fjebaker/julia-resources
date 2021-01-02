# Fourier Transforms with Julia

In Julia, we use the [`FFTW`](https://duckduckgo.com/?t=ffab&q=FFTW&ia=web) for fast-Fourier Transforms, through the wrapping library [`FFTW.jl`](https://juliamath.github.io/FFTW.jl/latest/).

The package extends [`AbstractFFTs.jl`](https://github.com/JuliaMath/AbstractFFTs.jl), thus uses the documentation at
- [AbstractFFTs](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#Public-Interface-1)
- [FFTW](https://juliamath.github.io/FFTW.jl/latest/fft.html)


## Real 1D data

To transform `x` and `y` data of 1D Real-valued series, use
```julia
function transform(y, dx)
    fk = FFTW.rfft(y)
    k = FFTW.rfftfreq(length(y))
    kfreq = k / dx
    (kfreq, fk)
end
```

We use `rfft` over `fft` since, as is written in [the docs](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.rfft):

> Multidimensional FFT of a real array A, exploiting the fact that the transform has conjugate symmetry in order to save roughly half the computational time and storage costs compared with fft.
