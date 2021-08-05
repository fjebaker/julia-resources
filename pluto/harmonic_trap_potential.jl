### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ cba7c858-f5ca-11ea-1449-1f140367445a
begin
	using Pkg
	Pkg.activate("../..")
	Pkg.add("QuantumOptics")
	Pkg.add("Plots")
end

# ╔═╡ 4603956e-f5cb-11ea-286f-a50c5a803a44
using QuantumOptics, Plots; plotly();

# ╔═╡ 7c21548e-f5ca-11ea-0c71-9db66028ef07
md"""
# Particle in Harmonic Trap Potential
Following from a tutorial in the [`QuantumOptics.jl` docs](https://docs.qojulia.org/examples/particle-in-harmonic-trap/).

We describe the particle through the hamiltonian

$$H = \frac{\hat{p}^2}{2m} + \frac{1}{2}m \omega^2 \hat{x}^2,$$

with trapping potential strength $\omega$. For a numerical solution, we must work in a specific basis. We may choose to work in real or momentum space.

The position basis is defined by the range of positions it spans between some minimal and maximal position, such that the number of points in the position dictates the dimensionality of the basis:
"""

# ╔═╡ 1d13da7e-f5cb-11ea-126c-a903d9b3f278
begin
	m = 1.0
	ω = 0.5
	
	# define position basis
	x_min = -5
	x_max = 5
	N_points = 100
	b_position = PositionBasis(x_min, x_max, N_points)
end

# ╔═╡ 8745f18e-f5cb-11ea-2aef-638b8e6ae3f1
md"Next, we specify the Hamiltonian momentum and position operators on our basis:"

# ╔═╡ 9703134a-f5cb-11ea-31b9-41dd4cd3534b
begin
	p_op = momentum(b_position) # dense
	x_op = position(b_position) # sparse
end;

# ╔═╡ b9678ae4-f5cb-11ea-318c-57fa5af61042
md"and define the Hamiltonian itself:"

# ╔═╡ c282b944-f5cb-11ea-2a90-6fd07289523c
H_pos = p_op^2 / 2m + 1/2 * m * ω^2 * dense(x_op^2);

# ╔═╡ e8fa2166-f5cb-11ea-038c-21acea19bc5e
md"Were we to use a momentum basis instead, we would write 
```
b_momentum = MomentumBasis(b_position);

p_op = momentum(b_momentum) # Sparse operator
x_op = position(b_momentum) # Dense operator

H = dense(p_op^2) / 2m + 1/2 * m *ω^2 * x_op^2
```

However, for a `PositionBasis`, the library can automatically infer the corresponding `MomentumBasis` by calculating

$$p_\text{min} = \frac{- \pi}{\text{d}x},$$

and

$$p_\text{max} = \frac{\pi}{\text{d}x},$$

with

$$\text{d}x = \frac{x_\text{max} - x_\text{min}}{N}$$

In Julia, this is as simple as:
"

# ╔═╡ 5a409d78-f5cc-11ea-1869-cd2b15baf62d
b_momentum = MomentumBasis(b_position)

# ╔═╡ 6f9c99e2-f5cc-11ea-100e-637af8a221ab
md"""
## Squeezing performance with FFT
An issues with the above formulation of the problem is that in position (real) space, the position operator is diagonal, whereas momentum is a dense matrix, and vice-versa for the momentum space. The calculation therefore scales with $N^2$ for a given dimension $N$. A solution is to apply a Fast Fourier Transformation (FFT) prescription to translate the space between momentum and real space, allowing us to use diagonal operator forms, and thus computational complexity $N\text{log}N$. 

In Julia, we create a map

$$T_{p,x} : \mathcal{H_p} \rightarrow \mathcal{H_x}$$

with
"""

# ╔═╡ c925e7f0-f5cc-11ea-1077-fd01800db887
T_px = transform(b_momentum, b_position);

# ╔═╡ 258b87fe-f5cd-11ea-3f02-c1283e60507f
md"To make use of this map, we also need to use the concept of [*lazy operators*](https://docs.qojulia.org/quantumobjects/operators/#Lazy-operators-1), which delay the calculations until they are needed. They also have the benefit of evaluating e.g.

$$A \times ( B \times x)$$

instead of

$$(A \times B) \times x$$

and thus never have to compute a matrix times matrix computation.

We obtain the inverse map by taking the Hermitian conjugate:
"

# ╔═╡ b66fde76-f5cd-11ea-1e3a-53f960657152
T_xp = dagger(T_px);

# ╔═╡ daf583ba-f5cd-11ea-224f-f9135a2a5910
md"And then totally define our Hamiltonian the *lazy* way with:"

# ╔═╡ e7afd29a-f5cd-11ea-351d-1d5a247267f4
begin
	x = position(b_position)
	p = momentum(b_momentum)
	
	H_kin = LazyProduct(T_xp, p^2 / 2m, T_px)
	V = ω * x^2 			# potential
	H = LazySum(H_kin, V)	# addition
end;

# ╔═╡ 3b486836-f5ce-11ea-128b-c922e428efa1
md"*NB:* our operators now live in their respective bases, and as such are sparse. We then define the kinetic term of the hamiltonian by invoking our map between the bases."

# ╔═╡ 57087ab6-f5ce-11ea-3f2b-552d56a5dbb9
md"""
## Schrödinger Evolution
We have now prepped our system for simulation, which we can evolve according to the usual [Schrödinger time evolution equation](https://docs.qojulia.org/timeevolution/schroedinger/).

We define our initial state:
"""

# ╔═╡ 81baf69c-f5ce-11ea-3a91-01a565d3bb2f
begin
	x0 = 1.5
	p0 = 0
	sigma0 = 0.6
	ψ0 = gaussianstate(b_position, x0, p0, sigma0)
end;

# ╔═╡ b3cd53ca-f5ce-11ea-11c1-213c799b4db1
md"Now we give the time evolution of the system:"

# ╔═╡ caf484a8-f5ce-11ea-3959-cf287018c9a2
begin
	T = [0:0.1:3;]
	tout, ψt = timeevolution.schroedinger(T, ψ0, H)
end;	

# ╔═╡ 16ca34ac-f5cf-11ea-26ff-37610cfc0156
md"This evolution is calculated by [integrating the Schrödinger equation](https://github.com/qojulia/QuantumOptics.jl/blob/6a90f3a68545fe973059382396e4a9f276e9a6ce/src/schroedinger.jl#L1-L14). We can then plot the result interactively in `Pluto.jl`:"

# ╔═╡ 6667f1ae-f5cf-11ea-04d5-d373ab6ba72f
@bind selector html"<input type=range min=0 max=100></input>"

# ╔═╡ c604c130-f5cf-11ea-11f4-5d15d43a4b99
begin
	i = trunc(Int, (selector/100)*(length(T)-1) + 1)
	md"``i=`` $i"
end

# ╔═╡ ec0beac2-f5cf-11ea-3c93-099fa1a28733
begin
	x_points = samplepoints(b_position)
	
	n = abs.(ψ0.data).^2
	V_plot = ω * x_points.^2
	C = maximum(V_plot)/maximum(n)
	
	plot(
		x_points, 
		(V_plot.-3)./C, 
		label="potential", 
		color=:black, 
		ls=:dash
	)
	plot!(
		x_points, 
		abs.(ψt[i].data).^2, 
		label="solution at i=$i"
	)
	
end

# ╔═╡ 6e88378c-f5d1-11ea-0b1a-15ee29ddc13f
md"Or show all plots in one:"

# ╔═╡ 1df2af04-f5d2-11ea-2930-5d796c522a78
begin
	output_plot = plot(
		x_points, 
		(V_plot.-3)./C, 
		label="potential", 
		color=:black, 
		ls=:dash
	)
	for ind=1:length(T)
		plot!(
			x_points, 
			abs.(ψt[ind].data).^2, 
			alpha=0.9 * ( float(ind) / length(T) )^8 + 0.1, 
			label=nothing)
	end
	output_plot
end

# ╔═╡ Cell order:
# ╠═cba7c858-f5ca-11ea-1449-1f140367445a
# ╟─7c21548e-f5ca-11ea-0c71-9db66028ef07
# ╠═4603956e-f5cb-11ea-286f-a50c5a803a44
# ╠═1d13da7e-f5cb-11ea-126c-a903d9b3f278
# ╟─8745f18e-f5cb-11ea-2aef-638b8e6ae3f1
# ╠═9703134a-f5cb-11ea-31b9-41dd4cd3534b
# ╟─b9678ae4-f5cb-11ea-318c-57fa5af61042
# ╠═c282b944-f5cb-11ea-2a90-6fd07289523c
# ╟─e8fa2166-f5cb-11ea-038c-21acea19bc5e
# ╠═5a409d78-f5cc-11ea-1869-cd2b15baf62d
# ╟─6f9c99e2-f5cc-11ea-100e-637af8a221ab
# ╠═c925e7f0-f5cc-11ea-1077-fd01800db887
# ╟─258b87fe-f5cd-11ea-3f02-c1283e60507f
# ╠═b66fde76-f5cd-11ea-1e3a-53f960657152
# ╟─daf583ba-f5cd-11ea-224f-f9135a2a5910
# ╠═e7afd29a-f5cd-11ea-351d-1d5a247267f4
# ╟─3b486836-f5ce-11ea-128b-c922e428efa1
# ╟─57087ab6-f5ce-11ea-3f2b-552d56a5dbb9
# ╠═81baf69c-f5ce-11ea-3a91-01a565d3bb2f
# ╟─b3cd53ca-f5ce-11ea-11c1-213c799b4db1
# ╠═caf484a8-f5ce-11ea-3959-cf287018c9a2
# ╟─16ca34ac-f5cf-11ea-26ff-37610cfc0156
# ╟─6667f1ae-f5cf-11ea-04d5-d373ab6ba72f
# ╟─c604c130-f5cf-11ea-11f4-5d15d43a4b99
# ╠═ec0beac2-f5cf-11ea-3c93-099fa1a28733
# ╟─6e88378c-f5d1-11ea-0b1a-15ee29ddc13f
# ╠═1df2af04-f5d2-11ea-2930-5d796c522a78
