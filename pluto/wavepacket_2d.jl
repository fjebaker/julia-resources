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

# ╔═╡ 291c7d5a-f6af-11ea-33b4-67d04e0a16ec
begin
	using Pkg; Pkg.activate("../..");
	Pkg.add("QuantumOptics")
	Pkg.add("Plots")
	Pkg.add("LinearAlgebra")
	
	using QuantumOptics, Plots
	plotly()
end

# ╔═╡ 6cf6ad0e-f6b2-11ea-269a-678bed52852e
using LinearAlgebra

# ╔═╡ 5e2299e6-f6af-11ea-3c09-8b6fe0aec753
md"""
# Two-Dimensional Wavepacket interacting with a Gaussian Potential

Following from the first notebook on a particle in a Harmonic Trap Potential.

We will simulate a 2-d wavepacket by taking the tensor product of two spaces. We start by defining a position and momentum space for each dimension:
"""

# ╔═╡ 82d80ba2-f6af-11ea-133b-11cf0eb8bde2
begin
	N_points = 100
	
	x_min = -30
	x_max = 50
	bx_x = PositionBasis(x_min, x_max, N_points)
	bp_x = MomentumBasis(bx_x)
	
	y_min = -30
	y_max = 50
	bx_y = PositionBasis(y_min, y_max, N_points)
	bp_y = MomentumBasis(bx_y)
end;

# ╔═╡ d113e17e-f6af-11ea-0376-2553b74c647e
md"We then create a composite space taking into account both dimensions, and define mappings between them."

# ╔═╡ e43c77de-f6af-11ea-3fdb-7577805c585f
begin
	b_comp_x = bx_x ⊗ bx_y
	b_comp_p = bp_x ⊗ bp_y
	
	T_xp = transform(b_comp_x, b_comp_p)
	T_px = T_xp' # transform(b_comp_p, b_comp_x)
end;

# ╔═╡ 16d904bc-f6b0-11ea-0e80-f7d0c24a214e
md"With these transformations we can now use sparse matrices in our operations. We define the operators through:"

# ╔═╡ 5870c362-f6b0-11ea-27fa-e3fc17a0fc03
begin
	px = momentum(bp_x)
	py = momentum(bp_y)
end;

# ╔═╡ 6ba7488e-f6b0-11ea-1d95-6d835ad9c4e8
md"This composite basis now allows us to write the kentix energy terms. We will do this using *lazy operations* as usual:"

# ╔═╡ a0347662-f6b0-11ea-1419-8bfc634f54ed
begin
	H_kx = LazyTensor(b_comp_p, [1, 2], [px^2 / 2, one(bp_y)])
	H_ky = LazyTensor(b_comp_p, [1, 2], [one(bp_x), py^2 / 2])
	
	H_kx_FFT = LazyProduct(T_xp, H_kx, T_px)
	H_ky_FFT = LazyProduct(T_xp, H_ky, T_px)
end;

# ╔═╡ 4d1baf3a-f6b1-11ea-3c7d-b90c9f713997
md"""
### Defining the Potential
The potential must have the same number of arguments equal to the number of bases, and the order of the function arguments must be the same as the order of the tensor product in constructing the composite basis.

We can use the `potentialoperator` function to easily construct a Gaussian potential on our composite basis:
"""

# ╔═╡ 8f6ced72-f6b1-11ea-0a8d-9720ce34a655
V = potentialoperator(
	b_comp_x, (x, y) -> (exp( - (x^2 + y^2) / 30.0 ))
);

# ╔═╡ c05cfc24-f6b1-11ea-18e6-592345831992
md"We can now define the full Hamiltonian for our setup:"

# ╔═╡ c9d3be6c-f6b1-11ea-19db-290f3211eaaa
H = LazySum(H_kx_FFT, H_ky_FFT, V);

# ╔═╡ ddc7bbf0-f6b1-11ea-0574-c7b85f463484
md"""
### Defining the Wavepacket
We define the initial conditions of our wavepacket, tensor-mulitply the basis vectors on the position spaces, and evolve the packet according to the Schrödinger equation:
"""

# ╔═╡ f04cc644-f6b1-11ea-19d9-0d83c22d45ae
begin
	x0 = -10
	y0 = -5
	p0_x = 1.5
	p0_y = 0.5
	σ = 2.0
	
	Ψx = gaussianstate(bx_x, x0, p0_x, σ)
	Ψy = gaussianstate(bx_y, y0, p0_y, σ)
	
	Ψ = Ψx ⊗ Ψy
	
	T = collect(0.0:0.1:24.0)
	tout, Ψt = timeevolution.schroedinger(T, Ψ, H)
end;

# ╔═╡ 5b7fc47a-f6b2-11ea-01fc-6f528bf4d44a
md"Now we can visualise our solution at different times."

# ╔═╡ 4176999a-f6b3-11ea-13ca-5d812008d825
md"First we build our data arrays:"

# ╔═╡ 71044bfe-f6b2-11ea-3085-53f44da1a702
begin
	density = [
		Array(
			transpose( 
				reshape((abs2.(i.data)), (N_points, N_points))
			)
		) for i ∈ Ψt
	]
	V_plot = Array(transpose(
		reshape(real.(diag(V.data)), (N_points, N_points))		
	))
	
	x_sample, y_sample = samplepoints(bx_x), samplepoints(bx_y)
end;

# ╔═╡ 5e50c93a-f6b5-11ea-3894-d5af50ce8ce9
@bind i html"<input type=range min=0 max=100>"

# ╔═╡ 691d09d0-f6b5-11ea-05f7-b9196eae9de3
begin
	ind = trunc(Int, (i/100) * (length(T) - 1) + 1)
	
	md"Examining ``t=``$(T[ind])"
end

# ╔═╡ 52674088-f6b3-11ea-0d25-394b26dd3e81
begin
	contour(
		x_sample, y_sample, 
		V_plot ./ maximum(V_plot), 
		fill=true, color=:blues, lw=0, cbar=false
	)
	contour!(
		x_sample, y_sample, 
		density[ind] ./ maximum(density[ind]), 
		color=:hot, alpha=0.1, fill=true
	)
end

# ╔═╡ Cell order:
# ╠═291c7d5a-f6af-11ea-33b4-67d04e0a16ec
# ╟─5e2299e6-f6af-11ea-3c09-8b6fe0aec753
# ╠═82d80ba2-f6af-11ea-133b-11cf0eb8bde2
# ╟─d113e17e-f6af-11ea-0376-2553b74c647e
# ╠═e43c77de-f6af-11ea-3fdb-7577805c585f
# ╟─16d904bc-f6b0-11ea-0e80-f7d0c24a214e
# ╠═5870c362-f6b0-11ea-27fa-e3fc17a0fc03
# ╟─6ba7488e-f6b0-11ea-1d95-6d835ad9c4e8
# ╠═a0347662-f6b0-11ea-1419-8bfc634f54ed
# ╟─4d1baf3a-f6b1-11ea-3c7d-b90c9f713997
# ╠═8f6ced72-f6b1-11ea-0a8d-9720ce34a655
# ╟─c05cfc24-f6b1-11ea-18e6-592345831992
# ╠═c9d3be6c-f6b1-11ea-19db-290f3211eaaa
# ╟─ddc7bbf0-f6b1-11ea-0574-c7b85f463484
# ╠═f04cc644-f6b1-11ea-19d9-0d83c22d45ae
# ╟─5b7fc47a-f6b2-11ea-01fc-6f528bf4d44a
# ╠═6cf6ad0e-f6b2-11ea-269a-678bed52852e
# ╟─4176999a-f6b3-11ea-13ca-5d812008d825
# ╠═71044bfe-f6b2-11ea-3085-53f44da1a702
# ╟─5e50c93a-f6b5-11ea-3894-d5af50ce8ce9
# ╟─691d09d0-f6b5-11ea-05f7-b9196eae9de3
# ╠═52674088-f6b3-11ea-0d25-394b26dd3e81
