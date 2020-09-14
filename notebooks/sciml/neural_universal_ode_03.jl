### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 26e049d6-f67d-11ea-2312-ebb78b7cf2d4
begin
	using Pkg; Pkg.activate("../..")
	Pkg.add("OrdinaryDiffEq")
	Pkg.add("Flux")
	Pkg.add("DiffEqFlux")
end

# ╔═╡ b3c4724c-f67e-11ea-1e9a-0b038c01f60e
using OrdinaryDiffEq, Flux, DiffEqFlux

# ╔═╡ 109bee32-f682-11ea-2790-c35e5d64d856
using Plots

# ╔═╡ 154addbc-f67d-11ea-344e-7bc98e04a82b
md"""
# Neural and Universal Ordinary Differential Equations: Part 02
Following along from notes by [Chris Rackauckas](https://mitmath.github.io/18S096SciML/lecture3/diffeq_ml).

Part 02 examined how we can explicitly create a neural network, and train it on an ODE, imbued with a-priori knowledge of the system, but without any consideration to the format of the parameters -- indeed, the parameter vector was merely an unfolded version of the network itself.
"""

# ╔═╡ 6b824314-f67d-11ea-38b4-2d1c105a3aa8
md"""
## Universal ODEs
Following from *knowledge embedded* NN, we might ask if we know something about the DE, could we use that in the DE definition itself? This introduces the idea of the *universal* DE; a differential equation embedding universal approximators in its definition. This allows the NN to learn arbitrary functions and pieces of the overall DE.

Let us examine this by example: we have a two-state system, and know that the second state is defined by a linear ODE. We note

$$x^\prime = \text{NN}(x, y),$$

$$y^\prime = p_1 x + p_2 y.$$

Expressed in Julia:
"""

# ╔═╡ 1056c2dc-f67e-11ea-2568-af76b7c0e392
begin
	u0 = [0.8, 0.8]
	tspan = (0.0f0, 25.0f0)
	
	ann = Chain(
		Dense(2, 10, tanh),
		Dense(10, 1)
	)
	
	p1, re = Flux.destructure(ann)
	p2 = [-2.0, 1.1]
	p3 = [p1;p2]
	ps = Flux.params(p3)
	
	function dudt_!(du, u, p, t)
		x, y = u
		du[1] = re(p[1:41])(u)[1]
		du[2] = p[end - 1] * y + p[end] * x # last two are p2
	end
	
	prob = ODEProblem(dudt_!, u0, tspan, p3)
	# concrete_solve is apparently deprecated 
	# https://www.juliabloggers.com/sciml-ecosystem-update-auto-parallelism-and-component-based-modeling/
	solve(prob, Tsit5(), abstol=1e-8, reltol=1e-6)
end

# ╔═╡ 09efef8a-f681-11ea-203c-f139f0e8eb96
md"We could now train our system to be stable at 1 as follows:"

# ╔═╡ 2c6eec6e-f681-11ea-07cd-ffa4230438fe
begin
	function predict_adjoint()
		Array(solve(
				prob, Tsit5(), saveat=0.0:0.1:25.0
			)
		)
	end
	
	loss_adjoint() = sum(abs2, predict_adjoint() .- 1)
	@show loss_adjoint()
	
	data = Iterators.repeated((), 300)
	opt = ADAM(0.01)
	
	training_plots = []
	iter = 0
	callback = function()
		global iter += 1
		if iter % 50 == 0
			@show loss_adjoint()
			push!(
				training_plots,
				plot(solve(
						remake(prob, p=p3, u0=u0),
						Tsit5(),
						saveat=0.1
					),
					ylim=(0,6)
				)
			)
		end
	end
	
	Flux.train!(loss_adjoint, ps, data, opt, cb=callback)
end

# ╔═╡ fc79c1cc-f681-11ea-3919-b154193db0c0
md"And view our training plots:"

# ╔═╡ 010bf944-f682-11ea-0c2f-3f22bf33c5aa
training_plots

# ╔═╡ Cell order:
# ╠═26e049d6-f67d-11ea-2312-ebb78b7cf2d4
# ╟─154addbc-f67d-11ea-344e-7bc98e04a82b
# ╟─6b824314-f67d-11ea-38b4-2d1c105a3aa8
# ╠═b3c4724c-f67e-11ea-1e9a-0b038c01f60e
# ╠═1056c2dc-f67e-11ea-2568-af76b7c0e392
# ╟─09efef8a-f681-11ea-203c-f139f0e8eb96
# ╠═109bee32-f682-11ea-2790-c35e5d64d856
# ╠═2c6eec6e-f681-11ea-07cd-ffa4230438fe
# ╟─fc79c1cc-f681-11ea-3919-b154193db0c0
# ╠═010bf944-f682-11ea-0c2f-3f22bf33c5aa
