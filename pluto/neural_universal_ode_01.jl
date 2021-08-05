### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ e610c9be-f5e4-11ea-1eca-d12d7e232c97
begin
	using Pkg; Pkg.activate("../..")
	Pkg.add("OrdinaryDiffEq")
	Pkg.add("Flux")
	Pkg.add("DiffEqFlux")
end

# ╔═╡ ab1fa440-f5e6-11ea-1331-7788f25321df
using OrdinaryDiffEq, Plots

# ╔═╡ 8d77e154-f5e7-11ea-2b89-fd0b39acfc70
using Flux, DiffEqFlux

# ╔═╡ 2345f746-f5e5-11ea-3892-29f10df5d54e
md"""
# Neural and Universal Ordinary Differential Equations: Part 01
Following along from notes by [Chris Rackauckas](https://mitmath.github.io/18S096SciML/lecture3/diffeq_ml).

We can relate neural networks to differential equations through the *neural differential equations*. To begin, consider the recurrent neural network:

$$x_{n+1} = x_n + \text{NN}(x_n).$$

In general, we can consider pulling out a multiplication factor $h$, such that $t_{n+1} = t_n + h$, and

$$x_{n+1} = x_n + h \cdot\text{NN}(x_n),$$

$$\frac{x_{n+1} - x_n}{h} = \text{NN}(x_n),$$

allowing us to rewrite the limit $h \rightarrow 0$

$$x^\prime = \text{NN}(x_n).$$
"""

# ╔═╡ 37126f6a-f5e6-11ea-0c15-25c47262abbd
md"""
## Training ODEs
Rackauckas has written exstensive notes on [training neural ODEs](https://mitmath.github.io/18337/lecture11/adjoints), which I will summarize in my own words here (todo). 

For simplicity, we will not concern ourselves with the details here, and instead use the default gradient calculation in `DiffEqFlux.jl`.

We will use the [`Flux.jl`](https://github.com/FluxML/Flux.jl) neural network library. Let us fist define a problem, namely the [Lotka-Volterra system](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations):
"""

# ╔═╡ b19175e2-f5e6-11ea-0792-7ff2d906559a
function lotka_volterra!(du, u, p, t)
	x, y = u
	α, β, δ, γ = p
	du[1] = α * x - β * x * y
	du[2] = - δ * y + γ * x * y
end

# ╔═╡ f69c5e90-f5e6-11ea-0e64-efbf763cc73c
md"Next, we solve this using a traditional `ODEProblem` approach:"

# ╔═╡ 029a2b66-f5e7-11ea-3295-cfc47e791b06
begin
	u0 = [1.0, 1.0]
	tspan = (0.0, 10.0)
	p = [1.5, 1.0, 3.0, 1.0]
	prob = ODEProblem(lotka_volterra!, u0, tspan, p)
	sol = solve(prob, Tsit5(), saveat=0.1)
	plot(sol)
end

# ╔═╡ 4062a76e-f5e7-11ea-06c3-ddb2ae8f40e3
md"""
### Building a neural network

Next, we define a *single layer* NN, that uses `concrete_solve` to return a $x(t)$ solution. The method `concrete_solve` is built on `DifferentialEquations.jl`'s `solve` method, which uses a backpropagation algorithm to determine the gradient:
"""

# ╔═╡ 9219095c-f5e7-11ea-1209-ebeff8f7bf57
begin 
 	test_data = Array(sol)
	p2 = [2.2, 1.0, 2.0, 0.4] # inital parameter vector
	
	function predict_adjoint() # single layer neural network
		Array(
			concrete_solve(
				prob, 
				Tsit5(), 
				u0,
				p2,
				saveat=0.1, 
				abstol=1e-6,
				reltol=1e-5
			)
		)
	end
end

# ╔═╡ c478e590-f5e7-11ea-33b8-efe407e1900e
md"Next we specify a loss function through which we can evaluate the model. Our aim is to ensure the Lotka-Volterra solution is a constant $x(t)=1$, so will define the loss as the distance from 1:"

# ╔═╡ ebf21e3e-f5e7-11ea-217f-e7b4247bd770
loss_adjoint() = sum(abs2, predict_adjoint() - test_data)

# ╔═╡ 1eacfbb4-f5e8-11ea-2e0f-f5d074f18db2
md"""
### Training the network
We will use ADAM optimization, with a callback to observe the training steps:
"""

# ╔═╡ 3473a06a-f5e8-11ea-2797-a574855b69ba
begin	
	iter = 0
	callback = function()
		global iter += 1
		if iter % 50 == 0
			@show loss_adjoint()
			
			# use remake to reconstruct problem with updated parameters
			pl = plot(
				solve(
					remake(
						prob,
						p=p2
					),
					Tsit5(),
					saveat=0.0:0.1:10.0
				),
				ylim=(0,8)
			)
			
			Flux.scatter!(
				pl, 
				0.0:0.1:10,
				test_data',
				markersize=2
			)
			
			push!(training_plots, pl)
		end
	end
	
	# ode with initial parameter values
	callback()
	
	data = Iterators.repeated((), 300) # 300 training cycles
	opt = ADAM(0.1)
	Flux.train!(loss_adjoint, Flux.params(p2), data, opt, cb=callback)
end

# ╔═╡ a3c97010-f5eb-11ea-236c-ad6a5f75af75
md"We can then examine the plots to see how the model fits to the data:"

# ╔═╡ fac89ad0-f5e9-11ea-0cce-ab48597fe5ee
iter

# ╔═╡ Cell order:
# ╠═e610c9be-f5e4-11ea-1eca-d12d7e232c97
# ╠═2345f746-f5e5-11ea-3892-29f10df5d54e
# ╠═37126f6a-f5e6-11ea-0c15-25c47262abbd
# ╠═ab1fa440-f5e6-11ea-1331-7788f25321df
# ╠═b19175e2-f5e6-11ea-0792-7ff2d906559a
# ╟─f69c5e90-f5e6-11ea-0e64-efbf763cc73c
# ╠═029a2b66-f5e7-11ea-3295-cfc47e791b06
# ╠═4062a76e-f5e7-11ea-06c3-ddb2ae8f40e3
# ╠═8d77e154-f5e7-11ea-2b89-fd0b39acfc70
# ╠═9219095c-f5e7-11ea-1209-ebeff8f7bf57
# ╠═c478e590-f5e7-11ea-33b8-efe407e1900e
# ╠═ebf21e3e-f5e7-11ea-217f-e7b4247bd770
# ╠═1eacfbb4-f5e8-11ea-2e0f-f5d074f18db2
# ╠═3473a06a-f5e8-11ea-2797-a574855b69ba
# ╟─a3c97010-f5eb-11ea-236c-ad6a5f75af75
# ╠═fac89ad0-f5e9-11ea-0cce-ab48597fe5ee
