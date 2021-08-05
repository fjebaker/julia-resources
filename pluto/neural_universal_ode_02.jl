### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 30455e5c-f5ed-11ea-28a1-bd4a99eea9b1
begin
	using Pkg; Pkg.activate("../..")
	Pkg.add("OrdinaryDiffEq")
	Pkg.add("Flux")
	Pkg.add("DiffEqFlux")
end

# ╔═╡ 1c9c164c-f5ee-11ea-2b18-71c14a1eb2e1
using OrdinaryDiffEq, Flux, DiffEqFlux

# ╔═╡ af5c6414-f5ee-11ea-1fe6-7b715a731a96
begin 
	using Plots; gr()
end

# ╔═╡ 427779e8-f5ed-11ea-10ef-5d3359dfc1fb
md"""
# Neural and Universal Ordinary Differential Equations: Part 02
Following along from notes by [Chris Rackauckas](https://mitmath.github.io/18S096SciML/lecture3/diffeq_ml).

In the previous notebook is documented how we can use a neural network to solve a parameter problem relating to an ODE -- we essentially trained a neural network on an ODE model, and attempted to find a best fit for the model parameters.
"""

# ╔═╡ 4b96ebb2-f5ed-11ea-010f-6dad229c6e64
md"""
## Defining and Training Neural ODEs

Defining a neural ODE is the same as defining a parameterized differential equation, except here the parameterized ODE is a NN. 

Consider the following example; we wish to match the following data:
"""

# ╔═╡ b1c2b29a-f5ed-11ea-0711-a327bfc701be
begin
	u0 = Float32[2.; 0.]
	datasize = 30
	tspan = (0.0f0, 1.5f0)
	
	function trueODE!(du, u, p, t)
		true_A = [-0.1 2.0; -2.0 -0.1]
		du .= ((u.^3)'true_A)'
	end
	
	t = range(tspan[1], tspan[2], length=datasize)
	prob = ODEProblem(trueODE!, u0, tspan)
	ode_data = Array(solve(prob, Tsit5(), saveat=t))
end

# ╔═╡ 36ef4d70-f661-11ea-3e3f-678e61ea6ca9
md"Let us quickly visualise this data:"

# ╔═╡ 762427ec-f65f-11ea-20aa-37a17b5d4ab5
plot(ode_data')

# ╔═╡ 8b50706a-f5ee-11ea-25bf-bdb7d40195ea
md"We will use a so-called *knowledge-infused approach*; that is to say, we assume that we knew the ODE had cubic behaviour. We can attempt to encode that physical information into a dense NN as:"

# ╔═╡ 4a6bc1f8-f661-11ea-3120-452194d8b281
dudt = Chain(
	x -> x.^3,
	Dense(2, 50, tanh),
	Dense(50, 2)
)

# ╔═╡ 7347598e-f661-11ea-0e94-69d25745b5ef
md"""
To train this network we will make use of `Flux.destructure` and `Flux.restructure`, allowing us to take the parameters out of a NN into a vector, and similarly rebuild a NN from a parameter vector. 

Using these, we define the ODE:
"""

# ╔═╡ 9f350fd2-f661-11ea-2443-679ed3be0e58
begin 
	p, re = Flux.destructure(dudt)
	dudt2_(u, p, t) = re(p)(u)
	prob2 = ODEProblem(dudt2_, u0, tspan, p)
end

# ╔═╡ c7cf8152-f661-11ea-23f9-91b6426b7731
md"""
This is equivalent to

$$u^\prime = \text{NN}(u)$$

where the parameters are the parameters of the NN. We then use the same structure as before to train the network to reconstruct the ODE:
"""

# ╔═╡ fd3cd57e-f661-11ea-31e2-f39184ce2d4d
begin
	function predict_n_ode()
		Array(concrete_solve(prob2, Tsit5(), u0, p, saveat=t))
	end
	loss_n_ode() = sum(abs2, ode_data .- predict_n_ode())
	
	data = Iterators.repeated((), 300)
	opt = ADAM(0.1)
	
	training_plots = []
	iter = 0
	callback = function()
		global iter += 1
		if iter % 50 == 0
			@show loss_n_ode()
			
			cur_pred = predict_n_ode()
			pl = scatter(t, ode_data[1,:], label="data")
			scatter!(pl, t, cur_pred[1,:], label="prediction")
			push!(training_plots, plot(pl))
		end
	end
	
	ps = Flux.params(p)
	Flux.train!(loss_n_ode, ps, data, opt, cb=callback)
end

# ╔═╡ a0ec1204-f662-11ea-2b38-6fcc370e2eb3
md"We can then view our plots:"

# ╔═╡ a704eb32-f662-11ea-222e-0bd5db9575aa
training_plots

# ╔═╡ d8247214-f662-11ea-2882-87ed9a6e240c
md"And true enough, we were able to fit the parameters of the neural network (can we extract true A from this?)."

# ╔═╡ 3d026146-f663-11ea-30be-d13962d99962
md"""
## Augmented Neural ODE
Not every function can be represented by an ODE. Specifically, some

$$u(t) : \mathbb{R} \rightarrow \mathbb{R}^n.$$

This is because the *flow* of the ODE must be uniquely defined at each point; the above mapping would have at least *two directions of flow* for a given point $u_i$, and thus at least *two solutions* in phase space to the general ODE

$$u^\prime = f(u, p, t),$$

using the convention $u(0) = u_i$.

We can rectify this by introducing additional degrees of freedom, to ensure that the ODE does not overlap and become degenerate. This is the so-called *augmented neural ODE*. 

This can be built using the following prescription:
- add a fake state to the ODE which is 0 everywhere
- allow this extra dimension to *bump* around to let the function become a [universal approximator](https://en.wikipedia.org/wiki/Universal_approximation_theorem)

In Julia, this is:
"""

# ╔═╡ 4823ccb2-f664-11ea-0bdd-ad1156429c47
md"""
```julia
dudt = Chain(...)
p, re = Flux.destructure(dudt)

dudt_(u, p, t) = re(p)(u)
prob = ODEProblem(dudt_, [u0,0f0], tspan, p)

augmented_data = vcat(
	ode_data,
	zeros(
		1,
		size(ode_data, 2)
	)
)
```
"""

# ╔═╡ Cell order:
# ╠═30455e5c-f5ed-11ea-28a1-bd4a99eea9b1
# ╟─427779e8-f5ed-11ea-10ef-5d3359dfc1fb
# ╟─4b96ebb2-f5ed-11ea-010f-6dad229c6e64
# ╠═1c9c164c-f5ee-11ea-2b18-71c14a1eb2e1
# ╠═b1c2b29a-f5ed-11ea-0711-a327bfc701be
# ╟─36ef4d70-f661-11ea-3e3f-678e61ea6ca9
# ╠═af5c6414-f5ee-11ea-1fe6-7b715a731a96
# ╠═762427ec-f65f-11ea-20aa-37a17b5d4ab5
# ╟─8b50706a-f5ee-11ea-25bf-bdb7d40195ea
# ╠═4a6bc1f8-f661-11ea-3120-452194d8b281
# ╟─7347598e-f661-11ea-0e94-69d25745b5ef
# ╠═9f350fd2-f661-11ea-2443-679ed3be0e58
# ╟─c7cf8152-f661-11ea-23f9-91b6426b7731
# ╠═fd3cd57e-f661-11ea-31e2-f39184ce2d4d
# ╟─a0ec1204-f662-11ea-2b38-6fcc370e2eb3
# ╠═a704eb32-f662-11ea-222e-0bd5db9575aa
# ╟─d8247214-f662-11ea-2882-87ed9a6e240c
# ╟─3d026146-f663-11ea-30be-d13962d99962
# ╟─4823ccb2-f664-11ea-0bdd-ad1156429c47
