### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ ad868e24-f5a8-11ea-11b8-09ebd833f68f
begin
	using Pkg; Pkg.activate("../..")
	Pkg.add("DifferentialEquations")
	Pkg.add("Plots")
end

# ╔═╡ 6289ffda-f5a7-11ea-1cc5-4f5a29a8e469
md"""
# Indroduction to `DifferentialEquations.jl`

Following from the [official tutorials](https://tutorials.sciml.ai/html/introduction/01-ode_introduction.html).
We define an ordinary differential equation (ODE) as 

$$u^\prime = f(u, p, t).$$
"""

# ╔═╡ 5269349e-f5a8-11ea-3a04-798828148c6d
md"## Exponential Growth
As an example, we can consider exponential growth

$$u^\prime = au,$$ 

with $u(t_0) = u_0$ . For an increasing rate of 98% per year, we can define the general function in Julia as
"

# ╔═╡ 86f896dc-f5a8-11ea-08fd-43d5706a1ac9
f(u, p, t) = 0.98u

# ╔═╡ de4f1f32-f5a8-11ea-00bd-1f681e5d73cf
begin
	using DifferentialEquations
	u0 = 1.0
	tspan = (0.0,1.0)
	prob = ODEProblem(f, u0, tspan)
end;

# ╔═╡ 90773196-f5a8-11ea-2ccd-ed537dc35f4c
md"If we want to solve this model from t=0.0 to t=1.0, we use the `ODEProblem` method available in `DifferentialEquations.jl`:"

# ╔═╡ fcf333ce-f5a8-11ea-1a42-c985bed9f141
md"With the problem define, we call `solve` to find a solution:"

# ╔═╡ 86825bfa-f5aa-11ea-2094-c13c39fdd4f4
sol = solve(prob);

# ╔═╡ 521488e4-f5b0-11ea-2746-2bfbadd217ee
begin
	using Plots; plotly()
	plot(sol, xaxis="Time(t)", yaxis="u(t)", label="ODE")
	plot!(sol.t, t-> 1.0 * exp(0.98t), ls=:dash, label="True Solution")
end

# ╔═╡ 06bce6c8-f5b0-11ea-1afb-3bcc253f865d
md"*NB:* Pluto has issue displaying this result in the notebook."

# ╔═╡ 165a0778-f5b0-11ea-2776-516690646892
md"### Examining the solution

There is an entire page dedicated to [solution handling](https://diffeq.sciml.ai/dev/basics/solution/) in the SciML docs, which describes what may be done with the solution instance. For example, we may plot the solution:"

# ╔═╡ 8f0b3a96-f5b2-11ea-108e-6d084edbd1c0
md"The solution object itself, if printed to the terminal, contains useful information:

```
retcode: Success
Interpolation: automatic order switching interpolation
t: 5-element Array{Float64,1}:
 0.0
 0.10042494449239292
 0.35218603951893646
 0.6934436028208104
 1.0
u: 5-element Array{Float64,1}:
 1.0
 1.1034222047865465
 1.4121908848175448
 1.9730384275622996
 2.664456142481451
```

We see that the solution has *order changing interpolation*. For non-stiff equations, this is a continous function of order 4 -- we can get an arbitrary point by calling the solution:"

# ╔═╡ c6ae6964-f5b2-11ea-3522-1dbdd4f2710a
sol(0.45)

# ╔═╡ dcba67da-f5b2-11ea-31cf-3f3a5958757f
md"""
## Solve Control

There are a myriad of solving algorithms, listed under [common solver options](https://diffeq.sciml.ai/dev/basics/common_solver_opts/). Common flags to use are `abstol` and `reltol`, which define how the time stepping engine should handle precisions. In general, `abstol` is the accuracy used when `u` is around zero.

A rule of thumb is that the total solution accuracy is approx 1-2 digits less that the `reltol`.

Let us solve under stricter tolerances:
"""

# ╔═╡ 69e3fbf8-f5b3-11ea-306c-25d8756241bc
begin
	sol2 = solve(prob, abstol=1e-8, reltol=1e-8)
	plot(sol2, xaxis="Time(t)", yaxis="u(t)", label="ODE 2")
	plot!(sol2.t, t-> 1.0 * exp(0.98t), ls=:dash, label="True Solution")
end

# ╔═╡ 77818a78-f5b3-11ea-36b0-c1ec5c82e27e
md"We see that the solution now much more closely matches the true solution. The tradeoff for this, however, is that previously $ t $ and $ u $ were $(length(sol.t)) - element arrays, however now are $(length(sol2.t)) - element arrays. This indicates a greater number of steps had to be taken to populate the solution.

We can also define a step delta through the `saveat` keyword, which accepts either an array of explicit steps, or a float for the delta.

If we need to reduce the amount of saving to the array, we can also pass the `dense=false` flag to the `solve` method, or save only the time span $(tspan[1]) and $(tspan[2]) values with `save_everystep=false`.

More delicate and advanced behaviour can be obtained by using a `SavingCallback`, documented in the [callback library](https://diffeq.sciml.ai/dev/features/callback_library/#saving_callback-1)."

# ╔═╡ 670ccff2-f5b5-11ea-36db-9be2acf0d9f9
md"We can examine how a specific `solve` call perfomed thorugh the `destats` property:"

# ╔═╡ 87c8d29a-f5b5-11ea-3b91-094d8af8abbc
sol.destats

# ╔═╡ 8a4d8bbe-f5b5-11ea-0c97-8d691975401a
sol2.destats

# ╔═╡ 916bf0ac-f5b5-11ea-0f99-276a3d38b6bc
md"We see more clearly the increase in computation required for stricter tolerances first hand."

# ╔═╡ 100a4514-f5b4-11ea-1501-8de4ecf66377
md"""
## Choosing Solver Algorithms

When invoking `solve`, the library makes a guess as to which algorithm is best suited to the defined problem. However, `DifferentialEquations.jl` provides a suite for more defined control. This will be examined in a later notebook here (todo).

As a taster, we will discuss the *stiffness* of a model. Stiffness is characterized by a Jacobian $ J $ with large eigenvalues. In more practical terms, a quote by Lawrence Shampine of MATLAB ODE Suite
> If the standard algorithms are slow, then it's stiff.

We can provide flags to our solver to suggest what we belive the model's behaviour is like, and aid the solver in picking an effective algorithm:
"""

# ╔═╡ 1b81911c-f5b5-11ea-0084-951608e48d2c
sol3 = solve(prob, alg_hints=[:stiff]);

# ╔═╡ 3803e34e-f5b5-11ea-3db1-91b5c6ffe612
md"Printing this to terminal yields a different interpolation type, namely
```
Interpolation: specialized 3rd order \"free\" stiffness-aware interpolation
```

Examining the `destats` property, we see new properties and computations which were omitted in the non-stiff examples.
"

# ╔═╡ 448ac542-f5b5-11ea-338b-79ac28aa0061
sol3.destats

# ╔═╡ 22736262-f5b6-11ea-284d-fb14a797a755
md"""
## System of ODEs: Lorenz Attractor
To illustrate an example of a system of ODEs being solved with `DifferentialEquations.jl`, we consider the set of equations:

$$\frac{dx}{dt} = \sigma(y-x),$$

$$\frac{dy}{dt} = x(\rho - z)-y,$$

$$\frac{dy}{dt} = xy- \beta z,$$

with adjustable parameters $\sigma$, $\rho$, $\beta$. We pass there via a `p` array into our Julia function. We define the above system of ODEs through:
"""

# ╔═╡ 7c694626-f5b7-11ea-00ee-7588b23c824a
function lorenz!(du, u, p, t)
	σ, ρ, β = p
	du[1] = σ * (u[2] - u[1])
	du[2] = u[1] * (ρ - u[3]) - u[2]
	du[3] = u[1] * u[2] - β*u[3]
end

# ╔═╡ bd008104-f5b7-11ea-24e9-1fcf7cb41549
md"*NB:* we use the in-place format function, as it behaves a little faster for systems of ODEs, and hence our function fingerprint includes the `du` parameter."

# ╔═╡ 16244112-f5b8-11ea-3e0a-7770ab2cf628
begin
	p = (10,28,8/3)
	la_tspan = (0.0, 100.0)
	la_u0 = [1.0,0.0,0.0]
	la_prob = ODEProblem(lorenz!, la_u0, la_tspan, p)
end

# ╔═╡ d8b5134c-f5b7-11ea-3361-73e921612ec9
md"We now provide parameter definitions, generate the time span, and construct the problem. We use inital values ``\sigma=`` $(p[1]), ``\rho=`` $(p[2]), ``\beta=`` $(round(p[3], digits=3)), over a time span from $(la_tspan[1]) to $(la_tspan[2])."

# ╔═╡ a6d89258-f5b8-11ea-301e-9d334303ea40
md"We solve simply with:"

# ╔═╡ accdd182-f5b8-11ea-191a-c5274a31e52f
la_sol = solve(la_prob);

# ╔═╡ b8910818-f5b8-11ea-285d-61e386034003
md"For standard arguments, the solution will again be an order-switching interpolation, but now we have $(length(la_sol.t)) elements in our solution arrays, demonstrating the complexity of the solution. Also the indexing of `sol` now returns a three element array, corresponding to ``x``, ``y``, and ``z``. As such, the index `sol[j, i]` yields the value of the `j`th variable, and time `i`. We can obtain a matrix of the solution by casting the solution object to an `Array`, which preserves the `[j, i]` indexing."

# ╔═╡ 33274542-f5b9-11ea-3713-6513aa254018
md"Directly polotting the solution would yield `u1,` `u2,` `u3` as a function of `t`, however these individual lines do not have much meaning to us. Instead, we want to create a plot with each axis acting as a given variable. E.g., for `1` against `2` against `3` we use:"

# ╔═╡ 6596fa70-f5b9-11ea-314b-a7305a5a68af
plot(la_sol, vars=(1, 2, 3))

# ╔═╡ 77bb08d6-f5b9-11ea-3cf9-f78cb8dc798d
md"This plot by default takes advantage of interpolation, however we can also turn this off:"

# ╔═╡ 86d1e768-f5b9-11ea-1a37-7d0ba0583fd4
plot(la_sol, vars=(1, 2, 3), denseplot=false)

# ╔═╡ 94417820-f5b9-11ea-2aa4-398c4c4089f9
md"*NB:* in `vars`, index 0 is time. Thus, we could analyse a single dimension with:"

# ╔═╡ a2ffa51c-f5b9-11ea-1f5c-57ff76af802d
plot(la_sol, vars=(0, 2))

# ╔═╡ bf8ebbdc-f5b9-11ea-13ee-b5fc8cda5834
md"""
## A note on Types
`DifferentialEquations.jl` determines output types based off of the input types. In the above examples, we have used `Float64` as inputs, and thus also as anticipated outputs. As a short example, let us solve an ODE with matrix input:

"""

# ╔═╡ eba72d12-f5b9-11ea-0ff0-a9844538421c
A  = [1. 0  0 -5
      4 -2  4 -3
     -4  0  0  1
      5 -2  2  3];

# ╔═╡ f112031c-f5b9-11ea-17f9-e15034f0a6c5
begin
	m_u0 = rand(4, 2)
	m_tspan = (0.0, 1.0)
	m_f(u, p, t) = A*u
	m_prob = ODEProblem(m_f, m_u0, m_tspan)
	m_sol = solve(prob)
end;

# ╔═╡ Cell order:
# ╠═ad868e24-f5a8-11ea-11b8-09ebd833f68f
# ╟─6289ffda-f5a7-11ea-1cc5-4f5a29a8e469
# ╟─5269349e-f5a8-11ea-3a04-798828148c6d
# ╠═86f896dc-f5a8-11ea-08fd-43d5706a1ac9
# ╟─90773196-f5a8-11ea-2ccd-ed537dc35f4c
# ╠═de4f1f32-f5a8-11ea-00bd-1f681e5d73cf
# ╟─fcf333ce-f5a8-11ea-1a42-c985bed9f141
# ╠═86825bfa-f5aa-11ea-2094-c13c39fdd4f4
# ╟─06bce6c8-f5b0-11ea-1afb-3bcc253f865d
# ╟─165a0778-f5b0-11ea-2776-516690646892
# ╠═521488e4-f5b0-11ea-2746-2bfbadd217ee
# ╟─8f0b3a96-f5b2-11ea-108e-6d084edbd1c0
# ╠═c6ae6964-f5b2-11ea-3522-1dbdd4f2710a
# ╟─dcba67da-f5b2-11ea-31cf-3f3a5958757f
# ╠═69e3fbf8-f5b3-11ea-306c-25d8756241bc
# ╟─77818a78-f5b3-11ea-36b0-c1ec5c82e27e
# ╟─670ccff2-f5b5-11ea-36db-9be2acf0d9f9
# ╠═87c8d29a-f5b5-11ea-3b91-094d8af8abbc
# ╠═8a4d8bbe-f5b5-11ea-0c97-8d691975401a
# ╟─916bf0ac-f5b5-11ea-0f99-276a3d38b6bc
# ╟─100a4514-f5b4-11ea-1501-8de4ecf66377
# ╠═1b81911c-f5b5-11ea-0084-951608e48d2c
# ╟─3803e34e-f5b5-11ea-3db1-91b5c6ffe612
# ╠═448ac542-f5b5-11ea-338b-79ac28aa0061
# ╟─22736262-f5b6-11ea-284d-fb14a797a755
# ╠═7c694626-f5b7-11ea-00ee-7588b23c824a
# ╟─bd008104-f5b7-11ea-24e9-1fcf7cb41549
# ╟─d8b5134c-f5b7-11ea-3361-73e921612ec9
# ╠═16244112-f5b8-11ea-3e0a-7770ab2cf628
# ╟─a6d89258-f5b8-11ea-301e-9d334303ea40
# ╠═accdd182-f5b8-11ea-191a-c5274a31e52f
# ╟─b8910818-f5b8-11ea-285d-61e386034003
# ╟─33274542-f5b9-11ea-3713-6513aa254018
# ╠═6596fa70-f5b9-11ea-314b-a7305a5a68af
# ╟─77bb08d6-f5b9-11ea-3cf9-f78cb8dc798d
# ╠═86d1e768-f5b9-11ea-1a37-7d0ba0583fd4
# ╟─94417820-f5b9-11ea-2aa4-398c4c4089f9
# ╠═a2ffa51c-f5b9-11ea-1f5c-57ff76af802d
# ╟─bf8ebbdc-f5b9-11ea-13ee-b5fc8cda5834
# ╠═eba72d12-f5b9-11ea-0ff0-a9844538421c
# ╠═f112031c-f5b9-11ea-17f9-e15034f0a6c5
