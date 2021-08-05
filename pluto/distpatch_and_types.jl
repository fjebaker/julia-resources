### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 6948ca0a-f4f2-11ea-15a0-8bfa436afb37
begin
	import Pkg; Pkg.activate("..")
end

# ╔═╡ a85043c0-f4f2-11ea-3d9b-433ba7fca2db
begin
	Pkg.add("AbstractTrees")
	using AbstractTrees
	AbstractTrees.children(x::Type) = subtypes(x)
	print_tree(Number)
end

# ╔═╡ 37d6ff3a-f4ef-11ea-01a9-4debaf19a839
md"""
# Types and dispatch
Julia has *concrete* and *abstract* types; that is, objects, and structures. Abstract types cannot be instantiated, but are used to generate hierarchies in the language.
"""

# ╔═╡ 84f3e29c-f4ef-11ea-3b86-2184415fe3be
typeof(3)

# ╔═╡ 6c437662-f4f0-11ea-299f-1db881081c32
isconcretetype(Float64)

# ╔═╡ 70a75318-f4f0-11ea-2db4-b79bc7608e5b
isabstracttype(Number)

# ╔═╡ 5706721c-f4f1-11ea-1e3b-2fc6f6a1dba7
md"We can also investigate the type tree:"

# ╔═╡ 5f2810c2-f4f1-11ea-1f7a-b73cca9d2d29
supertype(Float64)

# ╔═╡ 60c9d602-f4f1-11ea-0a43-b7f8989f69f1
subtypes(AbstractFloat)

# ╔═╡ 668a7b16-f4f1-11ea-32fa-a1aaf402ec62
md"In Julia, everything is a subtype of `Any`, cf. Python with `Object`. We can investigate subtyping with the `<:` operator for abstract types, and with the `isa` keyword for objects:"

# ╔═╡ a5019f6e-f4f1-11ea-3c72-2bd32ba04811
Number <: Any

# ╔═╡ a84c5fc4-f4f1-11ea-027a-c7d612a8eea9
3.0 isa Float64

# ╔═╡ b3be7a04-f4f1-11ea-27ec-7ffaeb86008c
md"Let us define a function to investigate the type hierarchy:"

# ╔═╡ cb59597c-f4f1-11ea-32b8-93fef1986454
function string_supertypes(T)
	""" type tree from leaf """ 
	x = string(T)
	while T != Any
		T = supertype(T)
		x = string(x, " <: $T")
	end
	x
end

# ╔═╡ 203087a4-f4f2-11ea-1782-73174d7bfa94
string_supertypes(Float64)

# ╔═╡ 8e0ca0a0-f4f2-11ea-10de-b520703c09a4
md"There exists a tool for facilitating the exploration of these trees, known as `AbstractTrees`:"

# ╔═╡ dab6b2e2-f4f2-11ea-004b-ffc2fad94fd8
md"Unfortunately, Pluto.jl will print this to the terminal spawning Pluto. The output, however, will be along the lines of
```
Number
├─ Complex
└─ Real
   ├─ AbstractFloat
   │  ├─ BigFloat
   │  ├─ Float16
   │  ├─ Float32
   │  └─ Float64
   ├─ AbstractIrrational
   │  └─ Irrational
   ├─ Integer
   │  ├─ Bool
   │  ├─ Signed
   │  │  ├─ BigInt
   │  │  ├─ Int128
   │  │  ├─ Int16
   │  │  ├─ Int32
   │  │  ├─ Int64
   │  │  └─ Int8
   │  └─ Unsigned
   │     ├─ UInt128
   │     ├─ UInt16
   │     ├─ UInt32
   │     ├─ UInt64
   │     └─ UInt8
   └─ Rational
```"

# ╔═╡ 71faf4a6-f4f3-11ea-2f46-29ef0fef4d00
md"We see that *concrete* types are leaf points, and *abstract* types are the nodes defining branches."

# ╔═╡ 311f6c5e-f4f4-11ea-04f0-b9e20b8fb580
md"""
## Functions and Dispatch
Let us work by example: say we want a function that calculates the absolute value of a number, and we want this to work for `Real` and `Complex` numbers. We create a type specific implementation of the function:
"""

# ╔═╡ 72f212c6-f4f4-11ea-2eb5-e7d892c29209
begin
	f_abs(x::Real) = sign(x) * x
	f_abs(z::Complex) = sqrt(real(z * conj(z)))
end

# ╔═╡ 9c7e8160-f4f4-11ea-1c3f-4f5f76819380
f_abs(-6.66)

# ╔═╡ c80bf8c6-f4f4-11ea-1908-236004d58da2
md"We can investigate the available methods for a function with the builtin 	`methods`.For example, `f_abs`:"

# ╔═╡ be1a1118-f4f4-11ea-1892-a54f2d006c1c
methods(f_abs)

# ╔═╡ c108dbe2-f4f5-11ea-1e79-0fc0b33440d8
md"Or using `@which` decorator to discern which method is being called"

# ╔═╡ d32857e4-f4f5-11ea-1c5d-ff6d0116f32d
@which f_abs(-6.66)

# ╔═╡ f4a50edc-f4f5-11ea-0ecd-6145fdb91b81
md"This concept is known as *multiple dispatch*, and allows for Julia to optimise and pick the most specific method for a given input parameter. This also allows for modification of standard functions for a given need:"

# ╔═╡ 178c27ee-f4f6-11ea-1bc3-8d7eb3142818
begin 
	import Base: +
	+(x::String, y::String) = string(x, " + ", y)
	
	"Hello" + "World"
end

# ╔═╡ a2ace4fa-f4f4-11ea-0b28-8b7e60357e77
f_abs(1.0 + 1.0im)

# ╔═╡ 44540f1e-f4f6-11ea-1325-759adb383839
md"The benefit of such modification is that is quickly allows for very complicated behaviour to trickle down from the abstractions:"

# ╔═╡ 5ef2b422-f4f6-11ea-37e0-9360235e685d
sum(["Hello", "World", "Goodbye"])

# ╔═╡ 79c1a254-f4f6-11ea-1aa4-399f707797a7
md"It should be noted that there is no such thing as a *unique* method either, which can be a pitfall for declarations. Consider"

# ╔═╡ 23043f3e-f4f7-11ea-304e-2b88e5c82c6c
begin
	test_type(x::Int, y::Any) = println("int")
	test_type(x::Any, y::String) = println("string")
end

# ╔═╡ 5628e752-f4f7-11ea-0708-ed3d2be76563
md"If we try to execute this method with e.g. an integer and a string, we get a REPL error and possible solution"

# ╔═╡ 2a23e2e2-f4f7-11ea-352a-2fb5a47d1780
test_type(3, "test")

# ╔═╡ 632894c0-f4f7-11ea-15ae-e91efc12db3c
md"""
## Parametric types
There are some data structures which are parametric, such as the `Array` which accepts parameters for the type.
"""

# ╔═╡ 96d98a20-f4f7-11ea-22f7-c3180192d2e4
rand(2, 2)

# ╔═╡ a306402e-f4f7-11ea-3f66-d15ee66f837a
md"We see the above is a `Float64` array, with dimension 2. We can also create arrays of other types."

# ╔═╡ cfb8eb3c-f4f7-11ea-218b-df4a75ecf006
M = fill("Hello World", 2, 2)

# ╔═╡ d8c463d0-f4f7-11ea-1895-9bccc580ac0b
eltype(M)

# ╔═╡ e8d58bbe-f4f7-11ea-0603-511b1c824ecc
md"#### Aside: Matrices and Vectors
We can also use list comprehension to create a vector of matrices generically."

# ╔═╡ f9b3dc10-f4f7-11ea-3ace-41f63ad6536c
[rand(2, 2) for i in 1:3]

# ╔═╡ 0d38cedc-f4f8-11ea-325d-bdf0fd4d1262
md"However Julia has builtin aliases for these data structures:"

# ╔═╡ 266627d6-f4f8-11ea-3a00-bb92286e9d0d
Matrix{Float64} == Array{Float64, 2}

# ╔═╡ 37f0a558-f4f8-11ea-3c7d-c3b53555c5f8
Vector{Float64} == Array{Float64, 1}

# ╔═╡ 53cda9b0-f4f8-11ea-0723-e5123c1cb4ba
md"""
### Union types and `where` keyword
There are some less intuitive behaviours in Julia. For example:
"""

# ╔═╡ 6f73a8fe-f4f8-11ea-0680-ebbfb3004a29
Vector{Float64} <: Vector{Real}

# ╔═╡ 716249cc-f4f8-11ea-29a9-795a3652bc3c
md"despite"

# ╔═╡ 7b0c8c88-f4f8-11ea-2d4c-b1a360ce246d
Float64 <: Real

# ╔═╡ 84ff3852-f4f8-11ea-0d64-e97673e424c3
md"This is precisely because `Vector{Real}` is a concrete type, with elements `T <: Real`. Concrete types *do not have* subtypes."

# ╔═╡ e759262c-f4f9-11ea-3fea-ed03a1264820
md"Let us examine how we can use parameter types in function signatures:"

# ╔═╡ af1d880a-f4f8-11ea-3ad8-29b4118e9889
begin
	some_func(x::Integer) = typeof(x)
	some_func(x::T) where T = T
	some_func(x::Matrix{T}) where T <: Real = "I will be overwritten by the next definition"
	some_func(x::Matrix{<:Real}) = eltype(x)
end

# ╔═╡ ad1ca218-f4f9-11ea-2a82-4d97d8db2842
md"Or a more practical example of genericism:"

# ╔═╡ 311946da-f4f9-11ea-01f6-93ebca76fae8
begin
	discern_type(x::T, y::T) where T = "same type"
	discern_type(x, y) = "different types"
	some_val1 = 3; some_val2 = 3.0; some_val3 = 1.0
	dt_res1 = discern_type(some_val1, some_val2)
	dt_res2 = discern_type(some_val2, some_val3)
end;

# ╔═╡ 7a5776a0-f4f9-11ea-2263-5faee9faf2da
md"Indeed, we see that $some_val1 and $some_val2 are of $dt_res1, and that $some_val2 and $some_val3 are of $dt_res2."

# ╔═╡ d68d67ea-f4f9-11ea-2d98-4f5fcef48cd0
md"There also exist union types:"

# ╔═╡ c0d9ccba-f501-11ea-2c05-05142b7fe2b0
Union{Float64, Int32} <: Real

# ╔═╡ 3ef45b5a-f503-11ea-08f7-775dbc5de4a4
md"Finally, it is worth noting the existence of [bit based typing](https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/#man-bits-types-1), which includes methods such as
- `isbits(x)`
- `isbitytype(T)`
"

# ╔═╡ Cell order:
# ╠═6948ca0a-f4f2-11ea-15a0-8bfa436afb37
# ╟─37d6ff3a-f4ef-11ea-01a9-4debaf19a839
# ╠═84f3e29c-f4ef-11ea-3b86-2184415fe3be
# ╠═6c437662-f4f0-11ea-299f-1db881081c32
# ╠═70a75318-f4f0-11ea-2db4-b79bc7608e5b
# ╟─5706721c-f4f1-11ea-1e3b-2fc6f6a1dba7
# ╠═5f2810c2-f4f1-11ea-1f7a-b73cca9d2d29
# ╠═60c9d602-f4f1-11ea-0a43-b7f8989f69f1
# ╟─668a7b16-f4f1-11ea-32fa-a1aaf402ec62
# ╠═a5019f6e-f4f1-11ea-3c72-2bd32ba04811
# ╠═a84c5fc4-f4f1-11ea-027a-c7d612a8eea9
# ╟─b3be7a04-f4f1-11ea-27ec-7ffaeb86008c
# ╠═cb59597c-f4f1-11ea-32b8-93fef1986454
# ╠═203087a4-f4f2-11ea-1782-73174d7bfa94
# ╟─8e0ca0a0-f4f2-11ea-10de-b520703c09a4
# ╠═a85043c0-f4f2-11ea-3d9b-433ba7fca2db
# ╟─dab6b2e2-f4f2-11ea-004b-ffc2fad94fd8
# ╟─71faf4a6-f4f3-11ea-2f46-29ef0fef4d00
# ╟─311f6c5e-f4f4-11ea-04f0-b9e20b8fb580
# ╠═72f212c6-f4f4-11ea-2eb5-e7d892c29209
# ╠═9c7e8160-f4f4-11ea-1c3f-4f5f76819380
# ╠═a2ace4fa-f4f4-11ea-0b28-8b7e60357e77
# ╟─c80bf8c6-f4f4-11ea-1908-236004d58da2
# ╠═be1a1118-f4f4-11ea-1892-a54f2d006c1c
# ╟─c108dbe2-f4f5-11ea-1e79-0fc0b33440d8
# ╠═d32857e4-f4f5-11ea-1c5d-ff6d0116f32d
# ╟─f4a50edc-f4f5-11ea-0ecd-6145fdb91b81
# ╠═178c27ee-f4f6-11ea-1bc3-8d7eb3142818
# ╟─44540f1e-f4f6-11ea-1325-759adb383839
# ╠═5ef2b422-f4f6-11ea-37e0-9360235e685d
# ╠═79c1a254-f4f6-11ea-1aa4-399f707797a7
# ╠═23043f3e-f4f7-11ea-304e-2b88e5c82c6c
# ╟─5628e752-f4f7-11ea-0708-ed3d2be76563
# ╠═2a23e2e2-f4f7-11ea-352a-2fb5a47d1780
# ╟─632894c0-f4f7-11ea-15ae-e91efc12db3c
# ╠═96d98a20-f4f7-11ea-22f7-c3180192d2e4
# ╟─a306402e-f4f7-11ea-3f66-d15ee66f837a
# ╠═cfb8eb3c-f4f7-11ea-218b-df4a75ecf006
# ╠═d8c463d0-f4f7-11ea-1895-9bccc580ac0b
# ╟─e8d58bbe-f4f7-11ea-0603-511b1c824ecc
# ╠═f9b3dc10-f4f7-11ea-3ace-41f63ad6536c
# ╟─0d38cedc-f4f8-11ea-325d-bdf0fd4d1262
# ╠═266627d6-f4f8-11ea-3a00-bb92286e9d0d
# ╠═37f0a558-f4f8-11ea-3c7d-c3b53555c5f8
# ╟─53cda9b0-f4f8-11ea-0723-e5123c1cb4ba
# ╠═6f73a8fe-f4f8-11ea-0680-ebbfb3004a29
# ╟─716249cc-f4f8-11ea-29a9-795a3652bc3c
# ╠═7b0c8c88-f4f8-11ea-2d4c-b1a360ce246d
# ╟─84ff3852-f4f8-11ea-0d64-e97673e424c3
# ╟─e759262c-f4f9-11ea-3fea-ed03a1264820
# ╠═af1d880a-f4f8-11ea-3ad8-29b4118e9889
# ╟─ad1ca218-f4f9-11ea-2a82-4d97d8db2842
# ╠═311946da-f4f9-11ea-01f6-93ebca76fae8
# ╟─7a5776a0-f4f9-11ea-2263-5faee9faf2da
# ╟─d68d67ea-f4f9-11ea-2d98-4f5fcef48cd0
# ╠═c0d9ccba-f501-11ea-2c05-05142b7fe2b0
# ╟─3ef45b5a-f503-11ea-08f7-775dbc5de4a4
