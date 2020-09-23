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

# ╔═╡ b213700c-fd35-11ea-1552-1794da9334d6
begin
	using Pkg; 
	Pkg.activate("../..")
	Pkg.add.(("CSV", "DataFrames", "Plots", "ZipFile", "Shapefile"))
	#Pkg.add.(["CSV", "DataFrames"])
	
	using CSV, DataFrames, Plots, PlutoUI, Statistics, Dates, ZipFile, Shapefile
	gr()
	
	try
		mkdir("./data")
	catch e
		@show e
	end
	
	md"Environment initialized..."
end

# ╔═╡ 430098da-fd34-11ea-064b-e9954e55543b
begin
	url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv";
	
	download(url, "data/covid_data.csv")
	csv_data = CSV.File("data/covid_data.csv");
	raw_data = DataFrame(csv_data)
	md"Data loaded..."
end

# ╔═╡ ee2149aa-fd35-11ea-350f-41ffff1bc8dd
begin
	df = rename(raw_data, 1=> "province", 2 => "country", 3 => "lat", 4 => "long")
	head(df)
	md"Data formatted..."
end

# ╔═╡ 75e5d342-fda4-11ea-07b4-0b03effe68d7
md"# Covid Growth Map"

# ╔═╡ 7ed3b6b0-fda5-11ea-2317-4d2cc3c31ee9
begin
	# get data
    zipfile = download("https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip")

    r = ZipFile.Reader(zipfile);
    for f in r.files
        println("Filename: $(f.name)")
        open(joinpath("data", f.name), "w") do io
            write(io, read(f))
        end
    end
    close(r)
	
	shp_countries = Shapefile.shapes(
		Shapefile.Table("data/ne_110m_admin_0_countries.shp")
	)
	
	md"Fetched map files 🗺 "
end

# ╔═╡ fb1f2e8a-fda6-11ea-07bf-e9ce7424de87
@bind clock_day Clock(0.5)

# ╔═╡ 36578262-fda8-11ea-3f53-375a4d1e402d
md"# World Data Normalized"

# ╔═╡ 7d270b0e-fda8-11ea-1e97-6dde7a729b3b
begin
	#TODO
end

# ╔═╡ b5ee5b0c-fd41-11ea-0beb-cb42134ce271
md"# Covid Statistics by Country"

# ╔═╡ 763c32b4-fd36-11ea-2eb5-91a7b1ec8037
begin
	country_selection = @bind country Select(unique(df[:, "country"]))
	md"""
	Country Select: $(country_selection)
	"""
end

# ╔═╡ d1a12aae-fd38-11ea-0b58-0d8d63dc15a8
begin
	country_data = filter(row -> row.country == country, df)
	ret = md"Only one province."
	if length(country_data.province) > 1
		province_select = @bind province Select(
			push!(unique(skipmissing(country_data.province)), country),
			default=country
		)
		ret = md"Province Select: $(province_select)"
	else
		province = ""
	end
	ret
end

# ╔═╡ 993e8c2e-fd36-11ea-0cfa-dd10ceb9d1ab
begin
	date_fmt = Dates.DateFormat("m/d/y")

	if length(country_data.province) > 1 # if there are provinces
		if province === "None"
			p_selector = missing
		else
			p_selector = province
		end
		data = filter(row -> row.province === p_selector, country_data)
	else
		data = country_data
	end
	date_strings = names(data)[5:end]
	dates = parse.(Date, date_strings, date_fmt) .+ Year("2000")
	data
end

# ╔═╡ 35d0671a-fda7-11ea-2f6f-693caac8d64f
begin
	day = min(length(dates)-1, clock_day)
	ppdate = dates[day]
	
	md"""### Statistics for $(ppdate) (day $(day))"""
end

# ╔═╡ 91687304-fda4-11ea-1540-637630e89679
begin
	# use country name where province is missing
	all_provinces = df.province
	missing_prov = ismissing.(all_provinces)
	all_provinces[missing_prov] .= df.country[missing_prov]
	
	daily_world = max.(0, diff(
		Array( df[:, 5:end] ),
		dims=2
	))
	
	world_plot = begin
		plot(shp_countries, alpha=0.2)
		scatter!(
			df.long, df.lat, 
			ms=2*log10.(daily_world[:, day]),
			alpha=0.7,
			leg=false
		)
	end
end

# ╔═╡ a22200f4-fd40-11ea-153e-9b2437f1ba84
begin
	num_data = Vector(data[1, 5:end])
	daily_cases = diff(num_data)
	
	total_cases = maximum(num_data)
	new_cases = daily_cases[end]
	avg_new = mean(daily_cases[end-7:end])
	
	md"""
	### 😷 Total Cases: $(total_cases)
	### 🤧 New Cases since yesterday: $(new_cases)
	### 🤒 New Cases 7 day average: $(avg_new)
	"""
end

# ╔═╡ 1beb0e06-fd3a-11ea-0a9b-e9780f1101e9
begin	
	avg_count = length(num_data) - 4
	avg = [mean(daily_cases[i-3:i+3]) for i in 4:avg_count]
	
	scaling_factor = maximum(daily_cases) / maximum(num_data)
	
	bar(
		dates, 
		num_data .* scaling_factor, 
		label="rescaled total cases", 
		c=:brown, 
		lw=0,
		alpha=0.2,
		leg=:topleft,
	)
	bar!(dates, daily_cases, label="daily cases", lw=0, c=:grey, alpha=0.6)
	plot!(dates[4:avg_count], avg, label="7 day average", lw=2, c=:red)
end

# ╔═╡ 24076c7c-fdd4-11ea-1a63-a718c7cd2c27
begin
	log_data = log.(10, max.(1, num_data))
	plot(
		dates,
		log_data / maximum(log_data),
		leg=:topleft
	)
	plot!(dates, num_data / maximum(num_data))
end

# ╔═╡ Cell order:
# ╟─b213700c-fd35-11ea-1552-1794da9334d6
# ╟─430098da-fd34-11ea-064b-e9954e55543b
# ╟─ee2149aa-fd35-11ea-350f-41ffff1bc8dd
# ╟─75e5d342-fda4-11ea-07b4-0b03effe68d7
# ╟─7ed3b6b0-fda5-11ea-2317-4d2cc3c31ee9
# ╠═fb1f2e8a-fda6-11ea-07bf-e9ce7424de87
# ╟─35d0671a-fda7-11ea-2f6f-693caac8d64f
# ╟─91687304-fda4-11ea-1540-637630e89679
# ╟─36578262-fda8-11ea-3f53-375a4d1e402d
# ╠═7d270b0e-fda8-11ea-1e97-6dde7a729b3b
# ╟─b5ee5b0c-fd41-11ea-0beb-cb42134ce271
# ╟─763c32b4-fd36-11ea-2eb5-91a7b1ec8037
# ╠═d1a12aae-fd38-11ea-0b58-0d8d63dc15a8
# ╟─993e8c2e-fd36-11ea-0cfa-dd10ceb9d1ab
# ╟─a22200f4-fd40-11ea-153e-9b2437f1ba84
# ╟─1beb0e06-fd3a-11ea-0a9b-e9780f1101e9
# ╠═24076c7c-fdd4-11ea-1a63-a718c7cd2c27
