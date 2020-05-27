import Pkg
Pkg.add("RDatasets")

using CSV
using RDatasets
using DataFrames

auto = dataset("ISLR", "Auto")
first(auto, 3)

describe(auto, :mean, :median, :std)

names(auto)

mpg = auto.MPG
mpg = auto[:, 1]
mpg = auto[:, :MPG]
mpg |> mean

@show size(auto)
@show nrow(auto)
@show ncol(auto)


Pkg.add("PyPlot")
using PyPlot

figure(figsize=(8,6))
plot(mpg)
