
import CSV
using DataFrames

file = "/Users/jnesnky/Downloads/train (1).csv"

df = DataFrame(CSV.File(file))

println(describe(df))

X,y = MLJ.X_and_y(MLJ.load_boston());
train, test = MLJ.partition(eachindex(y), 0.7);
