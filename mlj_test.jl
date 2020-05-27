# exploring Julia MLJ. From https://www.simonwenkel.com/2019/02/03/exploring-MLJjl.html


Pkg.add("MLJ")
Pkg.add("DataFrames")

import MLJ
import DataFrames

X,y = MLJ.X_and_y(MLJ.load_boston());

# train-test splitting (70:30) without validation set
train, test = MLJ.partition(eachindex(y), 0.7);

knn_model=MLJ.KNNRegressor(K=10)

knn = MLJ.machine(knn_model, X, y)

MLJ.fit!(knn, rows=train)

yhat = MLJ.predict(knn, X[test,:]);
yhat_rms = MLJ.rms(y[test], yhat)

knn_model.K = 20
MLJ.fit!(knn)
yhat = MLJ.predict(knn, X[test,:])
yhat_rms = MLJ.rms(y[test], yhat)

ensemble_model = MLJ.EnsembleModel(atom=knn_model, n=20)

MLJ.params(ensemble_model)

Params(:atom => Params(:K => 20, :metric => MLJ.KNN.euclidean, :kernel => MLJ.KNN.reciprocal), :weights => Float64[], :bagging_fraction => 0.8, :rng_seed => 0, :n => 20, :parallel => true)
