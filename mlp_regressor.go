package go_sklearn_predict

func NewMLPRegressor(ActivationHidden activation,
	layers []int,
	weights [][][]float64,
	bias [][]float64) *MLPRegressor {

	network := make([][]float64, len(layers)+1)
	for i, v := range layers {
		network[i+1] = make([]float64, v)
	}

	return &MLPRegressor{
		activationHidden: ActivationHidden,
		network:          network,
		weights:          weights,
		bias:             bias,
	}
}

type MLPRegressor struct {
	activationHidden activation
	weights          [][][]float64
	bias             [][]float64
	network          [][]float64
}

func (mlp *MLPRegressor) Predict(x []float64) (float64, error) {

	mlp.network[0] = x

	for i := 0; i < len(mlp.network)-1; i++ {
		for j := 0; j < len(mlp.network[i+1]); j++ {
			mlp.network[i+1][j] = mlp.bias[i][j]
			for l := 0; l < len(mlp.network[i]); l++ {
				mlp.network[i+1][j] += mlp.network[i][l] * mlp.weights[i][l][j]
			}
		}
		if (i + 1) < (len(mlp.network) - 1) {
			mlp.network[i+1] = compute(mlp.activationHidden, mlp.network[i+1])
		}
	}

	if len(mlp.network[len(mlp.network)-1]) > 1 {
		return mlp.network[len(mlp.network)-1][0], nil
	}
	return mlp.network[len(mlp.network)-1][0], nil
}
