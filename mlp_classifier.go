package go_sklearn_predict

func NewMLPClassifier(ActivationHidden activation,
	activationOutput activation,
	layers []int,
	weights [][][]float64,
	bias [][]float64) (*MLPClassifier, error) {

	network := make([][]float64, len(layers)+1)
	for i, v := range layers {
		network[i+1] = make([]float64, v)
	}
	model := &MLPClassifier{
		activationHidden: ActivationHidden,
		activationOutput: activationOutput,
		weights:          weights,
		bias:             bias,
		network:          network,
	}
	return model, nil
}

type MLPClassifier struct {
	activationHidden activation
	activationOutput activation
	weights          [][][]float64
	bias             [][]float64
	network          [][]float64
}

func (mlp *MLPClassifier) PredictRaw(x []float64) []float64 {
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
	mlp.network[len(mlp.network)-1] = compute(mlp.activationOutput, mlp.network[len(mlp.network)-1])

	return mlp.network[len(mlp.network)-1]
}

func (mlp *MLPClassifier) Predict(x []float64) (int, error) {

	raw := mlp.PredictRaw(x)

	if len(raw) == 1 {
		if raw[0] > .5 {
			return 1, nil
		}
		return 0, nil
	} else {
		var classIdx = 0
		for i := 0; i < len(raw); i++ {
			if raw[i] > raw[classIdx] {
				classIdx = i
			}
		}
		return classIdx, nil
	}

	return -1, ErrorNotFoundNodeResult
}
