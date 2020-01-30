package go_sklearn_predict

import (
	"math"
)

type activation string

const (
	ActivationLogistic = activation(`LOGISTIC`)
	ActivationRelu     = activation(`RELU`)
	ActivationTanh     = activation(`TANH`)
	ActivationSoftMax  = activation(`SOFTMAX`)
)

func NewMLPClassifier(ActivationHidden activation,
	ActivationOutput activation,
	Layers []int,
	Weights [][][]float64,
	Bias [][]float64) (*MLPClassifier, error) {

	network := make([][]float64, len(Layers)+1)
	for i, v := range Layers {
		network[i+1] = make([]float64, v)
	}
	model := &MLPClassifier{
		ActivationHidden: ActivationHidden,
		ActivationOutput: ActivationOutput,
		Weights:          Weights,
		Bias:             Bias,
		network:          network,
	}
	return model, nil
}

type MLPClassifier struct {
	ActivationHidden activation
	ActivationOutput activation
	Weights          [][][]float64
	Bias             [][]float64
	network          [][]float64
}

func (mlp *MLPClassifier) PredictRaw(x []float64) []float64 {
	mlp.network[0] = x

	for i := 0; i < len(mlp.network)-1; i++ {
		for j := 0; j < len(mlp.network[i+1]); j++ {
			mlp.network[i+1][j] = mlp.Bias[i][j]
			for l := 0; l < len(mlp.network[i]); l++ {
				mlp.network[i+1][j] += mlp.network[i][l] * mlp.Weights[i][l][j]
			}
		}
		if (i + 1) < (len(mlp.network) - 1) {
			mlp.network[i+1] = compute(mlp.ActivationHidden, mlp.network[i+1])
		}
	}
	mlp.network[len(mlp.network)-1] = compute(mlp.ActivationOutput, mlp.network[len(mlp.network)-1])

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

func compute(activ activation, v []float64) []float64 {
	switch activ {
	case ActivationLogistic:
		for i := 0; i < len(v); i++ {
			v[i] = 1. / (1. + math.Exp(-v[i]))
		}
		break
	case ActivationRelu:
		for i := 0; i < len(v); i++ {
			v[i] = math.Max(0, v[i])
		}
		break
	case ActivationTanh:
		for i := 0; i < len(v); i++ {
			v[i] = math.Tanh(v[i])
		}
		break
	case ActivationSoftMax:
		var max = math.Inf(-1)
		for i := 0; i < len(v); i++ {
			if v[i] > max {
				max = v[i]
			}
		}
		for i := 0; i < len(v); i++ {
			v[i] = math.Exp(v[i] - max)
		}
		var sum = 0.0
		for i := 0; i < len(v); i++ {
			sum += v[i]
		}
		for i := 0; i < len(v); i++ {
			v[i] /= sum
		}
		break
	}
	return v
}
