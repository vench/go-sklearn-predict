package go_sklearn_predict

import "math"

type activation string

const (
	ActivationLogistic = activation(`LOGISTIC`)
	ActivationRelu     = activation(`RELU`)
	ActivationTanh     = activation(`TANH`)
	ActivationSoftMax  = activation(`SOFTMAX`)
)

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
