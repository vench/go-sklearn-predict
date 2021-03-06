package go_sklearn_predict

import "testing"

func TestNewMLPClassifier(t *testing.T) {
	layers := []int{2, 1}
	weights := [][][]float64{{
		{0.2938741350525824, -1.2010742740364726},
		{-0.3943454456681259, -1.0438345984520652},
		{0.712114427850095, -0.7365274828404846},
		{-1.2185988843425422, -0.4252100269854478},
		{3.8695560304980517e-26, 1.1141002163878758e-66},
		{-2.524760965730412, -2.4494973303889376},
		{1.2181558230561722e-96, -1.4499070244467648e-24},
		{0.2871571610594222, -0.14282168541725787},
		{0.7611658435030262, 2.1525082872958529e-72},
		{-0.16754238530283352, -1.4815847205749806},
		{0.2778075453395965, -0.38224886639394334}},
		{{-0.820228881275432}, {-3.6370049373844617}}}
	bias := [][]float64{{0.03899700691803675, 1.2520733772112358}, {0.8763071894666827}}

	m, err := NewMLPClassifier(ActivationRelu, ActivationLogistic, layers, weights, bias)
	if err != nil {
		t.Fatal(err)
	}

	v := []float64{0.586, 0., 0.854, 1., 0., 0.064, 0., 0.922, 0., 0., 0.}
	y, err := m.Predict(v)
	if err != nil {
		t.Fatalf(``)
	}
	if y != 1 {
		t.Fatalf(`class: %d != 1`, y)
	}

	v = []float64{0.378, 0., 0.929, 1., 0., 0., 0., 0.952, 0., 0., 0.}
	y, err = m.Predict(v)
	if err != nil {
		t.Fatalf(``)
	}
	if y != 1 {
		t.Fatalf(`class: %d != 1`, y)
	}
	v = []float64{0.577, 0., 0., 0., 0., -0., 0., 0., 0., 0., 0.}
	y, err = m.Predict(v)
	if err != nil {
		t.Fatalf(``)
	}
	if y != 0 {
		t.Fatalf(`class: %d != 0`, y)
	}

	v = []float64{0.244, 0., 0., 0., 0., -0., 0., 0., 0., 0., 0.}
	y, err = m.Predict(v)
	if err != nil {
		t.Fatalf(``)
	}
	if y != 0 {
		t.Fatalf(`class: %d != 0`, y)
	}
}
