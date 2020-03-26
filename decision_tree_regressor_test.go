package go_sklearn_predict

import "testing"

func TestTreePredict_Predict(t *testing.T) {

	p := NewDecisionTreeRegressor(
		make([]*TreeNode, 0),
		make([]float64, 0),
	)

	if _, err := p.Predict([]float64{}); err != ErrorModelNotLoaded {
		t.Fatalf(``)
	}

	p.values = []float64{100}
	p.nodes = []*TreeNode{{-1, -1, -1, 0}}

	val, err := p.Predict([]float64{0})
	if err != nil {
		t.Fatalf(``)
	}
	if val != 100 {
		t.Fatalf(``)
	}

	p.values = []float64{}
	p.nodes = []*TreeNode{{-1, -1, -1, 0}}

	if _, err := p.Predict([]float64{}); err != ErrorNotFoundNodeResult {
		t.Fatalf(``)
	}

	p.values = []float64{}
	p.nodes = []*TreeNode{{1, 2, 0, 0}}

	if _, err := p.Predict([]float64{}); err != ErrorNotFoundNodeResult {
		t.Fatalf(``)
	}

	p.values = []float64{0, 500, 1000}
	p.nodes = []*TreeNode{
		{1, 2, 0, 60},
		{-1, -1, 0, 0},
		{-1, -1, 0, 0},
	}

	val, err = p.Predict([]float64{50})
	if err != nil {
		t.Fatalf(``)
	}
	if val != p.values[1] {
		t.Fatalf(``)
	}
	val, err = p.Predict([]float64{70})
	if err != nil {
		t.Fatalf(``)
	}
	if val != p.values[2] {
		t.Fatalf(``)
	}
}

func TestTreePredict_Predict2(t *testing.T) {
	nodes := []*TreeNode{{1, 4, 3, 736936.781250}, {2, 3, 3, 271022.218750}, {-1, -1, -2, -2.000000}, {-1, -1, -2, -2.000000}, {5, 6, 0, 350275.531250}, {-1, -1, -2, -2.000000}, {-1, -1, -2, -2.000000}}
	values := []float64{92532.134603, 26498.711659, 18377.824710, 137515.772113, 950070.180220, 406891.760946, 2043264.541900}

	p := &DecisionTreeRegressor{
		nodes:  nodes,
		values: values,
	}

	tests := []struct {
		x []float64
		y float64
	}{
		{[]float64{62944.435, 2271700., 0., 164301.31178, 401183.32083, 20750000.}, 18377.824710},
	}

	for _, test := range tests {
		val, err := p.Predict(test.x)
		if err != nil {
			t.Fatalf(``)
		}
		if val != test.y {
			t.Fatalf(``)
		}
	}

}
