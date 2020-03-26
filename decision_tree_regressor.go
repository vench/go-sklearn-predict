package go_sklearn_predict

func NewDecisionTreeRegressor(nodes []*TreeNode, values []float64) *DecisionTreeRegressor {
	p := &DecisionTreeRegressor{nodes,
		values,
	}
	return p
}

type DecisionTreeRegressor struct {
	nodes  []*TreeNode
	values []float64
}

func (t *DecisionTreeRegressor) Predict(x []float64) (float64, error) {

	var (
		i = 0
	)

	if len(t.nodes) == 0 {
		return 0, ErrorModelNotLoaded
	}

	for i >= 0 && t.nodes[i].Feature >= 0 && t.nodes[i].Feature < len(x) {

		if x[t.nodes[i].Feature] <= t.nodes[i].Threshold {
			if t.nodes[i].Left < 0 {
				break
			}
			i = t.nodes[i].Left
		} else {
			if t.nodes[i].Right < 0 {
				break
			}
			i = t.nodes[i].Right
		}
	}
	if len(t.values) > i {
		return t.values[i], nil
	}
	return 0, ErrorNotFoundNodeResult
}

type TreeNode struct {
	Left      int
	Right     int
	Feature   int
	Threshold float64
}
