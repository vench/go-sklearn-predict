package go_sklearn_predict

import "errors"

var (
	ErrorNotFoundNodeResult = errors.New(`not found node result`)
	ErrorModelNotLoaded     = errors.New(`model not loaded`)
)
