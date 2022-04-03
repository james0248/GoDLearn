package nn

import (
	"GoDLearn/tensor"
)

func Uniform[T tensor.Number](t *tensor.Tensor[T], from, to T) {
	t.Uniform(from, to)
}
