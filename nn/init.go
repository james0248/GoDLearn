package nn

import (
	godl "GoDLearn"
	"GoDLearn/tensor"
)

func Uniform[T godl.Number](t *tensor.Tensor[T], from, to T) {
	t.Uniform(from, to)
}
