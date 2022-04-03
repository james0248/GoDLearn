package nn

import (
	"GoDLearn/tensor"
	"math"
)

type LinearFuncOpt func(lo *LinearOptions)

type LinearOptions struct {
	bias bool
}

type Linear[T tensor.Number] struct {
	name        string
	inFeatures  int
	outFeatures int
	weight      *tensor.Tensor[T]
	bias        *tensor.Tensor[T]
}

func WithBias(bias bool) func(*LinearOptions) {
	return func(lo *LinearOptions) {
		lo.bias = bias
	}
}

func NewLinear[T tensor.Number](inFeatures, outFeatures int, opts ...LinearFuncOpt) *Linear[T] {
	deafultOpt := &LinearOptions{bias: true}
	for _, opt := range opts {
		opt(deafultOpt)
	}

	linear := &Linear[T]{
		name:        "",
		inFeatures:  inFeatures,
		outFeatures: outFeatures,
		weight:      tensor.NewZeroTensor[T](outFeatures, inFeatures),
		bias:        nil,
	}
	if deafultOpt.bias {
		linear.bias = tensor.NewZeroTensor[T](outFeatures)
	}
	linear.resetParameters()
	return linear
}

func (l *Linear[T]) resetParameters() {
	bound := T(1 / math.Sqrt(float64(l.inFeatures)))
	Uniform(l.weight, -bound, bound)
	if l.bias != nil {
		Uniform(l.bias, -bound, bound)
	}
}

func (l *Linear[T]) Forward(input *tensor.Tensor[T]) *tensor.Tensor[T] {
	output := l.weight.MatMul(input).Add(l.bias)
	return output
}

func (l *Linear[T]) Weight() *tensor.Tensor[T] {
	return l.weight
}
