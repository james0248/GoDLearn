package tensor

import (
	"errors"
	"fmt"
)

type Number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~float32 | ~float64
}

type Tensor[T Number] struct {
	data   []T
	dim    int
	stride Stride
	shape  Shape
}

func NewZeroTensor[T Number](_shape ...int) *Tensor[T] {
	shape := Shape(_shape)
	dim := len(shape)
	length, stride := shape.getLengthAndStride()
	data := make([]T, length)
	return &Tensor[T]{
		data:   data,
		dim:    dim,
		stride: stride,
		shape:  shape,
	}
}

func NewOneTensor[T Number](_shape ...int) *Tensor[T] {
	t := NewZeroTensor[T](_shape...)
	for i := range t.data {
		t.data[i] = 1
	}
	return t
}

func NewTensor[T Number](data []T, _shape ...int) *Tensor[T] {
	shape := Shape(_shape)
	dim := len(shape)
	stride := shape.getStride()
	return &Tensor[T]{
		data:   data,
		dim:    dim,
		stride: stride,
		shape:  shape,
	}
}

func (t *Tensor[T]) Data() []T {
	return t.data
}

func (t *Tensor[T]) SetData(data []T) error {
	if origLen, newLen := len(t.data), len(data); origLen == newLen {
		return errors.New(fmt.Sprintf("original data length %d is not equal to new data length %d", origLen, newLen))
	}
	t.data = data
	return nil
}

func (t *Tensor[T]) Shape() Shape {
	return t.shape
}

func (t *Tensor[T]) Resize(_shape ...int) (*Tensor[T], error) {
	shape := Shape(_shape)
	length, stride := shape.getLengthAndStride()
	if len(t.data) != length {
		return nil, errors.New("new shape does not match original data length")
	}

	t.shape = shape
	t.dim = len(shape)
	t.stride = stride
	return t, nil
}

func (t *Tensor[T]) Dim() int {
	return t.dim
}

func (t *Tensor[T]) Stride() Stride {
	return t.Stride()
}

func (t *Tensor[T]) Copy() *Tensor[T] {
	return NewTensor(t.data, t.shape...)
}

func (t *Tensor[T]) CopyShape() *Tensor[T] {
	return NewZeroTensor[T](t.shape...)
}
