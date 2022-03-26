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
	length, stride := shape.getSizeAndStride()
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

func (t *Tensor[T]) Reshape(_shape ...int) (*Tensor[T], error) {
	shape, err := InferShape(len(t.data), _shape)
	if err != nil {
		switch err.Error() {
		case "infer-mismatch":
			panic("Cannot reshape tensor: original tensor size does not match with unspecified dimension")
		case "size-mismatch":
			panic("Cannot reshape tensor: original tensor size does not match with given shape")
		case "infer-impossible":
			panic("Cannot reshape tensor: more than 2 dimensions cannot be inferred")
		}
	}
	stride := shape.getStride()

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

func (t *Tensor[T]) getLastTwoShapes() (int, int) {
	return t.shape.getLastTwoShapes()
}

func (t *Tensor[T]) getLastShape() int {
	return t.shape.getLastShape()
}

func (t *Tensor[T]) getFirstShapes(num int) Shape {
	return t.shape.getFirstShapes(num)
}
