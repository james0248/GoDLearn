package tensor

import (
	"errors"
	"fmt"
	"reflect"
)

type Number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~float32 | ~float64
}

type TVector[T Number] []T

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

func (t *Tensor[T]) Copy() *Tensor[T] {
	return NewTensor(t.data, t.shape...)
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

func (t *Tensor[T]) CopyShape() *Tensor[T] {
	return NewZeroTensor[T](t.shape...)
}

func (t *Tensor[T]) operatableWith(other *Tensor[T]) error {
	if !t.shape.equalTo(other.shape) {
		return errors.New("shape")
	}
	if inputType, otherType := reflect.TypeOf(t.data), reflect.TypeOf(other.data); inputType != otherType {
		return errors.New("type")
	}
	return nil
}

func (t *Tensor[T]) Add(other *Tensor[T]) *Tensor[T] {
	err := t.operatableWith(other)
	if err != nil {
		switch err.Error() {
		case "shape":
			errString := fmt.Sprint("Cannot do operations between two tensors with different shapes: ", t.shape, other.shape)
			panic(errString)
		case "type":
			errString := fmt.Sprint("Cannot do operations between two tensors with different types: ", reflect.TypeOf(t.data[0]), reflect.TypeOf(other.data[0]))
			panic(errString)
		}
	}

	result := t.CopyShape()
	for i := range result.data {
		result.data[i] = t.data[i] + other.data[i]
	}
	return result
}

func (t *Tensor[T]) Sub(other *Tensor[T]) *Tensor[T] {
	err := t.operatableWith(other)
	if err != nil {
		switch err.Error() {
		case "shape":
			errString := fmt.Sprint("Cannot do operations between two tensors with different shapes: ", t.shape, other.shape)
			panic(errString)
		case "type":
			errString := fmt.Sprint("Cannot do operations between two tensors with different types: ", reflect.TypeOf(t.data[0]), reflect.TypeOf(other.data[0]))
			panic(errString)
		}
	}

	result := t.CopyShape()
	for i := range result.data {
		result.data[i] = t.data[i] - other.data[i]
	}
	return result
}
