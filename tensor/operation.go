package tensor

import (
	"errors"
	"fmt"
	"reflect"
)

func (t *Tensor[T]) operatableWith(other *Tensor[T]) error {
	if !t.shape.equalTo(other.shape) {
		return errors.New("shape")
	}
	if inputType, otherType := reflect.TypeOf(t.data), reflect.TypeOf(other.data); inputType != otherType {
		return errors.New("type")
	}
	return nil
}

func (t *Tensor[T]) multipliableWith(other *Tensor[T]) error {
	if t.dim <= 2 && other.dim <= 2 {
		if t.shape[t.dim-1] != other.shape[0] {
			return errors.New("dimesion-size")
		}
		return nil
	}

	// TODO: Implement broadcasting
	return errors.New("dimension-high")
}

// TODO: AddScaled => add times alpha

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

func (t *Tensor[T]) MatMul(other *Tensor[T]) *Tensor[T] {
	err := t.multipliableWith(other)
	if err != nil {
		panic(err)
	}

	if t.dim == 1 {
		t.Resize(1, -1)
	}
	if other.dim == 1 {
		other.Resize(-1, 1)
	}
	// TODO: Implement actual matrix multiplication

	return nil
}
