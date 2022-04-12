package util

import (
	godl "GoDLearn"
	"math/rand"
	"reflect"
	"time"
)

func UniformDist[T godl.Number](size int, from, to T) []T {
	source := rand.NewSource(time.Now().UnixNano())
	rnd := rand.New(source)
	data := make([]T, size)
	// TODO: Add other data types
	switch reflect.TypeOf(from).Kind() {
	case reflect.Float32:
		for i := range data {
			data[i] = T(rnd.Float32()*float32(to-from) + float32(from))
		}
		break
	case reflect.Float64:
		for i := range data {
			data[i] = T(rnd.Float64()*float64(to-from) + float64(from))
		}
		break
	case reflect.Int:
		for i := range data {
			data[i] = T(rnd.Intn(int(to-from)) + int(from))
		}
		break
	case reflect.Int32:
		for i := range data {
			data[i] = T(rnd.Int31n(int32(to-from)) + int32(from))
		}
		break
	case reflect.Int64:
		for i := range data {
			data[i] = T(rnd.Int63n(int64(to-from)) + int64(from))
		}
		break
	}

	return data
}
