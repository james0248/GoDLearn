package util

func Reverse[T any](slice []T) []T {
	length := len(slice)
	rev := make([]T, length)
	for i := range slice {
		rev[length-i-1] = slice[i]
	}
	return rev
}
