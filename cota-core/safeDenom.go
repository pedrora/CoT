import "math"

func safeDenom(x float64) float64 {
    return math.Max(x, 1e-10)
}

// Usage anywhere in projection / normalization:
factor := 0.999 / safeDenom(norm)