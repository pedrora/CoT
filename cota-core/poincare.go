package main

import (
	"math"
	"encoding/binary"
)

const Dim = 64

type SoulMask struct {
	Data []byte      // The 1024-bit raw operator
	Pos  [Dim]float64 // The 64D Poincaré coordinate
}

func NewSoulMask(seedPath string) *SoulMask {
	// Initialize with 10MB DNA or a 1024-bit string
	return &SoulMask{
		Data: make([]byte, 128),
	}
}

// Rotate projects the 1024-bit vector into Hyperbolic Space
func (m *SoulMask) Rotate(curvature []byte) {
	// 1. Update the raw bit-operator (Interference)
	for i := 0; i < 128; i++ {
		m.Data[i] ^= curvature[i]
	}

	// 2. Map Bits to Euclidean 64D Vector
	var vec [Dim]float64
	for i := 0; i < Dim; i++ {
		// Take 16-bit segments (2 bytes)
		seg := binary.BigEndian.Uint16(m.Data[i*2 : i*2+2])
		// Normalize to [-1.0, 1.0]
		vec[i] = (float64(seg) / 32768.0) - 1.0
	}

	// 3. Project to Poincaré Ball (The 'Squash')
	// ds² = 4 * Σ dx² / (1 - Σ x²)²
	norm := 0.0
	for _, v := range vec {
		norm += v * v
	}
	norm = math.Sqrt(norm)

	// Hyperbolic constraint: tanh(norm) keeps the vector inside r < 1
	factor := math.Tanh(norm) / (norm + 1e-10)
	for i := 0; i < Dim; i++ {
		m.Pos[i] = vec[i] * factor
	}
}

// GetLinearAddress converts the 64D position back to a 1GB memory offset
func (m *SoulMask) GetLinearAddress(maxSize uintptr) uintptr {
	// We use the Norm (Radius) to determine 'Depth' 
	// and the first few dimensions to determine 'Angle' (Sector)
	norm := 0.0
	for _, v := range m.Pos {
		norm += v * v
	}
	norm = math.Sqrt(norm)

	// Radial Scaling: exponential growth toward the edge
	rFactor := (math.Exp(4*norm) - 1) / (math.Exp(4) - 1)
	
	// Angular Hash: deterministic mapping of the direction
	angleHash := uint64(m.Pos[0]*1000) ^ uint64(m.Pos[1]*1000)
	
	address := uintptr(rFactor * float64(maxSize))
	return (address + uintptr(angleHash)) % (maxSize - 128)
}