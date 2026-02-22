#!/usr/bin/env python3
"""
extract_dna_xor.py

Chunks a large file into 10 MB blocks, XORs them all together,
and outputs a single 10 MB seed file.

Usage: python extract_dna_xor.py input.bin output.bin
"""

import sys
import os

CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB

def xor_chunks(input_path, output_path):
    # Initialize accumulator with zeros
    accumulator = bytearray(CHUNK_SIZE)
    
    total_read = 0
    chunk_count = 0
    
    with open(input_path, 'rb') as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            # If chunk is smaller than CHUNK_SIZE, pad with zeros
            if len(chunk) < CHUNK_SIZE:
                chunk = chunk.ljust(CHUNK_SIZE, b'\x00')
            # XOR into accumulator
            for i in range(CHUNK_SIZE):
                accumulator[i] ^= chunk[i]
            total_read += len(chunk)
            chunk_count += 1
            print(f"Processed chunk {chunk_count} (total {total_read / (1024*1024):.2f} MB)", end='\r')
    
    print(f"\nFinished. {chunk_count} chunks XORed. Total bytes processed: {total_read}")
    
    # Write accumulator to output
    with open(output_path, 'wb') as f:
        f.write(accumulator)
    
    print(f"Seed written to {output_path} ({len(accumulator)} bytes)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_dna_xor.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: input file '{input_file}' not found.")
        sys.exit(1)
    
    xor_chunks(input_file, output_file)