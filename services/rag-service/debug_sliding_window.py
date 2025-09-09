"""Debug the sliding window function."""


def sliding_window_debug(text_length: int, chunk_size: int, overlap: int):
    """Debug version with print statements."""
    print(
        f"Input: text_length={text_length}, chunk_size={chunk_size}, overlap={overlap}"
    )

    if text_length == 0:
        return []

    positions = []
    start = 0
    iteration = 0

    while start < text_length:
        iteration += 1
        print(f"  Iteration {iteration}: start={start}")

        end = min(start + chunk_size, text_length)
        print(f"    end={end}")

        if start < end:  # Ensure valid chunk
            positions.append((start, end))
            print(f"    Added chunk: ({start}, {end})")

        # If we've reached the end, stop
        if end >= text_length:
            print(f"    Stopping: end ({end}) >= text_length ({text_length})")
            break

        # Calculate next start with overlap
        next_start = end - overlap
        print(f"    next_start (before adjustment)={next_start}")

        if next_start <= start:  # Prevent infinite loop - ensure progress
            next_start = start + 1
            print(f"    Adjusted next_start to {next_start} (ensuring progress)")

        start = next_start
        print(f"    New start: {start}")

        if iteration > 20:  # Safety break
            print("    SAFETY BREAK: Too many iterations")
            break

    print(f"Final positions: {positions}")
    return positions


# Test the problematic case
result = sliding_window_debug(100, 50, 10)
print(f"\nExpected: [(0, 50), (40, 90), (80, 100)]")
print(f"Actual:   {result}")

# Let's also test a simpler case
print("\n" + "=" * 50)
result2 = sliding_window_debug(10, 5, 1)
print(f"\nExpected for (10, 5, 1): [(0, 5), (4, 9), (8, 10)]")
print(f"Actual:   {result2}")
