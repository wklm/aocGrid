package wklm.aoc;

/**
 * Immutable record representing grid dimensions.
 *
 * @param nrows Number of rows in the grid.
 * @param ncols Number of columns in the grid.
 */
public record Dimensions(int nrows, int ncols) {
    public Dimensions {
        if (nrows < 0 || ncols < 0) {
            throw new IllegalArgumentException("Dimensions must be non-negative.");
        }
    }
}
