package wklm.aoc;

/**
 * Immutable record representing a point in the grid.
 *
 * @param <T> The type of the value held by the point.
 */
public record Point<T>(int x, int y, T value) {}
