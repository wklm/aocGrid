package wklm.aoc;

/**
 * Cardinal directions for grid navigation - optimized for Advent of Code puzzles.
 * Provides rotation, movement, and direction arithmetic commonly needed in AOC.
 */
public enum Direction {
    NORTH(-1, 0, '^'),
    SOUTH(1, 0, 'v'),
    WEST(0, -1, '<'),
    EAST(0, 1, '>');

    private final int dx;
    private final int dy;
    private final char symbol;

    Direction(int dx, int dy, char symbol) {
        this.dx = dx;
        this.dy = dy;
        this.symbol = symbol;
    }

    /**
     * Gets the row delta for this direction.
     */
    public int dx() {
        return dx;
    }

    /**
     * Gets the column delta for this direction.
     */
    public int dy() {
        return dy;
    }

    /**
     * Gets the character symbol representing this direction.
     */
    public char symbol() {
        return symbol;
    }

    /**
     * Rotates this direction 90 degrees clockwise.
     * NORTH -> EAST -> SOUTH -> WEST -> NORTH
     */
    public Direction turnRight() {
        return switch (this) {
            case NORTH -> EAST;
            case EAST -> SOUTH;
            case SOUTH -> WEST;
            case WEST -> NORTH;
        };
    }

    /**
     * Rotates this direction 90 degrees counter-clockwise.
     * NORTH -> WEST -> SOUTH -> EAST -> NORTH
     */
    public Direction turnLeft() {
        return switch (this) {
            case NORTH -> WEST;
            case WEST -> SOUTH;
            case SOUTH -> EAST;
            case EAST -> NORTH;
        };
    }

    /**
     * Reverses this direction (180 degree turn).
     * NORTH <-> SOUTH, EAST <-> WEST
     */
    public Direction reverse() {
        return switch (this) {
            case NORTH -> SOUTH;
            case SOUTH -> NORTH;
            case EAST -> WEST;
            case WEST -> EAST;
        };
    }

    /**
     * Applies this direction to a point, returning new coordinates.
     *
     * @param row Current row
     * @param col Current column
     * @return Array [newRow, newCol]
     */
    public int[] apply(int row, int col) {
        return new int[]{row + dx, col + dy};
    }

    /**
     * Applies this direction N times to a point.
     *
     * @param row Current row
     * @param col Current column
     * @param steps Number of steps to move
     * @return Array [newRow, newCol]
     */
    public int[] apply(int row, int col, int steps) {
        return new int[]{row + dx * steps, col + dy * steps};
    }

    /**
     * Creates a Direction from a character symbol.
     *
     * @param symbol '^', 'v', '<', '>', 'N', 'S', 'W', 'E', 'U', 'D', 'L', 'R'
     * @return The corresponding Direction
     * @throws IllegalArgumentException if symbol is invalid
     */
    public static Direction fromSymbol(char symbol) {
        return switch (Character.toUpperCase(symbol)) {
            case '^', 'N', 'U' -> NORTH;
            case 'v', 'S', 'D' -> SOUTH;
            case '<', 'W', 'L' -> WEST;
            case '>', 'E', 'R' -> EAST;
            default -> throw new IllegalArgumentException("Invalid direction symbol: " + symbol);
        };
    }

    /**
     * Gets all four cardinal directions.
     */
    public static Direction[] all() {
        return values();
    }

    /**
     * Checks if this direction is horizontal (EAST or WEST).
     */
    public boolean isHorizontal() {
        return this == EAST || this == WEST;
    }

    /**
     * Checks if this direction is vertical (NORTH or SOUTH).
     */
    public boolean isVertical() {
        return this == NORTH || this == SOUTH;
    }
}
