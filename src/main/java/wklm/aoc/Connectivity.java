package wklm.aoc;

/**
 * Enum representing grid connectivity types.
 */
public enum Connectivity {
    /**
     * 4-directional connectivity (up, down, left, right).
     */
    FOUR(4),

    /**
     * 8-directional connectivity (4-directional plus diagonals).
     */
    EIGHT(8);

    private final int value;

    Connectivity(int value) {
        this.value = value;
    }

    /**
     * Gets the numeric value of this connectivity type.
     *
     * @return 4 or 8
     */
    public int getValue() {
        return value;
    }

    /**
     * Creates a Connectivity enum from an integer value.
     *
     * @param value 4 or 8
     * @return Corresponding Connectivity enum
     * @throws IllegalArgumentException if value is not 4 or 8
     */
    public static Connectivity fromInt(int value) {
        return switch (value) {
            case 4 -> FOUR;
            case 8 -> EIGHT;
            default -> throw new IllegalArgumentException("Connectivity must be 4 or 8, got: " + value);
        };
    }
}
