package wklm.aoc;

/**
 * Interface for parsing characters into type T.
 *
 * @param <T> The type to parse characters into.
 */
@FunctionalInterface
public interface CharacterParser<T> {
    /**
     * Parses a character into type T.
     *
     * @param c the character to parse
     * @return the parsed value of type T
     */
    T parse(char c);
}
