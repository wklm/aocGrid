package wklm.aoc;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the CharacterParser interface.
 */
class CharacterParserTest {

    @Test
    void testIdentityParser() {
        CharacterParser<Character> parser = c -> c;
        assertEquals('A', parser.parse('A'));
        assertEquals('z', parser.parse('z'));
        assertEquals('0', parser.parse('0'));
    }

    @Test
    void testDigitParser() {
        CharacterParser<Integer> parser = c ->
            Character.isDigit(c) ? Character.getNumericValue(c) : null;

        assertEquals(0, parser.parse('0'));
        assertEquals(5, parser.parse('5'));
        assertEquals(9, parser.parse('9'));
        assertNull(parser.parse('A'));
        assertNull(parser.parse('.'));
    }

    @Test
    void testCustomParser() {
        CharacterParser<String> parser = c -> {
            if (c == '#') return "wall";
            if (c == '.') return "empty";
            return "unknown";
        };

        assertEquals("wall", parser.parse('#'));
        assertEquals("empty", parser.parse('.'));
        assertEquals("unknown", parser.parse('X'));
    }

    @Test
    void testBooleanParser() {
        CharacterParser<Boolean> parser = c -> c == '#';

        assertTrue(parser.parse('#'));
        assertFalse(parser.parse('.'));
        assertFalse(parser.parse('A'));
    }

    @Test
    void testComplexTypeParser() {
        record Cell(char symbol, boolean isWall) {}

        CharacterParser<Cell> parser = c -> new Cell(c, c == '#');

        Cell wall = parser.parse('#');
        assertEquals('#', wall.symbol());
        assertTrue(wall.isWall());

        Cell empty = parser.parse('.');
        assertEquals('.', empty.symbol());
        assertFalse(empty.isWall());
    }
}
