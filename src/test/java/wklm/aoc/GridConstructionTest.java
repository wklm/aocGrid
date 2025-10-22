package wklm.aoc;

import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Grid construction (constructors and factory methods).
 */
class GridConstructionTest {

    @Test
    void testFromStringSimple() {
        String input = """
            ABC
            DEF
            GHI""";

        Optional<Grid<Character>> gridOpt = Grid.fromString(input, c -> c);
        assertTrue(gridOpt.isPresent());

        Grid<Character> grid = gridOpt.get();
        assertEquals(3, grid.getRows());
        assertEquals(3, grid.getCols());
        assertEquals('A', grid.get(0, 0).get().value());
        assertEquals('I', grid.get(2, 2).get().value());
    }

    @Test
    void testFromStringWithParser() {
        String input = """
            123
            456
            789""";

        Optional<Grid<Integer>> gridOpt = Grid.fromString(input,
            c -> Character.isDigit(c) ? Character.getNumericValue(c) : null);

        assertTrue(gridOpt.isPresent());
        Grid<Integer> grid = gridOpt.get();

        assertEquals(1, grid.get(0, 0).get().value());
        assertEquals(5, grid.get(1, 1).get().value());
        assertEquals(9, grid.get(2, 2).get().value());
    }

    @Test
    void testFromStringInvalidUnevenRows() {
        String input = """
            ABC
            DE
            FGH""";

        Optional<Grid<Character>> gridOpt = Grid.fromString(input, c -> c);
        assertFalse(gridOpt.isPresent());
    }

    @Test
    void testFromStringSingleCell() {
        String input = "X";

        Optional<Grid<Character>> gridOpt = Grid.fromString(input, c -> c);
        assertTrue(gridOpt.isPresent());

        Grid<Character> grid = gridOpt.get();
        assertEquals(1, grid.getRows());
        assertEquals(1, grid.getCols());
        assertEquals('X', grid.get(0, 0).get().value());
    }

    @Test
    void testFromStringEmptyString() {
        String input = "";

        Optional<Grid<Character>> gridOpt = Grid.fromString(input, c -> c);
        assertTrue(gridOpt.isPresent());

        Grid<Character> grid = gridOpt.get();
        assertEquals(1, grid.getRows());
        assertEquals(0, grid.getCols());
    }

    @Test
    void testFromStringWithNullInputThrowsException() {
        assertThrows(NullPointerException.class, () -> {
            Grid.fromString(null, c -> c);
        });
    }

    @Test
    void testFromStringWithNullParserThrowsException() {
        assertThrows(NullPointerException.class, () -> {
            Grid.fromString("ABC", null);
        });
    }

    @Test
    void testFromStringWithNullBehaviorTypeThrowsException() {
        assertThrows(NullPointerException.class, () -> {
            Grid.fromString("ABC", c -> c, null);
        });
    }

    @Test
    void testFromStringStandardBehavior() {
        String input = "AB\nCD";

        Optional<Grid<Character>> gridOpt = Grid.fromString(input, c -> c, Grid.BehaviorType.STANDARD);
        assertTrue(gridOpt.isPresent());
    }

    @Test
    void testFromStringToroidalBehavior() {
        String input = "AB\nCD";

        Optional<Grid<Character>> gridOpt = Grid.fromString(input, c -> c, Grid.BehaviorType.TOROIDAL);
        assertTrue(gridOpt.isPresent());
    }

    @Test
    void testFromToroidalString() {
        String input = "AB\nCD";

        Optional<Grid<Character>> gridOpt = Grid.fromToroidalString(input, c -> c);
        assertTrue(gridOpt.isPresent());
    }

    @Test
    void testProgrammaticConstructorBasic() {
        Dimensions dims = new Dimensions(3, 3);
        List<Point<Character>> points = List.of(
            new Point<>(0, 0, 'A'),
            new Point<>(1, 1, 'B'),
            new Point<>(2, 2, 'C')
        );

        Grid<Character> grid = new Grid<>(points, dims, Optional.of('.'), Optional.empty());

        assertEquals(3, grid.getRows());
        assertEquals(3, grid.getCols());
        assertEquals('A', grid.get(0, 0).get().value());
        assertEquals('B', grid.get(1, 1).get().value());
        assertEquals('C', grid.get(2, 2).get().value());
        assertEquals('.', grid.get(0, 1).get().value()); // Default value
    }

    @Test
    void testProgrammaticConstructorWithoutDefaultValue() {
        Dimensions dims = new Dimensions(2, 2);
        List<Point<String>> points = List.of(
            new Point<>(0, 0, "X")
        );

        Grid<String> grid = new Grid<>(points, dims, Optional.empty(), Optional.empty());

        assertEquals("X", grid.get(0, 0).get().value());
        assertNull(grid.get(0, 1).get().value());
        assertNull(grid.get(1, 0).get().value());
        assertNull(grid.get(1, 1).get().value());
    }

    @Test
    void testProgrammaticConstructorOverwritesDefaultValue() {
        Dimensions dims = new Dimensions(2, 2);
        List<Point<Character>> points = List.of(
            new Point<>(0, 0, 'X'),
            new Point<>(0, 0, 'Y')  // Overwrites previous
        );

        Grid<Character> grid = new Grid<>(points, dims, Optional.of('.'), Optional.empty());

        assertEquals('Y', grid.get(0, 0).get().value());
    }

    @Test
    void testProgrammaticConstructorOutOfBoundsThrowsException() {
        Dimensions dims = new Dimensions(2, 2);
        List<Point<Character>> points = List.of(
            new Point<>(5, 5, 'X')  // Out of bounds
        );

        assertThrows(IllegalArgumentException.class, () -> {
            new Grid<>(points, dims, Optional.of('.'), Optional.empty());
        });
    }

    @Test
    void testProgrammaticConstructorNegativeCoordinatesThrowsException() {
        Dimensions dims = new Dimensions(2, 2);
        List<Point<Character>> points = List.of(
            new Point<>(-1, 0, 'X')
        );

        assertThrows(IllegalArgumentException.class, () -> {
            new Grid<>(points, dims, Optional.of('.'), Optional.empty());
        });
    }

    @Test
    void testProgrammaticConstructorWithNullPointsThrowsException() {
        Dimensions dims = new Dimensions(2, 2);

        assertThrows(NullPointerException.class, () -> {
            new Grid<>(null, dims, Optional.of('.'), Optional.empty());
        });
    }

    @Test
    void testProgrammaticConstructorWithNullDimensionsThrowsException() {
        List<Point<Character>> points = List.of();

        assertThrows(NullPointerException.class, () -> {
            new Grid<>(points, null, Optional.of('.'), Optional.empty());
        });
    }

    @Test
    void testProgrammaticConstructorWithToroidalBehavior() {
        Dimensions dims = new Dimensions(2, 2);
        List<Point<Character>> points = List.of(
            new Point<>(0, 0, 'A')
        );

        Grid<Character> grid = new Grid<>(points, dims, Optional.of('.'),
            Optional.of(Grid.BehaviorType.TOROIDAL));

        assertEquals('A', grid.get(0, 0).get().value());
    }

    @Test
    void testFromStringWithNewlines() {
        String input = "A\nB\nC";

        Optional<Grid<Character>> gridOpt = Grid.fromString(input, c -> c);
        assertTrue(gridOpt.isPresent());

        Grid<Character> grid = gridOpt.get();
        assertEquals(3, grid.getRows());
        assertEquals(1, grid.getCols());
    }

    @Test
    void testFromStringRectangularGrid() {
        String input = """
            ABCD
            EFGH
            IJKL
            MNOP
            QRST""";

        Optional<Grid<Character>> gridOpt = Grid.fromString(input, c -> c);
        assertTrue(gridOpt.isPresent());

        Grid<Character> grid = gridOpt.get();
        assertEquals(5, grid.getRows());
        assertEquals(4, grid.getCols());
        assertEquals('A', grid.get(0, 0).get().value());
        assertEquals('T', grid.get(4, 3).get().value());
    }

    @Test
    void testGridImmutability() {
        String input = "AB\nCD";
        Optional<Grid<Character>> gridOpt = Grid.fromString(input, c -> c);
        Grid<Character> grid = gridOpt.get();

        // Grid should be immutable - getting points should not allow mutation
        Optional<Point<Character>> pointOpt = grid.get(0, 0);
        assertTrue(pointOpt.isPresent());

        // Point is a record, so it's immutable by design
        Point<Character> point = pointOpt.get();
        assertEquals('A', point.value());
    }
}
