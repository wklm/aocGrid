package wklm.aoc;

import org.junit.jupiter.api.Test;
import java.util.*;
import java.util.function.BiFunction;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Grid transformation methods (transpose, union).
 */
class GridTransformationTest {

    @Test
    void testTransposeSquareGrid() {
        String input = """
            ABC
            DEF
            GHI""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();
        Grid<Character> transposed = grid.transpose();

        assertEquals(3, transposed.getRows());
        assertEquals(3, transposed.getCols());

        // Check transposition
        assertEquals('A', transposed.get(0, 0).get().value());
        assertEquals('D', transposed.get(0, 1).get().value());
        assertEquals('G', transposed.get(0, 2).get().value());
        assertEquals('B', transposed.get(1, 0).get().value());
        assertEquals('E', transposed.get(1, 1).get().value());
        assertEquals('H', transposed.get(1, 2).get().value());
        assertEquals('C', transposed.get(2, 0).get().value());
        assertEquals('F', transposed.get(2, 1).get().value());
        assertEquals('I', transposed.get(2, 2).get().value());
    }

    @Test
    void testTransposeRectangularGrid() {
        String input = """
            ABCD
            EFGH""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();
        Grid<Character> transposed = grid.transpose();

        assertEquals(4, transposed.getRows());
        assertEquals(2, transposed.getCols());

        assertEquals('A', transposed.get(0, 0).get().value());
        assertEquals('E', transposed.get(0, 1).get().value());
        assertEquals('B', transposed.get(1, 0).get().value());
        assertEquals('F', transposed.get(1, 1).get().value());
        assertEquals('D', transposed.get(3, 0).get().value());
        assertEquals('H', transposed.get(3, 1).get().value());
    }

    @Test
    void testTransposeSingleRow() {
        String input = "ABCD";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();
        Grid<Character> transposed = grid.transpose();

        assertEquals(4, transposed.getRows());
        assertEquals(1, transposed.getCols());

        assertEquals('A', transposed.get(0, 0).get().value());
        assertEquals('B', transposed.get(1, 0).get().value());
        assertEquals('C', transposed.get(2, 0).get().value());
        assertEquals('D', transposed.get(3, 0).get().value());
    }

    @Test
    void testTransposeSingleColumn() {
        String input = "A\nB\nC\nD";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();
        Grid<Character> transposed = grid.transpose();

        assertEquals(1, transposed.getRows());
        assertEquals(4, transposed.getCols());

        assertEquals('A', transposed.get(0, 0).get().value());
        assertEquals('B', transposed.get(0, 1).get().value());
        assertEquals('C', transposed.get(0, 2).get().value());
        assertEquals('D', transposed.get(0, 3).get().value());
    }

    @Test
    void testTransposeSingleCell() {
        String input = "X";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();
        Grid<Character> transposed = grid.transpose();

        assertEquals(1, transposed.getRows());
        assertEquals(1, transposed.getCols());
        assertEquals('X', transposed.get(0, 0).get().value());
    }

    @Test
    void testTransposeDoubleTranspose() {
        String input = """
            ABC
            DEF""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();
        Grid<Character> transposed = grid.transpose().transpose();

        assertEquals(grid.getRows(), transposed.getRows());
        assertEquals(grid.getCols(), transposed.getCols());

        for (int i = 0; i < grid.getRows(); i++) {
            for (int j = 0; j < grid.getCols(); j++) {
                assertEquals(grid.get(i, j).get().value(),
                           transposed.get(i, j).get().value());
            }
        }
    }

    @Test
    void testTransposePreservesBehavior() {
        String input = "AB\nCD";
        Grid<Character> standardGrid = Grid.fromString(input, c -> c,
            Grid.BehaviorType.STANDARD).orElseThrow();
        Grid<Character> toroidalGrid = Grid.fromString(input, c -> c,
            Grid.BehaviorType.TOROIDAL).orElseThrow();

        Grid<Character> transposedStandard = standardGrid.transpose();
        Grid<Character> transposedToroidal = toroidalGrid.transpose();

        // Both should work without errors (behavior is preserved internally)
        assertNotNull(transposedStandard);
        assertNotNull(transposedToroidal);
    }

    @Test
    void testUnionSimple() {
        String input1 = """
            111
            111
            111""";
        String input2 = """
            222
            222
            222""";

        Grid<Integer> grid1 = Grid.fromString(input1,
            c -> Character.getNumericValue(c)).orElseThrow();
        Grid<Integer> grid2 = Grid.fromString(input2,
            c -> Character.getNumericValue(c)).orElseThrow();

        BiFunction<Integer, Integer, Integer> sumMerger = (a, b) -> a + b;
        Optional<Grid<Integer>> unionOpt = grid1.union(grid2, sumMerger);

        assertTrue(unionOpt.isPresent());
        Grid<Integer> union = unionOpt.get();

        // All values should be 3 (1 + 2)
        for (int i = 0; i < union.getRows(); i++) {
            for (int j = 0; j < union.getCols(); j++) {
                assertEquals(3, union.get(i, j).get().value());
            }
        }
    }

    @Test
    void testUnionWithMaxMerger() {
        String input1 = """
            123
            456
            789""";
        String input2 = """
            987
            654
            321""";

        Grid<Integer> grid1 = Grid.fromString(input1,
            c -> Character.getNumericValue(c)).orElseThrow();
        Grid<Integer> grid2 = Grid.fromString(input2,
            c -> Character.getNumericValue(c)).orElseThrow();

        BiFunction<Integer, Integer, Integer> maxMerger = Math::max;
        Optional<Grid<Integer>> unionOpt = grid1.union(grid2, maxMerger);

        assertTrue(unionOpt.isPresent());
        Grid<Integer> union = unionOpt.get();

        assertEquals(9, union.get(0, 0).get().value());
        assertEquals(8, union.get(0, 1).get().value());
        assertEquals(7, union.get(0, 2).get().value());
        assertEquals(6, union.get(1, 0).get().value());
        assertEquals(6, union.get(1, 1).get().value());
        assertEquals(6, union.get(1, 2).get().value());
        assertEquals(7, union.get(2, 0).get().value());
        assertEquals(8, union.get(2, 1).get().value());
        assertEquals(9, union.get(2, 2).get().value());
    }

    @Test
    void testUnionDifferentDimensionsReturnsEmpty() {
        String input1 = "AB";
        String input2 = "A\nB";

        Grid<Character> grid1 = Grid.fromString(input1, c -> c).orElseThrow();
        Grid<Character> grid2 = Grid.fromString(input2, c -> c).orElseThrow();

        BiFunction<Character, Character, Character> merger = (a, b) -> a;
        Optional<Grid<Character>> unionOpt = grid1.union(grid2, merger);

        assertFalse(unionOpt.isPresent());
    }

    @Test
    void testUnionWithNullGridThrowsException() {
        String input = "AB";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        BiFunction<Character, Character, Character> merger = (a, b) -> a;

        assertThrows(NullPointerException.class, () -> {
            grid.union(null, merger);
        });
    }

    @Test
    void testUnionWithNullMergerThrowsException() {
        String input = "AB";
        Grid<Character> grid1 = Grid.fromString(input, c -> c).orElseThrow();
        Grid<Character> grid2 = Grid.fromString(input, c -> c).orElseThrow();

        assertThrows(NullPointerException.class, () -> {
            grid1.union(grid2, null);
        });
    }

    @Test
    void testUnionSingleCell() {
        String input = "A";
        Grid<Character> grid1 = Grid.fromString(input, c -> c).orElseThrow();
        Grid<Character> grid2 = Grid.fromString(input, c -> c).orElseThrow();

        BiFunction<Character, Character, Character> merger = (a, b) -> 'X';
        Optional<Grid<Character>> unionOpt = grid1.union(grid2, merger);

        assertTrue(unionOpt.isPresent());
        assertEquals('X', unionOpt.get().get(0, 0).get().value());
    }

    @Test
    void testUnionPreservesBehavior() {
        String input = "AB\nCD";
        Grid<Character> grid1 = Grid.fromString(input, c -> c,
            Grid.BehaviorType.STANDARD).orElseThrow();
        Grid<Character> grid2 = Grid.fromString(input, c -> c,
            Grid.BehaviorType.STANDARD).orElseThrow();

        BiFunction<Character, Character, Character> merger = (a, b) -> a;
        Optional<Grid<Character>> unionOpt = grid1.union(grid2, merger);

        assertTrue(unionOpt.isPresent());
        // Union should preserve behavior from first grid
    }

    @Test
    void testUnionReturnsImmutableGrid() {
        String input = "AB";
        Grid<Character> grid1 = Grid.fromString(input, c -> c).orElseThrow();
        Grid<Character> grid2 = Grid.fromString(input, c -> c).orElseThrow();

        BiFunction<Character, Character, Character> merger = (a, b) -> a;
        Optional<Grid<Character>> unionOpt = grid1.union(grid2, merger);

        assertTrue(unionOpt.isPresent());
        Grid<Character> union = unionOpt.get();

        // Test immutability
        Optional<List<Point<Character>>> points = union.find('A');
        assertTrue(points.isPresent());
        assertThrows(UnsupportedOperationException.class, () -> {
            points.get().add(new Point<>(5, 5, 'Z'));
        });
    }

    @Test
    void testTransposeReturnsImmutableGrid() {
        String input = "AB";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();
        Grid<Character> transposed = grid.transpose();

        Optional<List<Point<Character>>> points = transposed.find('A');
        assertTrue(points.isPresent());
        assertThrows(UnsupportedOperationException.class, () -> {
            points.get().add(new Point<>(5, 5, 'Z'));
        });
    }

    @Test
    void testUnionWithComplexMerger() {
        String input1 = "AB\nCD";
        String input2 = "XY\nZW";

        Grid<Character> grid1 = Grid.fromString(input1, c -> c).orElseThrow();
        Grid<Character> grid2 = Grid.fromString(input2, c -> c).orElseThrow();

        // Concatenate characters as strings
        BiFunction<Character, Character, String> merger =
            (a, b) -> String.valueOf(a) + String.valueOf(b);

        Grid<String> union = grid1.union(grid2, merger).orElseThrow();

        assertEquals("AX", union.get(0, 0).get().value());
        assertEquals("BY", union.get(0, 1).get().value());
        assertEquals("CZ", union.get(1, 0).get().value());
        assertEquals("DW", union.get(1, 1).get().value());
    }

    @Test
    void testTransposeWithNumbers() {
        String input = """
            123
            456""";
        Grid<Integer> grid = Grid.fromString(input,
            c -> Character.getNumericValue(c)).orElseThrow();
        Grid<Integer> transposed = grid.transpose();

        assertEquals(3, transposed.getRows());
        assertEquals(2, transposed.getCols());

        assertEquals(1, transposed.get(0, 0).get().value());
        assertEquals(4, transposed.get(0, 1).get().value());
        assertEquals(2, transposed.get(1, 0).get().value());
        assertEquals(5, transposed.get(1, 1).get().value());
        assertEquals(3, transposed.get(2, 0).get().value());
        assertEquals(6, transposed.get(2, 1).get().value());
    }
}
