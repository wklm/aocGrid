package wklm.aoc;

import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Grid visualization methods (toString, toGridString, toGraphString).
 */
class GridVisualizationTest {

    @Test
    void testToStringSimple() {
        String input = """
            ABC
            DEF
            GHI""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        String output = grid.toString();
        assertEquals("ABC\nDEF\nGHI", output);
    }

    @Test
    void testToStringSingleCell() {
        String input = "X";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        String output = grid.toString();
        assertEquals("X", output);
    }

    @Test
    void testToStringSingleRow() {
        String input = "ABCD";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        String output = grid.toString();
        assertEquals("ABCD", output);
    }

    @Test
    void testToStringSingleColumn() {
        String input = "A\nB\nC";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        String output = grid.toString();
        assertEquals("A\nB\nC", output);
    }

    @Test
    void testToStringWithNullValues() {
        Dimensions dims = new Dimensions(2, 2);
        List<Point<String>> points = List.of(
            new Point<>(0, 0, "A")
        );
        Grid<String> grid = new Grid<>(points, dims, Optional.empty(), Optional.empty());

        String output = grid.toString();
        // Null values should be represented as "_"
        assertTrue(output.contains("A"));
        assertTrue(output.contains("_"));
    }

    @Test
    void testToGridStringWithoutIndices() {
        String input = """
            AB
            CD""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        String output = grid.toGridString(false);
        assertTrue(output.contains("A"));
        assertTrue(output.contains("B"));
        assertTrue(output.contains("C"));
        assertTrue(output.contains("D"));
        assertFalse(output.contains("0")); // No indices
    }

    @Test
    void testToGridStringWithIndices() {
        String input = """
            AB
            CD""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        String output = grid.toGridString(true);
        assertTrue(output.contains("A"));
        assertTrue(output.contains("B"));
        assertTrue(output.contains("C"));
        assertTrue(output.contains("D"));
        assertTrue(output.contains("0")); // Row/col indices
        assertTrue(output.contains("1"));
        assertTrue(output.contains("|")); // Border
        assertTrue(output.contains("+")); // Border
    }

    @Test
    void testToGridStringSingleCellWithIndices() {
        String input = "X";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        String output = grid.toGridString(true);
        assertTrue(output.contains("X"));
        assertTrue(output.contains("0"));
    }

    @Test
    void testToGridStringLargerGrid() {
        String input = """
            ABCDE
            FGHIJ
            KLMNO""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        String output = grid.toGridString(true);
        // Should contain all letters
        for (char c = 'A'; c <= 'O'; c++) {
            assertTrue(output.contains(String.valueOf(c)));
        }
        // Should contain indices 0-4 for columns
        for (int i = 0; i < 5; i++) {
            assertTrue(output.contains(String.valueOf(i)));
        }
    }

    @Test
    void testToGraphStringFourConnectivity() {
        String input = """
            AB
            CD""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        String output = grid.toGraphString(Connectivity.FOUR, false, null);

        // Should contain "Node(x,y)" entries
        assertTrue(output.contains("Node(0,0)")); // A
        assertTrue(output.contains("Node(0,1)")); // B
        assertTrue(output.contains("Node(1,0)")); // C
        assertTrue(output.contains("Node(1,1)")); // D

        // Should contain adjacency information
        assertTrue(output.contains("Graph Representation"));
        assertTrue(output.contains("->")); // Arrow for adjacency
    }

    @Test
    void testToGraphStringEightConnectivity() {
        String input = """
            AB
            CD""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        String output = grid.toGraphString(Connectivity.EIGHT, false, null);

        // Should show more neighbors with 8-connectivity
        assertTrue(output.contains("Node(0,0)"));
        assertTrue(output.contains("->"));
    }

    @Test
    void testToGraphStringWithWeights() {
        String input = """
            12
            34""";
        Grid<Integer> grid = Grid.fromString(input,
            c -> Character.getNumericValue(c)).orElseThrow();

        Grid.MovementCostFunction<Integer> costFunc = (from, to) -> to.value().doubleValue();

        String output = grid.toGraphString(Connectivity.FOUR, true, costFunc);

        // Should contain cost information
        assertTrue(output.contains("cost="));
        assertTrue(output.contains("Node(0,0)"));
    }

    @Test
    void testToGraphStringSingleCell() {
        String input = "X";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        String output = grid.toGraphString(Connectivity.FOUR, false, null);

        assertTrue(output.contains("Node(0,0)"));
        assertTrue(output.contains("no neighbors"));
    }

    @Test
    void testToGraphStringNullConnectivityThrowsException() {
        String input = "AB";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        assertThrows(NullPointerException.class, () -> {
            grid.toGraphString(null, false, null);
        });
    }

    @Test
    void testToStringWithNumbers() {
        String input = """
            123
            456
            789""";
        Grid<Integer> grid = Grid.fromString(input,
            c -> Character.getNumericValue(c)).orElseThrow();

        String output = grid.toString();
        assertEquals("123\n456\n789", output);
    }

    @Test
    void testToGridStringWithNullValues() {
        Dimensions dims = new Dimensions(2, 2);
        List<Point<Character>> points = List.of(
            new Point<>(0, 0, 'A')
        );
        Grid<Character> grid = new Grid<>(points, dims, Optional.empty(), Optional.empty());

        String output = grid.toGridString(false);
        assertTrue(output.contains("A"));
        assertTrue(output.contains(".")); // Null values shown as "."
    }

    @Test
    void testToStringMultipleRows() {
        String input = """
            AAA
            BBB
            CCC
            DDD
            EEE""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        String output = grid.toString();
        String[] lines = output.split("\n");
        assertEquals(5, lines.length);
        assertEquals("AAA", lines[0]);
        assertEquals("EEE", lines[4]);
    }

    @Test
    void testToGraphStringCornerNode() {
        String input = """
            ABC
            DEF
            GHI""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        String output = grid.toGraphString(Connectivity.FOUR, false, null);

        // Top-left corner should have 2 neighbors
        assertTrue(output.contains("Node(0,0)"));

        // Check that adjacency list is present
        int node00Index = output.indexOf("Node(0,0)");
        int nextNodeIndex = output.indexOf("Node", node00Index + 1);
        String node00Section = output.substring(node00Index, nextNodeIndex);

        // Should list neighbors
        assertTrue(node00Section.contains("->"));
    }

    @Test
    void testToGraphStringWithoutWeightsDoesNotShowCost() {
        String input = "AB";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        String output = grid.toGraphString(Connectivity.FOUR, false, null);

        assertFalse(output.contains("cost="));
    }

    @Test
    void testToGridStringEmptyRow() {
        String input = "";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        String output = grid.toGridString(false);
        assertNotNull(output);
    }

    @Test
    void testToStringReturnsNotNull() {
        String input = "A";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        assertNotNull(grid.toString());
        assertNotNull(grid.toGridString(false));
        assertNotNull(grid.toGridString(true));
        assertNotNull(grid.toGraphString(Connectivity.FOUR, false, null));
    }
}
