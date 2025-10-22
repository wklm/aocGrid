package wklm.aoc;

import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for toroidal grid behavior (wrap-around at edges).
 */
class GridToroidalTest {

    @Test
    void testToroidalGridCreation() {
        String input = """
            ABC
            DEF
            GHI""";

        Optional<Grid<Character>> gridOpt = Grid.fromToroidalString(input, c -> c);
        assertTrue(gridOpt.isPresent());

        Grid<Character> grid = gridOpt.get();
        assertEquals(3, grid.getRows());
        assertEquals(3, grid.getCols());
    }

    @Test
    void testToroidalNeighborsTopEdge() {
        String input = """
            ABC
            DEF
            GHI""";
        Grid<Character> grid = Grid.fromToroidalString(input, c -> c).orElseThrow();

        // Top-left corner should wrap to bottom and right
        Point<Character> topLeft = grid.get(0, 0).get(); // 'A'
        List<Point<Character>> neighbors = grid.getNeighbors(topLeft, Connectivity.FOUR);

        assertEquals(4, neighbors.size());

        Set<Character> neighborValues = new HashSet<>();
        neighbors.forEach(p -> neighborValues.add(p.value()));

        // Should have: B (right), D (down), G (wrap up), C (wrap left)
        assertTrue(neighborValues.contains('B')); // Right
        assertTrue(neighborValues.contains('D')); // Down
        assertTrue(neighborValues.contains('G')); // Wrap up (bottom of same column)
        assertTrue(neighborValues.contains('C')); // Wrap left (right edge of same row)
    }

    @Test
    void testToroidalNeighborsBottomEdge() {
        String input = """
            ABC
            DEF
            GHI""";
        Grid<Character> grid = Grid.fromToroidalString(input, c -> c).orElseThrow();

        Point<Character> bottomRight = grid.get(2, 2).get(); // 'I'
        List<Point<Character>> neighbors = grid.getNeighbors(bottomRight, Connectivity.FOUR);

        assertEquals(4, neighbors.size());

        Set<Character> neighborValues = new HashSet<>();
        neighbors.forEach(p -> neighborValues.add(p.value()));

        // Should have: H (left), F (up), C (wrap down), G (wrap right)
        assertTrue(neighborValues.contains('H')); // Left
        assertTrue(neighborValues.contains('F')); // Up
        assertTrue(neighborValues.contains('C')); // Wrap down (top of same column)
        assertTrue(neighborValues.contains('G')); // Wrap right (left edge of same row)
    }

    @Test
    void testToroidalNeighborsLeftEdge() {
        String input = """
            ABC
            DEF
            GHI""";
        Grid<Character> grid = Grid.fromToroidalString(input, c -> c).orElseThrow();

        Point<Character> leftMiddle = grid.get(1, 0).get(); // 'D'
        List<Point<Character>> neighbors = grid.getNeighbors(leftMiddle, Connectivity.FOUR);

        assertEquals(4, neighbors.size());

        Set<Character> neighborValues = new HashSet<>();
        neighbors.forEach(p -> neighborValues.add(p.value()));

        // Should have: A (up), E (right), G (down), F (wrap left)
        assertTrue(neighborValues.contains('A')); // Up
        assertTrue(neighborValues.contains('E')); // Right
        assertTrue(neighborValues.contains('G')); // Down
        assertTrue(neighborValues.contains('F')); // Wrap left (right edge)
    }

    @Test
    void testToroidalNeighborsRightEdge() {
        String input = """
            ABC
            DEF
            GHI""";
        Grid<Character> grid = Grid.fromToroidalString(input, c -> c).orElseThrow();

        Point<Character> rightMiddle = grid.get(1, 2).get(); // 'F'
        List<Point<Character>> neighbors = grid.getNeighbors(rightMiddle, Connectivity.FOUR);

        assertEquals(4, neighbors.size());

        Set<Character> neighborValues = new HashSet<>();
        neighbors.forEach(p -> neighborValues.add(p.value()));

        // Should have: C (up), E (left), I (down), D (wrap right)
        assertTrue(neighborValues.contains('C')); // Up
        assertTrue(neighborValues.contains('E')); // Left
        assertTrue(neighborValues.contains('I')); // Down
        assertTrue(neighborValues.contains('D')); // Wrap right (left edge)
    }

    @Test
    void testToroidalNeighborsEightConnectivity() {
        String input = """
            ABC
            DEF
            GHI""";
        Grid<Character> grid = Grid.fromToroidalString(input, c -> c).orElseThrow();

        Point<Character> topLeft = grid.get(0, 0).get(); // 'A'
        List<Point<Character>> neighbors = grid.getNeighbors(topLeft, Connectivity.EIGHT);

        assertEquals(8, neighbors.size()); // Should have 8 neighbors including diagonals

        Set<Character> neighborValues = new HashSet<>();
        neighbors.forEach(p -> neighborValues.add(p.value()));

        // All 8 neighbors with wrapping
        assertTrue(neighborValues.contains('B')); // Right
        assertTrue(neighborValues.contains('D')); // Down
        assertTrue(neighborValues.contains('E')); // Down-Right
        assertTrue(neighborValues.contains('G')); // Wrap up
        assertTrue(neighborValues.contains('C')); // Wrap left
        assertTrue(neighborValues.contains('I')); // Wrap up-left (diagonal)
        assertTrue(neighborValues.contains('H')); // Wrap up-right (diagonal)
        assertTrue(neighborValues.contains('F')); // Wrap down-left (diagonal)
    }

    @Test
    void testToroidalVsStandardBehaviorCenter() {
        String input = """
            ABC
            DEF
            GHI""";
        Grid<Character> standardGrid = Grid.fromString(input, c -> c).orElseThrow();
        Grid<Character> toroidalGrid = Grid.fromToroidalString(input, c -> c).orElseThrow();

        Point<Character> centerStandard = standardGrid.get(1, 1).get();
        Point<Character> centerToroidal = toroidalGrid.get(1, 1).get();

        List<Point<Character>> standardNeighbors =
            standardGrid.getNeighbors(centerStandard, Connectivity.FOUR);
        List<Point<Character>> toroidalNeighbors =
            toroidalGrid.getNeighbors(centerToroidal, Connectivity.FOUR);

        // Center should have same neighbors in both cases
        assertEquals(4, standardNeighbors.size());
        assertEquals(4, toroidalNeighbors.size());
    }

    @Test
    void testToroidalVsStandardBehaviorCorner() {
        String input = """
            ABC
            DEF
            GHI""";
        Grid<Character> standardGrid = Grid.fromString(input, c -> c).orElseThrow();
        Grid<Character> toroidalGrid = Grid.fromToroidalString(input, c -> c).orElseThrow();

        Point<Character> cornerStandard = standardGrid.get(0, 0).get();
        Point<Character> cornerToroidal = toroidalGrid.get(0, 0).get();

        List<Point<Character>> standardNeighbors =
            standardGrid.getNeighbors(cornerStandard, Connectivity.FOUR);
        List<Point<Character>> toroidalNeighbors =
            toroidalGrid.getNeighbors(cornerToroidal, Connectivity.FOUR);

        // Standard grid has 2 neighbors at corner, toroidal has 4
        assertEquals(2, standardNeighbors.size());
        assertEquals(4, toroidalNeighbors.size());
    }

    @Test
    void testToroidalConnectedComponents() {
        String input = """
            A.A
            ...
            A.A""";
        Grid<Character> toroidalGrid = Grid.fromToroidalString(input, c -> c).orElseThrow();

        // With toroidal wrapping and 4-connectivity, corners are still separate
        List<List<Point<Character>>> components =
            toroidalGrid.getConnectedPoints('A', Connectivity.FOUR);

        assertEquals(4, components.size());
    }

    @Test
    void testToroidalSmallGrid() {
        String input = "AB";
        Grid<Character> grid = Grid.fromToroidalString(input, c -> c).orElseThrow();

        Point<Character> a = grid.get(0, 0).get();
        List<Point<Character>> neighbors = grid.getNeighbors(a, Connectivity.FOUR);

        assertEquals(4, neighbors.size()); // Left, right, up, down all wrap to itself or B
    }

    @Test
    void testToroidalSingleRowGrid() {
        String input = "ABCD";
        Grid<Character> grid = Grid.fromToroidalString(input, c -> c).orElseThrow();

        Point<Character> a = grid.get(0, 0).get();
        List<Point<Character>> neighbors = grid.getNeighbors(a, Connectivity.FOUR);

        // Single row: left/right neighbors exist, up/down wrap to same row
        assertEquals(4, neighbors.size());

        Set<Character> neighborValues = new HashSet<>();
        neighbors.forEach(p -> neighborValues.add(p.value()));

        assertTrue(neighborValues.contains('B')); // Right
        assertTrue(neighborValues.contains('D')); // Wrap left
        // Up and down both wrap to same row, so A appears twice in neighbors
        assertTrue(neighbors.stream().filter(p -> p.value() == 'A').count() == 2);
    }

    @Test
    void testToroidalSingleColumnGrid() {
        String input = "A\nB\nC\nD";
        Grid<Character> grid = Grid.fromToroidalString(input, c -> c).orElseThrow();

        Point<Character> a = grid.get(0, 0).get();
        List<Point<Character>> neighbors = grid.getNeighbors(a, Connectivity.FOUR);

        assertEquals(4, neighbors.size());

        Set<Character> neighborValues = new HashSet<>();
        neighbors.forEach(p -> neighborValues.add(p.value()));

        assertTrue(neighborValues.contains('B')); // Down
        assertTrue(neighborValues.contains('D')); // Wrap up
        // Left and right both wrap to same column, so A appears twice
        assertTrue(neighbors.stream().filter(p -> p.value() == 'A').count() == 2);
    }

    @Test
    void testToroidalPathfinding() {
        String input = """
            ...
            ...
            ...""";
        Grid<Character> grid = Grid.fromToroidalString(input, c -> c).orElseThrow();

        // In toroidal grid, wrapping might provide shorter paths
        Optional<List<Point<Character>>> pathOpt = grid.findShortestPathDijkstra(
            0, 0, 2, 2, Connectivity.FOUR
        );

        assertTrue(pathOpt.isPresent());
        // Path exists (exact length depends on pathfinding implementation with wrapping)
        assertTrue(pathOpt.get().size() > 0);
    }

    @Test
    void testToroidalTranspose() {
        String input = """
            ABC
            DEF""";
        Grid<Character> grid = Grid.fromToroidalString(input, c -> c).orElseThrow();
        Grid<Character> transposed = grid.transpose();

        // Transpose should preserve toroidal behavior
        Point<Character> corner = transposed.get(0, 0).get();
        List<Point<Character>> neighbors = transposed.getNeighbors(corner, Connectivity.FOUR);

        // Should still have 4 neighbors (toroidal wrapping preserved)
        assertEquals(4, neighbors.size());
    }

    @Test
    void testProgrammaticConstructorWithToroidalBehavior() {
        Dimensions dims = new Dimensions(3, 3);
        List<Point<Character>> points = List.of(
            new Point<>(0, 0, 'A'),
            new Point<>(2, 2, 'B')
        );

        Grid<Character> grid = new Grid<>(points, dims, Optional.of('.'),
            Optional.of(Grid.BehaviorType.TOROIDAL));

        Point<Character> a = grid.get(0, 0).get();
        List<Point<Character>> neighbors = grid.getNeighbors(a, Connectivity.FOUR);

        // With toroidal behavior, corner should have 4 neighbors
        assertEquals(4, neighbors.size());
    }
}
