package wklm.aoc;

import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Grid algorithm methods (pathfinding, connected components).
 */
class GridAlgorithmTest {

    @Test
    void testFindShortestPathDijkstraSimplePath() {
        String input = """
            ...
            ...
            ...""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        Optional<List<Point<Character>>> pathOpt = grid.findShortestPathDijkstra(
            0, 0, 2, 2, Connectivity.FOUR
        );

        assertTrue(pathOpt.isPresent());
        List<Point<Character>> path = pathOpt.get();
        assertEquals(5, path.size()); // Manhattan distance: 4 steps + start
        assertEquals(0, path.get(0).x());
        assertEquals(0, path.get(0).y());
        assertEquals(2, path.get(4).x());
        assertEquals(2, path.get(4).y());
    }

    @Test
    void testFindShortestPathDijkstraSameStartAndEnd() {
        String input = "...";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        Optional<List<Point<Character>>> pathOpt = grid.findShortestPathDijkstra(
            0, 0, 0, 0, Connectivity.FOUR
        );

        assertTrue(pathOpt.isPresent());
        assertEquals(1, pathOpt.get().size());
    }

    @Test
    void testFindShortestPathDijkstraAdjacentCells() {
        String input = "..";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        Optional<List<Point<Character>>> pathOpt = grid.findShortestPathDijkstra(
            0, 0, 0, 1, Connectivity.FOUR
        );

        assertTrue(pathOpt.isPresent());
        assertEquals(2, pathOpt.get().size());
    }

    @Test
    void testFindShortestPathDijkstraInvalidStartReturnsEmpty() {
        String input = "...";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        Optional<List<Point<Character>>> pathOpt = grid.findShortestPathDijkstra(
            -1, 0, 0, 2, Connectivity.FOUR
        );

        assertFalse(pathOpt.isPresent());
    }

    @Test
    void testFindShortestPathDijkstraInvalidEndReturnsEmpty() {
        String input = "...";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        Optional<List<Point<Character>>> pathOpt = grid.findShortestPathDijkstra(
            0, 0, 5, 5, Connectivity.FOUR
        );

        assertFalse(pathOpt.isPresent());
    }

    @Test
    void testFindShortestPathDijkstraWithCustomCosts() {
        String input = """
            123
            456
            789""";
        Grid<Integer> grid = Grid.fromString(input,
            c -> Character.getNumericValue(c)).orElseThrow();

        // Cost is the value of the destination cell
        Grid.MovementCostFunction<Integer> costFunc = (from, to) -> to.value().doubleValue();

        Optional<List<Point<Integer>>> pathOpt = grid.findShortestPathDijkstra(
            0, 0, 2, 2, Connectivity.FOUR, costFunc
        );

        assertTrue(pathOpt.isPresent());
        List<Point<Integer>> path = pathOpt.get();
        assertTrue(path.size() >= 1);
        assertEquals(1, path.get(0).value()); // Start at 1
        assertEquals(9, path.get(path.size() - 1).value()); // End at 9
    }

    @Test
    void testFindShortestPathDijkstraEightConnectivity() {
        String input = """
            ...
            ...
            ...""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        Optional<List<Point<Character>>> pathOpt = grid.findShortestPathDijkstra(
            0, 0, 2, 2, Connectivity.EIGHT
        );

        assertTrue(pathOpt.isPresent());
        // With 8-connectivity, diagonal is possible, so path is shorter
        assertEquals(3, pathOpt.get().size()); // Diagonal path: start, middle, end
    }

    @Test
    void testFindShortestPathDijkstraWithNullConnectivityThrowsException() {
        String input = "...";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        assertThrows(NullPointerException.class, () -> {
            grid.findShortestPathDijkstra(0, 0, 0, 2, null);
        });
    }

    @Test
    void testFindShortestPathDijkstraReturnsImmutableList() {
        String input = "...";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        Optional<List<Point<Character>>> pathOpt = grid.findShortestPathDijkstra(
            0, 0, 0, 2, Connectivity.FOUR
        );

        assertTrue(pathOpt.isPresent());
        assertThrows(UnsupportedOperationException.class, () -> {
            pathOpt.get().add(new Point<>(5, 5, 'Z'));
        });
    }

    @Test
    void testGetConnectedPointsSimple() {
        String input = """
            AAA
            BBB
            AAA""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        List<List<Point<Character>>> components = grid.getConnectedPoints('A', Connectivity.FOUR);

        assertEquals(2, components.size()); // Two separate A regions
        // Top row has 3 A's, bottom row has 3 A's
        assertTrue(components.stream().anyMatch(comp -> comp.size() == 3));
    }

    @Test
    void testGetConnectedPointsSingleComponent() {
        String input = """
            AAA
            AAA
            AAA""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        List<List<Point<Character>>> components = grid.getConnectedPoints('A', Connectivity.FOUR);

        assertEquals(1, components.size());
        assertEquals(9, components.get(0).size());
    }

    @Test
    void testGetConnectedPointsNoMatches() {
        String input = """
            AAA
            AAA
            AAA""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        List<List<Point<Character>>> components = grid.getConnectedPoints('B', Connectivity.FOUR);

        assertTrue(components.isEmpty());
    }

    @Test
    void testGetConnectedPointsEightConnectivity() {
        String input = """
            A.A
            ...
            A.A""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        List<List<Point<Character>>> components4 = grid.getConnectedPoints('A', Connectivity.FOUR);
        assertEquals(4, components4.size()); // All separate with 4-connectivity

        List<List<Point<Character>>> components8 = grid.getConnectedPoints('A', Connectivity.EIGHT);
        assertEquals(1, components8.size()); // All connected with 8-connectivity
        assertEquals(4, components8.get(0).size());
    }

    @Test
    void testGetConnectedPointsComplexShape() {
        String input = """
            AAA
            A.A
            AAA""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        List<List<Point<Character>>> components = grid.getConnectedPoints('A', Connectivity.FOUR);

        assertEquals(1, components.size()); // All A's are connected in a ring
        assertEquals(8, components.get(0).size());
    }

    @Test
    void testGetConnectedPointsWithNullValueThrowsException() {
        String input = "AAA";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        assertThrows(NullPointerException.class, () -> {
            grid.getConnectedPoints(null, Connectivity.FOUR);
        });
    }

    @Test
    void testGetConnectedPointsWithNullConnectivityThrowsException() {
        String input = "AAA";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        assertThrows(NullPointerException.class, () -> {
            grid.getConnectedPoints('A', null);
        });
    }

    @Test
    void testGetConnectedPointsReturnsImmutableLists() {
        String input = """
            AAA
            BBB""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        List<List<Point<Character>>> components = grid.getConnectedPoints('A', Connectivity.FOUR);

        // Outer list should be immutable
        assertThrows(UnsupportedOperationException.class, () -> {
            components.add(new ArrayList<>());
        });

        // Inner lists should be immutable
        assertThrows(UnsupportedOperationException.class, () -> {
            components.get(0).add(new Point<>(5, 5, 'Z'));
        });
    }

    @Test
    void testGetConnectedPointsMultipleComponents() {
        String input = """
            A.A
            ...
            A.A""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        List<List<Point<Character>>> components = grid.getConnectedPoints('A', Connectivity.FOUR);

        assertEquals(4, components.size());
        // Each component should have exactly 1 point
        assertTrue(components.stream().allMatch(comp -> comp.size() == 1));
    }

    @Test
    void testGetConnectedPointsSingleCell() {
        String input = "A";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        List<List<Point<Character>>> components = grid.getConnectedPoints('A', Connectivity.FOUR);

        assertEquals(1, components.size());
        assertEquals(1, components.get(0).size());
        assertEquals('A', components.get(0).get(0).value());
    }

    @Test
    void testFindShortestPathDijkstraLargerGrid() {
        String input = """
            .....
            .....
            .....
            .....
            .....""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        Optional<List<Point<Character>>> pathOpt = grid.findShortestPathDijkstra(
            0, 0, 4, 4, Connectivity.FOUR
        );

        assertTrue(pathOpt.isPresent());
        assertEquals(9, pathOpt.get().size()); // Manhattan distance 8 + start
    }

    @Test
    void testGetConnectedPointsWithDifferentValues() {
        String input = """
            AAB
            ABB
            BBA""";
        Grid<Character> grid = Grid.fromString(input, c -> c).orElseThrow();

        List<List<Point<Character>>> aComponents = grid.getConnectedPoints('A', Connectivity.FOUR);
        List<List<Point<Character>>> bComponents = grid.getConnectedPoints('B', Connectivity.FOUR);

        assertEquals(3, aComponents.size()); // Three separate A's
        assertEquals(1, bComponents.size()); // All B's connected
        assertEquals(6, bComponents.get(0).size());
    }

    @Test
    void testDijkstraWithHighCostPath() {
        String input = """
            111
            191
            111""";
        Grid<Integer> grid = Grid.fromString(input,
            c -> Character.getNumericValue(c)).orElseThrow();

        Grid.MovementCostFunction<Integer> costFunc = (from, to) -> to.value().doubleValue();

        Optional<List<Point<Integer>>> pathOpt = grid.findShortestPathDijkstra(
            0, 0, 2, 2, Connectivity.FOUR, costFunc
        );

        assertTrue(pathOpt.isPresent());
        List<Point<Integer>> path = pathOpt.get();

        // Should avoid the 9 in the middle
        assertFalse(path.stream().anyMatch(p -> p.value() == 9));
    }
}
