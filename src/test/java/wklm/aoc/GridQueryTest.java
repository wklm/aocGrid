package wklm.aoc;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Grid query methods (get, find, getNeighbors, etc.).
 */
class GridQueryTest {

    private Grid<Character> grid;

    @BeforeEach
    void setUp() {
        String input = """
            ABC
            DEF
            GHI""";
        grid = Grid.fromString(input, c -> c).orElseThrow();
    }

    @Test
    void testGetValidCoordinates() {
        Optional<Point<Character>> point = grid.get(0, 0);
        assertTrue(point.isPresent());
        assertEquals('A', point.get().value());
        assertEquals(0, point.get().x());
        assertEquals(0, point.get().y());
    }

    @Test
    void testGetAllCorners() {
        assertEquals('A', grid.get(0, 0).get().value()); // Top-left
        assertEquals('C', grid.get(0, 2).get().value()); // Top-right
        assertEquals('G', grid.get(2, 0).get().value()); // Bottom-left
        assertEquals('I', grid.get(2, 2).get().value()); // Bottom-right
    }

    @Test
    void testGetCenter() {
        assertEquals('E', grid.get(1, 1).get().value());
    }

    @Test
    void testGetNegativeRowReturnsEmpty() {
        Optional<Point<Character>> point = grid.get(-1, 0);
        assertFalse(point.isPresent());
    }

    @Test
    void testGetNegativeColReturnsEmpty() {
        Optional<Point<Character>> point = grid.get(0, -1);
        assertFalse(point.isPresent());
    }

    @Test
    void testGetRowTooLargeReturnsEmpty() {
        Optional<Point<Character>> point = grid.get(3, 0);
        assertFalse(point.isPresent());
    }

    @Test
    void testGetColTooLargeReturnsEmpty() {
        Optional<Point<Character>> point = grid.get(0, 3);
        assertFalse(point.isPresent());
    }

    @Test
    void testGetDimensions() {
        Dimensions dims = grid.getDimensions();
        assertEquals(3, dims.nrows());
        assertEquals(3, dims.ncols());
    }

    @Test
    void testGetRows() {
        assertEquals(3, grid.getRows());
    }

    @Test
    void testGetCols() {
        assertEquals(3, grid.getCols());
    }

    @Test
    void testFindExistingValue() {
        Optional<List<Point<Character>>> points = grid.find('E');
        assertTrue(points.isPresent());
        assertEquals(1, points.get().size());
        assertEquals('E', points.get().get(0).value());
        assertEquals(1, points.get().get(0).x());
        assertEquals(1, points.get().get(0).y());
    }

    @Test
    void testFindNonExistingValue() {
        Optional<List<Point<Character>>> points = grid.find('Z');
        assertFalse(points.isPresent());
    }

    @Test
    void testFindMultipleValues() {
        String input = """
            AAB
            ABA
            BAA""";
        Grid<Character> grid2 = Grid.fromString(input, c -> c).orElseThrow();

        Optional<List<Point<Character>>> aPoints = grid2.find('A');
        assertTrue(aPoints.isPresent());
        assertEquals(6, aPoints.get().size());

        Optional<List<Point<Character>>> bPoints = grid2.find('B');
        assertTrue(bPoints.isPresent());
        assertEquals(3, bPoints.get().size());
    }

    @Test
    void testGetUniqueValues() {
        List<Character> uniqueValues = grid.getUniqueValues();
        assertEquals(9, uniqueValues.size());
        assertTrue(uniqueValues.contains('A'));
        assertTrue(uniqueValues.contains('E'));
        assertTrue(uniqueValues.contains('I'));
    }

    @Test
    void testGetUniqueValuesWithDuplicates() {
        String input = """
            AAA
            BBB
            AAA""";
        Grid<Character> grid2 = Grid.fromString(input, c -> c).orElseThrow();

        List<Character> uniqueValues = grid2.getUniqueValues();
        assertEquals(2, uniqueValues.size());
        assertTrue(uniqueValues.contains('A'));
        assertTrue(uniqueValues.contains('B'));
    }

    @Test
    void testGetNeighborsFourConnectivityCenter() {
        Point<Character> center = grid.get(1, 1).get();
        List<Point<Character>> neighbors = grid.getNeighbors(center, Connectivity.FOUR);

        assertEquals(4, neighbors.size());
        assertTrue(neighbors.stream().anyMatch(p -> p.value() == 'B')); // Up
        assertTrue(neighbors.stream().anyMatch(p -> p.value() == 'H')); // Down
        assertTrue(neighbors.stream().anyMatch(p -> p.value() == 'D')); // Left
        assertTrue(neighbors.stream().anyMatch(p -> p.value() == 'F')); // Right
    }

    @Test
    void testGetNeighborsEightConnectivityCenter() {
        Point<Character> center = grid.get(1, 1).get();
        List<Point<Character>> neighbors = grid.getNeighbors(center, Connectivity.EIGHT);

        assertEquals(8, neighbors.size());
        // Check all 8 neighbors exist
        Set<Character> neighborValues = new HashSet<>();
        neighbors.forEach(p -> neighborValues.add(p.value()));

        assertTrue(neighborValues.contains('A')); // Up-Left
        assertTrue(neighborValues.contains('B')); // Up
        assertTrue(neighborValues.contains('C')); // Up-Right
        assertTrue(neighborValues.contains('D')); // Left
        assertTrue(neighborValues.contains('F')); // Right
        assertTrue(neighborValues.contains('G')); // Down-Left
        assertTrue(neighborValues.contains('H')); // Down
        assertTrue(neighborValues.contains('I')); // Down-Right
    }

    @Test
    void testGetNeighborsFourConnectivityTopLeftCorner() {
        Point<Character> topLeft = grid.get(0, 0).get();
        List<Point<Character>> neighbors = grid.getNeighbors(topLeft, Connectivity.FOUR);

        assertEquals(2, neighbors.size());
        assertTrue(neighbors.stream().anyMatch(p -> p.value() == 'B')); // Right
        assertTrue(neighbors.stream().anyMatch(p -> p.value() == 'D')); // Down
    }

    @Test
    void testGetNeighborsEightConnectivityTopLeftCorner() {
        Point<Character> topLeft = grid.get(0, 0).get();
        List<Point<Character>> neighbors = grid.getNeighbors(topLeft, Connectivity.EIGHT);

        assertEquals(3, neighbors.size());
        assertTrue(neighbors.stream().anyMatch(p -> p.value() == 'B')); // Right
        assertTrue(neighbors.stream().anyMatch(p -> p.value() == 'D')); // Down
        assertTrue(neighbors.stream().anyMatch(p -> p.value() == 'E')); // Down-Right
    }

    @Test
    void testGetNeighborsFourConnectivityEdge() {
        Point<Character> edge = grid.get(0, 1).get(); // 'B'
        List<Point<Character>> neighbors = grid.getNeighbors(edge, Connectivity.FOUR);

        assertEquals(3, neighbors.size());
        assertTrue(neighbors.stream().anyMatch(p -> p.value() == 'A')); // Left
        assertTrue(neighbors.stream().anyMatch(p -> p.value() == 'C')); // Right
        assertTrue(neighbors.stream().anyMatch(p -> p.value() == 'E')); // Down
    }

    @Test
    void testGetNeighborsWithNullPointReturnsEmptyList() {
        List<Point<Character>> neighbors = grid.getNeighbors(null, Connectivity.FOUR);
        assertTrue(neighbors.isEmpty());
    }

    @Test
    void testGetNeighborsWithNullConnectivityThrowsException() {
        Point<Character> point = grid.get(1, 1).get();
        assertThrows(NullPointerException.class, () -> {
            grid.getNeighbors(point, null);
        });
    }

    @Test
    void testFindReturnsImmutableList() {
        Optional<List<Point<Character>>> points = grid.find('E');
        assertTrue(points.isPresent());

        assertThrows(UnsupportedOperationException.class, () -> {
            points.get().add(new Point<>(5, 5, 'Z'));
        });
    }

    @Test
    void testGetUniqueValuesReturnsImmutableList() {
        List<Character> uniqueValues = grid.getUniqueValues();

        assertThrows(UnsupportedOperationException.class, () -> {
            uniqueValues.add('Z');
        });
    }

    @Test
    void testGetNeighborsReturnsImmutableList() {
        Point<Character> center = grid.get(1, 1).get();
        List<Point<Character>> neighbors = grid.getNeighbors(center, Connectivity.FOUR);

        assertThrows(UnsupportedOperationException.class, () -> {
            neighbors.add(new Point<>(5, 5, 'Z'));
        });
    }

    @Test
    void testSingleCellGridNeighbors() {
        String input = "X";
        Grid<Character> singleGrid = Grid.fromString(input, c -> c).orElseThrow();
        Point<Character> point = singleGrid.get(0, 0).get();

        List<Point<Character>> neighbors4 = singleGrid.getNeighbors(point, Connectivity.FOUR);
        assertEquals(0, neighbors4.size());

        List<Point<Character>> neighbors8 = singleGrid.getNeighbors(point, Connectivity.EIGHT);
        assertEquals(0, neighbors8.size());
    }

    @Test
    void testGetNeighborsInOneRowGrid() {
        String input = "ABC";
        Grid<Character> rowGrid = Grid.fromString(input, c -> c).orElseThrow();

        Point<Character> middle = rowGrid.get(0, 1).get(); // 'B'
        List<Point<Character>> neighbors = rowGrid.getNeighbors(middle, Connectivity.FOUR);

        assertEquals(2, neighbors.size());
        assertTrue(neighbors.stream().anyMatch(p -> p.value() == 'A'));
        assertTrue(neighbors.stream().anyMatch(p -> p.value() == 'C'));
    }

    @Test
    void testGetNeighborsInOneColumnGrid() {
        String input = "A\nB\nC";
        Grid<Character> colGrid = Grid.fromString(input, c -> c).orElseThrow();

        Point<Character> middle = colGrid.get(1, 0).get(); // 'B'
        List<Point<Character>> neighbors = colGrid.getNeighbors(middle, Connectivity.FOUR);

        assertEquals(2, neighbors.size());
        assertTrue(neighbors.stream().anyMatch(p -> p.value() == 'A'));
        assertTrue(neighbors.stream().anyMatch(p -> p.value() == 'C'));
    }
}
