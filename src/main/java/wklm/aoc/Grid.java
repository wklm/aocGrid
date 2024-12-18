package wklm.aoc;

import java.util.*;
import java.util.function.BiFunction;
import java.util.stream.Collectors;

/**
 * Core Grid class
 *
 * @param <T> The type of elements stored in the grid.
 */
public final class Grid<T> {

    // Direction constants shared by behaviors
    private static final int[][] FOUR_DIRECTIONS = {
            {-1, 0},  // Up
            {1, 0},   // Down
            {0, -1},  // Left
            {0, 1}    // Right
    };

    private static final int[][] EIGHT_DIRECTIONS = {
            {-1, 0},   // Up
            {1, 0},    // Down
            {0, -1},   // Left
            {0, 1},    // Right
            {-1, -1},  // Up-Left
            {-1, 1},   // Up-Right
            {1, -1},   // Down-Left
            {1, 1}     // Down-Right
    };

    /**
     * Enum representing grid behaviors.
     */
    public enum BehaviorType {
        STANDARD, TOROIDAL
    }

    /**
     * Internal interface for grid behaviors.
     */
    private interface Behavior<T> {
        List<Point<T>> getNeighbors(Point<T> point, Grid<T> grid, int connectivity);
    }

    /**
     * Standard grid behavior without wrap-around.
     *
     * @param <T> The type of elements in the grid.
     */
    private static final class StandardBehavior<T> implements Behavior<T> {
        @Override
        public List<Point<T>> getNeighbors(Point<T> point, Grid<T> grid, int connectivity) {
            return getNeighborsInternal(point, grid, connectivity, false);
        }
    }

    /**
     * Toroidal grid behavior with wrap-around.
     *
     * @param <T> The type of elements in the grid.
     */
    private static final class ToroidalBehavior<T> implements Behavior<T> {
        @Override
        public List<Point<T>> getNeighbors(Point<T> point, Grid<T> grid, int connectivity) {
            return getNeighborsInternal(point, grid, connectivity, true);
        }
    }

    // Shared neighbor retrieval logic
    private static <T> List<Point<T>> getNeighborsInternal(Point<T> point, Grid<T> grid, int connectivity, boolean toroidal) {
        final int[][] directions = switch (connectivity) {
            case 4 -> FOUR_DIRECTIONS;
            case 8 -> EIGHT_DIRECTIONS;
            default -> throw new IllegalArgumentException("Unsupported connectivity: " + connectivity);
        };

        List<Point<T>> neighbors = new ArrayList<>();
        int nrows = grid.dimensions.nrows();
        int ncols = grid.dimensions.ncols();

        for (int[] dir : directions) {
            int nr = point.x() + dir[0];
            int nc = point.y() + dir[1];
            if (toroidal) {
                nr = Math.floorMod(nr, nrows);
                nc = Math.floorMod(nc, ncols);
            }
            if (!toroidal && (nr < 0 || nr >= nrows || nc < 0 || nc >= ncols)) {
                continue;
            }
            neighbors.add(grid.rows.get(nr).get(nc));
        }
        return Collections.unmodifiableList(neighbors);
    }

    private final List<List<Point<T>>> rows;
    private final Map<T, List<Point<T>>> locations;
    private final Dimensions dimensions;
    private final Behavior<T> behavior;

    private Grid(List<List<Point<T>>> rows, Map<T, List<Point<T>>> locations, Dimensions dimensions, Behavior<T> behavior) {
        this.rows = Collections.unmodifiableList(rows);
        this.locations = Collections.unmodifiableMap(locations);
        this.dimensions = dimensions;
        this.behavior = behavior;
    }

    /**
     * Constructs a Grid from a list of points, dimensions, default value, and behavior type.
     *
     * @param points       List of points to populate the grid.
     * @param dimensions   Dimensions of the grid.
     * @param defaultValue Default value for empty points.
     * @param bt           Behavior type (STANDARD or TOROIDAL).
     */
    public Grid(List<Point<T>> points, Dimensions dimensions, Optional<T> defaultValue, Optional<BehaviorType> bt) {
        this.behavior = bt.<Behavior<T>>map(behaviorType -> switch (behaviorType) {
            case STANDARD -> new StandardBehavior<>();
            case TOROIDAL -> new ToroidalBehavior<>();
        }).orElseGet(StandardBehavior::new);

        this.dimensions = dimensions;
        this.locations = new HashMap<>();
        this.rows = new ArrayList<>(dimensions.nrows());
        for (int i = 0; i < dimensions.nrows(); i++) {
            rows.add(new ArrayList<>(dimensions.ncols()));
            for(int j = 0; j < dimensions.ncols(); j++) {
                this.rows.get(i).set(j, new Point<>(i, j, defaultValue.orElse(null)));
            }
        }

        for(var point : points) {
            this.rows.get(point.x()).set(point.y(), point);
            this.locations.computeIfAbsent(point.value(), _ -> new ArrayList<>()).add(point);
    }


}


/**
 * Factory method to create a Grid from a string input.
 *
 * @param input        String representation of the grid, with rows separated by newlines.
 * @param parser       Parser to convert characters to type T.
 * @param behaviorType Type of grid behavior (STANDARD or TOROIDAL).
 * @return Optional containing the created Grid, or empty if input is invalid.
 */
public static <T> Optional<Grid<T>> fromString(String input, CharacterParser<T> parser, BehaviorType behaviorType) {
    Objects.requireNonNull(input, "Input string cannot be null.");
    Objects.requireNonNull(parser, "CharacterParser cannot be null.");
    Objects.requireNonNull(behaviorType, "BehaviorType cannot be null.");

    String[] lines = input.split("\n");
    int nrows = lines.length;
    int ncols = nrows > 0 ? lines[0].length() : 0;

    // Validate that all lines have the same length
    if (!Arrays.stream(lines).allMatch(line -> line.length() == ncols)) {
        return Optional.empty();
    }

    List<List<Point<T>>> tmpRows = new ArrayList<>(nrows);
    Map<T, List<Point<T>>> tmpLocations = new HashMap<>();

    for (int i = 0; i < nrows; i++) {
        String line = lines[i];
        List<Point<T>> row = new ArrayList<>(ncols);
        for (int j = 0; j < ncols; j++) {
            char c = line.charAt(j);
            T val = parser.parse(c);
            Point<T> p = new Point<>(i, j, val);
            row.add(p);
            if (val != null) {
                tmpLocations.computeIfAbsent(val, _ -> new ArrayList<>()).add(p);
            }
        }
        tmpRows.add(Collections.unmodifiableList(row));
    }

    Dimensions dims = new Dimensions(nrows, ncols);
    Behavior<T> behavior = switch (behaviorType) {
        case STANDARD -> new StandardBehavior<>();
        case TOROIDAL -> new ToroidalBehavior<>();
    };

    return Optional.of(new Grid<>(tmpRows, tmpLocations, dims, behavior));
}

/**
 * Factory method to create a standard Grid from a string input.
 *
 * @param input  String representation of the grid, with rows separated by newlines.
 * @param parser Parser to convert characters to type T.
 * @return Optional containing the created Grid, or empty if input is invalid.
 */
public static <T> Optional<Grid<T>> fromString(String input, CharacterParser<T> parser) {
    return fromString(input, parser, BehaviorType.STANDARD);
}

/**
 * Factory method to create a Toroidal Grid from a string input.
 *
 * @param input  String representation of the grid, with rows separated by newlines.
 * @param parser Parser to convert characters to type T.
 * @return Optional containing the created Toroidal Grid, or empty if input is invalid.
 */
public static <T> Optional<Grid<T>> fromToroidalString(String input, CharacterParser<T> parser) {
    return fromString(input, parser, BehaviorType.TOROIDAL);
}

/**
 * Retrieves the point at the specified row and column.
 *
 * @param row Row index.
 * @param col Column index.
 * @return Optional containing the Point at the specified location, or empty if indices are invalid.
 */
public Optional<Point<T>> get(int row, int col) {
    if (row < 0 || row >= dimensions.nrows() || col < 0 || col >= dimensions.ncols()) {
        return Optional.empty();
    }
    return Optional.of(rows.get(row).get(col));
}

/**
 * Finds all points with the specified value.
 *
 * @param value Value to search for.
 * @return Optional containing the list of matching points, or empty if none found.
 */
public Optional<List<Point<T>>> find(T value) {
    List<Point<T>> points = locations.get(value);
    return (points == null || points.isEmpty()) ? Optional.empty() : Optional.of(points);
}

/**
 * Retrieves the neighbors of a given point based on connectivity.
 *
 * @param point        Point whose neighbors are to be found.
 * @param connectivity 4 or 8 for directional connectivity.
 * @return Optional containing the list of neighboring points, or empty if point is null.
 */
public Optional<List<Point<T>>> getNeighbors(Point<T> point, int connectivity) {
    if (point == null) {
        return Optional.empty();
    }
    List<Point<T>> neighbors = behavior.getNeighbors(point, this, connectivity);
    return Optional.of(neighbors);
}

/**
 * Retrieves a list of unique values present in the grid.
 *
 * @return Unmodifiable list of unique values.
 */
public List<T> getUniqueValues() {
    return List.copyOf(locations.keySet());
}

/**
 * Retrieves all connected points with the specified target value.
 *
 * @param targetValue  Value to find connected components for.
 * @param connectivity 4 or 8 for directional connectivity.
 * @return Optional containing the list of connected point lists, or empty if none found.
 */
public Optional<List<List<Point<T>>>> getConnectedPoints(T targetValue, int connectivity) {
    Objects.requireNonNull(targetValue, "Target value cannot be null.");
    List<Point<T>> startingPoints = locations.get(targetValue);
    if (startingPoints == null || startingPoints.isEmpty()) {
        return Optional.empty();
    }

    Set<Point<T>> visited = new HashSet<>();
    List<List<Point<T>>> connectedComponents = new ArrayList<>();

    for (Point<T> start : startingPoints) {
        if (visited.add(start)) {
            List<Point<T>> component = bfs(start, targetValue, connectivity, visited);
            connectedComponents.add(Collections.unmodifiableList(component));
        }
    }

    return connectedComponents.isEmpty() ? Optional.empty() : Optional.of(connectedComponents);
}

/**
 * Breadth-first search to find all connected points with the target value.
 *
 * @param start        Starting point.
 * @param targetValue  Value to search for.
 * @param connectivity 4 or 8 for directional connectivity.
 * @param visited      Set to track visited points.
 * @return List of connected points.
 */
private List<Point<T>> bfs(Point<T> start, T targetValue, int connectivity, Set<Point<T>> visited) {
    List<Point<T>> component = new ArrayList<>();
    Queue<Point<T>> queue = new ArrayDeque<>();
    queue.offer(start);

    while (!queue.isEmpty()) {
        Point<T> current = queue.poll();
        component.add(current);

        for (Point<T> neighbor : behavior.getNeighbors(current, this, connectivity)) {
            if (Objects.equals(neighbor.value(), targetValue) && visited.add(neighbor)) {
                queue.offer(neighbor);
            }
        }
    }

    return component;
}

/**
 * Transposes the grid (rows become columns and vice versa).
 *
 * @return New transposed Grid.
 */
public Grid<T> transpose() {
    int newNrows = dimensions.ncols();
    int newNcols = dimensions.nrows();

    List<List<Point<T>>> transposedRows = new ArrayList<>(newNrows);
    Map<T, List<Point<T>>> transposedLocations = new HashMap<>();

    for (int i = 0; i < newNrows; i++) {
        List<Point<T>> newRow = new ArrayList<>(newNcols);
        for (int j = 0; j < newNcols; j++) {
            Point<T> original = rows.get(j).get(i);
            Point<T> transposedPoint = new Point<>(i, j, original.value());
            newRow.add(transposedPoint);
            if (original.value() != null) {
                transposedLocations.computeIfAbsent(original.value(), _ -> new ArrayList<>()).add(transposedPoint);
            }
        }
        transposedRows.add(Collections.unmodifiableList(newRow));
    }

    Dimensions newDims = new Dimensions(newNrows, newNcols);
    Behavior<T> newBehavior = this.behavior instanceof ToroidalBehavior ? new ToroidalBehavior<>() : new StandardBehavior<>();

    // Convert locations to unmodifiable map of unmodifiable lists
    Map<T, List<Point<T>>> unmodifiableLocations = transposedLocations.entrySet().stream()
            .collect(Collectors.toUnmodifiableMap(
                    Map.Entry::getKey,
                    e -> Collections.unmodifiableList(e.getValue())
            ));

    return new Grid<>(Collections.unmodifiableList(transposedRows), unmodifiableLocations, newDims, newBehavior);
}

/**
 * Performs the union of this Grid with another Grid using a merging function.
 *
 * @param other  Another Grid to union with.
 * @param merger Function to merge two T values.
 * @return Optional containing the new Grid representing the union, or empty if dimensions mismatch.
 */
public Optional<Grid<T>> union(Grid<T> other, BiFunction<T, T, T> merger) {
    Objects.requireNonNull(other, "Other grid cannot be null.");
    Objects.requireNonNull(merger, "Merger function cannot be null.");

    if (!this.dimensions.equals(other.dimensions)) {
        return Optional.empty();
    }

    List<List<Point<T>>> combinedRows = new ArrayList<>(dimensions.nrows());
    Map<T, List<Point<T>>> combinedLocations = new HashMap<>();

    for (int i = 0; i < dimensions.nrows(); i++) {
        List<Point<T>> combinedRow = new ArrayList<>(dimensions.ncols());
        for (int j = 0; j < dimensions.ncols(); j++) {
            Point<T> thisPoint = this.rows.get(i).get(j);
            Point<T> otherPoint = other.rows.get(i).get(j);
            T mergedValue = merger.apply(thisPoint.value(), otherPoint.value());
            Point<T> mergedPoint = new Point<>(i, j, mergedValue);
            combinedRow.add(mergedPoint);
            if (mergedValue != null) {
                combinedLocations.computeIfAbsent(mergedValue, _ -> new ArrayList<>()).add(mergedPoint);
            }
        }
        combinedRows.add(Collections.unmodifiableList(combinedRow));
    }

    Map<T, List<Point<T>>> unmodifiableLocations = combinedLocations.entrySet().stream()
            .collect(Collectors.toUnmodifiableMap(
                    Map.Entry::getKey,
                    e -> Collections.unmodifiableList(e.getValue())
            ));

    return Optional.of(new Grid<>(Collections.unmodifiableList(combinedRows), unmodifiableLocations, dimensions, this.behavior));
}

/**
 * Finds the shortest path between two points using Dijkstra's algorithm.
 * Supports both uniform and custom movement costs.
 *
 * @param startRow         Starting point's row index.
 * @param startCol         Starting point's column index.
 * @param endRow           Ending point's row index.
 * @param endCol           Ending point's column index.
 * @param connectivity     4 or 8 for directional connectivity.
 * @param movementCostFunc Optional movement cost function. If null, uniform cost is assumed.
 * @return Optional containing the list of points representing the shortest path,
 * or empty if no path exists or indices are invalid.
 */
public Optional<List<Point<T>>> findShortestPathDijkstra(
        int startRow,
        int startCol,
        int endRow,
        int endCol,
        int connectivity,
        MovementCostFunction<T> movementCostFunc
) {
    Optional<Point<T>> startOpt = get(startRow, startCol);
    Optional<Point<T>> endOpt = get(endRow, endCol);

    if (startOpt.isEmpty() || endOpt.isEmpty()) {
        return Optional.empty();
    }

    Point<T> start = startOpt.get();
    Point<T> end = endOpt.get();

    PriorityQueue<DijkstraNode<T>> queue = new PriorityQueue<>();
    Map<Point<T>, Double> costs = new HashMap<>();
    Map<Point<T>, Point<T>> predecessors = new HashMap<>();
    Set<Point<T>> visited = new HashSet<>();

    queue.offer(new DijkstraNode<>(start, 0.0));
    costs.put(start, 0.0);

    while (!queue.isEmpty()) {
        DijkstraNode<T> currentNode = queue.poll();
        Point<T> currentPoint = currentNode.point;

        if (!visited.add(currentPoint)) {
            continue;
        }

        if (currentPoint.equals(end)) {
            return Optional.of(reconstructPath(predecessors, end));
        }

        for (Point<T> neighbor : behavior.getNeighbors(currentPoint, this, connectivity)) {
            if (visited.contains(neighbor)) {
                continue;
            }

            double movementCost = (movementCostFunc != null)
                    ? movementCostFunc.calculate(currentPoint, neighbor)
                    : 1.0; // Default uniform cost

            double newCost = costs.get(currentPoint) + movementCost;

            if (newCost < costs.getOrDefault(neighbor, Double.MAX_VALUE)) {
                costs.put(neighbor, newCost);
                predecessors.put(neighbor, currentPoint);
                queue.offer(new DijkstraNode<>(neighbor, newCost));
            }
        }
    }

    // No path found
    return Optional.empty();
}

/**
 * Overloaded method for Dijkstra's algorithm with uniform movement costs.
 *
 * @param startRow     Starting point's row index.
 * @param startCol     Starting point's column index.
 * @param endRow       Ending point's row index.
 * @param endCol       Ending point's column index.
 * @param connectivity 4 or 8 for directional connectivity.
 * @return Optional containing the list of points representing the shortest path,
 * or empty if no path exists or indices are invalid.
 */
public Optional<List<Point<T>>> findShortestPathDijkstra(
        int startRow,
        int startCol,
        int endRow,
        int endCol,
        int connectivity
) {
    return findShortestPathDijkstra(startRow, startCol, endRow, endCol, connectivity, null);
}

/**
 * Reconstructs the path from predecessors map.
 *
 * @param predecessors Map of point predecessors.
 * @param end          End point.
 * @return List of points representing the path from start to end.
 */
private List<Point<T>> reconstructPath(Map<Point<T>, Point<T>> predecessors, Point<T> end) {
    List<Point<T>> path = new ArrayList<>();
    Point<T> step = end;
    while (step != null) {
        path.add(step);
        step = predecessors.get(step);
    }
    Collections.reverse(path);
    return Collections.unmodifiableList(path);
}

/**
 * Overrides the toString method to provide a simple string representation of the grid.
 *
 * @return String representation with rows separated by newlines.
 */
@Override
public String toString() {
    return rows.stream()
            .map(row -> row.stream()
                    .map(p -> p.value() != null ? p.value().toString() : "_")
                    .collect(Collectors.joining()))
            .collect(Collectors.joining("\n"));
}

/**
 * Provides a beautifully formatted grid string representation.
 *
 * @param includeIndices Whether to include row and column indices.
 * @return String representation of the grid.
 */
public String toGridString(boolean includeIndices) {
    StringBuilder sb = new StringBuilder();

    if (includeIndices) {
        // Column indices
        sb.append("    ");
        for (int j = 0; j < dimensions.ncols(); j++) {
            sb.append(String.format("%3d", j));
        }
        sb.append("\n   +");
        sb.append("----".repeat(dimensions.ncols()));
        sb.append("\n");
    }

    for (int i = 0; i < dimensions.nrows(); i++) {
        if (includeIndices) {
            sb.append(String.format("%3d|", i));
        }
        for (int j = 0; j < dimensions.ncols(); j++) {
            Point<T> p = rows.get(i).get(j);
            sb.append(" ");
            sb.append(p.value() != null ? p.value().toString() : ".");
        }
        sb.append("\n");
    }

    return sb.toString();
}

/**
 * Provides a beautifully formatted graph string representation (Adjacency List).
 *
 * @param connectivity     4 or 8 for directional connectivity.
 * @param includeWeights   Whether to include movement costs (if applicable).
 * @param movementCostFunc Optional movement cost function. If provided, weights will be calculated.
 * @return String representation of the graph.
 */
public String toGraphString(int connectivity, boolean includeWeights, MovementCostFunction<T> movementCostFunc) {
    StringBuilder sb = new StringBuilder();
    sb.append("Graph Representation (Adjacency List):\n");

    for (List<Point<T>> row : rows) {
        for (Point<T> point : row) {
            sb.append(String.format("Node(%d,%d)", point.x(), point.y()));
            List<Point<T>> neighbors = behavior.getNeighbors(point, this, connectivity);
            if (neighbors.isEmpty()) {
                sb.append(" has no neighbors.\n");
                continue;
            }
            sb.append(" -> [");
            List<String> neighborStrings = new ArrayList<>();
            for (Point<T> neighbor : neighbors) {
                if (includeWeights && movementCostFunc != null) {
                    double cost = movementCostFunc.calculate(point, neighbor);
                    neighborStrings.add(String.format("(%d,%d, cost=%.2f)", neighbor.x(), neighbor.y(), cost));
                } else {
                    neighborStrings.add(String.format("(%d,%d)", neighbor.x(), neighbor.y()));
                }
            }
            sb.append(String.join(", ", neighborStrings));
            sb.append("]\n");
        }
    }

    return sb.toString();
}

/**
 * Functional interface for calculating movement costs between points.
 *
 * @param <T> The type of elements in the grid.
 */
@FunctionalInterface
public interface MovementCostFunction<T> {
    /**
     * Calculates the cost to move from one point to another.
     *
     * @param from The starting point.
     * @param to   The destination point.
     * @return The cost of moving from 'from' to 'to'.
     */
    double calculate(Point<T> from, Point<T> to);
}

/**
 * Helper class for Dijkstra's algorithm nodes.
 *
 * @param <T> The type of elements in the grid.
 */
private record DijkstraNode<T>(Point<T> point, double cost) implements Comparable<DijkstraNode<T>> {

    @Override
    public int compareTo(DijkstraNode<T> other) {
        return Double.compare(this.cost, other.cost);
    }
}
}
