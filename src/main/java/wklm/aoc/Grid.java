package wklm.aoc;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Immutable grid data structure for 2D algorithmic problems.
 * Designed for Advent of Code challenges with functional programming paradigm.
 *
 * @param <T> The type of elements stored in the grid.
 */
public final class Grid<T> {

    // Direction vectors as records for type safety
    private record Direction(int dx, int dy) {
        Point<Integer> apply(Point<Integer> point) {
            return new Point<>(point.x() + dx, point.y() + dy, null);
        }
    }

    private static final List<Direction> FOUR_DIRECTIONS = List.of(
        new Direction(-1, 0),  // Up
        new Direction(1, 0),   // Down
        new Direction(0, -1),  // Left
        new Direction(0, 1)    // Right
    );

    private static final List<Direction> EIGHT_DIRECTIONS = List.of(
        new Direction(-1, 0),   // Up
        new Direction(1, 0),    // Down
        new Direction(0, -1),   // Left
        new Direction(0, 1),    // Right
        new Direction(-1, -1),  // Up-Left
        new Direction(-1, 1),   // Up-Right
        new Direction(1, -1),   // Down-Left
        new Direction(1, 1)     // Down-Right
    );

    /**
     * Enum representing grid behaviors.
     */
    public enum BehaviorType {
        STANDARD, TOROIDAL
    }

    /**
     * Sealed interface for grid behaviors - only StandardBehavior and ToroidalBehavior allowed.
     */
    private sealed interface Behavior<T> permits StandardBehavior, ToroidalBehavior {
        List<Point<T>> getNeighbors(Point<T> point, Grid<T> grid, Connectivity connectivity);
    }

    /**
     * Standard grid behavior without wrap-around.
     */
    private static final class StandardBehavior<T> implements Behavior<T> {
        @Override
        public List<Point<T>> getNeighbors(Point<T> point, Grid<T> grid, Connectivity connectivity) {
            return getNeighborsStream(point, grid, connectivity, false)
                .toList();
        }
    }

    /**
     * Toroidal grid behavior with wrap-around.
     */
    private static final class ToroidalBehavior<T> implements Behavior<T> {
        @Override
        public List<Point<T>> getNeighbors(Point<T> point, Grid<T> grid, Connectivity connectivity) {
            return getNeighborsStream(point, grid, connectivity, true)
                .toList();
        }
    }

    /**
     * Functional neighbor retrieval using streams.
     */
    private static <T> Stream<Point<T>> getNeighborsStream(
            Point<T> point,
            Grid<T> grid,
            Connectivity connectivity,
            boolean toroidal) {

        var directions = connectivity == Connectivity.FOUR ? FOUR_DIRECTIONS : EIGHT_DIRECTIONS;
        var dims = grid.dimensions;

        return directions.stream()
            .map(dir -> {
                int nr = point.x() + dir.dx();
                int nc = point.y() + dir.dy();

                if (toroidal) {
                    nr = Math.floorMod(nr, dims.nrows());
                    nc = Math.floorMod(nc, dims.ncols());
                } else if (nr < 0 || nr >= dims.nrows() || nc < 0 || nc >= dims.ncols()) {
                    return null;
                }

                return grid.rows.get(nr).get(nc);
            })
            .filter(Objects::nonNull);
    }

    private final List<List<Point<T>>> rows;
    private final Map<T, List<Point<T>>> locations;
    private final Dimensions dimensions;
    private final Behavior<T> behavior;

    /**
     * Private constructor for internal use.
     */
    private Grid(List<List<Point<T>>> rows, Map<T, List<Point<T>>> locations,
                 Dimensions dimensions, Behavior<T> behavior) {
        this.rows = rows;
        this.locations = locations;
        this.dimensions = dimensions;
        this.behavior = behavior;
    }

    /**
     * Constructs a Grid from a list of points using functional composition.
     *
     * @param points       List of points to populate the grid.
     * @param dimensions   Dimensions of the grid.
     * @param defaultValue Default value for empty points.
     * @param bt           Behavior type (STANDARD or TOROIDAL).
     * @throws NullPointerException if any required parameter is null.
     * @throws IllegalArgumentException if points are out of bounds.
     */
    public Grid(List<Point<T>> points, Dimensions dimensions,
                Optional<T> defaultValue, Optional<BehaviorType> bt) {
        Objects.requireNonNull(points, "Points list cannot be null.");
        Objects.requireNonNull(dimensions, "Dimensions cannot be null.");
        Objects.requireNonNull(defaultValue, "Default value Optional cannot be null.");
        Objects.requireNonNull(bt, "Behavior type Optional cannot be null.");

        this.behavior = bt.map(behaviorType -> switch (behaviorType) {
            case STANDARD -> new StandardBehavior<T>();
            case TOROIDAL -> new ToroidalBehavior<T>();
        }).orElseGet(StandardBehavior::new);

        this.dimensions = dimensions;

        // Validate points are within bounds
        points.stream()
            .filter(p -> p.x() < 0 || p.x() >= dimensions.nrows() ||
                        p.y() < 0 || p.y() >= dimensions.ncols())
            .findFirst()
            .ifPresent(p -> {
                throw new IllegalArgumentException(
                    String.format("Point (%d,%d) is out of bounds for dimensions %dx%d",
                        p.x(), p.y(), dimensions.nrows(), dimensions.ncols()));
            });

        T defaultVal = defaultValue.orElse(null);

        // Create grid with default values using functional approach
        var initialGrid = IntStream.range(0, dimensions.nrows())
            .mapToObj(i -> IntStream.range(0, dimensions.ncols())
                .mapToObj(j -> new Point<>(i, j, defaultVal))
                .collect(Collectors.toList()))
            .collect(Collectors.toList());

        // Apply provided points (mutable for construction)
        var mutableGrid = initialGrid.stream()
            .map(ArrayList::new)
            .collect(Collectors.toList());

        points.forEach(p -> mutableGrid.get(p.x()).set(p.y(), p));

        // Build locations index functionally
        var allPoints = mutableGrid.stream()
            .flatMap(Collection::stream)
            .filter(p -> p.value() != null)
            .collect(Collectors.groupingBy(
                Point::value,
                Collectors.collectingAndThen(
                    Collectors.toList(),
                    Collections::unmodifiableList
                )
            ));

        // Make immutable
        this.rows = mutableGrid.stream()
            .map(Collections::unmodifiableList)
            .toList();
        this.locations = Collections.unmodifiableMap(allPoints);
    }

    /**
     * Factory method to create a Grid from a string input using functional parsing.
     */
    public static <T> Optional<Grid<T>> fromString(String input, CharacterParser<T> parser,
                                                     BehaviorType behaviorType) {
        Objects.requireNonNull(input, "Input string cannot be null.");
        Objects.requireNonNull(parser, "CharacterParser cannot be null.");
        Objects.requireNonNull(behaviorType, "BehaviorType cannot be null.");

        String[] lines = input.split("\n");
        if (lines.length == 0) return Optional.empty();

        int nrows = lines.length;
        int ncols = lines.length > 0 ? lines[0].length() : 0;

        // Validate all lines have same length
        if (!Arrays.stream(lines).allMatch(line -> line.length() == ncols)) {
            return Optional.empty();
        }

        // Parse grid functionally
        var tmpRows = IntStream.range(0, nrows)
            .mapToObj(i -> IntStream.range(0, ncols)
                .mapToObj(j -> new Point<>(i, j, parser.parse(lines[i].charAt(j))))
                .collect(Collectors.toUnmodifiableList()))
            .toList();

        // Build locations index
        var tmpLocations = tmpRows.stream()
            .flatMap(Collection::stream)
            .filter(p -> p.value() != null)
            .collect(Collectors.groupingBy(
                Point::value,
                Collectors.collectingAndThen(
                    Collectors.toList(),
                    Collections::unmodifiableList
                )
            ));

        var dims = new Dimensions(nrows, ncols);
        var behavior = switch (behaviorType) {
            case STANDARD -> new StandardBehavior<T>();
            case TOROIDAL -> new ToroidalBehavior<T>();
        };

        return Optional.of(new Grid<>(tmpRows, Collections.unmodifiableMap(tmpLocations), dims, behavior));
    }

    /**
     * Factory method to create a standard Grid from a string input.
     */
    public static <T> Optional<Grid<T>> fromString(String input, CharacterParser<T> parser) {
        return fromString(input, parser, BehaviorType.STANDARD);
    }

    /**
     * Factory method to create a Toroidal Grid from a string input.
     */
    public static <T> Optional<Grid<T>> fromToroidalString(String input, CharacterParser<T> parser) {
        return fromString(input, parser, BehaviorType.TOROIDAL);
    }

    /**
     * Gets the dimensions of this grid.
     */
    public Dimensions getDimensions() {
        return dimensions;
    }

    /**
     * Gets the number of rows in this grid.
     */
    public int getRows() {
        return dimensions.nrows();
    }

    /**
     * Gets the number of columns in this grid.
     */
    public int getCols() {
        return dimensions.ncols();
    }

    /**
     * Retrieves the point at the specified row and column.
     */
    public Optional<Point<T>> get(int row, int col) {
        return Optional.of(dimensions)
            .filter(d -> row >= 0 && row < d.nrows() && col >= 0 && col < d.ncols())
            .map(d -> rows.get(row).get(col));
    }

    /**
     * Finds all points with the specified value.
     */
    public Optional<List<Point<T>>> find(T value) {
        return Optional.ofNullable(locations.get(value))
            .filter(Predicate.not(List::isEmpty));
    }

    /**
     * Retrieves the neighbors of a given point based on connectivity.
     */
    public List<Point<T>> getNeighbors(Point<T> point, Connectivity connectivity) {
        return Optional.ofNullable(point)
            .map(p -> behavior.getNeighbors(p, this, connectivity))
            .orElse(List.of());
    }

    /**
     * Retrieves a list of unique values present in the grid.
     */
    public List<T> getUniqueValues() {
        return List.copyOf(locations.keySet());
    }

    /**
     * Retrieves all connected points with the specified target value using functional BFS.
     */
    public List<List<Point<T>>> getConnectedPoints(T targetValue, Connectivity connectivity) {
        Objects.requireNonNull(targetValue, "Target value cannot be null.");
        Objects.requireNonNull(connectivity, "Connectivity cannot be null.");

        var startingPoints = locations.get(targetValue);
        if (startingPoints == null || startingPoints.isEmpty()) {
            return List.of();
        }

        var visited = new HashSet<Point<T>>();

        return startingPoints.stream()
            .filter(visited::add)
            .map(start -> bfs(start, targetValue, connectivity, visited))
            .map(Collections::unmodifiableList)
            .toList();
    }

    /**
     * Functional BFS using iteration (tail recursion optimization).
     */
    private List<Point<T>> bfs(Point<T> start, T targetValue,
                               Connectivity connectivity, Set<Point<T>> visited) {
        var component = new ArrayList<Point<T>>();
        var queue = new ArrayDeque<Point<T>>();
        queue.offer(start);

        while (!queue.isEmpty()) {
            var current = queue.poll();
            component.add(current);

            behavior.getNeighbors(current, this, connectivity).stream()
                .filter(neighbor -> Objects.equals(neighbor.value(), targetValue))
                .filter(visited::add)
                .forEach(queue::offer);
        }

        return component;
    }

    /**
     * Transposes the grid using functional mapping.
     */
    public Grid<T> transpose() {
        var newDims = new Dimensions(dimensions.ncols(), dimensions.nrows());

        var transposedRows = IntStream.range(0, newDims.nrows())
            .mapToObj(i -> IntStream.range(0, newDims.ncols())
                .mapToObj(j -> {
                    var original = rows.get(j).get(i);
                    return new Point<>(i, j, original.value());
                })
                .collect(Collectors.toUnmodifiableList()))
            .toList();

        var transposedLocations = transposedRows.stream()
            .flatMap(Collection::stream)
            .filter(p -> p.value() != null)
            .collect(Collectors.groupingBy(
                Point::value,
                Collectors.collectingAndThen(
                    Collectors.toList(),
                    Collections::unmodifiableList
                )
            ));

        return new Grid<>(transposedRows, Collections.unmodifiableMap(transposedLocations),
                         newDims, this.behavior);
    }

    /**
     * Performs the union of this Grid with another Grid using functional composition.
     */
    public Optional<Grid<T>> union(Grid<T> other, BiFunction<T, T, T> merger) {
        Objects.requireNonNull(other, "Other grid cannot be null.");
        Objects.requireNonNull(merger, "Merger function cannot be null.");

        if (!this.dimensions.equals(other.dimensions)) {
            return Optional.empty();
        }

        var combinedRows = IntStream.range(0, dimensions.nrows())
            .mapToObj(i -> IntStream.range(0, dimensions.ncols())
                .mapToObj(j -> {
                    var thisPoint = this.rows.get(i).get(j);
                    var otherPoint = other.rows.get(i).get(j);
                    var mergedValue = merger.apply(thisPoint.value(), otherPoint.value());
                    return new Point<>(i, j, mergedValue);
                })
                .collect(Collectors.toUnmodifiableList()))
            .toList();

        var combinedLocations = combinedRows.stream()
            .flatMap(Collection::stream)
            .filter(p -> p.value() != null)
            .collect(Collectors.groupingBy(
                Point::value,
                Collectors.collectingAndThen(
                    Collectors.toList(),
                    Collections::unmodifiableList
                )
            ));

        return Optional.of(new Grid<>(combinedRows, Collections.unmodifiableMap(combinedLocations),
                                     dimensions, this.behavior));
    }

    /**
     * Finds the shortest path using Dijkstra's algorithm with functional approach.
     */
    public Optional<List<Point<T>>> findShortestPathDijkstra(
            int startRow, int startCol, int endRow, int endCol,
            Connectivity connectivity, MovementCostFunction<T> movementCostFunc) {

        Objects.requireNonNull(connectivity, "Connectivity cannot be null.");

        return get(startRow, startCol)
            .flatMap(start -> get(endRow, endCol)
                .flatMap(end -> dijkstra(start, end, connectivity, movementCostFunc)));
    }

    /**
     * Overloaded method for Dijkstra's algorithm with uniform movement costs.
     */
    public Optional<List<Point<T>>> findShortestPathDijkstra(
            int startRow, int startCol, int endRow, int endCol, Connectivity connectivity) {
        return findShortestPathDijkstra(startRow, startCol, endRow, endCol, connectivity, null);
    }

    /**
     * Dijkstra's algorithm implementation.
     */
    private Optional<List<Point<T>>> dijkstra(Point<T> start, Point<T> end,
                                               Connectivity connectivity,
                                               MovementCostFunction<T> movementCostFunc) {
        var queue = new PriorityQueue<DijkstraNode<T>>();
        var costs = new HashMap<Point<T>, Double>();
        var predecessors = new HashMap<Point<T>, Point<T>>();
        var visited = new HashSet<Point<T>>();

        queue.offer(new DijkstraNode<>(start, 0.0));
        costs.put(start, 0.0);

        while (!queue.isEmpty()) {
            var currentNode = queue.poll();
            var currentPoint = currentNode.point;

            if (!visited.add(currentPoint)) {
                continue;
            }

            if (currentPoint.equals(end)) {
                return Optional.of(reconstructPath(predecessors, end));
            }

            behavior.getNeighbors(currentPoint, this, connectivity).stream()
                .filter(Predicate.not(visited::contains))
                .forEach(neighbor -> {
                    double movementCost = Optional.ofNullable(movementCostFunc)
                        .map(func -> func.calculate(currentPoint, neighbor))
                        .orElse(1.0);

                    double newCost = costs.get(currentPoint) + movementCost;

                    if (newCost < costs.getOrDefault(neighbor, Double.MAX_VALUE)) {
                        costs.put(neighbor, newCost);
                        predecessors.put(neighbor, currentPoint);
                        queue.offer(new DijkstraNode<>(neighbor, newCost));
                    }
                });
        }

        return Optional.empty();
    }

    /**
     * Reconstructs the path from predecessors using functional stream.
     */
    private List<Point<T>> reconstructPath(Map<Point<T>, Point<T>> predecessors, Point<T> end) {
        return Stream.iterate(end, Objects::nonNull, predecessors::get)
            .collect(Collectors.collectingAndThen(
                Collectors.toList(),
                list -> {
                    Collections.reverse(list);
                    return Collections.unmodifiableList(list);
                }
            ));
    }

    /**
     * String representation of the grid.
     */
    @Override
    public String toString() {
        return rows.stream()
            .map(row -> row.stream()
                .map(p -> Optional.ofNullable(p.value())
                    .map(Object::toString)
                    .orElse("_"))
                .collect(Collectors.joining()))
            .collect(Collectors.joining("\n"));
    }

    /**
     * Formatted grid string representation.
     */
    public String toGridString(boolean includeIndices) {
        var sb = new StringBuilder();

        if (includeIndices) {
            // Column indices
            sb.append("    ");
            IntStream.range(0, dimensions.ncols())
                .forEach(j -> sb.append(String.format("%3d", j)));
            sb.append("\n   +");
            sb.append("----".repeat(dimensions.ncols()));
            sb.append("\n");
        }

        IntStream.range(0, dimensions.nrows()).forEach(i -> {
            if (includeIndices) {
                sb.append(String.format("%3d|", i));
            }
            rows.get(i).forEach(p -> {
                sb.append(" ");
                sb.append(Optional.ofNullable(p.value())
                    .map(Object::toString)
                    .orElse("."));
            });
            sb.append("\n");
        });

        return sb.toString();
    }

    /**
     * Graph string representation using functional approach.
     */
    public String toGraphString(Connectivity connectivity, boolean includeWeights,
                               MovementCostFunction<T> movementCostFunc) {
        Objects.requireNonNull(connectivity, "Connectivity cannot be null.");

        var sb = new StringBuilder("Graph Representation (Adjacency List):\n");

        rows.stream()
            .flatMap(Collection::stream)
            .forEach(point -> {
                sb.append(String.format("Node(%d,%d)", point.x(), point.y()));
                var neighbors = behavior.getNeighbors(point, this, connectivity);

                if (neighbors.isEmpty()) {
                    sb.append(" has no neighbors.\n");
                    return;
                }

                sb.append(" -> [");
                sb.append(neighbors.stream()
                    .map(neighbor -> includeWeights && movementCostFunc != null
                        ? String.format("(%d,%d, cost=%.2f)",
                            neighbor.x(), neighbor.y(),
                            movementCostFunc.calculate(point, neighbor))
                        : String.format("(%d,%d)", neighbor.x(), neighbor.y()))
                    .collect(Collectors.joining(", ")));
                sb.append("]\n");
            });

        return sb.toString();
    }

    /**
     * Functional interface for calculating movement costs between points.
     */
    @FunctionalInterface
    public interface MovementCostFunction<T> {
        double calculate(Point<T> from, Point<T> to);
    }

    /**
     * Helper record for Dijkstra's algorithm nodes.
     */
    private record DijkstraNode<T>(Point<T> point, double cost) implements Comparable<DijkstraNode<T>> {
        @Override
        public int compareTo(DijkstraNode<T> other) {
            return Double.compare(this.cost, other.cost);
        }
    }

    // ==================== AOC-OPTIMIZED METHODS ====================

    /**
     * Gets the neighbor in a specific direction (optimized for AOC direction-based navigation).
     *
     * @param point     The starting point
     * @param direction The direction to move
     * @return Optional containing the neighbor point, or empty if out of bounds
     */
    public Optional<Point<T>> getNeighbor(Point<T> point, wklm.aoc.Direction direction) {
        if (point == null) return Optional.empty();
        int newRow = point.x() + direction.dx();
        int newCol = point.y() + direction.dy();
        return get(newRow, newCol);
    }

    /**
     * BFS pathfinding - finds shortest path using breadth-first search.
     * Optimized for AOC - faster than Dijkstra for unweighted grids.
     *
     * @param startRow     Starting row
     * @param startCol     Starting column
     * @param endRow       Ending row
     * @param endCol       Ending column
     * @param connectivity Connectivity type (FOUR or EIGHT)
     * @param canMove      Predicate to determine if movement to a point is allowed
     * @return Optional containing the path, or empty if no path exists
     */
    public Optional<List<Point<T>>> findPathBFS(
            int startRow, int startCol, int endRow, int endCol,
            Connectivity connectivity, java.util.function.Predicate<Point<T>> canMove) {

        return get(startRow, startCol)
            .flatMap(start -> get(endRow, endCol)
                .flatMap(end -> bfsPath(start, end, connectivity, canMove)));
    }

    /**
     * BFS implementation for pathfinding.
     */
    private Optional<List<Point<T>>> bfsPath(
            Point<T> start, Point<T> end, Connectivity connectivity,
            java.util.function.Predicate<Point<T>> canMove) {

        var queue = new ArrayDeque<Point<T>>();
        var visited = new HashSet<Point<T>>();
        var predecessors = new HashMap<Point<T>, Point<T>>();

        queue.offer(start);
        visited.add(start);

        while (!queue.isEmpty()) {
            var current = queue.poll();

            if (current.equals(end)) {
                return Optional.of(reconstructPath(predecessors, end));
            }

            behavior.getNeighbors(current, this, connectivity).stream()
                .filter(canMove)
                .filter(visited::add)
                .forEach(neighbor -> {
                    predecessors.put(neighbor, current);
                    queue.offer(neighbor);
                });
        }

        return Optional.empty();
    }

    /**
     * A* pathfinding - finds shortest path using A* algorithm with Manhattan distance heuristic.
     * Optimized for AOC - much faster than Dijkstra for large grids.
     *
     * @param startRow         Starting row
     * @param startCol         Starting column
     * @param endRow           Ending row
     * @param endCol           Ending column
     * @param connectivity     Connectivity type (FOUR or EIGHT)
     * @param movementCostFunc Optional movement cost function (null for uniform cost)
     * @return Optional containing the path, or empty if no path exists
     */
    public Optional<List<Point<T>>> findPathAStar(
            int startRow, int startCol, int endRow, int endCol,
            Connectivity connectivity, MovementCostFunction<T> movementCostFunc) {

        return get(startRow, startCol)
            .flatMap(start -> get(endRow, endCol)
                .flatMap(end -> astar(start, end, connectivity, movementCostFunc)));
    }

    /**
     * A* algorithm implementation with Manhattan distance heuristic.
     */
    private Optional<List<Point<T>>> astar(
            Point<T> start, Point<T> end, Connectivity connectivity,
            MovementCostFunction<T> movementCostFunc) {

        var openSet = new PriorityQueue<AStarNode<T>>();
        var gScores = new HashMap<Point<T>, Double>();
        var predecessors = new HashMap<Point<T>, Point<T>>();
        var visited = new HashSet<Point<T>>();

        double startH = manhattanDistance(start, end);
        openSet.offer(new AStarNode<>(start, 0.0, startH));
        gScores.put(start, 0.0);

        while (!openSet.isEmpty()) {
            var current = openSet.poll();

            if (!visited.add(current.point)) {
                continue;
            }

            if (current.point.equals(end)) {
                return Optional.of(reconstructPath(predecessors, end));
            }

            double currentG = gScores.get(current.point);

            behavior.getNeighbors(current.point, this, connectivity).stream()
                .filter(java.util.function.Predicate.not(visited::contains))
                .forEach(neighbor -> {
                    double moveCost = Optional.ofNullable(movementCostFunc)
                        .map(func -> func.calculate(current.point, neighbor))
                        .orElse(1.0);

                    double tentativeG = currentG + moveCost;

                    if (tentativeG < gScores.getOrDefault(neighbor, Double.MAX_VALUE)) {
                        predecessors.put(neighbor, current.point);
                        gScores.put(neighbor, tentativeG);
                        double h = manhattanDistance(neighbor, end);
                        openSet.offer(new AStarNode<>(neighbor, tentativeG, h));
                    }
                });
        }

        return Optional.empty();
    }

    /**
     * A* node with f-score = g-score + heuristic.
     */
    private record AStarNode<T>(Point<T> point, double gScore, double hScore)
            implements Comparable<AStarNode<T>> {
        double fScore() {
            return gScore + hScore;
        }

        @Override
        public int compareTo(AStarNode<T> other) {
            return Double.compare(this.fScore(), other.fScore());
        }
    }

    /**
     * Calculates Manhattan distance between two points (L1 distance).
     * Primary distance metric for grid navigation in AOC.
     *
     * @param p1 First point
     * @param p2 Second point
     * @return Manhattan distance
     */
    public static <T> double manhattanDistance(Point<T> p1, Point<T> p2) {
        return Math.abs(p1.x() - p2.x()) + Math.abs(p1.y() - p2.y());
    }

    /**
     * Calculates Euclidean distance between two points (L2 distance).
     *
     * @param p1 First point
     * @param p2 Second point
     * @return Euclidean distance
     */
    public static <T> double euclideanDistance(Point<T> p1, Point<T> p2) {
        int dx = p1.x() - p2.x();
        int dy = p1.y() - p2.y();
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Rotates the grid 90 degrees clockwise.
     * Optimized for AOC grid rotation puzzles.
     */
    public Grid<T> rotate90Clockwise() {
        var newDims = new Dimensions(dimensions.ncols(), dimensions.nrows());

        var rotatedRows = IntStream.range(0, newDims.nrows())
            .mapToObj(i -> IntStream.range(0, newDims.ncols())
                .mapToObj(j -> {
                    // (i, j) in rotated = (nrows - 1 - j, i) in original
                    var original = rows.get(dimensions.nrows() - 1 - j).get(i);
                    return new Point<>(i, j, original.value());
                })
                .collect(Collectors.toUnmodifiableList()))
            .toList();

        var rotatedLocations = rotatedRows.stream()
            .flatMap(Collection::stream)
            .filter(p -> p.value() != null)
            .collect(Collectors.groupingBy(
                Point::value,
                Collectors.collectingAndThen(Collectors.toList(), Collections::unmodifiableList)
            ));

        return new Grid<>(rotatedRows, Collections.unmodifiableMap(rotatedLocations),
                         newDims, this.behavior);
    }

    /**
     * Rotates the grid 90 degrees counter-clockwise.
     */
    public Grid<T> rotate90CounterClockwise() {
        var newDims = new Dimensions(dimensions.ncols(), dimensions.nrows());

        var rotatedRows = IntStream.range(0, newDims.nrows())
            .mapToObj(i -> IntStream.range(0, newDims.ncols())
                .mapToObj(j -> {
                    // (i, j) in rotated = (j, ncols - 1 - i) in original
                    var original = rows.get(j).get(dimensions.ncols() - 1 - i);
                    return new Point<>(i, j, original.value());
                })
                .collect(Collectors.toUnmodifiableList()))
            .toList();

        var rotatedLocations = rotatedRows.stream()
            .flatMap(Collection::stream)
            .filter(p -> p.value() != null)
            .collect(Collectors.groupingBy(
                Point::value,
                Collectors.collectingAndThen(Collectors.toList(), Collections::unmodifiableList)
            ));

        return new Grid<>(rotatedRows, Collections.unmodifiableMap(rotatedLocations),
                         newDims, this.behavior);
    }

    /**
     * Flips the grid horizontally (mirror across vertical axis).
     */
    public Grid<T> flipHorizontal() {
        var flippedRows = IntStream.range(0, dimensions.nrows())
            .mapToObj(i -> IntStream.range(0, dimensions.ncols())
                .mapToObj(j -> {
                    var original = rows.get(i).get(dimensions.ncols() - 1 - j);
                    return new Point<>(i, j, original.value());
                })
                .collect(Collectors.toUnmodifiableList()))
            .toList();

        var flippedLocations = flippedRows.stream()
            .flatMap(Collection::stream)
            .filter(p -> p.value() != null)
            .collect(Collectors.groupingBy(
                Point::value,
                Collectors.collectingAndThen(Collectors.toList(), Collections::unmodifiableList)
            ));

        return new Grid<>(flippedRows, Collections.unmodifiableMap(flippedLocations),
                         dimensions, this.behavior);
    }

    /**
     * Flips the grid vertically (mirror across horizontal axis).
     */
    public Grid<T> flipVertical() {
        var flippedRows = IntStream.range(0, dimensions.nrows())
            .mapToObj(i -> IntStream.range(0, dimensions.ncols())
                .mapToObj(j -> {
                    var original = rows.get(dimensions.nrows() - 1 - i).get(j);
                    return new Point<>(i, j, original.value());
                })
                .collect(Collectors.toUnmodifiableList()))
            .toList();

        var flippedLocations = flippedRows.stream()
            .flatMap(Collection::stream)
            .filter(p -> p.value() != null)
            .collect(Collectors.groupingBy(
                Point::value,
                Collectors.collectingAndThen(Collectors.toList(), Collections::unmodifiableList)
            ));

        return new Grid<>(flippedRows, Collections.unmodifiableMap(flippedLocations),
                         dimensions, this.behavior);
    }

    /**
     * Counts points matching a predicate - optimized for AOC counting problems.
     *
     * @param predicate Condition to check
     * @return Count of matching points
     */
    public long count(java.util.function.Predicate<Point<T>> predicate) {
        return rows.stream()
            .flatMap(Collection::stream)
            .filter(predicate)
            .count();
    }

    /**
     * Finds first point matching a predicate - optimized for AOC search problems.
     *
     * @param predicate Condition to check
     * @return Optional containing first matching point
     */
    public Optional<Point<T>> findFirst(java.util.function.Predicate<Point<T>> predicate) {
        return rows.stream()
            .flatMap(Collection::stream)
            .filter(predicate)
            .findFirst();
    }

    /**
     * Applies an action to each point in the grid - optimized for AOC iteration.
     *
     * @param action Action to perform on each point
     */
    public void forEach(java.util.function.Consumer<Point<T>> action) {
        rows.stream()
            .flatMap(Collection::stream)
            .forEach(action);
    }

    /**
     * Returns a stream of all points for functional composition - enables parallel processing.
     *
     * @return Stream of all points
     */
    public Stream<Point<T>> stream() {
        return rows.stream().flatMap(Collection::stream);
    }

    /**
     * Returns a parallel stream of all points - optimized for large grid operations in AOC.
     *
     * @return Parallel stream of all points
     */
    public Stream<Point<T>> parallelStream() {
        return rows.parallelStream().flatMap(Collection::stream);
    }

    /**
     * Extracts a rectangular subgrid - useful for AOC region problems.
     *
     * @param startRow Starting row (inclusive)
     * @param startCol Starting column (inclusive)
     * @param endRow   Ending row (exclusive)
     * @param endCol   Ending column (exclusive)
     * @return Optional containing the subgrid, or empty if bounds are invalid
     */
    public Optional<Grid<T>> subGrid(int startRow, int startCol, int endRow, int endCol) {
        if (startRow < 0 || startCol < 0 || endRow > dimensions.nrows() ||
            endCol > dimensions.ncols() || startRow >= endRow || startCol >= endCol) {
            return Optional.empty();
        }

        int newNrows = endRow - startRow;
        int newNcols = endCol - startCol;

        var subRows = IntStream.range(0, newNrows)
            .mapToObj(i -> IntStream.range(0, newNcols)
                .mapToObj(j -> {
                    var original = rows.get(startRow + i).get(startCol + j);
                    return new Point<>(i, j, original.value());
                })
                .collect(Collectors.toUnmodifiableList()))
            .toList();

        var subLocations = subRows.stream()
            .flatMap(Collection::stream)
            .filter(p -> p.value() != null)
            .collect(Collectors.groupingBy(
                Point::value,
                Collectors.collectingAndThen(Collectors.toList(), Collections::unmodifiableList)
            ));

        return Optional.of(new Grid<>(subRows, Collections.unmodifiableMap(subLocations),
                                     new Dimensions(newNrows, newNcols), this.behavior));
    }

    /**
     * Detects if a grid contains a cycle starting from a point - useful for AOC cycle detection.
     * Uses Floyd's cycle detection algorithm.
     *
     * @param start        Starting point
     * @param connectivity Connectivity type
     * @param next         Function to determine next point
     * @return Optional containing cycle length if found
     */
    public Optional<Integer> detectCycle(
            Point<T> start, Connectivity connectivity,
            java.util.function.Function<Point<T>, Optional<Point<T>>> next) {

        var slow = start;
        var fast = start;

        // Floyd's algorithm: slow moves 1 step, fast moves 2 steps
        while (true) {
            var slowNext = next.apply(slow);
            if (slowNext.isEmpty()) return Optional.empty();
            slow = slowNext.get();

            var fastNext1 = next.apply(fast);
            if (fastNext1.isEmpty()) return Optional.empty();
            var fastNext2 = next.apply(fastNext1.get());
            if (fastNext2.isEmpty()) return Optional.empty();
            fast = fastNext2.get();

            if (slow.equals(fast)) {
                // Cycle detected, find cycle length
                int cycleLength = 1;
                fast = next.apply(slow).get();
                while (!slow.equals(fast)) {
                    fast = next.apply(fast).get();
                    cycleLength++;
                }
                return Optional.of(cycleLength);
            }
        }
    }
}
