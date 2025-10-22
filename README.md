# Grid Library for Advent of Code

A high-performance, type-safe Java library for grid-based algorithmic problems, specifically designed for [Advent of Code](https://adventofcode.com/) challenges.

[![](https://jitpack.io/v/wklm/aocGrid.svg)](https://jitpack.io/#wklm/aocGrid)

## Features

- **Immutable Grid Data Structure** - Thread-safe and prevents accidental mutations
- **Generic Type Support** - Works with any type `T`
- **Multiple Grid Behaviors** - Standard and Toroidal (wrap-around) grids
- **Pathfinding Algorithms** - Built-in Dijkstra's algorithm with custom cost functions
- **Connected Components** - BFS-based component detection
- **Flexible Connectivity** - 4-directional and 8-directional neighbor support
- **Rich API** - Query, transform, and visualize grids easily
- **Zero Dependencies** - Pure Java implementation

## Installation

### Gradle (Kotlin DSL)

```kotlin
repositories {
    maven { url = uri("https://jitpack.io") }
}

dependencies {
    implementation("com.github.wklm:aocGrid:0.0.2")
}
```

### Gradle (Groovy)

```groovy
repositories {
    maven { url 'https://jitpack.io' }
}

dependencies {
    implementation 'com.github.wklm:aocGrid:0.0.2'
}
```

### Maven

```xml
<repositories>
    <repository>
        <id>jitpack.io</id>
        <url>https://jitpack.io</url>
    </repository>
</repositories>

<dependency>
    <groupId>com.github.wklm</groupId>
    <artifactId>aocGrid</artifactId>
    <version>0.0.2</version>
</dependency>
```

## Quick Start

### Creating a Grid from String

```java
import wklm.aoc.*;

// Parse a simple character grid
String input = """
    ###.
    #..#
    ####
    """;

Optional<Grid<Character>> gridOpt = Grid.fromString(input, c -> c);
Grid<Character> grid = gridOpt.orElseThrow();

// Get dimensions
System.out.println("Rows: " + grid.getRows());      // 3
System.out.println("Cols: " + grid.getCols());      // 4

// Get a point
Optional<Point<Character>> point = grid.get(0, 0);
System.out.println(point.get().value());  // '#'
```

### Custom Parsing

```java
// Parse digits into integers
String input = """
    123
    456
    789
    """;

Optional<Grid<Integer>> grid = Grid.fromString(input,
    c -> Character.isDigit(c) ? Character.getNumericValue(c) : null
);
```

### Finding Points by Value

```java
// Find all walls in a maze
Optional<List<Point<Character>>> walls = grid.find('#');
if (walls.isPresent()) {
    System.out.println("Found " + walls.get().size() + " walls");
}
```

### Getting Neighbors

```java
import wklm.aoc.Connectivity;

Point<Character> point = grid.get(1, 1).get();

// 4-directional (up, down, left, right)
List<Point<Character>> neighbors4 = grid.getNeighbors(point, Connectivity.FOUR);

// 8-directional (including diagonals)
List<Point<Character>> neighbors8 = grid.getNeighbors(point, Connectivity.EIGHT);
```

## Advanced Features

### Toroidal Grids (Wrap-Around)

```java
// Create a grid where edges wrap around (Pac-Man style)
Optional<Grid<Character>> toroidalGrid = Grid.fromToroidalString(input, c -> c);

// Or specify behavior explicitly
Optional<Grid<Character>> grid = Grid.fromString(input, c -> c,
    Grid.BehaviorType.TOROIDAL);
```

### Connected Components (Flood Fill)

```java
// Find all connected regions of '.' (empty spaces)
List<List<Point<Character>>> components =
    grid.getConnectedPoints('.', Connectivity.FOUR);

System.out.println("Found " + components.size() + " separate regions");
for (List<Point<Character>> component : components) {
    System.out.println("  Region with " + component.size() + " points");
}
```

### Pathfinding with Dijkstra's Algorithm

```java
// Find shortest path with uniform cost (each step costs 1)
Optional<List<Point<Integer>>> path = grid.findShortestPathDijkstra(
    0, 0,        // start row, col
    9, 9,        // end row, col
    Connectivity.FOUR
);

if (path.isPresent()) {
    System.out.println("Path length: " + path.get().size());
    System.out.println("Path cost: " + (path.get().size() - 1));
}
```

### Custom Movement Costs

```java
// Use point values as movement costs (useful for weighted grids)
Grid.MovementCostFunction<Integer> costFunc =
    (from, to) -> to.value().doubleValue();

Optional<List<Point<Integer>>> path = grid.findShortestPathDijkstra(
    0, 0,
    9, 9,
    Connectivity.FOUR,
    costFunc
);
```

### Grid Transformations

```java
// Transpose the grid (swap rows and columns)
Grid<Character> transposed = grid.transpose();

// Union two grids with a merge function
Grid<Integer> grid1 = /* ... */;
Grid<Integer> grid2 = /* ... */;

Optional<Grid<Integer>> merged = grid1.union(grid2,
    (a, b) -> Math.max(a, b)  // Keep maximum value
);
```

### Programmatic Grid Construction

```java
import wklm.aoc.*;
import java.util.*;

// Create a 5x5 grid programmatically
Dimensions dims = new Dimensions(5, 5);
List<Point<Character>> points = new ArrayList<>();

// Add some specific points
points.add(new Point<>(0, 0, '#'));
points.add(new Point<>(0, 4, '#'));
points.add(new Point<>(4, 0, '#'));
points.add(new Point<>(4, 4, '#'));

// Create grid with '.' as default value and standard behavior
Grid<Character> grid = new Grid<>(
    points,
    dims,
    Optional.of('.'),
    Optional.of(Grid.BehaviorType.STANDARD)
);
```

## Visualization

### Simple String Output

```java
System.out.println(grid.toString());
// Output:
// ###.
// #..#
// ####
```

### Formatted Grid with Indices

```java
System.out.println(grid.toGridString(true));
// Output:
//       0  1  2  3
//    +----------------
//  0| # # # .
//  1| # . . #
//  2| # # # #
```

### Graph Representation

```java
// Show adjacency list
System.out.println(grid.toGraphString(Connectivity.FOUR, false, null));
// Output:
// Graph Representation (Adjacency List):
// Node(0,0) -> [(0,1), (1,0)]
// Node(0,1) -> [(0,0), (0,2), (1,1)]
// ...
```

## API Reference

### Core Classes

#### `Grid<T>`

The main grid class providing all grid operations.

**Factory Methods:**
- `static <T> Optional<Grid<T>> fromString(String input, CharacterParser<T> parser)` - Create standard grid
- `static <T> Optional<Grid<T>> fromString(String input, CharacterParser<T> parser, BehaviorType behaviorType)` - Create grid with specific behavior
- `static <T> Optional<Grid<T>> fromToroidalString(String input, CharacterParser<T> parser)` - Create toroidal grid

**Constructor:**
- `Grid(List<Point<T>> points, Dimensions dimensions, Optional<T> defaultValue, Optional<BehaviorType> bt)` - Programmatic construction

**Query Methods:**
- `Optional<Point<T>> get(int row, int col)` - Get point at coordinates
- `Optional<List<Point<T>>> find(T value)` - Find all points with value
- `List<Point<T>> getNeighbors(Point<T> point, Connectivity connectivity)` - Get neighboring points
- `List<T> getUniqueValues()` - Get all unique values in grid
- `Dimensions getDimensions()` - Get grid dimensions
- `int getRows()` - Get number of rows
- `int getCols()` - Get number of columns

**Algorithm Methods:**
- `List<List<Point<T>>> getConnectedPoints(T targetValue, Connectivity connectivity)` - Find connected components
- `Optional<List<Point<T>>> findShortestPathDijkstra(int startRow, int startCol, int endRow, int endCol, Connectivity connectivity)` - Find shortest path
- `Optional<List<Point<T>>> findShortestPathDijkstra(int startRow, int startCol, int endRow, int endCol, Connectivity connectivity, MovementCostFunction<T> costFunc)` - Find shortest path with custom costs

**Transformation Methods:**
- `Grid<T> transpose()` - Transpose rows and columns
- `Optional<Grid<T>> union(Grid<T> other, BiFunction<T, T, T> merger)` - Merge two grids

**Visualization Methods:**
- `String toString()` - Simple string representation
- `String toGridString(boolean includeIndices)` - Formatted grid
- `String toGraphString(Connectivity connectivity, boolean includeWeights, MovementCostFunction<T> costFunc)` - Adjacency list

#### `Point<T>`

Immutable record representing a grid cell.

```java
public record Point<T>(int x, int y, T value) {}
```

- `x` - Row index
- `y` - Column index
- `value` - Value at this position

#### `Dimensions`

Immutable record representing grid dimensions.

```java
public record Dimensions(int nrows, int ncols) {}
```

#### `Connectivity`

Enum for neighbor connectivity types.

- `FOUR` - 4-directional (up, down, left, right)
- `EIGHT` - 8-directional (4-directional + diagonals)

Methods:
- `int getValue()` - Get numeric value (4 or 8)
- `static Connectivity fromInt(int value)` - Create from int

#### `BehaviorType`

Enum for grid edge behavior.

- `STANDARD` - Edges are boundaries
- `TOROIDAL` - Edges wrap around

#### `CharacterParser<T>`

Functional interface for parsing characters.

```java
@FunctionalInterface
public interface CharacterParser<T> {
    T parse(char c);
}
```

#### `MovementCostFunction<T>`

Functional interface for custom pathfinding costs.

```java
@FunctionalInterface
public interface MovementCostFunction<T> {
    double calculate(Point<T> from, Point<T> to);
}
```

## Common Patterns

### Parsing AOC Input

```java
// Example: AOC 2023 Day X - height map
String input = Files.readString(Path.of("input.txt"));
Optional<Grid<Integer>> heightMap = Grid.fromString(input,
    c -> Character.isDigit(c) ? Character.getNumericValue(c) : 0
);
```

### Finding Shortest Path in Maze

```java
// Find start and end positions
Point<Character> start = grid.find('S').get().get(0);
Point<Character> end = grid.find('E').get().get(0);

// Find path avoiding walls
Optional<List<Point<Character>>> path = grid.findShortestPathDijkstra(
    start.x(), start.y(),
    end.x(), end.y(),
    Connectivity.FOUR
);
```

### Flood Fill / Region Counting

```java
// Count separate water regions
List<List<Point<Character>>> waterRegions =
    grid.getConnectedPoints('~', Connectivity.FOUR);

int largestRegion = waterRegions.stream()
    .mapToInt(List::size)
    .max()
    .orElse(0);
```

### Custom Grid Traversal

```java
// Visit all points in order
for (int row = 0; row < grid.getRows(); row++) {
    for (int col = 0; col < grid.getCols(); col++) {
        Point<Character> point = grid.get(row, col).get();
        // Process point...
    }
}
```

## Performance Considerations

### Efficient Value Lookups

The `Grid` class maintains an internal index (`Map<T, List<Point<T>>>`) for O(1) value lookups:

```java
// This is very fast - O(1) lookup
Optional<List<Point<Character>>> walls = grid.find('#');
```

### Immutability Benefits

All grids are immutable, providing:
- **Thread safety** - Share grids across threads safely
- **Predictability** - No unexpected mutations
- **Caching** - Safe to cache and reuse

### Memory Usage

- Each `Point<T>` object is created once and reused
- The locations index adds memory overhead but enables fast lookups
- For large grids (>1000x1000), consider memory constraints

## Design Decisions

### Why Immutable?

Immutability provides safety and enables:
- Fearless concurrency
- No defensive copying needed
- Easier reasoning about code
- Better cache locality

### Why Optional Returns?

Methods return `Optional` to:
- Make null-safety explicit
- Force handling of edge cases
- Follow modern Java best practices

However, some methods return empty lists instead of `Optional<List<>>` for better ergonomics:
- `getNeighbors()` returns `List` (empty if invalid)
- `getConnectedPoints()` returns `List<List<>>` (empty if none found)

### Why Enums Instead of Ints?

Using `Connectivity` enum instead of magic numbers (4/8):
- Type safety - can't pass invalid values
- Self-documenting code
- IDE autocomplete support
- Easier to extend in the future

## Changelog

### Version 0.0.2 (Current)

**Bug Fixes:**
- Fixed critical `IndexOutOfBoundsException` in Grid constructor
- Fixed broken immutability in public constructor
- Added null validation to constructors

**API Improvements:**
- Added `Connectivity` enum (replaced magic int values)
- Simplified Optional usage - some methods now return empty lists
- Added `getDimensions()`, `getRows()`, `getCols()` accessor methods
- `getNeighbors()` now returns `List` instead of `Optional<List>`
- `getConnectedPoints()` now returns `List<List>` instead of `Optional<List<List>>`

**Performance:**
- Optimized transpose to preserve behavior without instanceof check
- Better ArrayList initialization patterns

**Breaking Changes:**
- All methods using `int connectivity` now use `Connectivity` enum
- `getNeighbors()` returns `List` instead of `Optional<List>`
- `getConnectedPoints()` returns `List<List>` instead of `Optional<List<List>>`

### Version 0.0.1

- Initial release

## Contributing

Contributions welcome! This library is designed for Advent of Code, so features should align with common AOC problem patterns.

## License

This project is released under the MIT License.

## Author

**wklm** - Built for Advent of Code enthusiasts

## Acknowledgments

Inspired by the amazing [Advent of Code](https://adventofcode.com/) by Eric Wastl.
