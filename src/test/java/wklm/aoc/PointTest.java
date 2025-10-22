package wklm.aoc;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the Point record.
 */
class PointTest {

    @Test
    void testPointCreation() {
        Point<Integer> point = new Point<>(5, 10, 42);
        assertEquals(5, point.x());
        assertEquals(10, point.y());
        assertEquals(42, point.value());
    }

    @Test
    void testPointWithNullValue() {
        Point<String> point = new Point<>(0, 0, null);
        assertEquals(0, point.x());
        assertEquals(0, point.y());
        assertNull(point.value());
    }

    @Test
    void testPointEquality() {
        Point<Integer> p1 = new Point<>(1, 2, 10);
        Point<Integer> p2 = new Point<>(1, 2, 10);
        Point<Integer> p3 = new Point<>(1, 2, 20);
        Point<Integer> p4 = new Point<>(2, 1, 10);

        assertEquals(p1, p2);
        assertNotEquals(p1, p3); // Different value
        assertNotEquals(p1, p4); // Different coordinates
    }

    @Test
    void testPointHashCode() {
        Point<Integer> p1 = new Point<>(1, 2, 10);
        Point<Integer> p2 = new Point<>(1, 2, 10);

        assertEquals(p1.hashCode(), p2.hashCode());
    }

    @Test
    void testPointToString() {
        Point<Integer> point = new Point<>(3, 7, 99);
        String str = point.toString();
        assertTrue(str.contains("3"));
        assertTrue(str.contains("7"));
        assertTrue(str.contains("99"));
    }

    @Test
    void testPointWithDifferentTypes() {
        Point<String> stringPoint = new Point<>(0, 0, "test");
        Point<Character> charPoint = new Point<>(1, 1, 'A');
        Point<Double> doublePoint = new Point<>(2, 2, 3.14);

        assertEquals("test", stringPoint.value());
        assertEquals('A', charPoint.value());
        assertEquals(3.14, doublePoint.value());
    }

    @Test
    void testNegativeCoordinates() {
        Point<Integer> point = new Point<>(-5, -10, 42);
        assertEquals(-5, point.x());
        assertEquals(-10, point.y());
    }
}
