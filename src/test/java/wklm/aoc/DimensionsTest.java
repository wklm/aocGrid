package wklm.aoc;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the Dimensions record.
 */
class DimensionsTest {

    @Test
    void testValidDimensions() {
        Dimensions dims = new Dimensions(5, 10);
        assertEquals(5, dims.nrows());
        assertEquals(10, dims.ncols());
    }

    @Test
    void testZeroDimensions() {
        Dimensions dims = new Dimensions(0, 0);
        assertEquals(0, dims.nrows());
        assertEquals(0, dims.ncols());
    }

    @Test
    void testNegativeRowsThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> {
            new Dimensions(-1, 5);
        });
    }

    @Test
    void testNegativeColsThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> {
            new Dimensions(5, -1);
        });
    }

    @Test
    void testBothNegativeThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> {
            new Dimensions(-5, -10);
        });
    }

    @Test
    void testDimensionsEquality() {
        Dimensions d1 = new Dimensions(3, 4);
        Dimensions d2 = new Dimensions(3, 4);
        Dimensions d3 = new Dimensions(4, 3);

        assertEquals(d1, d2);
        assertNotEquals(d1, d3);
    }

    @Test
    void testDimensionsHashCode() {
        Dimensions d1 = new Dimensions(5, 7);
        Dimensions d2 = new Dimensions(5, 7);

        assertEquals(d1.hashCode(), d2.hashCode());
    }

    @Test
    void testDimensionsToString() {
        Dimensions dims = new Dimensions(10, 20);
        String str = dims.toString();
        assertTrue(str.contains("10"));
        assertTrue(str.contains("20"));
    }

    @Test
    void testLargeDimensions() {
        Dimensions dims = new Dimensions(10000, 50000);
        assertEquals(10000, dims.nrows());
        assertEquals(50000, dims.ncols());
    }
}
