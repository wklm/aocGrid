package wklm.aoc;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the Connectivity enum.
 */
class ConnectivityTest {

    @Test
    void testFourConnectivityValue() {
        assertEquals(4, Connectivity.FOUR.getValue());
    }

    @Test
    void testEightConnectivityValue() {
        assertEquals(8, Connectivity.EIGHT.getValue());
    }

    @Test
    void testFromIntFour() {
        assertEquals(Connectivity.FOUR, Connectivity.fromInt(4));
    }

    @Test
    void testFromIntEight() {
        assertEquals(Connectivity.EIGHT, Connectivity.fromInt(8));
    }

    @Test
    void testFromIntInvalidThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> {
            Connectivity.fromInt(6);
        });
    }

    @Test
    void testFromIntZeroThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> {
            Connectivity.fromInt(0);
        });
    }

    @Test
    void testFromIntNegativeThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> {
            Connectivity.fromInt(-1);
        });
    }

    @Test
    void testEnumValues() {
        Connectivity[] values = Connectivity.values();
        assertEquals(2, values.length);
        assertTrue(values[0] == Connectivity.FOUR || values[0] == Connectivity.EIGHT);
        assertTrue(values[1] == Connectivity.FOUR || values[1] == Connectivity.EIGHT);
    }

    @Test
    void testValueOf() {
        assertEquals(Connectivity.FOUR, Connectivity.valueOf("FOUR"));
        assertEquals(Connectivity.EIGHT, Connectivity.valueOf("EIGHT"));
    }

    @Test
    void testValueOfInvalidThrowsException() {
        assertThrows(IllegalArgumentException.class, () -> {
            Connectivity.valueOf("SIX");
        });
    }
}
