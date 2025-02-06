package lt.bongibau.scrapper.searching.formatter;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

class URLFormatterTest {

    public static Stream<Arguments> provideHrefIsValid() {
        return Stream.of(
                Arguments.of("https://example.com", true),
                Arguments.of("http://example.com", true),
                Arguments.of("http://example.com/", true),
                Arguments.of("http://example.com/home/", true),
                Arguments.of("http://example.com/home", true),
                Arguments.of("http://example.com/search?q=hello:world", true),
                Arguments.of("/home", true),
                Arguments.of("/home/", true),
                Arguments.of("/articles/012891141", true),
                Arguments.of("/articles/012891141/", true),
                Arguments.of("/home/?q=hello?b=world", true),
                Arguments.of("/home?q=hello?b=world", true),
                Arguments.of("/home?q=hello:world", true),
                Arguments.of("page.html", true),
                Arguments.of("page.html?q=hello?b=world", true),
                Arguments.of("page.html?q=hello:world", true),
                Arguments.of("page.html?q=hello?b=world#section", true),
                Arguments.of("home/page.html?q=hello?b=world#section", true),

                Arguments.of("javascript:alert('Hello')", false),
                Arguments.of("mailto:example@example.com", false),
                Arguments.of("ftp://example.com", false),
                Arguments.of("tel:123456789", false),
                Arguments.of("file:///home/user/file.txt", false),
                Arguments.of("data:text/plain;base64,SGVsbG8sIFdvcmxkIQ==", false),
                Arguments.of("data:application/octet-stream;base64,SGVsbG8sIFdvcmxkIQ==", false)
        );
    }

    @ParameterizedTest
    @MethodSource("provideHrefIsValid")
    @DisplayName("Test hrefIsValid")
    void hrefIsValidTest(String href, boolean expected) {
        assertEquals(expected, URLFormatter.hrefIsValid(href), "hrefIsValid should return expected value.");
    }

}