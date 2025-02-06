package lt.bongibau.scrapper.searching.formatter;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
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

    public static Stream<Arguments> provideHrefToURL() {
        try {
            return Stream.of(
                    Arguments.of(new URI("https://test.com"), "https://example.com", new URI("https://example.com")),
                    Arguments.of(new URI("https://example.com"), "http://example.com", new URI("http://example.com")),
                    Arguments.of(new URI("https://example.com"), "http://example.com/", new URI("http://example.com/")),
                    Arguments.of(new URI("https://example.com"), "http://example.com/home/", new URI("http://example.com/home/")),
                    Arguments.of(new URI("https://example.com"), "http://example.com/home", new URI("http://example.com/home")),
                    Arguments.of(new URI("https://example.com"), "http://example.com/search?q=hello:world", new URI("http://example.com/search?q=hello:world")),
                    Arguments.of(new URI("https://example.com"), "/home#hello-world", new URI("https://example.com/home#hello-world")),
                    Arguments.of(new URI("https://example.com"), "/home?test=1#hello-world", new URI("https://example.com/home?test=1#hello-world")),

                    Arguments.of(new URI("https://insa-toulouse.fr"), "/home", new URI("https://insa-toulouse.fr/home")),
                    Arguments.of(new URI("https://insa-toulouse.fr"), "/home/", new URI("https://insa-toulouse.fr/home/")),
                    Arguments.of(new URI("https://insa-toulouse.fr"), "/articles/012891141", new URI("https://insa-toulouse.fr/articles/012891141")),
                    Arguments.of(new URI("https://insa-toulouse.fr"), "/articles/012891141/", new URI("https://insa-toulouse.fr/articles/012891141/")),

                    Arguments.of(new URI("https://insa-toulouse.fr/hello/world"), "/home/?q=hello?b=world", new URI("https://insa-toulouse.fr/home/?q=hello?b=world")),
                    Arguments.of(new URI("https://insa-toulouse.fr/hello/world"), "/home?q=hello?b=world#test", new URI("https://insa-toulouse.fr/home?q=hello?b=world#test")),
                    Arguments.of(new URI("https://insa-toulouse.fr/hello/world"), "/home?q=hello:world", new URI("https://insa-toulouse.fr/home?q=hello:world")),

                    Arguments.of(new URI("https://insa-toulouse.fr/hello/world"), "page.html", new URI("https://insa-toulouse.fr/hello/page.html")),
                    Arguments.of(new URI("https://insa-toulouse.fr/hello/world"), "page.html?q=hello?b=world", new URI("https://insa-toulouse.fr/hello/page.html?q=hello?b=world")),
                    Arguments.of(new URI("https://insa-toulouse.fr/hello/world"), "page.html?q=hello:world", new URI("https://insa-toulouse.fr/hello/page.html?q=hello:world"))
            );
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    @ParameterizedTest
    @MethodSource("provideHrefToURL")
    @DisplayName("Test hrefToUrl")
    void hrefToUrlTest(URI baseUrl, String href, URI expected) {
        Assertions.assertDoesNotThrow(() -> {
            assertEquals(expected, URLFormatter.hrefToUrl(baseUrl, href), "hrefToUrl should return expected URL.");
        });
    }

    @ParameterizedTest
    @MethodSource("provideHrefIsValid")
    @DisplayName("Test hrefIsValid")
    void hrefIsValidTest(String href, boolean expected) {
        assertEquals(expected, URLFormatter.hrefIsValid(href), "hrefIsValid should return expected value.");
    }

}