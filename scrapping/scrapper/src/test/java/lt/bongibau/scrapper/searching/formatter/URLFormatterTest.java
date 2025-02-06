package lt.bongibau.scrapper.searching.formatter;

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
                    Arguments.of(new URI("https://test.com").toURL(), "https://example.com", new URI("https://example.com").toURL()),
                    Arguments.of(new URI("https://example.com").toURL(), "http://example.com", new URI("http://example.com").toURL()),
                    Arguments.of(new URI("https://example.com").toURL(), "http://example.com/", new URI("http://example.com/").toURL()),
                    Arguments.of(new URI("https://example.com").toURL(), "http://example.com/home/", new URI("http://example.com/home/").toURL()),
                    Arguments.of(new URI("https://example.com").toURL(), "http://example.com/home", new URI("http://example.com/home").toURL()),
                    Arguments.of(new URI("https://example.com").toURL(), "http://example.com/search?q=hello:world", new URI("http://example.com/search?q=hello:world").toURL()),
                    Arguments.of(new URI("https://example.com").toURL(), "/home#hello-world", new URI("https://example.com/home#hello-world").toURL()),
                    Arguments.of(new URI("https://example.com").toURL(), "/home?test=1#hello-world", new URI("https://example.com/home?test=1#hello-world").toURL()),

                    Arguments.of(new URI("https://insa-toulouse.fr").toURL(), "/home", new URI("https://insa-toulouse.fr/home").toURL()),
                    Arguments.of(new URI("https://insa-toulouse.fr").toURL(), "/home/", new URI("https://insa-toulouse.fr/home/").toURL()),
                    Arguments.of(new URI("https://insa-toulouse.fr").toURL(), "/articles/012891141", new URI("https://insa-toulouse.fr/articles/012891141").toURL()),
                    Arguments.of(new URI("https://insa-toulouse.fr").toURL(), "/articles/012891141/", new URI("https://insa-toulouse.fr/articles/012891141/").toURL()),

                    Arguments.of(new URI("https://insa-toulouse.fr/hello/world").toURL(), "/home/?q=hello?b=world", new URI("https://insa-toulouse.fr/home/?q=hello?b=world").toURL()),
                    Arguments.of(new URI("https://insa-toulouse.fr/hello/world").toURL(), "/home?q=hello?b=world#test", new URI("https://insa-toulouse.fr/home?q=hello?b=world#test").toURL()),
                    Arguments.of(new URI("https://insa-toulouse.fr/hello/world").toURL(), "/home?q=hello:world", new URI("https://insa-toulouse.fr/home?q=hello:world").toURL()),

                    Arguments.of(new URI("https://insa-toulouse.fr/hello/world").toURL(), "page.html", new URI("https://insa-toulouse.fr/hello/page.html").toURL()),
                    Arguments.of(new URI("https://insa-toulouse.fr/hello/world").toURL(), "page.html?q=hello?b=world", new URI("https://insa-toulouse.fr/hello/page.html?q=hello?b=world").toURL()),
                    Arguments.of(new URI("https://insa-toulouse.fr/hello/world").toURL(), "page.html?q=hello:world", new URI("https://insa-toulouse.fr/hello/page.html?q=hello:world").toURL())
            );
        } catch (MalformedURLException | URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    @ParameterizedTest
    @MethodSource("provideHrefToURL")
    @DisplayName("Test hrefToUrl")
    void hrefToUrlTest(URL baseUrl, String href, URL expected) {
        assertEquals(expected, URLFormatter.hrefToUrl(baseUrl, href).toString(), "hrefToUrl should return expected URL.");
    }

    @ParameterizedTest
    @MethodSource("provideHrefIsValid")
    @DisplayName("Test hrefIsValid")
    void hrefIsValidTest(String href, boolean expected) {
        assertEquals(expected, URLFormatter.hrefIsValid(href), "hrefIsValid should return expected value.");
    }

}