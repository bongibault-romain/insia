package lt.bongibau.scrapper.searching.filters;

import java.net.URL;

/**
 * Represents a filter for URLs, which can be used to accept or deny URLs
 */
public record Filter(String host, lt.bongibau.scrapper.searching.filters.Filter.Type type) {
    public enum Type {
        DENY,
        ACCEPT
    }

    /**
     * Check if the URL is accepted by the filter
     *
     * @param url URL to check
     * @return true if the URL is accepted by the filter
     */
    public boolean check(URL url) {
        return url.getHost().contains(host);
    }
}
