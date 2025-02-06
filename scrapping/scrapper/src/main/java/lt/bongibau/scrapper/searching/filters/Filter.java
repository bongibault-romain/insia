package lt.bongibau.scrapper.searching.filters;

import java.net.URI;
import java.net.URL;
import java.util.Arrays;

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
     * @param uri URI to check
     * @return true if the URL is accepted by the filter
     */
    public boolean check(URI uri) {
        String[] urlDomains = uri.getHost().split("\\.");
        String[] hosts = host.split("\\.");

        if (hosts.length > urlDomains.length) {
            return false;
        }

        for (int i = hosts.length - 1; i >= 0; i--) {
            if (!hosts[i].equals(urlDomains[i + urlDomains.length - hosts.length])) {
                return false;
            }
        }

        return true;
    }
}
