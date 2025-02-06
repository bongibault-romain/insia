package lt.bongibau.scrapper.searching.filters;

import org.jetbrains.annotations.Unmodifiable;

import java.net.URL;
import java.util.Collections;
import java.util.List;

public class FilterContainer {
    private final List<Filter> filters;

    private final Filter.Type defaultPolicy;

    public FilterContainer(List<Filter> filters, Filter.Type defaultPolicy) {
        this.filters = filters;
        this.defaultPolicy = defaultPolicy;
    }

    public FilterContainer(List<Filter> filters) {
        this(filters, Filter.Type.DENY);
    }

    /**
     * Check if the URL is accepted by the filters,
     * if no filter accepts the URL, the default policy is used
     *
     * @param url URL to check
     * @return true if the URL is accepted by the filters
     */
    public boolean check(URL url) {
        for (Filter filter : filters) {
            if (filter.check(url)) {
                return filter.type() == Filter.Type.ACCEPT;
            }
        }

        return defaultPolicy == Filter.Type.ACCEPT;
    }

    public Filter.Type getDefaultPolicy() {
        return defaultPolicy;
    }

    @Unmodifiable
    public List<Filter> getFilters() {
        return Collections.unmodifiableList(this.filters);
    }
}
