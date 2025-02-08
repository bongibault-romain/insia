package lt.bongibau.scrapper.searching;

import lt.bongibau.scrapper.Scrapper;
import lt.bongibau.scrapper.ScrapperLogger;
import lt.bongibau.scrapper.searching.filters.FilterContainer;
import lt.bongibau.scrapper.searching.formatter.NotValidHrefException;
import lt.bongibau.scrapper.searching.formatter.URLFormatter;
import org.jetbrains.annotations.Nullable;

import java.net.URI;
import java.net.URL;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Level;

public class SearchManager implements Searcher.Observer {

    private final FilterContainer filters;

    private List<Searcher> searchers;

    private List<URL> heap;

    private List<URL> visited;

    private List<URL> discovered;

    public SearchManager(List<URL> heap, FilterContainer filters) {
        this.heap = new LinkedList<>(heap);
        this.searchers = new ArrayList<>();
        this.visited = new LinkedList<>();
        this.discovered = new LinkedList<>();
        this.filters = filters;
    }

    public void start(int searcherCount) {
        for (int i = 0; i < searcherCount; i++) {
            Searcher searcher = new Searcher();
            searcher.setName("S" + i);

            searcher.subscribe(this);
            searchers.add(searcher);
            searcher.setRunning(true);

            searcher.start();
        }

        while (!this.isEmpty() || this.isSearchersWorking()) {
            URL url = this.pop();
            if (url == null) {
                ScrapperLogger.log("No work found, sleeping...");
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                continue;
            }

            Searcher searcher = this.findSearcher();

            ScrapperLogger.log("Total Discovered: " + getDiscoveredCount() + " Total visited: " + getVisitedCount() + ". Global Heap Size: " + getWorkload() + ". Assigning " + url + " to searcher " + searcher.getName());

            searcher.push(url);
        }

        searchers.forEach((s) -> s.setRunning(false));
        searchers.forEach((s) -> {
            try {
                s.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

    /**
     * Check if at least one searcher is currently working. This is
     * used to determine if the search manager should wait for the
     * searchers to finish their work.
     *
     * @return true if at least one searcher is working
     */
    private boolean isSearchersWorking() {
        return searchers.stream().anyMatch((s) -> s.getPhase() == Searcher.Phase.WORKING || s.hasWork());
    }

    /**
     * Find a searcher that is not working or find the searcher with
     * the least amount of work.
     *
     * @return Searcher that is not working or has the least amount of work
     */
    private Searcher findSearcher() {
        // s.getWorkload && s.getPhase
        return searchers.stream().min((s1, s2) -> {
            if (s1.getPhase() == Searcher.Phase.WORKING && s2.getPhase() == Searcher.Phase.WORKING) {
                return s1.getWorkload() - s2.getWorkload();
            }

            if (s1.getPhase() == Searcher.Phase.WORKING) {
                return 1;
            }

            if (s2.getPhase() == Searcher.Phase.WORKING) {
                return -1;
            }

            return 0;
        }).orElse(null);
    }

    private synchronized boolean isEmpty() {
        return heap.isEmpty();
    }

    private synchronized int getWorkload() {
        return heap.size();
    }

    private synchronized int getVisitedCount() {
        return visited.size();
    }

    private synchronized int getDiscoveredCount() {
        return discovered.size();
    }

    @Nullable
    private synchronized URL pop() {
        if (heap.isEmpty()) {
            return null;
        }

        return heap.removeFirst();
    }

    public synchronized void push(URL url) {
        heap.add(url);
    }

    public synchronized boolean isVisited(URL url) {
        return visited.contains(url);
    }

    public synchronized void markVisited(URL url) {
        if (visited.contains(url)) {
            ScrapperLogger.log(Level.WARNING, "Trying to mark already visited URL: " + url + ". Skipping...");
            return;
        }

        visited.add(url);
    }

    public synchronized void markDiscovered(URL url) {
        if (discovered.contains(url)) {
            ScrapperLogger.log(Level.WARNING, "Trying to mark already discovered URL: " + url + ". Skipping...");
            return;
        }

        discovered.add(url);
    }

    public synchronized boolean isDiscovered(URL url) {
        return discovered.contains(url);
    }

    @Override
    public synchronized void notify(URL baseUrl, List<String> links) {
        ScrapperLogger.log("Found " + links.size() + " links on " + baseUrl);

        this.markVisited(baseUrl);

        int added = 0;

        for (String link : links) {
            URL url;
            try {
                 url = URLFormatter.hrefToUrl(baseUrl, link);
            } catch (NotValidHrefException e) {
                continue;
            }

            try {
                url = URLFormatter.format(url);
            } catch (Exception e) {
                ScrapperLogger.log(Level.SEVERE, "Failed to format URL: " + url, e);
                continue;
            }

            if (this.isDiscovered(url)) {
                continue;
            }

            if (!filters.check(url)) {
                continue;
            }

            this.markDiscovered(url);
            this.push(url);
            added++;
        }

        ScrapperLogger.log("Added " + added + " new links to the heap");
    }
}
