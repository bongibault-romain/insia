package lt.bongibau.scrapper.searching;

import lt.bongibau.scrapper.searching.filters.FilterContainer;
import lt.bongibau.scrapper.searching.formatter.NotValidHrefException;
import lt.bongibau.scrapper.searching.formatter.URLFormatter;
import org.jetbrains.annotations.Nullable;

import java.net.URI;
import java.net.URL;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class SearchManager implements Searcher.Observer {

    private final FilterContainer filters;

    private List<Searcher> searchers;

    private List<URL> heap;

    private List<URL> visited;

    public SearchManager(List<URL> heap, FilterContainer filters) {
        this.heap = new LinkedList<>(heap);
        this.searchers = new ArrayList<>();
        this.visited = new LinkedList<>();
        this.filters = filters;
    }

    public void start(int searcherCount) {
        System.out.println("Starting search manager with " + searcherCount + " searchers");
        for (int i = 0; i < searcherCount; i++) {
            Searcher searcher = new Searcher();
            searcher.start();
            searcher.setRunning(true);
            searcher.subscribe(this);
            searchers.add(searcher);
        }

        System.out.println("[MAIN] Search manager started");
        while (!this.isEmpty() || this.isSearchersWorking()) {
            URL url = this.pop();
            if (url == null) {
                System.out.println("[MAIN] IDLE");
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println(isSearchersWorking());
                continue;
            }

            System.out.println("[MAIN] Popped " + url);

            Searcher searcher = this.findSearcher();
            searcher.push(url);

            System.out.println(isSearchersWorking());
        }

        System.out.println("[MAIN] Search manager is waiting for searchers to finish");

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
        return searchers.stream().min((s1, s2) -> {
            if (s1.getPhase() == Searcher.Phase.IDLE && s2.getPhase() == Searcher.Phase.WORKING) {
                return -1;
            } else if (s1.getPhase() == Searcher.Phase.WORKING && s2.getPhase() == Searcher.Phase.IDLE) {
                return 1;
            } else {
                return 0;
            }
        }).orElse(searchers.getFirst());
    }

    private synchronized boolean isEmpty() {
        return heap.isEmpty();
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
        if (visited.contains(url)) return;

        visited.add(url);
    }

    @Override
    public synchronized void notify(URL baseUrl, List<String> links) {
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
                System.out.println("Failed to format URL: " + url);
                continue;
            }

            if (this.isVisited(url)) {
                continue;
            }

            if (!filters.check(url)) {
                continue;
            }

            this.markVisited(url);
            this.push(url);
        }
    }
}
