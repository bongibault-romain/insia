package lt.bongibau.scrapper.searching;

import lt.bongibau.scrapper.searching.formatter.URLFormatter;
import org.jetbrains.annotations.Nullable;

import java.net.URI;
import java.net.URL;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class SearchManager implements Searcher.Observer {

    private static SearchManager instance;

    private List<Searcher> searchers = new ArrayList<>();

    private List<URL> heap = new LinkedList<>();

    private List<URL> visited = new LinkedList<>();

    public void start(int searcherCount) {
        for (int i = 0; i < searcherCount; i++) {
            Searcher searcher = new Searcher();
            searcher.start();
            searcher.setRunning(true);
            searcher.subscribe(this);
            searchers.add(searcher);
        }

        while (!this.isEmpty() || this.isSearchersWorking()) {
            URL url = this.pop();
            if (url == null) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                continue;
            }

            Searcher searcher = this.findSearcher();
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
        return searchers.stream().anyMatch((s) -> s.getPhase() == Searcher.Phase.WORKING);
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
        }
    }

    public static SearchManager getInstance() {
        if (instance == null) {
            instance = new SearchManager();
        }

        return instance;
    }
}
