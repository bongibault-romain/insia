package lt.bongibau.scrapper.searching;

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
            searchers.add(searcher);
        }

        while (!heap.isEmpty() || this.isSearchersWorking()) {
            // TODO: implement
            throw new UnsupportedOperationException("Not implemented");
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

    @Override
    public synchronized void notify(URL baseUrl, List<String> links) {
        // TODO: implement, add links to the heap, normalize them, etc.
        throw new UnsupportedOperationException("Not implemented");
    }

    public static SearchManager getInstance() {
        if (instance == null) {
            instance = new SearchManager();
        }

        return instance;
    }
}
