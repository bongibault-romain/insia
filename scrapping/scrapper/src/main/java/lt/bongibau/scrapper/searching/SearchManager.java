package lt.bongibau.scrapper.searching;

import java.util.ArrayList;
import java.util.List;

public class SearchManager {

    private static SearchManager instance;

    private List<Searcher> searchers = new ArrayList<>();

    public void start(int searcherCount) {
        for (int i = 0; i < searcherCount; i++) {
            Searcher searcher = new Searcher();
            searcher.start();
            searcher.setRunning(true);
            searchers.add(searcher);
        }

    }

    public static SearchManager getInstance() {
        if (instance == null) {
            instance = new SearchManager();
        }

        return instance;
    }
}
