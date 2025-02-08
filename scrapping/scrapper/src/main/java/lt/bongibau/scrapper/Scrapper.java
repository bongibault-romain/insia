package lt.bongibau.scrapper;

import lt.bongibau.scrapper.searching.SearchManager;
import lt.bongibau.scrapper.searching.Searcher;
import lt.bongibau.scrapper.searching.filters.Filter;
import lt.bongibau.scrapper.searching.filters.FilterContainer;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.*;
import java.util.Arrays;
import java.util.List;

public class Scrapper {
    public static void main(String[] args) throws IOException, InterruptedException, URISyntaxException {
        FilterContainer filters = new FilterContainer(List.of(
                new Filter("insa-toulouse.fr", Filter.Type.ACCEPT)
        ));

        SearchManager searchManager = new SearchManager(
                List.of(new URL("https://www.insa-toulouse.fr")),
                filters
        );

        searchManager.start(2);

    }
}
