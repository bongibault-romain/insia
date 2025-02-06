package lt.bongibau.scrapper;

import lt.bongibau.scrapper.searching.SearchManager;
import lt.bongibau.scrapper.searching.Searcher;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.*;

public class Scrapper {
    public static void main(String[] args) throws IOException, InterruptedException, URISyntaxException {
        URI test = new URI("https://api.google.com/search?q=java&a=2#salut");

        System.out.println(test.resolve("sapristie.j'ai/faim#hello"));

    }
}
