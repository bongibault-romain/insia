package lt.bongibau.scrapper.searching;

import lt.bongibau.scrapper.ScrapperLogger;
import org.jetbrains.annotations.Nullable;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Attribute;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.LinkedList;
import java.util.List;

public class Searcher extends Thread {
    public enum Phase {
        WORKING,
        IDLE
    }

    public static interface Observer {
        /**
         * Called when the searcher finds links on a page,
         * basically when the searcher is working and finished
         * one loop of searching
         *
         * @param baseUrl Base URL of the page, where the links were found
         * @param links List of links found on the page, links are relative to the base URL
         *              and are not formatted, they are just the href attribute values.
         *              They should be formatted before using them.
         */
        void notify(URL baseUrl, List<String> links);
    }

    private final List<URL> heap = new LinkedList<>();

    private final List<Searcher.Observer> observers = new LinkedList<>();

    private boolean running = false;

    private Phase phase = Phase.IDLE;

    @Override
    public void run() {
        while (this.isRunning()) {
            this.setPhase(Phase.WORKING);

            URL url = this.pop();
            if (url == null) {
                this.setPhase(Phase.IDLE);
                ScrapperLogger.log("No work found, sleeping...");

                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                continue;
            }

            ScrapperLogger.log("Processing URL: " + url);

            HttpURLConnection connection = null;
            try {
                connection = (HttpURLConnection) url.openConnection();

                connection.setInstanceFollowRedirects(false);
                connection.setRequestMethod("GET");
                connection.connect();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            int status = 0;
            try {
                status = connection.getResponseCode();
            } catch (IOException e) {
                e.printStackTrace();
                continue;
            }

            if (status == HttpURLConnection.HTTP_MOVED_TEMP
                    || status == HttpURLConnection.HTTP_MOVED_PERM) {
                String location = connection.getHeaderField("Location");

                ScrapperLogger.log("Redirecting to: " + location + ", adding to heap and notifying all.");

                this.notifyAll(url, List.of(location));
                continue;
            }

            if (status != HttpURLConnection.HTTP_OK) {
                ScrapperLogger.log("Failed to connect to: " + url + ", status: " + status);
                continue;
            }

            StringBuilder content = new StringBuilder();
            try {
                BufferedReader in = new BufferedReader(
                        new InputStreamReader(connection.getInputStream()));
                String inputLine;
                while ((inputLine = in.readLine()) != null) {
                    content.append(inputLine);
                }
                in.close();

            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            Document document = Jsoup.parse(content.toString());
            List<String> links = document.select("a").stream().map((a) -> a.attr("href")).toList();
            this.notifyAll(url, links);
        }

        ScrapperLogger.log("Searcher stopped.");
    }

    public synchronized void setPhase(Searcher.Phase phase) {
        this.phase = phase;
    }

    public synchronized Searcher.Phase getPhase() {
        return phase;
    }

    public synchronized boolean isRunning() {
        return running;
    }

    public synchronized void setRunning(boolean running) {
        this.running = running;
    }

    public synchronized void push(URL url) {
        heap.add(url);
    }

    public synchronized void notifyAll(URL baseUrl, List<String> links) {
        for (Searcher.Observer observer : observers) {
            observer.notify(baseUrl, links);
        }
    }

    public synchronized void subscribe(Searcher.Observer observer) {
        observers.add(observer);
    }

    public synchronized void unsubscribe(Searcher.Observer observer) {
        observers.remove(observer);
    }

    public synchronized boolean hasWork() {
        return !heap.isEmpty();
    }

    @Nullable
    public synchronized URL pop() {
        return heap.isEmpty() ? null : heap.removeFirst();
    }
}
