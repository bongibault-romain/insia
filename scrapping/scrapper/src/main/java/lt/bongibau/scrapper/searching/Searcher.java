package lt.bongibau.scrapper.searching;

import org.jetbrains.annotations.Nullable;

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

    private final List<String> heap = new LinkedList<>();

    private final List<Searcher.Observer> observers = new LinkedList<>();

    private boolean running = false;

    private Phase phase = Phase.IDLE;

    @Override
    public void run() {
        while (this.isRunning()) {
            String url = this.pop();
            if (url == null) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                continue;
            }

            System.out.println("Searching: " + url);

            this.notifyAll(null, List.of("link1", "link2"));
        }
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

    public synchronized void add(String url) {
        heap.add(url);
    }

    public synchronized void addAll(List<String> urls) {
        heap.addAll(urls);
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

    @Nullable
    public synchronized String pop() {
        return heap.isEmpty() ? null : heap.removeFirst();
    }
}
