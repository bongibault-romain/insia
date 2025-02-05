package lt.bongibau.scrapper.searching;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.LinkedList;
import java.util.List;

public class Searcher extends Thread {

    private List<String> heap = new LinkedList<>();

    private boolean running = false;

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
        }
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

    @Nullable
    public synchronized String pop() {
        return heap.isEmpty() ? null : heap.removeFirst();
    }
}
