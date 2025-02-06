package lt.bongibau.scrapper.searching;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.List;

public class URLRequester {

    private final URL url;

    public URLRequester(URL url) {
        this.url = url;
    }

    private void get() throws IOException {
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setInstanceFollowRedirects(false);
        connection.setRequestMethod("GET");
    }

    private List<>

}
