package lt.bongibau.scrapper.converter;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public interface IConverter {

    String translate(InputStream source);

    default String translate(URL source) throws IOException {
        HttpURLConnection connection = null;

        connection = (HttpURLConnection) source.openConnection();
        connection.setRequestMethod("GET");
        connection.connect();

        return translate(connection.getInputStream());
    }

}
