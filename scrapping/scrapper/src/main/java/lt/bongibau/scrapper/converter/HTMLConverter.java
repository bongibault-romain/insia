package lt.bongibau.scrapper.converter;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.select.Elements;

import java.io.IOException;
import java.io.InputStream;

public class HTMLConverter implements IConverter {
    @Override
    public String translate(InputStream source) {
        try {

            Document doc = Jsoup.parse(source, "UTF-8", "");

            // Texts longer than 10 characters
            Elements elements = doc.getElementsMatchingText(".{15,}");

            StringBuilder builder = new StringBuilder();
            for (var element : elements) {
                if (!element.is("p")) continue;
                builder.append(element.text());
                builder.append("\n");
            }

            return builder.toString();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
