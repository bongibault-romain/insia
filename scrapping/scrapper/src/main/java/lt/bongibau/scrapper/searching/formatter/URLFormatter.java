package lt.bongibau.scrapper.searching.formatter;

import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.Arrays;

public class URLFormatter {

    /**
     * Formats the URL to a static format
     * and sorts the query parameters
     * @param url URL to format
     * @return Formatted URL
     */
    public static URL format(URL url){
        try {
            String urlString = url.toString();

            int hashIndex = urlString.indexOf('#');
            if (hashIndex != -1) {
                urlString = urlString.substring(0, hashIndex);
            }

            String[] parts = urlString.split("\\?", 2);
            String baseUrl = parts[0];
            String queryParams;
            if(parts.length>1)if(parts[1].contains("="))queryParams = parts[1]; else queryParams = null;
            else queryParams = null;

            if (baseUrl.endsWith("/")) {
                baseUrl=baseUrl.substring(0, baseUrl.length() - 1);
            }

            // Trier les paramètres de requête
            if (queryParams != null) {
                String[] queryArray = queryParams.split("&");
                Arrays.sort(queryArray);
                queryParams = String.join("&", queryArray);
            }

            String formattedUrl;
            if(queryParams!=null) formattedUrl= baseUrl + "?" + queryParams;
            else formattedUrl = baseUrl;

            return new URI(formattedUrl).toURL();
        } catch (MalformedURLException | URISyntaxException e) {
            throw new RuntimeException("Failed to format URL", e);
        }
    }

    /**
     * Converts href to URI
     * and verifies if the href is valid
     * @param baseUrl Base URI
     * @param href Href to convert
     * @return URI
     * @throws NotValidHrefException If the href is not valid
     */
    public static URL hrefToUrl(URL baseUrl, String href) throws NotValidHrefException {
        if(!hrefIsValid(href))throw new NotValidHrefException();
        try {
            return baseUrl.toURI().resolve(href).toURL();
        } catch (URISyntaxException | MalformedURLException e) {
            throw new NotValidHrefException();
        }
    }

    /**
     * Checks if the href is valid
     * @param href Href to check
     * @return true if the href is valid
     */
    public static boolean hrefIsValid(String href){
        if(href==null)return false;
        if(href.startsWith("/")||href.startsWith("https://")||href.startsWith("http://"))return true;
        if(href.contains(":")){
            String prefix = href.split(":")[0];
            return prefix.contains("?");
        }
        return true;
    }
}
