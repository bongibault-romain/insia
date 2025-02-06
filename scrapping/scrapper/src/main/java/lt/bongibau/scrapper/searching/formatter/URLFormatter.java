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
    static URI format(URI url){
        String stringUrl = url.toString();
        if(stringUrl.contains("?")){
            String[] parts = stringUrl.split("\\?",1);
            if(!parts[0].endsWith("/")) stringUrl =parts[0]+"/?"+parts[1];
        }else{
            if(!stringUrl.endsWith("/")) stringUrl +="/";
        }
        if(stringUrl.contains("#")){
            stringUrl = stringUrl.split("#",1)[0];
        }

        String querrySelector=url.getQuery();
        if(querrySelector!=null){
            String[] querryParts = querrySelector.split("&");
            Arrays.sort(querryParts);
            querrySelector = String.join("&",querryParts);
        }

        stringUrl = stringUrl.split("\\?",1)[0];
        if(querrySelector!=null)stringUrl+="?"+querrySelector;

        try {
            return new URI(stringUrl);
        } catch (URISyntaxException  e) {
            e.printStackTrace();
        }
        return url;
    }

    /**
     * Converts href to URI
     * and verifies if the href is valid
     * @param baseUrl Base URI
     * @param href Href to convert
     * @return URI
     * @throws NotValidHrefException If the href is not valid
     */
    static URI hrefToUrl(URI baseUrl, String href) throws NotValidHrefException {
        if(!hrefIsValid(href))throw new NotValidHrefException();
        //if(href.startsWith("https://")||href.startsWith("http://"))return new URI(href).toURL();
        //else if(href.startsWith("/"))return new URI(baseUrl.getProtocol()+"://"+baseUrl.getHost()+href).toURL();
        //else return new URI(baseUrl.getProtocol()+"://"+baseUrl.getHost()+baseUrl.getPath()+'/'+href).toURL();
        return baseUrl.resolve(href);
    }

    /**
     * Checks if the href is valid
     * @param href Href to check
     * @return true if the href is valid
     */
    static boolean hrefIsValid(String href){
        if(href==null)return false;
        if(href.startsWith("/")||href.startsWith("https://")||href.startsWith("http://"))return true;
        if(href.contains(":")){
            String prefix = href.split(":",1)[0];
            return prefix.contains("?");
        }
        return true;
    }
}
