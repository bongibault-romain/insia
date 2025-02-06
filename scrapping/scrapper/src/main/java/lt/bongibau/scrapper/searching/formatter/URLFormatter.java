package lt.bongibau.scrapper.searching.formatter;

import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;

public class URLFormatter {
    static URL format(URL url){
        return url;
    }
    static URL hrefToUrl(URL baseUrl, String href)  {
        if(!hrefIsValid(href))throw new NotValidHrefException();
        try {
            if(href.startsWith("https://")||href.startsWith("http://")){
                return new URI(href).toURL();
            }
            if(href.startsWith("/"))return new URI(baseUrl.getProtocol()+"://"+baseUrl.getHost()+href).toURL();
        }catch (URISyntaxException | MalformedURLException e){
            throw new NotValidHrefException();
        }

        return baseUrl;
    }
    static boolean hrefIsValid(String href){
        if(href.startsWith("/")||href.startsWith("https://")||href.startsWith("http://"))return true;
        return !href.contains(":");
    }
}
