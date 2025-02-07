package lt.bongibau.scrapper;

import java.util.logging.Level;
import java.util.logging.Logger;

public class ScrapperLogger {

    private static ScrapperLogger instance;

    private Logger logger;

    private ScrapperLogger() {
        logger = Logger.getLogger(ScrapperLogger.class.getName());
    }

    public synchronized static void log(Level level, String message) {
        getInstance().getLogger().log(level, "[" + Thread.currentThread().getName() + "] " + message);
    }

    public synchronized static void log(Level level, String message, Throwable throwable) {
        getInstance().getLogger().log(level, message, throwable);
    }

    public synchronized static void log(String message) {
        log(Level.INFO, message);
    }

    public Logger getLogger() {
        return logger;
    }

    public static ScrapperLogger getInstance() {
        if (instance == null) {
            instance = new ScrapperLogger();
        }

        return instance;
    }
}
