package lt.bongibau.scrapper.converter;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

import java.io.IOException;
import java.io.InputStream;

public class PDFConverter implements IConverter {
    @Override
    public String translate(InputStream source) {
        // read the PDF file using PDFBox

        try {
            PDDocument document = Loader.loadPDF(source.readAllBytes());

            PDFTextStripper stripper = new PDFTextStripper();

            return stripper.getText(document);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
