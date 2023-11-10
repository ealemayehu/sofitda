package sofitda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class TRECTokenizer extends AbstractDocumentsTokenizer {
  final String[] TRAINING_FILES = { "train_2000.label", "train_3000.label", "train_4000.label", "train_5500.label" };
  final String[] VALIDATION_FILES = { "train_1000.label" };
  final String[] TESTING_FILES = { "TREC_10.label" };

  public TRECTokenizer() throws IOException {
    super("trec", true /* hasResponse */);

    createDataset("training", TRAINING_FILES, false /* isLastPrefix */);
    createDataset("validation", VALIDATION_FILES, false /* isLastPrefix */);
    createDataset("testing", TESTING_FILES, true /* isLastPrefix */);
  }

  private void createDataset(String prefix, String[] dataFiles, boolean isLastPrefix) throws IOException {
    initialize(prefix);
    processResponse(prefix, dataFiles);
    done(prefix, isLastPrefix);
  }

  protected List<String> tokenize(String document) {
    String[] tokens = document.split(" ");

    return Arrays.asList(tokens);
  }

  protected String sentenceTerminator() {
    return ".";
  }

  private void processResponse(String prefix, String[] dataFiles) throws IOException {
    for (String dataFile : dataFiles) {
      String dataFilePath = rawDataDirectory.getAbsolutePath() + "/" + dataFile;
      BufferedReader reader = new BufferedReader(new FileReader(dataFilePath));
      String line;

      for (int i = 0; (line = reader.readLine()) != null; i++) {
        if (i % 1000 == 0) {
          System.out.println("Processed " + i + " reviews for prefix: " + prefix);
        }

        String[] mainComponents = line.split(" ");

        if (mainComponents.length < 2) {
          continue;
        }

        String[] labelComponents = mainComponents[0].split(":");

        if (labelComponents.length < 2) {
          continue;
        }

        String text = line.substring(mainComponents[0].length() + 1);
        String label = labelComponents[0];

        processDocument(text, label);
      }

      reader.close();
    }
  }

  protected boolean excludeTerminators() {
    return true;
  }
}
