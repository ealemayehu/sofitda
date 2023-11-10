package sofitda;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class DataAugmentationTokenizer extends AbstractDocumentsTokenizer {
  public DataAugmentationTokenizer(String name) throws IOException {
    super(name, true /* hasResponse */);

    createDataset("training", false /* isLastPrefix */);
    createDataset("validation", false /* isLastPrefix */);
    createDataset("testing", true /* isLastPrefix */);
  }

  private void createDataset(String prefix, boolean isLastPrefix) throws NumberFormatException, IOException {
    initialize(prefix);

    String documentWordDatasetTextFilePath = this.rawDataDirectory.getAbsolutePath() + "/"
        + String.format(Constants.DOCUMENT_WORD_DATASET_TEXT_FILENAME_FORMAT, prefix);

    BufferedReader reader = new BufferedReader(new FileReader(documentWordDatasetTextFilePath));
    String line;

    while ((line = reader.readLine()) != null) {
      int firstSpaceIndex = line.indexOf(' ');
      int secondSpaceIndex = line.indexOf(' ', firstSpaceIndex + 1);
      int responseId = Integer.parseInt(line.substring(firstSpaceIndex + 1, secondSpaceIndex));
      String document = line.substring(secondSpaceIndex);

      processDocument(document, responseId);
    }

    if (prefix.contentEquals("training")) {
      for (int i = 0; i < this.responseIdMap.size(); i++) {
        augmentGenerated(i);
      }
    }

    reader.close();
    done(prefix, isLastPrefix);
  }

  private boolean augmentGenerated(int responseId) throws IOException {
    String generatedFilename = this.rawDataDirectory.getAbsolutePath() + "/"
        + String.format("generated_%d.txt", responseId);
    File file = new File(generatedFilename);

    if (!file.exists()) {
      return false;
    }

    System.out.println("Loading generated file " + generatedFilename);

    BufferedReader reader = new BufferedReader(new FileReader(generatedFilename));
    String line;

    while ((line = reader.readLine()) != null) {
      line = line.replace("[BEGIN] ", "").replace(" [END]", "");
      processDocument(line, responseId);
    }

    reader.close();
    return true;
  }

  protected List<String> tokenize(String document) {
    String[] tokens = document.split("[\\s]+");

    return Arrays.asList(tokens);
  }

  protected boolean excludeTerminators() {
    return true;
  }
}
