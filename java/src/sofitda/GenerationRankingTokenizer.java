package sofitda;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class GenerationRankingTokenizer extends AbstractDocumentsTokenizer {

  public GenerationRankingTokenizer(String name) throws IOException {
    super(name, true /* hasResponse */);

    createRealDataset("training", false /* isLastPrefix */);
    createRealDataset("validation", false /* isLastPrefix */);
    // createMixedDataset("training", responseId, false);
    createGeneratedDataSet("testing", true /* isLastPrefix */);
  }

  private void createRealDataset(String prefix, boolean isLastPrefix) throws NumberFormatException, IOException {
    System.out.println("Creating real dataset prefix: " + prefix);

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

    reader.close();
    done(prefix, isLastPrefix);
  }

  private void createGeneratedDataSet(String prefix, boolean isLastPrefix) throws IOException {
    System.out.println("Creating generated dataset prefix: " + prefix);

    initialize(prefix);

    String responseDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, "training");

    Map<Integer, int[]> responseMap = Helper.readMultiColumnIntegerMap(responseDatasetFilePath);

    for (Integer responseId : responseMap.keySet()) {
      String generatedFilename = this.rawDataDirectory.getAbsolutePath() + "/"
          + String.format("generated_%d.txt", responseId);
      File generatedFile = new File(generatedFilename);

      if (!generatedFile.exists()) {
        System.out.println("Generate file: " + generatedFilename + " does not exist");
        continue;
      }

      System.out.println("Merging generated file: " + generatedFilename);

      BufferedReader reader = new BufferedReader(new FileReader(generatedFilename));
      String line;
      int count = 0;

      while ((line = reader.readLine()) != null) {
        line = line.replace("[BEGIN] ", "").replace(" [END]", "");
        processDocument(line, responseId);
        count++;
      }

      System.out.println("Merged " + count + " generated documents into " + prefix + " dataset.");
      reader.close();
    }

    done(prefix, isLastPrefix);
  }

  protected List<String> tokenize(String document) {
    String[] tokens = document.split("[\\s]+");

    return Arrays.asList(tokens);
  }

  protected boolean excludeTerminators() {
    return true;
  }
}
