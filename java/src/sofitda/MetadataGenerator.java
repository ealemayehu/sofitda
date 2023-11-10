package sofitda;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;

public class MetadataGenerator {
  public MetadataGenerator(String type) throws IOException {
    String[] prefixes = { "training", "validation", "testing", "all" };

    for (String prefix : prefixes) {
      if (!prefix.equals("training") && isDatasetEmpty(prefix)) {
        copyMetadata("training", prefix);
        continue;
      }

      switch (type) {
      case "rdps":
        generateRdps(prefix);
        break;

      case "dps":
        generateDps(prefix);
        break;

      case "rds":
        generateRds(prefix);
        break;

      case "rdw":
        generateRdw(prefix);
        break;

      case "ds":
        generateDs(prefix);
        break;

      case "dw":
        generateDw(prefix);
        break;
      }
    }
  }

  private boolean isDatasetEmpty(String prefix) throws IOException {
    String documentDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.DOCUMENT_DATASET_FILENAME_FORMAT, prefix);

    return Helper.getMaxId(documentDatasetFilePath) == -1;
  }

  private void copyMetadata(String fromPrefix, String toPrefix) throws IOException {
    String fromMetadataDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.METADATA_DATASET_FILE_NAME_FORMAT, fromPrefix);
    String toMetadataDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.METADATA_DATASET_FILE_NAME_FORMAT, toPrefix);

    FileUtils.copyFile(new File(fromMetadataDatasetFilePath), new File(toMetadataDatasetFilePath));
  }

  private void generateRdps(String prefix) throws IOException {
    String metadataDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.METADATA_DATASET_FILE_NAME_FORMAT, prefix);
    String responseDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, prefix);
    String documentDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.DOCUMENT_DATASET_FILENAME_FORMAT, prefix);
    String paragraphDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.PARAGRAPH_DATASET_FILENAME_FORMAT, prefix);
    String sentenceDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.SENTENCE_DATASET_FILENAME_FORMAT, prefix);
    String vocabularyFilePath = Configuration.STAGE3_DIRECTORY + "/" + Constants.VOCABULARY_FILENAME;

    List<int[]> metadataDataset = new ArrayList<int[]>();
    int[] limits = new int[8];
    int index = 0;

    metadataDataset.add(limits);

    limits[index++] = Helper.getMaxId(responseDatasetFilePath) + 1;
    limits[index++] = Helper.getMaxId(documentDatasetFilePath) + 1;
    limits[index++] = Helper.getMaxColumnCount(documentDatasetFilePath) - 1;
    limits[index++] = Helper.getMaxId(paragraphDatasetFilePath) + 1;
    limits[index++] = Helper.getMaxColumnCount(paragraphDatasetFilePath) - 1;
    limits[index++] = Helper.getMaxId(sentenceDatasetFilePath) + 1;
    limits[index++] = Helper.getMaxColumnCount(sentenceDatasetFilePath) - 1;
    limits[index++] = Helper.getMaxId(vocabularyFilePath, 1, 1) + 1;
    Helper.writeMultiColumnIntegerDataset(metadataDatasetFilePath, metadataDataset);
  }

  private void generateDps(String prefix) throws IOException {
    String metadataDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.METADATA_DATASET_FILE_NAME_FORMAT, prefix);
    String documentDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.DOCUMENT_DATASET_FILENAME_FORMAT, prefix);
    String paragraphDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.PARAGRAPH_DATASET_FILENAME_FORMAT, prefix);
    String sentenceDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.SENTENCE_DATASET_FILENAME_FORMAT, prefix);
    String vocabularyFilePath = Configuration.STAGE3_DIRECTORY + "/" + Constants.VOCABULARY_FILENAME;

    List<int[]> metadataDataset = new ArrayList<int[]>();
    int[] limits = new int[7];
    int index = 0;

    metadataDataset.add(limits);

    limits[index++] = Helper.getMaxId(documentDatasetFilePath) + 1;
    limits[index++] = Helper.getMaxColumnCount(documentDatasetFilePath) - 1;
    limits[index++] = Helper.getMaxId(paragraphDatasetFilePath) + 1;
    limits[index++] = Helper.getMaxColumnCount(paragraphDatasetFilePath) - 1;
    limits[index++] = Helper.getMaxId(sentenceDatasetFilePath) + 1;
    limits[index++] = Helper.getMaxColumnCount(sentenceDatasetFilePath) - 1;
    limits[index++] = Helper.getMaxId(vocabularyFilePath, 1, 1) + 1;
    Helper.writeMultiColumnIntegerDataset(metadataDatasetFilePath, metadataDataset);
  }

  private void generateRds(String prefix) throws IOException {
    String metadataDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.METADATA_DATASET_FILE_NAME_FORMAT, prefix);
    String responseDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, prefix);
    String documentSentenceDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.DOCUMENT_SENTENCE_DATASET_FILENAME_FORMAT, prefix);
    String sentenceDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.SENTENCE_DATASET_FILENAME_FORMAT, prefix);
    String vocabularyFilePath = Configuration.STAGE3_DIRECTORY + "/" + Constants.VOCABULARY_FILENAME;

    List<int[]> metadataDataset = new ArrayList<int[]>();
    int[] limits = new int[6];
    int index = 0;

    metadataDataset.add(limits);

    limits[index++] = Helper.getMaxId(responseDatasetFilePath) + 1;
    limits[index++] = Helper.getMaxId(documentSentenceDatasetFilePath) + 1;
    limits[index++] = Helper.getMaxColumnCount(documentSentenceDatasetFilePath) - 1;
    limits[index++] = Helper.getMaxId(sentenceDatasetFilePath) + 1;
    limits[index++] = Helper.getMaxColumnCount(sentenceDatasetFilePath) - 1;
    limits[index++] = Helper.getMaxId(vocabularyFilePath, 1, 1) + 1;
    Helper.writeMultiColumnIntegerDataset(metadataDatasetFilePath, metadataDataset);
  }

  private void generateDs(String prefix) throws IOException {
    String metadataDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.METADATA_DATASET_FILE_NAME_FORMAT, prefix);
    String documentSentenceDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.DOCUMENT_SENTENCE_DATASET_FILENAME_FORMAT, prefix);
    String sentenceDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.SENTENCE_DATASET_FILENAME_FORMAT, prefix);
    String vocabularyFilePath = Configuration.STAGE3_DIRECTORY + "/" + Constants.VOCABULARY_FILENAME;

    List<int[]> metadataDataset = new ArrayList<int[]>();
    int[] limits = new int[5];
    int index = 0;

    metadataDataset.add(limits);
    limits[index++] = Helper.getMaxId(documentSentenceDatasetFilePath) + 1;
    limits[index++] = Helper.getMaxColumnCount(documentSentenceDatasetFilePath) - 1;
    limits[index++] = Helper.getMaxId(sentenceDatasetFilePath) + 1;
    limits[index++] = Helper.getMaxColumnCount(sentenceDatasetFilePath) - 1;
    limits[index++] = Helper.getMaxId(vocabularyFilePath, 1, 1) + 1;

    Helper.writeMultiColumnIntegerDataset(metadataDatasetFilePath, metadataDataset);
  }

  private void generateRdw(String prefix) throws IOException {
    String metadataDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.METADATA_DATASET_FILE_NAME_FORMAT, prefix);
    String responseDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, prefix);
    String documentWordDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.DOCUMENT_WORD_DATASET_FILENAME_FORMAT, prefix);
    String vocabularyFilePath = Configuration.STAGE3_DIRECTORY + "/" + Constants.VOCABULARY_FILENAME;

    List<int[]> metadataDataset = new ArrayList<int[]>();
    int[] limits = new int[4];
    int index = 0;

    metadataDataset.add(limits);
    limits[index++] = Helper.getMaxId(responseDatasetFilePath) + 1;
    limits[index++] = Helper.getMaxId(documentWordDatasetFilePath) + 1;
    limits[index++] = Helper.getMaxColumnCount(documentWordDatasetFilePath) - 1;
    limits[index++] = Helper.getMaxId(vocabularyFilePath, 1, 1) + 1;

    Helper.writeMultiColumnIntegerDataset(metadataDatasetFilePath, metadataDataset);
  }

  private void generateDw(String prefix) throws IOException {
    String metadataDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.METADATA_DATASET_FILE_NAME_FORMAT, prefix);
    String documentWordDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.DOCUMENT_WORD_DATASET_FILENAME_FORMAT, prefix);
    String vocabularyFilePath = Configuration.STAGE3_DIRECTORY + "/" + Constants.VOCABULARY_FILENAME;

    List<int[]> metadataDataset = new ArrayList<int[]>();
    int[] limits = new int[3];
    int index = 0;

    metadataDataset.add(limits);
    limits[index++] = Helper.getMaxId(documentWordDatasetFilePath) + 1;
    limits[index++] = Helper.getMaxColumnCount(documentWordDatasetFilePath) - 1;
    limits[index++] = Helper.getMaxId(vocabularyFilePath, 1, 1) + 1;

    Helper.writeMultiColumnIntegerDataset(metadataDatasetFilePath, metadataDataset);
  }
}
