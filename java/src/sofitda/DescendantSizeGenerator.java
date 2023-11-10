package sofitda;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;
import java.util.TreeMap;

public class DescendantSizeGenerator {
  public DescendantSizeGenerator() throws IOException {
    String[] prefixes = { "training", "validation", "testing" };

    for (String prefix : prefixes) {
      createDesendantSizeDatasets(prefix);
    }
  }

  @SuppressWarnings("unchecked")
  private void createDesendantSizeDatasets(String prefix) throws IOException {
    String documentSentenceDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.DOCUMENT_SENTENCE_DATASET_FILENAME_FORMAT, prefix);
    String sentenceDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.SENTENCE_DATASET_FILENAME_FORMAT, prefix);
    String documentSentenceDescendantSizeFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.DOCUMENT_SENTENCE_DESCENDANT_SIZE_FILENAME_FORMAT, prefix);
    String sentenceDescendantSizeFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.SENTENCE_DESCENDANT_SIZE_FILENAME_FORMAT, prefix);
    
    createDescendantSizeDatasets(
        new String[] {sentenceDescendantSizeFilePath, documentSentenceDescendantSizeFilePath},
        new Map[] {
            Helper.readMultiColumnIntegerMap(sentenceDatasetFilePath),
            Helper.readMultiColumnIntegerMap(documentSentenceDatasetFilePath)});
  }

  private void createDescendantSizeDatasets(String[] outputFilePaths,
      Map<Integer, int[]>[] hierarchyMaps) throws FileNotFoundException {
    @SuppressWarnings("unchecked")
    Map<Integer, Integer>[] descendantSizeMap = new Map[hierarchyMaps.length];
    
    for (int i = 0; i < hierarchyMaps.length; i++) {
      descendantSizeMap[i] = new TreeMap<>();
    }
    
    for (Map.Entry<Integer, int[]> entry: hierarchyMaps[0].entrySet()) {
      descendantSizeMap[0].put(entry.getKey(), entry.getValue().length);
    }
    
    Helper.writeIntegralDictionary(descendantSizeMap[0], outputFilePaths[0]);
    
    for (int i = 1; i < hierarchyMaps.length; i++) {      
      for (Map.Entry<Integer, int[]> entry: hierarchyMaps[i].entrySet()) {
        int totalSize = 0;
        
        for (int id: entry.getValue()) {
          totalSize += descendantSizeMap[i-1].get(id);
        }
        
        descendantSizeMap[i].put(entry.getKey(), totalSize);
      }
      
      Helper.writeIntegralDictionary(descendantSizeMap[i], outputFilePaths[i]);
    }
  }
}
