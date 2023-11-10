package sofitda;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class MinorityVsGeneratedPartitioner {
  public MinorityVsGeneratedPartitioner() throws IOException {
    String responseDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
        + String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, "all");

    Map<Integer, int[]> responseMap = Helper.readMultiColumnIntegerMap(responseDatasetFilePath);
    int[] generatedDocuments = responseMap.get(0);
    int[] actualDocuments = responseMap.get(1);

    int generatedClassSize = generatedDocuments.length;
    int actualClassSize = actualDocuments.length;
    int generatedChunkCount = generatedClassSize / actualClassSize;

    Set<Integer> actualDocumentIds = new HashSet<>();

    for (int documentId : actualDocuments) {
      actualDocumentIds.add(documentId);
    }

    System.out.println("Generated chunk count: " + generatedChunkCount);
    System.out.println("Actual class size: " + actualClassSize);

    for (int i = 0; i < generatedChunkCount; i++) {
      List<Integer> trainingIds = new ArrayList<>();
      List<Integer> testingIds = new ArrayList<>();
      int beginChunkIndex = i * actualClassSize;
      int endChunkIndex = beginChunkIndex + actualClassSize;

      trainingIds.addAll(actualDocumentIds);

      for (int j = 0; j < generatedDocuments.length; j++) {
        if (beginChunkIndex <= j && j < endChunkIndex) {
          trainingIds.add(generatedDocuments[j]);
        } else {
          testingIds.add(generatedDocuments[j]);
        }
      }

      String trainingIdFilePath = Configuration.STAGE3_DIRECTORY + "/"
          + String.format(Constants.ROOT_ID_CHUNK_FILENAME_FORMAT, "training", i);
      String testingIdFilePath = Configuration.STAGE3_DIRECTORY + "/"
          + String.format(Constants.ROOT_ID_CHUNK_FILENAME_FORMAT, "testing", i);

      System.out.println("Creating minority dataset chunk (" + beginChunkIndex + ", " + endChunkIndex + ")");

      Helper.writeSingleColumnDataset(trainingIdFilePath, trainingIds);
      Helper.writeSingleColumnDataset(testingIdFilePath, testingIds);
    }
  }
}
