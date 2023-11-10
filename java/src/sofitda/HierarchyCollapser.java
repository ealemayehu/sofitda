package sofitda;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class HierarchyCollapser {
  public HierarchyCollapser() throws IOException {
    // collapseParagraph("all");
    collapseParagraphAndSentence("training");
    collapseParagraphAndSentence("validation");
    collapseParagraphAndSentence("testing");
  }

  private void collapseParagraphAndSentence(String prefix) throws IOException {
    String documentFilePath = String
        .format(Configuration.STAGE3_DIRECTORY + "/" + Constants.DOCUMENT_DATASET_FILENAME_FORMAT, prefix);
    String paragraphFilePath = String
        .format(Configuration.STAGE3_DIRECTORY + "/" + Constants.PARAGRAPH_DATASET_FILENAME_FORMAT, prefix);
    String sentenceFilePath = String
        .format(Configuration.STAGE3_DIRECTORY + "/" + Constants.SENTENCE_DATASET_FILENAME_FORMAT, prefix);
    String documentSentenceFilePath = String
        .format(Configuration.STAGE3_DIRECTORY + "/" + Constants.DOCUMENT_SENTENCE_DATASET_FILENAME_FORMAT, prefix);
    String documentWordFilePath = String
        .format(Configuration.STAGE3_DIRECTORY + "/" + Constants.DOCUMENT_WORD_DATASET_FILENAME_FORMAT, prefix);

    collapse(documentFilePath, paragraphFilePath, documentSentenceFilePath,
        AbstractDocumentsTokenizer.PARAGRAPH_TERMINATOR_SENTENCE_ID);
    collapse(documentSentenceFilePath, sentenceFilePath, documentWordFilePath, -1);
  }

  private void collapse(String grandParentFilePath, String parentFilePath, String collapsedFilePath,
      int parentTerminatorId) throws IOException {
    TreeMap<Integer, int[]> grandParentMap = Helper.readMultiColumnIntegerMap(grandParentFilePath);
    TreeMap<Integer, int[]> parentMap = Helper.readMultiColumnIntegerMap(parentFilePath);
    TreeMap<Integer, int[]> collapsedMap = collapseParent(grandParentMap, parentMap, parentTerminatorId);

    Helper.writeMultiColumnIntegerArrayMap(collapsedMap, collapsedFilePath);
  }

  private TreeMap<Integer, int[]> collapseParent(TreeMap<Integer, int[]> grandParentMap,
      TreeMap<Integer, int[]> parentMap, int parentTerminatorId) {
    TreeMap<Integer, int[]> collapsedMap = new TreeMap<Integer, int[]>();

    for (Map.Entry<Integer, int[]> grandParentEntry : grandParentMap.entrySet()) {
      int grandParentId = grandParentEntry.getKey();
      List<Integer> grandParentGrandChildIds = new ArrayList<>();

      for (int parentId : grandParentEntry.getValue()) {
        int[] parentChildIds = parentMap.get(parentId);

        for (int parentChildId : parentChildIds) {
          if (parentChildId != parentTerminatorId) {
            grandParentGrandChildIds.add(parentChildId);
          }
        }
      }

      int[] childIds = new int[grandParentGrandChildIds.size()];

      for (int i = 0; i < childIds.length; i++) {
        childIds[i] = grandParentGrandChildIds.get(i);
      }

      collapsedMap.put(grandParentId, childIds);
    }

    return collapsedMap;
  }
}
