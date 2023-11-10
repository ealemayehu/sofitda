package sofitda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

public class ExtraDatasetGenerator {
  public ExtraDatasetGenerator() throws IOException {
    Map<Integer, String> vocabulary = Helper.readSingleColumnStringMap(
        Configuration.STAGE3_DIRECTORY + "/" + Constants.VOCABULARY_FILENAME, 1);
    String[] prefixes = { "all", "training", "validation", "testing" };

    for (String prefix : prefixes) {
      String sentenceDatasetFilePath = String.format(Configuration.STAGE3_DIRECTORY + "/"
          + Constants.SENTENCE_DATASET_FILENAME_FORMAT, prefix);
      String sentenceDatasetTextFilePath = String.format(Configuration.STAGE3_DIRECTORY + "/"
          + Constants.SENTENCE_DATASET_TEXT_FILENAME_FORMAT, prefix);
      String wordDatasetFilePath = String.format(Configuration.STAGE3_DIRECTORY + "/"
          + Constants.WORD_DATASET_FILENAME_FORMAT, prefix);
      String documentWordDatasetFilePath = String.format(Configuration.STAGE3_DIRECTORY + "/"
          + Constants.DOCUMENT_WORD_DATASET_FILENAME_FORMAT, prefix);
      String documentWordDatasetTextFilePath = String.format(Configuration.STAGE3_DIRECTORY + "/"
          + Constants.DOCUMENT_WORD_DATASET_TEXT_FILENAME_FORMAT, prefix);
  		String responseDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
  		    + String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, prefix);
      

      createSentenceTextDatasetFile(sentenceDatasetFilePath, vocabulary,
          sentenceDatasetTextFilePath);
      createDocumentWordTextDatasetFile(documentWordDatasetFilePath, responseDatasetFilePath, vocabulary,
      		documentWordDatasetTextFilePath);
      createWordDatasetFile(sentenceDatasetFilePath, wordDatasetFilePath);
    }
  }

  private void createSentenceTextDatasetFile(String inputFilePath, Map<Integer, String> vocabulary,
      String outputFilePath) throws IOException {
    System.out.println("Creating sentence dataset text file " + outputFilePath + "...");

    Map<Integer, int[]> sentenceMap = Helper.readMultiColumnIntegerMap(inputFilePath);
    PrintWriter writer = new PrintWriter(outputFilePath);

    for (int sentenceId : sentenceMap.keySet()) {
    	writer.write(sentenceId + " ");
    	
      for (int wordId : sentenceMap.get(sentenceId)) {
        writer.write(vocabulary.get(wordId));
        writer.write(" ");
      }

      writer.write("\n");
    }

    writer.close();
  }
  
  private void createDocumentWordTextDatasetFile(String documentWordDatasetFilePath, String responseDatasetFilePath, Map<Integer, String> vocabulary,
      String outputFilePath) throws IOException {
    System.out.println("Creating document word dataset text file " + outputFilePath + "...");

		Map<Integer, int[]> documentWordDatasetMap = Helper.readMultiColumnIntegerMap(documentWordDatasetFilePath);
		Map<Integer, int[]> responseMap = Helper.readMultiColumnIntegerMap(responseDatasetFilePath);
		Map<Integer, Integer> documentResponseMap = new HashMap<>();
		
		for (int responseId: responseMap.keySet()) {
			for (int documentId: responseMap.get(responseId)) {
				documentResponseMap.put(documentId, responseId);
			}
		}
		
		PrintWriter writer = new PrintWriter(outputFilePath);

		for (int documentId : documentWordDatasetMap.keySet()) {
			writer.write(documentId + " " + documentResponseMap.get(documentId) + " ");
			
			for (int wordId : documentWordDatasetMap.get(documentId)) {
				String word = vocabulary.get(wordId);
				
				writer.write(word);
				writer.write(" ");
			}
			
			writer.write("\n");
		}

		writer.close();
	}

  private void createWordDatasetFile(String inputFilePath, String outputFilePath)
      throws IOException {
    System.out.println("Creating word dataset file " + outputFilePath + "...");

    BufferedReader reader = new BufferedReader(new FileReader(inputFilePath));
    String line;
    Map<Integer, Set<Integer>> wordMap = new HashMap<>();

    while ((line = reader.readLine()) != null) {
      String[] idStrings = line.split("\\s+");
      int sentenceId = Integer.parseInt(idStrings[0]);

      for (int i = 1; i < idStrings.length; i++) {
        String wordIdString = idStrings[i];

        if (wordIdString != null && !wordIdString.isEmpty()) {
          int wordId = Integer.parseInt(wordIdString);
          Set<Integer> sentenceIds = wordMap.get(wordId);

          if (sentenceIds == null) {
            sentenceIds = new HashSet<>();
            wordMap.put(wordId, sentenceIds);
          }

          sentenceIds.add(sentenceId);
        }
      }
    }

    reader.close();

    SortedMap<Integer, Set<Integer>> sortedWordMap = new TreeMap<>(wordMap);
    PrintWriter writer = new PrintWriter(outputFilePath);

    for (Map.Entry<Integer, Set<Integer>> entry : sortedWordMap.entrySet()) {
      writer.write(String.valueOf(entry.getKey()));

      for (Integer sentenceId : entry.getValue()) {
        writer.print(" ");
        writer.print(String.valueOf(sentenceId));
      }

      writer.print("\n");
    }

    writer.close();
  }
}
