package sofitda;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Map;

public class TextGenerationDataset {
	public TextGenerationDataset() throws IOException {
		String documentWordDatasetFilePath = String
		    .format(Configuration.STAGE3_DIRECTORY + "/" + Constants.DOCUMENT_WORD_DATASET_FILENAME_FORMAT, "training");
		String responseDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, "training");

		this.createWordTextDatasetFile(documentWordDatasetFilePath, responseDatasetFilePath);
	}

	private void createWordTextDatasetFile(String documentWordDatasetFilePath, String responseDatasetFilePath)
	    throws IOException {
		Map<Integer, String> vocabulary = Helper
		    .readSingleColumnStringMap(Configuration.STAGE3_DIRECTORY + "/" + Constants.VOCABULARY_FILENAME, 1);
		Map<Integer, int[]> wordDatasetMap = Helper.readMultiColumnIntegerMap(documentWordDatasetFilePath);
		Map<Integer, int[]> responseMap = Helper.readMultiColumnIntegerMap(responseDatasetFilePath);

		for (int responseId : responseMap.keySet()) {
			String textGenerationDatasetFilePath = String
			    .format(Configuration.STAGE3_DIRECTORY + "/" + Constants.TEXT_GENERATION_DATASET_FILENAME_FORMAT, responseId);

			System.out.println("Creating word text dataset text file " + textGenerationDatasetFilePath + "...");

			PrintWriter writer = Helper.createPrintWriter(textGenerationDatasetFilePath);

			for (int documentId : responseMap.get(responseId)) {
				writer.write("[BEGIN] ");

				for (int wordId : wordDatasetMap.get(documentId)) {
					String word = vocabulary.get(wordId);

					writer.write(word);
					writer.write(" ");
				}

				writer.write(" [END]\n");
			}

			writer.close();
		}
	}
}
