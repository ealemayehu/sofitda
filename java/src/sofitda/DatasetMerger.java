package sofitda;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class DatasetMerger {
	private Map<Integer, int[]> responseDataset = new TreeMap<>();;
	private List<int[]> documentDataset = new ArrayList<>();
	private List<int[]> documentSentenceDataset = new ArrayList<>();
	private List<int[]> documentWordDataset = new ArrayList<>();
	private List<int[]> paragraphDataset = new ArrayList<>();
	private List<int[]> sentenceDataset = new ArrayList<>();

	public DatasetMerger() throws IOException {
		String[] prefixes = { "training", "validation", "testing" };

		for (String prefix : prefixes) {
			String responseDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
			    + String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, prefix);
			String documentDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
			    + String.format(Constants.DOCUMENT_DATASET_FILENAME_FORMAT, prefix);
			String documentSentenceDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
			    + String.format(Constants.DOCUMENT_SENTENCE_DATASET_FILENAME_FORMAT, prefix);
			String documentWordDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
			    + String.format(Constants.DOCUMENT_WORD_DATASET_FILENAME_FORMAT, prefix);
			String paragraphDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
			    + String.format(Constants.PARAGRAPH_DATASET_FILENAME_FORMAT, prefix);
			String sentenceDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
			    + String.format(Constants.SENTENCE_DATASET_FILENAME_FORMAT, prefix);

			addResponseDataset(Helper.readMultiColumnIntegerMap(responseDatasetFilePath));
			paragraphDataset.addAll(Helper.readMultiColumnIntegerDataset(paragraphDatasetFilePath));
			documentSentenceDataset.addAll(Helper.readMultiColumnIntegerDataset(documentSentenceDatasetFilePath));
			documentWordDataset.addAll(Helper.readMultiColumnIntegerDataset(documentWordDatasetFilePath));
			sentenceDataset.addAll(Helper.readMultiColumnIntegerDataset(sentenceDatasetFilePath));

			List<int[]> prefixDocumentDataset = Helper.readMultiColumnIntegerDataset(documentDatasetFilePath);

			documentDataset.addAll(prefixDocumentDataset);
			writeRootIds(prefixDocumentDataset, prefix);
		}

		String responseDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, "all");
		String documentDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.DOCUMENT_DATASET_FILENAME_FORMAT, "all");
		String documentSentenceDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.DOCUMENT_SENTENCE_DATASET_FILENAME_FORMAT, "all");
		String documentWordDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.DOCUMENT_WORD_DATASET_FILENAME_FORMAT, "all");
		String paragraphDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.PARAGRAPH_DATASET_FILENAME_FORMAT, "all");
		String sentenceDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.SENTENCE_DATASET_FILENAME_FORMAT, "all");

		Helper.writeMultiColumnIntegerArrayMap(responseDataset, responseDatasetFilePath);
		Helper.writeMultiColumnIntegerDataset(documentDatasetFilePath, documentDataset);
		Helper.writeMultiColumnIntegerDataset(documentSentenceDatasetFilePath, documentSentenceDataset);
		Helper.writeMultiColumnIntegerDataset(documentWordDatasetFilePath, documentWordDataset);
		Helper.writeMultiColumnIntegerDataset(paragraphDatasetFilePath, paragraphDataset);
		Helper.writeMultiColumnIntegerDataset(sentenceDatasetFilePath, sentenceDataset);
	}

	private void addResponseDataset(TreeMap<Integer, int[]> prefixResponseDataset) {
		for (Map.Entry<Integer, int[]> entry : prefixResponseDataset.entrySet()) {
			int[] rootIds = responseDataset.get(entry.getKey());

			if (rootIds == null) {
				responseDataset.put(entry.getKey(), entry.getValue());
			} else {
				int[] mergedRootIds = new int[rootIds.length + entry.getValue().length];

				for (int i = 0; i < rootIds.length; i++) {
					mergedRootIds[i] = rootIds[i];
				}

				for (int i = rootIds.length, j = 0; j < entry.getValue().length; i++, j++) {
					mergedRootIds[i] = entry.getValue()[j];
				}

				responseDataset.put(entry.getKey(), mergedRootIds);
			}
		}
	}

	private void writeRootIds(List<int[]> prefixDocumentDataset, String prefix) throws FileNotFoundException {
		List<Integer> rootIds = new ArrayList<>();

		for (int i = 0; i < prefixDocumentDataset.size(); i++) {
			rootIds.add(prefixDocumentDataset.get(i)[0]);
		}

		String datasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.ROOT_ID_FILENAME_FORMAT, prefix);
		Helper.writeSingleColumnDataset(datasetFilePath, rootIds);
	}
}
