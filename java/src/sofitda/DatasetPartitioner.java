package sofitda;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class DatasetPartitioner {
	private List<int[]> responseDataset;
	private List<int[]> documentDataset;
	private List<int[]> paragraphDataset;
	private List<int[]> sentenceDataset;

	private Map<Integer, List<Integer>> documentResponseMap = new HashMap<>();
	private Map<Integer, Integer> documentMap = new HashMap<>();
	private Map<Integer, Integer> paragraphMap = new HashMap<>();
	private Map<Integer, Integer> sentenceMap = new HashMap<>();

	private Comparator<int[]> rowComparator = new Comparator<int[]>() {
		public int compare(int[] row1, int[] row2) {
			return Integer.compare(row1[0], row2[0]);
		}
	};

	private final static int[] EXCLUDED_SENTENCE_IDS = new int[] {
	    AbstractDocumentsTokenizer.DOCUMENT_TERMINATOR_SENTENCE_ID,
	    AbstractDocumentsTokenizer.PARAGRAPH_TERMINATOR_SENTENCE_ID };

	private final static int[] EXCLUDED_PARAGRAPH_IDS = new int[] {
	    AbstractDocumentsTokenizer.DOCUMENT_TERMINATOR_PARAGRAPH_ID };

	public DatasetPartitioner() throws IOException {
		String responseDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, "all");
		String documentDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.DOCUMENT_DATASET_FILENAME_FORMAT, "all");
		String paragraphDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.PARAGRAPH_DATASET_FILENAME_FORMAT, "all");
		String sentenceDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.SENTENCE_DATASET_FILENAME_FORMAT, "all");

		responseDataset = Helper.readMultiColumnIntegerDataset(responseDatasetFilePath);
		documentDataset = Helper.readMultiColumnIntegerDataset(documentDatasetFilePath);
		paragraphDataset = Helper.readMultiColumnIntegerDataset(paragraphDatasetFilePath);
		sentenceDataset = Helper.readMultiColumnIntegerDataset(sentenceDatasetFilePath);

		populateDocumentResponseMap();
		Collections.shuffle(documentDataset);

		int trainingBeginIndex = 0;
		int trainingLength = (int) (documentDataset.size() * Constants.TRAINING_PARTITION_SIZE);
		int validationBeginIndex = trainingBeginIndex + trainingLength;
		int validationLength = (int) (documentDataset.size() * Constants.VALIDATION_PARTITION_SIZE);
		int testingBeginIndex = validationBeginIndex + validationLength;
		int testingLength = documentDataset.size() - trainingLength - validationLength;

		Map<Integer, List<Integer>> allResponseMap = new TreeMap<>();

		merge(allResponseMap, partition("training", trainingBeginIndex, trainingLength));
		merge(allResponseMap, partition("validation", validationBeginIndex, validationLength));
		merge(allResponseMap, partition("testing", testingBeginIndex, testingLength));

		Collections.sort(documentDataset, rowComparator);
		Collections.sort(paragraphDataset, rowComparator);
		Collections.sort(sentenceDataset, rowComparator);

		Helper.writeMultiColumnIntegerListMap(allResponseMap, responseDatasetFilePath);
		Helper.writeMultiColumnIntegerDataset(documentDatasetFilePath, documentDataset);
		Helper.writeMultiColumnIntegerDataset(paragraphDatasetFilePath, paragraphDataset);
		Helper.writeMultiColumnIntegerDataset(sentenceDatasetFilePath, sentenceDataset);
	}

	private void merge(Map<Integer, List<Integer>> allResponseMap, Map<Integer, List<Integer>> partitionResponseMap) {
		for (Map.Entry<Integer, List<Integer>> entry : partitionResponseMap.entrySet()) {
			List<Integer> partitionDocumentIds = partitionResponseMap.get(entry.getKey());
			List<Integer> allDocumentIds = partitionResponseMap.get(entry.getKey());

			if (allDocumentIds == null) {
				allDocumentIds = new ArrayList<>();
				allResponseMap.put(entry.getKey(), allDocumentIds);
			}

			allDocumentIds.addAll(partitionDocumentIds);
			Collections.sort(allDocumentIds);
		}

	}

	private void populateDocumentResponseMap() {
		for (int[] row : responseDataset) {
			int responseId = row[0];

			for (int i = 1; i < row.length; i++) {
				int documentId = row[i];

				List<Integer> responseIds = documentResponseMap.get(documentId);

				if (responseIds == null) {
					responseIds = new ArrayList<>();
					documentResponseMap.put(documentId, responseIds);
				}

				responseIds.add(responseId);
			}
		}
	}

	private Map<Integer, List<Integer>> partition(String prefix, int beginIndex, int length)
	    throws FileNotFoundException {
		System.out.println("Partitioning " + prefix + ": beginIndex = " + beginIndex + ", length = " + length + "...");

		List<Integer> oldDocumentIds = new ArrayList<>();
		List<int[]> documents = new ArrayList<>();
		List<int[]> paragraphs = new ArrayList<>();
		List<int[]> sentences = new ArrayList<>();

		int endIndex = beginIndex + length - 1;

		for (int paragraphId : EXCLUDED_PARAGRAPH_IDS) {
			paragraphs.add(paragraphDataset.get(paragraphId));
		}

		for (int sentenceId : EXCLUDED_SENTENCE_IDS) {
			sentences.add(sentenceDataset.get(sentenceId));
		}

		for (int i = beginIndex; i <= endIndex; i++) {
			int[] paragraphIds = documentDataset.get(i);
			int documentId = paragraphIds[0];

			for (int j = 1; j < paragraphIds.length; j++) {
				if (isExcluded(paragraphIds[j], EXCLUDED_PARAGRAPH_IDS)) {
					continue;
				}

				int[] sentenceIds = paragraphDataset.get(paragraphIds[j]);

				for (int k = 1; k < sentenceIds.length; k++) {
					if (isExcluded(sentenceIds[k], EXCLUDED_SENTENCE_IDS)) {
						continue;
					}

					int[] wordIds = sentenceDataset.get(sentenceIds[k]);

					sentences.add(wordIds);
				}

				paragraphs.add(sentenceIds);
			}

			documents.add(paragraphIds);
			oldDocumentIds.add(documentId);
		}

		updateIds(documents, documentMap, paragraphMap, null, EXCLUDED_PARAGRAPH_IDS);
		updateIds(paragraphs, paragraphMap, sentenceMap, EXCLUDED_PARAGRAPH_IDS, EXCLUDED_SENTENCE_IDS);
		updateIds(sentences, sentenceMap, null, EXCLUDED_SENTENCE_IDS, null);

		Map<Integer, List<Integer>> responses = getResponses(oldDocumentIds);

		String partitionResponseFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, prefix);
		String partitionDocumentFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.DOCUMENT_DATASET_FILENAME_FORMAT, prefix);
		String partitionParagraphFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.PARAGRAPH_DATASET_FILENAME_FORMAT, prefix);
		String partitionSentenceFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.SENTENCE_DATASET_FILENAME_FORMAT, prefix);

		Helper.writeMultiColumnIntegerListMap(responses, partitionResponseFilePath);
		Helper.writeMultiColumnIntegerDataset(partitionDocumentFilePath, documents);
		Helper.writeMultiColumnIntegerDataset(partitionParagraphFilePath, paragraphs);
		Helper.writeMultiColumnIntegerDataset(partitionSentenceFilePath, sentences);

		return responses;
	}

	private void updateIds(List<int[]> hierarchy, Map<Integer, Integer> parentMap, Map<Integer, Integer> childMap,
	    int[] excludedParentIds, int[] excludedChildIds) {
		for (int[] row : hierarchy) {
			if (!isExcluded(row[0], excludedParentIds)) {
				Integer newParentId = parentMap.get(row[0]);
				int begin = excludedParentIds != null ? excludedParentIds.length : 0;

				if (newParentId == null) {
					newParentId = parentMap.size() + begin;
					parentMap.put(row[0], newParentId);
				}

				row[0] = newParentId;
			}

			if (childMap == null) {
				continue;
			}

			for (int i = 1; i < row.length; i++) {
				if (!isExcluded(row[i], excludedChildIds)) {
					Integer newChildId = childMap.get(row[i]);
					int begin = excludedChildIds != null ? excludedChildIds.length : 0;

					if (newChildId == null) {
						newChildId = childMap.size() + begin;
						childMap.put(row[i], newChildId);
					}

					row[i] = newChildId;
				}
			}
		}

		Collections.sort(hierarchy, rowComparator);
	}

	private Map<Integer, List<Integer>> getResponses(List<Integer> oldDocumentIds) {
		Map<Integer, List<Integer>> responseMap = new TreeMap<>();

		if (documentResponseMap.size() > 0) {
			for (int oldDocumentId : oldDocumentIds) {
				int newDocumentId = documentMap.get(oldDocumentId);
				List<Integer> responseIds = documentResponseMap.get(oldDocumentId);

				if (responseIds == null) {
					continue;
				}

				for (int responseId : responseIds) {
					List<Integer> documentIds = responseMap.get(responseId);

					if (documentIds == null) {
						documentIds = new ArrayList<>();
						responseMap.put(responseId, documentIds);
					}

					documentIds.add(newDocumentId);
				}
			}
		}

		return responseMap;
	}

	private boolean isExcluded(int id, int[] excludedIds) {
		if (excludedIds == null) {
			return false;
		}
		for (int excludedId : excludedIds) {
			if (id == excludedId) {
				return true;
			}
		}

		return false;
	}
}
