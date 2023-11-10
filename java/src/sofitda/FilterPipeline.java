package sofitda;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

public class FilterPipeline {
	private static int MIN_DOCUMENT_WORD_COUNT = 1; // Includes document terminator
	private static int MAX_DOCUMENT_WORD_COUNT = 50; // Includes document terminator

	public static int[] EXEMPT_WORD_IDS = { AbstractDocumentsTokenizer.SENTENCE_TERMINATOR_WORD_ID,
	    AbstractDocumentsTokenizer.PARAGRAPH_TERMINATOR_WORD_ID, AbstractDocumentsTokenizer.DOCUMENT_TERMINATOR_WORD_ID,
	    AbstractDocumentsTokenizer.UNKNOWN_WORD_ID };

	private FilterData filterData = new FilterData();

	public FilterPipeline() throws IOException {
		filterData.trainingData = loadPartitionData("training");
		filterData.validationData = loadPartitionData("validation");
		filterData.testingData = loadPartitionData("testing");
		filterData.vocabulary = getVocabulary();

		System.out.println("Vocabulary Size: " + filterData.vocabulary.size());

		filterData.responseIdMap = getResponseIdMap();


		filterRemoveShortAndLongDocuments();


		savePartitionData(filterData.trainingData, "training");
		savePartitionData(filterData.validationData, "validation");
		savePartitionData(filterData.testingData, "testing");
		saveVocabulary(filterData.vocabulary);

		printAverageProportionOfUnknownWords("Training", filterData.trainingData);
		printAverageProportionOfUnknownWords("Testing", filterData.testingData);
		System.out.println("Final vocabularySize: " + filterData.vocabulary.size());
	}


	private void filterRemoveShortAndLongDocuments() {
		System.out.println("Filter: remove short and long documents.");

		PartitionData[] partitionDatas = { filterData.trainingData, filterData.validationData, filterData.testingData };

		for (PartitionData partitionData : partitionDatas) {
			Iterator<Integer> iterator = partitionData.documentMap.keySet().iterator();

			while (iterator.hasNext()) {
				int documentId = iterator.next();
				int wordCount = getWordCount(partitionData, documentId);

				if (wordCount < MIN_DOCUMENT_WORD_COUNT || MAX_DOCUMENT_WORD_COUNT < wordCount) {
					iterator.remove();
				}
			}
		}

		removeDanglingReferences();
		compactIds();
	}


	private void printAverageProportionOfUnknownWords(String partitionName, PartitionData partitionData) {
		float total = 0;

		for (Integer documentId : partitionData.documentMap.keySet()) {
			total += getProportionOfUnknownWords(partitionData, documentId);
		}

		System.out
		    .println(partitionName + ": average proportion of unknown words: " + total / partitionData.documentMap.size());
	}

	private double getProportionOfUnknownWords(PartitionData partitionData, int documentId) {
		int[] paragraphIds = partitionData.documentMap.get(documentId);
		int knownWordCount = 0;
		int unknownWordCount = 0;

		for (int paragraphId : paragraphIds) {
			int[] sentenceIds = partitionData.paragraphMap.get(paragraphId);

			for (int sentenceId : sentenceIds) {
				int[] wordIds = partitionData.sentenceMap.get(sentenceId);

				for (int wordId : wordIds) {
					if (wordId == AbstractDocumentsTokenizer.UNKNOWN_WORD_ID) {
						unknownWordCount++;
					} else {
						knownWordCount++;
					}
				}
			}
		}

		return ((double) unknownWordCount) / (unknownWordCount + knownWordCount + 1e-7);
	}


	private PartitionData loadPartitionData(String prefix) throws IOException {
		String responseDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, prefix);
		String documentDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.DOCUMENT_DATASET_FILENAME_FORMAT, prefix);
		String paragraphDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.PARAGRAPH_DATASET_FILENAME_FORMAT, prefix);
		String sentenceDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.SENTENCE_DATASET_FILENAME_FORMAT, prefix);
		PartitionData partitionData = new PartitionData();

		partitionData.responseMap = loadDataset(responseDatasetFilePath);
		partitionData.documentMap = loadDataset(documentDatasetFilePath);
		partitionData.paragraphMap = loadDataset(paragraphDatasetFilePath);
		partitionData.sentenceMap = loadDataset(sentenceDatasetFilePath);
		return partitionData;
	}

	private TreeMap<Integer, int[]> loadDataset(String datasetFilePath) throws IOException {
		File datasetFile = new File(datasetFilePath);

		if (datasetFile.exists()) {
			TreeMap<Integer, int[]> datasetMap = Helper.readMultiColumnIntegerMap(datasetFilePath);

			System.out.println("Loaded dataset: " + datasetFilePath);
			return datasetMap;
		} else {
			System.out.println("Did not find " + datasetFilePath + ". Skipping...");
			return new TreeMap<>();
		}
	}

	private void savePartitionData(PartitionData partitionData, String prefix) throws FileNotFoundException {
		String responseDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, prefix);
		String documentDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.DOCUMENT_DATASET_FILENAME_FORMAT, prefix);
		String paragraphDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.PARAGRAPH_DATASET_FILENAME_FORMAT, prefix);
		String sentenceDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
		    + String.format(Constants.SENTENCE_DATASET_FILENAME_FORMAT, prefix);

		Helper.writeMultiColumnIntegerArrayMap(partitionData.responseMap, responseDatasetFilePath);
		Helper.writeMultiColumnIntegerArrayMap(partitionData.documentMap, documentDatasetFilePath);
		Helper.writeMultiColumnIntegerArrayMap(partitionData.paragraphMap, paragraphDatasetFilePath);
		Helper.writeMultiColumnIntegerArrayMap(partitionData.sentenceMap, sentenceDatasetFilePath);
	}

	private Map<Integer, String> getVocabulary() throws IOException {
		String vocabularyFilePath = Configuration.STAGE3_DIRECTORY + "/" + Constants.VOCABULARY_FILENAME;
		return Helper.readSingleColumnStringMap(vocabularyFilePath, 1);
	}

	private Map<Integer, String> getResponseIdMap() throws IOException {
		String responseFilePath = Configuration.STAGE3_DIRECTORY + "/" + Constants.RESPONSE_DATASET_TEXT_FILENAME;
		return Helper.readSingleColumnStringMap(responseFilePath, 0);
	}

	private void saveVocabulary(Map<Integer, String> vocabulary) throws IOException {
		String vocabularyFilePath = Configuration.STAGE3_DIRECTORY + "/" + Constants.VOCABULARY_FILENAME;
		Helper.writeSingleColumnStringMap(vocabularyFilePath, "Word\tID", vocabulary);
	}


	private void compactIds() {
		compactWordIds();

		PartitionData[] partitionDatas = new PartitionData[] { filterData.trainingData, filterData.validationData,
		    filterData.testingData };

		int minSentenceId = 0;
		int minParagraphId = 0;

		for (PartitionData partitionData : partitionDatas) {
			minSentenceId = compactChildIds(minSentenceId, partitionData.paragraphMap, partitionData.sentenceMap);
			minParagraphId = compactChildIds(minParagraphId, partitionData.documentMap, partitionData.paragraphMap);
		}

		compactDocumentIds();
		compactResponseIds();
	}

	private int compactChildIds(int beginChildId, TreeMap<Integer, int[]> parentDataset,
	    TreeMap<Integer, int[]> childDataset) {
		Set<Integer> foundChildIds = new TreeSet<>();

		for (int[] childIds : parentDataset.values()) {
			for (int childId : childIds) {
				foundChildIds.add(childId);
			}
		}

		int previousFoundChildId = beginChildId;
		Map<Integer, Integer> compactIdMap = new HashMap<>();

		for (int foundChildId : childDataset.keySet()) {
			int compactChildId;

			if (foundChildId - previousFoundChildId > 1) {
				compactChildId = previousFoundChildId + 1;
			} else {
				compactChildId = foundChildId;
			}

			previousFoundChildId = beginChildId > compactChildId ? beginChildId : compactChildId;
			compactIdMap.put(foundChildId, compactChildId);
		}

		Map<Integer, int[]> newChildDataset = new TreeMap<>();

		for (Map.Entry<Integer, Integer> entry : compactIdMap.entrySet()) {
			int[] grandChildIds = childDataset.get(entry.getKey());

			newChildDataset.put(entry.getValue(), grandChildIds);
		}

		childDataset.clear();
		childDataset.putAll(newChildDataset);

		for (int[] childIds : parentDataset.values()) {
			for (int i = 0; i < childIds.length; i++) {
				if (!compactIdMap.containsKey(childIds[i])) {
					System.err.println("Missing " + childIds[i]);
				}

				childIds[i] = compactIdMap.get(childIds[i]);
			}
		}

		return previousFoundChildId;
	}

	private void compactDocumentIds() {
		PartitionData[] partitionDatas = new PartitionData[] { filterData.trainingData, filterData.validationData,
		    filterData.testingData };
		int previousDocumentId = 0;

		for (PartitionData partitionData : partitionDatas) {
			Map<Integer, Integer> compactIdMap = new HashMap<>();

			for (int documentId : partitionData.documentMap.keySet()) {
				int compactId;

				if (documentId - previousDocumentId > 1) {
					compactId = previousDocumentId + 1;
				} else {
					compactId = documentId;
				}

				previousDocumentId = compactId;
				compactIdMap.put(documentId, compactId);
			}

			Map<Integer, int[]> newDocumentMap = new TreeMap<>();

			for (Map.Entry<Integer, Integer> entry : compactIdMap.entrySet()) {
				int[] paragraphIds = partitionData.documentMap.get(entry.getKey());

				newDocumentMap.put(entry.getValue(), paragraphIds);
			}

			partitionData.documentMap.clear();
			partitionData.documentMap.putAll(newDocumentMap);

			for (int[] documentIds : partitionData.responseMap.values()) {
				for (int i = 0; i < documentIds.length; i++) {
					documentIds[i] = compactIdMap.get(documentIds[i]);
				}
			}
		}
	}

	private void compactResponseIds() {
		Set<Integer> foundResponseIds = new TreeSet<>();
		PartitionData[] partitionDatas = new PartitionData[] { filterData.trainingData, filterData.validationData,
		    filterData.testingData };

		for (PartitionData partitionData : partitionDatas) {
			for (Map.Entry<Integer, int[]> entry : partitionData.responseMap.entrySet()) {
				if (entry.getValue().length > 0) {
					foundResponseIds.add(entry.getKey());
				}
			}
		}

		Map<Integer, Integer> compactIdMap = new HashMap<>();
		int previousResponseId = -1;

		for (int responseId : foundResponseIds) {
			int compactId;

			if (responseId - previousResponseId > 1) {
				compactId = previousResponseId + 1;

				String response = filterData.responseIdMap.get(responseId);

				filterData.responseIdMap.put(compactId, response);
				filterData.responseIdMap.remove(responseId);
			} else {
				compactId = responseId;
			}

			compactIdMap.put(responseId, compactId);
			previousResponseId = compactId;
		}

		for (PartitionData partitionData : partitionDatas) {
			TreeMap<Integer, int[]> newResponseMap = new TreeMap<>();

			for (Map.Entry<Integer, int[]> entry : partitionData.responseMap.entrySet()) {
				if (entry.getValue().length > 0) {
					int newResponseId = compactIdMap.get(entry.getKey());

					newResponseMap.put(newResponseId, entry.getValue());
				}
			}

			partitionData.responseMap = newResponseMap;
		}
	}

	private void compactWordIds() {
		Set<Integer> foundWordIds = new TreeSet<>();
		PartitionData[] partitionDatas = new PartitionData[] { filterData.trainingData, filterData.validationData,
		    filterData.testingData };

		for (PartitionData partitionData : partitionDatas) {
			for (int[] wordIds : partitionData.sentenceMap.values()) {
				for (int wordId : wordIds) {
					foundWordIds.add(wordId);
				}
			}
		}

		for (int wordId : EXEMPT_WORD_IDS) {
			foundWordIds.add(wordId);
		}

		Map<Integer, Integer> compactIdMap = new HashMap<>();
		int previousFoundWordId = 1;

		for (int foundWordId : foundWordIds) {
			int compactWordId;

			if (foundWordId - previousFoundWordId > 1) {
				compactWordId = previousFoundWordId + 1;

				String word = filterData.vocabulary.get(foundWordId);

				if (word.isEmpty()) {
					throw new RuntimeException("Word cannot be empty. WordId: " + foundWordId);
				}

				filterData.vocabulary.put(compactWordId, word);
				filterData.vocabulary.remove(foundWordId);

			} else {
				compactWordId = foundWordId;
			}

			previousFoundWordId = compactWordId;
			compactIdMap.put(foundWordId, compactWordId);
		}

		for (PartitionData partitionData : partitionDatas) {
			for (int[] wordIds : partitionData.sentenceMap.values()) {
				for (int i = 0; i < wordIds.length; i++) {
					wordIds[i] = compactIdMap.get(wordIds[i]);
				}
			}
		}
	}

	private void removeDanglingReferences() {
		PartitionData[] partitionDatas = { filterData.trainingData, filterData.validationData, filterData.testingData };

		for (PartitionData partitionData : partitionDatas) {
			removeDanglingReferences(partitionData.documentMap, partitionData.paragraphMap);
			removeDanglingReferences(partitionData.paragraphMap, partitionData.sentenceMap);
		}

		removeDanglingWords();
		removeDanglingResponses();
	}

	private void removeDanglingResponses() {
		PartitionData[] partitionDatas = { filterData.trainingData, filterData.validationData, filterData.testingData };
		Set<Integer> referenced = new HashSet<>();

		for (PartitionData partitionData : partitionDatas) {
			TreeMap<Integer, int[]> newResponseMap = new TreeMap<>();

			for (Map.Entry<Integer, int[]> entry : partitionData.responseMap.entrySet()) {
				ArrayList<Integer> foundDocumentIds = new ArrayList<>();

				for (int documentId : entry.getValue()) {
					if (partitionData.documentMap.containsKey(documentId)) {
						foundDocumentIds.add(documentId);
					}
				}

				int[] documentIds = new int[foundDocumentIds.size()];

				for (int i = 0; i < documentIds.length; i++) {
					documentIds[i] = foundDocumentIds.get(i);
				}

				newResponseMap.put(entry.getKey(), documentIds);
				referenced.add(entry.getKey());
			}

			partitionData.responseMap = newResponseMap;
		}

		Iterator<Integer> iterator = filterData.responseIdMap.keySet().iterator();

		while (iterator.hasNext()) {
			int responseId = iterator.next();

			if (!referenced.contains(responseId)) {
				iterator.remove();
			}
		}
	}

	private void removeDanglingWords() {
		PartitionData[] partitionDatas = { filterData.trainingData, filterData.validationData, filterData.testingData };
		Set<Integer> referenced = new HashSet<>();

		for (PartitionData partitionData : partitionDatas) {
			for (int[] childIds : partitionData.sentenceMap.values()) {
				for (int childId : childIds) {
					referenced.add(childId);
				}
			}
		}

		Iterator<Integer> iterator = filterData.vocabulary.keySet().iterator();

		while (iterator.hasNext()) {
			int wordId = iterator.next();

			if (!referenced.contains(wordId)) {
				if (canBeRemoved(wordId)) {
					iterator.remove();
				}
			}
		}
	}

	private boolean canBeRemoved(int wordId) {
		String word = filterData.vocabulary.get(wordId);

		for (int exemptWordId : EXEMPT_WORD_IDS) {
			if (wordId == exemptWordId) {
				return false;
			}

			if (word.startsWith("<response-")) {
				return false;
			}
		}

		return true;
	}

	private void removeDanglingReferences(TreeMap<Integer, int[]> parentMap, TreeMap<Integer, int[]> childMap) {
		Set<Integer> referenced = new HashSet<>();

		for (int[] childIds : parentMap.values()) {
			for (int childId : childIds) {
				referenced.add(childId);
			}
		}

		Iterator<Integer> childIterator = childMap.keySet().iterator();

		while (childIterator.hasNext()) {
			int childId = childIterator.next();

			if (!referenced.contains(childId)) {
				childIterator.remove();
			}
		}

		for (int parentId : parentMap.keySet()) {
			int[] childIds = parentMap.get(parentId);
			List<Integer> newChildIds = new ArrayList<>();

			for (int childId : childIds) {
				if (childMap.containsKey(childId)) {
					newChildIds.add(childId);
				}
			}

			if (newChildIds.size() != childIds.length) {
				childIds = new int[newChildIds.size()];

				for (int i = 0; i < newChildIds.size(); i++) {
					childIds[i] = newChildIds.get(i);
				}

				parentMap.put(parentId, childIds);
			}
		}
	}

	private int getWordCount(PartitionData partitionData, int documentId) {
		int wordCount = 0;

		for (int paragraphId : partitionData.documentMap.get(documentId)) {
			int[] sentenceIds = partitionData.paragraphMap.get(paragraphId);

			for (int sentenceId : sentenceIds) {
				if (sentenceId == AbstractDocumentsTokenizer.PARAGRAPH_TERMINATOR_SENTENCE_ID) {
					continue;
				}

				// We include the terminator since a period normally counts as a word.
				wordCount += partitionData.sentenceMap.get(sentenceId).length;
			}
		}

		return wordCount;
	}


	static class FilterData {
		PartitionData trainingData;
		PartitionData validationData;
		PartitionData testingData;
		Map<Integer, String> vocabulary;
		Map<Integer, String> responseIdMap;
	}

	static class PartitionData {
		TreeMap<Integer, int[]> responseMap;
		TreeMap<Integer, int[]> documentMap;
		TreeMap<Integer, int[]> paragraphMap;
		TreeMap<Integer, int[]> sentenceMap;
	}
}
