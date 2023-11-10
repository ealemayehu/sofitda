package sofitda;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;

public abstract class AbstractDocumentsTokenizer {
	public static final int SENTENCE_TERMINATOR_WORD_ID = 2;
	public static final int PARAGRAPH_TERMINATOR_WORD_ID = 1;
	public static final int DOCUMENT_TERMINATOR_WORD_ID = 0;
	public static final int UNKNOWN_WORD_ID = 3;

	protected static final String SENTENCE_TERMINATOR_TEXT = "<end-sentence>";
	protected static final String PARAGRAPH_TERMINATOR_TEXT = "<end-paragraph>";
	protected static final String DOCUMENT_TERMINATOR_TEXT = "<end-document>";
	protected static final String UNKNOWN_WORD_TEXT = "<unknown>";

	public static final int DOCUMENT_TERMINATOR_SENTENCE_ID = 0;
	public static final int PARAGRAPH_TERMINATOR_SENTENCE_ID = 1;

	public static final int DOCUMENT_TERMINATOR_PARAGRAPH_ID = 0;

	protected int sentenceCount;
	protected int paragraphCount;
	protected int documentCount;
	protected int maxSentenceId;
	protected int maxParagraphId;
	protected int maxResponseId;
	protected PrintWriter sentenceDatasetWriter;
	protected PrintWriter paragraphDatasetWriter;
	protected PrintWriter documentDatasetWriter;
	protected Map<String, Integer> datasetVocabularyMap = new HashMap<>();
	protected Map<Integer, List<Integer>> responseMap = new TreeMap<>();
	protected Map<String, Integer> responseIdMap = new TreeMap<>();
	protected List<String> datasetVocabularyList = new ArrayList<>();
	protected List<String> currentSentence = new ArrayList<>();
	protected List<Integer> currentParagraph = new ArrayList<>();
	protected List<Integer> currentDocument = new ArrayList<>();
	protected File rawDataDirectory;
	protected String datasetName;
	protected boolean hasResponse;

	public AbstractDocumentsTokenizer(String datasetName, boolean hasResponse) throws IOException {
		this.datasetName = datasetName;
		this.hasResponse = hasResponse;

		rawDataDirectory = new File(Constants.STAGE1_BASE_DIRECTORY + "/" + datasetName);

		if (!rawDataDirectory.exists()) {
			System.err.println("Directory, " + rawDataDirectory.getAbsolutePath() + ", does not exist");
			return;
		}

		Helper.makeOutputDirectory(datasetName);
		initializeVocabulary();
	}

	protected boolean excludeTerminators() {
		return false;
	}

	protected void initializeVocabulary() {
		datasetVocabularyMap.clear();
		datasetVocabularyMap.put(SENTENCE_TERMINATOR_TEXT, SENTENCE_TERMINATOR_WORD_ID);
		datasetVocabularyList.add(SENTENCE_TERMINATOR_TEXT);
		datasetVocabularyMap.put(PARAGRAPH_TERMINATOR_TEXT, PARAGRAPH_TERMINATOR_WORD_ID);
		datasetVocabularyList.add(PARAGRAPH_TERMINATOR_TEXT);
		datasetVocabularyMap.put(DOCUMENT_TERMINATOR_TEXT, DOCUMENT_TERMINATOR_WORD_ID);
		datasetVocabularyList.add(DOCUMENT_TERMINATOR_TEXT);
		datasetVocabularyMap.put(UNKNOWN_WORD_TEXT, UNKNOWN_WORD_ID);
		datasetVocabularyList.add(UNKNOWN_WORD_TEXT);
	}

	protected void initializeSentenceInfo() {
		sentenceDatasetWriter.print(String.valueOf(DOCUMENT_TERMINATOR_SENTENCE_ID));
		sentenceDatasetWriter.print(" ");
		sentenceDatasetWriter.print(String.valueOf(DOCUMENT_TERMINATOR_WORD_ID));
		sentenceDatasetWriter.print("\n");
		sentenceCount++;

		sentenceDatasetWriter.print(String.valueOf(PARAGRAPH_TERMINATOR_SENTENCE_ID));
		sentenceDatasetWriter.print(" ");
		sentenceDatasetWriter.print(String.valueOf(PARAGRAPH_TERMINATOR_WORD_ID));
		sentenceDatasetWriter.print("\n");
		sentenceCount++;
	}

	protected void initializeParagraphInfo() {
		paragraphDatasetWriter.print(String.valueOf(DOCUMENT_TERMINATOR_PARAGRAPH_ID));
		paragraphDatasetWriter.print(" ");
		paragraphDatasetWriter.print(String.valueOf(DOCUMENT_TERMINATOR_SENTENCE_ID));
		paragraphDatasetWriter.print("\n");
		paragraphCount++;
	}

	protected void initializeResponseInfo() {
		responseMap.clear();
	}
	

	protected void initialize(String prefix) throws FileNotFoundException, UnsupportedEncodingException {
		sentenceDatasetWriter = Helper.createPrintWriter(Configuration.STAGE3_DIRECTORY + "/"
				+ String.format(Constants.SENTENCE_DATASET_FILENAME_FORMAT, prefix));
		paragraphDatasetWriter = Helper.createPrintWriter(Configuration.STAGE3_DIRECTORY + "/"
				+ String.format(Constants.PARAGRAPH_DATASET_FILENAME_FORMAT, prefix));
		documentDatasetWriter = Helper.createPrintWriter(Configuration.STAGE3_DIRECTORY + "/"
				+ String.format(Constants.DOCUMENT_DATASET_FILENAME_FORMAT, prefix));

		initializeSentenceInfo();
		initializeParagraphInfo();

		if (hasResponse) {
			initializeResponseInfo();
		}
	}

	protected void done(String prefix, boolean isLastPrefix) throws FileNotFoundException {
		System.out.println("Prefix " + prefix + " - Unique sentenceCount: " + sentenceCount
				+ ", Unique paragraphCount: " + paragraphCount);

		sentenceDatasetWriter.close();
		paragraphDatasetWriter.close();
		documentDatasetWriter.close();
		saveResponse(prefix);

		if (isLastPrefix) {
			Helper.saveDatasetVocabulary(datasetVocabularyMap);

			if (responseIdMap.size() > 0) {
				saveResponseMap(prefix);
			}
		}
	}

	protected void saveResponseMap(String prefix) throws FileNotFoundException {
		String filename = Configuration.STAGE3_DIRECTORY + "/" + Constants.RESPONSE_DATASET_TEXT_FILENAME;

		Helper.writeStringDictionary(responseIdMap, filename);
	}

	protected void processDocument(String document) {
		int[] responseIds = {};

		internalProcessDocument(document, responseIds);
	}

	protected void processDocument(String document, Integer responseId) {
		if (responseId == null) {
			int[] responseIds = {};

			processDocument(document, responseIds);
		} else {
			int[] responseIds = { responseId };

			processDocument(document, responseIds);
		}
	}
	
	protected void processDocument(String document, String responseId) {
		processDocument(document, new String[] {responseId});
	}

	protected void processDocument(String document, String[] responses) {
		List<String> responseList = new ArrayList<>();

		for (String response : responses) {
			responseList.add(response);
		}

		processDocument(document, responseList);
	}

	protected void processDocument(String document, List<String> responses) {
		int[] responseIds = new int[responses.size()];

		for (int i = 0; i < responses.size(); i++) {
			String response = responses.get(i);
			Integer responseId = responseIdMap.get(response);

			if (responseId == null) {
				responseId = maxResponseId++;
				responseIdMap.put(response, responseId);
			}

			responseIds[i] = responseId;
		}

		internalProcessDocument(document, responseIds);
	}

	protected void processDocument(String document, int[] responseIds) {
		for (int responseId : responseIds) {
			responseIdMap.put(String.valueOf(responseId), responseId);
		}

		internalProcessDocument(document, responseIds);
	}

	private void internalProcessDocument(String document, int[] responseIds) {
		List<String> tokens = tokenize(document);

		for (String token : tokens) {
			if (token.equals(sentenceTerminator())) {
				addSentence();
			} else if (token.equals(paragraphTerminator())) {
				addParagraph();
			} else {
				for (String subToken : token.split(" ")) {
					if (!subToken.isEmpty()) {
						currentSentence.add(subToken);
					}
				}
			}
		}

		addSentence();
		addParagraph();
		addDocument(responseIds);
	}

	protected String sentenceTerminator() {
		return ".";
	}

	protected String paragraphTerminator() {
		return "*nl*";
	}

	protected List<String> tokenize(String document) {
		List<String> tokens = new ArrayList<>();
		StringReader reader = new StringReader(document);
		PTBTokenizer<CoreLabel> tokenizer = new PTBTokenizer<>(reader, new CoreLabelTokenFactory(), "tokenizeNLs=true");

		while (tokenizer.hasNext()) {
			tokens.add(tokenizer.next().toString().toLowerCase());
		}

		return tokens;
	}

	protected void saveResponse(String prefix) throws FileNotFoundException {
		String responseFilePath = Configuration.STAGE3_DIRECTORY + "/"
				+ String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, prefix);

		Helper.writeMultiColumnIntegerListMap(responseMap, responseFilePath);
	}

	protected void addSentence() {
		if (currentSentence.size() == 0 || currentSentence.size() > Constants.MAX_SENTENCE_LENGTH) {
			currentSentence.clear();
			return;
		}

		List<Integer> wordIds = new ArrayList<>();

		for (int i = 0; i < currentSentence.size() && i < Constants.MAX_SENTENCE_LENGTH; i++) {
			String word = currentSentence.get(i);

			wordIds.add(getWordId(word));
		}

		if (!excludeTerminators()) {
			wordIds.add(SENTENCE_TERMINATOR_WORD_ID);
			currentSentence.add(SENTENCE_TERMINATOR_TEXT);
		}

		int sentenceId = sentenceCount++;

		currentParagraph.add(sentenceId);
		sentenceDatasetWriter.write(String.valueOf(sentenceId));

		for (int i = 0; i < wordIds.size(); i++) {
			int wordId = wordIds.get(i);

			sentenceDatasetWriter.write(" ");
			sentenceDatasetWriter.write(String.valueOf(wordId));
		}

		currentSentence.clear();
		sentenceDatasetWriter.write("\n");
	}

	protected void addParagraph() {
		if (currentParagraph.size() == 0) {
			return;
		}

		if (!excludeTerminators()) {
			currentParagraph.add(PARAGRAPH_TERMINATOR_SENTENCE_ID);
		}

		int paragraphId = paragraphCount++;

		currentDocument.add(paragraphId);
		paragraphDatasetWriter.write(String.valueOf(paragraphId));

		for (Integer sentenceId : currentParagraph) {
			paragraphDatasetWriter.write(" ");
			paragraphDatasetWriter.write(String.valueOf(sentenceId));
		}

		paragraphDatasetWriter.write("\n");
		currentParagraph.clear();
	}

	protected void addDocument(int[] responseIds) {
		if (currentDocument.size() == 0) {
			return;
		}

		if (!excludeTerminators()) {
			currentDocument.add(DOCUMENT_TERMINATOR_PARAGRAPH_ID);
		}

		documentDatasetWriter.write(String.valueOf(documentCount));

		if (hasResponse) {
			addResponse(responseIds, documentCount);
		}

		for (Integer paragraphId : currentDocument) {
			documentDatasetWriter.write(" ");
			documentDatasetWriter.write(String.valueOf(paragraphId));
		}

		documentDatasetWriter.write("\n");
		documentCount++;
		currentDocument.clear();
	}

	protected void addResponse(int[] responseIds, int documentId) {
		for (int responseId : responseIds) {
			List<Integer> documentIds = responseMap.get(responseId);

			if (documentIds == null) {
				documentIds = new ArrayList<>();
				responseMap.put(responseId, documentIds);
			}

			documentIds.add(documentId);
		}
	}

	protected int getWordId(String word) {
		Integer wordId = datasetVocabularyMap.get(word);

		if (wordId == null) {
			wordId = datasetVocabularyMap.size();

			if (word.isEmpty()) {
				throw new RuntimeException("Word cannot be empty");
			}

			datasetVocabularyMap.put(word, wordId);
			datasetVocabularyList.add(word);
		}

		return wordId;
	}
}
