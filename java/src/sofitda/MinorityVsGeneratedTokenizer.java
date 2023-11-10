package sofitda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class MinorityVsGeneratedTokenizer extends AbstractDocumentsTokenizer {
	private int responseId;

	public MinorityVsGeneratedTokenizer(String name, int responseId) throws IOException {
		super(name, true /* hasResponse */);

		this.responseId = responseId;
		
		initialize("all");
		
		String responseDatasetFilePath = this.rawDataDirectory.getAbsolutePath() + "/"
		    + String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, "training");
		BufferedReader responseReader = new BufferedReader(new FileReader(responseDatasetFilePath));
		Set<Integer> documentIds = new HashSet<>();
    String line;
		
		while ((line = responseReader.readLine()) != null) {
		  String[] tokens = line.split(" ");
		  int temp = Integer.parseInt(tokens[0]);
		  
		  if (temp != responseId) {
		    continue;
		  }
		  
		  for (int i = 1; i < tokens.length; i++) {
		    documentIds.add(Integer.parseInt(tokens[i]));
		  }
		}
		
		responseReader.close();

		String documentWordDatasetTextFilePath = this.rawDataDirectory.getAbsolutePath() + "/"
		    + String.format(Constants.DOCUMENT_WORD_DATASET_TEXT_FILENAME_FORMAT, "training");
		BufferedReader reader = new BufferedReader(new FileReader(documentWordDatasetTextFilePath));

		while ((line = reader.readLine()) != null) {
			int firstSpaceIndex = line.indexOf(' ');
			int secondSpaceIndex = line.indexOf(' ', firstSpaceIndex + 1);
			int documentId = Integer.parseInt(line.substring(0, firstSpaceIndex));
			
			if (!documentIds.contains(documentId)) {
			  continue;
			}
			
			String document = line.substring(secondSpaceIndex);

			processDocument(document, 1);
		}
		
		addGenerated();

		reader.close();
		done("all", true);
	}
	
	private void addGenerated() throws IOException {
		String generatedFile = this.rawDataDirectory.getAbsolutePath() + "/"
		    + String.format("generated_%d.txt", this.responseId);
		
		BufferedReader reader = new BufferedReader(new FileReader(generatedFile));
		String line;
		
		while ((line = reader.readLine()) != null) {
			processDocument(line, 0);
		}
		
		reader.close();
	}
	
	protected List<String> tokenize(String document) {
		String[] tokens = document.split("[\\s\"']+");
		List<String> tokenList = new ArrayList<>();
		
		for (String token: tokens) {
			if (token != null && !token.isEmpty()) {
				tokenList.add(token);
			}
		}

		return tokenList;
	}
	
	protected boolean excludeTerminators() {
		return true;
	}
}
