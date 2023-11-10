package sofitda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class QuoraInsincereTokenizer extends AbstractDocumentsTokenizer {
	private static final Pattern entry = Pattern.compile("[^\"]+,\"([^\"]*)\",(\\d)");
	private static final String QUOTE_REPLACER = "#U^$&*^#&(*$";

	public QuoraInsincereTokenizer(int maxRowCount) throws IOException {
		super("quora", true /* hasResponse */);

		System.out.println("Extracting quora dataset...");
		initialize("all");

		String trainFilePath = rawDataDirectory.getAbsolutePath() + "/train.csv";
		BufferedReader reader = null;
		int class0Count = 0;
		int class1Count = 0;
		int errorCount = 0;
		List<String> insincereList = new ArrayList<String>();

		try {
			reader = new BufferedReader(new FileReader(trainFilePath));

			reader.readLine(); // skip headers

			for (int i = 0; i < maxRowCount; i++) {
				String line = reader.readLine();

				try {
					if (line == null) {
						break;
					}

					String temp = line.replace("\"\"\"", "\"" + QUOTE_REPLACER);

					temp = temp.replace("\"\"", QUOTE_REPLACER);

					Matcher matcher = entry.matcher(temp);
					String question;
					int label;

					if (matcher.matches()) {
						question = matcher.group(1);
						label = Integer.parseInt(matcher.group(2));
					} else {
						String[] components = temp.split(",");

						question = components[1];
						label = Integer.parseInt(components[2]);
					}

					if (label != 0 && label != 1) {
						throw new Exception("Invalid label: " + label);
					}

					question = question.replace(QUOTE_REPLACER, "\"");

					if (label == 0) {
						class0Count++;
					} else {
						insincereList.add(question);
						class1Count++;
					}

					processDocument(question, label);
				} catch (Exception e) {
					// e.printStackTrace();
					errorCount++;
				}
			}
		} finally {
			if (reader != null) {
				reader.close();
			}
		}

		System.out.println(
		    "Class 0 Count: " + class0Count + "\n" + "Class 1 Count: " + class1Count + "\n" + "Error Count: " + errorCount);

		System.out.println("Completed extraction.");
		done("all", true /* isLastPrefix */);
	}

	protected List<String> tokenize(String document) {
		String[] tokens = document.split("[\\s]+");

		return Arrays.asList(tokens);
	}

	protected boolean excludeTerminators() {
		return true;
	}

	protected void initializeResponseInfo() {
		responseMap.put(0, new ArrayList<>());
		responseMap.put(1, new ArrayList<>());
	}
}
