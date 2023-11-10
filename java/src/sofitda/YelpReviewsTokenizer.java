package sofitda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.json.JSONObject;
import org.json.JSONTokener;

public class YelpReviewsTokenizer extends AbstractDocumentsTokenizer {
  private BufferedReader reader;

  public YelpReviewsTokenizer(int maxReviewCount) throws IOException {
    super("yelp", true /* hasResponse */);

    initialize("all");

    String reviewFilePath = rawDataDirectory.getAbsolutePath() + "/review.json";

    reader = new BufferedReader(new FileReader(reviewFilePath));

    String line = null;

    for (int i = 0; i < maxReviewCount && (line = reader.readLine()) != null; i++) {
      JSONTokener tokener = new JSONTokener(line);
      JSONObject json = new JSONObject(tokener);
      String text = json.getString("text");
      int response = json.getInt("stars");

      processDocument(text, response);
    }

    done("all", true /* isLastPrefix */);
    reader.close();
  }

  protected List<String> tokenize(String document) {
    String[] tokens = document.split("[\\s]+");

    return Arrays.asList(tokens);
  }

  protected boolean excludeTerminators() {
    return true;
  }
}
