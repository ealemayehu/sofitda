package sofitda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.json.JSONObject;
import org.json.JSONTokener;

import java.nio.charset.Charset;

public class TripAdvisorTokenizer extends AbstractDocumentsTokenizer {
  private BufferedReader reader;

  public TripAdvisorTokenizer(int maxReviewCount) throws IOException {
    super("tripadvisor", true /* hasResponse */);

    initialize("all");

    String reviewFilePath = rawDataDirectory.getAbsolutePath() + "/review.json";

    reader = new BufferedReader(new FileReader(reviewFilePath));

    String line = null;

    for (int i = 0; i < maxReviewCount && (line = reader.readLine()) != null; i++) {
      JSONTokener tokener = new JSONTokener(line);
      JSONObject json = new JSONObject(tokener);
      String text = json.getString("text").trim();

      if (!isPureAscii(text)) {
        continue;
      }

      int response = (int) Math.round((json.getJSONObject("ratings").getFloat("overall")));

      if (response == 0) {
        continue;
      }

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

  private boolean isPureAscii(String v) {
    return Charset.forName("US-ASCII").newEncoder().canEncode(v);
    // or "ISO-8859-1" for ISO Latin 1
    // or StandardCharsets.US_ASCII with JDK1.7+
  }
}
