package sofitda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.json.JSONObject;
import org.json.JSONTokener;

public class SST5Tokenizer extends AbstractDocumentsTokenizer {
  public SST5Tokenizer() throws IOException {
    super("sst5", true /* hasResponse */);

    createDataset("training", false /* isLastPrefix */);
    createDataset("validation", false /* isLastPrefix */);
    createDataset("testing", true /* isLastPrefix */);
  }

  private void createDataset(String prefix, boolean isLastPrefix) throws IOException {
    initialize(prefix);
    processResponse(prefix);
    done(prefix, isLastPrefix);
  }

  protected void initializeResponseInfo() {
    for (int i = 0; i < 5; i++) {
      responseMap.put(i, new ArrayList<>());
    }
    
    responseIdMap.put("very negative", 0);
    responseIdMap.put("negative", 1);
    responseIdMap.put("neutral", 2);
    responseIdMap.put("positive", 3);
    responseIdMap.put("very positive", 4);
  }

  protected List<String> tokenize(String document) {
    String[] tokens = document.split(" ");

    return Arrays.asList(tokens);
  }

  protected String sentenceTerminator() {
    return ".";
  }

  private void processResponse(String prefix) throws IOException {
    String reviewFilePath = rawDataDirectory.getAbsolutePath() + "/" + prefix + ".jsonl";
    BufferedReader reader = new BufferedReader(new FileReader(reviewFilePath));
    String line;

    for (int i = 0; (line = reader.readLine()) != null; i++) {
      if (i % 1000 == 0) {
        System.out.println("Processed " + i + " reviews for prefix: " + prefix);
      }

      JSONTokener tokener = new JSONTokener(line);
      JSONObject json = new JSONObject(tokener);
      String text = json.getString("text");
      String response = json.getString("label_text");

      processDocument(text, response);
    }

    reader.close();
  }
  
  protected boolean excludeTerminators() {
    return true;
  }
}