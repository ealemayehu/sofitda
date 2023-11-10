package sofitda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

//  Columns in the Amazon tsv file.
//
//  Column 0: marketplace
//  Column 1: customer_id
//  Column 2: review_id
//  Column 3: product_id
//  Column 4: product_parent
//  Column 5: product_title
//  Column 6: product_category
//  Column 7: star_rating
//  Column 8: helpful_votes
//  Column 9: total_votes
//  Column 10: vine
//  Column 11: verified_purchase
//  Column 12: review_headline
//  Column 13: review_body
//  Column 14: review_date
public class AmazonReviewsTokenizer extends AbstractDocumentsTokenizer {
  public AmazonReviewsTokenizer() throws IOException {
    super("amazon", true /* hasResponse */);

    initialize("all");

    processDocuments("amazon_reviews_us_Camera_v1_00.tsv", 40000);

    done("all", true /* isLastPrefix */);
  }

  public void processDocuments(String filename, int reviewCount) throws IOException {
    String reviewFilePath = rawDataDirectory.getAbsolutePath() + "/" + filename;
    BufferedReader reader = new BufferedReader(new FileReader(reviewFilePath));
    boolean firstLine = true;
    int count = 0;
    int errorCount = 0;

    while (true) {
      try {
        String[] columns = reader.readLine().split("\t");

        if (firstLine) {
          for (int j = 0; j < columns.length; j++) {
            System.out.println("Column " + j + ": " + columns[j]);
          }

          firstLine = false;
        } else {
          String review = columns[13];
          int rating = Integer.parseInt(columns[7]);

          processDocument(review.replace("<br />", "\n"), rating);
          count++;

          if (count == reviewCount) {
            break;
          }
        }
      } catch (Exception e) {
        e.printStackTrace();
        errorCount++;
      }
    }

    reader.close();

    System.out.println("Count: " + count + ", Error Count: " + errorCount);
  }

  protected List<String> tokenize(String document) {
    String[] tokens = document.split("[\\s\"']+");

    return Arrays.asList(tokens);
  }

  protected boolean excludeTerminators() {
    return true;
  }
}
