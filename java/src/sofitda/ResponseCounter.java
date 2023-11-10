package sofitda;

import java.io.IOException;
import java.util.Map;
import java.util.TreeMap;

public class ResponseCounter {
	public ResponseCounter() throws IOException {
		String[] prefixes = { "training", "validation", "testing", "all" };

		for (String prefix : prefixes) {
			String responseDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
			    + String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, prefix);

			Map<Integer, int[]> responseMap = Helper.readMultiColumnIntegerMap(responseDatasetFilePath);
			Map<Integer, Integer> responseCountMap = new TreeMap<>();

			for (Map.Entry<Integer, int[]> entry : responseMap.entrySet()) {
				responseCountMap.put(entry.getKey(), entry.getValue().length);
			}

			String responseCountDatasetFilePath = Configuration.STAGE3_DIRECTORY + "/"
			    + String.format(Constants.RESPONSE_COUNT_DATASET_FORMAT, prefix);

			Helper.writeIntegralDictionary(responseCountMap, responseCountDatasetFilePath);
		}
	}
}
