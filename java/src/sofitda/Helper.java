package sofitda;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public class Helper {
	public static void deleteDirectory(File file) {
		if (!file.isDirectory()) {
			file.delete();
			return;
		}

		for (File childFile : file.listFiles()) {
			deleteDirectory(childFile);
		}

		file.delete();
	}

	public static void makeOutputDirectory(String datasetName) {
		makeOutputDirectory(datasetName, true);
	}

	public static void makeOutputDirectory(String datasetName, boolean deleteExisting) {
		Configuration.STAGE2_DIRECTORY = Constants.STAGE2_BASE_DIRECTORY + "/" + datasetName;
		Configuration.STAGE3_DIRECTORY = Constants.STAGE3_BASE_DIRECTORY + "/" + datasetName;
		Configuration.DATASET_NAME = datasetName;

		File outputDirectory = new File(Configuration.STAGE3_DIRECTORY);

		if (deleteExisting && outputDirectory.exists()) {
			Helper.deleteDirectory(outputDirectory);
		}

		outputDirectory.mkdirs();
	}

	public static List<int[]> readMultiColumnIntegerDataset(String filePath) throws IOException {
		List<int[]> dataset = new ArrayList<>();
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		String line;

		while ((line = reader.readLine()) != null) {
			String[] idStrings = line.split("\\s+");
			int[] ids = new int[idStrings.length];

			for (int i = 0; i < ids.length; i++) {
				ids[i] = Integer.parseInt(idStrings[i]);
			}

			dataset.add(ids);
		}

		reader.close();
		return dataset;
	}

	public static void writeMultiColumnIntegerDataset(String filePath, List<int[]> dataset)
			throws FileNotFoundException {
		PrintWriter writer = new PrintWriter(filePath);

		for (int[] row : dataset) {
			for (int i = 0; i < row.length; i++) {
				if (i != 0) {
					writer.print(" ");
				}

				writer.write(String.valueOf(row[i]));
			}

			writer.write("\n");
		}

		writer.close();
	}

	public static <T> void writeMultiColumnObjectDataset(String filePath, List<T[]> dataset)
			throws FileNotFoundException {
		PrintWriter writer = new PrintWriter(filePath);

		for (Object[] row : dataset) {
			for (int i = 0; i < row.length; i++) {
				if (i != 0) {
					writer.print(" ");
				}

				writer.write(String.valueOf(row[i]));
			}

			writer.write("\n");
		}

		writer.close();
	}

	public static TreeMap<Integer, int[]> readMultiColumnIntegerMap(String filePath) throws IOException {
		TreeMap<Integer, int[]> rows = new TreeMap<>();
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		String line;

		while ((line = reader.readLine()) != null) {
			String[] idStrings = line.split("\\s+");

			if (idStrings.length > 0) {
				int parentId = Integer.parseInt(idStrings[0]);
				int[] childIds = new int[idStrings.length - 1];

				for (int i = 1; i < idStrings.length; i++) {
					childIds[i - 1] = Integer.parseInt(idStrings[i]);
				}

				rows.put(parentId, childIds);
			}
		}

		reader.close();
		return rows;
	}

	public static TreeMap<String, String[]> readMultiColumnStringMap(String filePath) throws IOException {
		TreeMap<String, String[]> rows = new TreeMap<>();
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		String line;

		while ((line = reader.readLine()) != null) {
			String[] idStrings = line.split("\\s+");

			if (idStrings.length > 0) {
				String parentId = idStrings[0];
				String[] childIds = new String[idStrings.length - 1];

				for (int i = 1; i < idStrings.length; i++) {
					childIds[i - 1] = idStrings[i];
				}

				rows.put(parentId, childIds);
			}
		}

		reader.close();
		return rows;
	}

	public static TreeMap<Integer, String[]> readMultiColumnStringMap(String filePath, int columnCount)
			throws IOException {
		TreeMap<Integer, String[]> rows = new TreeMap<>();
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		String line;

		while ((line = reader.readLine()) != null) {
			String[] idStrings = line.split("\\s+");

			if (idStrings.length >= columnCount) {
				int parentId = Integer.parseInt(idStrings[0]);
				String[] childIds = new String[columnCount];

				for (int i = 1; i < columnCount; i++) {
					if (i == columnCount - 1) {
						StringBuffer buffer = new StringBuffer();

						for (int j = i; j < idStrings.length; j++) {
							if (j > i) {
								buffer.append(' ');
							}

							buffer.append(idStrings[j]);
						}

						childIds[i - 1] = buffer.toString();
					} else {
						childIds[i - 1] = idStrings[i];
					}
				}

				rows.put(parentId, childIds);
			}
		}

		reader.close();
		return rows;
	}

	public static void writeIntegralDictionary(Map<Integer, Integer> map, String filePath)
			throws FileNotFoundException {
		PrintWriter writer = new PrintWriter(filePath);

		for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
			writer.print(entry.getKey().toString());
			writer.print(" ");
			writer.print(entry.getValue().toString());
			writer.print("\n");
		}

		writer.close();
	}

	public static void writeStringDictionary(Map<String, Integer> map, String filePath) throws FileNotFoundException {
		PrintWriter writer = new PrintWriter(filePath);

		for (Map.Entry<String, Integer> entry : map.entrySet()) {
			writer.print(entry.getKey().toString());
			writer.print(" ");
			writer.print(entry.getValue().toString());
			writer.print("\n");
		}

		writer.close();
	}

	public static void writeMultiColumnIntegerArrayMap(Map<Integer, int[]> map, String filePath)
			throws FileNotFoundException {
		PrintWriter writer = new PrintWriter(filePath);

		for (Map.Entry<Integer, int[]> entry : map.entrySet()) {
			writer.print(entry.getKey().toString());

			for (int value : entry.getValue()) {
				writer.print(" ");
				writer.print(String.valueOf(value));
			}

			writer.print("\n");
		}

		writer.close();
	}

	public static void writeMultiColumnDoubleArrayMap(Map<Integer, double[]> map, String filePath)
			throws FileNotFoundException {
		PrintWriter writer = new PrintWriter(filePath);

		for (Map.Entry<Integer, double[]> entry : map.entrySet()) {
			writer.print(entry.getKey().toString());

			for (double value : entry.getValue()) {
				writer.print(" ");
				writer.print(String.valueOf(value));
			}

			writer.print("\n");
		}

		writer.close();
	}

	public static void writeMultiColumnIntegerListMap(Map<Integer, List<Integer>> map, String filePath)
			throws FileNotFoundException {
		PrintWriter writer = new PrintWriter(filePath);

		for (Map.Entry<Integer, List<Integer>> entry : map.entrySet()) {
			writer.print(entry.getKey().toString());

			for (int value : entry.getValue()) {
				writer.print(" ");
				writer.print(String.valueOf(value));
			}

			writer.print("\n");
		}

		writer.close();
	}

	public static List<Integer> readSingleColumnIntegerDataset(String filePath) throws IOException {
		List<Integer> dataset = new ArrayList<>();
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		String line;

		while ((line = reader.readLine()) != null) {
			dataset.add(Integer.parseInt(line.trim()));
		}

		reader.close();
		return dataset;
	}

	public static void writeSingleColumnDataset(String filePath, Collection<?> collection)
			throws FileNotFoundException {
		PrintWriter writer = new PrintWriter(filePath);

		for (Object object : collection) {
			writer.print(object);
			writer.write("\n");
		}

		writer.close();
	}

	public static Map<Integer, String> readSingleColumnStringMap(String filePath, int skipLineCount)
			throws IOException {
		Map<Integer, String> rows = new TreeMap<>();
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		String line;
		int count = 0;

		while ((line = reader.readLine()) != null) {
			if (count++ < skipLineCount) {
				continue;
			}

			String[] idStrings = line.split("\\s+");

			if (idStrings.length == 2) {
				String value = idStrings[0];
				int key = Integer.parseInt(idStrings[1]);

				rows.put(key, value);
			}
		}

		reader.close();
		return rows;
	}

	public static Map<Integer, Integer> readSingleColumnIntegerMap(String filePath, int skipLineCount)
			throws IOException {
		Map<Integer, Integer> rows = new TreeMap<>();
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		String line;
		int count = 0;

		while ((line = reader.readLine()) != null) {
			if (count++ < skipLineCount) {
				continue;
			}

			String[] idStrings = line.split("\\s+");

			if (idStrings.length == 2) {
				int key = Integer.parseInt(idStrings[0]);
				int value = Integer.parseInt(idStrings[1]);

				rows.put(key, value);
			}
		}

		reader.close();
		return rows;
	}

	public static void writeSingleColumnStringMap(String filePath, String header, Map<Integer, String> map)
			throws IOException {
		PrintWriter writer = new PrintWriter(filePath);

		if (header != null) {
			writer.write(header);
			writer.write("\n");
		}

		for (Map.Entry<Integer, String> entry : map.entrySet()) {
			writer.write(entry.getValue());
			writer.write(" ");
			writer.write(entry.getKey().toString());
			writer.write("\n");
		}

		writer.close();
	}

	public static int getMaxId(String filePath) throws IOException {
		return getMaxId(filePath, 0, 0);
	}

	public static int getMaxId(String filePath, int idColumn, int skipLineCount) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		String line;
		int maxId = -1;
		int lineCount = 0;

		while ((line = reader.readLine()) != null) {
			if (lineCount++ < skipLineCount) {
				continue;
			}

			String[] idStrings = line.split("\\s+");

			if (idStrings.length > 0) {
				int id = Integer.parseInt(idStrings[idColumn]);

				if (id > maxId) {
					maxId = id;
				}
			}
		}

		reader.close();
		return maxId;
	}

	public static int getMaxColumnCount(String filePath) throws IOException {
		return getMaxColumnCount(filePath, 0, 0);
	}

	public static int getMaxColumnCount(String filePath, int idColumn, int skipLineCount) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		String line;
		int lineCount = 0;
		int maxColumnCount = -1;

		while ((line = reader.readLine()) != null) {
			if (lineCount++ < skipLineCount) {
				continue;
			}

			String[] idStrings = line.split("\\s+");

			if (idStrings.length > maxColumnCount) {
				maxColumnCount = idStrings.length;
			}
		}

		reader.close();
		return maxColumnCount;
	}

	public static void saveDatasetVocabulary(Map<String, Integer> datasetVocabularyMap) throws FileNotFoundException {
		PrintWriter writer = new PrintWriter(Configuration.STAGE3_DIRECTORY + "/" + Constants.VOCABULARY_FILENAME);

		writer.write("Word\tID\n");

		Map<Integer, String> reverseVocabularyMap = new TreeMap<Integer, String>();

		for (Map.Entry<String, Integer> entry : datasetVocabularyMap.entrySet()) {
			reverseVocabularyMap.put(entry.getValue(), entry.getKey());
		}

		for (Map.Entry<Integer, String> entry : reverseVocabularyMap.entrySet()) {
			writer.write(entry.getValue());
			writer.write("\t");
			writer.write(String.valueOf(entry.getKey()));
			writer.write("\n");
		}

		writer.close();
	}

	public static void touch(String filePath) throws FileNotFoundException, IOException {
		new FileOutputStream(filePath).close();
	}

	public static void zip(String directoryPath, String zipFilename) throws IOException {
		File directory = new File(directoryPath);
		ZipOutputStream zipOut = new ZipOutputStream(new FileOutputStream(directory.getParent() + "/" + zipFilename));

		zipFile(directory, directory.getName(), zipOut);
		zipOut.close();
	}

	private static void zipFile(File fileToZip, String fileName, ZipOutputStream zipOut) throws IOException {
		if (fileToZip.isHidden()) {
			return;
		}

		if (fileToZip.isDirectory()) {
			if (fileName.endsWith("/")) {
				zipOut.putNextEntry(new ZipEntry(fileName));
				zipOut.closeEntry();
			} else {
				zipOut.putNextEntry(new ZipEntry(fileName + "/"));
				zipOut.closeEntry();
			}

			File[] children = fileToZip.listFiles();

			for (File childFile : children) {
				zipFile(childFile, fileName + "/" + childFile.getName(), zipOut);
			}

			return;
		}

		FileInputStream fis = new FileInputStream(fileToZip);
		ZipEntry zipEntry = new ZipEntry(fileName);
		zipOut.putNextEntry(zipEntry);
		byte[] bytes = new byte[1024];
		int length;

		while ((length = fis.read(bytes)) >= 0) {
			zipOut.write(bytes, 0, length);
		}

		fis.close();
	}
	
	 public static PrintWriter createPrintWriter(String filePath) throws FileNotFoundException, UnsupportedEncodingException {
	    FileOutputStream fos = new FileOutputStream(filePath);
	  
	    return new PrintWriter(new OutputStreamWriter(fos, "UTF-8"));
	  }


	public static Map<String, String> nextJson(Reader reader) throws IOException {
		char character = (char) -1;
		Map<String, String> attributes = null;
		boolean isInJson = false;
		boolean isInDoubleQuote = false;
		boolean isEscapingNextCharacter = false;
		boolean isEndOfReview = false;
		boolean isInKey = false;
		boolean isInValue = false;
		StringBuilder keyBuilder = new StringBuilder(100);
		StringBuilder valueBuilder = new StringBuilder(4096);
		int count = 0;

		while (!isEndOfReview && (character = (char) reader.read()) != -1) {
			if (count++ > 1000000) {
				System.out.println("Max loop count exceeded");
				return null;
			}

			if (isInDoubleQuote == false) {
				if (character == '{') {
					if (isInJson) {
						throw new IllegalStateException("{ unexpectedly encountered");
					} else {
						isInJson = true;
						attributes = new HashMap<>();
					}
				} else if (character == '\"') {
					isInDoubleQuote = true;

					if (isInValue == false) {
						isInKey = true;
					}
				} else if (character == ':') {
					isInValue = true;
				} else if (character == ',' || character == '}') {
					String key = keyBuilder.toString().trim();
					String value = valueBuilder.toString().trim();

					keyBuilder.setLength(0);
					valueBuilder.setLength(0);
					isInValue = false;

					if (key != null && !key.isEmpty() && value != null && !value.isEmpty()) {
						attributes.put(key, value);
					}

					if (character == '}') {
						isInJson = false;
						isEndOfReview = true;
					}
				} else if (isInValue) {
					valueBuilder.append(character);
				}
			} else {
				if (character == '[' || character == ']') {
				} else if (character == '\\' && isEscapingNextCharacter == false) {
					isEscapingNextCharacter = true;
				} else if (character == '\"' && isEscapingNextCharacter == false) {
					if (isInKey == true) {
						isInKey = false;
					}

					isInDoubleQuote = false;
				} else {
					if (isEscapingNextCharacter == true) {
						switch (character) {
						case 'n':
							character = '\n';
							break;

						case 't':
							character = '\t';
						}

						isEscapingNextCharacter = false;
					}

					if (isInKey) {
						keyBuilder.append(character);
					} else if (isInValue) {
						valueBuilder.append(character);
					}
				}
			}
		}

		if (isInJson || character == -1) {
			return null;
		} else {
			return attributes;
		}
	}
}
