package sofitda;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;

import javax.xml.bind.JAXBException;

import org.apache.commons.io.FileUtils;

public class Main {
	public static void main(String[] args) throws IOException, JAXBException {
		if (args.length < 1) {
			System.out.println("Usage: dataset-name [optional args]");
			return;
		}

		switch (args[0]) {
		case "yelp":
			createYelpDatasets();
			break;

		case "amazon":
			createAmazonDatasets();
			break;

		case "tripadvisor":
			createTripAdvisorDatasets();
			break;
			
    case "sst5":
      createSST5Datasets();
      break;

		case "quora":
			createQuoraDatasets();
			break;

		case "trec":
		  createTRECDatasets();
		  break;

		default:
			if (args[0].startsWith("mingen")) {
				createMingenDatasets(args[0]);
			} else if (args[0].startsWith("aug")) {
				createAugmentedDatasets(args[0]);
			} else if (args[0].startsWith("genrank")) {
				createGenrankDatasets(args[0]);
			} else {
				System.out.println("Unknown dataset: " + args[0]);
			}
		}

		System.out.println("DONE!");
	}



	private static void createYelpDatasets() throws IOException {
		new YelpReviewsTokenizer(100000);
		new DatasetPartitioner();
		new FilterPipeline();
		new HierarchyCollapser();
		new DatasetMerger();
		new MetadataGenerator("dw");
		new ExtraDatasetGenerator();
		new ResponseCounter();
		new TextGenerationDataset();
	}
	
  private static void createSST5Datasets() throws IOException {
    new SST5Tokenizer();
    new FilterPipeline();
    new HierarchyCollapser();
    new DatasetMerger();
    new MetadataGenerator("dw");
    new ExtraDatasetGenerator();
    new ResponseCounter();
    new TextGenerationDataset();
  }
  
  private static void createTRECDatasets() throws IOException {
    new TRECTokenizer();
    new FilterPipeline();
    new HierarchyCollapser();
    new DatasetMerger();
    new MetadataGenerator("dw");
    new ExtraDatasetGenerator();
    new ResponseCounter();
    new TextGenerationDataset();
  }

	private static void createAmazonDatasets() throws IOException {
		new AmazonReviewsTokenizer();
		new DatasetPartitioner();
		new FilterPipeline();
		new HierarchyCollapser();
		new DatasetMerger();
		new MetadataGenerator("dw");
		new ExtraDatasetGenerator();
		new ResponseCounter();
		new TextGenerationDataset();
	}

	private static void createTripAdvisorDatasets() throws IOException {
		new TripAdvisorTokenizer(300000);
		new DatasetPartitioner();
		new FilterPipeline();
		new HierarchyCollapser();
		new DatasetMerger();
		new MetadataGenerator("dw");
		new ExtraDatasetGenerator();
		new ResponseCounter();
		new TextGenerationDataset();
	}

	private static void createQuoraDatasets() throws IOException {
		new QuoraInsincereTokenizer(40000);
		new DatasetPartitioner();
		new FilterPipeline();
		new HierarchyCollapser();
		new DatasetMerger();
		new MetadataGenerator("dw");
		new ExtraDatasetGenerator();
		new TextGenerationDataset();
		new ResponseCounter();
	}
	
	private static void createAugmentedDatasets(String datasetName) throws IOException {
		int index = datasetName.indexOf("aug");
		
		if (index != 0) {
			throw new RuntimeException(datasetName + " is not a valid augmented dataset name");
		}
		
		index = datasetName.indexOf("_");
		
		if (index == -1) {
			throw new RuntimeException(datasetName + " has an underscore missing");
		}
		
		String originalDatasetName = datasetName.substring("aug".length(), index);
		
		System.out.println("Original dataset name: " + originalDatasetName);
		
		File originalDataset3Directory = new File(Constants.STAGE3_BASE_DIRECTORY + "/" + originalDatasetName);
		File augStage1Directory = new File(Constants.STAGE1_BASE_DIRECTORY + "/" + datasetName);

		FileUtils.copyDirectory(originalDataset3Directory, augStage1Directory);

		new DataAugmentationTokenizer(datasetName);
		new HierarchyCollapser();
		new DatasetMerger();
		new MetadataGenerator("dw");
		new ExtraDatasetGenerator();
		new ResponseCounter();
	}

	private static void createMingenDatasets(String datasetName) throws IOException {
		File quora3Directory = new File(Constants.STAGE3_BASE_DIRECTORY + "/quora");
		File mingenStage1Directory = new File(Constants.STAGE1_BASE_DIRECTORY + "/" + datasetName);

		FileUtils.copyDirectory(quora3Directory, mingenStage1Directory, new FileFilter() {

			String[] filenames = new String[] { String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, "training"),
			    String.format(Constants.DOCUMENT_WORD_DATASET_TEXT_FILENAME_FORMAT, "training") };

			@Override
			public boolean accept(File file) {
				for (String filename : filenames) {
					if (file.getName().equals(filename)) {
						return true;
					}
				}

				return false;
			}
		});

		new MinorityVsGeneratedTokenizer(datasetName, 1);
		new DatasetPartitioner();
		new FilterPipeline();
		new HierarchyCollapser();
		new DatasetMerger();
		new MetadataGenerator("dw");
		new ExtraDatasetGenerator();
		new ResponseCounter();
		new MinorityVsGeneratedPartitioner();
	}

	private static void createGenrankDatasets(String genrankDatasetName) throws IOException {
		String datasetName = genrankDatasetName.split("_")[2];
		File datasetName3Directory = new File(Constants.STAGE3_BASE_DIRECTORY + "/" + datasetName);
		File genrankStage1Directory = new File(Constants.STAGE1_BASE_DIRECTORY + "/" + genrankDatasetName);

		FileUtils.copyDirectory(datasetName3Directory, genrankStage1Directory, new FileFilter() {

			String[] filenames = new String[] { String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, "training"),
			    String.format(Constants.DOCUMENT_WORD_DATASET_TEXT_FILENAME_FORMAT, "training"),
			    String.format(Constants.RESPONSE_DATASET_FILENAME_FORMAT, "validation"),
			    String.format(Constants.DOCUMENT_WORD_DATASET_TEXT_FILENAME_FORMAT, "validation") };

			@Override
			public boolean accept(File file) {
				for (String filename : filenames) {
					if (file.getName().equals(filename)) {
						return true;
					}
				}

				return false;
			}
		});

		new GenerationRankingTokenizer(genrankDatasetName);
		new HierarchyCollapser();
		new DatasetMerger();
		new MetadataGenerator("dw");
		new ExtraDatasetGenerator();
		new ResponseCounter();
	}
}
