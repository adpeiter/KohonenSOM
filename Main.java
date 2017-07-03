/*
 * UFFS - Universidade Federal da Fronteira Sul
 * 2017/1 - Inteligência Artificial - Trabalho 3
 * Kohonen Self Organized Maps
 * Aristides Darlan Peiter Tondolo, Maicon Ghidolin, Wagner Frana
 */

class Main {
 
	private static String FILE_NAME_TRAIN = "database/optdigits-orig.tra";
	private static String FILE_NAME_VALIDATION = "database/optdigits-orig.cv";
	private static double ERROR = 0.000001;
	private static int EPOCHS = 100;
	private static int DATA_DIMENSION = 32*32;
	private static int SOM_SIZE = 15;
	
	private final static String argIteration = "e:";
	private final static String argSomSize = "s:";
	
	private final static String argPattern = "[a-zA-Z]{1,2}[:](\\d{1,}|(\\d|[a-zA-Z.-_])*)";
	
    public static void main (String args[]) {
    
		int testResult[] = new int[2];
		int totalTests;
		
		if (args != null) {
			try {
				for (int i = 0; i < args.length; i++) {
					if (args[i].startsWith(argIteration))
						EPOCHS = Integer.parseInt(args[i].substring(argIteration.length()));
					else if (args[i].startsWith(argSomSize))
						SOM_SIZE = Integer.parseInt(args[i].substring(argSomSize.length()));
				}
			}
			catch (Exception ex) {
				System.out.println("Read parameters fail...");
			}
		}
				
		SOM som = new SOM(SOM_SIZE, DATA_DIMENSION, EPOCHS, ERROR);
		
		som.readTrainingDataSet(FILE_NAME_TRAIN);
		som.initialiseNeurons();
		som.train();
		som.readCrossValidadionDataSet(FILE_NAME_VALIDATION);
		som.test(testResult);
		som.printTest();
        
		totalTests = testResult[0] + testResult[1];
		System.out.println("Hits: " + testResult[0] + " de " + totalTests + " rate: " + ((double)testResult[0] / (double)totalTests)); 
		
    }
}