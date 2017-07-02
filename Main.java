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
	private static int MAX_ITERATIONS = 5;
	private static int DATA_DIMENSION = 32*32;
	private static int SOM_SIZE = 10;
	
	private final static String argIteration = "i:";
	private int argIx;
	
	private final static String argPattern = "[a-zA-Z]{1,2}[:](\\d{1,}|(\\d|[a-zA-Z.-_])*)";
	
    public static void main (String args[]) {
        
		if (args != null) {
			try {
				for (int i = 0; i < args.length; i++) {
					if (args[i].startsWith(argIteration))
						MAX_ITERATIONS = Integer.parseInt(args[i].substring(argIteration.length()));
				}
			}
			catch (Exception ex) {
				System.out.println("Read parameters fail...");
			}
		}
		
		System.out.println(MAX_ITERATIONS);
		
		SOM som = new SOM(SOM_SIZE, DATA_DIMENSION, MAX_ITERATIONS, ERROR);
		
		som.readDataSet(FILE_NAME_TRAIN);
		som.initialiseNeurons();
		som.train();
		        
    }
}