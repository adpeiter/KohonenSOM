/*
 * UFFS - Universidade Federal da Fronteira Sul
 * 2017/1 - Inteligência Artificial - Trabalho 3
 * Kohonen Self Organized Maps
 * Aristides Darlan Peiter Tondolo, Maicon Ghidolin, Wagner Frana
 */

class Main {
 
	private static String FILE_NAME_TRAIN = "database/optdigits-orig.tra";
	private static String FILE_NAME_VALIDATION = "database/optdigits-orig.cv";
	private static double VARIATION = 0.000001;
	private static double START_LEARNING_RATE = 0.1;
	private static int EPOCHS = 100;
	private static int DATA_DIMENSION = 32*32;
	private static int SOM_SIZE = 15;
	private static double START_RADIUS = 0.5;
	
	private final static String ARG_MAXEPOCHS = "e:";
	private final static String ARG_SOMSIZE = "s:";
	private final static String ARG_STARTLEARNING = "l:";
	private final static String ARG_VARIATION = "v:";
	private final static String ARG_STARTRADIUS = "r:";
	

	private final static String argPattern = "[a-zA-Z]{1,2}[:](\\d{1,}|(\\d|[a-zA-Z.-_])*)";
	
    public static void main (String args[]) {
    
		int testResult[] = new int[2];
		int totalTests;
		
		if (args != null) {
			try {
				for (int i = 0; i < args.length; i++) {
					if (args[i].startsWith(ARG_MAXEPOCHS)) // máximo de épocas
						EPOCHS = Integer.parseInt(args[i].substring(ARG_MAXEPOCHS.length()));
					else if (args[i].startsWith(ARG_SOMSIZE)) // tamanho da camada de saída (rede será n x n)
						SOM_SIZE = Integer.parseInt(args[i].substring(ARG_SOMSIZE.length()));
					else if (args[i].startsWith(ARG_STARTLEARNING)) // taxa de aprendizado inicial (padrão 0.1)
						START_LEARNING_RATE = Double.parseDouble(args[i].substring(ARG_STARTLEARNING.length()));
					else if (args[i].startsWith(ARG_VARIATION)) // variação máxima (valor da alteração dos pesos entre épocas)
						VARIATION = Integer.parseInt(args[i].substring(ARG_VARIATION.length()));
					else if (args[i].startsWith(ARG_STARTRADIUS)) // coeficiente da distância inicial da função de vizinhança, em relação ao tamanho da rede (ex: 0.5 será o raio)
						START_RADIUS = Double.parseDouble(args[i].substring(ARG_STARTRADIUS.length()));
					
				}
			}
			catch (Exception ex) {
				System.out.println("Read parameters fail...");
				System.exit(1);
			}
		}
				
		SOM som = new SOM(SOM_SIZE, DATA_DIMENSION, EPOCHS, VARIATION, START_LEARNING_RATE, START_RADIUS);
		
		som.readTrainingDataSet(FILE_NAME_TRAIN);
		som.initialiseNeurons();
		som.train();
		som.readCrossValidadionDataSet(FILE_NAME_VALIDATION);
		som.test(testResult);
		som.dumpTest();
        
		totalTests = testResult[0] + testResult[1];
		System.out.println("Hits: " + testResult[0] + " de " + totalTests + " rate: " + ((double)testResult[0] / (double)totalTests)); 
		
    }
}