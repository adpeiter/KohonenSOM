import java.util.*;
import java.util.stream.*;
import java.io.*;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;
import java.lang.Math;
import java.awt.Desktop;
import java.net.URI;

class SOM {

	private static final String STREMPTY = "";
	private static final String BLKSPACE = " ";
	private static final String UNDERSCORE = "_";

    private int currentEpoch;
	private int dataSize; // tamanho do dado
	private double variation;
	private int maxEpochs;
	private Neuron[][] neurons;
	private double startRadius;
	private int size; // tamanho da rede (size x size)
	private double startLearningRate;
	private double timeConstant;
	private long totalTrainTime;
	
	private String[][] crossValidationPlaceholder;
	private ArrayList<ManuscriptChar> inputDataSet, crossValidationDataSet;
	
	private final static String pattern1 = "[01]{32}"; // cadeia binária de tamanho 32 
	private final static String pattern2 = "([ ][0-9]){1}"; // espaço + 1 dígito 
	private final static String pattern3 = "(\\d\\(\\d*\\)){1}"; // espaço + 1 dígito 
	
	private int[] examples;
	private int[] errors;
	private int[] hits;
	private float[] hitRate;

	public SOM(int size, int dataSize, int maxEpochs, double variation, double startLearningRate, double startRadius) {
		
		this.currentEpoch = 0;
		this.dataSize = dataSize;
		this.variation = variation;
		this.maxEpochs = maxEpochs;
		this.startRadius = startRadius;
		this.size = size;
		this.startLearningRate = startLearningRate;
		this.timeConstant = this.maxEpochs / Math.log(this.size / 2); // constante para o cálculo da função de vizinhança
		
		initialiseNeurons();
		
	}
	
	public void readTrainingDataSet(String file) {
		this.inputDataSet = new ArrayList<ManuscriptChar>();
		readFile(file, this.inputDataSet);
	}
	
	public void readCrossValidadionDataSet(String file) {
		this.crossValidationDataSet = new ArrayList<ManuscriptChar>();
		readFile(file, this.crossValidationDataSet);
	}
	
	public void readFile(String file, ArrayList<ManuscriptChar> dataSet) {
		
		BufferedReader br;
		StringBuilder sb;
		String line;
		ManuscriptChar data;
		double charRepresentation[];
		int i, j;
		
		try {
			
			br = new BufferedReader(new FileReader(file));
			sb = new StringBuilder();
			
			while (true) {
								
				while (true) {
					line = br.readLine();
					if (line == null)
						return;
					if (!line.matches(pattern1))
						break;
					sb.append(line);
				}
				
				if (sb.length() > 0) {
					
					charRepresentation = new double[this.dataSize];
					
					for (i = 0; i < sb.length(); i++)
						charRepresentation[i] = sb.charAt(i) == '0' ? 0 : 1;
					
					sb.setLength(0);
					data = new ManuscriptChar();
					data.value = line.matches(pattern2) ? Integer.parseInt(line.charAt(1) + STREMPTY) : 0;
					data.representation = charRepresentation;
					dataSet.add(data);
					
				}
				
			}
			
		}
		catch (Exception ex) {
			System.out.println("Fail to read data file: ");
			ex.printStackTrace();
			dataSet = null;
		}
		
	}
	
	public void initialiseNeurons() {
		
		Random rand = new Random();
		int i, j, k, l;
		
		// os neurônios da 2ª camada são uma matriz quadrada (incialmente 10)
		this.neurons = new Neuron[this.size][this.size];
		for (i = 0; i < this.size; i++) {
			for (j = 0; j < this.size; j++) {
				this.neurons[i][j] = new Neuron(i, j);
				this.neurons[i][j].weights = new double[this.dataSize];
				for (k = 0; k < this.dataSize; k++) {
					this.neurons[i][j].weights[k] = rand.nextDouble();
				}
			}
		}
	}
	
	public void train() {
		
		// o treinamento é repetido pra cada entrada, uma por vez, para aprender generalizar
		// escolher randomicamente um representante do dataset
		// comparar os valores com os vetores de pesos de todos os neurônios
		// definir o vetor vencedor
		// ajustar os pesos do vencedor e dos vizinhos adjacentes pela função neighboorhood
		// atualizar função neighboorhood
		// atualizar taxa de aprendizado
		// repetir até o máximo de épocas ou mínimo de variação
		
		long msStartTrain = 0, msEndTrain = 0, msStartEpoch = 0, msEndEpoch = 0;
		double variation, neighborhoodStrength, data[], learningRate;
		int i;
		ArrayList<ManuscriptChar> trainingSet;
		ManuscriptChar tempManuscriptChar;
		Neuron bmu;
		Random rand = new Random();
		
		this.currentEpoch = 0;
		variation = 1; // valor de atualização dos pesos
		
		msStartTrain = System.currentTimeMillis();

		while (this.currentEpoch++ < this.maxEpochs && variation > this.variation) {
			
			msStartEpoch = System.currentTimeMillis();
			
			i = 0;
			variation = 0;
			trainingSet = new ArrayList<ManuscriptChar>();
			
			for (i = 0; i < this.inputDataSet.size(); i++) {
				tempManuscriptChar = new ManuscriptChar();
				tempManuscriptChar.value = this.inputDataSet.get(i).value;
				tempManuscriptChar.representation = this.inputDataSet.get(i).representation.clone();
				trainingSet.add(tempManuscriptChar);
			}
			
			learningRate = this.learningRate(); // calcula taxa de aprendizado em função da iteração atual
			neighborhoodStrength = this.neighborhoodStrength(); // atualiza o valor da influência da função neighboorhood
			
			while (trainingSet.size() > 0) {
				
				i =  trainingSet.size() > 1 ? rand.nextInt(trainingSet.size()-1) : 0;
				data = trainingSet.get(i).representation;
				bmu = discoverBMU(data);
				bmu.mappedChars[trainingSet.get(i).value]++;
				bmu.lastMappedChar = trainingSet.get(i).value;
				
				for (int j = 0; j < this.size; j++) {
					for (int k = 0; k < this.size; k++) {
						// atualiza os pesos com base na função neighboorhood e acumula a variação aplicada
						variation += this.neurons[j][k].updateWeights(data, bmu, learningRate, neighborhoodStrength);
					}
				}
				trainingSet.remove(i);
			}
			variation = Math.abs(variation / Math.pow((double)this.dataSize, 2));
			msEndEpoch = System.currentTimeMillis();
			
			System.out.println("E: " + this.currentEpoch + " L: " + learningRate + " N: " + neighborhoodStrength +
				" V: " + variation + " T: " + (msEndEpoch - msStartEpoch) + " ms"); 
			
		}
		
		msEndTrain = System.currentTimeMillis();
		this.totalTrainTime = msEndTrain - msStartTrain;
		System.out.println("Totail training time: " + this.totalTrainTime + " s");
		
	}
	
	private Neuron discoverBMU(double[] data) {
		
		Neuron bmu = null;
		double min = Double.MAX_VALUE;
		
		for (int i = 0; i < this.size; i++) {
			for (int j = 0; j < this.size; j++) {
				this.neurons[i][j].distance = this.neurons[i][j].distance(data);
				if (this.neurons[i][j].distance < min) {
					min = this.neurons[i][j].distance;
					bmu = this.neurons[i][j];
				}
			}
		}
		
		return bmu;
		
	}
	
	private double neighborhoodStrength() {
		return (double)this.size * this.startRadius * Math.exp(-(double)this.currentEpoch / this.timeConstant);
	}
	
	private double learningRate() {
		return Math.exp(-(double)this.currentEpoch / (double)this.maxEpochs) * this.startLearningRate;
	}
	
	public int totalErrors() {
		return IntStream.of(this.errors).sum();
	}
	
	public int totalHits() {
		return IntStream.of(this.hits).sum();
	}
	
	public float totalHitRate() {
		return (float)this.totalHits() / (float)(this.totalErrors() + this.totalHits());
	}
	
	public String totalHitRate3f() {
		return String.format("%.3f", this.totalHitRate() * 100);
	}
	
	public void test() {
		
		this.crossValidationPlaceholder = new String[this.size][this.size];
		this.errors = new int[10];
		this.hits = new int[10];
		this.examples = new int[10];
		this.hitRate = new float[10];
		
		int i, j;
		ManuscriptChar item;
		Neuron mapped;
		String label, phLabel;
		
		for (i = 0; i < 10; i++) {
			this.hits[i] = 0;
			this.errors[i] = 0;
		}
		
		for (i = 0; i < this.crossValidationDataSet.size(); i++) {
			
			item = this.crossValidationDataSet.get(i);
			mapped = discoverBMU(item.representation);
			
			mapped.tested[item.value]++;
			
			if (mapped.lastMappedChar == item.value) {
				this.hits[item.value]++;
			}
			else {
				this.errors[item.value]++;
			}
			
		}
		
		for (i = 0; i < 10; i++) {
			this.examples[i] = this.hits[i] + this.errors[i];
			this.hitRate[i] = (float)this.hits[i] / (float)this.examples[i];
			this.hitRate[i] *= 100;
		}
		
	}
	
	public void dumpTest() {
		
		StringBuilder sb = new StringBuilder();
		String phValue, cssClass, dumpFileName, tdTitle;
		int i, j, k;
		
		dumpFileName = "dump/S" + this.size + "_E" + this.maxEpochs + "_L" + (this.startLearningRate  + "_R" + this.startRadius).replace(".", "d") + ".html";
		
		sb.append("<!DOCTYPE html>");
		sb.append("<html>");
		sb.append("<head>");
		sb.append("<meta charset=\"utf-8\"><title>Dump Kohonen SOM</title>");
		sb.append("<link rel=\"stylesheet\" href=\"css/style.css\"></link>");
		sb.append("</head>");
				  
		sb.append("<body>");
		sb.append("<h2>");
		sb.append("SOM properties - Size: " + this.size + " nodes, Epochs: " + this.maxEpochs + ", Variation: " + this.variation + ", Start Learning Rate:" + this.startLearningRate + ", Radius: " + this.startRadius);
		sb.append("<br/>Total train time: " + this.totalTrainTime + " ms");
		sb.append("<br/>Total hit rate: " + this.totalHitRate3f());
		sb.append("</h2>");
		sb.append("<hr/>");
		sb.append("<table>");
		sb.append("<tbody>");
		sb.append("<tr>");
		sb.append("<td>VALUE</td>");
		
		for (i = 0; i < 10; i++) {
			sb.append("<td class=\"bg" + i + " text-white\">" + i + "</td>");
		}
		
		sb.append("<td>TOTAL</td>");
		sb.append("</tr>");
		sb.append("<tr>");
		sb.append("<td>HITS</td>");
		
		for (i = 0; i < 10; i++) {
			sb.append("<td class=\"bg" + i + " text-white\">" + this.hits[i] + "</td>");
		}
		sb.append("<td>" + this.totalHits() + "</td>");
		sb.append("</tr>");
		
		sb.append("<tr>");
		sb.append("<td>ERRORS</td>");
		
		for (i = 0; i < 10; i++) {
			sb.append("<td class=\"bg" + i + " text-white\">" + this.errors[i] + "</td>");
		}
		sb.append("<td>" + this.totalErrors() + "</td>");
		
		sb.append("</tr>");
		sb.append("<tr>");
		sb.append("<td>HIT RATE</td>");
		
		for (i = 0; i < 10; i++) {
			sb.append("<td class=\"bg" + i + " text-white\">" + String.format("%.3f", this.hitRate[i]) + "%</td>");
		}
		sb.append("<td>" + this.totalHitRate3f() + "%</td>");
		
		sb.append("</tr>");
		sb.append("</tbody>");
		sb.append("</table>");
		
		sb.append("<hr/>");
		sb.append("<input type=\"checkbox\" id=\"ckbChangeColorErrorNodes\" onchange=\"startStopAlternateClass(this.checked)\" />");
		sb.append("<label for=\"ckbChangeColorErrorNodes\"> Change color error nodes</label>");		
		
		sb.append("<table><tbody>");
			
		for (i = 0; i < this.size; i++) {
			sb.append("<tr> ");
			for (j = 0; j < this.size; j++) {
				
				phValue = STREMPTY;
				tdTitle = STREMPTY;
				
				for (k = 0; k < 10; k++) {
					if (this.neurons[i][j].tested[k] > 0) {
						phValue += k;
						if (tdTitle.length() > 0)
							tdTitle += ", ";
						tdTitle += k + ":" + this.neurons[i][j].tested[k];
					}
				}
				
				if (phValue == STREMPTY) {
					cssClass = "empty";
					phValue = "-";
				}
				else if (phValue.length() > 1) {
					cssClass = "error";
				}
				else {
					cssClass = "bg" + phValue.charAt(0) + " text-white";
				}
				sb.append("<td id=\"" + i + UNDERSCORE + j + "\" class=\"" + cssClass + "\" title=\"" + tdTitle + "\">" + phValue + "</td> ");
			}
			sb.append("</tr> ");
		}
		
		sb.append("</tbody></table>");
		sb.append("<script type=\"text/javascript\" src=\"js/script.js\"></script>");
		sb.append("</body>");
		sb.append("</html>");
		
		try {
			
			PrintWriter writer = new PrintWriter(dumpFileName, "UTF-8");			
			writer.print(sb.toString());
			writer.close();
			
			File htmlFile = new File(dumpFileName);
			Desktop.getDesktop().browse(htmlFile.toURI());
			
		}
		catch (Exception ex) {
			System.out.println("Fail to open dump: ");
			ex.printStackTrace();
		}
		
	}
	
}