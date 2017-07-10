import java.util.*;
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
			System.out.println("Crash on read data file...");
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

		while (this.currentEpoch++ < this.maxEpochs) {
			
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
		System.out.println("Tempo total do treinamento: " + this.totalTrainTime + " s");
		
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
	
	public void test(int[] result) {
		
		this.crossValidationPlaceholder = new String[this.size][this.size];
		
		int hits = 0, errors = 0, i = 0;
		ManuscriptChar item;
		Neuron mapped;
		String label;
		
		for (i = 0; i < this.crossValidationDataSet.size(); i++) {
			
			item = this.crossValidationDataSet.get(i);
			mapped = discoverBMU(item.representation);
			
			if (mapped.lastMappedChar == item.value) {
				hits++;
			}
			else {
				errors++;
			}
			
			label = item.value + STREMPTY;
			
			if (this.crossValidationPlaceholder[mapped.x][mapped.y] == null) {
				this.crossValidationPlaceholder[mapped.x][mapped.y] = label;
			}
			else if (this.crossValidationPlaceholder[mapped.x][mapped.y].indexOf(label) < 0) {
				this.crossValidationPlaceholder[mapped.x][mapped.y] += label;
			}
						
		}
		
		result[0] = hits;
		result[1] = errors;
		
	}
	
	public void dumpTest() {
		
		StringBuilder sb = new StringBuilder();
		String phValue, cssClass, dumpFileName;
		
		dumpFileName = "E_" + this.maxEpochs + "_S_" + this.size + "_E_" + this.variation + "_" + System.currentTimeMillis() + ".html";
		
		sb.append("<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Dump Kohonen SOM</title><style>");
		sb.append("td { text-align: center; width: 30px; } ");
		sb.append(".empty { background-color: #FFF; color: #000; } ");
		sb.append(".error { background-color: #FFF; color: #000; text-decoration: underline; } ");
		sb.append(".bg0 { background-color: #000; color: #FFF; } ");
		sb.append(".bg1 { background-color: #0099CC; color: #FFF; } ");
		sb.append(".bg2 { background-color: #339933; color: #FFF; } ");
		sb.append(".bg3 { background-color: #996633; color: #FFF; } ");
		sb.append(".bg4 { background-color: #FF9933; color: #FFF; } ");
		sb.append(".bg5 { background-color: #e63900; color: #FFF; } ");
		sb.append(".bg6 { background-color: #ff66cc; color: #FFF; } ");
		sb.append(".bg7 { background-color: #9900cc; color: #FFF; } ");
		sb.append(".bg8 { background-color: #8585ad; color: #FFF; } ");
		sb.append(".bg9 { background-color: #0039e6; color: #FFF; } ");
		sb.append("</style></head><body><h2>");
		sb.append("SOM parameters - Size: " + this.size + " nodes, Epochs: " + this.maxEpochs + ", Variation: " + this.variation); 
		sb.append("<br/>Total train time: " + this.totalTrainTime + " ms</h2>");
		sb.append("<table> <tbody> ");
			
		for (int i = 0; i < this.size; i++) {
			sb.append("<tr> ");
			for (int j = 0; j < this.size; j++) {
				phValue = this.crossValidationPlaceholder[i][j];
				if (phValue == null) {
					phValue = STREMPTY;
					cssClass = "empty";
				}
				else if (phValue.length() > 1) {
					cssClass = "error";
				}
				else {
					cssClass = "bg" + phValue;
				}
				
				sb.append("<td id=\"" + i + UNDERSCORE + j + "\" class=\"" + cssClass + "\">" + phValue + "</td> ");
			}
			sb.append("</tr> ");
		}
		
		sb.append("</tbody></table><hr/>");
		sb.append("Refresh <span id=\"spnErrorNodeCount\"></span> class error node interval: <select id=\"sltInterval\">");
		sb.append("<option value=\"2000\">2 sec.</option><option value=\"3000\">3 sec.</option>");
		sb.append("<option value=\"5000\">5 sec.</option><option value=\"10000\">10 sec.</option>");
		sb.append("</select></body>");

		sb.append("<script type=\"text/javascript\">");
		sb.append("  var cellsError;");
		sb.append("  var ids = [];");
		sb.append("  var values = [];");
		sb.append("  cellsError = document.getElementsByClassName(\"error\");");
		sb.append("  for (i = 0; i < cellsError.length; i++) {");
		sb.append("	   ids.push(cellsError[i].id);");
		sb.append("	   values.push(cellsError[i].innerText);");
		sb.append("  };" );
		sb.append("  document.getElementById(\"spnErrorNodeCount\").innerText = ids.length;");
		sb.append("  for (i = 0; i < ids.length; i++) alternateClass(i, 0);");
		sb.append("  function alternateClass(i, j) {");
		sb.append("	   if (j >= values[i].length) j = 0;");
		sb.append("	   var elem = document.getElementById(ids[i]);");
		sb.append("	   elem.className = \"bg\" + values[i][j];");
		sb.append("	   setTimeout(function () { alternateClass(i, ++j) }, document.getElementById(\"sltInterval\").value);");
		sb.append("  }");
		sb.append("</script></html>");
		
		try {
			
			PrintWriter writer = new PrintWriter(dumpFileName, "UTF-8");			
			writer.print(sb.toString());
			writer.close();
			Desktop.getDesktop().browse(new URI(dumpFileName));
			
		}
		catch (Exception ex) {
			System.out.println("Fail to open dump...");
		}
		
	}
	
}