import java.util.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Random;
import java.lang.Math;

class SOM {

    private int currentEpoch;
	private int dataSize; // tamanho do dado
	private double variation;
	private int maxEpochs;
	private Neuron[][] neurons;
    private int radius;
	private int size; // tamanho da rede (size x size)
	private double startLearningRate;
	private double timeConstant;
	
	private String[][] crossValidationPlaceholder;
	
	private ArrayList<ManuscriptChar> inputDataSet, crossValidationDataSet;
	
	private final static String pattern1 = "[01]{32}"; // cadeia binária de tamanho 32 
	private final static String pattern2 = "([ ][0-9]){1}"; // espaço + 1 dígito 
	
	public SOM(int size, int dataSize, int maxEpochs, double variation) {
		
		this.currentEpoch = 0;
		this.dataSize = dataSize;
		this.variation = variation;
		this.maxEpochs = maxEpochs;
		this.radius = size / 2;
		this.size = size;
		this.startLearningRate = 0.1;
		this.timeConstant = this.maxEpochs / Math.log(this.radius); // constante para o cálculo da função de vizinhança
		
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
					data.value = line.matches(pattern2) ? line.charAt(1) : '0';
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
				//System.out.println("\n\nCreating neuron " + i + " " + j + ":");
				this.neurons[i][j] = new Neuron('0', i, j);
				this.neurons[i][j].weights = new double[this.dataSize];
				for (k = 0; k < this.dataSize; k++) {
					this.neurons[i][j].weights[k] = rand.nextDouble();
					//System.out.print(k + " " + this.neurons[i][j].weights[k] + " ");
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
		// atualizar taxa de aprendizado?
		
		Random rand = new Random();
		double variation, neighborhoodStrength, data[], learningRate;
		int i;
		ArrayList<ManuscriptChar> trainingSet;
		ManuscriptChar tempManuscriptChar;
		Neuron bmu;
		
		this.currentEpoch = 0;
		variation = 1; // valor de atualização dos pesos
		
		while (variation > this.variation && this.currentEpoch++ < this.maxEpochs) {
			
			i = 0;
			variation = 1;
			trainingSet = new ArrayList<ManuscriptChar>();
			
			for (i = 0; i < this.inputDataSet.size(); i++) {
				tempManuscriptChar = new ManuscriptChar();
				tempManuscriptChar.value = this.inputDataSet.get(i).value;
				tempManuscriptChar.representation = this.inputDataSet.get(i).representation.clone();
				trainingSet.add(tempManuscriptChar);
			}
			
			while (trainingSet.size() > 0) {
				
				i =  trainingSet.size() > 1 ? rand.nextInt(trainingSet.size()-1) : 0;
				data = trainingSet.get(i).representation;
				bmu = discoverBMU(data);
				bmu.value = trainingSet.get(i).value;
				learningRate = this.learningRate(); // calcula taxa de aprendizado em função da iteração atual
				neighborhoodStrength = this.neighborhoodStrength(); // atualiza o valor da influência da função neighboorhood
				
				for (int j = 0; j < this.size; j++) {
					for (int k = 0; k < this.size; k++) {
						// atualiza os pesos com base na função neighboorhood
						variation += this.neurons[j][k].updateWeights(data, bmu, learningRate, neighborhoodStrength);
					}
				}
				trainingSet.remove(i);
			}
			variation = Math.abs(variation / (this.dataSize * this.dataSize));
			System.out.println("Epoch: " + this.currentEpoch + " Variation: " + variation); 
			
		}
		
	}
	
	private Neuron discoverBMU(double[] data)
	{
		Neuron bmu = null;
		double min = Double.MAX_VALUE, dist;
		
		for (int i = 0; i < this.size; i++) {
			for (int j = 0; j < this.size; j++) {
				dist = this.neurons[i][j].distance(data);
				//System.out.println("N " + i + "," + j + ": " + dist); 
				if (dist < min) {
					min = dist;
					bmu = this.neurons[i][j];
				}
			}
		}
		//System.out.println("N " + bmu.x + "," + bmu.y + ": " + min);
		return bmu;
	}
	
	private double neighborhoodStrength() {
		return this.radius * Math.exp(-(double)this.currentEpoch / this.timeConstant);
	}
	
	private double learningRate() {
		return Math.exp(-this.currentEpoch / this.maxEpochs) * this.startLearningRate;
	}
		
	public void test(int[] result) {
		
		this.crossValidationPlaceholder = new String[this.size][this.size];
		
		int hits = 0, errors = 0;
		ManuscriptChar item;
		int maxMapped = 0;
		Neuron mapped;
		String ph;
		String pad = "";
		int i;
		
		for (i = 0; i < this.crossValidationDataSet.size(); i++) {
			
			item = this.crossValidationDataSet.get(i);
			mapped = discoverBMU(item.representation);
			
			if (item.value == mapped.value) {
				hits++;
			}
			else {
				errors++;
			}
			
			ph = this.crossValidationPlaceholder[mapped.x][mapped.y];
			if (ph == null) {
				ph = item.value + "";
			}
			else if (ph.indexOf(item.value) < 0) {
				ph += " " + item.value;
			}
			
			if (ph.length() > maxMapped) {
				maxMapped = ph.length();
			}
			this.crossValidationPlaceholder[mapped.x][mapped.y] = ph;
			
		}
		
		for (i = 0; i < maxMapped; i++)
			pad += " ";
		
		for (i = 0; i < this.size; i++) {
			for (int j = 0; j < this.size; j++) {
				ph = this.crossValidationPlaceholder[i][j];
				if (ph == null) {
					this.crossValidationPlaceholder[i][j] = pad;
				}
				else {
					this.crossValidationPlaceholder[i][j] = ph + pad.substring(ph.length());
				}
			}
		}
		
		result[0] = hits;
		result[1] = errors;
		
	}
	
	public void printTest() {
		for (int i = 0; i < this.size; i++) {
			for (int j = 0; j < this.size; j++) {
				System.out.print(this.crossValidationPlaceholder[i][j]);
				if (j == this.size - 1)
					System.out.print("\n");
				else
					System.out.print("|");
			}
		}
	}
	
}