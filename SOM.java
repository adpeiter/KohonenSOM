import java.util.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Random;
import java.lang.Math;

class SOM {

    private int currentIteration;
	private int dataSize; // tamanho do dado
	private double error;
	private int maxIterations;
	private Neuron[][] neurons;
    private int radius;
	private int size; // tamanho da rede (size x size)
	private double startLearningRate;
	private double timeConstant;
	
	private ArrayList<double[]> dataSet;
	
	private final static String pattern1 = "[01]{32}"; // cadeia binária de tamanho 32 
	private final static String pattern2 = "([ ][0-9]){1}"; // espaço + 1 dígito 
	
	public SOM(int size, int dataSize, int maxIterations, double error) {
		
		this.currentIteration = 0;
		this.dataSize = dataSize;
		this.error = error;
		this.maxIterations = maxIterations;
		this.radius = size / 2;
		this.size = size;
		this.startLearningRate = 0.1;
		this.timeConstant = this.maxIterations / Math.log(this.radius); // constante para o cálculo da função de vizinhança
		
		initialiseNeurons();
		
	}
	
	public void readDataSet(String file) {
		
		BufferedReader br;
		StringBuilder sb;
		String line;
		double data[];
		int i, j;
		
		try {
			
			br = new BufferedReader(new FileReader(file));
			this.dataSet = new ArrayList<double[]>();
			sb = new StringBuilder();
			
			while (true) {
								
				while (true) {
					line = br.readLine();
					//System.out.println(line);
					if (line == null)
						return;
					if (!line.matches(pattern1))
						break;
					sb.append(line);
				}
				
				if (sb.length() > 0) {
				
					data = new double[this.dataSize];
					
					for (i = 0; i < sb.length(); i++)
						data[i] = sb.charAt(i) == '0' ? 0 : 1;
					
					sb.setLength(0);
					this.dataSet.add(data);
					
				}
				
			}
			
		}
		catch (Exception ex) {
			System.out.println("Crash on read data set...");
			this.dataSet = null;
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
		double error, neighborhoodStrength, data[], learningRate;
		int i;
		ArrayList<double[]> trainingSet;
		Neuron bmu;

		this.currentIteration = 0;
		error = 1; // valor de atualização dos pesos
		
		System.out.println("Iterações: " + this.currentIteration + "/" + this.maxIterations);
		System.out.println("Erro: " + error + "/" + this.error);
		
		while (error > this.error && this.currentIteration++ < this.maxIterations) {
			
			i = 0;
			error = 1;
			trainingSet = new ArrayList<double[]>();
			
			for (i = 0; i < this.dataSet.size(); i++) {
				trainingSet.add(this.dataSet.get(i).clone());
			}
			
			while (trainingSet.size() > 0) {
				
				i =  trainingSet.size() > 1 ? rand.nextInt(trainingSet.size()-1) : 0;
				data = trainingSet.get(i);
				bmu = discoverBMU(data);
				
				learningRate = this.learningRate(); // calcula taxa de aprendizado em função da iteração atual
				neighborhoodStrength = this.neighborhoodStrength(); // atualiza o valor da influência da função neighboorhood
				
				for (int j = 0; j < this.size; j++) {
					for (int k = 0; k < this.size; k++) {
						// atualiza os pesos com base na função neighboorhood
						error += this.neurons[j][k].updateWeights(data, bmu, learningRate, neighborhoodStrength);
					}
				}
				trainingSet.remove(i);
			}
			error = Math.abs(error / (this.dataSize * this.dataSize));
			System.out.println("Error: " + error); 
			
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
		return this.radius * Math.exp(-(double)this.currentIteration / this.timeConstant);
	}
	
	private double learningRate() {
		return Math.exp(-this.currentIteration / this.maxIterations) * this.startLearningRate;
	}
	
	public void print() {
		
	}
	
	public void Test() {
		
	}
	
}