import java.lang.Math;

class Neuron {

    double weights[];
    int x;
    int y;
	double distance; // a última distância calculada
	int mappedChars[];
	int lastMappedChar;
	
	
	public Neuron(int x, int y) {
		this.x = x;
		this.y = y;
		this.mappedChars = new int[10];
		for (int i = 0; i < 10; i++)
			this.mappedChars[i] = 0;
	}
	
	public double distance(double[] data) { // calcula a distância de um vetor de dados para o vetor de pesos de um neurônio
	
		double dist = 0;
		
		for (int i = 0; i < data.length; i++) {
			dist += Math.pow(data[i] - this.weights[i], 2); // raíz quadrada do somatório do quadrado de todas as diferenças data(i) - weihghts(i)
		}
		dist = Math.sqrt(dist);
		return dist;
	}
		
	private double neighborhoodInfluence(Neuron bmu, double neighborhoodStrength) {
		double distance = Math.sqrt(Math.pow(bmu.x - this.x, 2) + Math.pow(bmu.y - this.y, 2)); // distância euclideana entre o nó e o bmu 
		return Math.exp(-Math.pow(distance, 2) / (2 * Math.pow(neighborhoodStrength, 2)));
	}
	
	public double updateWeights(double[] data, Neuron bmu, double learningRate, double neighborhoodStrength) {
		
		double delta, update, neighborhoodInfluence;
		
		delta = update = 0;
		
		//System.out.println("BMU: " + bmu.x + "," + bmu.y + " NEURON: " + this.x + "," + this.y);
		
		for (int i = 0; i < this.weights.length; i++) {
			neighborhoodInfluence = this.neighborhoodInfluence(bmu, neighborhoodStrength);
			//System.out.println("W: " + i + ": " + neighborhoodInfluence);
			delta = learningRate * neighborhoodInfluence * (data[i] - this.weights[i]);
			this.weights[i] += delta;
			update += delta;
		}
		
		return update / this.weights.length;
		
	}
	
	public String dump() {
		
		String text = "";
		
		for (int i = 0; i < this.mappedChars.length; i++) {
			if (this.mappedChars[i] > 0) {
				text = text + i + "(" + this.mappedChars[i] + ") ";
			}
		}
		System.out.println(text);
		return text;
		
	}

}