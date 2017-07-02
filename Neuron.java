import java.lang.Math;

class Neuron {

    double weights[];
    int x;
    int y;
	double value;
	char label;
	
	public Neuron(char label, int x, int y) {
		this.label = label;
		this.x = x;
		this.y = y;
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
		return Math.exp(-Math.pow(distance, 2) / (Math.pow(neighborhoodStrength, 2)));
	}
	
	public double updateWeights(double[] data, Neuron bmu, double learningRate, double neighborhoodStrength) {
		
		double delta, update;
		
		delta = update = 0;
		
		for (int i = 0; i < this.weights.length; i++) {
			delta = learningRate * neighborhoodInfluence(bmu, neighborhoodStrength) * (data[i] - this.weights[i]);
			this.weights[i] += delta;
			update += delta;
		}
		
		return update;
		
	}

}