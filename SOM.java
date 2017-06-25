import java.util.*;
import java.util.File;
import java.util.Scanner;

package kohonen;

class SOM {

    private Neuron[][] neurons;
    private int currentIteration;
    private int maxIterations;
	private int dataDimension; // dimensão do dado
	private int databaseSize; // quantidade de registros
	private int[][] database;
	private double error;
	
	public void readDatabase(String file) {
		
		Scanner scanner = new Scanner(new File(file));
		
		for (i = 0; i < this.databaseSize - 1; i++) {
			this.database[i] = scanner.readLine().split(",");
		}
		
	}
	
	public SOM(int dataDimension, int databaseSize) {
		
		 // os neurônios da 2ª camada são uma matriz quadrada com a dimensão do dado (64 no caso da base de dados fornecida pelo professor)
		this.neurons = new Neuron[dataDimension][dataDimension];
		this.database = new int[databaseSize][dataDimension];
		this.currentIteration = 0;
		this.limitIteration = limitIteration;
		this.error = error;
		
	}
	
	public void defMaxIterations(int maxIterations) {
		this.maxIterations = maxIterations;
	}
	
	public void defError(int error) {
		this.error = error;
	}
	
	
	public void Train() {
		
		// o treinamento é repetido pra cada entrada, uma por vez, para aprender generalizar
		// escolher randomicamente um representante do database
		// comparar os valores com os vetores de pesos de todos os neurônios
		// definir o vetor vencedor
		// ajustar os pesos do vencedor e dos vizinhos adjacentes pela função neighboorhood
		// atualizar função neighboorhood
		// atualizar taxa de aprendizado?
		
	}
	
	public void print() {
		
	}
	
}