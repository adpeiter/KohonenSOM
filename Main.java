/*
 * Kohonen SOM
 * 2017/1 - IA - T3
 * Aristides Darlan Peiter Tondolo, Maicon Ghidolin, Wagner Frana
 */
package kohonen;
class Main {
 
    public static void main (String args[]) {
        
        String fileNameTrain, fileNameTest;

        if (args) {
            for (int i = 0; args.length; i++) {
                if (args[i].equals("-t"))
                    fileNameTrain = args[i+1];
                else if (args[i].equals("-v"))
                    fileNameTest = args[i+1];
            }
        }

        if (!fileNameTrain)
            System.out.println("Faltando parâmetro -t (dados para treinamento).");
        if (!fileNameTest)
            System.out.println("Faltando parâmetro -v (dados para validação).");
        
    }
}