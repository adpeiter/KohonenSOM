Para rodar o programa basta compilar os arquivos .java e, em seguida, executar a classe Main, com os comandos básicos para java e parâmetros opcionais de execução:
Compilação: javac *.java;
Execução: java Main [e: ][s: ][l: ][r: ][v: ]

Os parâmetros opcionais são: quantidade de épocas (e), tamanho da rede (s), taxa inicial de aprendizado (l), tamanho do raio inicial para a função de vizinhança (r) e variação máxima (v).

No parâmetro s, deve ser informado apenas o tamanho de uma dimensão, visto que a rede é uma matriz quadrada. No parâmetro r deve ser informado um valor a ser multiplicado pelo tamanho da rede. Por exemplo, o valor 0.5 fará com que o raio inicial seja metade do tamanho da rede, neste caso, o raio propriamente dito da matriz. Nos parâmetros e, l e v deve ser indicado o valor absoluto desejado para estas propriedades.