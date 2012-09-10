# Renan Teruo Carneiro
# Numero USP 6514157
# MAC 0460 - Aprendizagem Computacional
# Tarefa 2

from scipy.stats import *
import sys

def argmax(array):
  return array.index(max(array))

dists = ["normal", "exponencial", "uniforme"]

########################################################
# Classe Data: Armazena dados do conjunto de treinamento
########################################################

class Data:
  def __init__(self):
    # Cada posicao de raw data guarda um vetor com as amostras de cada classe
    self.raw_data = [[],[],[]]
    # Eventualmente, cada posicao de pdfs guarda tres fdps, uma para cada distribuicao, indexada pelo nome da mesma
    self.pdfs = [{},{},{}]
    # Cada posicao de mean guarda a media de cada classe; o mesmo vale para var, maximum e minimum
    self.mean = [[], [], []]
    self.var = [[], [], []]
    self.maximum = [[], [], []]
    self.minimum = [[], [], []]
  
  # Insere um novo dado na classe especificada
  def add_data(self, data, data_class):
    self.raw_data[data_class-1] += [data]
  
  # Gera media, variancia, maximo, minimo e fdps
  def calculate_extra_data(self):
    for i in xrange(0,3):
      self.mean[i] = tmean(self.raw_data[i])
      self.var[i] = tvar(self.raw_data[i])
      self.maximum[i] = max(self.raw_data[i])
      self.minimum[i] = min(self.raw_data[i])
      self.generate_pdfs(i)  

  # Gera as fdps
  def generate_pdfs(self, data_class):
    self.pdfs[data_class]["normal"] = norm(loc = self.mean[data_class], scale=self.var[data_class])
    self.pdfs[data_class]["uniforme"] = uniform(loc = self.minimum[data_class], scale = self.maximum[data_class] - self.minimum[data_class])
    self.pdfs[data_class]["exponencial"] = expon(scale = self.mean[data_class])

#######################################################
# Classe Classifier: Classifica amostras
#######################################################

class Classifier:
  def __init__(self, data, dist1, dist2, dist3):
    self.pdfs = [{},{},{}]
    self.successes = 0
    self.pdfs[0]["dist"] = data.pdfs[0][dist1]
    self.pdfs[0]["name"] = dist1
    self.pdfs[1]["dist"] = data.pdfs[1][dist2]
    self.pdfs[1]["name"] = dist2
    self.pdfs[2]["dist"] = data.pdfs[2][dist3]
    self.pdfs[2]["name"] = dist3
    self.confusion_table = [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]]
  
  # Classifica a amostra dada, escolhendo o maior valor P(x|w), e verifica se acertou
  def choose_class(self, x, data_class):
    results = []
    data_class -= 1
    for i in self.pdfs:
      results += [i["dist"].pdf(x)]
    result = argmax(results)
    if result == data_class:
      self.successes += 1
    self.add_to_confusion_table(data_class, result)
    return result

  # Retorna o numero de acertos
  def get_results(self):
    return self.successes

  # Imprime as distribuicoes usadas nesse classificador
  def print_self(self):
    print (self.pdfs[0]["name"], self.pdfs[1]["name"], self.pdfs[2]["name"])
    
  # Adiciona, na tabela de confusao, o valor esperado (i) e o valor devolvido(j)
  def add_to_confusion_table(self, i, j):
    self.confusion_table[i][j] += 1
  
  # Imprime a tabela de confusao
  def print_confusion_table(self):
    print "\t\t\t\tClassificado"
    print "\t\t\t\tClasse 1\tClasse 2\tClasse 3"
    print "\t\tClasse 1\t",self.confusion_table[0][0],"\t\t",self.confusion_table[0][1],"\t\t", self.confusion_table[0][2]
    print "Esperado\tClasse 2\t",self.confusion_table[1][0],"\t\t",self.confusion_table[1][1],"\t\t", self.confusion_table[1][2]
    print "\t\tClasse 3\t",self.confusion_table[2][0],"\t\t",self.confusion_table[2][1],"\t\t", self.confusion_table[2][2]
       
#########################################################

if len(sys.argv) < 3:
  print "Especifique o conjunto de treinamento e o conjunto de testes!"
  exit()

training_file = open(sys.argv[1])

training_data = training_file.readlines()

training_file.close()

data = Data()
  
for i in training_data:
  vals = i.split(' ')
  val = float(vals[0])
  data_class = int(vals[1])
  data.add_data(val, data_class)
    
data.calculate_extra_data()
    
classifiers = []
    
for i in dists:
  for j in dists:
    for k in dists:
      classifiers += [Classifier(data, i, j, k)]
          
test_file = open(sys.argv[2])
                  
test_data = test_file.readlines()
  
test_file.close()
  
total_cases = 0

for i in test_data:
  vals = i.split(' ')
  val = float(vals[0])
  data_class = int(vals[1])
  total_cases += 1
  for j in classifiers:
    result = j.choose_class(val, data_class)

results = []
for i in classifiers:
  i.print_self()
  results += [i.get_results()]
  print "Erros: ", i.get_results(), "\n"
  
maximum = max(results)

maxargs = []

for i in xrange(0,27):
  if results[i] == maximum:
    maxargs += [i]
    
print "Modelos escolhidos: \n"

for i in maxargs:
  classifiers[i].print_self()
  print "Erros: ", total_cases - results[i]
  classifiers[i].print_confusion_table()
  print "\n"
