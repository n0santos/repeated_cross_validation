'''
Autor: Nicolau Santos
Disciplina: INF1017/CMP263

Implementação da validação cruzada repetida para avaliação de modelos de classificação
'''
import numpy as np

# Function that normalizes an array based on its maximum and minimum value
def normalize(v):
    norm_v = np.zeros(v.shape)
    maximo = max(v)
    minimo = min(v)
    for i in range(len(v)):
        norm_v[i] = (v[i] - minimo)/(maximo - minimo)
    return norm_v

# Function that normalizes a matrix by column
def normalize_matrix(matrix):
    norm_matrix = np.zeros(matrix.shape)
    i = 0
    for linha in np.transpose(matrix):
        np.transpose(norm_matrix)[i] = normalize(linha)
        i += 1
    return norm_matrix

# Função que calcula a diferença quadrática entre 2 valores
def diference(a,b):
    return (a-b)**2

# Função que calcula a distância euclidiana entre 2 vetores
def euclidiana(a,b):
    sum = 0
    for i in range(len(a)-1):
        sum += diference(a[i],b[i])
    return sum**0.5

# Function that counts number of occurences of a value in an array
def how_many(x,array):
    count = 0
    for i in array:
        if i == x:
            count += 1
    return count

# Function that performs a stratified division of a matrix
def proportional_division(k, matrix):
    '''
        Input:
            - k: number of folds
            - matrix: sorted matrix to be divided
        Output:
            - 3D matrix of matrices resulted from division
    '''
    result_matrix = np.zeros((int(matrix.shape[0]/k)+1, matrix.shape[1], k))
    n = 0
    for i in range(k):
        m = 0
        for j in range(n,matrix.shape[0],k):
            result_matrix[m,:,i] = matrix[j,:]
            m += 1
        n += 1
    return result_matrix

# Function that returns a matrix with non-zero lines
def no_zero_lines(matrix):
    return matrix[~(matrix==0).all(1)]

# Function that returns the mean average of an array of values
def media(vetor):
      sum = 0
      for i in vetor:
             sum += i
      media = sum/len(vetor)
      return media

# Function that returns the standard deviation of an array of values
def desvio_padrao(vetor):
      media = media(vetor)
      sum = 0
      for i in vetor:
           sum += (i - media)**2
      desvio_padrao = (sum/len(vetor))**0.5
      return desvio_padrao

# Function that creates k-folds using stratified division of the dataset
def stratified_k_folds(k, dataset):
    '''
        Input:
            - k: number of folds
            - dataset: csv file with datavalues and attributes
        Output:
            - 3D matrix with k-fold
    '''

    data = np.loadtxt(open(dataset, "rb"), delimiter=",", skiprows=1)

    sorted_data = data_shuffler(data) # sorting the matrix according to classes and shuffling rows

    norm_sorted_data = normalize_matrix(sorted_data) # normalizing sorted dataset by column

    k_folds = proportional_division(k, norm_sorted_data) # making a stratified division of instances

    return k_folds

# Function that shuffles the lines of a matrix leaving it sorted by classes
def data_shuffler(matrix):
    '''
        Input:
            - matrix: normalized matrix
        Output:
            - shuffled matrix sorted by classes
    '''
    sorted_matrix = matrix[matrix[:,-1].argsort()]
    classes = set(sorted_matrix[:,-1])
    division_indices = []
    for i in range(sorted_matrix.shape[0]-1):
        if sorted_matrix[i,-1] != sorted_matrix[i+1,-1]:
            division_indices.append(i+1)
    if len(division_indices) == 1:
        np.random.shuffle(sorted_matrix[:division_indices[0],:])
        np.random.shuffle(sorted_matrix[division_indices[0]:,:])
    else:
        for i in range(len(division_indices)):
            if i == 0: 
                np.random.shuffle(sorted_matrix[:division_indices[i+1],:])
            elif i == len(division_indices) - 1:
                np.random.shuffle(sorted_matrix[division_indices[i]:,:])
            else:
                np.random.shuffle(sorted_matrix[division_indices[i]:division_indices[i+1],:])
    return sorted_matrix

# Function that returns a tuple with training fold indices
def training_folds(k,i):
    training_folds = []
    for j in range(k):
        if j == i:
            continue
        training_folds.append(j)
    return tuple(training_folds)

# Function that creates test and training datasets for the entire cross-validation process
def knn(i, k_folds):
    '''
        Input:
            - i: index of repetition during cross-validation
            - k_folds: 3d matrix with k-folds
        Output:
            - accuracy: number of correct classifications / number of total classifications 
            - F-1 score: measure of the test accuracy based on precision and recall values
    '''    
    test_data = no_zero_lines(k_folds[:,:,i])

    k = k_folds.shape[2]

    k_neighbours = 5

    training_partitions = training_folds(k,i)

    for x in range(len(training_partitions)):
        if x == 0:
            training_data = no_zero_lines(k_folds[:,:,training_partitions[x]])
        else:
            training_data = np.concatenate((training_data,no_zero_lines(k_folds[:,:,training_partitions[x]])))
    
    distances = np.zeros([training_data.shape[0],test_data.shape[0]])
    for x in range(test_data.shape[0]):
        for y in range(training_data.shape[0]):
            distances[y,x] = euclidiana(test_data[x,:],training_data[y,:])

    k_nearest_neighbours = np.zeros([k_neighbours,training_data.shape[1],test_data.shape[0]])
    k_nearest = np.zeros((distances.shape[0]))

    for x in range(distances.shape[1]):
        for y in range(k_neighbours):
                k_nearest = distances[:,x].argsort()[:k_neighbours]
                k_nearest_neighbours[y,:,x] = training_data[k_nearest[y],:]

    predictions = np.zeros((k_nearest_neighbours.shape[2]))

    for x in range(k_nearest_neighbours.shape[2]):
        count_zeros = 0
        count_ones = 0
        for instance in k_nearest_neighbours[:,:,x]:
            if instance[-1] == 0:
                count_zeros += 1
            else:
                count_ones += 1
        if count_zeros > count_ones:
            predictions[x] = 0
        else:
            predictions[x] = 1

    corrects_count = 0
    wrongs_count = 0
    for x in range(predictions.shape[0]):
        if predictions[x] == test_data[x,-1]:
            corrects_count += 1
        else:
            wrongs_count += 1

    accuracy = corrects_count/test_data.shape[0]
    #print(f'Accuracy: {accuracy*100:.2f}%\n')


    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for x in range(predictions.shape[0]):
        if predictions[x] == 1:
            if test_data[x,-1] == 1:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if test_data[x,-1] == 1:
                false_negatives += 1
            else:
                true_negatives += 1

    #print(f'True Positives: {true_positives}\n')
    #print(f'False Positives: {false_positives}\n')
    #print(f'True Negatives: {true_negatives}\n')
    #print(f'False Negaties: {false_negatives}\n')

    precision = precision_eq(true_positives, false_positives)
    recall = recall_eq(true_positives, false_negatives)

    f1_measure = f1_measure_eq(precision,recall)

    return accuracy, f1_measure

def f1_measure_eq(precision,recall):
    return 2*(precision*recall)/(precision+recall)

def precision_eq(true_positives, false_positives):
    return true_positives/(true_positives+false_positives)

def recall_eq(true_positives, false_negatives):
    return true_positives/(true_positives+false_negatives)

# Function that prints a table with performance metrics of an algorithm based on its accuracy and f-1 score
def print_table(accuracy, f1_measure):
    from prettytable import PrettyTable
    
    x = PrettyTable()

    x.field_names = ["Fold", "Accuracy", "F-1 Measure"]
    for i in range(accuracy.shape[1]):
        x.add_row(['','',''])
        for j in range(accuracy.shape[0]):
            x.add_row([j+1, f'{accuracy[j,i]*100:.2f}%', f'{f1_measure[j,i]:.2f}'])
    
    x.add_row(['','',''])
    x.add_row(['','',''])
    x.add_row(['Average', f'{np.mean(accuracy)*100:.2f}%', f'{np.mean(f1_measure):.2f}'])
    x.add_row(['Standard Deviation', f'{np.std(accuracy)*100:.2f}%', f'{np.std(f1_measure):.2f}'])

    print(x)

# Function that performs a repeated cross-validation of a knn analysis
def repeated_cross_validation(r,k,dataset):
    accuracy = np.zeros((k,r))
    f1_measure = np.zeros((k,r))
    for i in range(r):
        k_folds = stratified_k_folds(k,dataset)
        for j in range(k):
            accuracy[j,i], f1_measure[j,i] = knn(j,k_folds)
    print_table(accuracy, f1_measure)

repeated_cross_validation(5,3,'diabetes.csv')
