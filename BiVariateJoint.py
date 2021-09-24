import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

def get_class_prob_naive(x_data, y_data, joint_class_1, joint_class_2, likelihood_indep_class_1, likelihood_indep_class_2):
    prior_class_1 = joint_class_1.N/ (joint_class_1.N + joint_class_2.N)
    prior_class_2 = joint_class_2.N/ (joint_class_1.N + joint_class_2.N)
    likelihood_class_1 = likelihood_indep_class_1[joint_class_1.data_to_index(x_data, y_data)] 
    likelihood_class_2 = likelihood_indep_class_2[joint_class_2.data_to_index(x_data, y_data)]
    total = likelihood_class_1*prior_class_1 + prior_class_2*likelihood_class_2
    # Evita division por cero
    total[total==0] = 1
    p_class_1 = prior_class_1*likelihood_class_1/total
    p_class_2 = prior_class_2*likelihood_class_2/total
    # Las indeterminadas en 0.5
    p_class_1[total==1] = 0.5
    p_class_2[total==1] = 0.5
    return p_class_1, p_class_2

def get_class_prob(x1_data, x2_data, joint_class_1, joint_class_2):
    ''' Devuelve la probabilidad de que los pares de datos x1_data y 
        x2_data pertenezcan a la clase 1 o 2, dado un conjunto de pares de variables para cada clase (joint_class).
    '''
        # Cantidad de la clase dividido el total de valores. Es decir, son las marginales
        prior_class_1 = joint_class_1.N/ (joint_class_1.N + joint_class_2.N)
        prior_class_2 = joint_class_2.N/ (joint_class_1.N + joint_class_2.N)
        
        # Probabilidad de ocurrencia del par de valores, calculado como la cantidad de observaciones sobre el total para esa clase
        likelihood_class_1 = joint_class_1.get_prob(x1_data, x2_data)
        likelihood_class_2 = joint_class_2.get_prob(x1_data, x2_data)
        
        # Termino normalizador: Probabilidad total de ocurrencia del par de valores
        total = likelihood_class_1*prior_class_1 + prior_class_2*likelihood_class_2
        
        # Evita division por cero
        total[total==0] = 1
        p_class_1 = prior_class_1*likelihood_class_1/total
        p_class_2 = prior_class_2*likelihood_class_2/total
        # Las indeterminadas en 0.5
        p_class_1[total==1] = 0.5
        p_class_2[total==1] = 0.5
        return p_class_1, p_class_2

class BiVariateJoint:
    '''Serie de métodos y propiedades para un conjunto de dos variables. Ej: (Pesos,Alturas).'''
    
    def __init__(self, data, step_X = 1, step_Y = 1, mins=None, maxs=None):
        # Data tienen que ser un np.array de dos columnas
        self.step_X = step_X
        self.step_Y = step_Y
        step = np.array([step_X, step_Y])
        self.data = data
        
        ## Esto redondea a múltiplos del STEP dado ## 
        self.data_rounded = (np.round(data/step)*step)
        
        # Seteo máximos
        if maxs is None:
            self.maxs = np.max(self.data_rounded, axis = 0) + 1
        else:
            self.maxs = maxs
        if mins is None:
            self.mins = np.min(self.data_rounded, axis = 0) - 1
        else:
            self.mins = mins
        
        # Lista de tuplas que contienen cada par de valores de data (redondeada).
        tuples = [tuple(row) for row in self.data_rounded]
        # Diccionario donde key: Par de valores (únicos) y value: Cantidad de pares existentes en la data.
        self.frequencies = Counter(tuples)
        
        # Cantidad de intervalos (MAX-MIN)/INTERVALO
        # Agrego uno adelante y otro atras para cubrirme
        count_X = int(np.round((self.maxs[0] - self.mins[0])/step_X)) + 1
        count_Y = int(np.round((self.maxs[1] - self.mins[1])/step_Y)) + 1
        # Serie completa de valores, desde MIN a MAX según intervalo = step
        self.X = np.linspace(self.mins[0] - step_X, self.mins[0] + step_X*count_X, count_X + 2)
        self.Y = np.linspace(self.mins[1] - step_Y, self.mins[1] + step_Y*count_Y, count_Y + 2)
        
        self.joint_matrix = self.freq_2_matrix()
        self.N = len(data)
    
    def plot_data(self, color='b'):
        plt.scatter(self.data[:,0], self.data[:,1], color=color, s=2)
    
    def plot_rounded(self, color='b'):
        plt.scatter(self.data_rounded[:,0], self.data_rounded[:,1], color=color, s=2)
    
    def data_to_index(self, x, y):
        ''' Para un par de valores x,y devuelve sus índices en 
            la matriz de frecuencias (joint_matrix)
        '''
        x = np.round((x - self.X[0])/self.step_X).astype(int)
        y = np.round((y - self.Y[0])/self.step_Y).astype(int)
        return x, y

    def get_prob(self, x, y, normalized=True):
        '''
        Devuelve la probabilidad de un par de valores, calculado como la frecuencia sobre el total de valores. La frecuencia va a estar influenciada por los tamaños de intervalos escogidos.
        '''
        x, y = self.data_to_index(x, y)
        if normalized:
            prob = self.joint_matrix[x , y]/self.N
        else:
            prob = self.joint_matrix[x , y]
        return prob
    
    def freq_2_matrix(self):
        ''' Genera una matiz 2D, donde las filas corresponden a la variable X
        y las columnas corresponden a la variable Y. Cada valor de la matriz equivale a la cantidad de observaciones para el par de valores x,y
        '''
        # Inicializa matriz de 0, de dimensiones X*Y
        joint = np.zeros([len(self.X), len(self.Y)])
        # Itera en pares de valores y su frecuencia
        for index, frec in self.frequencies.items():
            # Cantidad de intervalos de distancia desde valor mínimo de X o Y
            x = (index[0] - self.X[0])/self.step_X
            y = (index[1] - self.Y[0])/self.step_Y
            # Trunco al entero para obtener la posición de en la matriz de frecuencias.
            joint[int(x), int(y)] = frec
        return joint
    
    def get_Marginals(self, normalized=True):
        if normalized:
            marg_1 = self.joint_matrix.sum(axis=1)/self.N
            marg_2 = self.joint_matrix.sum(axis=0)/self.N
        else:
            marg_1 = self.joint_matrix.sum(axis=1)
            marg_2 = self.joint_matrix.sum(axis=0)
        return marg_1, marg_2
    
    def plot_joint_3d(self, joint_matrix = None, el=50, az=-5, ax=None, color='b', title=''):
    # Plotea el histograma en 3D
        xpos, ypos = np.meshgrid(self.X, self.Y)
        xpos = xpos.T.flatten()
        ypos = ypos.T.flatten()
        zpos = np.zeros(xpos.shape)
        dx = self.step_X * np.ones_like(zpos)
        dy = self.step_Y * np.ones_like(zpos)
        if joint_matrix is None:
            dz = self.joint_matrix.astype(int).flatten()
        else:
            dz = joint_matrix.flatten()
        if ax == None:
            fig = plt.figure(figsize=(20,20))
            ax = fig.add_subplot(111, projection='3d')
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color, alpha=0.5)
        ax.set_title(title)
        ax.view_init(el, az)