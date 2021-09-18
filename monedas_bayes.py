def monedas_bayes(n_monedas, p_monedas, tirada):
    ''' Estima la probabilidad de haber sacado una moneda de cada tipo, 
        dada una serie de tiradas.
        
        * n_monedas : Array tamaño n, indicando la cantidad de monedas de cada tipo.
        * p_monedas: Array tamaño n (ídem n_monedas) indicando la probabilidad de obtener cara para cada tipo de moneda
        * tirada: Array tamaño m, con el resultado de una serie de tiradas, donde 1 es cara y 0 ceca.
    '''
    
    # Probabilidad de extraer cada moneda (Prior)
    pB = [n/sum(n_monedas) for n in n_monedas]
    
    # Likelihood para cada tipo de moneda (probabilidad de observar la tirada X, suponiendo que la moneda fuera Y)
    l = []
    for i, _ in enumerate(n_monedas):
        # li es el likelihood a calcular como la probabilidad conjunta de todas las tiradas realizadas. Lo seteo a 1 para cada tipo de moneda
        li = 1
        for t in tirada:
            if t == 1:
                li = li * p_monedas[i]
            else:
                li = li * (1 - p_monedas[i])
        l.append(li)
        
    # Calculo término normalizador (probabilidad total de obtener la tirada X considerando las monedas existentes)
    pA = 0
    for i in range(len(n_monedas)):
        pA += l[i] * pB[i]
        
    # Calculo el posterior
    pos = []
    for i in range(len(n_monedas)):
        pos.append(l[i]*pB[i]/pA)
        
    # Mostrar resultados:
    print(f'Dada la tirada: {tirada}, la probabilidad de que se trate de la moneda de cada tipo es:')
    for i, _ in enumerate(n_monedas):
        print(f'Moneda tipo {i+1}: {round(pos[i]*100,2)}%')
    
    print(f'La suma de los posteriors da: {sum(pos)}')
    print('\nNOTA: Esta función no es recomendable para calcular tiradas demasiado largas ya que el cálculo de P(B|A) implica el producto de muchos valores pequeños, por lo tanto existe el riesgo de introducirse error en los resultados finales por inestabilidad numérica. Para solucionar esto, se recomienda calcular el log(likelihood)')