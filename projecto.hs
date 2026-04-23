-- Diogo Borges     64277
-- Pedro Batalheiro 62271

--
-- ENTREGA 1
--

-- | Divide uma lista em grupos de tamanho n.
-- Exemplo: chunksOf 2 [1,2,3,4,5] == [[1,2],[3,4],[5]]
chunksOf :: Int -> [a] -> [[a]]
chunksOf 0 _ = []
chunksOf _ [] = []
chunksOf n xs = take n xs : chunksOf n (drop n xs)

-- | Função de ativação sigmoid.
-- Exemplo: sigmoid 0 == 0.5
sigmoid :: Double -> Double
sigmoid x = 1 / (1 + exp (-x))

-- | Derivada da sigmoid (em termos da saída, não da entrada).
-- Exemplo: sigmoid' 0.5 == 0.25
sigmoid' :: Double -> Double
sigmoid' x = x * (1 - x)

-- | Transpõe uma matriz (troca linhas por colunas).
-- Exemplo: transpose [[1,2],[3,4],[5,6]] == [[1,3,5],[2,4,6]]
transpose :: [[a]] -> [[a]]
transpose [] = []
transpose ([] : _)  = []
transpose xss = [x | (x:_) <- xss] : transpose [xs | (_:xs) <- xss]

-- | Multiplica uma matriz por um vetor: y[i] = sum(W[i][j] * x[j]).
-- Exemplo: multMatrix [[1,0],[0,1]] [3,4] == [3,4]
multMatrix :: [[Double]] -> [Double] -> [Double]
multMatrix xss ys = [sum (zipWith (*) xs ys) | xs <- xss]

-- | Soma ponto-a-ponto de dois vetores.
-- Exemplo: somaVectorial [1,2] [3,4] == [4,6]
somaVectorial :: [Double] -> [Double] -> [Double]
somaVectorial = zipWith (+)

--
-- ENTREGA 2
--

-- Em Haskell, um vetor é uma lista [Double] e uma
-- matriz é uma lista de listas [[Double]]

-- Camada de rede neuronal, com pesos para cada neurónio, e a bias de cada neurónio
data Layer = Layer {pesos ::[[Double]], 
                    bias :: [Double]
                    } deriving (Show, Eq, Ord) -- Retirar Ord se não for preciso

-- Rede neuronal
type Network = [Layer] -- newtype deve ser usado quando apenas tem um atributo


-- Para facilitar, vamos usar uma matriz dos pesos por cada camada (W ∈ Rm×n), 
-- bem como um vector dos bias (b ∈ Rm). O array de output de cada camada
-- é definido por:
-- y = σ(W x + b)

-- | Constrói uma rede neuronal.
-- Exemplo: length (buildNetwork 2 [4,1] (repeat 0.1)) == 2
buildNetwork :: Int -> [Int] -> [Double] -> Network

buildNetwork _ [] _ = [] -- quando não há mais layers
buildNetwork inputSize (l:ls) vals = layer : buildNetwork l ls rest -- adiciona layer à network e chama a função recursivamente
  where
    numW    = l * inputSize -- obtem numero de weights para criar
    weights = chunksOf inputSize (take numW vals) -- fazemos chunks de tamanho size, de numW elementos de vals
    biases  = take l (drop numW vals) -- cria as biases, pegando em l vals, excluindo os que já foram usados como pesos
    layer   = Layer weights biases -- cria o layer
    rest    = drop (numW + l) vals -- guarda o resto dos valores que ainda não foram utilizados


-- | Diferença entre previsão e alvo (por elemento).
-- Exemplo: outputError [0.9] [1.0] == [-0.1]
-- outputError :: [Double] -> [Double] -> [Double]

-- | Erro quadrático médio entre previsão e alvo.
-- Exemplo: mse [1.0] [0.0] == 1.0
-- mse :: [Double] -> [Double] -> Double

-- | MSE médio sobre um conjunto de previsões.
-- msePredictions :: [[Double]] -> [[Double]] -> Double

-- | Propaga a entrada pela rede e devolve todas as ativações,
-- da entrada até à saída: [entrada, act1, act2, ..., saída].
-- Exemplo: length (forwardPass [0,1] net) == length net + 1
-- forwardPass :: [Double] -> Network -> [[Double]]

-- | Executa um passo de backpropagation dado uma taxa de aprendizagem, 
-- o input e o output esperado para um exemplo concreto.
-- backPropagation :: Double -> [Double] -> [Double] -> Network -> Network

--
-- ENTREGA 3
--

-- | Treina a rede durante n iterações, com uma taxa de
-- aprendizagem e percorrendo os exemplos ciclicamente.
-- training :: Int -> Double -> ([[Double]], [[Double]]) -> Network -> Network