-- Diogo Borges     64277
-- Pedro Batalheiro 62271

module Training
(training) where

import NeuralNetwork (Network, Layer(..), backPropagation)

-- | Treina a rede durante n iterações, com uma taxa de
-- aprendizagem e percorrendo os exemplos ciclicamente.
<<<<<<< Updated upstream
training :: Int -> Double -> ([[Double]], [[Double]])
=======
training :: Int -> Double -> ([[Double]], [[Double]]) -> NeuralNetwork.Network -> NeuralNetwork.Network
>>>>>>> Stashed changes

training 0 _ _ net = net
training it eta (inputMtx, expOutputs) net = 
  let
    -- emparelhar input com output
    dataset = zip inputMtx expOutputs
    -- para cada valor do dataset, foldl vai aplicar uma função lambda
    -- esta tem parâmetros currentNet (net) e dataset com PatternMatching em (input, output)
    -- dataset faz backPropagation pelos inputs e outputs
<<<<<<< Updated upstream
    newNet = foldl (\currentNet (input, output) eta input output currentNet) net dataset
=======
    newNet = foldl (\currentNet (input, output) -> NeuralNetwork.backPropagation eta input output currentNet) net dataset
>>>>>>> Stashed changes
  -- nova iteração
  in training (it - 1) eta (inputMtx, expOutputs) newNet