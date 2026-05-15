-- Diogo Borges     64277
-- Pedro Batalheiro 62271

module PF.Training
(training) where

import PF.NeuralNetwork (Network, Layer(..), backPropagation)

-- | Treina a rede durante n iterações, com uma taxa de
-- aprendizagem e percorrendo os exemplos ciclicamente.
training :: Int -> Double -> ([[Double]], [[Double]]) -> PF.NeuralNetwork.Network -> PF.NeuralNetwork.Network

training 0 _ _ net = net
training it eta (inputMtx, expOutputs) net = 
  let
    -- emparelhar input com output
    dataset = zip inputMtx expOutputs
    -- para cada valor do dataset, foldl vai aplicar uma função lambda
    -- esta tem parâmetros currentNet (net) e dataset com PatternMatching em (input, output)
    -- dataset faz backPropagation pelos inputs e outputs
    newNet = foldl (\currentNet (input, output) -> PF.NeuralNetwork.backPropagation eta input output currentNet) net dataset
  -- nova iteração
  in training (it - 1) eta (inputMtx, expOutputs) newNet