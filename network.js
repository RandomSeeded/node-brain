var numeric = require('numeric');
var randgen = require('randgen');
var _ = require('underscore');
var helpers = require('./helpers');
var operationScalarMatrix = helpers.operationScalarMatrix;
var sigmoid = helpers.sigmoid;

// Sizes is a array representing the number of nodes in each layer of the network
var Network = module.exports.Network = function(sizes) {
  this.num_layers = sizes.length;
  this.sizes = sizes;

  // Randomly generate a starting bias for each non-input node
  this.biases = [];
  for (var layer = 1; layer < sizes.length; layer++) {
    var layerSize = sizes[layer];
    var layerBiases = [];
    for (var node = 0; node < layerSize; node++) {
      layerBiases.push([randgen.rnorm(0,1)]);
    }
    this.biases.push(layerBiases);
  }

  // Randomly generate starting weights for each connection between nodes
  this.weights = [];
  for (var layer = 0; layer < sizes.length-1; layer++) {
    var layerWeights = [];
    for (var j = 0; j < sizes[layer+1]; j++) {
      var nodeWeights = [];
      for (var k = 0; k < sizes[layer]; k++) {
        nodeWeights.push(randgen.rnorm(0,1));
      }
      layerWeights.push(nodeWeights);
    }
    this.weights.push(layerWeights);
  }

  // Output the network for a given input 'a' (array which represents input to each input node)
  this.feedForward = function(a) {
    var zippedMatrix = _.zip(this.biases, this.weights);
    for (var layer = 0; layer < zippedMatrix.length; layer++) {
      var biases = zippedMatrix[layer][0];
      var weights = zippedMatrix[layer][1];

      // The new activation matrix is the dot product of weights * a + the biases, as BROADCASTED (eg 1x3 + 3x1 -> 3x3)
      var dotProduct = numeric.dot(weights, a);

      // Weights * a -> always a 1x[something] matrix. 
      var dims = dotProduct.length;

      // dotProduct can be either an array or an array within an array. This standardizes to an array
      if (Array.isArray(dotProduct[0])) {
        dotProduct = dotProduct[0];
      }

      // Implement broadcasting for dot product and biases for broadcast addition of different sized matrices
      var broadcastDotProduct = [];
      for (var i = 0; i < dims; i++) { 
        broadcastDotProduct.push(dotProduct.slice(0)); 
      }
      var broadcastBiases = [];
      for (var i = 0; i < dims; i++) {
        var val = biases[i][0];
        var row = [];
        for (var j = 0; j < dims; j++) {
          row.push(val);
        }
        broadcastBiases.push(row);
      }

      a = sigmoid(numeric.add(broadcastDotProduct, broadcastBiases));
    }
    return a;
  }
}

var myNetwork = new Network([2,3,1]);
console.log(myNetwork.feedForward([2,2]));

