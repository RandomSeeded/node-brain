var numeric = require('numeric');
var randgen = require('randgen');
var _ = require('underscore');
var helpers = require('./helpers');
var operationScalarMatrix = helpers.operationScalarMatrix;
var sigmoid = helpers.sigmoid;
var shuffle = helpers.shuffle;

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
      // console.log('w',weights);
      // console.log('a', a);
      var dotProduct = numeric.dot(weights, a);

      // console.log('dp',dotProduct);
      // Implement broadcasting for dot product and biases for broadcast addition of different sized matrices
      // If the dot product is a 1xX matrix, stretch it
      // Otherwise, LEAVE IT
      var broadcastDotProduct = [];
      //if (dotProduct.length !== 1) {
      if (!Array.isArray(dotProduct[0])) {
        var dims = dotProduct.length
        for (var i = 0; i < dims; i++) { 
          broadcastDotProduct.push(dotProduct.slice(0)); 
        }
      } else {
        broadcastDotProduct = dotProduct;
      }

      // Get width to stretch biases to
      var numCols;
      if (Array.isArray(dotProduct[0])) {
        numCols = dotProduct[0].length;
      } else {
        numCols = dotProduct.length;
      }
      var broadcastBiases = [];
      for (var i = 0; i < biases.length; i++) {
        var val = biases[i][0];
        var row = [];

        for (var j = 0; j < numCols; j++) {
          row.push(val);
        }
        broadcastBiases.push(row);
      }
      // console.log('b',biases);
      // console.log('broadcast dp', broadcastDotProduct);
      // console.log('broadcast b', broadcastBiases);
      a = sigmoid(numeric.add(broadcastDotProduct, broadcastBiases));
      // console.log('enda', a);
    }
    return a;
  }

  this.SGD = function(training_data, epochs, miniBatchSize, eta, testData) {
    // Train the neural network using mini-batch stochastic
    // gradient descent.  The training_data is a list of tuples
    // (x, y) representing the training inputs and the desired
    // outputs.  The other non-optional parameters are
    // self-explanatory.  If testData is provided then the
    // network will be evaluated against the test data after each
    // epoch, and partial progress printed out.  This is useful for
    // tracking progress, but slows things down substantially.

    // Initializations
    testData = testData || null;
    if (testData) { n_test = testData.length; }
    n = training_data.length;

    for (var epoch = 0; epoch < epochs; epoch++) {
      shuffle(training_data);
      var miniBatches = [];
      for (var j = 0; j < n; j += miniBatchSize) {
        miniBatches.push(training_data.splice(j, j+ miniBatchSize));
      }

      for (var k = 0; k < miniBatches.length; k++) {
        //update mini batch
        this.updateMiniBatch(miniBatches[k], eta);
      }

      if (testData) {
        console.log("Epoch " + epoch + ": " + this.evaluate(testData) + " / " + n_test);
      } else {
        console.log("Epoch " + epoch + " complete");
      }
    }
  }

  this.updateMiniBatch = function(miniBatch, eta) {
    //console.log('updating mini batch', miniBatch);
  }

  this.evaluate = function(testData) {
    var numCorrect = 0;
    for (var i = 0; i < testData.length; i++) {
      var resultArr = this.feedForward(testData[i].pixels);
      var maxResult = 0;
      var result = 0;
      // Turn array of output nodes into one best guess
      for (var nodeIdx = 0; nodeIdx < resultArr.length; nodeIdx++) {
        if (resultArr[nodeIdx] > maxResult) {
          maxResult = resultArr[nodeIdx];
          result = nodeIdx;
        }
      }
      if (result === testData[i].label) {
        numCorrect++;
      }
    }
    return numCorrect;
  }
}

// net = new Network([2,3,2]);
// console.log('output',net.feedForward([1,1]));

