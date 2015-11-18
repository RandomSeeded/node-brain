var numeric = require('numeric');
var randgen = require('randgen');
var _ = require('underscore');
var helpers = require('./helpers');
var operationScalarMatrix = helpers.operationScalarMatrix;

// Sizes is a array representing the number of nodes in each layer of the network
var Network = module.exports.Network = function(sizes) {
  this.numLayers = sizes.length;
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
      var b = zippedMatrix[layer][0];
      var w = zippedMatrix[layer][1];
      var z = helpers.calcZ(w, a, b);
      a = helpers.sigmoid(z);
    }
    return a;
  }

  this.SGD = function(trainingData, epochs, miniBatchSize, eta, testData) {
    // Train the neural network using mini-batch stochastic
    // gradient descent.  The trainingData is a list of tuples
    // (x, y) representing the training inputs and the desired
    // outputs.  The other non-optional parameters are
    // self-explanatory.  If testData is provided then the
    // network will be evaluated against the test data after each
    // epoch, and partial progress printed out.  This is useful for
    // tracking progress, but slows things down substantially.

    // Initializations
    testData = testData || null;
    if (testData) { n_test = testData.length; }
    n = trainingData.length;

    for (var epoch = 0; epoch < epochs; epoch++) {
      helpers.shuffle(trainingData);
      var miniBatches = [];
      for (var j = 0; j < n; j += miniBatchSize) {
        miniBatches.push(trainingData.slice(j, j+miniBatchSize));
      }

      //console.log('epoch weights last', this.weights[this.weights.length-1]);
      for (var k = 0; k < miniBatches.length; k++) {
        //update mini batch
        this.updateMiniBatch(miniBatches[k], eta);
      }
      //console.log('epoch post-weights', this.weights[this.weights.length-1]);

      if (testData) {
        console.log("Epoch " + epoch + ": " + this.evaluate(testData) + " / " + n_test);
      } else {
        console.log("Epoch " + epoch + " complete");
      }
    }
  }

  this.updateMiniBatch = function(miniBatch, eta) {
    // Update the network's weights and biases by applying gradient descent using
    // backpropogation to a single mini batch, where a mini batch is an array of
    // training data. ETA is learning rate.
    // Training data: 784x1 inputs, 10x1 output
    var nabla_b = helpers.zeros(this.biases);
    var nabla_w = helpers.zeros(this.weights);

    for (var i = 0; i < miniBatch.length; i++) {
      var datum = miniBatch[i];

      // Backpropogate to get delta nablas
      // NOTE: you should tuple this so that it's not dependent on this specific input
      var backprop = this.backprop(datum.pixels, datum.label);
      var delta_nabla_b = backprop.nabla_b;
      var delta_nabla_w = backprop.nabla_w;

      // Update nablas for each layer with the deltas calculated in the backpropogation
      for (var j = 0; j < nabla_b.length; j++) {
        nabla_b[j] = numeric.add(nabla_b[j], delta_nabla_b[j]);
        nabla_w[j] = numeric.add(nabla_w[j], delta_nabla_w[j]);
      }
    }

    // Finally, update the weights and biases
    for (var i = 0; i < this.weights.length; i++) {
      var w = this.weights[i];
      var b = this.biases[i];
      var nw = nabla_w[i];
      var nb = nabla_b[i];

      // new weight matrix: w-(eta/len(mini_batch))*nw
      // eta/len(mini_batch) -> CONSTANT
      // (eta/len(mini_batch)*nw -> matrix [nw is a matrix]
      // w - [^] -> matrix subtraction
      this.weights[i] = numeric['-'](w, operationScalarMatrix('multiply',(eta/miniBatch.length),nw));
      this.biases[i] = numeric['-'](b, operationScalarMatrix('multiply',(eta/miniBatch.length),nb));
      if (isNaN(this.weights[i][0][0])) {
        //console.log('w',w);
        //console.log('nw', nw);
        //console.log('eta minibatch len', eta/miniBatch.length);
        console.log('minibatch len', miniBatch.length);
        throw "die";
      }
    }
    //console.log('final weights',this.weights[this.weights.length-1]);
  }

  this.backprop = function(x, y) {
    // List of zero-matrices shaped like biases & weights
    var nabla_b = helpers.zeros(this.biases);
    var nabla_w = helpers.zeros(this.weights);

    // Feedforward
    var activation = x;
    // Store all the activations, layer by layer
    var activations = [x]; 
    // Store all the z vectors, layer by layer (Z = w * a + b)
    var zs = [];

    for (var i = 0; i < this.biases.length; i++) {
      var b = this.biases[i];
      var w = this.weights[i];
      z = helpers.calcZ(w, activation, b);
      zs.push(z);
      activation = helpers.sigmoid(z);
      activations.push(activation)
    }

    // Backward pass
    var costDeriv = this.costDerivative(activations[activations.length-1], y);
    var sigPrime = helpers.sigmoidPrime(zs[zs.length-1]);
    var delta = helpers.hadamardProduct(costDeriv, sigPrime);
    
    nabla_b[nabla_b.length-1] = delta;
    nabla_w[nabla_w.length-1] = numeric.dot(delta, numeric.transpose(activations[activations.length-2]));

    // Work backwards through the layers
    for (var l = 2; l < this.numLayers; l++) {
      z = zs[zs.length - l];
      var sp = helpers.sigmoidPrime(z);
      // Delta = (weights[-l+1].transpose() *[dot] delta) *[hammond] sp
      delta = helpers.hadamardProduct(numeric.dot(numeric.transpose(this.weights[this.weights.length-l+1]), delta), sp);
      nabla_b[nabla_b.length-l] = delta;
      nabla_w[nabla_w.length-l] = numeric.dot(delta, numeric.transpose(activations[activations.length-l-1]));
    }

    return {
      nabla_b: nabla_b,
      nabla_w: nabla_w
    }
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

  // Returns the vector of partial derivatives dC/dW or dC/dB
  // (change in cost as a result of change in weight or bias)
  // Remember: cost function is mean-squared-error: aka how far off perfect we are
  // By changing W or B, we change the MSE of cost, and thereby improve our network
  this.costDerivative = function(outputActivations, y) {
    return numeric['-'](outputActivations, y);
  }
}

// net = new Network([2,3,2]);
// console.log('output',net.feedForward([1,1]));

