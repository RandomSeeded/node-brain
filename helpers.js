var numeric = require('numeric');

// Element-wise multiply two vectors
var hadamardProduct = module.exports.hadamardProduct = function(a, b) {
  results = [];
  for (var row = 0; row < a.length; row++) {
    results.push([a[row][0] * b[row][0]]);
  }
  return results;
}

var sigmoid = module.exports.sigmoid = function(z) {
  var negMatrix = operationScalarMatrix('multiply', -1, z);
  var eMatrix = numeric.exp(negMatrix);
  var eMatrixPlusOne = operationScalarMatrix('add', 1, eMatrix);
  var oneOver = operationScalarMatrix('divide', 1, eMatrixPlusOne);
  return oneOver;
}

module.exports.sigmoidPrime = function(z) {
  // SigmoidPrime(z) = sigmoid(z) * (1-sigmoid(z))
  // NOTE: the * is NOT dot product, it's element-wise multiplication (Hadamard product)
   var sigZ = sigmoid(z);
   var oneMinus = operationScalarMatrix('subtract', 1, sigZ);
   var result = hadamardProduct(sigZ, oneMinus);
   return result;
}

var operationScalarMatrix = module.exports.operationScalarMatrix = function(operation, scalar, matrix) {
  matrix = clone(matrix);
  for (var row = 0; row < matrix.length; row++) {
    for (var col = 0; col < matrix[row].length; col++) {
      if (operation === 'add') {
        matrix[row][col] = scalar + matrix[row][col];
      }
      else if (operation === 'multiply') {
        matrix[row][col] = scalar * matrix[row][col];
      }
      else if (operation === 'divide') {
        matrix[row][col] = scalar / matrix[row][col];
      }
      else if (operation === 'subtract') {
        matrix[row][col] = scalar - matrix[row][col];
      }
    }
  }
  return matrix;
}

// Fisher-yates shuffle
module.exports.shuffle = function(array) {
  var currentIndex = array.length, temporaryValue, randomIndex ;

  // While there remain elements to shuffle...
  while (0 !== currentIndex) {
    // Pick a remaining element...
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex -= 1;

    // And swap it with the current element.
    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }

  return array;
}

// Calculates w*a + b, where w = weight matrix, a = activation matrix, and b = bias matrix
module.exports.calcZ = function(w, a, b) {
  var dotProduct = numeric.dot(w, a);

  // Implement broadcasting for dot product and biases for broadcast addition of different sized matrices
  // If the dot product is a 1xX matrix, stretch it
  // Otherwise, LEAVE IT
  var broadcastDotProduct = [];
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
  for (var i = 0; i < b.length; i++) {
    var val = b[i][0];
    var row = [];

    for (var j = 0; j < numCols; j++) {
      row.push(val);
    }
    broadcastBiases.push(row);
  }
  return numeric.add(broadcastDotProduct, broadcastBiases);
}

// Create zero matrices corresponding to an array of matrices 
module.exports.zeros = function(array) {
  var zeros = [];
  for (var matrixIdx = 0; matrixIdx < array.length; matrixIdx++) {
    var matrix = array[matrixIdx];
    var rows = matrix.length;
    var cols = matrix[0].length;
    zeros.push(zero(rows, cols));
  }
  return zeros;
}

function zero(rows, cols) {
  var result = [];
  for (var row = 0; row < rows; row++) {
    var newRow = [];
    for (var col = 0; col < cols; col++) {
      newRow.push(0);
    }
    result.push(newRow);
  }
  return result;
}

function clone (existingArray) {
   var newObj = (existingArray instanceof Array) ? [] : {};
   for (i in existingArray) {
      if (i == 'clone') continue;
      if (existingArray[i] && typeof existingArray[i] == "object") {
         newObj[i] = clone(existingArray[i]);
      } else {
         newObj[i] = existingArray[i]
      }
   }
   return newObj;
}

