module.exports.sigmoid = function(z) {
  var negMatrix = operationScalarMatrix('multiply', -1, z);
  var eMatrix = numeric.exp(negMatrix);
  var eMatrixPlusOne = operationScalarMatrix('add', 1, eMatrix);
  var oneOver = operationScalarMatrix('divide', 1, eMatrixPlusOne);
  return oneOver;
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
