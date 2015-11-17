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
