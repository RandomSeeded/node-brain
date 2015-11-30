/*
Loads the MNIST image data
*/

var _ = require('underscore-contrib');
var fs = require('fs');
var numeric = require('numeric');

//var numTotalImages = 60000;
//var trainingDataSize = 50000;
var numTotalImages = 10000;
var trainingDataSize = 9000;

// Returns an array of objects with format:
// [
// { 'label': [pixel, pixel, pixel] },
// { 'label': [pixel, pixel, pixel] }
// ]
// Each pixel having a # from 0-255
var readData = module.exports.readData = function() {
  var pixelValues = [];
  var dataFileBuffer = fs.readFileSync(__dirname + '/data/train-images.idx3-ubyte');
  var labelFileBuffer = fs.readFileSync(__dirname + '/data/train-labels.idx1-ubyte');
  for (var image = 0; image < numTotalImages; image++) {
    var pixels = [];

    for (var x = 0; x < 28; x++) {
      for (var y = 0; y < 28; y++) {
        pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 15]);
      }
    }
    var imageData = {};
    imageData.label = labelFileBuffer[image+8];
    imageData.pixels = pixels;
    pixelValues.push(imageData);
  }
  return pixelValues;
}

var loadDataWrapper = module.exports.loadDataWrapper = function() {
  // We return an object which contains training data, validation data, and test data
  // Every set of data has an array of inputs and outputs
  // Inputs is a 1x784 array, representing each pixel sequentially
  // For training data, outputs is a 1x10 array, representing the vector of outputs
  // For the others, output is the integer digit value representing the output
  // NOTE: this could be split out into training/validation/test. Here we have instead just done training/test
  var trainingData = readData();
  var testData = trainingData.splice(trainingDataSize);

  for (var i = 0; i < trainingData.length; i++) {
    // Reshape training data input into 1x784 array
    trainingData[i].pixels = numeric.transpose([trainingData[i].pixels]);

    // Reshape training data outputs into a 10x1 array
    var labelArr = [];
    for (var digit = 0; digit < 10; digit++) {
      labelArr[digit] = [0];
    }
    labelArr[trainingData[i].label] = [1];
    trainingData[i].label = labelArr;
  }

  // Reshape test data input into 1x784 array
  for (var i = 0; i < testData.length; i++) {
    testData[i].pixels = numeric.transpose([testData[i].pixels]);
  }

  return {
    trainingData: trainingData,
    testData: testData
  }
}
