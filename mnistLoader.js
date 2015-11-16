/*
Loads the MNIST image data
*/

var fs = require('fs');

// Returns an array of objects with format:
// [
// { 'label': [pixel, pixel, pixel] },
// { 'label': [pixel, pixel, pixel] }
// ]
// Each pixel having a # from 0-255
module.exports = function() {
  var pixelValues = [];
  var dataFileBuffer = fs.readFileSync(__dirname + '/data/train-images.idx3-ubyte');
  var labelFileBuffer = fs.readFileSync(__dirname + '/data/train-labels.idx1-ubyte');
  for (var image = 0; image < 60000; image++) {
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
