debugger;
var data = require('./mnistLoader').loadDataWrapper();
var network = require('./network');

// var testOut = [];
// var testNum = 4;
// var counter = 0;
// for (var i = 0; i < 28; i++) {
//   var row = [];
//   for (var j = 0; j < 28; j++) {
//     row.push(data.trainingData[testNum].pixels[counter++][0]);
//   }
//   testOut.push(row);
// }
// console.log(data.trainingData[testNum].label, testOut);

training_data = [
  {
    pixels: [[0],[0]],
    label: [[1],[0]] //0
  },
  {
    pixels: [[1],[0]],
    label: [[0],[1]] //1
  },
  {
    pixels: [[0],[1]],
    label: [[0],[1]] //1
  },
  {
    pixels: [[1],[1]],
    label: [[1],[0]] //0
  },
]
test_data = [
  {
    pixels: [[0],[0]],
    label: 0
  },
  {
    pixels: [[1],[0]],
    label: 1
  },
  {
    pixels: [[0],[1]],
    label: 1
  },
  {
    pixels: [[1],[1]],
    label: 0
  }
]

net = new network.Network([2,30,2]);
console.log('preliminary weights', net.weights);
console.log('preliminary biases', net.biases);
console.log('preliminary results');
console.log('1 1',net.feedForward([[1],[1]])); // 0
console.log('0 1',net.feedForward([[0],[1]])); // 1
console.log('1 0',net.feedForward([[1],[0]])); // 1
console.log('0 0',net.feedForward([[0],[0]])); // 0
console.log('Total: ', net.evaluate(test_data) + " / " + test_data.length);
net.SGD(training_data, 1000, 4, 3.0, test_data);
console.log('-----------');
// console.log('final weights', net.weights);
// console.log('final biases', net.biases);
console.log('1 1',net.feedForward([[1],[1]])); // 0
console.log('0 1',net.feedForward([[0],[1]])); // 1
console.log('1 0',net.feedForward([[1],[0]])); // 1
console.log('0 0',net.feedForward([[0],[0]])); // 0

// net = new network.Network([784,30,10]);
// net.SGD(data.trainingData, 30, 10, 3.0, data.testData);

