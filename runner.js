var data = require('./mnistLoader').loadDataWrapper();
var network = require('./network');

net = new network.Network([784,30,10]);
net.SGD(data.trainingData, 30, 10, 3.0, data.testData);

