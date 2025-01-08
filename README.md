# qtNeuralNetwork
A simple neural network with a single hidden layer.

## Usage
```cpp
    Network * network = new Network();

    int inputCount = 2;
    int hiddenCount = 5;
    int outputCount = 1;

    double * inputs = new double[inputCount];
    inputs[0] = 0.3;
    inputs[1] = 0.4;

    double * outputs = new double[outputCount];
    outputs[0] = 0.12;

    for(int i = 0; i < 2; i++){
        Neuron * neuron = new Neuron();
        neuron->setValue(inputs[i]);
        network->appendInputNeuron(neuron);
    }

    for(int i = 0; i < hiddenCount; i++){
        Neuron * neuron = new Neuron(inputCount);
        network->appendHiddenNeuron(neuron);
    }

    for(int i =0 ; i < outputCount; i++){
        Neuron * neuron = new Neuron(hiddenCount);
        neuron->setValue(outputs[i]);
        network->appendOutputNeuron(neuron);
    }

    network->train();

    double * testInuts = new double[inputCount];
    testInuts[0] = 0.3;
    testInuts[1] = 0.4;

    QList<Neuron*>* result = network->predict(testInuts);

    for(int i = 0; i < result->size(); i++){
        qDebug() << (*result)[i]->getPredictedValue();
    }
```
