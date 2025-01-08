#include "network.h"
#include "neuron.h"

Network::Network() {
    this->inputNeurons = new QList<Neuron*>();
    this->hiddenNeurons = new QList<Neuron*>();
    this->outputNeurons = new QList<Neuron*>();
}

void Network::appendInputNeuron(Neuron *neuron){
    this->inputNeurons->append(neuron);
}

void Network::appendHiddenNeuron(Neuron *neuron){
    this->hiddenNeurons->append(neuron);
}

void Network::appendOutputNeuron(Neuron *neuron){
    this->outputNeurons->append(neuron);
}

void Network::setIteration(int i){
    this->iteration = i;
}

int Network::getInputSize() const
{
    return this->inputNeurons ? this->inputNeurons->size() : 0;
}

int Network::getHiddenSize() const{
    return this->hiddenNeurons ? this->hiddenNeurons->size() : 0;
}

int Network::getOutputSize() const{
    return this->outputNeurons ? this->outputNeurons->size() : 0;
}

QList<Neuron *> *Network::getInputNeurons(){
    return this->inputNeurons;
}

QList<Neuron *> *Network::getHiddenNeurons(){
    return this->hiddenNeurons;
}

QList<Neuron *> *Network::getOutputNeurons(){
    return this->outputNeurons;
}

double Network::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double Network::sigmoidDerivative(double x) {
    return x * (1 - x);
}

void Network::train(){
    for(int i = 0; i < this->iteration; i++){
        double sumError = 0;
        for(int inputIndex = 0; inputIndex < getInputSize(); inputIndex++){
            Neuron *inputNeuron = getInputNeurons()->at(inputIndex);
            double hiddenWeightSum = 0.0;
            for(int hiddenIndex = 0; hiddenIndex < getHiddenSize(); hiddenIndex++){
                Neuron *hiddenNeuron = getHiddenNeurons()->at(hiddenIndex);

                for(int j = 0; j < getInputSize(); j++){
                    Neuron * tempInputNeuron = getInputNeurons()->at(j);
                    hiddenWeightSum += hiddenNeuron->getWeight(j) * tempInputNeuron->getValue();
                }
                hiddenWeightSum += hiddenNeuron->getBiase();
                hiddenNeuron->setValue(sigmoid(hiddenWeightSum));
            }

            for(int outputIndex = 0; outputIndex < getOutputSize(); outputIndex++){
                double outputWeightSum = 0.0;
                Neuron *outputNeuron = getOutputNeurons()->at(outputIndex);
                for(int hiddenIndex = 0; hiddenIndex < getHiddenSize(); hiddenIndex++){
                    Neuron *hiddenNeuron = getHiddenNeurons()->at(hiddenIndex);
                    outputWeightSum += outputNeuron->getWeight(hiddenIndex) * hiddenNeuron->getValue();
                }
                outputWeightSum += outputNeuron->getBiase();
                outputNeuron->setPredictedValue(sigmoid(outputWeightSum));
            }

            Neuron * output = getOutputNeurons()->at(0);
            if(std::abs(output->getValue() - output->getPredictedValue()) <= 1e-7){
                double avgError = sumError / (getOutputSize() * getInputSize());
                qDebug() << "Return;";
                qDebug() << "Epoch: " << i+1 << "Average Error: " << avgError << "Output: " << getOutputNeurons()->at(0)->getPredictedValue();
                return;
            }

            //compute error and update weights/biases
            for(int outputIndex = 0; outputIndex < getOutputSize(); outputIndex++){
                Neuron *outputNeuron = getOutputNeurons()->at(outputIndex);
                double error = outputNeuron->getError();
                sumError += error * error;

                // update weights and biases of the output layer
                for(int hiddenIndex = 0; hiddenIndex < getHiddenSize(); hiddenIndex++){
                    Neuron * hiddenNeuron = getHiddenNeurons()->at(hiddenIndex);
                    double outputNeuronCurrentWeight = outputNeuron->getWeight(hiddenIndex);
                    outputNeuronCurrentWeight += this->rate * error * sigmoidDerivative(outputNeuron->getPredictedValue()) * hiddenNeuron->getValue();
                    outputNeuron->setWeight(hiddenIndex,outputNeuronCurrentWeight);
                }
                double outputNeuronCurrentBiase = outputNeuron->getBiase();
                outputNeuronCurrentBiase += this->rate * error * sigmoidDerivative(outputNeuron->getPredictedValue());
                outputNeuron->setBiase(outputNeuronCurrentBiase);

                // update weights and biases of the hidden layer
                double hiddenError = 0.0;
                for(int j = 0; j < getOutputSize(); j++){
                    hiddenError += error * sigmoidDerivative(outputNeuron->getPredictedValue()) * outputNeuron->getWeight(j);
                }
                for(int hiddenIndex = 0 ; hiddenIndex < getHiddenSize() ; hiddenIndex++){
                    Neuron * hiddenNeuron = getHiddenNeurons()->at(hiddenIndex);
                    double hiddenNeuronCurrentWeight = hiddenNeuron->getWeight(inputIndex);
                    hiddenNeuronCurrentWeight += this->rate * hiddenError * sigmoidDerivative(hiddenNeuron->getValue()) * inputNeuron->getValue();
                    hiddenNeuron->setWeight(inputIndex,hiddenNeuronCurrentWeight);
                }

            }
        }
        if(!(i%10000)){
            double avgError = sumError / (getOutputSize() * getInputSize());
            qDebug() << "Epoch: " << i+1 << "Average Error: " << avgError << "Output: " << getOutputNeurons()->at(0)->getPredictedValue();
        }
    }

}

QList<Neuron*>* Network::predict(double *inputs){
    double hiddenWeightSum = 0.0;
    for(int hiddenIndex = 0; hiddenIndex < getHiddenSize(); hiddenIndex++){
        Neuron *hiddenNeuron = getHiddenNeurons()->at(hiddenIndex);

        for(int j = 0; j < getInputSize(); j++){
            hiddenWeightSum += hiddenNeuron->getWeight(j) * inputs[j];
        }
        hiddenWeightSum += hiddenNeuron->getBiase();
        hiddenNeuron->setPredictedValue(sigmoid(hiddenWeightSum));
    }

    for(int outputIndex = 0; outputIndex < getOutputSize(); outputIndex++){
        double outputWeightSum = 0.0;
        Neuron *outputNeuron = getOutputNeurons()->at(outputIndex);
        for(int hiddenIndex = 0; hiddenIndex < getHiddenSize(); hiddenIndex++){
            Neuron *hiddenNeuron = getHiddenNeurons()->at(hiddenIndex);
            outputWeightSum += outputNeuron->getWeight(hiddenIndex) * hiddenNeuron->getPredictedValue();
        }
        outputWeightSum += outputNeuron->getBiase();
        outputNeuron->setPredictedValue(sigmoid(outputWeightSum));
    }

    return getOutputNeurons();
}

Network::~Network() {
    delete inputNeurons;
    delete hiddenNeurons;
    delete outputNeurons;
}
