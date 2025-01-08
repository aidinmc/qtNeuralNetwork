#ifndef NETWORK_H
#define NETWORK_H

#include <QDebug>
#include <QList>

class Neuron;

class Network {
public:
    Network();

    void appendInputNeuron(Neuron *neuron);
    void appendHiddenNeuron(Neuron *neuron);
    void appendOutputNeuron(Neuron *neuron);

    void setIteration(int);

    int getInputSize() const;
    int getHiddenSize() const;
    int getOutputSize() const;

    QList<Neuron*>* getInputNeurons();
    QList<Neuron*>* getHiddenNeurons();
    QList<Neuron*>* getOutputNeurons();

    double sigmoid(double x);
    double sigmoidDerivative(double x);

    void train();
    QList<Neuron*>* predict(double *inputs);

    ~Network();
private:
    QList<Neuron*> *inputNeurons;
    QList<Neuron*> *hiddenNeurons;
    QList<Neuron*> *outputNeurons;

    int iteration = 500000;
    float rate = 0.0003;
    float eps = 1e-3;
};

#endif // NETWORK_H
