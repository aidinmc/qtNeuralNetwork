#ifndef NEURON_H
#define NEURON_H
#include <QMap>
#include <cmath>

class Neuron {
public:
    Neuron(int connectedNeurons = 0);
    void setValue(double v) {
        this->v = v;
    }

    void setBiase(double b){
        this->b = b;
    }

    void setWeight(int index,double w){
        this->w[index] = w;
    }

    void setPredictedValue(double v){
        this->p = v;
    }

    double getWeight(int index){
        return this->w[index];
    }

    double getBiase(){
        return this->b;
    }

    double getValue() {
        return this->v;
    }

    double getPredictedValue(){
        return this->p;
    }

    double getError(){
        return this->v - this->p;
    }

    double randomNumber(double min = 0, double max = 1) {
        return (rand() / static_cast<double>(RAND_MAX));
    }

private:
    QMap<int,double> w; //weights
    double b = 0; //biase
    double v = 0; //value
    double p = 0; //Network predicted value

};

#endif // NEURON_H
