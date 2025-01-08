#include "neuron.h"

Neuron::Neuron(int connectedNeurons) {
    if(connectedNeurons > 0)
        for(int i = 0 ; i < connectedNeurons; i++){
            this->w[i] = randomNumber();
        }
    else
        this->w[0] = randomNumber();
    this->b = randomNumber();
}
