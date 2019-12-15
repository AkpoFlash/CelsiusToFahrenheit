class NanoNeuron {
    constructor(w, b) {
        this.w = w;
        this.b = b;
    }

    predict = (x) => x * this.w + this.b;

    celsiusToFahrenheit = (c) =>  c * 1.8 + 32;

    generateDataSet = (amount) => {
        const xTrain = [];
        const yTrain = [];

        for(let x = 0; x < amount; x++){
            xTrain.push(x);
            yTrain.push(this.celsiusToFahrenheit(x));
        }

        const xTest = [];
        const yTest = [];

        for(let x = 0.5; x < amount; x++){
            xTest.push(x);
            yTest.push(this.celsiusToFahrenheit(x));
        }

        return [xTrain, yTrain, xTest, yTest];
    };

    predictCost = (y, prediction) => (y - prediction) ** 2 / 2;

    forwardPropagation = (model, xTrain, yTrain) => {
        const count = xTrain.length;
        const predictions = [];
        let predictCostSum = 0;

        for(let i = 0; i < count; i += 1){
            const prediction = this.predict(xTrain[i]);
            predictCostSum += this.predictCost(yTrain[i], prediction);
            predictions.push(prediction);
        }

        return [predictions, predictCostSum / count];
    };

    backwardPropagation = (predictions, xTrain, yTrain) => {
        const count = xTrain.length;
        let dW = 0;
        let dB = 0;

        for(let i = 0; i < count; i++){
            dW += (yTrain[i] - predictions[i]) * xTrain[i];
            dB += yTrain[i] - predictions[i];
        }

        return [dW / count, dB / count];
    };

    trainModel = ({model, epochs, alpha, xTrain, yTrain}) => {
        const costHistory = [];

        for(let epoch = 0; epoch < epochs; epoch++){
            const [predictions, cost] = this.forwardPropagation(model, xTrain, yTrain);
            costHistory.push(cost);

            const [dW, dB] = this.backwardPropagation(predictions, xTrain, yTrain);

            this.w += alpha * dW;
            this.b += alpha * dB;
        }

        return costHistory;
    }
}

const epochs = 70000;
const alpha = 0.0005;
const w = Math.random();
const b = Math.random();

const nanoNeuron = new NanoNeuron(w, b);
const [xTrain, yTrain, xTest, yTest] = nanoNeuron.generateDataSet(100);

const trainingCostHistory = nanoNeuron.trainModel({model: nanoNeuron, epochs, alpha, xTrain, yTrain});

console.log('Ошибка до тренировки:', trainingCostHistory[0]);
console.log('Ошибка после тренировки:', trainingCostHistory[epochs - 1]);
console.log('Параметры нано-нейрона:', {w: nanoNeuron.w, b: nanoNeuron.b});

[testPredictions, testCost] = nanoNeuron.forwardPropagation(nanoNeuron, xTest, yTest);
console.log('Ошибка на новых данных:', testCost);

const tempInCelsius = 70;
const customPrediction = nanoNeuron.predict(tempInCelsius);
console.log(`Нано-нейрон "думает", что ${tempInCelsius}°C в Фаренгейтах будет:`, customPrediction);
console.log('А правильный ответ:', nanoNeuron.celsiusToFahrenheit(tempInCelsius))