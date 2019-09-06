load bodyfat_dataset.mat

inputs = bodyfatInputs;
outputs = bodyfatTargets;

net = fitnet([5 5], 'trainbr');
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

[net, tr] = train(net, inputs, outputs);
outputTest = net(inputs);
error = gsubtract(outputs, outputTest);
performance = perform(net, outputs, outputTest);
