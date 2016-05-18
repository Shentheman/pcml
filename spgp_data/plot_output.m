x = load('train_inputs');
y = load('train_outputs');
xtest = load('test_inputs');
data = load('test_outputs');
mu = data(:,1);
s2 = data(:,2);

%%%%%%%%%
% Plotting - just for 1D demo - remove for real data set
% Hopefully, the predictions should look reasonable

clf
hold on
plot(x,y,'.m') % data points in magenta
plot(xtest,mu,'b') % mean predictions in blue
plot(xtest,mu+2*sqrt(s2),'r') % plus/minus 2 std deviation
                              % predictions in red
plot(xtest,mu-2*sqrt(s2),'r')
% x-location of pseudo-inputs as black crosses
hold off
axis([-3 10 -3 2])