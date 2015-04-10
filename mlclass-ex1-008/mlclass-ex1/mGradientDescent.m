function theta = mGradientDescent(X, y, alpha)

%Full gradient descent function, including cost function and normalization
%At the end it also uses the normal equiation and compares values

Xraw = X;
Xraw = [ones(size(X)(1), 1) X];
deviation = std(X);
average = mean(X);

Xnorm = X.-average;
Xnorm = Xnorm./deviation;

X = Xnorm;

X = [ones(size(X)(1), 1) X];
theta = zeros(3, 1);

iterations = 1500;

m = size(X)(1);

h = X * theta;

diff = h - y;
cost = (1/(2*m)) * sum(diff.^2);

disp(cost);

while iterations > 1,
    iterations = iterations - 1;
    h = X * theta;

    diff = h - y;
    cost = (1/(2*m)) * sum(diff.^2);

    T = X'*diff;
    U = (alpha/m) * T;

    theta = theta - U;
end
   
sample = [1650 3];
 


disp("\n--THETA Normal equiation--")
theta2 = pinv(Xraw'*Xraw) * (Xraw'*y)
[1 sample] * theta2

disp("\n---THETA Gradient descent--")
theta
sampleN = (sample.-average)./deviation;
sampleN = [1 sampleN];
sampleN * theta




