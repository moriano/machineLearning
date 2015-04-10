alpha = 0.01;
thetaZero = theta(1, 1);
thetaOne = theta(2, 1);

iterations = 1500;
range = 1:1:size(X)(1);
while(iterations > 0),
    iterations = iterations - 1;
    sumZero = 0;
    sumOne = 0;
    for i = range, 
        sumZero = sumZero + thetaZero + ( (thetaOne * X(i)) - y(i) );
        sumOne = sumOne + thetaOne + ( (thetaOne * X(i)) - y(i) ) * X(i);
    end;
    tmpZero = thetaZero - ((alpha/m) * sumZero);
    tmpOne = thetaOne - ((alpha/m) * sumOne);

    thetaOne = tmpOne
    thetaZero = tmpZero
end;
