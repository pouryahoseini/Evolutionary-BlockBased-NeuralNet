function [output]=ActivationFunction(input, FunctionType, Slope_Parameter, FunctionAmplitude)
%This function implements sigmoid, and gaussian functions or simply a sign
%function which maps values greater than or equal zero to "1" and maps 
%values less than zero to "-1". 

if strcmp(FunctionType,'sigmoid')
    output=FunctionAmplitude/(1+exp(-Slope_Parameter*input));
elseif strcmp(FunctionType,'gaussian')
    output=FunctionAmplitude*exp(-Slope_Parameter*(input^2));
else
    output=FunctionAmplitude*sign(input);
end

end