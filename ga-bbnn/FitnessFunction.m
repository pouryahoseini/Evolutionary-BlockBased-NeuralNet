function [Fitness, Weight_Matrix, Bias_Matrix, SignalFlow_Matrix] = FitnessFunction(Example_Matrix, Example_Number, NetworkRows, NetworkColumns, SignalFlow_Matrix, Weight_Matrix, Bias_Matrix, FunctionType, Slope_Parameter, FunctionAmplitude, ComparisonType, OutputNumber)
%This function computes fitness of BBNN structures by feeding sample inputs
%and observing output of neural network. Actual outputs of neural network
%compared to expected outputs produce error (cost) and based on the error,
%fitness value can be obtained.

%set error to zero
Cumulative_Error=0;

%determine computation form of error
if strcmp(ComparisonType,'numeric')
    
    %executing all examples on the neural network and accumulating errors

    for example_counter=1:Example_Number
        %call neural network
        [ NeuralNetwork_Output, Weight_Matrix, Bias_Matrix, SignalFlow_Matrix ] = NeuralNetwork( Example_Matrix(example_counter,1:NetworkColumns), Weight_Matrix, Bias_Matrix, SignalFlow_Matrix, NetworkRows, NetworkColumns, FunctionType, Slope_Parameter, FunctionAmplitude );
        
        %converting binary to decimal
        %Note: left bits are more significant
        %Note: decimal=
        %value-of-msb-of-output*(2^n)+.....+value-of-lsb-of-output*(2^0)
        %Note: values multiplying in (2^n) are in range of [-1 1] in the
        %case of sigmoid activation functions.
        decimal_example_output=sum(Example_Matrix(example_counter,(NetworkColumns+OutputNumber)).*((2*ones(1,numel(OutputNumber))).^(numel(OutputNumber)-1:-1:0)));
        decimal_neural_network_output=sum(NeuralNetwork_Output(OutputNumber).*((2*ones(1,numel(OutputNumber))).^(numel(OutputNumber)-1:-1:0)));
               
        %accumulating error
        Cumulative_Error=Cumulative_Error+abs(decimal_example_output-decimal_neural_network_output);
        
    end
else
    %executing all examples on the neural network and accumulating errors
    
    for example_counter=1:Example_Number
        %call neural network
       
        [ NeuralNetwork_Output, Weight_Matrix, Bias_Matrix, SignalFlow_Matrix ] = NeuralNetwork( Example_Matrix(example_counter,1:NetworkColumns), Weight_Matrix, Bias_Matrix, SignalFlow_Matrix, NetworkRows, NetworkColumns, FunctionType, Slope_Parameter, FunctionAmplitude );
        
        %accumulating error
        Cumulative_Error=Cumulative_Error+sum(abs(Example_Matrix(example_counter,(NetworkColumns+OutputNumber))-NeuralNetwork_Output(OutputNumber)));
    end
    
end
    
%generating fitness based on the error
Fitness=1/(1+Cumulative_Error);

end

