function [ Output, Weight_Matrix, Bias_Matrix, SignalFlow_Matrix ] = NeuralNetwork( Input, Weight_Matrix, Bias_Matrix, SignalFlow_Matrix, NetworkRows, NetworkColumns, FunctionType, Slope_Parameter, FunctionAmplitude )
%This function is the implementation of a NetworkRows*NetworkColumns Block Based Neural Network (BBNN).
%
%Note: Side inputs of the BBNN are set to "0". This is identical to the state
%where all side inputs/outputs (left and right) are outputs. Therefore in
%the BBNN's right and left sides we just have outputs.
%
%Note: The process of calculating blocks' outputs goes through layers
%(rows).

%%%iniatialization of all network connections
InputOutput_Matrix=zeros(NetworkRows,NetworkColumns,4);
InputOutput_Matrix(1,:,1)=Input;

%%%initializing a matrix that defines which block can begin to process and
%%%which one's input(s) are still needed. "0" indicates an input without a
%%%processed value from neighbor output. "1" stands for a processed input
%%%or an output.
ProcessCommander_Matrix=SignalFlow_Matrix(:,:,:,1:2);

%%%performing network calculations
column_database=zeros(1,NetworkColumns);

for row=1:NetworkRows
    
    for column_counter=1:NetworkColumns
        
        %finding ready to process cells
        first_index=find(ProcessCommander_Matrix(:,row,:,1));
        column=first_index(find(ProcessCommander_Matrix(:,row,first_index,2)==1,1,'first'));
                
        %call the block for processing
        [Weight_Matrix(:,row,column,:), Bias_Matrix(:,row,column,:), InputOutput_Matrix(row,column,:), SignalFlow_Matrix(:,row,column,:)]=NeuronBlock(SignalFlow_Matrix(:,row,column,:), Weight_Matrix(:,row,column,:), Bias_Matrix(:,row,column,:), InputOutput_Matrix(row,column,:), FunctionType, Slope_Parameter, FunctionAmplitude);

        %applying new values (from outputs) to neighbor cells
        InputOutput_Matrix(min(row+1,NetworkRows),column,1)=InputOutput_Matrix(row,column,2);
        InputOutput_Matrix(row,min(column+1,NetworkColumns),3)=InputOutput_Matrix(row,column,4);
        InputOutput_Matrix(row,max(column-1,1),4)=InputOutput_Matrix(row,column,3);
        
        %updating ProcessCommander matrix
        ProcessCommander_Matrix(:,row,min(column+1,NetworkColumns),1)=1;
        ProcessCommander_Matrix(:,row,max(column-1,1),2)=1;
            
        %removing the processed cell from ProcessCommander matrix
        column_database(1,column_counter)=column;
        ProcessCommander_Matrix(:,row,column_database(1:column_counter),:)=0;
    end
end

%%%store output of the network
Output=InputOutput_Matrix(NetworkRows,:,2);

end

