clc;
clear;

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
images_teste = loadMNISTImages('t10k-images.idx3-ubyte');
labels_teste = loadMNISTLabels('t10k-labels.idx1-ubyte');


%cria uma matriz em que tudo é 0.05, exceto o respectivo ao número da
%label, que é 0.95

resultado = 0.05*ones(60000, 10);

for i=1:60000
    aux=labels(i, 1);
    resultado(i, aux+1)=0.95;
end
resultado=resultado';

%declaração de variáveis que serão usadas
confere_resultado=[60000, 1];
eqm_rec = [];                
erro=0;

%acrescenta 1 no fim de cada coluna da matriz imagens
images = [images
    ones(1,60000)];       

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 


    %define o eta da rede
    eta=0.9;%0.99;

    %número de neurônios da primeira camada
    %a terceira camada é sempre 3
    neuronios1= 50; 
    neuronios2= 50; 

    %matrizes de pesos de entradas
    w1 = rand(neuronios1,785)*2-1; 
    w2 = rand(neuronios2, neuronios1 + 1)*2-1;
    w3 = rand(10, neuronios2 + 1)*2-1;
    
    %% 
    %%%%%%%%%%%%%%%%%%%%%%%%
    %    For Principal     %
    %%%%%%%%%%%%%%%%%%%%%%%%
    
   for k=1:10 
       eta=0.1;
       erro=0;
    for j=1:60000                       
        
        
        %%%%%%%%%%%%%%%%%%%%%
        %  Primeira camada  %
        %%%%%%%%%%%%%%%%%%%%%
        
        s1 = w1*images(:, j); %Soma das entradas multiplicadas pelos pesos
        y2 = 1./(1+exp( - s1)); %Saída da função do perceptron

        %Acrescenta o bias para a entrada na segunda camada
        y2 = [ y2 
            1 ];
        
        
        %%%%%%%%%%%%%%%%%%%%
        %  Segunda camada  %
        %%%%%%%%%%%%%%%%%%%%

        s2 = w2*y2; %Soma das entradas multiplicadas pelos pesos

        y3= 1./(1+exp( - s2)); %Saída dos perceptrons da segunda camada

        y3 = [ y3 
        1 ];

        %%%%%%%%%%%%%%%%%%%%%
        %  Terceira camada  %
        %%%%%%%%%%%%%%%%%%%%%

        s3 = w3*y3; %Soma das entradas multiplicadas pelos pesos

        y4= 1./(1+exp( - s3)); %Saída final


        %%%%%%%%%%%%%%%%%%%%%
        %       Erros       %
        %%%%%%%%%%%%%%%%%%%%%

        e =(resultado(:, j)-y4); %resultado menos a saida
        eqm = 0.1*(e'*e); %erro quadratico medio
        eqm_rec = [eqm_rec eqm]; %recorda os valores do erro quadratico médio


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %      Atualização de pesos      %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        e3=((resultado(:, j))-y4).*(y4.*(1-y4));
        dw3=eta*e3*y3';

        e2=((w3(:,1:end-1))'*e3).*y3(1:end-1, 1).*(1-y3(1:end-1, 1));
        dw2=eta*e2*y2';

        e1=(w2(:,1:end-1)'*e2).*y2(1:end-1, 1).*(1-y2(1:end-1, 1));
        dw1=eta*e1*images(:, j)';

        w1=w1+dw1;
        w2=w2+dw2;
        w3=w3+dw3;

        check = 0;
        
        %apenas para fins de conferir o resultado
        
        for k=1:10
            if y4(k)>check
                check=y4(k);
                confere_resultado(j) = k-1;
            end
        end

        if confere_resultado(j) ~= labels(j, 1)
            erro=erro+1;
        end

        eta=0.9999*eta; %atualiza o valor de eta;
    end
    erro
   end
   
    figure(1);
    plot(eqm_rec);
    bkprop_teste;

