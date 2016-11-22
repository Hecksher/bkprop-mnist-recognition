
images_teste = [images_teste
    ones(1,10000)];

erro=0;

for j=1:10000                       
        
        s1 = w1*images_teste(:, j);  %entrada passa pela primeira camada
        y2 = 1./(1+exp( - s1)); % Sigmoid of weighted input
        
        y2 = [ y2 
            1 ];    % Add bias node
        
        s2 = w2*y2; % Pass hidden acts through weights
        
        y3= 1./(1+exp( - s2));
        
        y3 = [ y3 
            1 ];
        
        s3 = w3*y3;
        y4= 1./(1+exp( - s3));
       
        check = 0;

        for k=1:10
            if y4(k)>check
                check=y4(k);
                confere_resultado(j) = k-1;
            end
        end
        
    confere_resultado(j);    
    if confere_resultado(j) ~= labels_teste(j, 1)
        erro=erro+1;
    end

    %j
end 
erro
porcentagem_de_acerto=(10000-erro)/100