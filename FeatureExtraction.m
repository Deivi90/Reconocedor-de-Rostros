%% Extrae los Features de las imagenes

function [Rank,LDAfeatures,disc_vect,vectorClass,lenghtClass]=FeatureExtraction(CantFeatures,CantClass,CantSamples,CantTrainSamples)


    %% Primera etapa de la extraccion de features.
    %% K-MEANs para reducir la dimensionalidad del problema.
    %% Se hace algo parecido al ejercicio de compresion.
    %Tamanio imagenes
    HEIGHT = 60;
    WIDTH = 60;
%     CantFeatures = 32;                     % Cantidad de clusters
    %% CARGO IMAGENES NORMALIZADAS
    Allfiles = dir('BasedeDatos\*.jpg');                                   %Directorio de las Imagenes

    %Agarro solo los files de train
    for k=1:CantClass  
            FilesClassK=Allfiles(...         
            CantSamples*(k-1)+1:...
            CantSamples*(k-1)+CantSamples,:);  
    
    files(CantTrainSamples*(k-1)+1:...
        CantTrainSamples*(k-1)+CantTrainSamples,:)=...
                                    FilesClassK(1:CantTrainSamples,:);
    end
    image = zeros(HEIGHT,WIDTH,length(files));
    for k=1:length(files)
        image(:,:,k) = imread(strcat('BasedeDatos\',files(k,1).name));  %Cargo imagenes
    end
    %Convierto imagenes en vector.
    vector = zeros(length(files),HEIGHT*WIDTH);
    k = 1;
    for j = 1 : HEIGHT
        for i = 1 : WIDTH
            vector(:,k) = image(j,i,:);
            k = k+1;
        end
    end

    %% K-MEANs
    Nsamples = length(files);   % cantidad de imagenes // Features
    % Centroides iniciales: Son las medias de la clases
    classMeans = rand(Nsamples,CantFeatures).*(mean(vector(:)*(1.1)) - mean(vector(:)*(0.9))) + mean(vector(:)*(0.9));

    % Algoritmo K-means
    iteraciones = 1;
    NewclassMeans = zeros(size(classMeans));
    vectorClass = zeros(1,length(files));           %Clase a la que pertenece cada vector
    Error = 1;
    while(Error > 1e-2 || nnz(lenghtClass) < CantFeatures)
        iteraciones = iteraciones + 1;
        lenghtClass = zeros(CantFeatures,1);                   %Cantidad de elementos en la clase
        class = zeros(floor(60*60/CantFeatures),Nsamples,CantFeatures);   %Contiene los vectores de la clase
        %Clasifico
        for k = 1 : length(vector)
            %Busco la clase con minima distancia al vector
            distancia = sum( (((vector(:,k)*ones(1,CantFeatures)) - classMeans).^2) );
            [~,vectorClass(k)] = min(distancia);
            %Acomodo el vector en la clase que le pertenece    
            class( lenghtClass(vectorClass(k))+1,:,vectorClass(k) ) = vector(:,k);  
            lenghtClass(vectorClass(k)) = lenghtClass(vectorClass(k)) + 1;
        end
        %Calculo nuevas medias
        for i=1:CantFeatures
            if lenghtClass(i) == 0                                              %Se cambian de lugar las clases vacias hacia donde hay mas muestras
                [~,index] = max(lenghtClass);
                NewclassMeans(:,i) = classMeans(:,index).*(rand*0.05+0.95);     %Ceranias de la mayor clase
            else
                NewclassMeans(:,i) = mean(class(1:lenghtClass(i),:,i),1);
            end      
        end
        Error = sum( sum((classMeans-NewclassMeans).^2) ); 
        classMeans = NewclassMeans;
    end

    %% Este codigo comentado es para ver como quedan las imagenes al comprimir
    % k=1;
    % for j=1:HEIGHT
        % for i=1:WIDTH   
            % for H=1:7
                % reimage(j,i,H)=classMeans(H,vectorClass(k));
            % end
             % k=k+1;
        % end
    % end
    % for H=1:7
                % figure
                % imshow(uint8(reimage(:,:,H)))
    % end

    % Los nuevos features sean las medias de cada clase 
    % NewVector sera la entrada del LDA. Newvector NxM. N son la cantidad de
    % imagenes y M es la cantidad de features
    FeaturesVector = classMeans;
%     save('FeaturesVector.mat','FeaturesVector')
%     save('vectorClass.mat','vectorClass','lenghtClass','HEIGHT','WIDTH')


    %%                      LDA
    %% Segunda etapa de la extraccion de features: LDA
%     CantClass=6;     %Cantidad de personas
%     CantSamples=10;   %Cantidad de fotos por persona

    %% Calculo matrices(Sb y Sw) para iniciar el algoritmo LDA
    [~, nFeatures ] = size(FeaturesVector);  

    % Sw:WITHIN CLASS SCATTER MATRIX
    Sw = zeros(nFeatures,nFeatures);              
    % Sb:BETWEEN  CLASS SCATTER MATRIX
    Sb = zeros(nFeatures,nFeatures); 


    %Media total de los features. En el paper lo llama X raya
    FeatureMeans=(1/(CantClass*CantTrainSamples))*sum(FeaturesVector); %En el paper X

    for i = 1 : CantClass         %En el paper i es K

        % Todos los vectores de la clase i, es decir que se toman los i-esimos
        % "Cant_TrainSamples" elementos de trainFeatures
          FeaturesClassI=FeaturesVector(...         
                CantTrainSamples*(i-1)+1:...
                CantTrainSamples*(i-1)+CantTrainSamples,:);  

        %Media de la clase en el paper la llama X^k
        ClassMean=(1/CantTrainSamples)*sum(FeaturesClassI); 

        for j = 1 : CantTrainSamples       %En el paper j es M

             xi = FeaturesClassI(j,:); % Un solo vector de features en el paper {x_m}^k    
             xi_clases = xi - ClassMean ; 
             Sw = Sw + (xi_clases' * xi_clases);
        end

        xi_res = ClassMean - FeatureMeans;
        Sb = Sb + (xi_res' * xi_res);              %Parece que esta bien este                     
    end

    %% Calculo de Aves

    Rank = rank(Sw);
    if( nFeatures == Rank )
        Sb_hat = inv(Sb + Sw) * Sb;
    else
%         disp('Aparece size problem')

        % svd(A) performs a singular value decomposition of matrix A, such that A = U*S*V'
        [U,S,V] = svd(Sw);
        Q = V( : , Rank+1 : end); 
        aux = (Q * Q');
        Sb_hat = aux * Sb * aux';
    end
    % eig computes the eigenvectors and stores the eigenvalues in a diagonal matrix:
    [Ave,Ava] = eig(Sb_hat);    %Ave autovectores en columnas

    % Ordeno los Ava Y Ave en funcion del mayor autovalor con su autovector 
    vec_ava = zeros(1,length(Ava));
    for i = 1 : length(Ava)
        vec_ava(i) = Ava(i,i);
    end
    [max_ava, posicion] = sort(vec_ava,'descend');

    max_ave = zeros(size(Ave));
    for i = 1 : length(Ava)
        max_ave(:,i) = Ave(:,posicion(i));
    end

    % Busco los autovalores maximos fijando una cota
    cota = 0.95;
    aux1 = max_ava(1);
    sum_tot = sum(max_ava);
    i = 1;
    scape = 1;
    while ( i < length(max_ava) && scape ) 
        if ( (aux1/sum_tot) <= cota )
            aux1 = max_ava(i) + aux1;
            disc_vect(:,i) = max_ave(:,i);
        else if (i)
                disc_vect(:,i) = max_ave(:,i);    
                scape = 0;
            end
        end
        i = i + 1;
    end

    LDAfeatures=disc_vect'*FeaturesVector';
    save('LDAfeatures.mat','LDAfeatures','disc_vect','vectorClass','lenghtClass')
    
end


